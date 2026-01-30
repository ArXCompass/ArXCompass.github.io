# llm - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19970v1">Benchmarking LLAMA Model Security Against OWASP Top 10 For LLM Applications</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) move from research prototypes to enterprise systems, their security vulnerabilities pose serious risks to data privacy and system integrity. This study benchmarks various Llama model variants against the OWASP Top 10 for LLM Applications framework, evaluating threat detection accuracy, response safety, and computational overhead. Using the FABRIC testbed with NVIDIA A30 GPUs, we tested five standard Llama models and five Llama Guard variants on 100 adversarial prompts covering ten vulnerability categories. Our results reveal significant differences in security performance: the compact Llama-Guard-3-1B model achieved the highest detection rate of 76% with minimal latency (0.165s per test), whereas base models such as Llama-3.1-8B failed to detect threats (0% accuracy) despite longer inference times (0.754s). We observe an inverse relationship between model size and security effectiveness, suggesting that smaller, specialized models often outperform larger general-purpose ones in security tasks. Additionally, we provide an open-source benchmark dataset including adversarial prompts, threat labels, and attack metadata to support reproducible research in AI security, [1].
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19899v1">Evaluation of Oncotimia: An LLM based system for supporting tumour boards</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 9 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Multidisciplinary tumour boards (MDTBs) play a central role in oncology decision-making but require manual processes and structuring large volumes of heterogeneous clinical information, resulting in a substantial documentation burden. In this work, we present ONCOTIMIA, a modular and secure clinical tool designed to integrate generative artificial intelligence (GenAI) into oncology workflows and evaluate its application to the automatic completion of lung cancer tumour board forms using large language models (LLMs). The system combines a multi-layer data lake, hybrid relational and vector storage, retrieval-augmented generation (RAG) and a rule-driven adaptive form model to transform unstructured clinical documentation into structured and standardised tumour board records. We assess the performance of six LLMs deployed through AWS Bedrock on ten lung cancer cases, measuring both completion form accuracy and end-to-end latency. The results demonstrate high performance across models, with the best performing configuration achieving an 80% of correct field completion and clinically acceptable response time for most LLMs. Larger and more recent models exhibit best accuracies without incurring prohibitive latency. These findings provide empirical evidence that LLM- assisted autocompletion form is technically feasible and operationally viable in multidisciplinary lung cancer workflows and support its potential to significantly reduce documentation burden while preserving data quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.13187v2">"Not in My Backyard": LLMs Uncover Online and Offline Social Biases Against Homelessnes</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Homelessness is a persistent social challenge, impacting millions worldwide. Over 876,000 people experienced homelessness (PEH) in the U.S. in 2025. Social bias is a significant barrier to alleviation, shaping public perception and influencing policymaking. Given that online textual media and offline city council discourse reflect and influence part of public opinion, it provides valuable insights to identify and track social biases against PEH. We present a new, manually-annotated multi-domain dataset compiled from Reddit, X (formerly Twitter), news articles, and city council meeting minutes across ten U.S. cities. Our 16-category multi-label taxonomy creates a challenging long-tail classification problem: some categories appear in less than 1% of samples, while others exceed 70%. We find that small human-annotated datasets (1,702 samples) are insufficient for training effective classifiers, whether used to fine-tune encoder models or as few-shot examples for LLMs. To address this, we use GPT-4.1 to generate pseudo-labels on a larger unlabeled corpus. Training on this expanded dataset enables even small encoder models (ModernBERT, 150M parameters) to achieve 35.23 macro-F1, approaching GPT-4.1's 41.57. This demonstrates that \textbf{data quantity matters more than model size}, enabling low-cost, privacy-preserving deployment without relying on commercial APIs. Our results reveal that negative bias against PEH is prevalent both offline and online (especially on Reddit), with "not in my backyard" narratives showing the highest engagement. These findings uncover a type of ostracism that directly impacts poverty-reduction policymaking and provide actionable insights for practitioners addressing homelessness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02091v4">Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted by ICASSP 2026
    </div>
    <details class="paper-abstract">
      Recent studies suggest that the deeper layers of Large Language Models (LLMs) contribute little to representation learning and can often be removed without significant performance loss. However, such claims are typically drawn from narrow evaluations and may overlook important aspects of model behavior. In this work, we present a systematic study of depth utilization across diverse dimensions, including evaluation protocols, task categories, and model architectures. Our analysis confirms that very deep layers are generally less effective than earlier ones, but their contributions vary substantially with the evaluation setting. Under likelihood-based metrics without generation, pruning most layers preserves performance, with only the initial few being critical. By contrast, generation-based evaluation uncovers indispensable roles for middle and deeper layers in enabling reasoning and maintaining long-range coherence. We further find that knowledge and retrieval are concentrated in shallow components, whereas reasoning accuracy relies heavily on deeper layers -- yet can be reshaped through distillation. These results highlight that depth usage in LLMs is highly heterogeneous and context-dependent, underscoring the need for task-, metric-, and model-aware perspectives in both interpreting and compressing large models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08512v2">MLVTG: Mamba-Based Feature Alignment and LLM-Driven Purification for Multi-Modal Video Temporal Grounding</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Video Temporal Grounding (VTG), which aims to localize video clips corresponding to natural language queries, is a fundamental yet challenging task in video understanding. Existing Transformer-based methods often suffer from redundant attention and suboptimal multi-modal alignment. To address these limitations, we propose MLVTG, a novel framework that integrates two key modules: MambaAligner and LLMRefiner. MambaAligner uses stacked Vision Mamba blocks as a backbone instead of Transformers to model temporal dependencies and extract robust video representations for multi-modal alignment. LLMRefiner leverages the specific frozen layer of a pre-trained Large Language Model (LLM) to implicitly transfer semantic priors, enhancing multi-modal alignment without fine-tuning. This dual alignment strategy, temporal modeling via structured state-space dynamics and semantic purification via textual priors, enables more precise localization. Extensive experiments on QVHighlights, Charades-STA, and TVSum demonstrate that MLVTG achieves state-of-the-art performance and significantly outperforms existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.17768v2">The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Sparse attention offers a promising strategy to extend long-context capabilities in Transformer LLMs, yet its efficiency-accuracy trade-offs remain unclear due to the lack of comprehensive evaluation. We address this gap with the largest-scale empirical analysis to date of training-free sparse attention, evaluating six methods across multiple model families and sizes, sequences up to 128K tokens, and sparsity levels up to 0.95 (i.e., $1/20$ attention budget) on nine diverse tasks. We first organise the rapidly evolving landscape of sparse attention methods into a taxonomy along four design axes. Our analysis then yields actionable insights: 1) sparse attention is effective -- larger sparse models outperform smaller dense ones at equivalent cost, improving the Pareto frontier; 2) due to computational constraints, token-to-page importance estimation is unfeasible during prefilling, where the choice of an alternative solution (global-to-token or block-to-block) depends on the task, but is possible during decoding, enabling better generalisation and tolerance to higher sparsity; 3) longer sequences tolerate higher sparsity, suggesting that fixed-budget methods in production are suboptimal. Together, these findings provide practical guidance for deploying sparse attention and methodological recommendations for future evaluations. Our code is available at https://github.com/PiotrNawrot/sparse-frontier.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19847v1">Identifying and Transferring Reasoning-Critical Neurons: Improving LLM Inference Reliability via Activation Steering</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Despite the strong reasoning capabilities of recent large language models (LLMs), achieving reliable performance on challenging tasks often requires post-training or computationally expensive sampling strategies, limiting their practical efficiency. In this work, we first show that a small subset of neurons in LLMs exhibits strong predictive correlations with reasoning correctness. Based on this observation, we propose AdaRAS (Adaptive Reasoning Activation Steering), a lightweight test-time framework that improves reasoning reliability by selectively intervening on neuron activations. AdaRAS identifies Reasoning-Critical Neurons (RCNs) via a polarity-aware mean-difference criterion and adaptively steers their activations during inference, enhancing incorrect reasoning traces while avoiding degradation on already-correct cases. Experiments on 10 mathematics and coding benchmarks demonstrate consistent improvements, including over 13% gains on AIME-24 and AIME-25. Moreover, AdaRAS exhibits strong transferability across datasets and scalability to stronger models, outperforming post-training methods without additional training or sampling cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19839v1">HARMONI: Multimodal Personalization of Multi-User Human-Robot Interactions with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Existing human-robot interaction systems often lack mechanisms for sustained personalization and dynamic adaptation in multi-user environments, limiting their effectiveness in real-world deployments. We present HARMONI, a multimodal personalization framework that leverages large language models to enable socially assistive robots to manage long-term multi-user interactions. The framework integrates four key modules: (i) a perception module that identifies active speakers and extracts multimodal input; (ii) a world modeling module that maintains representations of the environment and short-term conversational context; (iii) a user modeling module that updates long-term speaker-specific profiles; and (iv) a generation module that produces contextually grounded and ethically informed responses. Through extensive evaluation and ablation studies on four datasets, as well as a real-world scenario-driven user-study in a nursing home environment, we demonstrate that HARMONI supports robust speaker identification, online memory updating, and ethically aligned personalization, outperforming baseline LLM-driven approaches in user modeling accuracy, personalization quality, and user satisfaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.00961v2">LLM-Generated Explanations Do Not Suffice for Ultra-Strong Machine Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Ultra Strong Machine Learning (USML) refers to symbolic learning systems that not only improve their own performance but can also teach their acquired knowledge to quantifiably improve human performance. We introduce LENS (Logic Programming Explanation via Neural Summarisation), a neuro-symbolic framework that combines symbolic program synthesis with large language models (LLMs). This framework automatically generates natural language explanations of learned logic programs, replacing hand-crafted templates used in prior USML work. Using LLMs-as-judges evaluation and expert validation, we show that LENS produces higher-quality explanations than both direct LLM prompting and hand-crafted templates. We then examine whether LENS explanations suffice for achieving USML in a human trial teaching active learning strategies across three related domains. Our exploratory analysis suggests that concise, expert-written explanations may benefit learners with higher initial performance, while LLM-generated explanations provide no advantage over human self learning despite being rated as higher quality. This case study reveals that achieving USML requires methods grounded in human learning, where current LLM-generated explanations do not capture human cognitive constraints and LLMs-as-judges evaluations do not reflect what effectively supports human learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16846v4">BASIL: Bayesian Assessment of Sycophancy in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Sycophancy (overly agreeable or flattering behavior) poses a fundamental challenge for human-AI collaboration, particularly in high-stakes decision-making domains such as health, law, and education. A central difficulty in studying sycophancy in large language models (LLMs) is disentangling sycophantic belief shifts from rational changes in behavior driven by new evidence or user-provided information. Existing approaches either measure descriptive behavior changes or apply normative evaluations that rely on objective ground truth, limiting their applicability to subjective or uncertain tasks. We introduce a Bayesian probabilistic framework, grounded in behavioral economics and rational decision theory, that explicitly separates sycophancy from rational belief updating. Within this framework, we achieve three objectives: (i) a descriptive metric that measures sycophancy while controlling for rational responses to evidence; (ii) a normative metric that quantifies how sycophancy leads models astray from Bayesian-consistent belief updating; and (iii) the ability to apply both metrics in settings without ground-truth labels. Applying our framework across multiple LLMs and three uncertainty-driven tasks, we find robust evidence of sycophantic belief shifts and show that their impact on rationality depends on whether models systematically over- or under-update their beliefs. Finally, we demonstrate that a post-hoc calibration method and two fine-tuning strategies (SFT and DPO) substantially reduce Bayesian inconsistency, with particularly strong improvements under explicit sycophancy prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22603v3">Mitigating Attention Sinks and Massive Activations in Audio-Visual Speech Recognition with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 IEEE ICASSP 2026. The code is available at https://github.com/umbertocappellazzo/Llama-AVSR
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently advanced auditory speech recognition (ASR), visual speech recognition (VSR), and audio-visual speech recognition (AVSR). However, understanding of their internal dynamics under fine-tuning remains limited. In natural language processing, recent work has revealed attention sinks, tokens that attract disproportionately high attention, and associated massive activations in which some features of sink tokens exhibit huge activation in LLMs. In this work, we are the first to study these phenomena in multimodal speech recognition. Through a detailed analysis of audio-visual LLMs, we identify attention sinks and massive activations not only at the BOS token but also at intermediate low-semantic tokens across ASR, VSR, and AVSR. We show that massive activations originate in the MLP layers and correspond to fixed feature indices across all sink tokens. We further show that intermediate sink tokens exhibit high cosine similarity to the BOS token, thereby amplifying attention and activation. Building on these insights, we introduce a simple decorrelation loss that reduces cosine similarity between BOS and other tokens, effectively mitigating intermediate sinks and massive activations. Furthermore, our method improves word error rate (WER) under high audio-visual feature downsampling while remaining stable at lower downsampling rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.10113v4">What Does Neuro Mean to Cardio? Investigating the Role of Clinical Specialty Data in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      In this paper, we introduce S-MedQA, an English medical question-answering (QA) dataset designed for benchmarking large language models (LLMs) in fine-grained clinical specialties. S-MedQA consists of over 24k examples, covering 15 medical specialties, with QA pairs that can have multiple specialty annotations, such as when a question is cross-disciplinary. The dataset is constructed using both machine and expert verification to maximize data availability and reliability. We use S-MedQA to investigate the role of clinical specialties in the knowledge-intensive scenario of medical QA. Our results show that training on data from a clinical specialty does not necessarily lead to the best performance on that specialty. Additionally, regardless of the specialty the LLM was fine-tuned on, token probabilities of clinically relevant terms consistently increase across all specialties. Based on these findings, we hypothesize that improvement gains, at least in our settings, are derived primarily from domain shifting (e.g., general to medical) rather than from injecting specialty-specific knowledge. This suggests a need to rethink the role of fine-tuning data in the medical domain. To encourage further advancements in the clinical NLP field, we release S-MedQA along with all the code required to reproduce our experiments for the research community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19693v1">Using LLMs to Evaluate Architecture Documents: Results from a Digital Marketplace Environment</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Generative AI plays an increasing role during software engineering activities to make them, e.g., more efficient or provide better quality. However, it is often unclear how much benefit LLMs really provide. We concentrate on software architects and investigated how an LLM-supported evaluation of architecture documents can support software architects to improve such artefacts. In the context of a research project where a digital marketplace is developed and digital solutions should be analyzed, we used different LLMs to analyze the quality of architecture documents and compared the results with evaluations from software architects. We found out that the quality of the artifact has a strong influence on the quality of the LLM, i.e., the better the quality of the architecture document was, the more consistent were the LLM-based evaluation and the human expert evaluation. While using LLMs in this architecture task is promising, our results showed inconsistencies that need further analyses before generalizing them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19684v1">LLM-Assisted Authentication and Fraud Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 20 pages, 7 figures, 2 tables
    </div>
    <details class="paper-abstract">
      User authentication and fraud detection face growing challenges as digital systems expand and adversaries adopt increasingly sophisticated tactics. Traditional knowledge-based authentication remains rigid, requiring exact word-for-word string matches that fail to accommodate natural human memory and linguistic variation. Meanwhile, fraud-detection pipelines struggle to keep pace with rapidly evolving scam behaviors, leading to high false-positive rates and frequent retraining cycles required. This work introduces two complementary LLM-enabled solutions, namely, an LLM-assisted authentication mechanism that evaluates semantic correctness rather than exact wording, supported by document segmentation and a hybrid scoring method combining LLM judgement with cosine-similarity metrics and a RAG-based fraud-detection pipeline that grounds LLM reasoning in curated evidence to reduce hallucinations and adapt to emerging scam patterns without model retraining. Experiments show that the authentication system accepts 99.5% of legitimate non-exact answers while maintaining a 0,1% false-acceptance rate, and that the RAG-enhanced fraud detection reduces false positives from 17.2% to 35%. Together, these findings demonstrate that LLMs can significantly improve both usability and robustness in security workflows, offering a more adaptive , explainable, and human-aligned approach to authentication and fraud detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12360v2">Discovering 100+ Compiler Defects in 72 Hours via LLM-Driven Semantic Logic Recomposition</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Compilers constitute the foundational root-of-trust in software supply chains; however, their immense complexity inevitably conceals critical defects. Recent research has attempted to leverage historical bugs to design new mutation operators or fine-tune models to increase program diversity for compiler fuzzing.We observe, however, that bugs manifest primarily based on the semantics of input programs rather than their syntax. Unfortunately, current approaches, whether relying on syntactic mutation or general Large Language Model (LLM) fine-tuning, struggle to preserve the specific semantics found in the logic of bug-triggering programs. Consequently, these critical semantic triggers are often lost, resulting in a limitation of the diversity of generated programs. To explicitly reuse such semantics, we propose FeatureFuzz, a compiler fuzzer that combines features to generate programs. We define a feature as a decoupled primitive that encapsulates a natural language description of a bug-prone invariant, such as an out-of-bounds array access, alongside a concrete code witness of its realization. FeatureFuzz operates via a three-stage workflow: it first extracts features from historical bug reports, synthesizes coherent groups of features, and finally instantiates these groups into valid programs for compiler fuzzing. We evaluated FeatureFuzz on GCC and LLVM. Over 24-hour campaigns, FeatureFuzz uncovered 167 unique crashes, which is 2.78x more than the second-best fuzzer. Furthermore, through a 72-hour fuzzing campaign, FeatureFuzz identified 113 bugs in GCC and LLVM, 97 of which have already been confirmed by compiler developers, validating the approach's ability to stress-test modern compilers effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19622v1">Algorithmic Prompt-Augmentation for Efficient LLM-Based Heuristic Design for A* Search</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 accepted at EvoStar conference; Code: https://github.com/tb-git-tud/a-ceoh-evolution-of-heuristics?tab=readme-ov-file
    </div>
    <details class="paper-abstract">
      Heuristic functions are essential to the performance of tree search algorithms such as A*, where their accuracy and efficiency directly impact search outcomes. Traditionally, such heuristics are handcrafted, requiring significant expertise. Recent advances in large language models (LLMs) and evolutionary frameworks have opened the door to automating heuristic design. In this paper, we extend the Evolution of Heuristics (EoH) framework to investigate the automated generation of guiding heuristics for A* search. We introduce a novel domain-agnostic prompt augmentation strategy that includes the A* code into the prompt to leverage in-context learning, named Algorithmic - Contextual EoH (A-CEoH). To evaluate the effectiveness of A-CeoH, we study two problem domains: the Unit-Load Pre-Marshalling Problem (UPMP), a niche problem from warehouse logistics, and the classical sliding puzzle problem (SPP). Our computational experiments show that A-CEoH can significantly improve the quality of the generated heuristics and even outperform expert-designed heuristics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19607v1">ComAgent: Multi-LLM based Agentic AI Empowered Intelligent Wireless Networks</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Emerging 6G networks rely on complex cross-layer optimization, yet manually translating high-level intents into mathematical formulations remains a bottleneck. While Large Language Models (LLMs) offer promise, monolithic approaches often lack sufficient domain grounding, constraint awareness, and verification capabilities. To address this, we present ComAgent, a multi-LLM agentic AI framework. ComAgent employs a closed-loop Perception-Planning-Action-Reflection cycle, coordinating specialized agents for literature search, coding, and scoring to autonomously generate solver-ready formulations and reproducible simulations. By iteratively decomposing problems and self-correcting errors, the framework effectively bridges the gap between user intent and execution. Evaluations demonstrate that ComAgent achieves expert-comparable performance in complex beamforming optimization and outperforms monolithic LLMs across diverse wireless tasks, highlighting its potential for automating design in emerging wireless networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19588v1">From Atoms to Chains: Divergence-Guided Reasoning Curriculum for Unlabeled LLM Domain Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Code: https://github.com/bytedance/DGRC
    </div>
    <details class="paper-abstract">
      Adapting Large Language Models (LLMs) to specialized domains without human-annotated data is a crucial yet formidable challenge. Widely adopted knowledge distillation methods often devolve into coarse-grained mimicry, where the student model inefficiently targets its own weaknesses and risks inheriting the teacher's reasoning flaws. This exposes a critical pedagogical dilemma: how to devise a reliable curriculum when the teacher itself is not an infallible expert. Our work resolves this by capitalizing on a key insight: while LLMs may exhibit fallibility in complex, holistic reasoning, they often exhibit high fidelity on focused, atomic sub-problems. Based on this, we propose Divergence-Guided Reasoning Curriculum (DGRC), which constructs a learning path from atomic knowledge to reasoning chains by dynamically deriving two complementary curricula from disagreements in reasoning pathways. When a student and teacher produce conflicting results, DGRC directs the teacher to perform a diagnostic analysis: it analyzes both reasoning paths to formulate atomic queries that target the specific points of divergence, and then self-answers these queries to create high-confidence atomic question-answer pairs. These pairs then serve a dual purpose: (1) providing an atomic curriculum to rectify the student's knowledge gaps, and (2) serving as factual criteria to filter the teacher's original reasoning chains, yielding a verified CoT curriculum that teaches the student how to integrate atomic knowledge into complete reasoning paths. Experiments across the medical and legal domains on student models of various sizes demonstrate the effectiveness of our DGRC framework. Notably, our method achieves a 7.76% relative improvement for the 1.5B student model in the medical domain over strong unlabeled baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19585v1">LLM-Enhanced Reinforcement Learning for Long-Term User Satisfaction in Interactive Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Interactive recommender systems can dynamically adapt to user feedback, but often suffer from content homogeneity and filter bubble effects due to overfitting short-term user preferences. While recent efforts aim to improve content diversity, they predominantly operate in static or one-shot settings, neglecting the long-term evolution of user interests. Reinforcement learning provides a principled framework for optimizing long-term user satisfaction by modeling sequential decision-making processes. However, its application in recommendation is hindered by sparse, long-tailed user-item interactions and limited semantic planning capabilities. In this work, we propose LLM-Enhanced Reinforcement Learning (LERL), a novel hierarchical recommendation framework that integrates the semantic planning power of LLM with the fine-grained adaptability of RL. LERL consists of a high-level LLM-based planner that selects semantically diverse content categories, and a low-level RL policy that recommends personalized items within the selected semantic space. This hierarchical design narrows the action space, enhances planning efficiency, and mitigates overexposure to redundant content. Extensive experiments on real-world datasets demonstrate that LERL significantly improves long-term user satisfaction when compared with state-of-the-art baselines. The implementation of LERL is available at https://anonymous.4open.science/r/code3-18D3/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19583v1">Toward Architecture-Aware Evaluation Metrics for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted at CAIN 2026 (IEEE/ACM 5th International Conference on AI Engineering)
    </div>
    <details class="paper-abstract">
      LLM-based agents are becoming central to software engineering tasks, yet evaluating them remains fragmented and largely model-centric. Existing studies overlook how architectural components, such as planners, memory, and tool routers, shape agent behavior, limiting diagnostic power. We propose a lightweight, architecture-informed approach that links agent components to their observable behaviors and to the metrics capable of evaluating them. Our method clarifies what to measure and why, and we illustrate its application through real world agents, enabling more targeted, transparent, and actionable evaluation of LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.26201v2">LLM Agents for Knowledge Discovery in Atomic Layer Processing</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted submission to the AI4MAT workshop@NEURIPS 2025. As submitted, except author names added
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have garnered significant attention for several years now. Recently, their use as independently reasoning agents has been proposed. In this work, we test the potential of such agents for knowledge discovery in materials science. We repurpose LangGraph's tool functionality to supply agents with a black box function to interrogate. In contrast to process optimization or performing specific, user-defined tasks, knowledge discovery consists of freely exploring the system, posing and verifying statements about the behavior of this black box, with the sole objective of generating and verifying generalizable statements. We provide proof of concept for this approach through a children's parlor game, demonstrating the role of trial-and-error and persistence in knowledge discovery, and the strong path-dependence of results. We then apply the same strategy to show that LLM agents can explore, discover, and exploit diverse chemical interactions in an advanced Atomic Layer Processing reactor simulation using intentionally limited probe capabilities without explicit instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.17540v2">SGCR: A Specification-Grounded Framework for Trustworthy LLM Code Review</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted at ASE 2025
    </div>
    <details class="paper-abstract">
      Automating code review with Large Language Models (LLMs) shows immense promise, yet practical adoption is hampered by their lack of reliability, context-awareness, and control. To address this, we propose Specification-Grounded Code Review (SGCR), a framework that grounds LLMs in human-authored specifications to produce trustworthy and relevant feedback. SGCR features a novel dual-pathway architecture: an explicit path ensures deterministic compliance with predefined rules derived from these specifications, while an implicit path heuristically discovers and verifies issues beyond those rules. Deployed in a live industrial environment at HiThink Research, SGCR's suggestions achieved a 42% developer adoption rate-a 90.9% relative improvement over a baseline LLM (22%). Our work demonstrates that specification-grounding is a powerful paradigm for bridging the gap between the generative power of LLMs and the rigorous reliability demands of software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21372v2">Improving LLM-based Global Optimization with Search Space Partitioning</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 31 pages, 19 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently emerged as effective surrogate models and candidate generators within global optimization frameworks for expensive blackbox functions. Despite promising results, LLM-based methods often struggle in high-dimensional search spaces or when lacking domain-specific priors, leading to sparse or uninformative suggestions. To overcome these limitations, we propose HOLLM, a novel global optimization algorithm that enhances LLM-driven sampling by partitioning the search space into promising subregions. Each subregion acts as a ``meta-arm'' selected via a bandit-inspired scoring mechanism that effectively balances exploration and exploitation. Within each selected subregion, an LLM then proposes high-quality candidate points, without any explicit domain knowledge. Empirical evaluation on standard optimization benchmarks shows that HOLLM consistently matches or surpasses leading global optimization methods, while substantially outperforming global LLM-based sampling strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.07639v3">PowerGraph-LLM: Novel Power Grid Graph Embedding and Optimization with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Published at IEEE Transactions on Power Systems
    </div>
    <details class="paper-abstract">
      Efficiently solving Optimal Power Flow (OPF) problems in power systems is crucial for operational planning and grid management. There is a growing need for scalable algorithms capable of handling the increasing variability, constraints, and uncertainties in modern power networks while providing accurate and fast solutions. To address this, machine learning techniques, particularly Graph Neural Networks (GNNs) have emerged as promising approaches. This letter introduces PowerGraph-LLM, the first framework explicitly designed for solving OPF problems using Large Language Models (LLMs). The proposed approach combines graph and tabular representations of power grids to effectively query LLMs, capturing the complex relationships and constraints in power systems. A new implementation of in-context learning and fine-tuning protocols for LLMs is introduced, tailored specifically for the OPF problem. PowerGraph-LLM demonstrates reliable performances using off-the-shelf LLM. Our study reveals the impact of LLM architecture, size, and fine-tuning and demonstrates our framework's ability to handle realistic grid components and constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23785v2">Meaning Is Not A Metric: Using LLMs to make cultural context legible at scale</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      This position paper argues that large language models (LLMs) can make cultural context, and therefore human meaning, legible at an unprecedented scale in AI-based sociotechnical systems. We argue that such systems have previously been unable to represent human meaning because they rely on thin descriptions (numerical representations that enforce standardization and therefore strip human activity of the cultural context which gives it meaning). By contrast, scholars in the humanities and qualitative social sciences have developed frameworks for representing meaning through thick description (verbal representations that accommodate heterogeneity and retain contextual information needed to represent human meaning). The verbal capabilities of LLMs now provide a means of at least partially automating the generation and processing of thick descriptions, offering new ways to deploy them at scale. We argue that the problem of rendering human meaning legible is not just about selecting better metrics but about developing new representational formats based on thick description. We frame this as a crucial direction for the application of generative AI and identify five key challenges: preserving context, maintaining interpretive pluralism, integrating perspectives based on lived experience and critical distance, distinguishing qualitative content from quantitative magnitude, and acknowledging meaning as dynamic rather than static.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19540v1">"Do I Trust the AI?" Towards Trustworthy AI-Assisted Diagnosis: Understanding User Perception in LLM-Supported Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI'26), April 13--17, 2026, Barcelona, Spain
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown considerable potential in supporting medical diagnosis. However, their effective integration into clinical workflows is hindered by physicians' difficulties in perceiving and trusting LLM capabilities, which often results in miscalibrated trust. Existing model evaluations primarily emphasize standardized benchmarks and predefined tasks, offering limited insights into clinical reasoning practices. Moreover, research on human-AI collaboration has rarely examined physicians' perceptions of LLMs' clinical reasoning capability. In this work, we investigate how physicians perceive LLMs' capabilities in the clinical reasoning process. We designed clinical cases, collected the corresponding analyses, and obtained evaluations from physicians (N=37) to quantitatively represent their perceived LLM diagnostic capabilities. By comparing the perceived evaluations with benchmark performance, our study highlights the aspects of clinical reasoning that physicians value and underscores the limitations of benchmark-based evaluation. We further discuss the implications of opportunities for enhancing trustworthy collaboration between physicians and LLMs in LLM-supported clinical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11358v2">LLM-Specific Utility: A New Perspective for Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 13 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) is typically optimized for topical relevance, yet its success ultimately depends on whether retrieved passages are useful for a large language model (LLM) to generate correct and complete answers. We argue that such utility is often LLM-specific rather than universal, due to differences in models' knowledge, reasoning, and ability to leverage evidence. We formalize LLM-specific utility as the performance improvement of a target LLM when a passage is provided, compared to answering without evidence. To systematically study LLM-specific utility, we construct a benchmark of LLM-specific gold utilitarian passages for four LLMs (Qwen3-8B/14B/32B and Llama3.1-8B) on three QA datasets (Natural Questions, TriviaQA, and MS MARCO-FQA). Our analysis shows that utilitarian passages are model-dependent and non-transferable: each LLM performs best with its own utilitarian evidence, while evidence optimized for other LLMs is consistently suboptimal. Human-annotated evidence remains a strong general baseline but does not fully match individual LLM utility needs. We further introduce the LLM-specific utility judgment task and find that existing utility-aware selection and scoring methods largely capture model-agnostic usefulness and struggle to reliably estimate LLM-specific utility. Overall, our findings highlight the limitations of current utility-aware retrieval and motivate generator-tailored evidence selection for improving RAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19510v1">ALRM: Agentic LLM for Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently empowered agentic frameworks to exhibit advanced reasoning and planning capabilities. However, their integration in robotic control pipelines remains limited in two aspects: (1) prior \ac{llm}-based approaches often lack modular, agentic execution mechanisms, limiting their ability to plan, reflect on outcomes, and revise actions in a closed-loop manner; and (2) existing benchmarks for manipulation tasks focus on low-level control and do not systematically evaluate multistep reasoning and linguistic variation. In this paper, we propose Agentic LLM for Robot Manipulation (ALRM), an LLM-driven agentic framework for robotic manipulation. ALRM integrates policy generation with agentic execution through a ReAct-style reasoning loop, supporting two complementary modes: Code-asPolicy (CaP) for direct executable control code generation, and Tool-as-Policy (TaP) for iterative planning and tool-based action execution. To enable systematic evaluation, we also introduce a novel simulation benchmark comprising 56 tasks across multiple environments, capturing linguistically diverse instructions. Experiments with ten LLMs demonstrate that ALRM provides a scalable, interpretable, and modular approach for bridging natural language reasoning with reliable robotic execution. Results reveal Claude-4.1-Opus as the top closed-source model and Falcon-H1-7B as the top open-source model under CaP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19503v1">GradPruner: Gradient-Guided Layer Pruning Enabling Efficient Fine-Tuning and Inference for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted by ICLR2026
    </div>
    <details class="paper-abstract">
      Fine-tuning Large Language Models (LLMs) with downstream data is often considered time-consuming and expensive. Structured pruning methods are primarily employed to improve the inference efficiency of pre-trained models. Meanwhile, they often require additional time and memory for training, knowledge distillation, structure search, and other strategies, making efficient model fine-tuning challenging to achieve. To simultaneously enhance the training and inference efficiency of downstream task fine-tuning, we introduce GradPruner, which can prune layers of LLMs guided by gradients in the early stages of fine-tuning. GradPruner uses the cumulative gradients of each parameter during the initial phase of fine-tuning to compute the Initial Gradient Information Accumulation Matrix (IGIA-Matrix) to assess the importance of layers and perform pruning. We sparsify the pruned layers based on the IGIA-Matrix and merge them with the remaining layers. Only elements with the same sign are merged to reduce interference from sign variations. We conducted extensive experiments on two LLMs across eight downstream datasets. Including medical, financial, and general benchmark tasks. The results demonstrate that GradPruner has achieved a parameter reduction of 40% with only a 0.99% decrease in accuracy. Our code is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19487v1">LLM-VA: Resolving the Jailbreak-Overrefusal Trade-off via Vector Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Safety-aligned LLMs suffer from two failure modes: jailbreak (answering harmful inputs) and over-refusal (declining benign queries). Existing vector steering methods adjust the magnitude of answer vectors, but this creates a fundamental trade-off -- reducing jailbreak increases over-refusal and vice versa. We identify the root cause: LLMs encode the decision to answer (answer vector $v_a$) and the judgment of input safety (benign vector $v_b$) as nearly orthogonal directions, treating them as independent processes. We propose LLM-VA, which aligns $v_a$ with $v_b$ through closed-form weight updates, making the model's willingness to answer causally dependent on its safety assessment -- without fine-tuning or architectural changes. Our method identifies vectors at each layer using SVMs, selects safety-relevant layers, and iteratively aligns vectors via minimum-norm weight modifications. Experiments on 12 LLMs demonstrate that LLM-VA achieves 11.45% higher F1 than the best baseline while preserving 95.92% utility, and automatically adapts to each model's safety bias without manual tuning. Code and models are available at https://hotbento.github.io/LLM-VA-Web/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14401v2">The Role of Social Learning and Collective Norm Formation in Fostering Cooperation in LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted at the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026)
    </div>
    <details class="paper-abstract">
      A growing body of multi-agent studies with LLMs explores how norms and cooperation emerge in mixed-motive scenarios, where pursuing individual gain can undermine the collective good. While prior work has explored these dynamics in both richly contextualized simulations and simplified game-theoretic environments, most LLM systems featuring common-pool resource (CPR) games provide agents with explicit reward functions directly tied to their actions. In contrast, human cooperation often emerges without explicit knowledge of the payoff structure or how individual actions translate into long-run outcomes, relying instead on heuristics, communication, and enforcement. We introduce a CPR simulation framework that removes explicit reward signals and embeds cultural-evolutionary mechanisms: social learning (adopting strategies and beliefs from successful peers) and norm-based punishment, grounded in Ostrom's principles of resource governance. Agents also individually learn from the consequences of harvesting, monitoring, and punishing via environmental feedback, enabling norms to emerge endogenously. We establish the validity of our simulation by reproducing key findings from existing studies on human behavior. Building on this, we examine norm evolution across a $2\times2$ grid of environmental and social initialisations (resource-rich vs. resource-scarce; altruistic vs. selfish) and benchmark how agentic societies comprised of different LLMs perform under these conditions. Our results reveal systematic model differences in sustaining cooperation and norm formation, positioning the framework as a rigorous testbed for studying emergent norms in mixed-motive LLM societies. Such analysis can inform the design of AI systems deployed in social and organizational contexts, where alignment with cooperative norms is critical for stability, fairness, and effective governance of AI-mediated environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19447v1">KG-CRAFT: Knowledge Graph-based Contrastive Reasoning with LLMs for Enhancing Automated Fact-checking</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted to publication at the 19th Conference of the European Chapter of the Association for Computational Linguistics, EACL 2026
    </div>
    <details class="paper-abstract">
      Claim verification is a core component of automated fact-checking systems, aimed at determining the truthfulness of a statement by assessing it against reliable evidence sources such as documents or knowledge bases. This work presents KG-CRAFT, a method that improves automatic claim verification by leveraging large language models (LLMs) augmented with contrastive questions grounded in a knowledge graph. KG-CRAFT first constructs a knowledge graph from claims and associated reports, then formulates contextually relevant contrastive questions based on the knowledge graph structure. These questions guide the distillation of evidence-based reports, which are synthesised into a concise summary that is used for veracity assessment by LLMs. Extensive evaluations on two real-world datasets (LIAR-RAW and RAWFC) demonstrate that our method achieves a new state-of-the-art in predictive performance. Comprehensive analyses validate in detail the effectiveness of our knowledge graph-based contrastive reasoning approach in improving LLMs' fact-checking capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19435v1">Ad Insertion in LLM-Generated Responses</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 31 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Sustainable monetization of Large Language Models (LLMs) remains a critical open challenge. Traditional search advertising, which relies on static keywords, fails to capture the fleeting, context-dependent user intents--the specific information, goods, or services a user seeks--embedded in conversational flows. Beyond the standard goal of social welfare maximization, effective LLM advertising imposes additional requirements on contextual coherence (ensuring ads align semantically with transient user intents) and computational efficiency (avoiding user interaction latency), as well as adherence to ethical and regulatory standards, including preserving privacy and ensuring explicit ad disclosure. Although various recent solutions have explored bidding on token-level and query-level, both categories of approaches generally fail to holistically satisfy this multifaceted set of constraints. We propose a practical framework that resolves these tensions through two decoupling strategies. First, we decouple ad insertion from response generation to ensure safety and explicit disclosure. Second, we decouple bidding from specific user queries by using ``genres'' (high-level semantic clusters) as a proxy. This allows advertisers to bid on stable categories rather than sensitive real-time response, reducing computational burden and privacy risks. We demonstrate that applying the VCG auction mechanism to this genre-based framework yields approximately dominant strategy incentive compatibility (DSIC) and individual rationality (IR), as well as approximately optimal social welfare, while maintaining high computational efficiency. Finally, we introduce an "LLM-as-a-Judge" metric to estimate contextual coherence. Our experiments show that this metric correlates strongly with human ratings (Spearman's $ρ\approx 0.66$), outperforming 80% of individual human evaluators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19423v1">UniRec: Unified Multimodal Encoding for LLM-Based Recommendations</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Large language models have recently shown promise for multimodal recommendation, particularly with text and image inputs. Yet real-world recommendation signals extend far beyond these modalities. To reflect this, we formalize recommendation features into four modalities: text, images, categorical features, and numerical attributes, and highlight the unique challenges this heterogeneity poses for LLMs in understanding multimodal information. In particular, these challenges arise not only across modalities but also within them, as attributes such as price, rating, and time may all be numeric yet carry distinct semantic meanings. Beyond this intra-modality ambiguity, another major challenge is the nested structure of recommendation signals, where user histories are sequences of items, each associated with multiple attributes. To address these challenges, we propose UniRec, a unified multimodal encoder for LLM-based recommendation. UniRec first employs modality-specific encoders to produce consistent embeddings across heterogeneous signals. It then adopts a triplet representation, comprising attribute name, type, and value, to separate schema from raw inputs and preserve semantic distinctions. Finally, a hierarchical Q-Former models the nested structure of user interactions while maintaining their layered organization. Across multiple real-world benchmarks, UniRec outperforms state-of-the-art multimodal and LLM-based recommenders by up to 15%, and extensive ablation studies further validate the contributions of each component.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19410v1">Do LLMs Truly Benefit from Longer Context in Automatic Post-Editing?</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Automatic post-editing (APE) aims to refine machine translations by correcting residual errors. Although recent large language models (LLMs) demonstrate strong translation capabilities, their effectiveness for APE--especially under document-level context--remains insufficiently understood. We present a systematic comparison of proprietary and open-weight LLMs under a naive document-level prompting setup, analyzing APE quality, contextual behavior, robustness, and efficiency. Our results show that proprietary LLMs achieve near human-level APE quality even with simple one-shot prompting, regardless of whether document context is provided. While these models exhibit higher robustness to data poisoning attacks than open-weight counterparts, this robustness also reveals a limitation: they largely fail to exploit document-level context for contextual error correction. Furthermore, standard automatic metrics do not reliably reflect these qualitative improvements, highlighting the continued necessity of human evaluation. Despite their strong performance, the substantial cost and latency overheads of proprietary LLMs render them impractical for real-world APE deployment. Overall, our findings elucidate both the promise and current limitations of LLM-based document-aware APE, and point toward the need for more efficient long-context modeling approaches for translation refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19402v1">PROTEUS: SLA-Aware Routing via Lagrangian RL for Multi-LLM Serving Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Production LLM deployments serve diverse workloads where cost and quality requirements vary by customer tier, time of day, and query criticality. Model serving systems accept latency SLOs directly. LLM routers do not. They force operators to tune parameters offline and guess what accuracy might result. The relationship between parameters and outcomes is indirect, non-monotonic, and dataset-dependent. Operators need to specify accuracy targets, not infer them from opaque settings. We present PROTEUS (Polymorphic Router for Operational Target Enforcement with Unified SLA), a router that accepts accuracy targets tau as runtime input. PROTEUS uses Lagrangian dual control. A learned dual variable lambda tracks constraint violations during training and conditions the policy network. This lets the router translate specified tau values into routing decisions that satisfy them. A single trained model serves the full accuracy spectrum without retraining.We evaluate on RouterBench (11 models, 405K queries) and SPROUT (14 models, 45K queries). PROTEUS achieves consistent floor compliance where accuracy meets or exceeds tau. The target-response correlation reaches 0.97 to 0.98. The closest baseline, OmniRouter, meets floors only 22% of the time despite also using Lagrangian optimization. PROTEUS operates across tau in [0.85, 0.95] from a single model. On RouterBench it achieves 90.1% accuracy, within 1.3% of oracle. On SPROUT it achieves 94.0% accuracy, within 4.6% of oracle. Cost savings reach 89.8% versus the best fixed model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.12481v2">SAC-GLAM: Improving Online RL for LLM agents with Soft Actor-Critic and Hindsight Relabeling</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      The past years have seen Large Language Models (LLMs) strive not only as generative models but also as agents solving textual sequential decision-making tasks. When facing complex environments where their zero-shot abilities are insufficient, recent work showed online Reinforcement Learning (RL) could be used for the LLM agent to discover and learn efficient strategies interactively. However, most prior work sticks to on-policy algorithms, which greatly reduces the scope of methods such agents could use for both exploration and exploitation, such as experience replay and hindsight relabeling. Yet, such methods may be key for LLM learning agents, and in particular when designing autonomous intrinsically motivated agents sampling and pursuing their own goals (i.e. autotelic agents). This paper presents and studies an adaptation of Soft Actor-Critic and hindsight relabeling to LLM agents. Our method not only paves the path towards autotelic LLM agents that learn online but can also outperform on-policy methods in more classic multi-goal RL environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.15098v4">TextMineX: Data, Evaluation Framework and Ontology-guided LLM Pipeline for Humanitarian Mine Action</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Humanitarian Mine Action (HMA) addresses the challenge of detecting and removing landmines from conflict regions. Much of the life-saving operational knowledge produced by HMA agencies is buried in unstructured reports, limiting the transferability of information between agencies. To address this issue, we propose TextMineX: the first dataset, evaluation framework and ontology-guided large language model (LLM) pipeline for knowledge extraction from text in the HMA domain. TextMineX structures HMA reports into (subject, relation, object)-triples, thus creating domain-specific knowledge. To ensure real-world relevance, we utilized the dataset from our collaborator Cambodian Mine Action Centre (CMAC). We further introduce a bias-aware evaluation framework that combines human-annotated triples with an LLM-as-Judge protocol to mitigate position bias in reference-free scoring. Our experiments show that ontology-aligned prompts improve extraction accuracy by up to 44.2%, reduce hallucinations by 22.5%, and enhance format adherence by 20.9% compared to baseline models. We publicly release the dataset and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19362v1">Revisiting Parameter Server in LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Accepted in ICLR'26
    </div>
    <details class="paper-abstract">
      Modern data parallel (DP) training favors collective communication over parameter servers (PS) for its simplicity and efficiency under balanced workloads. However, the balanced workload assumption no longer holds in large language model (LLM) post-training due to the high variance in sequence lengths. Under imbalanced workloads, collective communication creates synchronization barriers, leading to under-utilization of devices with smaller workloads. This change in training dynamics calls for a revisit of the PS paradigm for its robustness to such imbalance. We propose \textbf{On-Demand Communication (ODC)}, which adapts PS into Fully Sharded Data Parallel (FSDP) by replacing collective all-gather and reduce-scatter with direct point-to-point communication. Compared to FSDP, ODC reduces the synchronization barrier from once per layer to once per minibatch and decouples the workload on each device so that faster workers are not stalled. It also enables simpler and more effective load balancing at the minibatch level. Across diverse LLM post-training tasks, ODC consistently improves device utilization and training throughput, achieving up to a 36\% speedup over standard FSDP. These results demonstrate that ODC is a superior fit for the prevalent imbalanced workloads in LLM post-training. Our implementation of ODC and integration with FSDP is open-sourced at https://github.com/sail-sg/odc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19334v1">When Benchmarks Leak: Inference-Time Decontamination for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Benchmark-based evaluation is the de facto standard for comparing large language models (LLMs). However, its reliability is increasingly threatened by test set contamination, where test samples or their close variants leak into training data and artificially inflate reported performance. To address this issue, prior work has explored two main lines of mitigation. One line attempts to identify and remove contaminated benchmark items before evaluation, but this inevitably alters the evaluation set itself and becomes unreliable when contamination is moderate or severe. The other line preserves the benchmark and instead suppresses contaminated behavior at evaluation time; however, such interventions often interfere with normal inference and lead to noticeable performance degradation on clean inputs. We propose DeconIEP, a decontamination framework that operates entirely during evaluation by applying small, bounded perturbations in the input embedding space. Guided by a relatively less-contaminated reference model, DeconIEP learns an instance-adaptive perturbation generator that steers the evaluated model away from memorization-driven shortcut pathways. Across multiple open-weight LLMs and benchmarks, extensive empirical results show that DeconIEP achieves strong decontamination effectiveness while incurring only minimal degradation in benign utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19332v1">CaseMaster: Designing and Evaluating a Probe for Oral Case Presentation Training with LLM Assistance</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI '26), April 13--17, 2026, Barcelona, Spain
    </div>
    <details class="paper-abstract">
      Preparing an oral case presentation (OCP) is a crucial skill for medical students, requiring clear communication of patient information, clinical findings, and treatment plans. However, inconsistent student participation and limited guidance can make this task challenging. While Large Language Models (LLMs) can provide structured content to streamline the process, their role in facilitating skill development and supporting medical education integration remains underexplored. To address this, we conducted a formative study with six medical educators and developed CaseMaster, an interactive probe that leverages LLM-generated content tailored to medical education to help users enhance their OCP skills. The controlled study suggests CaseMaster has the potential to both improve presentation quality and reduce workload compared to traditional methods, an implication reinforced by expert feedback. We propose guidelines for educators to develop adaptive, user-centered training methods using LLMs, while considering the implications of integrating advanced technologies into medical education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10609v3">iTIMO: An LLM-empowered Synthesis Dataset for Travel Itinerary Modification</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Addressing itinerary modification is crucial for enhancing the travel experience as it is a frequent requirement during traveling. However, existing research mainly focuses on fixed itinerary planning, leaving modification underexplored due to the scarcity of need-to-modify itinerary data. To bridge this gap, we formally define the itinerary modification task and propose a general pipeline to construct the corresponding dataset, namely iTIMO. This pipeline frames the generation of need-to-modify itinerary data as an intent-driven perturbation task. It instructs large language models to perturb real-world itineraries using three operations: REPLACE, ADD, and DELETE. Each perturbation is grounded in three intents: disruptions of popularity, spatial distance, and category diversity. Furthermore, hybrid evaluation metrics are introduced to ensure perturbation effectiveness. We conduct comprehensive benchmarking on iTIMO to analyze the capabilities and limitations of state-of-the-art LLMs. Overall, iTIMO provides a comprehensive testbed for the modification task, and empowers the evolution of traditional travel recommender systems into adaptive frameworks capable of handling dynamic travel needs. Dataset, code and supplementary materials are available at https://github.com/zelo2/iTIMO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10294v2">Reasoning Hijacking: Subverting LLM Classification via Decision-Criteria Injection</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Current LLM safety research predominantly focuses on mitigating Goal Hijacking, preventing attackers from redirecting a model's high-level objective (e.g., from "summarizing emails" to "phishing users"). In this paper, we argue that this perspective is incomplete and highlight a critical vulnerability in Reasoning Alignment. We propose a new adversarial paradigm: Reasoning Hijacking and instantiate it with Criteria Attack, which subverts model judgments by injecting spurious decision criteria without altering the high-level task goal. Unlike Goal Hijacking, which attempts to override the system prompt, Reasoning Hijacking accepts the high-level goal but manipulates the model's decision-making logic by injecting spurious reasoning shortcut. Though extensive experiments on three different tasks (toxic comment, negative review, and spam detection), we demonstrate that even newest models are prone to prioritize injected heuristic shortcuts over rigorous semantic analysis. The results are consistent over different backbones. Crucially, because the model's "intent" remains aligned with the user's instructions, these attacks can bypass defenses designed to detect goal deviation (e.g., SecAlign, StruQ), exposing a fundamental blind spot in the current safety landscape. Data and code are available at https://github.com/Yuan-Hou/criteria_attack
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19280v1">Group Distributionally Robust Optimization-Driven Reinforcement Learning for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 Keywords: Large Language Models, Reasoning Models, Reinforcement Learning, Distributionally Robust Optimization, GRPO
    </div>
    <details class="paper-abstract">
      Recent progress in Large Language Model (LLM) reasoning is increasingly driven by the refinement of post-training loss functions and alignment strategies. However, standard Reinforcement Learning (RL) paradigms like Group Relative Policy Optimization (GRPO) remain constrained by static uniformity: uniform prompt sampling and a fixed number of rollouts per prompt. For heterogeneous, heavy-tailed reasoning data, this creates structural inefficiencies that waste compute on already-solved patterns while under-training the long tail of hard problems. To address this, we propose Multi-Adversary Group Distributionally Robust Optimization (GDRO), an optimization-first framework that moves beyond uniform reasoning models by dynamically adapting the training distribution. We introduce an Online Difficulty Classifier that partitions prompts into dynamic pass@k difficulty groups. We then propose two independent GDRO games for post-training: (1) Prompt-GDRO, which employs an EMA-debiased multiplicative-weights bandit sampler to target the intensive difficulty margin and upweight persistently hard groups without frequency bias; and (2) Rollout-GDRO, which uses a shadow-price controller to reallocate rollouts across groups, maximizing gradient variance reduction on hard tasks under a fixed mean budget (compute-neutral). We provide no-regret guarantees for both controllers and additionally a variance-proxy analysis motivating a square-root optimal rollout allocation for Rollout-GDRO. We validate our framework on the DAPO 14.1k dataset using Qwen3-Base models. Prompt-GDRO and Rollout-GDRO achieve average relative gains of +10.6% and +10.1%, respectively, in pass@8 accuracy across 1.7B, 4B, and 8B scales compared to the GRPO baseline. Qualitative analysis shows an emergent curriculum: the adversaries shift resources to the evolving reasoning frontier, enhancing the reasoning model's performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22137v3">HybridFlow: Resource-Adaptive Subtask Routing for Efficient Edge-Cloud LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Edge-cloud collaborative inference is becoming a practical necessity for LLM-powered edge devices: on-device models often cannot afford the required reasoning capability, while cloud-only inference could be prohibitively costly and slow under strict latency and token/API budgets. However, existing edge-cloud collaboration methods often route per query or fixed steps simply based-on the estimated difficulty. Such coarse and static heuristics overlook subtask dependencies, missing opportunities for parallel execution and budget-adaptive routing. To this end, we propose \textbf{HybridFlow}, a resource-adaptive edge-cloud inference framework that (i) builds a dependency-aware DAG for each query and executes newly unlocked subtasks in parallel, reducing end-to-end latency; (ii) routes each subtask online to the edge or cloud via a learned benefit--cost utility model that dynamically trades accuracy gains against token/API and latency budgets, thereby reducing unnecessary cloud usage while preserving reasoning quality. Across GPQA, MMLU-Pro, AIME24, and LiveBench-Reasoning, HybridFlow improves the cost-accuracy trade-off, reducing latency and cloud API usage while maintaining competitive accuracy against strong structured reasoning baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19278v1">DART: Diffusion-Inspired Speculative Decoding for Fast LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Speculative decoding is an effective and lossless approach for accelerating LLM inference. However, existing widely adopted model-based draft designs, such as EAGLE3, improve accuracy at the cost of multi-step autoregressive inference, resulting in high drafting latency and ultimately rendering the drafting stage itself a performance bottleneck. Inspired by diffusion-based large language models (dLLMs), we propose DART, which leverages parallel generation to reduce drafting latency. DART predicts logits for multiple future masked positions in parallel within a single forward pass based on hidden states of the target model, thereby eliminating autoregressive rollouts in the draft model while preserving a lightweight design. Based on these parallel logit predictions, we further introduce an efficient tree pruning algorithm that constructs high-quality draft token trees with N-gram-enforced semantic continuity. DART substantially reduces draft-stage overhead while preserving high draft accuracy, leading to significantly improved end-to-end decoding speed. Experimental results demonstrate that DART achieves a 2.03x--3.44x wall-clock time speedup across multiple datasets, surpassing EAGLE3 by 30% on average and offering a practical speculative decoding framework. Code is released at https://github.com/fvliang/DART.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17717v2">The LLM Data Auditor: A Metric-oriented Survey on Quality and Trustworthiness in Evaluating Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2026-01-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful tools for generating data across various modalities. By transforming data from a scarce resource into a controllable asset, LLMs mitigate the bottlenecks imposed by the acquisition costs of real-world data for model training, evaluation, and system iteration. However, ensuring the high quality of LLM-generated synthetic data remains a critical challenge. Existing research primarily focuses on generation methodologies, with limited direct attention to the quality of the resulting data. Furthermore, most studies are restricted to single modalities, lacking a unified perspective across different data types. To bridge this gap, we propose the \textbf{LLM Data Auditor framework}. In this framework, we first describe how LLMs are utilized to generate data across six distinct modalities. More importantly, we systematically categorize intrinsic metrics for evaluating synthetic data from two dimensions: quality and trustworthiness. This approach shifts the focus from extrinsic evaluation, which relies on downstream task performance, to the inherent properties of the data itself. Using this evaluation system, we analyze the experimental evaluations of representative generation methods for each modality and identify substantial deficiencies in current evaluation practices. Based on these findings, we offer concrete recommendations for the community to improve the evaluation of data generation. Finally, the framework outlines methodologies for the practical application of synthetic data across different modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19260v1">"ENERGY STAR" LLM-Enabled Software Engineering Tools</a></div>
    <div class="paper-meta">
      📅 2026-01-27
      | 💬 CAIN 2026 - 5th International Conference on AI Engineering - Software Engineering for AI
    </div>
    <details class="paper-abstract">
      The discussion around AI-Engineering, that is, Software Engineering (SE) for AI-enabled Systems, cannot ignore a crucial class of software systems that are increasingly becoming AI-enhanced: Those used to enable or support the SE process, such as Computer-Aided SE (CASE) tools and Integrated Development Environments (IDEs). In this paper, we study the energy efficiency of these systems. As AI becomes seamlessly available in these tools and, in many cases, is active by default, we are entering a new era with significant implications for energy consumption patterns throughout the Software Development Lifecycle (SDLC). We focus on advanced Machine Learning (ML) capabilities provided by Large Language Models (LLMs). Our proposed approach combines Retrieval-Augmented Generation (RAG) with Prompt Engineering Techniques (PETs) to enhance both the quality and energy efficiency of LLM-based code generation. We present a comprehensive framework that measures real-time energy consumption and inference time across diverse model architectures ranging from 125M to 7B parameters, including GPT-2, CodeLlama, Qwen 2.5, and DeepSeek Coder. These LLMs, chosen for practical reasons, are sufficient to validate the core ideas and provide a proof of concept for more in-depth future analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18306v1">Calibrating Beyond English: Language Diversity for Better Quantized Multilingual LLM</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted to EACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      Quantization is an effective technique for reducing the storage footprint and computational costs of Large Language Models (LLMs), but it often results in performance degradation. Existing post-training quantization methods typically use small, English-only calibration sets; however, their impact on multilingual models remains underexplored. We systematically evaluate eight calibration settings (five single-language and three multilingual mixes) on two quantizers (GPTQ, AWQ) on data from 10 languages. Our findings reveal a consistent trend: non-English and multilingual calibration sets significantly improve perplexity compared to English-only baselines. Specifically, we observe notable average perplexity gains across both quantizers on Llama3.1 8B and Qwen2.5 7B, with multilingual mixes achieving the largest overall reductions of up to 3.52 points in perplexity. Furthermore, our analysis indicates that tailoring calibration sets to the evaluation language yields the largest improvements for individual languages, underscoring the importance of linguistic alignment. We also identify specific failure cases where certain language-quantizer combinations degrade performance, which we trace to differences in activation range distributions across languages. These results highlight that static one-size-fits-all calibration is suboptimal and that tailoring calibration data, both in language and diversity, plays a crucial role in robustly quantizing multilingual LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01289v2">OntoMetric: An Ontology-Driven LLM-Assisted Framework for Automated ESG Metric Knowledge Graph Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Environmental, Social, and Governance (ESG) metric knowledge is inherently structured, connecting industries, reporting frameworks, metric categories, metrics, and calculation models through compositional dependencies, yet in practice this structure remains embedded implicitly in regulatory documents such as SASB, TCFD, and IFRS S2 and rarely exists as an explicit, governed, or machine-actionable artefact. Existing ESG ontologies define formal schemas but do not address scalable population and governance from authoritative regulatory sources, while unconstrained large language model (LLM) extraction frequently produces semantically incorrect entities, hallucinated relationships, and structurally invalid graphs. OntoMetric is an ontology-guided framework for the automated construction and governance of ESG metric knowledge graphs from regulatory documents that operationalises the ESG Metric Knowledge Graph (ESGMKG) ontology as a first-class constraint embedded directly into the extraction and population process. The framework integrates structure-aware segmentation, ontology-constrained LLM extraction enriched with semantic fields and deterministic identifiers, and two-phase validation combining semantic type verification with rule-based schema checking, while preserving segment-level and page-level provenance to ensure traceability to regulatory source text. Evaluation on five ESG regulatory standards shows that ontology-guided extraction achieves 65-90 percent semantic accuracy and over 80 percent schema compliance, compared with 3-10 percent for unconstrained baseline extraction, and yields stable cost efficiency with a cost per validated entity of 0.01-0.02 USD and a 48 times efficiency improvement over baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18292v1">TriPlay-RL: Tri-Role Self-Play Reinforcement Learning for LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      In recent years, safety risks associated with large language models have become increasingly prominent, highlighting the urgent need to mitigate the generation of toxic and harmful content. The mainstream paradigm for LLM safety alignment typically adopts a collaborative framework involving three roles: an attacker for adversarial prompt generation, a defender for safety defense, and an evaluator for response assessment. In this paper, we propose a closed-loop reinforcement learning framework called TriPlay-RL that enables iterative and co-improving collaboration among three roles with near-zero manual annotation. Experimental results show that the attacker preserves high output diversity while achieving a 20%-50% improvement in adversarial effectiveness; the defender attains 10%-30% gains in safety performance without degrading general reasoning capability; and the evaluator continuously refines its fine-grained judgment ability through iterations, accurately distinguishing unsafe responses, simple refusals, and useful guidance. Overall, our framework establishes an efficient and scalable paradigm for LLM safety alignment, enabling continuous co-evolution within a unified learning loop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18282v1">Think-Augmented Function Calling: Improving LLM Parameter Accuracy Through Embedded Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in function calling for autonomous agents, yet current mechanisms lack explicit reasoning transparency during parameter generation, particularly for complex functions with interdependent parameters. While existing approaches like chain-of-thought prompting operate at the agent level, they fail to provide fine-grained reasoning guidance for individual function parameters. To address these limitations, we propose Think-Augmented Function Calling (TAFC), a novel framework that enhances function calling accuracy through explicit reasoning at both function and parameter levels. Our method introduces a universal "think" parameter augmentation that enables models to articulate their decision-making process, with dynamic optimization for parameter descriptions to improve reasoning quality. For complex parameters, TAFC automatically triggers granular reasoning based on complexity scoring, ensuring appropriate justification for critical decisions. Additionally, we propose reasoning-guided optimization to align generated reasoning with human expectations. TAFC requires no architectural modifications to existing LLMs while maintaining full API compatibility. Evaluation on ToolBench across proprietary and open-source models demonstrates significant improvements in parameter generation accuracy and reasoning coherence for multi-parameter functions, while providing enhanced interpretability for debugging AI agent behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15727v2">Towards Automated Kernel Generation in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 10 pages, 1 figure
    </div>
    <details class="paper-abstract">
      The performance of modern AI systems is fundamentally constrained by the quality of their underlying kernels, which translate high-level algorithmic semantics into low-level hardware operations. Achieving near-optimal kernels requires expert-level understanding of hardware architectures and programming models, making kernel engineering a critical but notoriously time-consuming and non-scalable process. Recent advances in large language models (LLMs) and LLM-based agents have opened new possibilities for automating kernel generation and optimization. LLMs are well-suited to compress expert-level kernel knowledge that is difficult to formalize, while agentic systems further enable scalable optimization by casting kernel development as an iterative, feedback-driven loop. Rapid progress has been made in this area. However, the field remains fragmented, lacking a systematic perspective for LLM-driven kernel generation. This survey addresses this gap by providing a structured overview of existing approaches, spanning LLM-based approaches and agentic optimization workflows, and systematically compiling the datasets and benchmarks that underpin learning and evaluation in this domain. Moreover, key open challenges and future research directions are further outlined, aiming to establish a comprehensive reference for the next generation of automated kernel optimization. To keep track of this field, we maintain an open-source GitHub repository at https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18255v1">Beyond Retention: Orchestrating Structural Safety and Plasticity in Continual Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Continual learning in Large Language Models (LLMs) faces the critical challenge of balancing stability (retaining old knowledge) and plasticity (learning new tasks). While Experience Replay (ER) is a standard countermeasure against catastrophic forgetting, its impact across diverse capabilities remains underexplored. In this work, we uncover a critical dichotomy in ER's behavior: while it induces positive backward transfer on robust, unstructured tasks (e.g., boosting performance on previous NLP classification tasks through repeated rehearsal), it causes severe negative transfer on fragile, structured domains like code generation (e.g., a significant relative drop in coding accuracy). This reveals that ER trades structural integrity for broad consolidation. To address this dilemma, we propose \textbf{Orthogonal Subspace Wake-up (OSW)}. OSW identifies essential parameter subspaces of previous tasks via a brief "wake-up" phase and enforces orthogonal updates for new tasks, providing a mathematically grounded "safety guarantee" for established knowledge structures. Empirical results across a diverse four-task sequence demonstrate that OSW uniquely succeeds in preserving fragile coding abilities where Replay fails, while simultaneously maintaining high plasticity for novel tasks. Our findings emphasize the necessity of evaluating structural safety alongside average retention in LLM continual learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18253v1">BoRP: Bootstrapped Regression Probing for Scalable and Human-Aligned LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 This is a pre-print
    </div>
    <details class="paper-abstract">
      Accurate evaluation of user satisfaction is critical for iterative development of conversational AI. However, for open-ended assistants, traditional A/B testing lacks reliable metrics: explicit feedback is sparse, while implicit metrics are ambiguous. To bridge this gap, we introduce BoRP (Bootstrapped Regression Probing), a scalable framework for high-fidelity satisfaction evaluation. Unlike generative approaches, BoRP leverages the geometric properties of LLM latent space. It employs a polarization-index-based bootstrapping mechanism to automate rubric generation and utilizes Partial Least Squares (PLS) to map hidden states to continuous scores. Experiments on industrial datasets show that BoRP (Qwen3-8B/14B) significantly outperforms generative baselines (even Qwen3-Max) in alignment with human judgments. Furthermore, BoRP reduces inference costs by orders of magnitude, enabling full-scale monitoring and highly sensitive A/B testing via CUPED.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18241v1">TAM-Eval: Evaluating LLMs for Automated Unit Test Maintenance</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted for publication at the 9th Workshop on Validation, Analysis and Evolution of Software Tests (VST 2026), co-located with the the 33rd IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER 2026)
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have shown promise in software engineering, their application to unit testing remains largely confined to isolated test generation or oracle prediction, neglecting the broader challenge of test suite maintenance. We introduce TAM-Eval (Test Automated Maintenance Evaluation), a framework and benchmark designed to evaluate model performance across three core test maintenance scenarios: creation, repair, and updating of test suites. Unlike prior work limited to function-level tasks, TAM-Eval operates at the test file level, while maintaining access to full repository context during isolated evaluation, better reflecting real-world maintenance workflows. Our benchmark comprises 1,539 automatically extracted and validated scenarios from Python, Java, and Go projects. TAM-Eval supports system-agnostic evaluation of both raw LLMs and agentic workflows, using a reference-free protocol based on test suite pass rate, code coverage, and mutation testing. Empirical results indicate that state-of-the-art LLMs have limited capabilities in realistic test maintenance processes and yield only marginal improvements in test effectiveness. We release TAM-Eval as an open-source framework to support future research in automated software testing. Our data and code are publicly available at https://github.com/trndcenter/TAM-Eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00847v4">Pay for The Second-Best Service: A Game-Theoretic Approach Against Dishonest LLM Providers</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 To appear in WWW 2026; 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) through Application Programming Interfaces (APIs) induces a critical vulnerability: the potential for dishonest manipulation by service providers. This manipulation can manifest in various forms, such as secretly substituting a proclaimed high-performance model with a low-cost alternative, or inflating responses with meaningless tokens to increase billing. This work tackles the issue through the lens of algorithmic game theory and mechanism design. We are the first to propose a formal economic model for a realistic user-provider ecosystem, where a user can iteratively delegate $T$ queries to multiple model providers, and providers can engage in a range of strategic behaviors. As our central contribution, we prove that for a continuous strategy space and any $ε\in(0,\frac12)$, there exists an approximate incentive-compatible mechanism with an additive approximation ratio of $O(T^{1-ε}\log T)$, and a guaranteed quasi-linear second-best user utility. We also prove an impossibility result, stating that no mechanism can guarantee an expected user utility that is asymptotically better than our mechanism. Furthermore, we demonstrate the effectiveness of our mechanism in simulation experiments with real-world API settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18225v1">ShopSimulator: Evaluating and Exploring RL-Driven LLM Agent for Shopping Assistants</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents are increasingly deployed in e-commerce shopping. To perform thorough, user-tailored product searches, agents should interpret personal preferences, engage in multi-turn dialogues, and ultimately retrieve and discriminate among highly similar products. However, existing research has yet to provide a unified simulation environment that consistently captures all of these aspects, and always focuses solely on evaluation benchmarks without training support. In this paper, we introduce ShopSimulator, a large-scale and challenging Chinese shopping environment. Leveraging ShopSimulator, we evaluate LLMs across diverse scenarios, finding that even the best-performing models achieve less than 40% full-success rate. Error analysis reveals that agents struggle with deep search and product selection in long trajectories, fail to balance the use of personalization cues, and to effectively engage with users. Further training exploration provides practical guidance for overcoming these weaknesses, with the combination of supervised fine-tuning (SFT) and reinforcement learning (RL) yielding significant performance improvements. Code and data will be released at https://github.com/ShopAgent-Team/ShopSimulator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16781v2">Persuasion Tokens for Editing Factual Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted at EACL Main 2026
    </div>
    <details class="paper-abstract">
      In-context knowledge editing (IKE) is a promising technique for updating Large Language Models (LLMs) with new information. However, IKE relies on lengthy, fact-specific demonstrations which are costly to create and consume significant context window space. In this paper, we introduce persuasion tokens (P-Tokens) -- special tokens trained to replicate the effect of IKE demonstrations, enabling efficient knowledge editing without requiring fact-specific demonstrations. We evaluate P-Tokens across two editing datasets and three LLMs, demonstrating performance comparable to, and often exceeding, IKE. We further find that editing performance is robust to distractors with small negative effects to neighboring facts, and that increasing the number of P-Tokens improves performance. Our work addresses key limitations of IKE and provides a more practical and scalable alternative for editing LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18220v1">LLM-ForcedAligner: A Non-Autoregressive and Accurate LLM-Based Forced Aligner for Multilingual and Long-Form Speech</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Forced alignment (FA) predicts start and end timestamps for words or characters in speech, but existing methods are language-specific and prone to cumulative temporal shifts. The multilingual speech understanding and long-sequence processing abilities of speech large language models (SLLMs) make them promising for FA in multilingual, crosslingual, and long-form speech settings. However, directly applying the next-token prediction paradigm of SLLMs to FA results in hallucinations and slow inference. To bridge the gap, we propose LLM-ForcedAligner, reformulating FA as a slot-filling paradigm: timestamps are treated as discrete indices, and special timestamp tokens are inserted as slots into the transcript. Conditioned on the speech embeddings and the transcript with slots, the SLLM directly predicts the time indices at slots. During training, causal attention masking with non-shifted input and label sequences allows each slot to predict its own timestamp index based on itself and preceding context, with loss computed only at slot positions. Dynamic slot insertion enables FA at arbitrary positions. Moreover, non-autoregressive inference is supported, avoiding hallucinations and improving speed. Experiments across multilingual, crosslingual, and long-form speech scenarios show that LLM-ForcedAligner achieves a 69%~78% relative reduction in accumulated averaging shift compared with prior methods. The checkpoint and inference code will be released later.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18217v1">Paying Less Generalization Tax: A Cross-Domain Generalization Study of RL Training for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Generalist LLM agents are often post-trained on a narrow set of environments but deployed across far broader, unseen domains. In this work, we investigate the challenge of agentic post-training when the eventual test domains are unknown. Specifically, we analyze which properties of reinforcement learning (RL) environments and modeling choices have the greatest influence on out-of-domain performance. First, we identify two environment axes that strongly correlate with cross-domain generalization: (i) state information richness, i.e., the amount of information for the agent to process from the state, and (ii) planning complexity, estimated via goal reachability and trajectory length under a base policy. Notably, domain realism and text-level similarity are not the primary factors; for instance, the simple grid-world domain Sokoban leads to even stronger generalization in SciWorld than the more realistic ALFWorld. Motivated by these findings, we further show that increasing state information richness alone can already effectively improve cross-domain robustness. We propose a randomization technique, which is low-overhead and broadly applicable: add small amounts of distractive goal-irrelevant features to the state to make it richer without altering the task. Beyond environment-side properties, we also examine several modeling choices: (a) SFT warmup or mid-training helps prevent catastrophic forgetting during RL but undermines generalization to domains that are not included in the mid-training datamix; and (b) turning on step-by-step thinking during RL, while not always improving in-domain performance, plays a crucial role in preserving generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10132v2">Is More Context Always Better? Examining LLM Reasoning Capability for Time Interval Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted at The Web Conference 2026 (WWW 2026)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities in reasoning and prediction across different domains. Yet, their ability to infer temporal regularities from structured behavioral data remains underexplored. This paper presents a systematic study investigating whether LLMs can predict time intervals between recurring user actions, such as repeated purchases, and how different levels of contextual information shape their predictive behavior. Using a simple but representative repurchase scenario, we benchmark state-of-the-art LLMs in zero-shot settings against both statistical and machine-learning models. Two key findings emerge. First, while LLMs surpass lightweight statistical baselines, they consistently underperform dedicated machine-learning models, showing their limited ability to capture quantitative temporal structure. Second, although moderate context can improve LLM accuracy, adding further user-level detail degrades performance. These results challenge the assumption that "more context leads to better reasoning". Our study highlights fundamental limitations of today's LLMs in structured temporal inference and offers guidance for designing future context-aware hybrid models that integrate statistical precision with linguistic flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.07314v2">MEDIC: Comprehensive Evaluation of Leading Indicators for LLM Safety and Utility in Clinical Applications</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) achieve superhuman performance on standardized medical licensing exams, these static benchmarks have become saturated and increasingly disconnected from the functional requirements of clinical workflows. To bridge the gap between theoretical capability and verified utility, we introduce MEDIC, a comprehensive evaluation framework establishing leading indicators across various clinical dimensions. Beyond standard question-answering, we assess operational capabilities using deterministic execution protocols and a novel Cross-Examination Framework (CEF), which quantifies information fidelity and hallucination rates without reliance on reference texts. Our evaluation across a heterogeneous task suite exposes critical performance trade-offs: we identify a significant knowledge-execution gap, where proficiency in static retrieval does not predict success in operational tasks such as clinical calculation or SQL generation. Furthermore, we observe a divergence between passive safety (refusal) and active safety (error detection), revealing that models fine-tuned for high refusal rates often fail to reliably audit clinical documentation for factual accuracy. These findings demonstrate that no single architecture dominates across all dimensions, highlighting the necessity of a portfolio approach to clinical model deployment. As part of this investigation, we released a public leaderboard on Hugging Face.\footnote{https://huggingface.co/spaces/m42-health/MEDIC-Benchmark}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12869v2">On the Fundamental Limits of LLMs at Scale</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Submitted to TMLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have benefited enormously from scaling, yet these gains are bounded by five fundamental limitations: (1) hallucination, (2) context compression, (3) reasoning degradation, (4) retrieval fragility, and (5) multimodal misalignment. While existing surveys describe these phenomena empirically, they lack a rigorous theoretical synthesis connecting them to the foundational limits of computation, information, and learning. This work closes that gap by presenting a unified, proof-informed framework that formalizes the innate theoretical ceilings of LLM scaling. First, computability and uncomputability imply an irreducible residue of error: for any computably enumerable model family, diagonalization guarantees inputs on which some model must fail, and undecidable queries (e.g., halting-style tasks) induce infinite failure sets for all computable predictors. Second, information-theoretic and statistical constraints bound attainable accuracy even on decidable tasks, finite description length enforces compression error, and long-tail factual knowledge requires prohibitive sample complexity. Third, geometric and computational effects compress long contexts far below their nominal size due to positional under-training, encoding attenuation, and softmax crowding. We further show how likelihood-based training favors pattern completion over inference, how retrieval under token limits suffers from semantic drift and coupling noise, and how multimodal scaling inherits shallow cross-modal alignment. Across sections, we pair theorems and empirical evidence to outline where scaling helps, where it saturates, and where it cannot progress, providing both theoretical foundations and practical mitigation paths like bounded-oracle retrieval, positional curricula, and sparse or hierarchical attention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22922v4">Improving Human Verification of LLM Reasoning through Interactive Explanation Interfaces</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 19 pages, 14 figures
    </div>
    <details class="paper-abstract">
      The reasoning capabilities of Large Language Models (LLMs) have led to their increasing employment in several critical applications, particularly education, where they support problem-solving, tutoring, and personalized study. Chain-of-thought (CoT) reasoning capabilities [1, 2] are well-known to help LLMs decompose a problem into steps and explore the solution spaces more effectively, leading to impressive performance on mathematical and reasoning benchmarks. As the length of CoT tokens per question increases substantially to even thousands of tokens per question [ 1], it is unknown how users could comprehend LLM reasoning and detect errors or hallucinations. To address this problem and understand how reasoning can improve human-AI interaction, we present three new interactive reasoning interfaces: interactive CoT (iCoT), interactive Program-of-Thought (iPoT), and interactive Graph (iGraph). That is, we ask LLMs themselves to generate an interactive web interface wrapped around the original CoT content, which may be presented in text (iCoT), graphs (iGraph) or code (iPoT). This interface allows users to interact with and provide a novel experience in reading and validating the reasoning chains of LLMs. Across a study of 125 participants, interactive interfaces significantly improve user performance. Specifically, iGraph users score the highest error detection rate (85.6%), followed by iPoT (82.5%), iCoT (80.6%), all outperforming standard CoT (73.5%). Interactive interfaces also lead to faster user validation time-iGraph users are faster (57.9 secs per question) than the users of iCoT and iPoT (60 secs) and the standard CoT (64.7 secs). A post-study questionnaire shows that users prefer iGraph, citing its superior ability to enable them to follow the LLM's reasoning. We discuss the implications of these results and provide recommendations for the future design of reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18150v1">FP8-RL: A Practical and Stable Low-Precision Stack for LLM Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) for large language models (LLMs) is increasingly bottlenecked by rollout (generation), where long output sequence lengths make attention and KV-cache memory dominate end-to-end step time. FP8 offers an attractive lever for accelerating RL by reducing compute cost and memory traffic during rollout, but applying FP8 in RL introduces unique engineering and algorithmic challenges: policy weights change every step (requiring repeated quantization and weight synchronization into the inference engine) and low-precision rollouts can deviate from the higher-precision policy assumed by the trainer, causing train-inference mismatch and potential instability. This report presents a practical FP8 rollout stack for LLM RL, implemented in the veRL ecosystem with support for common training backends (e.g., FSDP/Megatron-LM) and inference engines (e.g., vLLM/SGLang). We (i) enable FP8 W8A8 linear-layer rollout using blockwise FP8 quantization, (ii) extend FP8 to KV-cache to remove long-context memory bottlenecks via per-step QKV scale recalibration, and (iii) mitigate mismatch using importance-sampling-based rollout correction (token-level TIS/MIS variants). Across dense and MoE models, these techniques deliver up to 44% rollout throughput gains while preserving learning behavior comparable to BF16 baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18146v1">Think When Needed: Model-Aware Reasoning Routing for LLM-based Ranking</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied to ranking tasks in retrieval and recommendation. Although reasoning prompting can enhance ranking utility, our preliminary exploration reveals that its benefits are inconsistent and come at a substantial computational cost, suggesting that when to reason is as crucial as how to reason. To address this issue, we propose a reasoning routing framework that employs a lightweight, plug-and-play router head to decide whether to use direct inference (Non-Think) or reasoning (Think) for each instance before generation. The router head relies solely on pre-generation signals: i) compact ranking-aware features (e.g., candidate dispersion) and ii) model-aware difficulty signals derived from a diagnostic checklist reflecting the model's estimated need for reasoning. By leveraging these features before generation, the router outputs a controllable token that determines whether to apply the Think mode. Furthermore, the router can adaptively select its operating policy along the validation Pareto frontier during deployment, enabling dynamic allocation of computational resources toward instances most likely to benefit from Think under varying system constraints. Experiments on three public ranking datasets with different scales of open-source LLMs show consistent improvements in ranking utility with reduced token consumption (e.g., +6.3\% NDCG@10 with -49.5\% tokens on MovieLens with Qwen3-4B), demonstrating reasoning routing as a practical solution to the accuracy-efficiency trade-off.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09053v2">Who Fails Where? LLM and Human Error Patterns in Endometriosis Ultrasound Report Extraction</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      In this study, we evaluate a locally-deployed large-language model (LLM) to convert unstructured endometriosis transvaginal ultrasound (eTVUS) scan reports into structured data for imaging informatics workflows. Across 49 eTVUS reports, we compared three LLMs (7B/8B and a 20B-parameter model) against expert human extraction. The 20B model achieved a mean accuracy of 86.02%, substantially outperforming smaller models and confirming the importance of scale in handling complex clinical text. Crucially, we identified a highly complementary error profile: the LLM excelled at syntactic consistency (e.g., date/numeric formatting) where humans faltered, while human experts provided superior semantic and contextual interpretation. We also found that the LLM's semantic errors were fundamental limitations that could not be mitigated by simple prompt engineering. These findings strongly support a human-in-the-loop (HITL) workflow in which the on-premise LLM serves as a collaborative tool, not a full replacement. It automates routine structuring and flags potential human errors, enabling imaging specialists to focus on high-level semantic validation. We discuss implications for structured reporting and interactive AI systems in clinical practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13481v2">neuralFOMO: Can LLMs Handle Being Second Best? Measuring Envy-Like Preferences in Multi-Agent Settings</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Envy shapes competitiveness and cooperation in human groups, yet its role in large language model interactions remains largely unexplored. As LLMs increasingly operate in multi-agent settings, it is important to examine whether they exhibit envy-like preferences under social comparison. We evaluate LLM behavior across two scenarios: (1) a point-allocation game testing sensitivity to relative versus absolute payoff, and (2) comparative evaluations across general and contextual settings. To ground our analysis in psychological theory, we adapt four established psychometric questionnaires spanning general, domain-specific, workplace, and sibling-based envy. Our results reveal heterogeneous envy-like patterns across models and contexts, with some models sacrificing personal gain to reduce a peer's advantage, while others prioritize individual maximization. These findings highlight competitive dispositions as a design and safety consideration for multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18119v1">Beyond Text-to-SQL: Can LLMs Really Debug Enterprise ETL SQL?</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      SQL is central to enterprise data engineering, yet generating fully correct SQL code in a single attempt remains difficult, even for experienced developers and advanced text-to-SQL LLMs, often requiring multiple debugging iterations. We introduce OurBench, the first benchmark for enterprise-level SQL reasoning and debugging. Our benchmark is built on two key innovations: (1) an automated construction workflow that uses reverse engineering to systematically inject realistic bugs into large-scale SQL code, enabling scalable and diverse benchmark generation; and (2) an execution-free evaluation framework tailored to enterprise settings, providing fast, accurate, and resource-efficient assessment. OurBench comprises 469 OurBenchSyn queries featuring syntax errors with explicit error messages, and 516 OurBenchSem queries targeting semantic errors in which the code fails to meet user intent. The queries are highly complex, averaging over 140 lines and featuring deep and wide abstract syntax trees. Evaluation of nearly 30 LLMs reveals a substantial performance gap: the best-performing model, Claude-4-Sonnet, achieves only 36.46 percent accuracy on OurBenchSyn and 32.17 percent on OurBenchSem, while most models score below 20 percent. We further explore four solution strategies, identify key challenges, and outline promising directions for enterprise SQL debugging with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18116v1">FABLE: Forest-Based Adaptive Bi-Path LLM-Enhanced Retrieval for Multi-Document Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      The rapid expansion of long-context Large Language Models (LLMs) has reignited debate on whether Retrieval-Augmented Generation (RAG) remains necessary. However, empirical evidence reveals persistent limitations of long-context inference, including the lost-in-the-middle phenomenon, high computational cost, and poor scalability for multi-document reasoning. Conversely, traditional RAG systems, while efficient, are constrained by flat chunk-level retrieval that introduces semantic noise and fails to support structured cross-document synthesis. We present \textbf{FABLE}, a \textbf{F}orest-based \textbf{A}daptive \textbf{B}i-path \textbf{L}LM-\textbf{E}nhanced retrieval framework that integrates LLMs into both knowledge organization and retrieval. FABLE constructs LLM-enhanced hierarchical forest indexes with multi-granularity semantic structures, then employs a bi-path strategy combining LLM-guided hierarchical traversal with structure-aware propagation for fine-grained evidence acquisition, with explicit budget control for adaptive efficiency trade-offs. Extensive experiments demonstrate that FABLE consistently outperforms SOTA RAG methods and achieves comparable accuracy to full-context LLM inference with up to 94\% token reduction, showing that long-context LLMs amplify rather than fully replace the need for structured retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18110v1">AttenMIA: LLM Membership Inference Attack through Attention Signals</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed to enable or improve a multitude of real-world applications. Given the large size of their training data sets, their tendency to memorize training data raises serious privacy and intellectual property concerns. A key threat is the membership inference attack (MIA), which aims to determine whether a given sample was included in the model's training set. Existing MIAs for LLMs rely primarily on output confidence scores or embedding-based features, but these signals are often brittle, leading to limited attack success. We introduce AttenMIA, a new MIA framework that exploits self-attention patterns inside the transformer model to infer membership. Attention controls the information flow within the transformer, exposing different patterns for memorization that can be used to identify members of the dataset. Our method uses information from attention heads across layers and combines them with perturbation-based divergence metrics to train an effective MIA classifier. Using extensive experiments on open-source models including LLaMA-2, Pythia, and Opt models, we show that attention-based features consistently outperform baselines, particularly under the important low-false-positive metric (e.g., achieving up to 0.996 ROC AUC & 87.9% TPR@1%FPR on the WikiMIA-32 benchmark with Llama2-13b). We show that attention signals generalize across datasets and architectures, and provide a layer- and head-level analysis of where membership leakage is most pronounced. We also show that using AttenMIA to replace other membership inference attacks in a data extraction framework results in training data extraction attacks that outperform the state of the art. Our findings reveal that attention mechanisms, originally introduced to enhance interpretability, can inadvertently amplify privacy risks in LLMs, underscoring the need for new defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18096v1">Enhancing LLM-based Recommendation with Preference Hint Discovery from Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      LLMs have garnered substantial attention in recommendation systems. Yet they fall short of traditional recommenders when capturing complex preference patterns. Recent works have tried integrating traditional recommendation embeddings into LLMs to resolve this issue, yet a core gap persists between their continuous embedding and discrete semantic spaces. Intuitively, textual attributes derived from interactions can serve as critical preference rationales for LLMs' recommendation logic. However, directly inputting such attribute knowledge presents two core challenges: (1) Deficiency of sparse interactions in reflecting preference hints for unseen items; (2) Substantial noise introduction from treating all attributes as hints. To this end, we propose a preference hint discovery model based on the interaction-integrated knowledge graph, enhancing LLM-based recommendation. It utilizes traditional recommendation principles to selectively extract crucial attributes as hints. Specifically, we design a collaborative preference hint extraction schema, which utilizes semantic knowledge from similar users' explicit interactions as hints for unseen items. Furthermore, we develop an instance-wise dual-attention mechanism to quantify the preference credibility of candidate attributes, identifying hints specific to each unseen item. Using these item- and user-based hints, we adopt a flattened hint organization method to shorten input length and feed the textual hint information to the LLM for commonsense reasoning. Extensive experiments on both pair-wise and list-wise recommendation tasks verify the effectiveness of our proposed framework, indicating an average relative improvement of over 3.02% against baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18091v1">From LLMs to LRMs: Rethinking Pruning for Reasoning-Centric Models</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 18 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly costly to deploy, motivating extensive research on model pruning. However, most existing studies focus on instruction-following LLMs, leaving it unclear whether established pruning strategies transfer to reasoning-augmented models that explicitly generate long intermediate reasoning traces. In this work, we conduct a controlled study of pruning for both instruction-following ($\textbf{LLM-instruct}$) and reasoning-augmented ($\textbf{LLM-think}$) models. To isolate the effects of pruning, we align pruning calibration and post-pruning recovery data with each model's original training distribution, which we show yields more stable and reliable pruning behavior. We evaluate static depth pruning, static width pruning, and dynamic pruning across 17 tasks spanning classification, generation, and reasoning. Our results reveal clear paradigm-dependent differences: depth pruning outperforms width pruning on classification tasks, while width pruning is more robust for generation and reasoning. Moreover, static pruning better preserves reasoning performance, whereas dynamic pruning excels on classification and generation but remains challenging for long-chain reasoning. These findings underscore the need for pruning strategies that explicitly account for the distinct characteristics of reasoning-augmented LLMs. Our code is publicly available at https://github.com/EIT-NLP/LRM-Pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16432v2">iPDB -- Optimizing SQL Queries with ML and LLM Predicates</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Structured Query Language (SQL) has remained the standard query language for databases. SQL is highly optimized for processing structured data laid out in relations. Meanwhile, in the present application development landscape, it is highly desirable to utilize the power of learned models to perform complex tasks. Large language models (LLMs) have been shown to understand and extract information from unstructured textual data. However, SQL as a query language and accompanying relational database systems are either incompatible or inefficient for workloads that require leveraging learned models. This results in complex engineering and multiple data migration operations that move data between the data sources and the model inference platform. In this paper, we present iPDB, a relational system that supports in-database machine learning (ML) and large language model (LLM) inferencing using extended SQL syntax. In iPDB, LLMs and ML calls can function as semantic projects, as predicates to perform semantic selects and semantic joins, or for semantic grouping in group-by clauses. iPDB has a novel relational predict operator and semantic query optimizations that enable users to write and efficiently execute semantic SQL queries, outperforming the state-of-the-art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.13358v5">Bridging the Editing Gap in LLMs: FineEdit for Precise and Targeted Text Modifications</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 We resolved some issues in this paper
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced natural language processing, demonstrating strong capabilities in tasks such as text generation, summarization, and reasoning. Recently, their potential for automating precise text editing tasks across specialized domains, such as programming code, LaTeX, and structured database languages, has gained attention. However, current state-of-the-art LLMs still struggle with executing precise, instruction-driven edits, particularly when structural accuracy and strict adherence to domain conventions are required. To address these challenges, we introduce InstrEditBench, an automated benchmark dataset comprising over 30,000 structured editing tasks spanning diverse domains, including Wikipedia articles, LaTeX documents, source code, and database languages. Using this benchmark, we develop FineEdit, a specialized editing model explicitly trained for accurate, context-aware text modifications. Experimental evaluations demonstrate that FineEdit outperforms state-of-the-art models, achieving improvements of approximately 10\% over Gemini models on single-turn edits, up to 30\% over Llama-3.2-3B, and exceeding Mistral-7B-OpenOrca performance by over 40\% on direct editing tasks. FineEdit also effectively generalizes to realistic multi-turn editing scenarios, highlighting its practical applicability. To facilitate further research and reproducibility, we release FineEdit at https://github.com/StuRinDQB/FineEdit} and https://huggingface.co/datasets/YimingZeng/FineEdit_bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18077v1">Sparks of Cooperative Reasoning: LLMs as Strategic Hanabi Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Cooperative reasoning under incomplete information remains challenging for both humans and multi-agent systems. The card game Hanabi embodies this challenge, requiring theory-of-mind reasoning and strategic communication. We benchmark 17 state-of-the-art LLM agents in 2-5 player games and study the impact of context engineering across model scales (4B to 600B+) to understand persistent coordination failures and robustness to scaffolding: from a minimal prompt with only explicit card details (Watson setting), to scaffolding with programmatic, Bayesian-motivated deductions (Sherlock setting), to multi-turn state tracking via working memory (Mycroft setting). We show that (1) agents can maintain an internal working memory for state tracking and (2) cross-play performance between different LLMs smoothly interpolates with model strength. In the Sherlock setting, the strongest reasoning models exceed 15 points on average across player counts, yet still trail experienced humans and specialist Hanabi agents, both consistently scoring above 20. We release the first public Hanabi datasets with annotated trajectories and move utilities: (1) HanabiLogs, containing 1,520 full game logs for instruction tuning, and (2) HanabiRewards, containing 560 games with dense move-level value annotations for all candidate moves. Supervised and RL finetuning of a 4B open-weight model (Qwen3-Instruct) on our datasets improves cooperative Hanabi play by 21% and 156% respectively, bringing performance to within ~3 points of a strong proprietary reasoning model (o4-mini) and surpassing the best non-reasoning model (GPT-4.1) by 52%. The HanabiRewards RL-finetuned model further generalizes beyond Hanabi, improving performance on a cooperative group-guessing benchmark by 11%, temporal reasoning on EventQA by 6.4%, instruction-following on IFBench-800K by 1.7 Pass@10, and matching AIME 2025 mathematical reasoning Pass@10.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18067v1">EvolVE: Evolutionary Search for LLM-based Verilog Generation and Optimization</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 17 pages, 6 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Verilog's design cycle is inherently labor-intensive and necessitates extensive domain expertise. Although Large Language Models (LLMs) offer a promising pathway toward automation, their limited training data and intrinsic sequential reasoning fail to capture the strict formal logic and concurrency inherent in hardware systems. To overcome these barriers, we present EvolVE, the first framework to analyze multiple evolution strategies on chip design tasks, revealing that Monte Carlo Tree Search (MCTS) excels at maximizing functional correctness, while Idea-Guided Refinement (IGR) proves superior for optimization. We further leverage Structured Testbench Generation (STG) to accelerate the evolutionary process. To address the lack of complex optimization benchmarks, we introduce IC-RTL, targeting industry-scale problems derived from the National Integrated Circuit Contest. Evaluations establish EvolVE as the new state-of-the-art, achieving 98.1% on VerilogEval v2 and 92% on RTLLM v2. Furthermore, on the industry-scale IC-RTL suite, our framework surpasses reference implementations authored by contest participants, reducing the Power, Performance, Area (PPA) product by up to 66% in Huffman Coding and 17% in the geometric mean across all problems. The source code of the IC-RTL benchmark is available at https://github.com/weiber2002/ICRTL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18053v1">Addressing LLM Diversity by Infusing Random Concepts</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to produce outputs with limited diversity. In this work, we study whether infusing random concepts in the prompts can improve the diversity of the generated outputs. To benchmark the approach, we design a systematic evaluation protocol which involves prompting an LLM with questions of the form "Name 10 Hollywood actors", and analyzing diversity measures of the resulting LLM outputs. Our experiments on multiple LLMs show that prepending random words/sentences unrelated to the prompt result in greater diversity in the outputs of LLMs. We believe that this promising result and the evaluation protocol opens up interesting avenues for future work, such as how infusing randomness into LLMs could be applied to other domains. Further, the evaluation protocol could also inspire research into benchmarking LLM diversity more systematically.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.14852v2">Agentic Plan Caching: Test-Time Memory for Fast and Cost-Efficient LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 NeurIPS 2025. 27 pages
    </div>
    <details class="paper-abstract">
      LLM-based agent applications have shown increasingly remarkable capabilities in complex workflows but incur substantial costs and latency due to extensive planning and reasoning requirements. Existing LLM caching techniques (like context caching and semantic caching), primarily designed for serving chatbots, are insufficient for agent applications where outputs depend on external data and environmental contexts. We propose Agentic Plan Caching (APC), a novel test-time memory that extracts, stores, adapts, and reuses structured plan templates from planning stages of agent applications across semantically similar tasks to reduce the cost and latency of serving. Unlike traditional semantic caching, our system extracts plan templates from completed agent executions at test-time, employs keyword extraction to match new requests against cached plans, and utilizes lightweight models to adapt these templates to task-specific plans with contexts. Evaluation across multiple real-world agent applications shows that our system can reduce costs by 50.31% and latency by 27.28% on average while maintaining performance, offering a more efficient solution for serving LLM-based agents that complements existing LLM serving infrastructures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06201v2">K2-V2: A 360-Open, Reasoning-Enhanced LLM</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      We introduce K2-V2, a 360-open LLM built from scratch as a superior base for reasoning adaptation, in addition to functions such as conversation and knowledge retrieval from general LLMs. It stands as the strongest fully open model, rivals open-weight leaders in its size class, outperforms Qwen2.5-72B and approaches the performance of Qwen3-235B. We actively infuse domain knowledge, reasoning, long-context, and tool use throughout the training process. This explicitly prepares the model for complex reasoning tasks. We demonstrate this potential using simple supervised fine-tuning, establishing a strong baseline that indicates significant headroom for advanced alignment. By releasing the full training history and data composition, we maximize the effectiveness of continuous training, a key open source production scenario. We release the model weights and signature LLM360 artifacts, such as complete training data, to empower the community with a capable, reasoning-centric foundation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18987v1">LLMs versus the Halting Problem: Revisiting Program Termination Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Determining whether a program terminates is a central problem in computer science. Turing's foundational result established the Halting Problem as undecidable, showing that no algorithm can universally determine termination for all programs and inputs. Consequently, automatic verification tools approximate termination, sometimes failing to prove or disprove; these tools rely on problem-specific architectures and abstractions, and are usually tied to particular programming languages. Recent success and progress in large language models (LLMs) raises the following question: can LLMs reliably predict program termination? In this work, we evaluate LLMs on a diverse set of C programs from the Termination category of the International Competition on Software Verification (SV-Comp) 2025. Our results suggest that LLMs perform remarkably well at predicting program termination, where GPT-5 and Claude Sonnet-4.5 would rank just behind the top-ranked tool (using test-time-scaling), and Code World Model (CWM) would place just behind the second-ranked tool. While LLMs are effective at predicting program termination, they often fail to provide a valid witness as a proof. Moreover, LLMs performance drops as program length increases. We hope these insights motivate further research into program termination and the broader potential of LLMs for reasoning about undecidable problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18984v1">Save the Good Prefix: Precise Error Penalization via Process-Supervised RL to Enhance LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as a powerful framework for improving the reasoning capabilities of large language models (LLMs). However, most existing RL approaches rely on sparse outcome rewards, which fail to credit correct intermediate steps in partially successful solutions. Process reward models (PRMs) offer fine-grained step-level supervision, but their scores are often noisy and difficult to evaluate. As a result, recent PRM benchmarks focus on a more objective capability: detecting the first incorrect step in a reasoning path. However, this evaluation target is misaligned with how PRMs are typically used in RL, where their step-wise scores are treated as raw rewards to maximize. To bridge this gap, we propose Verifiable Prefix Policy Optimization (VPPO), which uses PRMs only to localize the first error during RL. Given an incorrect rollout, VPPO partitions the trajectory into a verified correct prefix and an erroneous suffix based on the first error, rewarding the former while applying targeted penalties only after the detected mistake. This design yields stable, interpretable learning signals and improves credit assignment. Across multiple reasoning benchmarks, VPPO consistently outperforms sparse-reward RL and prior PRM-guided baselines on both Pass@1 and Pass@K.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18949v1">Tricky$^2$: Towards a Benchmark for Evaluating Human and LLM Error Interactions</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into software development workflows, yet they often introduce subtle logic or data-misuse errors that differ from human bugs. To study how these two error types interact, we construct Tricky$^2$, a hybrid dataset that augments the existing TrickyBugs corpus of human-written defects with errors injected by both GPT-5 and OpenAI-oss-20b across C++, Python, and Java programs. Our approach uses a taxonomy-guided prompting framework to generate machine-originated bugs while preserving original human defects and program structure. The resulting corpus spans human-only, LLM-only, and human+LLM splits, enabling analysis of mixed-origin error behavior, multi-bug repair robustness, and reliability in hybrid human-machine code. This paper outlines the dataset construction pipeline and illustrates its use through small-scale baseline evaluations of classification, localization, and repair tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18904v1">SICL-AT: Another way to adapt Auditory LLM to low-resource task</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Auditory Large Language Models (LLMs) have demonstrated strong performance across a wide range of speech and audio understanding tasks. Nevertheless, they often struggle when applied to low-resource or unfamiliar tasks. In case of labeled in-domain data is scarce or mismatched to the true test distribution, direct fine-tuning can be brittle. In-Context Learning (ICL) provides a training-free, inference-time solution by adapting auditory LLMs through conditioning on a few in-domain demonstrations. In this work, we first show that \emph{Vanilla ICL}, improves zero-shot performance across diverse speech and audio tasks for selected models which suggest this ICL adaptation capability can be generalized to multimodal setting. Building on this, we propose \textbf{Speech In-Context Learning Adaptation Training (SICL-AT)}, a post-training recipe utilizes only high resource speech data intending to strengthen model's in-context learning capability. The enhancement can generalize to audio understanding/reasoning task. Experiments indicate our proposed method consistently outperforms direct fine-tuning in low-resource scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18899v1">Language Family Matters: Evaluating LLM-Based ASR Across Linguistic Boundaries</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-powered Automatic Speech Recognition (ASR) systems achieve strong performance with limited resources by linking a frozen speech encoder to a pretrained LLM via a lightweight connector. Prior work trains a separate connector per language, overlooking linguistic relatedness. We propose an efficient and novel connector-sharing strategy based on linguistic family membership, enabling one connector per family, and empirically validate its effectiveness across two multilingual LLMs and two real-world corpora spanning curated and crowd-sourced speech. Our results show that family-based connectors reduce parameter count while improving generalization across domains, offering a practical and scalable strategy for multilingual ASR deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18846v1">LLM Driven Design of Continuous Optimization Problems with Controllable High-level Properties</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 17 pages, accepted at EvoApplications 2026
    </div>
    <details class="paper-abstract">
      Benchmarking in continuous black-box optimisation is hindered by the limited structural diversity of existing test suites such as BBOB. We explore whether large language models embedded in an evolutionary loop can be used to design optimisation problems with clearly defined high-level landscape characteristics. Using the LLaMEA framework, we guide an LLM to generate problem code from natural-language descriptions of target properties, including multimodality, separability, basin-size homogeneity, search-space homogeneity and globallocal optima contrast. Inside the loop we score candidates through ELA-based property predictors. We introduce an ELA-space fitness-sharing mechanism that increases population diversity and steers the generator away from redundant landscapes. A complementary basin-of-attraction analysis, statistical testing and visual inspection, verifies that many of the generated functions indeed exhibit the intended structural traits. In addition, a t-SNE embedding shows that they expand the BBOB instance space rather than forming an unrelated cluster. The resulting library provides a broad, interpretable, and reproducible set of benchmark problems for landscape analysis and downstream tasks such as automated algorithm selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18844v1">Reducing False Positives in Static Bug Detection with LLMs: An Empirical Study in Industry</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Static analysis tools (SATs) are widely adopted in both academia and industry for improving software quality, yet their practical use is often hindered by high false positive rates, especially in large-scale enterprise systems. These false alarms demand substantial manual inspection, creating severe inefficiencies in industrial code review. While recent work has demonstrated the potential of large language models (LLMs) for false alarm reduction on open-source benchmarks, their effectiveness in real-world enterprise settings remains unclear. To bridge this gap, we conduct the first comprehensive empirical study of diverse LLM-based false alarm reduction techniques in an industrial context at Tencent, one of the largest IT companies in China. Using data from Tencent's enterprise-customized SAT on its large-scale Advertising and Marketing Services software, we construct a dataset of 433 alarms (328 false positives, 105 true positives) covering three common bug types. Through interviewing developers and analyzing the data, our results highlight the prevalence of false positives, which wastes substantial manual effort (e.g., 10-20 minutes of manual inspection per alarm). Meanwhile, our results show the huge potential of LLMs for reducing false alarms in industrial settings (e.g., hybrid techniques of LLM and static analysis eliminate 94-98% of false positives with high recall). Furthermore, LLM-based techniques are cost-effective, with per-alarm costs as low as 2.1-109.5 seconds and $0.0011-$0.12, representing orders-of-magnitude savings compared to manual review. Finally, our case analysis further identifies key limitations of LLM-based false alarm reduction in industrial settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18785v1">Design Techniques for LLM-Powered Interactive Storytelling: A Case Study of the Dramamancer System</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Extended abstract presented at the 2025 Wordplay Workshop at EMNLP
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has enabled a new paradigm for bridging authorial intent and player agency in interactive narrative. We consider this paradigm through the example of Dramamancer, a system that uses an LLM to transform author-created story schemas into player-driven playthroughs. This extended abstract outlines some design techniques and evaluation considerations associated with this system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18777v1">PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted at AAAI 2026 - Innovative Applications of AI (IAAI-26)
    </div>
    <details class="paper-abstract">
      Evaluating the quality of search, ranking and RAG systems traditionally requires a significant number of human relevance annotations. In recent times, several deployed systems have explored the usage of Large Language Models (LLMs) as automated judges for this task while their inherent biases prevent direct use for metric estimation. We present a statistical framework extending Prediction-Powered Inference (PPI) that combines minimal human annotations with LLM judgments to produce reliable estimates of metrics which require sub-instance annotations. Our method requires as few as 100 human-annotated queries and 10,000 unlabeled examples, reducing annotation requirements significantly compared to traditional approaches. We formulate our proposed framework (PRECISE) for inference of relevance uplift for an LLM-based query reformulation application, extending PPI to sub-instance annotations at the query-document level. By reformulating the metric-integration space, we reduced the computational complexity from O(2^|C|) to O(2^K), where |C| represents corpus size (in order of millions). Detailed experiments across prominent retrieval datasets demonstrate that our method reduces the variance of estimates for the business-critical Precision@K metric, while effectively correcting for LLM bias in low-resource settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18754v1">$α^3$-SecBench: A Large-Scale Evaluation Suite of Security, Resilience, and Trust for LLM-based UAV Agents over 6G Networks</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Autonomous unmanned aerial vehicle (UAV) systems are increasingly deployed in safety-critical, networked environments where they must operate reliably in the presence of malicious adversaries. While recent benchmarks have evaluated large language model (LLM)-based UAV agents in reasoning, navigation, and efficiency, systematic assessment of security, resilience, and trust under adversarial conditions remains largely unexplored, particularly in emerging 6G-enabled settings. We introduce $α^{3}$-SecBench, the first large-scale evaluation suite for assessing the security-aware autonomy of LLM-based UAV agents under realistic adversarial interference. Building on multi-turn conversational UAV missions from $α^{3}$-Bench, the framework augments benign episodes with 20,000 validated security overlay attack scenarios targeting seven autonomy layers, including sensing, perception, planning, control, communication, edge/cloud infrastructure, and LLM reasoning. $α^{3}$-SecBench evaluates agents across three orthogonal dimensions: security (attack detection and vulnerability attribution), resilience (safe degradation behavior), and trust (policy-compliant tool usage). We evaluate 23 state-of-the-art LLMs from major industrial providers and leading AI labs using thousands of adversarially augmented UAV episodes sampled from a corpus of 113,475 missions spanning 175 threat types. While many models reliably detect anomalous behavior, effective mitigation, vulnerability attribution, and trustworthy control actions remain inconsistent. Normalized overall scores range from 12.9% to 57.1%, highlighting a significant gap between anomaly detection and security-aware autonomous decision-making. We release $α^{3}$-SecBench on GitHub: https://github.com/maferrag/AlphaSecBench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18753v1">HalluGuard: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Have been accepted by ICLR'26
    </div>
    <details class="paper-abstract">
      The reliability of Large Language Models (LLMs) in high-stakes domains such as healthcare, law, and scientific discovery is often compromised by hallucinations. These failures typically stem from two sources: data-driven hallucinations and reasoning-driven hallucinations. However, existing detection methods usually address only one source and rely on task-specific heuristics, limiting their generalization to complex scenarios. To overcome these limitations, we introduce the Hallucination Risk Bound, a unified theoretical framework that formally decomposes hallucination risk into data-driven and reasoning-driven components, linked respectively to training-time mismatches and inference-time instabilities. This provides a principled foundation for analyzing how hallucinations emerge and evolve. Building on this foundation, we introduce HalluGuard, an NTK-based score that leverages the induced geometry and captured representations of the NTK to jointly identify data-driven and reasoning-driven hallucinations. We evaluate HalluGuard on 10 diverse benchmarks, 11 competitive baselines, and 9 popular LLM backbones, consistently achieving state-of-the-art performance in detecting diverse forms of LLM hallucinations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.02599v2">Next Token Knowledge Tracing: Exploiting Pretrained LLM Representations to Decode Student Behaviour</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Modelling student knowledge is a key challenge when leveraging AI in education, with major implications for personalised learning. The Knowledge Tracing (KT) task aims to predict how students will respond to educational questions in learning environments, based on their prior interactions. Existing KT models typically use response correctness along with metadata like skill tags and timestamps, often overlooking the question text, which is an important source of pedagogical insight. This omission poses a lost opportunity while limiting predictive performance. We propose Next Token Knowledge Tracing (NTKT), a novel approach that reframes KT as a next-token prediction task using pretrained Large Language Models (LLMs). NTKT represents both student histories and question content as sequences of text, allowing LLMs to learn patterns in both behaviour and language. Our series of experiments significantly improves performance over state-of-the-art neural KT models and generalises much better to cold-start questions and users. These findings highlight the importance of question content in KT and demonstrate the benefits of leveraging pretrained representations of LLMs to model student learning more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18731v1">One Adapts to Any: Meta Reward Modeling for Personalized LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Alignment of Large Language Models (LLMs) aims to align outputs with human preferences, and personalized alignment further adapts models to individual users. This relies on personalized reward models that capture user-specific preferences and automatically provide individualized feedback. However, developing these models faces two critical challenges: the scarcity of feedback from individual users and the need for efficient adaptation to unseen users. We argue that addressing these constraints requires a paradigm shift from fitting data to learn user preferences to learn the process of preference adaptation. To realize this, we propose Meta Reward Modeling (MRM), which reformulates personalized reward modeling as a meta-learning problem. Specifically, we represent each user's reward model as a weighted combination of base reward functions, and optimize the initialization of these weights using a Model-Agnostic Meta-Learning (MAML)-style framework to support fast adaptation under limited feedback. To ensure robustness, we introduce the Robust Personalization Objective (RPO), which places greater emphasis on hard-to-learn users during meta optimization. Extensive experiments on personalized preference datasets validate that MRM enhances few-shot personalization, improves user robustness, and consistently outperforms baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18706v1">Health-SCORE: Towards Scalable Rubrics for Improving Health-LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Rubrics are essential for evaluating open-ended LLM responses, especially in safety-critical domains such as healthcare. However, creating high-quality and domain-specific rubrics typically requires significant human expertise time and development cost, making rubric-based evaluation and training difficult to scale. In this work, we introduce Health-SCORE, a generalizable and scalable rubric-based training and evaluation framework that substantially reduces rubric development costs without sacrificing performance. We show that Health-SCORE provides two practical benefits beyond standalone evaluation: it can be used as a structured reward signal to guide reinforcement learning with safety-aware supervision, and it can be incorporated directly into prompts to improve response quality through in-context learning. Across open-ended healthcare tasks, Health-SCORE achieves evaluation quality comparable to human-created rubrics while significantly lowering development effort, making rubric-based evaluation and training more scalable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17853v3">CiteGuard: Faithful Citation Attribution for LLMs via Retrieval-Augmented Validation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as promising assistants for scientific writing. However, there have been concerns regarding the quality and reliability of the generated text, one of which is the citation accuracy and faithfulness. While most recent work relies on methods such as LLM-as-a-Judge, the reliability of LLM-as-a-Judge alone is also in doubt. In this work, we reframe citation evaluation as a problem of citation attribution alignment, which assesses whether LLM-generated citations match those a human author would include for the same text. We propose CiteGuard, a retrieval-aware agent framework designed to provide more faithful grounding for citation validation. CiteGuard improves the prior baseline by 17%, and achieves up to 68.1% accuracy on the CiteME benchmark, approaching human-level performance (69.7%). It also enables the identification of alternative but valid citations and demonstrates generalization ability for cross-domain citation attribution.Our code is available at https://github.com/KathCYM/CiteGuard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.12945v3">LLMPopcorn: Exploring LLMs as Assistants for Popular Micro-video Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 Accepted by ICASSP2026
    </div>
    <details class="paper-abstract">
      In an era where micro-videos dominate platforms like TikTok and YouTube, AI-generated content is nearing cinematic quality. The next frontier is using large language models (LLMs) to autonomously create viral micro-videos, a largely untapped potential that could shape the future of AI-driven content creation. To address this gap, this paper presents the first exploration of LLM-assisted popular micro-video generation (LLMPopcorn). We selected popcorn as the icon for this paper because it symbolizes leisure and entertainment, aligning with this study on leveraging LLMs as assistants for generating popular micro-videos that are often consumed during leisure time. Specifically, we empirically study the following research questions: (i) How can LLMs be effectively utilized to assist popular micro-video generation? (ii) To what extent can prompt-based enhancements optimize the LLM-generated content for higher popularity? (iii) How well do various LLMs and video generators perform in the popular micro-video generation task? Exploring these questions, we show that advanced LLMs like DeepSeek-V3 can generate micro-videos with popularity rivaling human content. Prompt enhancement further boosts results, while benchmarking highlights DeepSeek-V3 and R1 for LLMs, and LTX-Video and HunyuanVideo for video generation. This work advances AI-assisted micro-video creation and opens new research directions. The code is publicly available at https://github.com/GAIR-Lab/LLMPopcorn.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.06822v2">RAFFLES: Reasoning-based Attribution of Faults for LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      The advent of complex, interconnected long-horizon LLM systems has made it incredibly tricky to identify where and when these systems break down. Evaluation capabilities that currently exist today are limited in that they often focus on simple metrics, end-to-end outcomes, and are dependent on the perspectives of humans. In order to match the increasing complexity of these many component systems, evaluation frameworks must also be able to reason, probe, iterate, and understand the nuanced logic passing through these systems. In this paper, we present RAFFLES, an offline evaluation architecture that incorporates iterative reasoning. Specifically, RAFFLES operates as an iterative, multi-component pipeline, using a central Judge to systematically identify faults and a set of specialized Evaluators to assess the quality of the candidate faults as well as rationales of the Judge. We evaluated RAFFLES with several benchmarks - the Who&When dataset to identify step-level faults in multi-agent systems and the ReasonEval datasets to diagnose step-level mathematical reasoning errors. RAFFLES outperforms strong baselines, achieving an accuracy of over 20% and 50% on the Who&When Hand-Crafted and Algorithmically-Generated datasets, and over 80% on the ReasonEval datasets. These results demonstrate a key step towards introducing automated fault detection for autonomous systems over labor-intensive manual review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15765v2">Data Valuation for LLM Fine-Tuning: Efficient Shapley Value Approximation via Language Model Arithmetic</a></div>
    <div class="paper-meta">
      📅 2026-01-26
      | 💬 11 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Data is a critical asset for training large language models (LLMs), alongside compute resources and skilled workers. While some training data is publicly available, substantial investment is required to generate proprietary datasets, such as human preference annotations or to curate new ones from existing sources. As larger datasets generally yield better model performance, two natural questions arise. First, how can data owners make informed decisions about curation strategies and data sources investment? Second, how can multiple data owners collaboratively pool their resources to train superior models while fairly distributing the benefits? This problem, data valuation, which is not specific to large language models, has been addressed by the machine learning community through the lens of cooperative game theory, with the Shapley value being the prevalent solution concept. However, computing Shapley values is notoriously expensive for data valuation, typically requiring numerous model retrainings, which can become prohibitive for large machine learning models. In this work, we demonstrate that this computational challenge is dramatically simplified for LLMs trained with Direct Preference Optimization (DPO). We show how the specific mathematical structure of DPO enables scalable Shapley value computation. We believe this observation unlocks many applications at the intersection of data valuation and large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18630v1">Assessing the Quality of Mental Health Support in LLM Responses through Multi-Attribute Human Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-26
    </div>
    <details class="paper-abstract">
      The escalating global mental health crisis, marked by persistent treatment gaps, availability, and a shortage of qualified therapists, positions Large Language Models (LLMs) as a promising avenue for scalable support. While LLMs offer potential for accessible emotional assistance, their reliability, therapeutic relevance, and alignment with human standards remain challenging to address. This paper introduces a human-grounded evaluation methodology designed to assess LLM generated responses in therapeutic dialogue. Our approach involved curating a dataset of 500 mental health conversations from datasets with real-world scenario questions and evaluating the responses generated by nine diverse LLMs, including closed source and open source models. More specifically, these responses were evaluated by two psychiatric trained experts, who independently rated each on a 5 point Likert scale across a comprehensive 6 attribute rubric. This rubric captures Cognitive Support and Affective Resonance, providing a multidimensional perspective on therapeutic quality. Our analysis reveals that LLMs provide strong cognitive reliability by producing safe, coherent, and clinically appropriate information, but they demonstrate unstable affective alignment. Although closed source models (e.g., GPT-4o) offer balanced therapeutic responses, open source models show greater variability and emotional flatness. We reveal a persistent cognitive-affective gap and highlight the need for failure aware, clinically grounded evaluation frameworks that prioritize relational sensitivity alongside informational accuracy in mental health oriented LLMs. We advocate for balanced evaluation protocols with human in the loop that center on therapeutic sensitivity and provide a framework to guide the responsible design and clinical oversight of mental health oriented conversational AI.
    </details>
</div>
