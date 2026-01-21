# llm - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- Part 8
- [Part 9](papers_9.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14803v3">OnlineMate: An LLM-Based Multi-Agent Companion System for Cognitive Support in Online Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      In online learning environments, students often lack personalized peer interactions, which are crucial for cognitive development and learning engagement. Although previous studies have employed large language models (LLMs) to simulate interactive learning environments, these interactions are limited to conversational exchanges, failing to adapt to learners' individualized cognitive and psychological states. As a result, students' engagement is low and they struggle to gain inspiration. To address this challenge, we propose OnlineMate, a multi-agent learning companion system driven by LLMs integrated with Theory of Mind (ToM). OnlineMate simulates peer-like roles, infers learners' psychological states such as misunderstandings and confusion during collaborative discussions, and dynamically adjusts interaction strategies to support higher-order thinking. Comprehensive evaluations, including simulation-based experiments, human assessments, and real classroom trials, demonstrate that OnlineMate significantly promotes deep learning and cognitive engagement by elevating students' average cognitive level while substantially improving emotional engagement scores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10449v3">When Reject Turns into Accept: Quantifying the Vulnerability of LLM-Based Scientific Reviewers to Indirect Prompt Injection</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Driven by surging submission volumes, scientific peer review has catalyzed two parallel trends: individual over-reliance on LLMs and institutional AI-powered assessment systems. This study investigates the robustness of "LLM-as-a-Judge" systems to adversarial PDF manipulation via invisible text injections and layout aware encoding attacks. We specifically target the distinct incentive of flipping "Reject" decisions to "Accept," a vulnerability that fundamentally compromises scientific integrity. To measure this, we introduce the Weighted Adversarial Vulnerability Score (WAVS), a novel metric that quantifies susceptibility by weighting score inflation against the severity of decision shifts relative to ground truth. We adapt 15 domain-specific attack strategies, ranging from semantic persuasion to cognitive obfuscation, and evaluate them across 13 diverse language models (including GPT-5 and DeepSeek) using a curated dataset of 200 official and real-world accepted and rejected submissions (e.g., ICLR OpenReview). Our results demonstrate that obfuscation techniques like "Maximum Mark Magyk" and "Symbolic Masking & Context Redirection" successfully manipulate scores, achieving decision flip rates of up to 86.26% in open-source models, while exposing distinct "reasoning traps" in proprietary systems. We release our complete dataset and injection framework to facilitate further research on the topic (https://anonymous.4open.sciencer/llm-jailbreak-FC9E/).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02957v1">LLM-Augmented Changepoint Detection: A Framework for Ensemble Detection and Automated Explanation</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      This paper introduces a novel changepoint detection framework that combines ensemble statistical methods with Large Language Models (LLMs) to enhance both detection accuracy and the interpretability of regime changes in time series data. Two critical limitations in the field are addressed. First, individual detection methods exhibit complementary strengths and weaknesses depending on data characteristics, making method selection non-trivial and prone to suboptimal results. Second, automated, contextual explanations for detected changes are largely absent. The proposed ensemble method aggregates results from ten distinct changepoint detection algorithms, achieving superior performance and robustness compared to individual methods. Additionally, an LLM-powered explanation pipeline automatically generates contextual narratives, linking detected changepoints to potential real-world historical events. For private or domain-specific data, a Retrieval-Augmented Generation (RAG) solution enables explanations grounded in user-provided documents. The open source Python framework demonstrates practical utility in diverse domains, including finance, political science, and environmental science, transforming raw statistical output into actionable insights for analysts and decision-makers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01887v2">Safety at One Shot: Patching Fine-Tuned LLMs with A Single Instance</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Fine-tuning safety-aligned large language models (LLMs) can substantially compromise their safety. Previous approaches require many safety samples or calibration sets, which not only incur significant computational overhead during realignment but also lead to noticeable degradation in model utility. Contrary to this belief, we show that safety alignment can be fully recovered with only a single safety example, without sacrificing utility and at minimal cost. Remarkably, this recovery is effective regardless of the number of harmful examples used in fine-tuning or the size of the underlying model, and convergence is achieved within just a few epochs. Furthermore, we uncover the low-rank structure of the safety gradient, which explains why such efficient correction is possible. We validate our findings across five safety-aligned LLMs and multiple datasets, demonstrating the generality of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02950v1">Batch-of-Thought: Cross-Instance Learning for Enhanced LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Current Large Language Model reasoning systems process queries independently, discarding valuable cross-instance signals such as shared reasoning patterns and consistency constraints. We introduce Batch-of-Thought (BoT), a training-free method that processes related queries jointly to enable cross-instance learning. By performing comparative analysis across batches, BoT identifies high-quality reasoning templates, detects errors through consistency checks, and amortizes computational costs. We instantiate BoT within a multi-agent reflection architecture (BoT-R), where a Reflector performs joint evaluation to unlock mutual information gain unavailable in isolated processing. Experiments across three model families and six benchmarks demonstrate that BoT-R consistently improves accuracy and confidence calibration while reducing inference costs by up to 61%. Our theoretical and experimental analysis reveals when and why batch-aware reasoning benefits LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20002v2">LoFT-LLM: Low-Frequency Time-Series Forecasting with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 This submission is withdrawn due to internal review and compliance considerations
    </div>
    <details class="paper-abstract">
      Time-series forecasting in real-world applications such as finance and energy often faces challenges due to limited training data and complex, noisy temporal dynamics. Existing deep forecasting models typically supervise predictions using full-length temporal windows, which include substantial high-frequency noise and obscure long-term trends. Moreover, auxiliary variables containing rich domain-specific information are often underutilized, especially in few-shot settings. To address these challenges, we propose LoFT-LLM, a frequency-aware forecasting pipeline that integrates low-frequency learning with semantic calibration via a large language model (LLM). Firstly, a Patch Low-Frequency forecasting Module (PLFM) extracts stable low-frequency trends from localized spectral patches. Secondly, a residual learner then models high-frequency variations. Finally, a fine-tuned LLM refines the predictions by incorporating auxiliary context and domain knowledge through structured natural language prompts. Extensive experiments on financial and energy datasets demonstrate that LoFT-LLM significantly outperforms strong baselines under both full-data and few-shot regimes, delivering superior accuracy, robustness, and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02931v1">Memorization, Emergence, and Explaining Reversal Failures: A Controlled Study of Relational Semantics in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Autoregressive LLMs perform well on relational tasks that require linking entities via relational words (e.g., father/son, friend), but it is unclear whether they learn the logical semantics of such relations (e.g., symmetry and inversion logic) and, if so, whether reversal-type failures arise from missing relational semantics or left-to-right order bias. We propose a controlled Knowledge Graph-based synthetic framework that generates text from symmetric/inverse triples, train GPT-style autoregressive models from scratch, and evaluate memorization, logical inference, and in-context generalization to unseen entities to address these questions. We find a sharp phase transition in which relational semantics emerge with sufficient logic-bearing supervision, even in shallow (2-3 layer) models, and that successful generalization aligns with stable intermediate-layer signals. Finally, order-matched forward/reverse tests and a diffusion baseline indicate that reversal failures are primarily driven by autoregressive order bias rather than deficient inversion semantics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04252v1">Sphinx: Benchmarking and Modeling for LLM-Driven Pull Request Review</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Pull request (PR) review is essential for ensuring software quality, yet automating this task remains challenging due to noisy supervision, limited contextual understanding, and inadequate evaluation metrics. We present Sphinx, a unified framework for LLM-based PR review that addresses these limitations through three key components: (1) a structured data generation pipeline that produces context-rich, semantically grounded review comments by comparing pseudo-modified and merged code; (2) a checklist-based evaluation benchmark that assesses review quality based on structured coverage of actionable verification points, moving beyond surface-level metrics like BLEU; and (3) Checklist Reward Policy Optimization (CRPO), a novel training paradigm that uses rule-based, interpretable rewards to align model behavior with real-world review practices. Extensive experiments show that models trained with Sphinx achieve state-of-the-art performance on review completeness and precision, outperforming both proprietary and open-source baselines by up to 40\% in checklist coverage. Together, Sphinx enables the development of PR review models that are not only fluent but also context-aware, technically precise, and practically deployable in real-world development workflows. The data will be released after review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.20910v2">Emergence and Localisation of Semantic Role Circuits in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Despite displaying semantic competence, large language models' internal mechanisms that ground abstract semantic structure remain insufficiently characterised. We propose a method integrating role-cross minimal pairs, temporal emergence analysis, and cross-model comparison to study how LLMs implement semantic roles. Our analysis uncovers: (i) highly concentrated circuits (89-94% attribution within 28 nodes); (ii) gradual structural refinement rather than phase transitions, with larger models sometimes bypassing localised circuits; and (iii) moderate cross-scale conservation (24-59% component overlap) alongside high spectral similarity. These findings suggest that LLMs form compact, causally isolated mechanisms for abstract semantic structure, and these mechanisms exhibit partial transfer across scales and architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02902v1">Logical Phase Transitions: Understanding Collapse in LLM Logical Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Symbolic logical reasoning is a critical yet underexplored capability of large language models (LLMs), providing reliable and verifiable decision-making in high-stakes domains such as mathematical reasoning and legal judgment. In this study, we present a systematic analysis of logical reasoning under controlled increases in logical complexity, and reveal a previously unrecognized phenomenon, which we term Logical Phase Transitions: rather than degrading smoothly, logical reasoning performance remains stable within a regime but collapses abruptly beyond a critical logical depth, mirroring physical phase transitions such as water freezing beyond a critical temperature threshold. Building on this insight, we propose Neuro-Symbolic Curriculum Tuning, a principled framework that adaptively aligns natural language with logical symbols to establish a shared representation, and reshapes training dynamics around phase-transition boundaries to progressively strengthen reasoning at increasing logical depths. Experiments on five benchmarks show that our approach effectively mitigates logical reasoning collapse at high complexity, yielding average accuracy gains of +1.26 in naive prompting and +3.95 in CoT, while improving generalization to unseen logical compositions. Code and data are available at https://github.com/AI4SS/Logical-Phase-Transitions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02880v1">ReTreVal: Reasoning Tree with Validation -- A Hybrid Framework for Enhanced LLM Multi-Step Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 14 pages, 1 figure, 5 tables
    </div>
    <details class="paper-abstract">
      Multi-step reasoning remains a key challenge for Large Language Models (LLMs), particularly in complex domains such as mathematics and creative writing. While recent approaches including ReAct, Reflexion, and Self-Refine improve reasoning through iterative refinement and reflection, they often lack structured exploration of alternative solution paths and persistent learning across problems. We propose ReTreVal (Reasoning Tree with Validation), a hybrid framework that integrates Tree-of-Thoughts exploration, self-refinement, LLM-based critique scoring, and reflexion memory to enable bounded and validated multi-step reasoning. ReTreVal constructs a structured reasoning tree with adaptive depth based on problem complexity, where each node undergoes iterative self-critique and refinement guided by explicit LLM-generated feedback. A dual validation mechanism evaluates reasoning quality, coherence, and correctness at each node while persistently storing insights from successful reasoning paths and failure patterns in a reflexion memory buffer, enabling cross-problem learning. Critique-based pruning retains only the top-k highest-scoring nodes at each level, controlling computational cost while preserving high-quality solution paths. We evaluate ReTreVal against ReAct, Reflexion, and Self-Refine across 500 mathematical problems and creative writing tasks using Qwen 2.5 7B as the underlying LLM, and demonstrate that ReTreVal consistently outperforms existing methods through its combination of structured exploration, critique-driven refinement, and cross-problem memory, making it particularly effective for tasks requiring exploratory reasoning, rigorous verification, and knowledge transfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.18784v4">Successor-Generator Planning with LLM-generated Heuristics</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Heuristics are a central component of deterministic planning, particularly in domain-independent settings where general applicability is prioritized over task-specific tuning. This work revisits that paradigm in light of recent advances in large language models (LLMs), which enable the automatic synthesis of heuristics directly from problem definitions -- bypassing the need for handcrafted domain knowledge. We present a method that employs LLMs to generate problem-specific heuristic functions from planning tasks specified through successor generators, goal tests, and initial states written in a general-purpose programming language. These heuristics are compiled and integrated into standard heuristic search algorithms, such as greedy best-first search. Our approach achieves competitive, and in many cases state-of-the-art, performance across a broad range of established planning benchmarks. Moreover, it enables the solution of problems that are difficult to express in traditional formalisms, including those with complex numeric constraints or custom transition dynamics. We provide an extensive empirical evaluation that characterizes the strengths and limitations of the approach across diverse planning settings, demonstrating its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02858v1">To Generate or Discriminate? Methodological Considerations for Measuring Cultural Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 IJCNLP-AACL 2025
    </div>
    <details class="paper-abstract">
      Socio-demographic prompting (SDP) - prompting Large Language Models (LLMs) using demographic proxies to generate culturally aligned outputs - often shows LLM responses as stereotypical and biased. While effective in assessing LLMs' cultural competency, SDP is prone to confounding factors such as prompt sensitivity, decoding parameters, and the inherent difficulty of generation over discrimination tasks due to larger output spaces. These factors complicate interpretation, making it difficult to determine if the poor performance is due to bias or the task design. To address this, we use inverse socio-demographic prompting (ISDP), where we prompt LLMs to discriminate and predict the demographic proxy from actual and simulated user behavior from different users. We use the Goodreads-CSI dataset (Saha et al., 2025), which captures difficulty in understanding English book reviews for users from India, Mexico, and the USA, and test four LLMs: Aya-23, Gemma-2, GPT-4o, and LLaMA-3.1 with ISDP. Results show that models perform better with actual behaviors than simulated ones, contrary to what SDP suggests. However, performance with both behavior types diminishes and becomes nearly equal at the individual level, indicating limits to personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24867v2">Encyclo-K: Evaluating LLMs with Dynamically Composed Knowledge Statements</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Benchmarks play a crucial role in tracking the rapid advancement of large language models (LLMs) and identifying their capability boundaries. However, existing benchmarks predominantly curate questions at the question level, suffering from three fundamental limitations: vulnerability to data contamination, restriction to single-knowledge-point assessment, and reliance on costly domain expert annotation. We propose Encyclo-K, a statement-based benchmark that rethinks benchmark construction from the ground up. Our key insight is that knowledge statements, not questions, can serve as the unit of curation, and questions can then be constructed from them. We extract standalone knowledge statements from authoritative textbooks and dynamically compose them into evaluation questions through random sampling at test time. This design directly addresses all three limitations: the combinatorial space is too vast to memorize, and model rankings remain stable across dynamically generated question sets, enabling reliable periodic dataset refresh; each question aggregates 8-10 statements for comprehensive multi-knowledge assessment; annotators only verify formatting compliance without requiring domain expertise, substantially reducing annotation costs. Experiments on over 50 LLMs demonstrate that Encyclo-K poses substantial challenges with strong discriminative power. Even the top-performing OpenAI-GPT-5.1 achieves only 62.07% accuracy, and model performance displays a clear gradient distribution--reasoning models span from 16.04% to 62.07%, while chat models range from 9.71% to 50.40%. These results validate the challenges introduced by dynamic evaluation and multi-statement comprehensive understanding. These findings establish Encyclo-K as a scalable framework for dynamic evaluation of LLMs' comprehensive understanding over multiple fine-grained disciplinary knowledge statements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10411v3">SWAA: Sliding Window Attention Adaptation for Efficient Long-Context LLMs Without Pretraining</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      The quadratic complexity of self-attention in Transformer-based Large Language Models (LLMs) renders long-context inference prohibitively expensive. While Sliding Window Attention (SWA), the simplest sparse attention pattern, offers a linear-complexity alternative, naively applying it to models pretrained with Full Attention (FA) causes catastrophic long-context performance collapse due to the training-inference mismatch. To address this, we propose Sliding Window Attention Adaptation (SWAA), a plug-and-play toolkit of recipes that adapt FA models to SWA without costly pretraining. SWAA systematically combines five strategies: (1) applying SWA only during prefilling; (2) preserving "sink" tokens; (3) interleaving FA/SWA layers; (4) chain-of-thought (CoT); and (5) fine-tuning. Our experiments demonstrate that while individual methods are insufficient, specific synergistic combinations can effectively recover original long-context capabilities. After further analyzing performance-efficiency trade-offs, we identify recommended SWAA configurations for diverse scenarios, which achieve 30% to 100% speedups for long-context LLM inference with acceptable quality loss. Our code is available at https://github.com/yuyijiong/sliding-window-attention-adaptation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.09510v3">Communication Compression for Tensor Parallel LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have pushed the frontier of artificial intelligence but are comprised of hundreds of billions of parameters and operations. For faster inference latency, LLMs are deployed on multiple hardware accelerators through various Model Parallelism strategies. Our paper looks into the details on one such strategy - Tensor Parallel - and proposes to reduce latency by compressing inter-accelerator communication. We leverage fine grained quantization techniques to compress selected activations by 3.5 - 4.5x. Our proposed method leads up to 2x reduction of time-to-first-token (TTFT) with negligible model performance degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02025v3">Style over Story: Measuring LLM Narrative Preferences via Structured Selection</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      We introduce a constraint-selection-based experiment design for measuring narrative preferences of Large Language Models (LLMs). This design offers an interpretable lens on LLMs' narrative behavior. We developed a library of 200 narratology-grounded constraints and prompted selections from six LLMs under three different instruction types: basic, quality-focused, and creativity-focused. Findings demonstrate that models consistently prioritize Style over narrative content elements like Event, Character, and Setting. Style preferences remain stable across models and instruction types, whereas content elements show cross-model divergence and instructional sensitivity. These results suggest that LLMs have latent narrative preferences, which should inform how the NLP community evaluates and deploys models in creative domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02813v1">HAL: Inducing Human-likeness in LLMs with Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Conversational human-likeness plays a central role in human-AI interaction, yet it has remained difficult to define, measure, and optimize. As a result, improvements in human-like behavior are largely driven by scale or broad supervised training, rather than targeted alignment. We introduce Human Aligning LLMs (HAL), a framework for aligning language models to conversational human-likeness using an interpretable, data-driven reward. HAL derives explicit conversational traits from contrastive dialogue data, combines them into a compact scalar score, and uses this score as a transparent reward signal for alignment with standard preference optimization methods. Using this approach, we align models of varying sizes without affecting their overall performance. In large-scale human evaluations, models aligned with HAL are more frequently perceived as human-like in conversation. Because HAL operates over explicit, interpretable traits, it enables inspection of alignment behavior and diagnosis of unintended effects. More broadly, HAL demonstrates how soft, qualitative properties of language--previously outside the scope for alignment--can be made measurable and aligned in an interpretable and explainable way.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13271v2">Do You Get the Hint? Benchmarking LLMs on the Board Game Concept</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved striking successes on many benchmarks, yet recent studies continue to expose fundamental weaknesses. In this paper, we introduce Concept, a simple word-guessing board game, as a benchmark for probing abductive reasoning. Our results show that this game, easily solved by humans (with a success rate of over 90\%), is still very challenging for state-of-the-art LLMs (no model exceeds 40\% success rate). Specifically, we observe that LLMs struggle with interpreting other players' strategic intents, and with correcting initial hypotheses given sequential information updates. In addition, we extend the evaluation across multiple languages, and find that the LLM performance drops further in lower-resource languages (Dutch, French, and Spanish) compared to English.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07842v3">Alignment-Aware Quantization for LLM Safety</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 9 pages, 4 figures. Includes 8 pages of supplementary material
    </div>
    <details class="paper-abstract">
      Safety and efficiency are paramount yet often conflicting requirements for deploying Large Language Models (LLMs). While LLMs are trained to follow human alignment for safety, Post-Training Quantization (PTQ) is applied afterward to ensure efficiency. Here we identify a fundamental flaw in the conventional PTQ paradigm: quantization can turn into a safety vulnerability if it only aims to achieve low perplexity. To address this, we propose Alignment-Aware Quantization (AAQ), a novel approach that integrates an Alignment-Preserving Contrastive (APC) loss into the PTQ pipeline. Our method explicitly preserves alignment by encouraging the quantized model to mimic its safe, instruction-tuned model while diverging from the unaligned, pre-trained counterpart. AAQ achieves robust safety alignment without specialized safety-focused datasets, using only standard calibration data. We show that AAQ is compatible with standard PTQ techniques and enables robust 4-bit (W4A4) quantization across diverse model families. Our work resolves the critical trade-off between efficiency and safety, paving the way toward LLMs that are both efficient and trustworthy. Anonymized code is available in the supplementary material.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.07261v3">MemHunter: Automated and Verifiable Memorization Detection at Dataset-scale in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Withdrawn by the authors due to an inconsistency in the reported base model: Section 4 (Experiments) states "Llama-2-7B" while Fig. 3 labels "Llama-2-7B-Chat". Because this affects the experimental configuration, parts of the results must be re-verified by rerunning experiments; we withdraw to avoid misleading readers
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to memorize and reproduce content from their training data, raising significant privacy concerns, especially with web-scale datasets. Existing methods for detecting memorization are primarily sample-specific, relying on manually crafted or discretely optimized memory-inducing prompts generated on a per-sample basis, which become impractical for dataset-level detection due to the prohibitive computational cost of iterating through all samples. In real-world scenarios, data owners may need to verify whether a susceptible LLM has memorized their dataset, particularly if the LLM may have collected the data from the web without authorization. To address this, we introduce MemHunter, which trains a memory-inducing LLM and employs hypothesis testing to efficiently detect memorization at the dataset level, without requiring sample-specific memory inducing. Experiments on models like Pythia and Llama demonstrate that MemHunter can extract up to 40% more training data than existing methods under constrained time resources and reduce search time by up to 80% when integrated as a plug-in. Crucially, MemHunter is the first method capable of dataset-level memorization detection, providing a critical tool for assessing privacy risks in LLMs powered by large-scale datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02764v1">Netflix Artwork Personalization via LLM Post-training</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated success in various applications of user recommendation and personalization across e-commerce and entertainment. On many entertainment platforms such as Netflix, users typically interact with a wide range of titles, each represented by an artwork. Since users have diverse preferences, an artwork that appeals to one type of user may not resonate with another with different preferences. Given this user heterogeneity, our work explores the novel problem of personalized artwork recommendations according to diverse user preferences. Similar to the multi-dimensional nature of users' tastes, titles contain different themes and tones that may appeal to different viewers. For example, the same title might feature both heartfelt family drama and intense action scenes. Users who prefer romantic content may like the artwork emphasizing emotional warmth between the characters, while those who prefer action thrillers may find high-intensity action scenes more intriguing. Rather than a one-size-fits-all approach, we conduct post-training of pre-trained LLMs to make personalized artwork recommendations, selecting the most preferred visual representation of a title for each user and thereby improving user satisfaction and engagement. Our experimental results with Llama 3.1 8B models (trained on a dataset of 110K data points and evaluated on 5K held-out user-title pairs) show that the post-trained LLMs achieve 3-5\% improvements over the Netflix production model, suggesting a promising direction for granular personalized recommendations using LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02757v1">LLM Agent Framework for Intelligent Change Analysis in Urban Environment using Remote Sensing Imagery</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Existing change detection methods often lack the versatility to handle diverse real-world queries and the intelligence for comprehensive analysis. This paper presents a general agent framework, integrating Large Language Models (LLM) with vision foundation models to form ChangeGPT. A hierarchical structure is employed to mitigate hallucination. The agent was evaluated on a curated dataset of 140 questions categorized by real-world scenarios, encompassing various question types (e.g., Size, Class, Number) and complexities. The evaluation assessed the agent's tool selection ability (Precision/Recall) and overall query accuracy (Match). ChangeGPT, especially with a GPT-4-turbo backend, demonstrated superior performance, achieving a 90.71 % Match rate. Its strength lies particularly in handling change-related queries requiring multi-step reasoning and robust tool selection. Practical effectiveness was further validated through a real-world urban change monitoring case study in Qianhai Bay, Shenzhen. By providing intelligence, adaptability, and multi-type change analysis, ChangeGPT offers a powerful solution for decision-making in remote sensing applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02744v1">SYNAPSE: Empowering LLM Agents with Episodic-Semantic Memory via Spreading Activation</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) excel at generalized reasoning, standard retrieval-augmented approaches fail to address the disconnected nature of long-term agentic memory. To bridge this gap, we introduce Synapse (Synergistic Associative Processing Semantic Encoding), a unified memory architecture that transcends static vector similarity. Drawing from cognitive science, Synapse models memory as a dynamic graph where relevance emerges from spreading activation rather than pre-computed links. By integrating lateral inhibition and temporal decay, the system dynamically highlights relevant sub-graphs while filtering interference. We implement a Triple Hybrid Retrieval strategy that fuses geometric embeddings with activation-based graph traversal. Comprehensive evaluations on the LoCoMo benchmark show that Synapse significantly outperforms state-of-the-art methods in complex temporal and multi-hop reasoning tasks, offering a robust solution to the "Contextual Tunneling" problem. Our code and data will be made publicly available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00454v4">Learning an Efficient Multi-Turn Dialogue Evaluator from Multiple LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 20 pages, 4 pages, under review
    </div>
    <details class="paper-abstract">
      Evaluating the conversational abilities of large language models (LLMs) remains a challenging task. Current mainstream approaches primarily rely on the "LLM-as-a-judge" paradigm, where an LLM is prompted to serve as an evaluator to assess dialogue quality. However, such methods often suffer from various biases, which undermine the reliability and consistency of the evaluation results. To mitigate these biases, recent methods employ multiple LLMs as judges and aggregate their judgments to select the optimal assessment. Although effective, this multi-judge approach incurs significant computational overhead during inference. In this paper, we propose an efficient dialogue evaluator that captures the collective wisdom of multiple LLM judges by aggregating their preference knowledge into a single model. Our approach preserves the advantages of diverse multi-judge feedback while drastically reducing the evaluation cost, enabling fast, flexible, and fine-grained dialogue quality assessment. Extensive experiments on seven single rating and pairwise comparison dialogue evaluation benchmarks demonstrate that our method outperforms existing baselines across diverse scenarios, showcasing its efficiency and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.12645v4">Diagnostic-Guided Dynamic Profile Optimization for LLM-based User Simulators in Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled realistic user simulators for developing and evaluating recommender systems (RSs). However, existing LLM-based simulators for RSs face two major limitations: (1) static and single-step prompt-based inference that leads to inaccurate and incomplete user profile construction; (2) unrealistic and single-round recommendation-feedback interaction pattern that fails to capture real-world scenarios. To address these limitations, we propose DGDPO (Diagnostic-Guided Dynamic Profile Optimization), a novel framework that constructs user profile through a dynamic and iterative optimization process to enhance the simulation fidelity. Specifically, DGDPO incorporates two core modules within each optimization loop: firstly, a specialized LLM-based diagnostic module, calibrated through our novel training strategy, accurately identifies specific defects in the user profile. Subsequently, a generalized LLM-based treatment module analyzes the diagnosed defect and generates targeted suggestions to refine the profile. Furthermore, unlike existing LLM-based user simulators that are limited to single-round interactions, we are the first to integrate DGDPO with sequential recommenders, enabling a bidirectional evolution where user profiles and recommendation strategies adapt to each other over multi-round interactions. Extensive experiments conducted on three real-world datasets demonstrate the effectiveness of our proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02695v1">EvoRoute: Experience-Driven Self-Routing LLM Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Complex agentic AI systems, powered by a coordinated ensemble of Large Language Models (LLMs), tool and memory modules, have demonstrated remarkable capabilities on intricate, multi-turn tasks. However, this success is shadowed by prohibitive economic costs and severe latency, exposing a critical, yet underexplored, trade-off. We formalize this challenge as the \textbf{Agent System Trilemma}: the inherent tension among achieving state-of-the-art performance, minimizing monetary cost, and ensuring rapid task completion. To dismantle this trilemma, we introduce EvoRoute, a self-evolving model routing paradigm that transcends static, pre-defined model assignments. Leveraging an ever-expanding knowledge base of prior experience, EvoRoute dynamically selects Pareto-optimal LLM backbones at each step, balancing accuracy, efficiency, and resource use, while continually refining its own selection policy through environment feedback. Experiments on challenging agentic benchmarks such as GAIA and BrowseComp+ demonstrate that EvoRoute, when integrated into off-the-shelf agentic systems, not only sustains or enhances system performance but also reduces execution cost by up to $80\%$ and latency by over $70\%$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01310v2">Making MoE-based LLM Inference Resilient with Tarragon</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) models are increasingly used to serve LLMs at scale, but failures become common as deployment scale grows. Existing systems exhibit poor failure resilience: even a single worker failure triggers a coarse-grained, service-wide restart, discarding accumulated progress and halting the entire inference pipeline during recovery--an approach clearly ill-suited for latency-sensitive, LLM services. We present Tarragon, a resilient MoE inference framework that confines the failures impact to individual workers while allowing the rest of the pipeline to continue making forward progress. Tarragon exploits the natural separation between the attention and expert computation in MoE-based transformers, treating attention workers (AWs) and expert workers (EWs) as distinct failure domains. Tarragon introduces a reconfigurable datapath to mask failures by rerouting requests to healthy workers. On top of this datapath, Tarragon implements a self-healing mechanism that relaxes the tightly synchronized execution of existing MoE frameworks. For stateful AWs, Tarragon performs asynchronous, incremental KV cache checkpointing with per-request restoration, and for stateless EWs, it leverages residual GPU memory to deploy shadow experts. These together keep recovery cost and recomputation overhead extremely low. Our evaluation shows that, compared to state-of-the-art MegaScale-Infer, Tarragon reduces failure-induced stalls by 160-213x (from ~64 s down to 0.3-0.4 s) while preserving performance when no failures occur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00500v2">Pro2Guard: Proactive Runtime Enforcement of LLM Agent Safety via Probabilistic Model Checking</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents demonstrate strong autonomy, but their stochastic behavior introduces unpredictable safety risks. Existing rule-based enforcement systems, such as AgentSpec, are reactive, intervening only when unsafe behavior is imminent or has occurred, lacking foresight for long-horizon dependencies. To overcome these limitations, we present a proactive runtime enforcement framework for LLM agents. The framework abstracts agent behaviors into symbolic states and learns a Discrete-Time Markov Chain (DTMC) from execution traces. At runtime, it predicts the probability of leading to undesired behaviors and intervenes before violations occur when the estimated risk exceeds a user-defined threshold. Designed to provide PAC-correctness guarantee, the framework achieves statistically reliable enforcement of agent safety. We evaluate the framework across two safety-critical domains: autonomous vehicles and embodied agents. It proactively enforces safety and maintains high task performance, outperforming existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02680v1">Adversarial Contrastive Learning for LLM Quantization Attacks</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 14 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Model quantization is critical for deploying large language models (LLMs) on resource-constrained hardware, yet recent work has revealed severe security risks that benign LLMs in full precision may exhibit malicious behaviors after quantization. In this paper, we propose Adversarial Contrastive Learning (ACL), a novel gradient-based quantization attack that achieves superior attack effectiveness by explicitly maximizing the gap between benign and harmful responses probabilities. ACL formulates the attack objective as a triplet-based contrastive loss, and integrates it with a projected gradient descent two-stage distributed fine-tuning strategy to ensure stable and efficient optimization. Extensive experiments demonstrate ACL's remarkable effectiveness, achieving attack success rates of 86.00% for over-refusal, 97.69% for jailbreak, and 92.40% for advertisement injection, substantially outperforming state-of-the-art methods by up to 44.67%, 18.84%, and 50.80%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02670v1">Multi-Turn Jailbreaking of Aligned LLMs via Lexical Anchor Tree Search</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Most jailbreak methods achieve high attack success rates (ASR) but require attacker LLMs to craft adversarial queries and/or demand high query budgets. These resource limitations make jailbreaking expensive, and the queries generated by attacker LLMs often consist of non-interpretable random prefixes. This paper introduces Lexical Anchor Tree Search (), addressing these limitations through an attacker-LLM-free method that operates purely via lexical anchor injection. LATS reformulates jailbreaking as a breadth-first tree search over multi-turn dialogues, where each node incrementally injects missing content words from the attack goal into benign prompts. Evaluations on AdvBench and HarmBench demonstrate that LATS achieves 97-100% ASR on latest GPT, Claude, and Llama models with an average of only ~6.4 queries, compared to 20+ queries required by other methods. These results highlight conversational structure as a potent and under-protected attack surface, while demonstrating superior query efficiency in an era where high ASR is readily achievable. Our code will be released to support reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02663v1">When Do Tools and Planning Help LLMs Think? A Cost- and Latency-Aware Benchmark</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) increasingly rely on inference-time planning and external tools to improve reasoning. We benchmark this behavior on two real-world settings: event-centric question answering over graph-structured knowledge (Event-QA) and persuasive response generation in Reddit ChangeMyView (CMV). Using LangChain and LangGraph, we compare a one-shot baseline against a plan-execute-replan agent equipped with task-specific tools (DBpedia SPARQL/lookup/schema exploration, Wikipedia-focused retrieval, and topical web search). We evaluate on 60 examples each from Event-QA and CMV (3 splits of 20), and report both mean end-to-end latency and per-example token cost estimates. We evaluate GPT-4o and GPT-4o-mini under identical workflows and report accuracy and end-to-end latency. On Event-QA, the best tool-augmented configuration improves accuracy (e.g., 47.5\% $\rightarrow$ 67.5\% for GPT-4o) while increasing latency by orders of magnitude ($\sim$8s $\rightarrow$ $\sim$317s per example). On CMV, one-shot prompting is strongest (e.g., GPT-4o-mini achieves 75\% at $\sim$6s), and planning+search increases latency substantially without consistent gains. However, complex multi-tool orchestration exposes failure modes where the smaller model degrades. Overall, the findings highlight the need for task-specific, cost-aware choices of both model size and agent/tooling complexity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02632v1">TAAF: A Trace Abstraction and Analysis Framework Synergizing Knowledge Graphs and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Accepted to ICSE 2026. DOI 10.1145/3744916.3787832
    </div>
    <details class="paper-abstract">
      Execution traces are a critical source of information for understanding, debugging, and optimizing complex software systems. However, traces from OS kernels or large-scale applications like Chrome or MySQL are massive and difficult to analyze. Existing tools rely on predefined analyses, and custom insights often require writing domain-specific scripts, which is an error-prone and time-consuming task. This paper introduces TAAF (Trace Abstraction and Analysis Framework), a novel approach that combines time-indexing, knowledge graphs (KGs), and large language models (LLMs) to transform raw trace data into actionable insights. TAAF constructs a time-indexed KG from trace events to capture relationships among entities such as threads, CPUs, and system resources. An LLM then interprets query-specific subgraphs to answer natural-language questions, reducing the need for manual inspection and deep system expertise. To evaluate TAAF, we introduce TraceQA-100, a benchmark of 100 questions grounded in real kernel traces. Experiments across three LLMs and multiple temporal settings show that TAAF improves answer accuracy by up to 31.2%, particularly in multi-hop and causal reasoning tasks. We further analyze where graph-grounded reasoning helps and where limitations remain, offering a foundation for next-generation trace analysis tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02627v1">Improved Evidence Extraction for Document Inconsistency Detection with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming useful in many domains due to their impressive abilities that arise from large training datasets and large model sizes. However, research on LLM-based approaches to document inconsistency detection is relatively limited. There are two key aspects of document inconsistency detection: (i) classification of whether there exists any inconsistency, and (ii) providing evidence of the inconsistent sentences. We focus on the latter, and introduce new comprehensive evidence-extraction metrics and a redact-and-retry framework with constrained filtering that substantially improves LLM-based document inconsistency detection over direct prompting. We back our claims with promising experimental results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02624v1">LAsset: An LLM-assisted Security Asset Identification Framework for System-on-Chip (SoC) Verification</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      The growing complexity of modern system-on-chip (SoC) and IP designs is making security assurance difficult day by day. One of the fundamental steps in the pre-silicon security verification of a hardware design is the identification of security assets, as it substantially influences downstream security verification tasks, such as threat modeling, security property generation, and vulnerability detection. Traditionally, assets are determined manually by security experts, requiring significant time and expertise. To address this challenge, we present LAsset, a novel automated framework that leverages large language models (LLMs) to identify security assets from both hardware design specifications and register-transfer level (RTL) descriptions. The framework performs structural and semantic analysis to identify intra-module primary and secondary assets and derives inter-module relationships to systematically characterize security dependencies at the design level. Experimental results show that the proposed framework achieves high classification accuracy, reaching up to 90% recall rate in SoC design, and 93% recall rate in IP designs. This automation in asset identification significantly reduces manual overhead and supports a scalable path forward for secure hardware development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15674v2">Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 36 pages
    </div>
    <details class="paper-abstract">
      Large language model (LLM) activations are notoriously difficult to understand, with most existing techniques using complex, specialized methods for interpreting them. Recent work has proposed a simpler approach known as LatentQA: training LLMs to directly accept LLM activations as inputs and answer arbitrary questions about them in natural language. However, prior work has focused on narrow task settings for both training and evaluation. In this paper, we instead take a generalist perspective. We evaluate LatentQA-trained models, which we call Activation Oracles (AOs), in far out-of-distribution settings and examine how performance scales with training data diversity. We find that AOs can recover information fine-tuned into a model (e.g., biographical knowledge or malign propensities) that does not appear in the input text, despite never being trained with activations from a fine-tuned model. Our main evaluations are four downstream tasks where we can compare to prior white- and black-box techniques. We find that even narrowly-trained LatentQA models can generalize well, and that adding additional training datasets (such as classification tasks and a self-supervised context prediction task) yields consistent further improvements. Our best AOs match or exceed white-box baselines on all four tasks and the best overall baseline on 3 of 4. These results suggest that diversified training to answer natural-language queries imparts a general capability to verbalize information about LLM activations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03464v1">Prompting Underestimates LLM Capability for Time Series Classification</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 8 pages + Appendix and References, 9 figures
    </div>
    <details class="paper-abstract">
      Prompt-based evaluations suggest that large language models (LLMs) perform poorly on time series classification, raising doubts about whether they encode meaningful temporal structure. We show that this conclusion reflects limitations of prompt-based generation rather than the model's representational capacity by directly comparing prompt outputs with linear probes over the same internal representations. While zero-shot prompting performs near chance, linear probes improve average F1 from 0.15-0.26 to 0.61-0.67, often matching or exceeding specialized time series models. Layer-wise analyses further show that class-discriminative time series information emerges in early transformer layers and is amplified by visual and multimodal inputs. Together, these results demonstrate a systematic mismatch between what LLMs internally represent and what prompt-based evaluation reveals, leading current evaluations to underestimate their time series understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23842v4">Fair Document Valuation in LLM Summaries via Shapley Values</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in systems that retrieve and summarize content from multiple sources, such as search engines and AI assistants. While these systems enhance user experience through coherent summaries, they obscure the individual contributions of original content creators, raising concerns about credit attribution and compensation. We address the challenge of valuing individual documents used in LLM-generated summaries by proposing a Shapley value-based framework for fair document valuation. Although theoretically appealing, exact Shapley value computation is prohibitively expensive at scale. To improve efficiency, we develop Cluster Shapley, a simple approximation algorithm that leverages semantic similarity among documents to reduce computation while maintaining attribution accuracy. Using Amazon product review data, we empirically show that off-the-shelf Shapley approximations, such as Monte Carlo sampling and Kernel SHAP, perform suboptimally in LLM settings, whereas Cluster Shapley substantially improves the efficiency-accuracy frontier. Moreover, simple attribution rules (e.g., equal or relevance-based allocation), though computationally cheap, lead to highly unfair outcomes. Together, our findings highlight the potential of structure-aware Shapley approximations tailored to LLM summarization and offer guidance for platforms seeking scalable and fair content attribution mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03444v1">Grading Scale Impact on LLM-as-a-Judge: Human-LLM Alignment Is Highest on 0-5 Grading Scale</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as automated evaluators, yet prior works demonstrate that these LLM judges often lack consistency in scoring when the prompt is altered. However, the effect of the grading scale itself remains underexplored. We study the LLM-as-a-judge problem by comparing two kinds of raters: humans and LLMs. We collect ratings from both groups on three scales and across six benchmarks that include objective, open-ended subjective, and mixed tasks. Using intraclass correlation coefficients (ICC) to measure absolute agreement, we find that LLM judgments are not perfectly consistent across scales on subjective benchmarks, and that the choice of scale substantially shifts human-LLM agreement, even when within-group panel reliability is high. Aggregated over tasks, the grading scale of 0-5 yields the strongest human-LLM alignment. We further demonstrate that pooled reliability can mask benchmark heterogeneity and reveal systematic subgroup differences in alignment across gender groups, strengthening the importance of scale design and sub-level diagnostics as essential components of LLM-as-a-judge protocols.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03420v1">Jailbreaking LLMs Without Gradients or Priors: Effective and Transferable Attacks</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed in safety-critical domains, rigorously evaluating their robustness against adversarial jailbreaks is essential. However, current safety evaluations often overestimate robustness because existing automated attacks are limited by restrictive assumptions. They typically rely on handcrafted priors or require white-box access for gradient propagation. We challenge these constraints by demonstrating that token-level iterative optimization can succeed without gradients or priors. We introduce RAILS (RAndom Iterative Local Search), a framework that operates solely on model logits. RAILS matches the effectiveness of gradient-based methods through two key innovations: a novel auto-regressive loss that enforces exact prefix matching, and a history-based selection strategy that bridges the gap between the proxy optimization objective and the true attack success rate. Crucially, by eliminating gradient dependency, RAILS enables cross-tokenizer ensemble attacks. This allows for the discovery of shared adversarial patterns that generalize across disjoint vocabularies, significantly enhancing transferability to closed-source systems. Empirically, RAILS achieves near 100% success rates on multiple open-source models and high black-box attack transferability to closed-source systems like GPT and Gemini.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03401v1">Rendering Data Unlearnable by Exploiting LLM Alignment Mechanisms</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly trained on massive, heterogeneous text corpora, raising serious concerns about the unauthorised use of proprietary or personal data during model training. In this work, we address the problem of data protection against unwanted model learning in a realistic black-box setting. We propose Disclaimer Injection, a novel data-level defence that renders text unlearnable to LLMs. Rather than relying on model-side controls or explicit data removal, our approach exploits the models' own alignment mechanisms: by injecting carefully designed alignment-triggering disclaimers to prevent effective learning. Through layer-wise analysis, we find that fine-tuning on such protected data induces persistent activation of alignment-related layers, causing alignment constraints to override task learning even on common inputs. Consequently, models trained on such data exhibit substantial and systematic performance degradation compared to standard fine-tuning. Our results identify alignment behaviour as a previously unexplored lever for data protection and, to our knowledge, present the first practical method for restricting data learnability at LLM scale without requiring access to or modification of the training pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.20650v3">FinTagging: Benchmarking LLMs for Extracting and Structuring Financial Information</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Accurate interpretation of numerical data in financial reports is critical for markets and regulators. Although XBRL (eXtensible Business Reporting Language) provides a standard for tagging financial figures, mapping thousands of facts to over ten thousand US-GAAP concepts remains costly and error-prone. Existing benchmarks oversimplify this task as flat, single-step classification over small subsets of concepts, ignoring the hierarchical semantics of the taxonomy and the structured nature of financial documents. As a result, these benchmarks fail to evaluate Large Language Models (LLMs) under realistic reporting conditions. To bridge this gap, we introduce FinTagging, the first comprehensive benchmark for structure-aware and full-scope XBRL tagging. We decompose the complex tagging process into two subtasks: (1) FinNI (Financial Numeric Identification), which extracts entities and types from heterogeneous contexts such as text and tables; and (2) FinCL (Financial Concept Linking), which maps extracted entities to the full US-GAAP taxonomy. This two-stage formulation enables a fair assessment of LLM capabilities in numerical reasoning and taxonomy alignment. Evaluating diverse LLMs in zero-shot settings shows that while models generalize well in extraction, they struggle with fine-grained concept linking, revealing important limitations in domain-specific, structure-aware reasoning. Code is available on GitHub, and datasets are available on Hugging Face.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03385v1">SIGMA: Scalable Spectral Insights for LLM Collapse</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      The rapid adoption of synthetic data for training Large Language Models (LLMs) has introduced the technical challenge of "model collapse"-a degenerative process where recursive training on model-generated content leads to a contraction of distributional variance and representational quality. While the phenomenology of collapse is increasingly evident, rigorous methods to quantify and predict its onset in high-dimensional spaces remain elusive. In this paper, we introduce SIGMA (Spectral Inequalities for Gram Matrix Analysis), a unified framework that benchmarks model collapse through the spectral lens of the embedding Gram matrix. By deriving and utilizing deterministic and stochastic bounds on the matrix's spectrum, SIGMA provides a mathematically grounded metric to track the contraction of the representation space. Crucially, our stochastic formulation enables scalable estimation of these bounds, making the framework applicable to large-scale foundation models where full eigendecomposition is intractable. We demonstrate that SIGMA effectively captures the transition towards degenerate states, offering both theoretical insights into the mechanics of collapse and a practical, scalable tool for monitoring the health of recursive training pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06485v3">Task Matters: Knowledge Requirements Shape LLM Responses to Context-Memory Conflict</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Major revision
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) draw on both contextual information and parametric memory, yet these sources can conflict. Prior studies have largely examined this issue in contextual question answering, implicitly assuming that tasks should rely on the provided context, leaving unclear how LLMs behave when tasks require different types and degrees of knowledge utilization. We address this gap with a model-agnostic diagnostic framework that holds underlying knowledge constant while introducing controlled conflicts across tasks with varying knowledge demands. Experiments on representative open-source LLMs show that performance degradation under conflict is driven by both task-specific knowledge reliance and conflict plausibility; that strategies such as rationales or context reiteration increase context reliance, helping context-only tasks but harming those requiring parametric knowledge; and that these effects bias model-based evaluation, calling into question the reliability of LLMs as judges. Overall, our findings reveal that context-memory conflict is inherently task-dependent and motivate task-aware approaches to balancing context and memory in LLM deployment and evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03359v1">Enhancing LLM Instruction Following: An Evaluation-Driven Multi-Agentic Workflow for Prompt Instructions Optimization</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate substantively relevant content but fail to adhere to formal constraints, leading to outputs that are conceptually correct but procedurally flawed. Traditional prompt refinement approaches focus on rephrasing the description of the primary task an LLM has to perform, neglecting the granular constraints that function as acceptance criteria for its response. We propose a novel multi-agentic workflow that decouples optimization of the primary task description from its constraints, using quantitative scores as feedback to iteratively rewrite and improve them. Our evaluation demonstrates this method produces revised prompts that yield significantly higher compliance scores from models like Llama 3.1 8B and Mixtral-8x 7B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03335v1">Digital Red Queen: Adversarial Program Evolution in Core War with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 14 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being used to evolve solutions to problems in many domains, in a process inspired by biological evolution. However, unlike biological evolution, most LLM-evolution frameworks are formulated as static optimization problems, overlooking the open-ended adversarial dynamics that characterize real-world evolutionary processes. Here, we study Digital Red Queen (DRQ), a simple self-play algorithm that embraces these so-called "Red Queen" dynamics via continual adaptation to a changing objective. DRQ uses an LLM to evolve assembly-like programs, called warriors, which compete against each other for control of a virtual machine in the game of Core War, a Turing-complete environment studied in artificial life and connected to cybersecurity. In each round of DRQ, the model evolves a new warrior to defeat all previous ones, producing a sequence of adapted warriors. Over many rounds, we observe that warriors become increasingly general (relative to a set of held-out human warriors). Interestingly, warriors also become less behaviorally diverse across independent runs, indicating a convergence pressure toward a general-purpose behavioral strategy, much like convergent evolution in nature. This result highlights a potential value of shifting from static objectives to dynamic Red Queen objectives. Our work positions Core War as a rich, controllable sandbox for studying adversarial adaptation in artificial systems and for evaluating LLM-based evolution methods. More broadly, the simplicity and effectiveness of DRQ suggest that similarly minimal self-play approaches could prove useful in other more practical multi-agent adversarial domains, like real-world cybersecurity or combating drug resistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03328v1">LLM-Enabled Multi-Agent Systems: Empirical Evaluation and Insights into Emerging Design Patterns & Paradigms</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      This paper formalises the literature on emerging design patterns and paradigms for Large Language Model (LLM)-enabled multi-agent systems (MAS), evaluating their practical utility across various domains. We define key architectural components, including agent orchestration, communication mechanisms, and control-flow strategies, and demonstrate how these enable rapid development of modular, domain-adaptive solutions. Three real-world case studies are tested in controlled, containerised pilots in telecommunications security, national heritage asset management, and utilities customer service automation. Initial empirical results show that, for these case studies, prototypes were delivered within two weeks and pilot-ready solutions within one month, suggesting reduced development overhead compared to conventional approaches and improved user accessibility. However, findings also reinforce limitations documented in the literature, including variability in LLM behaviour that leads to challenges in transitioning from prototype to production maturity. We conclude by outlining critical research directions for improving reliability, scalability, and governance in MAS architectures and the further work needed to mature MAS design patterns to mitigate the inherent challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03320v1">Ratio-Variance Regularized Policy Optimization for Efficient LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      On-policy reinforcement learning (RL), particularly Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO), has become the dominant paradigm for fine-tuning large language models (LLMs). While policy ratio clipping stabilizes training, this heuristic hard constraint incurs a fundamental cost: it indiscriminately truncates gradients from high-return yet high-divergence actions, suppressing rare but highly informative "eureka moments" in complex reasoning. Moreover, once data becomes slightly stale, hard clipping renders it unusable, leading to severe sample inefficiency. In this work, we revisit the trust-region objective in policy optimization and show that explicitly constraining the \emph{variance (second central moment) of the policy ratio} provides a principled and smooth relaxation of hard clipping. This distributional constraint stabilizes policy updates while preserving gradient signals from valuable trajectories. Building on this insight, we propose $R^2VPO$ (Ratio-Variance Regularized Policy Optimization), a novel primal-dual framework that supports stable on-policy learning and enables principled off-policy data reuse by dynamically reweighting stale samples rather than discarding them. We extensively evaluate $R^2VPO$ on fine-tuning state-of-the-art LLMs, including DeepSeek-Distill-Qwen-1.5B and the openPangu-Embedded series (1B and 7B), across challenging mathematical reasoning benchmarks. Experimental results show that $R^2VPO$ consistently achieves superior asymptotic performance, with average relative gains of up to 17% over strong clipping-based baselines, while requiring approximately 50% fewer rollouts to reach convergence. These findings establish ratio-variance control as a promising direction for improving both stability and data efficiency in RL-based LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03315v1">Why LLMs Aren't Scientists Yet: Lessons from Four Autonomous Research Attempts</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      We report a case study of four end-to-end attempts to autonomously generate ML research papers using a pipeline of six LLM agents mapped to stages of the scientific workflow. Of these four, three attempts failed during implementation or evaluation. One completed the pipeline and was accepted to Agents4Science 2025, an experimental inaugural venue that required AI systems as first authors, passing both human and multi-AI review. From these attempts, we document six recurring failure modes: bias toward training data defaults, implementation drift under execution pressure, memory and context degradation across long-horizon tasks, overexcitement that declares success despite obvious failures, insufficient domain intelligence, and weak scientific taste in experimental design. We conclude by discussing four design principles for more robust AI-scientist systems, implications for autonomous scientific discovery, and we release all prompts, artifacts, and outputs at https://github.com/Lossfunk/ai-scientist-artefacts-v1
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03251v1">NavAI: A Generalizable LLM Framework for Navigation Tasks in Virtual Reality Environments</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Navigation is one of the fundamental tasks for automated exploration in Virtual Reality (VR). Existing technologies primarily focus on path optimization in 360-degree image datasets and 3D simulators, which cannot be directly applied to immersive VR environments. To address this gap, we present NavAI, a generalizable large language model (LLM)-based navigation framework that supports both basic actions and complex goal-directed tasks across diverse VR applications. We evaluate NavAI in three distinct VR environments through goal-oriented and exploratory tasks. Results show that it achieves high accuracy, with an 89% success rate in goal-oriented tasks. Our analysis also highlights current limitations of relying entirely on LLMs, particularly in scenarios that require dynamic goal assessment. Finally, we discuss the limitations observed during the experiments and offer insights for future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.05665v3">Characterizing the Robustness of Black-Box LLM Planners Under Perturbed Observations with Adaptive Stress Testing</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 30 pages, 24 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated success in decision-making tasks including planning, control, and prediction, but their tendency to hallucinate unsafe and undesired outputs poses risks. This unwanted behavior is further exacerbated in environments where sensors are noisy or unreliable. Characterizing the behavior of LLM planners to varied observations is necessary to proactively avoid failures in safety-critical scenarios. We specifically investigate the response of LLMs along two different perturbation dimensions. Like prior works, one dimension generates semantically similar prompts with varied phrasing by randomizing order of details, modifying access to few-shot examples, etc. Unique to our work, the second dimension simulates access to varied sensors and noise to mimic raw sensor or detection algorithm failures. An initial case study in which perturbations are manually applied show that both dimensions lead LLMs to hallucinate in a multi-agent driving environment. However, manually covering the entire perturbation space for several scenarios is infeasible. As such, we propose a novel method for efficiently searching the space of prompt perturbations using adaptive stress testing (AST) with Monte-Carlo tree search (MCTS). Our AST formulation enables discovery of scenarios, sensor configurations, and prompt phrasing that cause language models to act with high uncertainty or even crash. By generating MCTS prompt perturbation trees across diverse scenarios, we show through extensive experiments that offline analyses can be used to proactively understand potential failures that may arise at runtime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03248v1">STReasoner: Empowering LLMs for Spatio-Temporal Reasoning in Time Series via Spatial-Aware Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 preprint, we release our code publicly at https://github.com/LingFengGold/STReasoner
    </div>
    <details class="paper-abstract">
      Spatio-temporal reasoning in time series involves the explicit synthesis of temporal dynamics, spatial dependencies, and textual context. This capability is vital for high-stakes decision-making in systems such as traffic networks, power grids, and disease propagation. However, the field remains underdeveloped because most existing works prioritize predictive accuracy over reasoning. To address the gap, we introduce ST-Bench, a benchmark consisting of four core tasks, including etiological reasoning, entity identification, correlation reasoning, and in-context forecasting, developed via a network SDE-based multi-agent data synthesis pipeline. We then propose STReasoner, which empowers LLM to integrate time series, graph structure, and text for explicit reasoning. To promote spatially grounded logic, we introduce S-GRPO, a reinforcement learning algorithm that rewards performance gains specifically attributable to spatial information. Experiments show that STReasoner achieves average accuracy gains between 17% and 135% at only 0.004X the cost of proprietary models and generalizes robustly to real-world data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03205v1">UltraLogic: Enhancing LLM Reasoning through Large-Scale Data Synthesis and Bipolar Float Reward</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 19 pages, 6 figures, 7 tables
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated significant potential in natural language processing , complex general-purpose reasoning requiring multi-step logic, planning, and verification remains a critical bottleneck. Although Reinforcement Learning with Verifiable Rewards (RLVR) has succeeded in specific domains , the field lacks large-scale, high-quality, and difficulty-calibrated data for general reasoning. To address this, we propose UltraLogic, a framework that decouples the logical core of a problem from its natural language expression through a Code-based Solving methodology to automate high-quality data production. The framework comprises hundreds of unique task types and an automated calibration pipeline across ten difficulty levels. Furthermore, to mitigate binary reward sparsity and the Non-negative Reward Trap, we introduce the Bipolar Float Reward (BFR) mechanism, utilizing graded penalties to effectively distinguish perfect responses from those with logical flaws. Our experiments demonstrate that task diversity is the primary driver for reasoning enhancement , and that BFR, combined with a difficulty matching strategy, significantly improves training efficiency, guiding models toward global logical optima.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03194v1">X-MuTeST: A Multilingual Benchmark for Explainable Hate Speech Detection and A Novel LLM-consulted Explanation Framework</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Accepted in the proceedings of AAAI 2026
    </div>
    <details class="paper-abstract">
      Hate speech detection on social media faces challenges in both accuracy and explainability, especially for underexplored Indic languages. We propose a novel explainability-guided training framework, X-MuTeST (eXplainable Multilingual haTe Speech deTection), for hate speech detection that combines high-level semantic reasoning from large language models (LLMs) with traditional attention-enhancing techniques. We extend this research to Hindi and Telugu alongside English by providing benchmark human-annotated rationales for each word to justify the assigned class label. The X-MuTeST explainability method computes the difference between the prediction probabilities of the original text and those of unigrams, bigrams, and trigrams. Final explanations are computed as the union between LLM explanations and X-MuTeST explanations. We show that leveraging human rationales during training enhances both classification performance and explainability. Moreover, combining human rationales with our explainability method to refine the model attention yields further improvements. We evaluate explainability using Plausibility metrics such as Token-F1 and IOU-F1 and Faithfulness metrics such as Comprehensiveness and Sufficiency. By focusing on under-resourced languages, our work advances hate speech detection across diverse linguistic contexts. Our dataset includes token-level rationale annotations for 6,004 Hindi, 4,492 Telugu, and 6,334 English samples. Data and code are available on https://github.com/ziarehman30/X-MuTeST
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.02790v3">Leveraging the true depth of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      The remarkable capabilities of Large Language Models (LLMs) are overshadowed by their immense computational cost. While recent work has shown that many LLM layers can be reordered or even removed with minimal impact on accuracy, these insights have not been translated into significant inference speedups. To bridge this gap, we introduce a novel method that restructures the computational graph by grouping and evaluating consecutive layer pairs in parallel. This approach, requiring no retraining, yields a 1.19x throughput gain on Llama 2 7B while reducing the average benchmark accuracy by only 1.5\%. We demonstrate the practical value of this method for large-scale LLM deployment and show that some of the lost accuracy can be recovered with lightweight fine-tuning of the parallelized layers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03190v1">Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Machine unlearning aims to forget sensitive knowledge from Large Language Models (LLMs) while maintaining general utility. However, existing approaches typically treat all tokens in a response indiscriminately and enforce uncertainty over the entire vocabulary. This global treatment results in unnecessary utility degradation and extends optimization to content-agnostic regions. To address these limitations, we propose PALU (Prefix-Aware Localized Unlearning), a framework driven by a local entropy maximization objective across both temporal and vocabulary dimensions. PALU reveals that (i) suppressing the sensitive prefix alone is sufficient to sever the causal generation link, and (ii) flattening only the top-$k$ logits is adequate to maximize uncertainty in the critical subspace. These findings allow PALU to avoid redundant optimization across the full vocabulary and parameter space while minimizing collateral damage to general model performance. Extensive experiments validate that PALU achieves superior forgetting efficacy and utility preservation compared to state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15125v2">Iterative Topic Taxonomy Induction with LLMs: A Case Study of Electoral Advertising</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Under-submission
    </div>
    <details class="paper-abstract">
      Social media platforms play a pivotal role in shaping political discourse, but analyzing their vast and rapidly evolving content remains a major challenge. We introduce an end-to-end framework for automatically inducing an interpretable topic taxonomy from unlabeled text corpora. By combining unsupervised clustering with prompt-based inference, our method leverages large language models (LLMs) to iteratively construct a taxonomy without requiring seed sets (predefined labels) or domain expertise. We validate the framework through a study of political advertising ahead of the 2024 U.S. presidential election. The induced taxonomy yields semantically rich topic labels and supports downstream analyses, including moral framing, in this setting. Results suggest that structured, iterative labeling yields more consistent and interpretable topic labels than existing approaches under human evaluation, and is practical for analyzing large-scale political advertising data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03178v1">DiffBench Meets DiffAgent: End-to-End LLM-Driven Diffusion Acceleration Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Diffusion models have achieved remarkable success in image and video generation. However, their inherently multiple step inference process imposes substantial computational overhead, hindering real-world deployment. Accelerating diffusion models is therefore essential, yet determining how to combine multiple model acceleration techniques remains a significant challenge. To address this issue, we introduce a framework driven by large language models (LLMs) for automated acceleration code generation and evaluation. First, we present DiffBench, a comprehensive benchmark that implements a three stage automated evaluation pipeline across diverse diffusion architectures, optimization combinations and deployment scenarios. Second, we propose DiffAgent, an agent that generates optimal acceleration strategies and codes for arbitrary diffusion models. DiffAgent employs a closed-loop workflow in which a planning component and a debugging component iteratively refine the output of a code generation component, while a genetic algorithm extracts performance feedback from the execution environment to guide subsequent code refinements. We provide a detailed explanation of the DiffBench construction and the design principles underlying DiffAgent. Extensive experiments show that DiffBench offers a thorough evaluation of generated codes and that DiffAgent significantly outperforms existing LLMs in producing effective diffusion acceleration strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.06254v5">EviRerank: Adaptive Evidence Construction for Long-Document LLM Reranking</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Decoder-only LLM rerankers struggle with long documents: inference is costly and relevance signals can be diluted by irrelevant context. Motivated by an attention analysis indicating a consistent degradation trend when non-relevant text is appended, we propose EviRerank, an evidence-based long-document reranking framework for decoder-only LLMs. EviRerank (i) scores document blocks with a lightweight selector (BM25, bi-encoder, or cross-encoder), (ii) constructs a compact reranking context under a hard token cap by dynamically budgeting evidence blocks with Adaptive Evidence Budgeting (AEB) and adding a global summary cue via Summary Augmentation (SA), and (iii) reranks with a decoder-only LLM. Across TREC DL'19, DL'23, and MLDR-zh, EviRerank consistently outperforms full-document LLM reranking and strong block-selection baselines while substantially reducing the required input length. On TREC DL'19, EviRerank achieves 0.743 nDCG@10 and 0.307 MAP, establishing a new best result and improving over RankLLaMA (0.701/0.288) by +0.042 nDCG@10 (+6.0%) and +0.019 MAP (+6.6%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03149v1">PersonaLedger: Generating Realistic Financial Transactions with Persona Conditioned LLMs and Rule Grounded Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Strict privacy regulations limit access to real transaction data, slowing open research in financial AI. Synthetic data can bridge this gap, but existing generators do not jointly achieve behavioral diversity and logical groundedness. Rule-driven simulators rely on hand-crafted workflows and shallow stochasticity, which miss the richness of human behavior. Learning-based generators such as GANs capture correlations yet often violate hard financial constraints and still require training on private data. We introduce PersonaLedger, a generation engine that uses a large language model conditioned on rich user personas to produce diverse transaction streams, coupled with an expert configurable programmatic engine that maintains correctness. The LLM and engine interact in a closed loop: after each event, the engine updates the user state, enforces financial rules, and returns a context aware "nextprompt" that guides the LLM toward feasible next actions. With this engine, we create a public dataset of 30 million transactions from 23,000 users and a benchmark suite with two tasks, illiquidity classification and identity theft segmentation. PersonaLedger offers a realistic, privacy preserving resource that supports rigorous evaluation of forecasting and anomaly detection models. PersonaLedger offers the community a rich, realistic, and privacy preserving resource -- complete with code, rules, and generation logs -- to accelerate innovation in financial AI and enable rigorous, reproducible evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03137v1">Accurate Table Question Answering with Accessible LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 accepted for publication in the Proceedings of the IEEE International Conference on Data Engineering (ICDE) 2026
    </div>
    <details class="paper-abstract">
      Given a table T in a database and a question Q in natural language, the table question answering (TQA) task aims to return an accurate answer to Q based on the content of T. Recent state-of-the-art solutions leverage large language models (LLMs) to obtain high-quality answers. However, most rely on proprietary, large-scale LLMs with costly API access, posing a significant financial barrier. This paper instead focuses on TQA with smaller, open-weight LLMs that can run on a desktop or laptop. This setting is challenging, as such LLMs typically have weaker capabilities than large proprietary models, leading to substantial performance degradation with existing methods. We observe that a key reason for this degradation is that prior approaches often require the LLM to solve a highly sophisticated task using long, complex prompts, which exceed the capabilities of small open-weight LLMs. Motivated by this observation, we present Orchestra, a multi-agent approach that unlocks the potential of accessible LLMs for high-quality, cost-effective TQA. Orchestra coordinates a group of LLM agents, each responsible for a relatively simple task, through a structured, layered workflow to solve complex TQA problems -- akin to an orchestra. By reducing the prompt complexity faced by each agent, Orchestra significantly improves output reliability. We implement Orchestra on top of AgentScope, an open-source multi-agent framework, and evaluate it on multiple TQA benchmarks using a wide range of open-weight LLMs. Experimental results show that Orchestra achieves strong performance even with small- to medium-sized models. For example, with Qwen2.5-14B, Orchestra reaches 72.1% accuracy on WikiTQ, approaching the best prior result of 75.3% achieved with GPT-4; with larger Qwen, Llama, or DeepSeek models, Orchestra outperforms all prior methods and establishes new state-of-the-art results across all benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03134v1">The Anatomy of Conversational Scams: A Topic-Based Red Teaming Analysis of Multi-Turn Interactions in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      As LLMs gain persuasive agentic capabilities through extended dialogues, they introduce novel risks in multi-turn conversational scams that single-turn safety evaluations fail to capture. We systematically study these risks using a controlled LLM-to-LLM simulation framework across multi-turn scam scenarios. Evaluating eight state-of-the-art models in English and Chinese, we analyze dialogue outcomes and qualitatively annotate attacker strategies, defensive responses, and failure modes. Results reveal that scam interactions follow recurrent escalation patterns, while defenses employ verification and delay mechanisms. Furthermore, interactional failures frequently stem from safety guardrail activation and role instability. Our findings highlight multi-turn interactional safety as a critical, distinct dimension of LLM behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.24045v2">Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Personal LLM agents increasingly combine foreground reactive interactions with background proactive monitoring, forming long-lived, stateful LLM flows that interleave prefill and token-by-token decode. While modern heterogeneous SoCs integrate CPUs, iGPUs, and NPUs to support on-device intelligence, existing LLM engines assume static, single-shot inference and lack mechanisms for flow-level concurrency, prioritization, and efficient accelerator coordination. As a result, commodity SoCs remain poorly matched to the dynamic, mixed-criticality execution patterns of personal agents. This paper presents Agent$.$xpu, the first LLM engine that orchestrates concurrent reactive and proactive LLM flows on commodity SoCs. Extensive profiling uncovers unique SoC characteristics of operator-accelerator affinity, asymmetric DDR contention, and stage-divergent batching behaviors distinct from cloud-serving assumptions. Agent$.$xpu introduces three key techniques: a heterogeneous execution graph (HEG) capturing NPU/iGPU affinity and elastic operator binding; flow-aware NPU-iGPU coordination with stage elasticity, decoupling prefill and decode to reduce bandwidth contention and enforce priorities; and fine-grained preemption with slack-aware piggybacking to guarantee reactive responsiveness without starving proactive work. Across realistic personal-agent workloads, Agent$.$xpu delivers 1.2-4.9$\times$ proactive throughput and reduces reactive latency by at least 91%, compared with both industrial iGPU-only serving engine and NPU-iGPU static inference with optimal tensor-partitioning schemes. Agent$.$xpu also minimizes energy consumption and graphics interference via controlled iGPU usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01836v1">COMPASS: A Framework for Evaluating Organization-Specific Policy Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      As large language models are deployed in high-stakes enterprise applications, from healthcare to finance, ensuring adherence to organization-specific policies has become essential. Yet existing safety evaluations focus exclusively on universal harms. We present COMPASS (Company/Organization Policy Alignment Assessment), the first systematic framework for evaluating whether LLMs comply with organizational allowlist and denylist policies. We apply COMPASS to eight diverse industry scenarios, generating and validating 5,920 queries that test both routine compliance and adversarial robustness through strategically designed edge cases. Evaluating seven state-of-the-art models, we uncover a fundamental asymmetry: models reliably handle legitimate requests (>95% accuracy) but catastrophically fail at enforcing prohibitions, refusing only 13-40% of adversarial denylist violations. These results demonstrate that current LLMs lack the robustness required for policy-critical deployments, establishing COMPASS as an essential evaluation framework for organizational AI safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.12885v3">VAR-MATH: Probing True Mathematical Reasoning in LLMS via Symbolic Multi-Instance Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) have led to substantial improvements in the mathematical reasoning abilities of LLMs, as measured by standard benchmarks. Yet these gains often persist even when models are trained with flawed signals, such as random or inverted rewards. This raises a fundamental question: do such improvements reflect genuine reasoning, or are they merely artifacts of overfitting to benchmark-specific patterns? To answer this question, we adopt an evaluation-centric perspective and highlight two critical shortcomings in existing protocols. First, benchmark contamination arises because test problems are publicly available, thereby increasing the risk of data leakage. Second, evaluation fragility results from reliance on single-instance assessments, which are sensitive to stochastic outputs and fail to capture reasoning consistency. These limitations suggest the need for a new evaluation paradigm that can probe reasoning ability beyond memorization and one-off success. As response, we propose VAR-MATH, a symbolic evaluation framework that converts fixed numerical problems into parameterized templates and requires models to solve multiple instantiations of each. This design enforces consistency across structurally equivalent variants, mitigates contamination, and enhances robustness through bootstrapped metrics. We apply VAR-MATH to transform three popular benchmarks, AMC23, AIME24, and AIME25, into their symbolic counterparts, VAR-AMC23, VAR-AIME24, and VAR-AIME25. Experimental results show substantial performance drops for RL-trained models on these variabilized benchmarks, especially for smaller models, with average declines of 47.9\% on AMC23, 58.8\% on AIME24, and 72.9\% on AIME25. These findings indicate that some existing RL methods rely on superficial heuristics and fail to generalize beyond specific numerical forms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01804v1">Causality-Aware Temporal Projection for Video Understanding in Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 7 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recent Video Large Language Models (Video-LLMs) have shown strong multimodal reasoning capabilities, yet remain challenged by video understanding tasks that require consistent temporal ordering and causal coherence. Many parameter-efficient Video-LLMs rely on unconstrained bidirectional projectors to model inter-frame interactions, which can blur temporal ordering by allowing later frames to influence earlier representations, without explicit architectural mechanisms to respect the directional nature of video reasoning. To address this limitation, we propose V-CORE, a parameter-efficient framework that introduces explicit temporal ordering constraints for video understanding. V-CORE consists of two key components: (1) Learnable Spatial Aggregation (LSA), which adaptively selects salient spatial tokens to reduce redundancy, and (2) a Causality-Aware Temporal Projector (CATP), which enforces structured unidirectional information flow via block-causal attention and a terminal dynamic summary token acting as a causal sink. This design preserves intra-frame spatial interactions while ensuring that temporal information is aggregated in a strictly ordered manner. With 4-bit QLoRA and a frozen LLM backbone, V-CORE can be trained efficiently on a single consumer GPU. Experiments show that V-CORE achieves strong performance on the challenging NExT-QA benchmark, reaching 61.2% accuracy, and remains competitive across MSVD-QA, MSRVTT-QA, and TGIF-QA, with gains concentrated in temporal and causal reasoning subcategories (+3.5% and +5.2% respectively), directly validating the importance of explicit temporal ordering constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00588v2">CSSBench: Evaluating the Safety of Lightweight LLMs against Chinese-Specific Adversarial Patterns</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in cost-sensitive and on-device scenarios, and safety guardrails have advanced mainly in English. However, real-world Chinese malicious queries typically conceal intent via homophones, pinyin, symbol-based splitting, and other Chinese-specific patterns. These Chinese-specific adversarial patterns create the safety evaluation gap that is not well captured by existing benchmarks focused on English. This gap is particularly concerning for lightweight models, which may be more vulnerable to such specific adversarial perturbations. To bridge this gap, we introduce the Chinese-Specific Safety Benchmark (CSSBench) that emphasizes these adversarial patterns and evaluates the safety of lightweight LLMs in Chinese. Our benchmark covers six domains that are common in real Chinese scenarios, including illegal activities and compliance, privacy leakage, health and medical misinformation, fraud and hate, adult content, and public and political safety, and organizes queries into multiple task types. We evaluate a set of popular lightweight LLMs and measure over-refusal behavior to assess safety-induced performance degradation. Our results show that the Chinese-specific adversarial pattern is a critical challenge for lightweight LLMs. This benchmark offers a comprehensive evaluation of LLM safety in Chinese, assisting robust deployments in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01768v1">Can LLMs Track Their Output Length? A Dynamic Feedback Mechanism for Precise Length Regulation</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Precisely controlling the length of generated text is a common requirement in real-world applications. However, despite significant advancements in following human instructions, Large Language Models (LLMs) still struggle with this task. In this work, we demonstrate that LLMs often fail to accurately measure input text length, leading to poor adherence to length constraints. To address this issue, we propose a novel length regulation approach that incorporates dynamic length feedback during generation, enabling adaptive adjustments to meet target lengths. Experiments on summarization and biography tasks show our training-free approach significantly improves precision in achieving target token, word, or sentence counts without compromising quality. Additionally, we demonstrate that further supervised fine-tuning allows our method to generalize effectively to broader text-generation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01751v1">Query-Document Dense Vectors for LLM Relevance Judgment Bias Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Accepted for presentation at the ECIR 2026 Full Papers track
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been used as relevance assessors for Information Retrieval (IR) evaluation collection creation due to reduced cost and increased scalability as compared to human assessors. While previous research has looked at the reliability of LLMs as compared to human assessors, in this work, we aim to understand if LLMs make systematic mistakes when judging relevance, rather than just understanding how good they are on average. To this aim, we propose a novel representational method for queries and documents that allows us to analyze relevance label distributions and compare LLM and human labels to identify patterns of disagreement and localize systematic areas of disagreement. We introduce a clustering-based framework that embeds query-document (Q-D) pairs into a joint semantic space, treating relevance as a relational property. Experiments on TREC Deep Learning 2019 and 2020 show that systematic disagreement between humans and LLMs is concentrated in specific semantic clusters rather than distributed randomly. Query-level analyses reveal recurring failures, most often in definition-seeking, policy-related, or ambiguous contexts. Queries with large variation in agreement across their clusters emerge as disagreement hotspots, where LLMs tend to under-recall relevant content or over-include irrelevant material. This framework links global diagnostics with localized clustering to uncover hidden weaknesses in LLM judgments, enabling bias-aware and more reliable IR evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.14397v3">Thunder-NUBench: A Benchmark for LLMs' Sentence-Level Negation Understanding</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Negation is a fundamental linguistic phenomenon that poses ongoing challenges for Large Language Models (LLMs), particularly in tasks requiring deep semantic understanding. Current benchmarks often treat negation as a minor detail within broader tasks, such as natural language inference. Consequently, there is a lack of benchmarks specifically designed to evaluate comprehension of negation. In this work, we introduce Thunder-NUBench, a novel benchmark explicitly created to assess sentence-level understanding of negation in LLMs. Thunder-NUBench goes beyond merely identifying surface-level cues by contrasting standard negation with structurally diverse alternatives, such as local negation, contradiction, and paraphrase. This benchmark includes manually curated sentence-negation pairs and a multiple-choice dataset, allowing for a comprehensive evaluation of models' understanding of negation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24618v2">Youtu-LLM: Unlocking the Native Agentic Potential for Lightweight Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 57 pages, 26 figures
    </div>
    <details class="paper-abstract">
      We introduce Youtu-LLM, a lightweight yet powerful language model that harmonizes high computational efficiency with native agentic intelligence. Unlike typical small models that rely on distillation, Youtu-LLM (1.96B) is pre-trained from scratch to systematically cultivate reasoning and planning capabilities. The key technical advancements are as follows: (1) Compact Architecture with Long-Context Support: Built on a dense Multi-Latent Attention (MLA) architecture with a novel STEM-oriented vocabulary, Youtu-LLM supports a 128k context window. This design enables robust long-context reasoning and state tracking within a minimal memory footprint, making it ideal for long-horizon agent and reasoning tasks. (2) Principled "Commonsense-STEM-Agent" Curriculum: We curated a massive corpus of approximately 11T tokens and implemented a multi-stage training strategy. By progressively shifting the pre-training data distribution from general commonsense to complex STEM and agentic tasks, we ensure the model acquires deep cognitive abilities rather than superficial alignment. (3) Scalable Agentic Mid-training: Specifically for the agentic mid-training, we employ diverse data construction schemes to synthesize rich and varied trajectories across math, coding, and tool-use domains. This high-quality data enables the model to internalize planning and reflection behaviors effectively. Extensive evaluations show that Youtu-LLM sets a new state-of-the-art for sub-2B LLMs. On general benchmarks, it achieves competitive performance against larger models, while on agent-specific tasks, it significantly surpasses existing SOTA baselines, demonstrating that lightweight models can possess strong intrinsic agentic capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06781v3">From Description to Score: Can LLMs Quantify Vulnerabilities?</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Manual vulnerability scoring, such as assigning Common Vulnerability Scoring System (CVSS) scores, is a resource-intensive process that is often influenced by subjective interpretation. This study investigates the potential of general-purpose large language models (LLMs), namely ChatGPT, Llama, Grok, DeepSeek, and Gemini, to automate this process by analyzing over 31{,}000 recent Common Vulnerabilities and Exposures (CVE) entries. The results show that LLMs substantially outperform the baseline on certain metrics (e.g., \textit{Availability Impact}), while offering more modest gains on others (e.g., \textit{Attack Complexity}). Moreover, model performance varies across both LLM families and individual CVSS metrics, with ChatGPT-5 attaining the highest precision. Our analysis reveals that LLMs tend to misclassify many of the same CVEs, and ensemble-based meta-classifiers only marginally improve performance. Further examination shows that CVE descriptions often lack critical context or contain ambiguous phrasing, which contributes to systematic misclassifications. These findings underscore the importance of enhancing vulnerability descriptions and incorporating richer contextual details to support more reliable automated reasoning and alleviate the growing backlog of CVEs awaiting triage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02604v1">Scalable Construction of a Lung Cancer Knowledge Base: Profiling Semantic Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into biomedical research offers new opportunities for domainspecific reasoning and knowledge representation. However, their performance depends heavily on the semantic quality of training data. In oncology, where precision and interpretability are vital, scalable methods for constructing structured knowledge bases are essential for effective fine-tuning. This study presents a pipeline for developing a lung cancer knowledge base using Open Information Extraction (OpenIE). The process includes: (1) identifying medical concepts with the MeSH thesaurus; (2) filtering open-access PubMed literature with permissive licenses (CC0); (3) extracting (subject, relation, object) triplets using OpenIE method; and (4) enriching triplet sets with Named Entity Recognition (NER) to ensure biomedical relevance. The resulting triplet sets provide a domain-specific, large-scale, and noise-aware resource for fine-tuning LLMs. We evaluated T5 models finetuned on this dataset through Supervised Semantic Fine-Tuning. Comparative assessments with ROUGE and BERTScore show significantly improved performance and semantic coherence, demonstrating the potential of OpenIE-derived resources as scalable, low-cost solutions for enhancing biomedical NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02598v1">LongDA: Benchmarking LLM Agents for Long-Document Data Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      We introduce LongDA, a data analysis benchmark for evaluating LLM-based agents under documentation-intensive analytical workflows. In contrast to existing benchmarks that assume well-specified schemas and inputs, LongDA targets real-world settings in which navigating long documentation and complex data is the primary bottleneck. To this end, we manually curate raw data files, long and heterogeneous documentation, and expert-written publications from 17 publicly available U.S. national surveys, from which we extract 505 analytical queries grounded in real analytical practice. Solving these queries requires agents to first retrieve and integrate key information from multiple unstructured documents, before performing multi-step computations and writing executable code, which remains challenging for existing data analysis agents. To support the systematic evaluation under this setting, we develop LongTA, a tool-augmented agent framework that enables document access, retrieval, and code execution, and evaluate a range of proprietary and open-source models. Our experiments reveal substantial performance gaps even among state-of-the-art models, highlighting the challenges researchers should consider before applying LLM agents for decision support in real-world, high-stakes analytical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05431v3">Self-Filtered Distillation with LLMs-generated Trust Indicators for Reliable Patent Classification</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly generate natural language rationales to enhance interpretability, but these often contain logical errors, label mismatches, and domain-specific misalignments. Directly using such rationales as supervision risks propagating noise and undermining training stability. To address this challenge, we introduce Self-Filtered Distillation, a framework tailored for patent classification that treats LLM-generated rationales as trust signals rather than ground-truth supervision. The framework employs selective distillation guided by three unsupervised trust metrics: (1) Self-Consistency, which measures the stability of LLM-generated rationales across multiple generations; (2) Class Entailment Alignment, which assesses semantic coherence with patent-specific class definitions; and (3) LLM Agreement Scoring, which validates rationale-label plausibility. These metrics are integrated into a unified trust score that primarily weights training samples while optionally filtering out extremely low-trust cases, enabling reasoning-aware supervision. Experiments on the USPTO-2M dataset show that our method consistently outperforms label-based learning and conventional distillation in accuracy, stability, and interpretability across diverse student architectures, establishing a reliable paradigm for leveraging reasoning-aware trust indicators in patent analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02569v1">LoRA-Drop: Temporal LoRA Decoding for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Autoregressive large language models (LLMs) are bottlenecked by sequential decoding, where each new token typically requires executing all transformer layers. Existing dynamic-depth and layer-skipping methods reduce this cost, but often rely on auxiliary routing mechanisms or incur accuracy degradation when bypassed layers are left uncompensated. We present \textbf{LoRA-Drop}, a plug-and-play inference framework that accelerates decoding by applying a \emph{temporal compute schedule} to a fixed subset of intermediate layers: on most decoding steps, selected layers reuse the previous-token hidden state and apply a low-rank LoRA correction, while periodic \emph{refresh} steps execute the full model to prevent drift. LoRA-Drop requires no routing network, is compatible with standard KV caching, and can reduce KV-cache footprint by skipping KV updates in droppable layers during LoRA steps and refreshing periodically. Across \textbf{LLaMA2-7B}, \textbf{LLaMA3-8B}, \textbf{Qwen2.5-7B}, and \textbf{Qwen2.5-14B}, LoRA-Drop achieves up to \textbf{2.6$\times$ faster decoding} and \textbf{45--55\% KV-cache reduction} while staying within \textbf{0.5 percentage points (pp)} of baseline accuracy. Evaluations on reasoning (GSM8K, MATH, BBH), code generation (HumanEval, MBPP), and long-context/multilingual benchmarks (LongBench, XNLI, XCOPA) identify a consistent \emph{safe zone} of scheduling configurations that preserves quality while delivering substantial efficiency gains, providing a simple path toward adaptive-capacity inference in LLMs. Codes are available at https://github.com/hosseinbv/LoRA-Drop.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11150v2">Causal Judge Evaluation: Calibrated Surrogate Metrics for LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Code: https://github.com/cimo-labs/cje Experiments for Reproducibility: https://github.com/cimo-labs/cje-arena-experiments Original Preprint: https://zenodo.org/records/17903629
    </div>
    <details class="paper-abstract">
      Measuring long-run LLM outcomes (user satisfaction, expert judgment, downstream KPIs) is expensive. Teams default to cheap LLM judges, but uncalibrated proxies can invert rankings entirely. Causal Judge Evaluation (CJE) makes it affordable to aim at the right target: calibrate cheap scores against 5% oracle labels, then evaluate at scale with valid uncertainty. On 4,961 Arena prompts, CJE achieves 99% ranking accuracy at 14x lower cost. Key findings: naive confidence intervals on uncalibrated scores achieve 0% coverage (CJE: ~95%); importance-weighted estimators fail despite 90%+ effective sample size. We introduce the Coverage-Limited Efficiency (CLE) diagnostic explaining why. CJE combines mean-preserving calibration (AutoCal-R), weight stabilization (SIMCal-W), and bootstrap inference that propagates calibration uncertainty (OUA), grounded in semiparametric efficiency theory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02559v1">PerspectiveCoach: Exploring LLMs for Developer Reflection</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 48th International Conference of Software Engineering
    </div>
    <details class="paper-abstract">
      Despite growing awareness of ethical challenges in software development, practitioners still lack structured tools that help them critically engage with the lived experiences of marginalized users. This paper presents PerspectiveCoach, a large language model (LLM)-powered conversational tool designed to guide developers through structured perspective-taking exercises and deepen critical reflection on how software design decisions affect marginalized communities. Through a controlled study with 18 front-end developers (balanced by sex), who interacted with the tool using a real case of online gender-based harassment, we examine how PerspectiveCoach supports ethical reasoning and engagement with user perspectives. Qualitative analysis revealed increased self-awareness, broadened perspectives, and more nuanced ethical articulation, while a complementary human-human study contextualized these findings. Text similarity analyses demonstrated that participants in the human-PerspectiveCoach study improved the fidelity of their restatements over multiple attempts, capturing both surface-level and semantic aspects of user concerns. However, human-PerspectiveCoach's restatements had a lower baseline than the human-human conversations, highlighting contextual differences in impersonal and interpersonal perspective-taking. Across the study, participants rated the tool highly for usability and relevance. This work contributes an exploratory design for LLM-powered end-user perspective-taking that supports critical, ethical self-reflection and offers empirical insights (i.e., enhancing adaptivity, centering plurality) into how such tools can help practitioners build more inclusive and socially responsive technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02553v1">SimpleMem: Efficient Lifelong Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      To support reliable long-term interaction in complex environments, LLM agents require memory systems that efficiently manage historical experiences. Existing approaches either retain full interaction histories via passive context extension, leading to substantial redundancy, or rely on iterative reasoning to filter noise, incurring high token costs. To address this challenge, we introduce SimpleMem, an efficient memory framework based on semantic lossless compression. We propose a three-stage pipeline designed to maximize information density and token utilization: (1) \textit{Semantic Structured Compression}, which applies entropy-aware filtering to distill unstructured interactions into compact, multi-view indexed memory units; (2) \textit{Recursive Memory Consolidation}, an asynchronous process that integrates related units into higher-level abstract representations to reduce redundancy; and (3) \textit{Adaptive Query-Aware Retrieval}, which dynamically adjusts retrieval scope based on query complexity to construct precise context efficiently. Experiments on benchmark datasets show that our method consistently outperforms baseline approaches in accuracy, retrieval efficiency, and inference cost, achieving an average F1 improvement of 26.4% while reducing inference-time token consumption by up to 30-fold, demonstrating a superior balance between performance and efficiency. Code is available at https://github.com/aiming-lab/SimpleMem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03278v2">Thucy: An LLM-based Multi-Agent System for Claim Verification across Relational Databases</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Accepted at AAAI 2026 Workshop on LLM-based Multi-Agent Systems (LaMAS)
    </div>
    <details class="paper-abstract">
      In today's age, it is becoming increasingly difficult to decipher truth from lies. Every day, politicians, media outlets, and public figures make conflicting claims -- often about topics that can, in principle, be verified against structured data. For instance, statements about crime rates, economic growth or healthcare can all be verified against official public records and structured datasets. Building a system that can automatically do that would have sounded like science fiction just a few years ago. Yet, with the extraordinary progress in LLMs and agentic AI, this is now within reach. Still, there remains a striking gap between what is technically possible and what is being demonstrated by recent work. Most existing verification systems operate only on small, single-table databases -- typically a few hundred rows -- that conveniently fit within an LLM's context window. In this paper we report our progress on Thucy, the first cross-database, cross-table multi-agent claim verification system that also provides concrete evidence for each verification verdict. Thucy remains completely agnostic to the underlying data sources before deployment and must therefore autonomously discover, inspect, and reason over all available relational databases to verify claims. Importantly, Thucy also reports the exact SQL queries that support its verdict (whether the claim is accurate or not) offering full transparency to expert users familiar with SQL. When evaluated on the TabFact dataset -- the standard benchmark for fact verification over structured data -- Thucy surpasses the previous state of the art by 5.6 percentage points in accuracy (94.3% vs. 88.7%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.15173v2">Uncovering Autoregressive LLM Knowledge of Thematic Fit in Event Representation</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Significant update with massive changes: all experiments rerun with current LLMs; includes new probability estimate analysis and expanded results in Sections 4 and 5
    </div>
    <details class="paper-abstract">
      We show closed models possess much thematic fit knowledge and set a new state of the art, while open models also seem to capture much relevant knowledge (in semantic filtering), but yield lower scores. Surprisingly, multi-step reasoning only helped closed models (with few exceptions); generated sentences hurt closed models' performance; and output form had little to no effect. We analyze the reasons for these findings, and conclude that more foundational work is needed for a single LLM to perform the best on all tasks with the same experimental condition, let alone improve results further. Source code is available at: https://github.com/SafeyahShemali/LLM_Thematic_Fit_25
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.16891v3">LLMs as Layout Designers: Enhanced Spatial Reasoning for Content-Aware Layout Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated impressive reasoning and planning abilities in textual domains and can effectively follow instructions for complex tasks, their ability to understand and manipulate spatial relationships remains limited. Such capabilities are crucial for content-aware graphic layout design, where the goal is to arrange heterogeneous elements onto a canvas so that final design remains visually balanced and structurally feasible. This problem requires precise coordination of placement, alignment, and structural organization of multiple elements within a constrained visual space. To address this limitation, we introduce LaySPA, a reinforcement learning-based framework that augments LLM-based agents with explicit spatial reasoning capabilities for layout design. LaySPA employs hybrid reward signals that jointly capture geometric constraints, structural fidelity, and visual quality, enabling agents to navigate the canvas, model inter-element relationships, and optimize spatial arrangements. Through group-relative policy optimization, the agent generates content-aware layouts that reflect salient regions, respect spatial constraints, and produces an interpretable reasoning trace explaining placement decisions and a structured layout specification. Experimental results show that LaySPA substantially improves the generation of structurally valid and visually appealing layouts, outperforming larger general-purpose LLMs and achieving performance comparable to state-of-the-art specialized layout models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.16882v4">SaVe-TAG: LLM-based Interpolation for Long-Tailed Text-Attributed Graphs</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Accepted KDD 2026 Research Track Paper
    </div>
    <details class="paper-abstract">
      Real-world graph data often follows long-tailed distributions, making it difficult for Graph Neural Networks (GNNs) to generalize well across both head and tail classes. Recent advances in Vicinal Risk Minimization (VRM) have shown promise in mitigating class imbalance with numeric interpolation; however, existing approaches largely rely on embedding-space arithmetic, which fails to capture the rich semantics inherent in text-attributed graphs. In this work, we propose our method, SaVe-TAG (Semantic-aware Vicinal Risk Minimization for Long-Tailed Text-Attributed Graphs), a novel VRM framework that leverages Large Language Models (LLMs) to perform text-level interpolation, generating on-manifold, boundary-enriching synthetic samples for minority classes. To mitigate the risk of noisy generation, we introduce a confidence-based edge assignment mechanism that uses graph topology as a natural filter to ensure structural consistency. We provide theoretical justification for our method and conduct extensive experiments on benchmark datasets, showing that our approach consistently outperforms both numeric interpolation and prior long-tailed node classification baselines. Our results highlight the importance of integrating semantic and structural signals for balanced and effective learning on text-attributed graphs. The source code is publicly available at: https://github.com/LWang-Laura/SaVe-TAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02512v1">Green LLM Techniques in Action: How Effective Are Existing Techniques for Improving the Energy Efficiency of LLM-Based Applications in Industry?</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Accepted for publication at the 2026 International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP'26)
    </div>
    <details class="paper-abstract">
      The rapid adoption of large language models (LLMs) has raised concerns about their substantial energy consumption, especially when deployed at industry scale. While several techniques have been proposed to address this, limited empirical evidence exists regarding the effectiveness of applying them to LLM-based industry applications. To fill this gap, we analyzed a chatbot application in an industrial context at Schuberg Philis, a Dutch IT services company. We then selected four techniques, namely Small and Large Model Collaboration, Prompt Optimization, Quantization, and Batching, applied them to the application in eight variations, and then conducted experiments to study their impact on energy consumption, accuracy, and response time compared to the unoptimized baseline. Our results show that several techniques, such as Prompt Optimization and 2-bit Quantization, managed to reduce energy use significantly, sometimes by up to 90%. However, these techniques especially impacted accuracy negatively, to a degree that is not acceptable in practice. The only technique that achieved significant and strong energy reductions without harming the other qualities substantially was Small and Large Model Collaboration via Nvidia's Prompt Task and Complexity Classifier (NPCC) with prompt complexity thresholds. This highlights that reducing the energy consumption of LLM-based applications is not difficult in practice. However, improving their energy efficiency, i.e., reducing energy use without harming other qualities, remains challenging. Our study provides practical insights to move towards this goal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02511v1">LLM-Enhanced Reinforcement Learning for Time Series Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Detecting anomalies in time series data is crucial for finance, healthcare, sensor networks, and industrial monitoring applications. However, time series anomaly detection often suffers from sparse labels, complex temporal patterns, and costly expert annotation. We propose a unified framework that integrates Large Language Model (LLM)-based potential functions for reward shaping with Reinforcement Learning (RL), Variational Autoencoder (VAE)-enhanced dynamic reward scaling, and active learning with label propagation. An LSTM-based RL agent leverages LLM-derived semantic rewards to guide exploration, while VAE reconstruction errors add unsupervised anomaly signals. Active learning selects the most uncertain samples, and label propagation efficiently expands labeled data. Evaluations on Yahoo-A1 and SMD benchmarks demonstrate that our method achieves state-of-the-art detection accuracy under limited labeling budgets and operates effectively in data-constrained settings. This study highlights the promise of combining LLMs with RL and advanced unsupervised techniques for robust, scalable anomaly detection in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02360v1">Heterogeneous Low-Bandwidth Pre-Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Pre-training large language models (LLMs) increasingly requires distributed compute, yet bandwidth constraints make it difficult to scale beyond well-provisioned datacenters-especially when model parallelism forces frequent, large inter-device communications. We study whether SparseLoCo, a low-communication data parallel method based on infrequent synchronization and sparse pseudo-gradient exchange, can be combined with low-bandwidth pipeline model parallelism via activation and activation-gradient compression. We introduce a heterogeneous distributed training framework where some participants host full replicas on high-bandwidth interconnects, while resource-limited participants are grouped to jointly instantiate a replica using pipeline parallelism with subspace-projected inter-stage communication. To make the recently introduced subspace pipeline compression compatible with SparseLoCo, we study a number of adaptations. Across large-scale language modeling experiments (178M-1B parameters) on standard pretraining corpora, we find that activation compression composes with SparseLoCo at modest cost, while selective (heterogeneous) compression consistently improves the loss-communication tradeoff relative to compressing all replicas-especially at aggressive compression ratios. These results suggest a practical path to incorporating low-bandwidth model parallelism and heterogeneous participants into LLM pre-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.18773v3">BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      The growth of long-context Large Language Models (LLMs) significantly increases memory and bandwidth pressure during autoregressive decoding due to the expanding Key-Value (KV) cache. While accuracy-preserving KV-cache quantization (e.g., 4-bit or 2-bit) reduces memory footprint, existing systems decode inefficiently by relying solely on CUDA cores, underutilizing Tensor Cores-the dominant compute resource on GPUs. We present BitDecoding, the first inference system to efficiently decode low-bit KV caches by cooperatively leveraging CUDA cores and Tensor Cores. BitDecoding smartly induces Tensor-Core-friendly layouts, introduces warp-level dequantization parallelism, and provides unified system support through query transformation, high-performance tensor- and channel-wise quantization, and a software-pipelined dequantization kernel enabling mixed-precision execution. Architecture-aware optimizations further leverage Hopper's warpgroup tensor instructions and Blackwell's NVFP4 (MXFP4) tensor formats. Evaluated on Blackwell, Hopper, and Ampere GPUs, BitDecoding achieves an average 7.5x decoding speedup over FP16 FlashDecoding-v2, up to 8.6x on Blackwell with NVFP4, and up to 4.3x over state-of-the-art approaches. On LLaMA-3.1-8B with a 128K context, BitDecoding reduces single-batch decoding latency by 3x. BitDecoding is open-sourced at https://github.com/OpenBitSys/BitDecoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02314v1">Project Ariadne: A Structural Causal Framework for Auditing Faithfulness in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM) agents are increasingly tasked with high-stakes autonomous decision-making, the transparency of their reasoning processes has become a critical safety concern. While \textit{Chain-of-Thought} (CoT) prompting allows agents to generate human-readable reasoning traces, it remains unclear whether these traces are \textbf{faithful} generative drivers of the model's output or merely \textbf{post-hoc rationalizations}. We introduce \textbf{Project Ariadne}, a novel XAI framework that utilizes Structural Causal Models (SCMs) and counterfactual logic to audit the causal integrity of agentic reasoning. Unlike existing interpretability methods that rely on surface-level textual similarity, Project Ariadne performs \textbf{hard interventions} ($do$-calculus) on intermediate reasoning nodes -- systematically inverting logic, negating premises, and reversing factual claims -- to measure the \textbf{Causal Sensitivity} ($φ$) of the terminal answer. Our empirical evaluation of state-of-the-art models reveals a persistent \textit{Faithfulness Gap}. We define and detect a widespread failure mode termed \textbf{Causal Decoupling}, where agents exhibit a violation density ($ρ$) of up to $0.77$ in factual and scientific domains. In these instances, agents arrive at identical conclusions despite contradictory internal logic, proving that their reasoning traces function as "Reasoning Theater" while decision-making is governed by latent parametric priors. Our findings suggest that current agentic architectures are inherently prone to unfaithful explanation, and we propose the Ariadne Score as a new benchmark for aligning stated logic with model action.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.09665v3">Tales of the 2025 Los Angeles Fire: Hotwash for Public Health Concerns in Reddit via LLM-Enhanced Topic Modeling</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Fix typos in Method Section. Add data/code availability
    </div>
    <details class="paper-abstract">
      Wildfires have become increasingly frequent, irregular, and severe in recent years. Understanding how affected populations perceive and respond during wildfire crises is critical for timely and empathetic disaster response. Social media platforms offer a crowd-sourced channel to capture evolving public discourse, providing hyperlocal information and insight into public sentiment. This study analyzes Reddit discourse during the 2025 Los Angeles wildfires, spanning from the onset of the disaster to full containment. We collect 385 posts and 114,879 comments related to the Palisades and Eaton fires. We adopt topic modeling methods to identify the latent topics, enhanced by large language models (LLMs) and human-in-the-loop (HITL) refinement. Furthermore, we develop a hierarchical framework to categorize latent topics, consisting of two main categories, Situational Awareness (SA) and Crisis Narratives (CN). The volume of SA category closely aligns with real-world fire progressions, peaking within the first 2-5 days as the fires reach the maximum extent. The most frequent co-occurring category set of public health and safety, loss and damage, and emergency resources expands on a wide range of health-related latent topics, including environmental health, occupational health, and one health. Grief signals and mental health risks consistently accounted for 60 percentage and 40 percentage of CN instances, respectively, with the highest total volume occurring at night. This study contributes the first annotated social media dataset on the 2025 LA fires, and introduces a scalable multi-layer framework that leverages topic modeling for crisis discourse analysis. By identifying persistent public health concerns, our results can inform more empathetic and adaptive strategies for disaster response, public health communication, and future research in comparable climate-related disaster events.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.06254v4">Adaptive Evidence Budgeting for Scalable Long-Document Reranking with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Decoder-only LLM rerankers are powerful but often struggle with long documents: inference is costly and relevance signals can be diluted as irrelevant text accumulates in the context window. Motivated by an attention analysis showing that relevance-aligned heads degrade when non-relevant text is appended, we propose EviRerank, a scalable framework that (i) scores document blocks with a lightweight selector (BM25, bi-encoder, or cross-encoder), (ii) constructs a compact evidence context under a strict token budget, and (iii) reranks with a decoder-only LLM. Our key contribution is Adaptive Evidence Budgeting (AEB), an information-density-aware dynamic stopping strategy that avoids low-utility tail blocks, and we further study Summary Augmentation (SA) within the same budget. Across TREC DL'19, DL'23, and MLDR-zh, EviRerank consistently improves over full-document LLM reranking and strong block-selection baselines while substantially reducing the required input length. On TREC DL'19, EviRerank achieves 0.743 nDCG@10 and 0.307 MAP, improving over RankLLaMA (0.701/0.288) by +0.042 nDCG@10 (+6.0%) and +0.019 MAP (+6.6%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04847v3">Grounded Test-Time Adaptation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Our code is available here: https://github.com/r2llab/GTTA
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents struggle to generalize to novel and complex environments, such as unseen websites or new sets of functions, due to a fundamental mismatch between their pre-training and test-time conditions. This challenge stems from two distinct failure modes: a syntactic misunderstanding of environment-specific components like observation formats, and a semantic misunderstanding of state-transition dynamics, which are only revealed at test time. To address these issues, we propose two distinct and complementary strategies for adapting LLM agents by leveraging environment-specific information available during deployment. First, an online distributional adaptation method parameterizes environmental nuances by learning a lightweight adaptation vector that biases the model's output distribution, enabling rapid alignment with an environment response format. Second, a deployment-time dynamics grounding method employs a persona-driven exploration phase to systematically probe and learn the environment's causal dynamics before task execution, equipping the agent with a nonparametric world model. We evaluate these strategies across diverse agentic benchmarks, including function calling and web navigation. Our empirical results show the effectiveness of both strategies across all benchmarks with minimal computational cost. We find that dynamics grounding is particularly effective in complex environments where unpredictable dynamics pose a major obstacle, demonstrating a robust path toward more generalizable and capable LLM-based agents. For example, on the WebArena multi-site split, this method increases the agent's success rate from 2% to 23%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02298v1">Power-of-Two Quantization-Aware-Training (PoT-QAT) in Large Language Models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      In Large Language Models (LLMs), the number of parameters has grown exponentially in the past few years, e.g., from 1.5 billion parameters in GPT-2 to 175 billion in GPT-3 to possibly more than trillion in higher versions. This raises a significant challenge for implementation, especially for Edge devices. Unlike cloud computing, memory and processing power for Edge devices are very limited, which necessitates developing novel ideas to make such applications feasible. In this work, we investigate compressing weights with a special quantization that limits numbers to only power-of-two (PoT). This helps save a huge amount of memory as only exponents need to be stored, more importantly, it significantly reduces processing power by replacing costly multiplication with low cost bit shifting. To overcome performance loss due to this strict quantization, we investigate Quantization Aware Training (QAT) to enhance performance through additional training. Results on GPT-2 124M show a major enhancement for quantized PoT model after additional training, with a perplexity enhancement of 66% and BERT-Score loss to baseline GPT-2 of 1%. The memory saving is estimated to be 87.5% while the inference speed is expected to be 3-10x faster with PoT quantization versus full-precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00469v2">DSL or Code? Evaluating the Quality of LLM-Generated Algebraic Specifications: A Case Study in Optimization at Kinaxis</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Accepted for publication in ICSE-SEIP 2026
    </div>
    <details class="paper-abstract">
      Model-driven engineering (MDE) provides abstraction and analytical rigour, but industrial adoption in many domains has been limited by the cost of developing and maintaining models. Large language models (LLMs) can help shift this cost balance by supporting direct generation of models from natural-language (NL) descriptions. For domain-specific languages (DSLs), however, LLM-generated models may be less accurate than LLM-generated code in mainstream languages such as Python, due to the latter's dominance in LLM training corpora. We investigate this issue in mathematical optimization, with AMPL, a DSL with established industrial use. We introduce EXEOS, an LLM-based approach that derives AMPL models and Python code from NL problem descriptions and iteratively refines them with solver feedback. Using a public optimization dataset and real-world supply-chain cases from our industrial partner Kinaxis, we evaluate generated AMPL models against Python code in terms of executability and correctness. An ablation study with two LLM families shows that AMPL is competitive with, and sometimes better than, Python, and that our design choices in EXEOS improve the quality of generated specifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.01752v3">Tuning without Peeking: Provable Generalization Bounds and Robust LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Gradient-based optimization is the workhorse of deep learning, offering efficient and scalable training via backpropagation. However, exposing gradients during training can leak sensitive information about the underlying data, raising privacy and security concerns such as susceptibility to data poisoning attacks. In contrast, black box optimization methods, which treat the model as an opaque function, relying solely on function evaluations to guide optimization, offer a promising alternative in scenarios where data access is restricted, adversarial risks are high, or overfitting is a concern. This paper introduces BBoxER, an evolutionary black-box method for LLM post-training that induces an information bottleneck via implicit compression of the training data. Leveraging the tractability of information flow, we provide non-vacuous generalization bounds and strong theoretical guarantees for privacy, robustness to data poisoning attacks, and extraction attacks. In experiments with LLMs, we demonstrate empirically that black-box optimization methods, despite the scalability and computational challenges inherent to black-box approaches, are able to learn, showing how a few iterations of BBoxER improve performance, generalize well on a benchmark of reasoning datasets, and are robust to membership inference attacks. This positions BBoxER as an attractive add-on on top of gradient-based optimization, offering suitability for deployment in restricted or privacy-sensitive environments while also providing non-vacuous generalization guarantees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02224v1">From XAI to Stories: A Factorial Study of LLM-Generated Explanation Quality</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Explainable AI (XAI) methods like SHAP and LIME produce numerical feature attributions that remain inaccessible to non expert users. Prior work has shown that Large Language Models (LLMs) can transform these outputs into natural language explanations (NLEs), but it remains unclear which factors contribute to high-quality explanations. We present a systematic factorial study investigating how Forecasting model choice, XAI method, LLM selection, and prompting strategy affect NLE quality. Our design spans four models (XGBoost (XGB), Random Forest (RF), Multilayer Perceptron (MLP), and SARIMAX - comparing black-box Machine-Learning (ML) against classical time-series approaches), three XAI conditions (SHAP, LIME, and a no-XAI baseline), three LLMs (GPT-4o, Llama-3-8B, DeepSeek-R1), and eight prompting strategies. Using G-Eval, an LLM-as-a-judge evaluation method, with dual LLM judges and four evaluation criteria, we evaluate 660 explanations for time-series forecasting. Our results suggest that: (1) XAI provides only small improvements over no-XAI baselines, and only for expert audiences; (2) LLM choice dominates all other factors, with DeepSeek-R1 outperforming GPT-4o and Llama-3; (3) we observe an interpretability paradox: in our setting, SARIMAX yielded lower NLE quality than ML models despite higher prediction accuracy; (4) zero-shot prompting is competitive with self-consistency at 7-times lower cost; and (5) chain-of-thought hurts rather than helps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02215v1">LLM-Empowered Functional Safety and Security by Design in Automotive Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      This paper presents LLM-empowered workflow to support Software Defined Vehicle (SDV) software development, covering the aspects of security-aware system topology design, as well as event-driven decision-making code analysis. For code analysis we adopt event chains model which provides formal foundations to systematic validation of functional safety, taking into account the semantic validity of messages exchanged between key components, including both CAN and Vehicle Signal Specification (VSS). Analysis of security aspects for topology relies on synergy with Model-Driven Engineering (MDE) approach and Object Constraint Language (OCL) rules. Both locally deployable and proprietary solution are taken into account for evaluation within Advanced Driver-Assistance Systems (ADAS)-related scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.22458v2">Evaluating LLM-based Agents for Multi-Turn Conversations: A Survey</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      This survey examines evaluation methods for large language model (LLM)-based agents in multi-turn conversational settings. Using a PRISMA-inspired framework, we systematically reviewed nearly 250 scholarly sources, capturing the state of the art from various venues of publication, and establishing a solid foundation for our analysis. Our study offers a structured approach by developing two interrelated taxonomy systems: one that defines \emph{what to evaluate} and another that explains \emph{how to evaluate}. The first taxonomy identifies key components of LLM-based agents for multi-turn conversations and their evaluation dimensions, including task completion, response quality, user experience, memory and context retention, as well as planning and tool integration. These components ensure that the performance of conversational agents is assessed in a holistic and meaningful manner. The second taxonomy system focuses on the evaluation methodologies. It categorizes approaches into annotation-based evaluations, automated metrics, hybrid strategies that combine human assessments with quantitative measures, and self-judging methods utilizing LLMs. This framework not only captures traditional metrics derived from language understanding, such as BLEU and ROUGE scores, but also incorporates advanced techniques that reflect the dynamic, interactive nature of multi-turn dialogues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02179v1">Confidence Estimation for LLMs in Multi-turn Interactions</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      While confidence estimation is a promising direction for mitigating hallucinations in Large Language Models (LLMs), current research dominantly focuses on single-turn settings. The dynamics of model confidence in multi-turn conversations, where context accumulates and ambiguity is progressively resolved, remain largely unexplored. Reliable confidence estimation in multi-turn settings is critical for many downstream applications, such as autonomous agents and human-in-the-loop systems. This work presents the first systematic study of confidence estimation in multi-turn interactions, establishing a formal evaluation framework grounded in two key desiderata: per-turn calibration and monotonicity of confidence as more information becomes available. To facilitate this, we introduce novel metrics, including a length-normalized Expected Calibration Error (InfoECE), and a new "Hinter-Guesser" paradigm for generating controlled evaluation datasets. Our experiments reveal that widely-used confidence techniques struggle with calibration and monotonicity in multi-turn dialogues. We propose P(Sufficient), a logit-based probe that achieves comparatively better performance, although the task remains far from solved. Our work provides a foundational methodology for developing more reliable and trustworthy conversational agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.11320v2">Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 49 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) power many modern applications, but their inference procedure poses unique scheduling challenges: the Key-Value (KV) cache grows dynamically during response generation, and memory overflow triggers eviction that can cascade into system-wide failures. Even when memory capacity exceeds the theoretical requirement, conventional scheduling algorithms fail because they do not account for this dynamic memory growth -- a system that should be stable can become unstable under poor scheduling. This paper formulates LLM inference optimization as a multi-stage online scheduling problem. We develop a fluid dynamics approximation to establish a tractable benchmark and derive the Waiting for Accumulated Inference Threshold (WAIT) algorithm. WAIT uses threshold-based batching to prevent eviction by keeping the system near load balance, achieving near-optimal throughput when output lengths are known. For practical settings where output lengths are unknown at arrival, we introduce Nested WAIT. Rather than predicting output lengths, Nested WAIT classifies prompts on-the-fly: short prompts complete early and exit, while longer prompts naturally advance to later segments. A safety buffer provides high-probability protection against memory overflow with only logarithmic overhead. Theoretical analysis establishes near-optimal performance in the asymptotic regime. Experiments on Llama-7B with an A100 GPU demonstrate that our approach achieves superior throughput and reduced latency compared to vLLM and Sarathi. This work applies operations research principles to establish a theoretical framework for LLM deployment under memory constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05623v2">Deployability-Centric Infrastructure-as-Code Generation: Fail, Learn, Refine, and Succeed through LLM-Empowered DevOps Simulation</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Accepted by FSE 2026
    </div>
    <details class="paper-abstract">
      Infrastructure-as-Code (IaC) generation holds significant promise for automating cloud infrastructure provisioning. Recent advances in Large Language Models (LLMs) present a promising opportunity to democratize IaC development by generating deployable infrastructure templates from natural language descriptions. However, current evaluation focuses on syntactic correctness while ignoring deployability, the critical measure of the utility of IaC configuration files. Six state-of-the-art LLMs performed poorly on deployability, achieving only 20.8$\sim$30.2% deployment success rate on the first attempt. In this paper, we construct DPIaC-Eval, the first deployability-centric IaC template benchmark consisting of 153 real-world scenarios cross 58 unique services. Also, we propose an LLM-based deployability-centric framework, dubbed IaCGen, that uses iterative feedback mechanism encompassing format verification, syntax checking, and live deployment stages, thereby closely mirroring the real DevOps workflows. Results show that IaCGen can make 54.6$\sim$91.6% generated IaC templates from all evaluated models deployable in the first 10 iterations. Additionally, human-in-the-loop feedback that provide direct guidance for the deployability errors, can further boost the performance to over 90% passItr@25 on all evaluated LLMs. Furthermore, we explore the trustworthiness of the generated IaC templates on user intent alignment and security compliance. The poor performance (25.2% user requirement coverage and 8.4% security compliance rate) indicates a critical need for continued research in this domain.
    </details>
</div>
