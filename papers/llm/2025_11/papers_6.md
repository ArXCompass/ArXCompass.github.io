# llm - 2025_11

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10480v1">Scalable Synthesis of distributed LLM workloads through Symbolic Tensor Graphs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Optimizing the performance of large language models (LLMs) on large-scale AI training and inference systems requires a scalable and expressive mechanism to model distributed workload execution. Such modeling is essential for pre-deployment system-level optimizations (e.g., parallelization strategies) and design-space explorations. While recent efforts have proposed collecting execution traces from real systems, access to large-scale infrastructure remains limited to major cloud providers. Moreover, traces obtained from existing platforms cannot be easily adapted to study future larger-scale system configurations. We introduce Symbolic Tensor grAph GEnerator(STAGE), a framework that synthesizes high-fidelity execution traces to accurately model LLM workloads. STAGE supports a comprehensive set of parallelization strategies, allowing users to systematically explore a wide spectrum of LLM architectures and system configurations. STAGE demonstrates its scalability by synthesizing high-fidelity LLM traces spanning over 32K GPUs, while preserving tensor-level accuracy in compute, memory, and communication. STAGE will be publicly available to facilitate further research in distributed machine learning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10459v1">LocalBench: Benchmarking LLMs on County-Level Local Knowledge and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely evaluated on macro-scale geographic tasks, such as global factual recall, event summarization, and regional reasoning. Yet, their ability to handle hyper-local knowledge remains poorly understood. This gap is increasingly consequential as real-world applications, from civic platforms to community journalism, demand AI systems that can reason about neighborhood-specific dynamics, cultural narratives, and local governance. Existing benchmarks fall short in capturing this complexity, often relying on coarse-grained data or isolated references. We present LocalBench, the first benchmark designed to systematically evaluate LLMs on county-level local knowledge across the United States. Grounded in the Localness Conceptual Framework, LocalBench includes 14,782 validated question-answer pairs across 526 U.S. counties in 49 states, integrating diverse sources such as Census statistics, local subreddit discourse, and regional news. It spans physical, cognitive, and relational dimensions of locality. Using LocalBench, we evaluate 13 state-of-the-art LLMs under both closed-book and web-augmented settings. Our findings reveal critical limitations: even the best-performing models reach only 56.8% accuracy on narrative-style questions and perform below 15.5% on numerical reasoning. Moreover, larger model size and web augmentation do not guarantee better performance, for example, search improves Gemini's accuracy by +13.6%, but reduces GPT-series performance by -11.4%. These results underscore the urgent need for language models that can support equitable, place-aware AI systems: capable of engaging with the diverse, fine-grained realities of local communities across geographic and cultural contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02635v2">Test Set Quality in Multilingual LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 to appear in the proceedings of Eval4NLP workshop at AACL 2025. Camera ready version
    </div>
    <details class="paper-abstract">
      Several multilingual benchmark datasets have been developed in a semi-automatic manner in the recent past to measure progress and understand the state-of-the-art in the multilingual capabilities of Large Language Models. However, there is not a lot of attention paid to the quality of the datasets themselves, despite the existence of previous work in identifying errors in even fully human-annotated test sets. In this paper, we manually analyze recent multilingual evaluation sets in two languages - French and Telugu, identifying several errors in the process. We compare the performance difference across several LLMs with the original and revised versions of the datasets and identify large differences (almost 10% in some cases) in both languages). Based on these results, we argue that test sets should not be considered immutable and should be revisited, checked for correctness, and potentially versioned. We end with some recommendations for both the dataset creators as well as consumers on addressing the dataset quality issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10394v1">LLM-YOLOMS: Large Language Model-based Semantic Interpretation and Fault Diagnosis for Wind Turbine Components</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Journal resubmission
    </div>
    <details class="paper-abstract">
      The health condition of wind turbine (WT) components is crucial for ensuring stable and reliable operation. However, existing fault detection methods are largely limited to visual recognition, producing structured outputs that lack semantic interpretability and fail to support maintenance decision-making. To address these limitations, this study proposes an integrated framework that combines YOLOMS with a large language model (LLM) for intelligent fault analysis and diagnosis. Specifically, YOLOMS employs multi-scale detection and sliding-window cropping to enhance fault feature extraction, while a lightweight key-value (KV) mapping module bridges the gap between visual outputs and textual inputs. This module converts YOLOMS detection results into structured textual representations enriched with both qualitative and quantitative attributes. A domain-tuned LLM then performs semantic reasoning to generate interpretable fault analyses and maintenance recommendations. Experiments on real-world datasets demonstrate that the proposed framework achieves a fault detection accuracy of 90.6\% and generates maintenance reports with an average accuracy of 89\%, thereby improving the interpretability of diagnostic results and providing practical decision support for the operation and maintenance of wind turbines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10381v1">Position: On the Methodological Pitfalls of Evaluating Base LLMs for Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Existing work investigates the reasoning capabilities of large language models (LLMs) to uncover their limitations, human-like biases and underlying processes. Such studies include evaluations of base LLMs (pre-trained on unlabeled corpora only) for this purpose. Our position paper argues that evaluating base LLMs' reasoning capabilities raises inherent methodological concerns that are overlooked in such existing studies. We highlight the fundamental mismatch between base LLMs' pretraining objective and normative qualities, such as correctness, by which reasoning is assessed. In particular, we show how base LLMs generate logically valid or invalid conclusions as coincidental byproducts of conforming to purely linguistic patterns of statistical plausibility. This fundamental mismatch challenges the assumptions that (a) base LLMs' outputs can be assessed as their bona fide attempts at correct answers or conclusions; and (b) conclusions about base LLMs' reasoning can generalize to post-trained LLMs optimized for successful instruction-following. We call for a critical re-examination of existing work that relies implicitly on these assumptions, and for future work to account for these methodological pitfalls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10354v1">Knowledge Graphs Generation from Cultural Heritage Texts: Combining LLMs and Ontological Engineering for Scholarly Debates</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 46 pages
    </div>
    <details class="paper-abstract">
      Cultural Heritage texts contain rich knowledge that is difficult to query systematically due to the challenges of converting unstructured discourse into structured Knowledge Graphs (KGs). This paper introduces ATR4CH (Adaptive Text-to-RDF for Cultural Heritage), a systematic five-step methodology for Large Language Model-based Knowledge Extraction from Cultural Heritage documents. We validate the methodology through a case study on authenticity assessment debates. Methodology - ATR4CH combines annotation models, ontological frameworks, and LLM-based extraction through iterative development: foundational analysis, annotation schema development, pipeline architecture, integration refinement, and comprehensive evaluation. We demonstrate the approach using Wikipedia articles about disputed items (documents, artifacts...), implementing a sequential pipeline with three LLMs (Claude Sonnet 3.7, Llama 3.3 70B, GPT-4o-mini). Findings - The methodology successfully extracts complex Cultural Heritage knowledge: 0.96-0.99 F1 for metadata extraction, 0.7-0.8 F1 for entity recognition, 0.65-0.75 F1 for hypothesis extraction, 0.95-0.97 for evidence extraction, and 0.62 G-EVAL for discourse representation. Smaller models performed competitively, enabling cost-effective deployment. Originality - This is the first systematic methodology for coordinating LLM-based extraction with Cultural Heritage ontologies. ATR4CH provides a replicable framework adaptable across CH domains and institutional resources. Research Limitations - The produced KG is limited to Wikipedia articles. While the results are encouraging, human oversight is necessary during post-processing. Practical Implications - ATR4CH enables Cultural Heritage institutions to systematically convert textual knowledge into queryable KGs, supporting automated metadata enrichment and knowledge discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10333v1">EDGC: Entropy-driven Dynamic Gradient Compression for Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) poses significant challenges regarding computational resources and memory capacity. Although distributed training techniques help mitigate these issues, they still suffer from considerable communication overhead. Existing approaches primarily rely on static gradient compression to enhance communication efficiency; however, these methods neglect the dynamic nature of evolving gradients during training, leading to performance degradation. Accelerating LLM training via compression without sacrificing performance remains a challenge. In this paper, we propose an entropy-driven dynamic gradient compression framework called EDGC. The core concept is to adjust the compression rate during LLM training based on the evolving trends of gradient entropy, taking into account both compression efficiency and error. EDGC consists of three key components.First, it employs a down-sampling method to efficiently estimate gradient entropy, reducing computation overhead. Second, it establishes a theoretical model linking compression rate with gradient entropy, enabling more informed compression decisions. Lastly, a window-based adjustment mechanism dynamically adapts the compression rate across pipeline stages, improving communication efficiency and maintaining model performance. We implemented EDGC on a 32-NVIDIA-V100 cluster and a 64-NVIDIA-H100 cluster to train GPT2-2.5B and GPT2-12.1B, respectively. The results show that EDGC significantly reduces communication latency and training time by up to 46.45% and 16.13% while preserving LLM accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02573v2">Guess or Recall? Training CNNs to Classify and Localize Memorization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 This paper has been accepted for publication at AAAI-26
    </div>
    <details class="paper-abstract">
      Verbatim memorization in Large Language Models (LLMs) is a multifaceted phenomenon involving distinct underlying mechanisms. We introduce a novel method to analyze the different forms of memorization described by the existing taxonomy. Specifically, we train Convolutional Neural Networks (CNNs) on the attention weights of the LLM and evaluate the alignment between this taxonomy and the attention weights involved in decoding. We find that the existing taxonomy performs poorly and fails to reflect distinct mechanisms within the attention blocks. We propose a new taxonomy that maximizes alignment with the attention weights, consisting of three categories: memorized samples that are guessed using language modeling abilities, memorized samples that are recalled due to high duplication in the training set, and non-memorized samples. Our results reveal that few-shot verbatim memorization does not correspond to a distinct attention mechanism. We also show that a significant proportion of extractable samples are in fact guessed by the model and should therefore be studied separately. Finally, we develop a custom visual interpretability technique to localize the regions of the attention weights involved in each form of memorization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10303v1">Rectify Evaluation Preference: Improving LLMs' Critique on Math Reasoning via Perplexity-aware Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Accepted by AAAI2026
    </div>
    <details class="paper-abstract">
      To improve Multi-step Mathematical Reasoning (MsMR) of Large Language Models (LLMs), it is crucial to obtain scalable supervision from the corpus by automatically critiquing mistakes in the reasoning process of MsMR and rendering a final verdict of the problem-solution. Most existing methods rely on crafting high-quality supervised fine-tuning demonstrations for critiquing capability enhancement and pay little attention to delving into the underlying reason for the poor critiquing performance of LLMs. In this paper, we orthogonally quantify and investigate the potential reason -- imbalanced evaluation preference, and conduct a statistical preference analysis. Motivated by the analysis of the reason, a novel perplexity-aware reinforcement learning algorithm is proposed to rectify the evaluation preference, elevating the critiquing capability. Specifically, to probe into LLMs' critiquing characteristics, a One-to-many Problem-Solution (OPS) benchmark is meticulously constructed to quantify the behavior difference of LLMs when evaluating the problem solutions generated by itself and others. Then, to investigate the behavior difference in depth, we conduct a statistical preference analysis oriented on perplexity and find an intriguing phenomenon -- ``LLMs incline to judge solutions with lower perplexity as correct'', which is dubbed as \textit{imbalanced evaluation preference}. To rectify this preference, we regard perplexity as the baton in the algorithm of Group Relative Policy Optimization, supporting the LLMs to explore trajectories that judge lower perplexity as wrong and higher perplexity as correct. Extensive experimental results on our built OPS and existing available critic benchmarks demonstrate the validity of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10301v1">Rethinking Visual Information Processing in Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Despite the remarkable success of the LLaVA architecture for vision-language tasks, its design inherently struggles to effectively integrate visual features due to the inherent mismatch between text and vision modalities. We tackle this issue from a novel perspective in which the LLM not only serves as a language model but also a powerful vision encoder. To this end, we present LLaViT - Large Language Models as extended Vision Transformers - which enables the LLM to simultaneously function as a vision encoder through three key modifications: (1) learning separate QKV projections for vision modality, (2) enabling bidirectional attention on visual tokens, and (3) incorporating both global and local visual representations. Through extensive controlled experiments on a wide range of LLMs, we demonstrate that LLaViT significantly outperforms the baseline LLaVA method on a multitude of benchmarks, even surpassing models with double its parameter count, establishing a more effective approach to vision-language modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.11034v2">The Impact of Large Language Models (LLMs) on Code Review Process</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently gained prominence in the field of software development, significantly boosting productivity and simplifying teamwork. Although prior studies have examined task-specific applications, the phase-specific effects of LLM assistance on the efficiency of code review processes remain underexplored. This research investigates the effect of GPT on GitHub pull request (PR) workflows, with a focus on reducing resolution time, optimizing phase-specific performance, and assisting developers. We curated a dataset of 25,473 PRs from 9,254 GitHub projects and identified GPT-assisted PRs using a semi-automated heuristic approach that combines keyword-based detection, regular expression filtering, and manual verification until achieving 95% labeling accuracy. We then applied statistical modeling, including multiple linear regression and Mann-Whitney U test, to evaluate differences between GPT-assisted and non-assisted PRs, both at the overall resolution level and across distinct review phases. Our research has revealed that early adoption of GPT can substantially boost the effectiveness of the PR process, leading to considerable time savings at various stages. Our findings suggest that GPT-assisted PRs reduced median resolution time by more than 60% (9 hours compared to 23 hours for non-assisted PRs). We discovered that utilizing GPT can reduce the review time by 33% and the waiting time before acceptance by 87%. Analyzing a sample dataset of 300 GPT-assisted PRs, we discovered that developers predominantly use GPT for code optimization (60%), bug fixing (26%), and documentation updates (12%). This research sheds light on the impact of the GPT model on the code review process, offering actionable insights for software teams seeking to enhance workflows and promote seamless collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10271v1">Quality Assurance of LLM-generated Code: Addressing Non-Functional Quality Characteristics</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      In recent years, LLMs have been widely integrated into software engineering workflows, supporting tasks like code generation. However, while these models often generate functionally correct outputs, we still lack a systematic understanding and evaluation of their non-functional qualities. Existing studies focus mainly on whether generated code passes the tests rather than whether it passes with quality. Guided by the ISO/IEC 25010 quality model, this study conducted three complementary investigations: a systematic review of 108 papers, two industry workshops with practitioners from multiple organizations, and an empirical analysis of patching real-world software issues using three LLMs. Motivated by insights from both the literature and practitioners, the empirical study examined the quality of generated patches on security, maintainability, and performance efficiency. Across the literature, we found that security and performance efficiency dominate academic attention, while maintainability and other qualities are understudied. In contrast, industry experts prioritize maintainability and readability, warning that generated code may accelerate the accumulation of technical debt. In our evaluation of functionally correct patches generated by three LLMs, improvements in one quality dimension often come at the cost of others. Runtime and memory results further show high variance across models and optimization strategies. Overall, our findings reveal a mismatch between academic focus, industry priorities, and model performance, highlighting the urgent need to integrate quality assurance mechanisms into LLM code generation pipelines to ensure that future generated code not only passes tests but truly passes with quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.19794v5">Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 AAAI 2026 (oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) hold promise in automating data analysis tasks, yet open-source models face significant limitations in these kinds of reasoning-intensive scenarios. In this work, we investigate strategies to enhance the data analysis capabilities of open-source LLMs. By curating a seed dataset of diverse, realistic scenarios, we evaluate model behavior across three core dimensions: data understanding, code generation, and strategic planning. Our analysis reveals three key findings: (1) Strategic planning quality serves as the primary determinant of model performance; (2) Interaction design and task complexity significantly influence reasoning capabilities; (3) Data quality demonstrates a greater impact than diversity in achieving optimal performance. We leverage these insights to develop a data synthesis methodology, demonstrating significant improvements in open-source LLMs' analytical reasoning capabilities. Code is available at https://github.com/zjunlp/DataMind.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10234v1">Lost in Serialization: Invariance and Generalization of LLM Graph Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 AAAI 2026 Workshop on Graphs and more Complex Structures For Learning and Reasoning (GCLR)
    </div>
    <details class="paper-abstract">
      While promising, graph reasoners based on Large Language Models (LLMs) lack built-in invariance to symmetries in graph representations. Operating on sequential graph serializations, LLMs can produce different outputs under node reindexing, edge reordering, or formatting changes, raising robustness concerns. We systematically analyze these effects, studying how fine-tuning impacts encoding sensitivity as well generalization on unseen tasks. We propose a principled decomposition of graph serializations into node labeling, edge encoding, and syntax, and evaluate LLM robustness to variations of each of these factors on a comprehensive benchmarking suite. We also contribute a novel set of spectral tasks to further assess generalization abilities of fine-tuned reasoners. Results show that larger (non-fine-tuned) models are more robust. Fine-tuning reduces sensitivity to node relabeling but may increase it to variations in structure and format, while it does not consistently improve performance on unseen tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10233v1">Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 21 pages; To be published in AAAI-26
    </div>
    <details class="paper-abstract">
      Recent advances in Neural Combinatorial Optimization (NCO) methods have significantly improved the capability of neural solvers to handle synthetic routing instances. Nonetheless, existing neural solvers typically struggle to generalize effectively from synthetic, uniformly-distributed training data to real-world VRP scenarios, including widely recognized benchmark instances from TSPLib and CVRPLib. To bridge this generalization gap, we present Evolutionary Realistic Instance Synthesis (EvoReal), which leverages an evolutionary module guided by large language models (LLMs) to generate synthetic instances characterized by diverse and realistic structural patterns. Specifically, the evolutionary module produces synthetic instances whose structural attributes statistically mimics those observed in authentic real-world instances. Subsequently, pre-trained NCO models are progressively refined, firstly aligning them with these structurally enriched synthetic distributions and then further adapting them through direct fine-tuning on actual benchmark instances. Extensive experimental evaluations demonstrate that EvoReal markedly improves the generalization capabilities of state-of-the-art neural solvers, yielding a notable reduced performance gap compared to the optimal solutions on the TSPLib (1.05%) and CVRPLib (2.71%) benchmarks across a broad spectrum of problem scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10222v1">Speech-Audio Compositional Attacks on Multimodal LLMs and Their Mitigation with SALMONN-Guard</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Recent progress in large language models (LLMs) has enabled understanding of both speech and non-speech audio, but exposing new safety risks emerging from complex audio inputs that are inadequately handled by current safeguards. We introduce SACRED-Bench (Speech-Audio Composition for RED-teaming) to evaluate the robustness of LLMs under complex audio-based attacks. Unlike existing perturbation-based methods that rely on noise optimization or white-box access, SACRED-Bench exploits speech-audio composition mechanisms. SACRED-Bench adopts three mechanisms: (a) speech overlap and multi-speaker dialogue, which embeds harmful prompts beneath or alongside benign speech; (b) speech-audio mixture, which imply unsafe intent via non-speech audio alongside benign speech or audio; and (c) diverse spoken instruction formats (open-ended QA, yes/no) that evade text-only filters. Experiments show that, even Gemini 2.5 Pro, the state-of-the-art proprietary LLM, still exhibits 66% attack success rate in SACRED-Bench test set, exposing vulnerabilities under cross-modal, speech-audio composition attacks. To bridge this gap, we propose SALMONN-Guard, a safeguard LLM that jointly inspects speech, audio, and text for safety judgments, reducing attack success down to 20%. Our results highlight the need for audio-aware defenses for the safety of multimodal LLMs. The benchmark and SALMONN-Guard checkpoints can be found at https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench. Warning: this paper includes examples that may be offensive or harmful.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10182v1">Beyond the Black Box: Demystifying Multi-Turn LLM Reasoning with VISTA</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Recent research has increasingly focused on the reasoning capabilities of Large Language Models (LLMs) in multi-turn interactions, as these scenarios more closely mirror real-world problem-solving. However, analyzing the intricate reasoning processes within these interactions presents a significant challenge due to complex contextual dependencies and a lack of specialized visualization tools, leading to a high cognitive load for researchers. To address this gap, we present VISTA, an web-based Visual Interactive System for Textual Analytics in multi-turn reasoning tasks. VISTA allows users to visualize the influence of context on model decisions and interactively modify conversation histories to conduct "what-if" analyses across different models. Furthermore, the platform can automatically parse a session and generate a reasoning dependency tree, offering a transparent view of the model's step-by-step logical path. By providing a unified and interactive framework, VISTA significantly reduces the complexity of analyzing reasoning chains, thereby facilitating a deeper understanding of the capabilities and limitations of current LLMs. The platform is open-source and supports easy integration of custom benchmarks and local models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.01939v2">Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Accepted to NeurIPS 2025. 25 pages, 17 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful approach to enhancing the reasoning capabilities of Large Language Models (LLMs), while its mechanisms are not yet well understood. In this work, we undertake a pioneering exploration of RLVR through the novel perspective of token entropy patterns, comprehensively analyzing how different tokens influence reasoning performance. By examining token entropy patterns in Chain-of-Thought (CoT) reasoning, we observe that only a small fraction of tokens exhibit high entropy, and these tokens act as critical forks that steer the model toward diverse reasoning pathways. Furthermore, studying how entropy patterns evolve during RLVR training reveals that RLVR largely adheres to the base model's entropy patterns, primarily adjusting the entropy of high-entropy tokens. These findings highlight the significance of high-entropy tokens (i.e., forking tokens) to RLVR. We ultimately improve RLVR by restricting policy gradient updates to forking tokens and uncover a finding even beyond the 80/20 rule: utilizing only 20% of the tokens while maintaining performance comparable to full-gradient updates on the Qwen3-8B base model and significantly surpassing full-gradient updates on the Qwen3-32B (+11.04 on AIME'25 and +7.71 on AIME'24) and Qwen3-14B (+4.79 on AIME'25 and +5.21 on AIME'24) base models, highlighting a strong scaling trend. In contrast, training exclusively on the 80% lowest-entropy tokens leads to a marked decline in performance. These findings indicate that the efficacy of RLVR primarily arises from optimizing the high-entropy tokens that decide reasoning directions. Collectively, our results highlight the potential to understand RLVR through a token-entropy perspective and optimize RLVR by leveraging high-entropy minority tokens to further improve LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08903v2">LLM-Guided Probabilistic Fusion for Label-Efficient Document Layout Analysis</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Document layout understanding remains data-intensive despite advances in semi-supervised learning. We present a framework that enhances semi-supervised detection by fusing visual predictions with structural priors from text-pretrained LLMs via principled probabilistic weighting. Given unlabeled documents, an OCR-LLM pipeline infers hierarchical regions which are combined with teacher detector outputs through inverse-variance fusion to generate refined pseudo-labels.Our method demonstrates consistent gains across model scales. With a lightweight SwiftFormer backbone (26M params), we achieve 88.2$\pm$0.3 AP using only 5\% labels on PubLayNet. When applied to document-pretrained LayoutLMv3 (133M params), our fusion framework reaches 89.7$\pm$0.4 AP, surpassing both LayoutLMv3 with standard semi-supervised learning (89.1$\pm$0.4 AP, p=0.02) and matching UDOP~\cite{udop} (89.8 AP) which requires 100M+ pages of multimodal pretraining. This demonstrates that LLM structural priors are complementary to both lightweight and pretrained architectures. Key findings include: (1) learned instance-adaptive gating improves over fixed weights by +0.9 AP with data-dependent PAC bounds correctly predicting convergence; (2) open-source LLMs enable privacy-preserving deployment with minimal loss (Llama-3-70B: 87.1 AP lightweight, 89.4 AP with LayoutLMv3); (3) LLMs provide targeted semantic disambiguation (18.7\% of cases, +3.8 AP gain) beyond simple text heuristics.Total system cost includes \$12 for GPT-4o-mini API or 17 GPU-hours for local Llama-3-70B per 50K pages, amortized across training runs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10075v1">Format Matters: The Robustness of Multimodal LLMs in Reviewing Evidence from Tables and Charts</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Accepted at AAAI 2026
    </div>
    <details class="paper-abstract">
      With the growing number of submitted scientific papers, there is an increasing demand for systems that can assist reviewers in evaluating research claims. Experimental results are a core component of scientific work, often presented in varying formats such as tables or charts. Understanding how robust current multimodal large language models (multimodal LLMs) are at verifying scientific claims across different evidence formats remains an important and underexplored challenge. In this paper, we design and conduct a series of experiments to assess the ability of multimodal LLMs to verify scientific claims using both tables and charts as evidence. To enable this evaluation, we adapt two existing datasets of scientific papers by incorporating annotations and structures necessary for a multimodal claim verification task. Using this adapted dataset, we evaluate 12 multimodal LLMs and find that current models perform better with table-based evidence while struggling with chart-based evidence. We further conduct human evaluations and observe that humans maintain strong performance across both formats, unlike the models. Our analysis also reveals that smaller multimodal LLMs (under 8B) show weak correlation in performance between table-based and chart-based tasks, indicating limited cross-modal generalization. These findings highlight a critical gap in current models' multimodal reasoning capabilities. We suggest that future multimodal LLMs should place greater emphasis on improving chart understanding to better support scientific claim verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10067v1">Enhancing the Medical Context-Awareness Ability of LLMs via Multifaceted Self-Refinement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 20 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great promise in the medical domain, achieving strong performance on several benchmarks. However, they continue to underperform in real-world medical scenarios, which often demand stronger context-awareness, i.e., the ability to recognize missing or critical details (e.g., user identity, medical history, risk factors) and provide safe, helpful, and contextually appropriate responses. To address this issue, we propose Multifaceted Self-Refinement (MuSeR), a data-driven approach that enhances LLMs' context-awareness along three key facets (decision-making, communication, and safety) through self-evaluation and refinement. Specifically, we first design a attribute-conditioned query generator that simulates diverse real-world user contexts by varying attributes such as role, geographic region, intent, and degree of information ambiguity. An LLM then responds to these queries, self-evaluates its answers along three key facets, and refines its responses to better align with the requirements of each facet. Finally, the queries and refined responses are used for supervised fine-tuning to reinforce the model's context-awareness ability. Evaluation results on the latest HealthBench dataset demonstrate that our method significantly improves LLM performance across multiple aspects, with particularly notable gains in the context-awareness axis. Furthermore, by incorporating knowledge distillation with the proposed method, the performance of a smaller backbone LLM (e.g., Qwen3-32B) surpasses its teacher model, achieving a new SOTA across all open-source LLMs on HealthBench (63.8%) and its hard subset (43.1%). Code and dataset will be released at https://muser-llm.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10049v1">Continuous Benchmark Generation for Evaluating Enterprise-scale LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 5 pages
    </div>
    <details class="paper-abstract">
      The rapid adoption of AI agents across domains has made systematic evaluation crucial for ensuring their usefulness and successful production deployment. Evaluation of AI agents typically involves using a fixed set of benchmarks and computing multiple evaluation metrics for the agent. While sufficient for simple coding tasks, these benchmarks fall short for enterprise-scale agents, where services and requirements evolve continuously and ground-truth examples are sparse. We propose a process of benchmark generation that helps evolve the benchmarks as the requirements change and perform robust evaluation of evolving AI agents. We instantiate this approach for a case study of service migration from one deployment platform to another at a large public enterprise. Our approach relies on semi-structured documents where developers express the high-level intent, and uses state-of-the-art LLMs to generate benchmarks from just a small number of such documents. Overall, this process results in a maintainable evaluation framework, enabling rapid feedback on agent performance and facilitating targeted improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10037v1">Beyond ReAct: A Planner-Centric Framework for Complex Tool-Augmented LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Existing tool-augmented large language models (LLMs) encounter significant challenges when processing complex queries. Current frameworks such as ReAct are prone to local optimization traps due to their reliance on incremental decision-making processes. To address these limitations, we propose a novel Planner-centric Plan-Execute paradigm that fundamentally resolves local optimization bottlenecks through architectural innovation. Central to our approach is a novel Planner model that performs global Directed Acyclic Graph (DAG) planning for complex queries, enabling optimized execution beyond conventional tool coordination. We also introduce ComplexTool-Plan, a large-scale benchmark dataset featuring complex queries that demand sophisticated multi-tool composition and coordination capabilities. Additionally, we develop a two-stage training methodology that integrates Supervised Fine-Tuning (SFT) with Group Relative Policy Optimization (GRPO), systematically enhancing the Planner's tool selection accuracy and global planning awareness through structured DAG-based planning. When integrated with a capable executor, our framework achieves state-of-the-art performance on the StableToolBench benchmark for complex user queries, demonstrating superior end-to-end execution capabilities and robust handling of intricate multi-tool workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.19319v2">FHIR-AgentBench: Benchmarking LLM Agents for Realistic Interoperable EHR Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 ML4H 2025 Proceedings
    </div>
    <details class="paper-abstract">
      The recent shift toward the Health Level Seven Fast Healthcare Interoperability Resources (HL7 FHIR) standard opens a new frontier for clinical AI, demanding LLM agents to navigate complex, resource-based data models instead of conventional structured health data. However, existing benchmarks have lagged behind this transition, lacking the realism needed to evaluate recent LLMs on interoperable clinical data. To bridge this gap, we introduce FHIR-AgentBench, a benchmark that grounds 2,931 real-world clinical questions in the HL7 FHIR standard. Using this benchmark, we systematically evaluate agentic frameworks, comparing different data retrieval strategies (direct FHIR API calls vs. specialized tools), interaction patterns (single-turn vs. multi-turn), and reasoning strategies (natural language vs. code generation). Our experiments highlight the practical challenges of retrieving data from intricate FHIR resources and the difficulty of reasoning over them, both of which critically affect question answering performance. We publicly release the FHIR-AgentBench dataset and evaluation suite (https://github.com/glee4810/FHIR-AgentBench) to promote reproducible research and the development of robust, reliable LLM agents for clinical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10007v1">AssertMiner: Module-Level Spec Generation and Assertion Mining using Static Analysis Guided LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 6 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Assertion-based verification (ABV) is a key approach to checking whether a logic design complies with its architectural specifications. Existing assertion generation methods based on design specifications typically produce only top-level assertions, overlooking verification needs on the implementation details in the modules at the micro-architectural level, where design errors occur more frequently. To address this limitation, we present AssertMiner, a module-level assertion generation framework that leverages static information generated from abstract syntax tree (AST) to assist LLMs in mining assertions. Specifically, it performs AST-based structural extraction to derive the module call graph, I/O table, and dataflow graph, guiding the LLM to generate module-level specifications and mine module-level assertions. Our evaluation demonstrates that AssertMiner outperforms existing methods such as AssertLLM and Spec2Assertion in generating high-quality assertions for modules. When integrated with these methods, AssertMiner can enhance the structural coverage and significantly improve the error detection capability, enabling a more comprehensive and efficient verification process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09062v2">Pricing Online LLM Services with Data-Calibrated Stackelberg Routing Game</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 Extended version
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) has established LLM routing as a standard service delivery mechanism, where users select models based on cost, Quality of Service (QoS), among other things. However, optimal pricing in LLM routing platforms requires precise modeling for dynamic service markets, and solving this problem in real time at scale is computationally intractable. In this paper, we propose \PriLLM, a novel practical and scalable solution for real-time dynamic pricing in competitive LLM routing. \PriLLM models the service market as a Stackelberg game, where providers set prices and users select services based on multiple criteria. To capture real-world market dynamics, we incorporate both objective factors (\eg~cost, QoS) and subjective user preferences into the model. For scalability, we employ a deep aggregation network to learn provider abstraction that preserve user-side equilibrium behavior across pricing strategies. Moreover, \PriLLM offers interpretability by explaining its pricing decisions. Empirical evaluation on real-world data shows that \PriLLM achieves over 95\% of the optimal profit while only requiring less than 5\% of the optimal solution's computation time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09998v1">DemoTuner: Efficient DBMS Knobs Tuning via LLM-Assisted Demonstration Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 14 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The performance of modern DBMSs such as MySQL and PostgreSQL heavily depends on the configuration of performance-critical knobs. Manual tuning these knobs is laborious and inefficient due to the complex and high-dimensional nature of the configuration space. Among the automated tuning methods, reinforcement learning (RL)-based methods have recently sought to improve the DBMS knobs tuning process from several different perspectives. However, they still encounter challenges with slow convergence speed during offline training. In this paper, we mainly focus on how to leverage the valuable tuning hints contained in various textual documents such as DBMS manuals and web forums to improve the offline training of RL-based methods. To this end, we propose an efficient DBMS knobs tuning framework named DemoTuner via a novel LLM-assisted demonstration reinforcement learning method. Specifically, to comprehensively and accurately mine tuning hints from documents, we design a structured chain of thought prompt to employ LLMs to conduct a condition-aware tuning hints extraction task. To effectively integrate the mined tuning hints into RL agent training, we propose a hint-aware demonstration reinforcement learning algorithm HA-DDPGfD in DemoTuner. As far as we know, DemoTuner is the first work to introduce the demonstration reinforcement learning algorithm for DBMS knobs tuning. Experimental evaluations conducted on MySQL and PostgreSQL across various workloads demonstrate the significant advantages of DemoTuner in both performance improvement and online tuning cost reduction over three representative baselines including DB-BERT, GPTuner and CDBTune. Additionally, DemoTuner also exhibits superior adaptability to application scenarios with unknown workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09969v1">Owlgorithm: Supporting Self-Regulated Learning in Competitive Programming through LLM-Driven Reflection</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 7 pages, 1 figure, to be published in SIGCSE '26
    </div>
    <details class="paper-abstract">
      We present Owlgorithm, an educational platform that supports Self-Regulated Learning (SRL) in competitive programming (CP) through AI-generated reflective questions. Leveraging GPT-4o, Owlgorithm produces context-aware, metacognitive prompts tailored to individual student submissions. Integrated into a second- and third-year CP course, the system-provided reflective prompts adapted to student outcomes: guiding deeper conceptual insight for correct solutions and structured debugging for partial or failed ones. Our exploratory assessment of student ratings and TA feedback revealed both promising benefits and notable limitations. While many found the generated questions useful for reflection and debugging, concerns were raised about feedback accuracy and classroom usability. These results suggest advantages of LLM-supported reflection for novice programmers, though refinements are needed to ensure reliability and pedagogical value for advanced learners. From our experience, several key insights emerged: GenAI can effectively support structured reflection, but careful prompt design, dynamic adaptation, and usability improvements are critical to realizing their potential in education. We offer specific recommendations for educators using similar tools and outline next steps to enhance Owlgorithm's educational impact. The underlying framework may also generalize to other reflective learning contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09964v1">EnvTrace: Simulation-Based Semantic Evaluation of LLM Code via Execution Trace Alignment -- Demonstrated at Synchrotron Beamlines</a></div>
    <div class="paper-meta">
      📅 2025-11-13
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) for instrument control requires methods that go beyond standard, stateless algorithmic benchmarks, since the behavior of physical systems cannot be fully captured by unit tests alone. Here we introduce EnvTrace, a simulation-based method that evaluates execution traces to assess semantic code equivalence. EnvTrace is demonstrated with a beamline control-logic digital twin to facilitate the evaluation of instrument control code, with the digital twin itself also enabling the pre-execution validation of live experiments. Over 30 LLMs were evaluated using trace alignment to generate a multi-faceted score for functional correctness across key behavioral dimensions, showing that many top-tier models can approach human-level performance in rapid control-code generation. This is a first step toward a broader vision where LLMs and digital twins work symbiotically: LLMs providing intuitive control and agentic orchestration, and digital twins offering safe and high-fidelity environments, paving the way towards autonomous embodied AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.20194v3">DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 NeurIPS 2025. The code is available on Github (see hyperlink in the paper)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deliver strong performance but are difficult to deploy due to high memory and compute costs. While pruning reduces these demands, most methods ignore activation sparsity observed at runtime. We reinterpret activation sparsity as dynamic structured weight sparsity and propose DuoGPT, a unified framework that constructs dual-sparse (spMspV) workloads by combining unstructured weight pruning with activation sparsity. To preserve accuracy, we extend the Optimal Brain Compression (OBC) framework with activation-aware calibration and introduce output residuals from the dense model as correction terms. We further optimize the solution for efficient GPU execution, enabling scalability to billion-parameter LLMs. Evaluations on LLaMA-2 and LLaMA-3 show that DuoGPT outperforms state-of-the-art structured pruning methods by up to 9.17% accuracy at an iso-speedup of 1.39$\times$ compared to the baseline dense model. Code is available at Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09865v1">In-Token Rationality Optimization: Towards Accurate and Concise LLM Reasoning via Self-Feedback</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 AAAI 2026 Oral
    </div>
    <details class="paper-abstract">
      Training Large Language Models (LLMs) for chain-of-thought reasoning presents a significant challenge: supervised fine-tuning on a single "golden" rationale hurts generalization as it penalizes equally valid alternatives, whereas reinforcement learning with verifiable rewards struggles with credit assignment and prohibitive computational cost. To tackle these limitations, we introduce InTRO (In-Token Rationality Optimization), a new framework that enables both token-level exploration and self-feedback for accurate and concise reasoning. Instead of directly optimizing an intractable objective over all valid reasoning paths, InTRO leverages correction factors-token-wise importance weights estimated by the information discrepancy between the generative policy and its answer-conditioned counterpart, for informative next token selection. This approach allows the model to perform token-level exploration and receive self-generated feedback within a single forward pass, ultimately encouraging accurate and concise rationales. Across six math-reasoning benchmarks, InTRO consistently outperforms other baselines, raising solution accuracy by up to 20% relative to the base model. Its chains of thought are also notably more concise, exhibiting reduced verbosity. Beyond this, InTRO enables cross-domain transfer, successfully adapting to out-of-domain reasoning tasks that extend beyond the realm of mathematics, demonstrating robust generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09855v1">Unlearning Imperative: Securing Trustworthy and Responsible LLMs through Engineered Forgetting</a></div>
    <div class="paper-meta">
      📅 2025-11-13
      | 💬 14 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      The growing use of large language models in sensitive domains has exposed a critical weakness: the inability to ensure that private information can be permanently forgotten. Yet these systems still lack reliable mechanisms to guarantee that sensitive information can be permanently removed once it has been used. Retraining from the beginning is prohibitively costly, and existing unlearning methods remain fragmented, difficult to verify, and often vulnerable to recovery. This paper surveys recent research on machine unlearning for LLMs and considers how far current approaches can address these challenges. We review methods for evaluating whether forgetting has occurred, the resilience of unlearned models against adversarial attacks, and mechanisms that can support user trust when model complexity or proprietary limits restrict transparency. Technical solutions such as differential privacy, homomorphic encryption, federated learning, and ephemeral memory are examined alongside institutional safeguards including auditing practices and regulatory frameworks. The review finds steady progress, but robust and verifiable unlearning is still unresolved. Efficient techniques that avoid costly retraining, stronger defenses against adversarial recovery, and governance structures that reinforce accountability are needed if LLMs are to be deployed safely in sensitive applications. By integrating technical and organizational perspectives, this study outlines a pathway toward AI systems that can be required to forget, while maintaining both privacy and public trust.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.11122v3">Navigating the Alpha Jungle: An LLM-Powered MCTS Framework for Formulaic Factor Mining</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Alpha factor mining is pivotal in quantitative investment for identifying predictive signals from complex financial data. While traditional formulaic alpha mining relies on human expertise, contemporary automated methods, such as those based on genetic programming or reinforcement learning, often struggle with search inefficiency or yield alpha factors that are difficult to interpret. This paper introduces a novel framework that integrates Large Language Models (LLMs) with Monte Carlo Tree Search (MCTS) to overcome these limitations. Our framework leverages the LLM's instruction-following and reasoning capability to iteratively generate and refine symbolic alpha formulas within an MCTS-driven exploration. A key innovation is the guidance of MCTS exploration by rich, quantitative feedback from financial backtesting of each candidate factor, enabling efficient navigation of the vast search space. Furthermore, a frequent subtree avoidance mechanism is introduced to enhance search diversity and prevent formulaic homogenization, further improving performance. Experimental results on real-world stock market data demonstrate that our LLM-based framework outperforms existing methods by mining alphas with superior predictive accuracy and trading performance. The resulting formulas are also more amenable to human interpretation, establishing a more effective and efficient paradigm for formulaic alpha mining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09030v1">Solving a Million-Step LLM Task with Zero Errors</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 Main paper: 14 pages, 29 pages with references and appendix
    </div>
    <details class="paper-abstract">
      LLMs have achieved remarkable breakthroughs in reasoning, insights, and tool use, but chaining these abilities into extended processes at the scale of those routinely executed by humans, organizations, and societies has remained out of reach. The models have a persistent error rate that prevents scale-up: for instance, recent experiments in the Towers of Hanoi benchmark domain showed that the process inevitably becomes derailed after at most a few hundred steps. Thus, although LLM research is often still benchmarked on tasks with relatively few dependent logical steps, there is increasing attention on the ability (or inability) of LLMs to perform long range tasks. This paper describes MAKER, the first system that successfully solves a task with over one million LLM steps with zero errors, and, in principle, scales far beyond this level. The approach relies on an extreme decomposition of a task into subtasks, each of which can be tackled by focused microagents. The high level of modularity resulting from the decomposition allows error correction to be applied at each step through an efficient multi-agent voting scheme. This combination of extreme decomposition and error correction makes scaling possible. Thus, the results suggest that instead of relying on continual improvement of current LLMs, massively decomposed agentic processes (MDAPs) may provide a way to efficiently solve problems at the level of organizations and societies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09025v1">FLAD: Federated Learning for LLM-based Autonomous Driving in Vehicle-Edge-Cloud Networks</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have impressive data fusion and reasoning capabilities for autonomous driving (AD). However, training LLMs for AD faces significant challenges including high computation transmission costs, and privacy concerns associated with sensitive driving data. Federated Learning (FL) is promising for enabling autonomous vehicles (AVs) to collaboratively train models without sharing raw data. We present Federated LLM-based Autonomous Driving (FLAD), an FL framework that leverages distributed multimodal sensory data across AVs in heterogeneous environment. FLAD has three key innovations: (1) a cloud-edge-vehicle collaborative architecture that reduces communication delay and preserving data privacy; (2) an intelligent parallelized collaborative training with a communication scheduling mechanism that optimizes training efficiency, leveraging end-devices otherwise having insufficient resources for model training; and (3) a knowledge distillation method that personalizes LLM according to heterogeneous edge data. In addition, we prototype FLAD in a testbed with NVIDIA Jetsons, overcoming practical implementation challenges including CPU/GPU memory sharing in resource-constrained devices, dynamic model partitions, and fault-tolerant training.Extensive experimental evaluation demonstrates that FLAD achieves superior end-to-end AD performance while efficiently utilizing distributed vehicular resources, opening up new possibilities for future collaborative AD model training and knowledge sharing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.15311v2">Trajectory Bellman Residual Minimization: A Simple Value-Based Method for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Policy-based methods currently dominate reinforcement learning (RL) pipelines for large language model (LLM) reasoning, leaving value-based approaches largely unexplored. We revisit the classical paradigm of Bellman Residual Minimization and introduce Trajectory Bellman Residual Minimization (TBRM), an algorithm that naturally adapts this idea to LLMs, yielding a simple yet effective off-policy algorithm that optimizes a single trajectory-level Bellman objective using the model's own logits as $Q$-values. TBRM removes the need for critics, importance-sampling ratios, or clipping, and operates with only one rollout per prompt. We prove convergence to the near-optimal KL-regularized policy from arbitrary off-policy data via an improved change-of-trajectory-measure analysis. Experiments on standard mathematical-reasoning benchmarks show that TBRM consistently outperforms policy-based baselines, like PPO and GRPO, with comparable or lower computational and memory overhead. Our results indicate that value-based RL might be a principled and efficient alternative for enhancing reasoning capabilities in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06029v2">Lethe: Layer- and Time-Adaptive KV Cache Pruning for Reasoning-Intensive LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 aaai26 camera-ready version, 12 pages
    </div>
    <details class="paper-abstract">
      Generative reasoning with large language models (LLMs) often involves long decoding sequences, leading to substantial memory and latency overheads from accumulating key-value (KV) caches. While existing KV compression methods primarily focus on reducing prefill memory from long input sequences, they fall short in addressing the dynamic and layer-sensitive nature of long-form generation, which is central to reasoning tasks. We propose Lethe, a dynamic KV cache management framework that introduces adaptivity along both the spatial and temporal dimensions of decoding. Along the spatial dimension, Lethe performs layerwise sparsity-aware allocation, assigning token pruning budgets to each transformer layer based on estimated attention redundancy. Along the temporal dimension, Lethe conducts multi-round token pruning during generation, driven by a Recency-Aware Selective Retention} (RASR) mechanism. RASR extends traditional recency-based heuristics by also considering token relevance derived from evolving attention patterns, enabling informed decisions about which tokens to retain or evict. Empirical results demonstrate that Lethe achieves a favorable balance between efficiency and generation quality across diverse models and tasks, increases throughput by up to 2.56x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07419v2">Routing Manifold Alignment Improves Generalization of Mixture-of-Experts LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Sparse Mixture-of-Experts (MoE) have been widely adopted in recent large language models since it can efficiently scale up the model capability without increasing the inference cost. However, evaluations on broad downstream tasks reveal a consistent suboptimality of the routers in existing MoE LLMs, which results in a severe performance gap (e.g., 10-20% in accuracy) to the optimal routing. In this paper, we show that aligning the manifold of routing weights with that of task embedding can effectively reduce the gap and improve MoE LLMs' generalization performance. Our method, "Routing Manifold Alignment (RoMA)", introduces an additional manifold regularization term in the post-training objective and only requires lightweight finetuning of routers (with other parameters frozen). Specifically, the regularization encourages the routing weights of each sample to be close to those of its successful neighbors (whose routing weights lead to correct answers) in a task embedding space. Consequently, samples targeting similar tasks will share similar expert choices across layers. Building such bindings between tasks and experts over different samples is essential to achieve better generalization. Moreover, RoMA demonstrates the advantage of unifying the task understanding (by embedding models) with solution generation (by MoE LLMs). In experiments, we finetune routers in OLMoE, DeepSeekMoE, and Qwen3-MoE using RoMA. Evaluations on diverse benchmarks and extensive comparisons with baselines show the substantial improvement brought by RoMA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08949v1">EVADE: LLM-Based Explanation Generation and Validation for Error Detection in NLI</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      High-quality datasets are critical for training and evaluating reliable NLP models. In tasks like natural language inference (NLI), human label variation (HLV) arises when multiple labels are valid for the same instance, making it difficult to separate annotation errors from plausible variation. An earlier framework VARIERR (Weber-Genzel et al., 2024) asks multiple annotators to explain their label decisions in the first round and flag errors via validity judgments in the second round. However, conducting two rounds of manual annotation is costly and may limit the coverage of plausible labels or explanations. Our study proposes a new framework, EVADE, for generating and validating explanations to detect errors using large language models (LLMs). We perform a comprehensive analysis comparing human- and LLM-detected errors for NLI across distribution comparison, validation overlap, and impact on model fine-tuning. Our experiments demonstrate that LLM validation refines generated explanation distributions to more closely align with human annotations, and that removing LLM-detected errors from training data yields improvements in fine-tuning performance than removing errors identified by human annotators. This highlights the potential to scale error detection, reducing human effort while improving dataset quality under label variation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08947v1">AlphaCast: A Human Wisdom-LLM Intelligence Co-Reasoning Framework for Interactive Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Time series forecasting plays a critical role in high-stakes domains such as energy, healthcare, and climate. Although recent advances have improved accuracy, most approaches still treat forecasting as a static one-time mapping task, lacking the interaction, reasoning, and adaptability of human experts. This gap limits their usefulness in complex real-world environments. To address this, we propose AlphaCast, a human wisdom-large language model (LLM) intelligence co-reasoning framework that redefines forecasting as an interactive process. The key idea is to enable step-by-step collaboration between human wisdom and LLM intelligence to jointly prepare, generate, and verify forecasts. The framework consists of two stages: (1) automated prediction preparation, where AlphaCast builds a multi-source cognitive foundation comprising a feature set that captures key statistics and time patterns, a domain knowledge base distilled from corpora and historical series, a contextual repository that stores rich information for each time window, and a case base that retrieves optimal strategies via pattern clustering and matching; and (2) generative reasoning and reflective optimization, where AlphaCast integrates statistical temporal features, prior knowledge, contextual information, and forecasting strategies, triggering a meta-reasoning loop for continuous self-correction and strategy refinement. Extensive experiments on short- and long-term datasets show that AlphaCast consistently outperforms state-of-the-art baselines in predictive accuracy. Code is available at this repository: https://github.com/SkyeGT/AlphaCast_Official .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.20157v2">rLLM: Relational Table Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      We introduce rLLM (relationLLM), a PyTorch library designed for Relational Table Learning (RTL) with Large Language Models (LLMs). The core idea is to decompose state-of-the-art Graph Neural Networks, LLMs, and Table Neural Networks into standardized modules, to enable the fast construction of novel RTL-type models in a simple "combine, align, and co-train" manner. To illustrate the usage of rLLM, we introduce a simple RTL method named \textbf{BRIDGE}. Additionally, we present three novel relational tabular datasets (TML1M, TLF2K, and TACM12K) by enhancing classic datasets. We hope rLLM can serve as a useful and easy-to-use development framework for RTL-related tasks. Our code is available at: https://github.com/rllm-project/rllm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.00202v2">RouterArena: An Open Platform for Comprehensive Comparison of LLM Routers</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 16 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Today's LLM ecosystem comprises a wide spectrum of models that differ in size, capability, and cost. No single model is optimal for all scenarios; hence, LLM routers have become essential for selecting the most appropriate model under varying circumstances. However, the rapid emergence of various routers makes choosing the right one increasingly challenging. To address this problem, we need a comprehensive router comparison and a standardized leaderboard, similar to those available for models. In this work, we introduce RouterArena, the first open platform enabling comprehensive comparison of LLM routers. RouterArena has (1) a principally constructed dataset with broad knowledge domain coverage, (2) distinguishable difficulty levels for each domain, (3) an extensive list of evaluation metrics, and (4) an automated framework for leaderboard updates. Leveraging our framework, we have produced the initial leaderboard with detailed metrics comparison as shown in Figure 1. Our framework for evaluating new routers is on https://github.com/RouteWorks/RouterArena
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08916v1">HalluClean: A Unified Framework to Combat Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved impressive performance across a wide range of natural language processing tasks, yet they often produce hallucinated content that undermines factual reliability. To address this challenge, we introduce HalluClean, a lightweight and task-agnostic framework for detecting and correcting hallucinations in LLM-generated text. HalluClean adopts a reasoning-enhanced paradigm, explicitly decomposing the process into planning, execution, and revision stages to identify and refine unsupported claims. It employs minimal task-routing prompts to enable zero-shot generalization across diverse domains, without relying on external knowledge sources or supervised detectors. We conduct extensive evaluations on five representative tasks-question answering, dialogue, summarization, math word problems, and contradiction detection. Experimental results show that HalluClean significantly improves factual consistency and outperforms competitive baselines, demonstrating its potential to enhance the trustworthiness of LLM outputs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08905v1">iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Given the high cost of large language model (LLM) training from scratch, safeguarding LLM intellectual property (IP) has become increasingly crucial. As the standard paradigm for IP ownership verification, LLM fingerprinting thus plays a vital role in addressing this challenge. Existing LLM fingerprinting methods verify ownership by extracting or injecting model-specific features. However, they overlook potential attacks during the verification process, leaving them ineffective when the model thief fully controls the LLM's inference process. In such settings, attackers may share prompt-response pairs to enable fingerprint unlearning or manipulate outputs to evade exact-match verification. We propose iSeal, the first fingerprinting method designed for reliable verification when the model thief controls the suspected LLM in an end-to-end manner. It injects unique features into both the model and an external module, reinforced by an error-correction mechanism and a similarity-based verification strategy. These components are resistant to verification-time attacks, including collusion-based fingerprint unlearning and response manipulation, backed by both theoretical analysis and empirical results. iSeal achieves 100 percent Fingerprint Success Rate (FSR) on 12 LLMs against more than 10 attacks, while baselines fail under unlearning and response manipulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.01908v3">UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents equipped with external tools have become increasingly powerful for complex tasks such as web shopping, automated email replies, and financial trading. However, these advancements amplify the risks of adversarial attacks, especially when agents can access sensitive external functionalities. Nevertheless, manipulating LLM agents into performing targeted malicious actions or invoking specific tools remains challenging, as these agents extensively reason or plan before executing final actions. In this work, we present UDora, a unified red teaming framework designed for LLM agents that dynamically hijacks the agent's reasoning processes to compel malicious behavior. Specifically, UDora first generates the model's reasoning trace for the given task, then automatically identifies optimal points within this trace to insert targeted perturbations. The resulting perturbed reasoning is then used as a surrogate response for optimization. By iteratively applying this process, the LLM agent will then be induced to undertake designated malicious actions or to invoke specific malicious tools. Our approach demonstrates superior effectiveness compared to existing methods across three LLM agent datasets. The code is available at https://github.com/AI-secure/UDora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06838v2">P3-LLM: An Integrated NPU-PIM Accelerator for LLM Inference Using Hybrid Numerical Formats</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      The substantial memory bandwidth and computational demands of large language models (LLMs) present critical challenges for efficient inference. To tackle this, the literature has explored heterogeneous systems that combine neural processing units (NPUs) with DRAM-based processing-in-memory (PIM) for LLM acceleration. However, existing high-precision (e.g., FP16) PIM compute units incur significant area and power overhead in DRAM technology, limiting the effective computation throughput. In this paper, we introduce P3-LLM, a novel NPU-PIM integrated accelerator for LLM inference using hybrid numerical formats. Our approach is threefold: First, we propose a flexible mixed-precision quantization scheme, which leverages hybrid numerical formats to quantize different LLM operands with high compression efficiency and minimal accuracy loss. Second, we architect an efficient PIM accelerator for P3-LLM, featuring enhanced compute units to support hybrid numerical formats. Our careful choice of numerical formats allows to co-design low-precision PIM compute units that significantly boost the computation throughput under iso-area constraints. Third, we optimize the low-precision dataflow of different LLM modules by applying operator fusion to minimize the overhead of runtime dequantization. Evaluation on a diverse set of representative LLMs and tasks demonstrates that P3-LLM achieves state-of-the-art accuracy in terms of both KV-cache quantization and weight-activation quantization. Combining the proposed quantization scheme with PIM architecture co-design, P3-LLM yields an average of $4.9\times$, $2.0\times$, and $3.4\times$ speedups over the state-of-the-art LLM accelerators HBM-PIM, Ecco, and Pimba, respectively. Our quantization code is available at https://github.com/yc2367/P3-LLM.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.04387v3">FedP$^2$EFT: Federated Learning to Personalize PEFT for Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 Accepted at the 40th Annual AAAI Conference on Artificial Intelligence (AAAI-26)
    </div>
    <details class="paper-abstract">
      Federated learning (FL) has enabled the training of multilingual large language models (LLMs) on diverse and decentralized multilingual data, especially on low-resource languages. To improve client-specific performance, personalization via the use of parameter-efficient fine-tuning (PEFT) modules such as LoRA is common. This involves a personalization strategy (PS), such as the design of the PEFT adapter structures (e.g., in which layers to add LoRAs and what ranks) and choice of hyperparameters (e.g., learning rates) for fine-tuning. Instead of manual PS configuration, we propose FedP$^2$EFT, a federated learning-to-personalize method for multilingual LLMs in cross-device FL settings. Unlike most existing PEFT structure selection methods, which are prone to overfitting low-data regimes, FedP$^2$EFT collaboratively learns the optimal personalized PEFT structure for each client via Bayesian sparse rank selection. Evaluations on both simulated and real-world multilingual FL benchmarks demonstrate that FedP$^2$EFT largely outperforms existing personalized fine-tuning methods, while complementing other existing FL methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.08060v1">From LLMs to Agents: A Comparative Evaluation of LLMs and LLM-based Agents in Security Patch Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      The widespread adoption of open-source software (OSS) has accelerated software innovation but also increased security risks due to the rapid propagation of vulnerabilities and silent patch releases. In recent years, large language models (LLMs) and LLM-based agents have demonstrated remarkable capabilities in various software engineering (SE) tasks, enabling them to effectively address software security challenges such as vulnerability detection. However, systematic evaluation of the capabilities of LLMs and LLM-based agents in security patch detection remains limited. To bridge this gap, we conduct a comprehensive evaluation of the performance of LLMs and LLM-based agents for security patch detection. Specifically, we investigate three methods: Plain LLM (a single LLM with a system prompt), Data-Aug LLM (data augmentation based on the Plain LLM), and the ReAct Agent (leveraging the thought-action-observation mechanism). We also evaluate the performance of both commercial and open-source LLMs under these methods and compare these results with those of existing baselines. Furthermore, we analyze the detection performance of these methods across various vulnerability types, and examine the impact of different prompting strategies and context window sizes on the results. Our findings reveal that the Data-Aug LLM achieves the best overall performance, whereas the ReAct Agent demonstrates the lowest false positive rate (FPR). Although baseline methods exhibit strong accuracy, their false positive rates are significantly higher. In contrast, our evaluated methods achieve comparable accuracy while substantially reducing the FPR. These findings provide valuable insights into the practical applications of LLMs and LLM-based agents in security patch detection, highlighting their advantage in maintaining robust performance while minimizing false positive rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.07127v2">REACT-LLM: A Benchmark for Evaluating LLM Integration with Causal Features in Clinical Prognostic Tasks</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and causal learning each hold strong potential for clinical decision making (CDM). However, their synergy remains poorly understood, largely due to the lack of systematic benchmarks evaluating their integration in clinical risk prediction. In real-world healthcare, identifying features with causal influence on outcomes is crucial for actionable and trustworthy predictions. While recent work highlights LLMs' emerging causal reasoning abilities, there lacks comprehensive benchmarks to assess their causal learning and performance informed by causal features in clinical risk prediction. To address this, we introduce REACT-LLM, a benchmark designed to evaluate whether combining LLMs with causal features can enhance clinical prognostic performance and potentially outperform traditional machine learning (ML) methods. Unlike existing LLM-clinical benchmarks that often focus on a limited set of outcomes, REACT-LLM evaluates 7 clinical outcomes across 2 real-world datasets, comparing 15 prominent LLMs, 6 traditional ML models, and 3 causal discovery (CD) algorithms. Our findings indicate that while LLMs perform reasonably in clinical prognostics, they have not yet outperformed traditional ML models. Integrating causal features derived from CD algorithms into LLMs offers limited performance gains, primarily due to the strict assumptions of many CD methods, which are often violated in complex clinical data. While the direct integration yields limited improvement, our benchmark reveals a more promising synergy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2511.07568v1">Procedural Knowledge Improves Agentic LLM Workflows</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle when performing agentic tasks without substantial tool support, prom-pt engineering, or fine tuning. Despite research showing that domain-dependent, procedural knowledge can dramatically increase planning efficiency, little work evaluates its potential for improving LLM performance on agentic tasks that may require implicit planning. We formalize, implement, and evaluate an agentic LLM workflow that leverages procedural knowledge in the form of a hierarchical task network (HTN). Empirical results of our implementation show that hand-coded HTNs can dramatically improve LLM performance on agentic tasks, and using HTNs can boost a 20b or 70b parameter LLM to outperform a much larger 120b parameter LLM baseline. Furthermore, LLM-created HTNs improve overall performance, though less so. The results suggest that leveraging expertise--from humans, documents, or LLMs--to curate procedural knowledge will become another important tool for improving LLM workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09820v1">From Street to Orbit: Training-Free Cross-View Retrieval via Location Semantics and LLM Guidance</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 Accepted to WACV 2026, 10pages, 4 figures
    </div>
    <details class="paper-abstract">
      Cross-view image retrieval, particularly street-to-satellite matching, is a critical task for applications such as autonomous navigation, urban planning, and localization in GPS-denied environments. However, existing approaches often require supervised training on curated datasets and rely on panoramic or UAV-based images, which limits real-world deployment. In this paper, we present a simple yet effective cross-view image retrieval framework that leverages a pretrained vision encoder and a large language model (LLM), requiring no additional training. Given a monocular street-view image, our method extracts geographic cues through web-based image search and LLM-based location inference, generates a satellite query via geocoding API, and retrieves matching tiles using a pretrained vision encoder (e.g., DINOv2) with PCA-based whitening feature refinement. Despite using no ground-truth supervision or finetuning, our proposed method outperforms prior learning-based approaches on the benchmark dataset under zero-shot settings. Moreover, our pipeline enables automatic construction of semantically aligned street-to-satellite datasets, which is offering a scalable and cost-efficient alternative to manual annotation. All source codes will be made publicly available at https://jeonghomin.github.io/street2orbit.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.18638v3">Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 14 pages, 5 figures; published in EMNLP 2025 ; Code at: https://github.com/dsbuddy/GAP-LLM-Safety
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP's superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning. Our implementation is available at https://github.com/dsbuddy/GAP-LLM-Safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09785v1">AI Annotation Orchestration: Evaluating LLM verifiers to Improve the Quality of LLM Annotations in Learning Analytics</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to annotate learning interactions, yet concerns about reliability limit their utility. We test whether verification-oriented orchestration-prompting models to check their own labels (self-verification) or audit one another (cross-verification)-improves qualitative coding of tutoring discourse. Using transcripts from 30 one-to-one math sessions, we compare three production LLMs (GPT, Claude, Gemini) under three conditions: unverified annotation, self-verification, and cross-verification across all orchestration configurations. Outputs are benchmarked against a blinded, disagreement-focused human adjudication using Cohen's kappa. Overall, orchestration yields a 58 percent improvement in kappa. Self-verification nearly doubles agreement relative to unverified baselines, with the largest gains for challenging tutor moves. Cross-verification achieves a 37 percent improvement on average, with pair- and construct-dependent effects: some verifier-annotator pairs exceed self-verification, while others reduce alignment, reflecting differences in verifier strictness. We contribute: (1) a flexible orchestration framework instantiating control, self-, and cross-verification; (2) an empirical comparison across frontier LLMs on authentic tutoring data with blinded human "gold" labels; and (3) a concise notation, verifier(annotator) (e.g., Gemini(GPT) or Claude(Claude)), to standardize reporting and make directional effects explicit for replication. Results position verification as a principled design lever for reliable, scalable LLM-assisted annotation in Learning Analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.17741v2">Chameleon: Adaptive Caching and Scheduling for Many-Adapter LLM Inference Environments</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 Accepted at MICRO '25
    </div>
    <details class="paper-abstract">
      The widespread adoption of LLMs has driven an exponential rise in their deployment, imposing substantial demands on inference clusters. These clusters must handle numerous concurrent queries for different LLM downstream tasks. To handle multi-task settings with vast LLM parameter counts, methods like Low-Rank Adaptation (LoRA) enable task-specific fine-tuning while sharing most of the base LLM model across tasks. Hence, they allow concurrent task serving with minimal memory requirements. However, existing LLM serving systems face inefficiencies: they overlook workload heterogeneity, impose high link bandwidth from frequent adapter loading, and suffer from head-of-line blocking in their schedulers. To address these challenges, we present Chameleon, a novel LLM serving system optimized for many adapter environments, that relies on two core ideas: adapter caching and adapter-aware scheduling. First, Chameleon caches popular adapters in GPU memory, minimizing the adapter loading times. Importantly, it uses the otherwise idle GPU memory, avoiding extra memory costs. Second, Chameleon uses a non-preemptive multi-queue scheduling to efficiently account for workload heterogeneity. In this way, Chameleon simultaneously prevents head of line blocking and starvation. We implement Chameleon on top of a state-of-the-art LLM serving platform and evaluate it with real-world production traces and open-source LLMs. Under high loads, Chameleon reduces P99 and P50 TTFT latency by 80.7% and 48.1%, respectively, while improving throughput by 1.5x compared to state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.14617v3">SageServe: Optimizing LLM Serving on Cloud Data Centers with Forecast Aware Auto-Scaling</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 25 pages, 16 figures, 2 tables. The workload traces, our simulator harness and the SageServe scheduler are available at https://github.com/shashwatj07/SageServe
    </div>
    <details class="paper-abstract">
      Global cloud service providers handle inference workloads for Large Language Models (LLMs) that span latency-sensitive (e.g., chatbots) and insensitive (e.g., report writing) tasks, resulting in diverse and often conflicting Service Level Agreement (SLA) requirements. Managing such mixed workloads is challenging due to the complexity of the inference serving stack, which encompasses multiple models, GPU hardware, and global data centers. Existing solutions often silo such fast and slow tasks onto separate GPU resource pools with different SLAs, but this leads to significant under-utilization of expensive accelerators due to load mismatch. In this article, we characterize the LLM serving workloads at Microsoft Office 365, one of the largest users of LLMs within Microsoft Azure cloud with over 10 million requests per day, and highlight key observations across workloads in different data center regions and across time. This is one of the first such public studies of Internet-scale LLM workloads. We use these insights to propose SageServe, a comprehensive LLM serving framework that dynamically adapts to workload demands using multi-timescale control knobs. It combines short-term request routing to data centers with long-term scaling of GPU VMs and model placement with higher lead times, and co-optimizes the routing and resource allocation problem using a traffic forecast model and an Integer Linear Programming (ILP) solution. We evaluate SageServe through real runs and realistic simulations on 10 million production requests across three regions and four open-source models. We achieve up to 25% savings in GPU-hours compared to the current baseline deployment and reduce GPU-hour wastage due to inefficient auto-scaling by 80%, resulting in a potential monthly cost savings of up to $2.5 million, while maintaining tail latency and meeting SLAs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09710v1">Echoing: Identity Failures when LLM Agents Talk to Each Other</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      As large language model (LLM) based agents interact autonomously with one another, a new class of failures emerges that cannot be predicted from single agent performance: behavioral drifts in agent-agent conversations (AxA). Unlike human-agent interactions, where humans ground and steer conversations, AxA lacks such stabilizing signals, making these failures unique. We investigate one such failure, echoing, where agents abandon their assigned roles and instead mirror their conversational partners, undermining their intended objectives. Through experiments across $60$ AxA configurations, $3$ domains, and $2000+$ conversations, we demonstrate that echoing occurs across three major LLM providers, with echoing rates from $5\%$ to $70\%$ depending on the model and domain. Moreover, we find that echoing is persistent even in advanced reasoning models with substantial rates ($32.8\%$) that are not reduced by increased reasoning efforts. We analyze prompt impacts, conversation dynamics, showing that echoing arises as interaction grows longer ($7+$ turns in experiments) and is not merely an artifact of sub-optimal prompting. Finally, we introduce a protocol-level mitigation in which targeted use of structured responses reduces echoing to $9\%$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09693v1">ConstrainedSQL: Training LLMs for Text2SQL via Constrained Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has demonstrated significant promise in enhancing the reasoning capabilities of Text2SQL LLMs, especially with advanced algorithms such as GRPO and DAPO. However, the performance of these methods is highly sensitive to the design of reward functions. Inappropriate rewards can lead to reward hacking, where models exploit loopholes in the reward structure to achieve high scores without genuinely solving the task. This work considers a constrained RL framework for Text2SQL that incorporates natural and interpretable reward and constraint signals, while dynamically balancing trade-offs among them during the training. We establish the theoretical guarantees of our constrained RL framework and our numerical experiments on the well-known Text2SQL datasets substantiate the improvement of our approach over the state-of-the-art RL-trained LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05294v4">Towards Embodied Agentic AI: Review and Classification of LLM- and VLM-Driven Robot Autonomy and Interaction</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Foundation models, including large language models (LLMs) and vision-language models (VLMs), have recently enabled novel approaches to robot autonomy and human-robot interfaces. In parallel, vision-language-action models (VLAs) or large behavior models (LBMs) are increasing the dexterity and capabilities of robotic systems. This survey paper reviews works that advance agentic applications and architectures, including initial efforts with GPT-style interfaces and more complex systems where AI agents function as coordinators, planners, perception actors, or generalist interfaces. Such agentic architectures allow robots to reason over natural language instructions, invoke APIs, plan task sequences, or assist in operations and diagnostics. In addition to peer-reviewed research, due to the fast-evolving nature of the field, we highlight and include community-driven projects, ROS packages, and industrial frameworks that show emerging trends. We propose a taxonomy for classifying model integration approaches and present a comparative analysis of the role that agents play in different solutions in today's literature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09606v1">How Can We Effectively Use LLMs for Phishing Detection?: Evaluating the Effectiveness of Large Language Model-based Phishing Detection Models</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have emerged as a promising phishing detection mechanism, addressing the limitations of traditional deep learning-based detectors, including poor generalization to previously unseen websites and a lack of interpretability. However, LLMs' effectiveness for phishing detection remains unexplored. This study investigates how to effectively leverage LLMs for phishing detection (including target brand identification) by examining the impact of input modalities (screenshots, logos, HTML, and URLs), temperature settings, and prompt engineering strategies. Using a dataset of 19,131 real-world phishing websites and 243 benign sites, we evaluate seven LLMs -- two commercial models (GPT 4.1 and Gemini 2.0 flash) and five open-source models (Qwen, Llama, Janus, DeepSeek-VL2, and R1) -- alongside two deep learning (DL)-based baselines (PhishIntention and Phishpedia). Our findings reveal that commercial LLMs generally outperform open-source models in phishing detection, while DL models demonstrate better performance on benign samples. For brand identification, screenshot inputs achieve optimal results, with commercial LLMs reaching 93-95% accuracy and open-source models, particularly Qwen, achieving up to 92%. However, incorporating multiple input modalities simultaneously or applying one-shot prompts does not consistently enhance performance and may degrade results. Furthermore, higher temperature values reduce performance. Based on these results, we recommend using screenshot inputs with zero temperature to maximize accuracy for LLM-based detectors with HTML serving as auxiliary context when screenshot information is insufficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06942v2">HLPD: Aligning LLMs to Human Language Preference for Machine-Revised Text Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 20 pages, 10 figures, accepted by AAAI'26
    </div>
    <details class="paper-abstract">
      To prevent misinformation and social issues arising from trustworthy-looking content generated by LLMs, it is crucial to develop efficient and reliable methods for identifying the source of texts. Previous approaches have demonstrated exceptional performance in detecting texts fully generated by LLMs. However, these methods struggle when confronting more advanced LLM output or text with adversarial multi-task machine revision, especially in the black-box setting, where the generating model is unknown. To address this challenge, grounded in the hypothesis that human writing possesses distinctive stylistic patterns, we propose Human Language Preference Detection (HLPD). HLPD employs a reward-based alignment process, Human Language Preference Optimization (HLPO), to shift the scoring model's token distribution toward human-like writing, making the model more sensitive to human writing, therefore enhancing the identification of machine-revised text. We test HLPD in an adversarial multi-task evaluation framework that leverages a five-dimensional prompt generator and multiple advanced LLMs to create diverse revision scenarios. When detecting texts revised by GPT-series models, HLPD achieves a 15.11% relative improvement in AUROC over ImBD, surpassing Fast-DetectGPT by 45.56%. When evaluated on texts generated by advanced LLMs, HLPD achieves the highest average AUROC, exceeding ImBD by 5.53% and Fast-DetectGPT by 34.14%. Code will be made available at https://github.com/dfq2021/HLPD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.03789v2">Steve: LLM Powered ChatBot for Career Progression</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      The advancements in systems deploying large language models (LLMs), as well as improvements in their ability to act as agents with predefined templates, provide an opportunity to conduct qualitative, individualized assessments, creating a bridge between qualitative and quantitative methods for candidates seeking career progression. In this paper, we develop a platform that allows candidates to run AI-led interviews to assess their current career stage and curate coursework to enable progression to the next level. Our approach incorporates predefined career trajectories, associated skills, and a method to recommend the best resources for gaining the necessary skills for advancement. We employ OpenAI API calls along with expertly compiled chat templates to assess candidate competence. Our platform is highly configurable due to the modularity of the development, is easy to deploy and use, and available as a web interface where the only requirement is candidate resumes in PDF format. We demonstrate a use-case centered on software engineering and intend to extend this platform to be domain-agnostic, requiring only regular updates to chat templates as industries evolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09458v1">Exploring The Interaction-Outcome Paradox: Seemingly Richer and More Self-Aware Interactions with LLMs May Not Yet Lead to Better Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have transformed the user interface for learning, moving from keyword search to natural language dialogue, their impact on educational outcomes remains unclear. We present a controlled study (N=20) that directly compares the learning interaction and outcomes between LLM and search-based interfaces. We found that although LLMs elicit richer and nuanced interactions from a learner, they do not produce broadly better learning outcomes. In this paper, we explore this the ``Interaction-Outcome Paradox.'' To explain this, we discuss the concept of a cognitive shift: the locus of student effort moves from finding and synthesizing disparate sources (search) to a more self-aware identification and articulation of their knowledge gaps and strategies to bridge those gaps (LLMs). This insight provides a new lens for evaluating educational technologies, suggesting that the future of learning tools lies not in simply enriching interaction, but in designing systems that scaffold productive cognitive work by leveraging this student expressiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.09474v2">Surgical AI Copilot: Energy-Based Fourier Gradient Low-Rank Adaptation for Surgical LLM Agent Reasoning and Planning</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Image-guided surgery demands adaptive, real-time decision support, yet static AI models struggle with structured task planning and providing interactive guidance. Large language models (LLMs)-powered agents offer a promising solution by enabling dynamic task planning and predictive decision support. Despite recent advances, the absence of surgical agent datasets and robust parameter-efficient fine-tuning techniques limits the development of LLM agents capable of complex intraoperative reasoning. In this paper, we introduce Surgical AI Copilot, an LLM agent for image-guided pituitary surgery, capable of conversation, planning, and task execution in response to queries involving tasks such as MRI tumor segmentation, endoscope anatomy segmentation, overlaying preoperative imaging with intraoperative views, instrument tracking, and surgical visual question answering (VQA). To enable structured agent planning, we develop the PitAgent dataset, a surgical context-aware planning dataset covering surgical tasks like workflow analysis, instrument localization, anatomical segmentation, and query-based reasoning. Additionally, we propose DEFT-GaLore, a Deterministic Energy-based Fourier Transform (DEFT) gradient projection technique for efficient low-rank adaptation of recent LLMs (e.g., LLaMA 3.2, Qwen 2.5), enabling their use as surgical agent planners. We extensively validate our agent's performance and the proposed adaptation technique against other state-of-the-art low-rank adaptation methods on agent planning and prompt generation tasks, including a zero-shot surgical VQA benchmark, demonstrating the significant potential for truly efficient and scalable surgical LLM agents in real-time operative settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09438v1">LLM-Guided Dynamic-UMAP for Personalized Federated Graph Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      We propose a method that uses large language models to assist graph machine learning under personalization and privacy constraints. The approach combines data augmentation for sparse graphs, prompt and instruction tuning to adapt foundation models to graph tasks, and in-context learning to supply few-shot graph reasoning signals. These signals parameterize a Dynamic UMAP manifold of client-specific graph embeddings inside a Bayesian variational objective for personalized federated learning. The method supports node classification and link prediction in low-resource settings and aligns language model latent representations with graph structure via a cross-modal regularizer. We outline a convergence argument for the variational aggregation procedure, describe a differential privacy threat model based on a moments accountant, and present applications to knowledge graph completion, recommendation-style link prediction, and citation and product graphs. We also discuss evaluation considerations for benchmarking LLM-assisted graph machine learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09407v1">CARE-Bench: A Benchmark of Diverse Client Simulations Guided by Expert Principles for Evaluating LLMs in Psychological Counseling</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      The mismatch between the growing demand for psychological counseling and the limited availability of services has motivated research into the application of Large Language Models (LLMs) in this domain. Consequently, there is a need for a robust and unified benchmark to assess the counseling competence of various LLMs. Existing works, however, are limited by unprofessional client simulation, static question-and-answer evaluation formats, and unidimensional metrics. These limitations hinder their effectiveness in assessing a model's comprehensive ability to handle diverse and complex clients. To address this gap, we introduce \textbf{CARE-Bench}, a dynamic and interactive automated benchmark. It is built upon diverse client profiles derived from real-world counseling cases and simulated according to expert guidelines. CARE-Bench provides a multidimensional performance evaluation grounded in established psychological scales. Using CARE-Bench, we evaluate several general-purpose LLMs and specialized counseling models, revealing their current limitations. In collaboration with psychologists, we conduct a detailed analysis of the reasons for LLMs' failures when interacting with clients of different types, which provides directions for developing more comprehensive, universal, and effective counseling models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.19254v4">Uncertainty Quantification for Language Models: A Suite of Black-Box, White-Box, LLM Judge, and Ensemble Scorers</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 Accepted by TMLR; UQLM repository: https://github.com/cvs-health/uqlm
    </div>
    <details class="paper-abstract">
      Hallucinations are a persistent problem with Large Language Models (LLMs). As these models become increasingly used in high-stakes domains, such as healthcare and finance, the need for effective hallucination detection is crucial. To this end, we outline a versatile framework for closed-book hallucination detection that practitioners can apply to real-world use cases. To achieve this, we adapt a variety of existing uncertainty quantification (UQ) techniques, including black-box UQ, white-box UQ, and LLM-as-a-Judge, transforming them as necessary into standardized response-level confidence scores ranging from 0 to 1. To enhance flexibility, we propose a tunable ensemble approach that incorporates any combination of the individual confidence scores. This approach enables practitioners to optimize the ensemble for a specific use case for improved performance. To streamline implementation, the full suite of scorers is offered in this paper's companion Python toolkit, UQLM. To evaluate the performance of the various scorers, we conduct an extensive set of experiments using several LLM question-answering benchmarks. We find that our tunable ensemble typically surpasses its individual components and outperforms existing hallucination detection methods. Our results demonstrate the benefits of customized hallucination detection strategies for improving the accuracy and reliability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.01166v2">Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 AAAI 2026
    </div>
    <details class="paper-abstract">
      Automatic Speech Recognition (ASR) aims to convert human speech content into corresponding text. In conversational scenarios, effectively utilizing context can enhance its accuracy. Large Language Models' (LLMs) exceptional long-context understanding and reasoning abilities enable LLM-based ASR (LLM-ASR) to leverage historical context for recognizing conversational speech, which has a high degree of contextual relevance. However, existing conversational LLM-ASR methods use a fixed number of preceding utterances or the entire conversation history as context, resulting in significant ASR confusion and computational costs due to massive irrelevant and redundant information. This paper proposes a multi-modal retrieval-and-selection method named MARS that augments conversational LLM-ASR by enabling it to retrieve and select the most relevant acoustic and textual historical context for the current utterance. Specifically, multi-modal retrieval obtains a set of candidate historical contexts, each exhibiting high acoustic or textual similarity to the current utterance. Multi-modal selection calculates the acoustic and textual similarities for each retrieved candidate historical context and, by employing our proposed near-ideal ranking method to consider both similarities, selects the best historical context. Evaluations on the Interspeech 2025 Multilingual Conversational Speech Language Model Challenge dataset show that the LLM-ASR, when trained on only 1.5K hours of data and equipped with the MARS, outperforms the state-of-the-art top-ranking system trained on 179K hours of data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.22354v2">LLMs Struggle to Reject False Presuppositions when Misinformation Stakes are High</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 8 pages (including References). Published at CogSci 2025: https://escholarship.org/uc/item/4932r1hx
    </div>
    <details class="paper-abstract">
      This paper examines how LLMs handle false presuppositions and whether certain linguistic factors influence their responses to falsely presupposed content. Presuppositions subtly introduce information as given, making them highly effective at embedding disputable or false information. This raises concerns about whether LLMs, like humans, may fail to detect and correct misleading assumptions introduced as false presuppositions, even when the stakes of misinformation are high. Using a systematic approach based on linguistic presupposition analysis, we investigate the conditions under which LLMs are more or less sensitive to adopt or reject false presuppositions. Focusing on political contexts, we examine how factors like linguistic construction, political party, and scenario probability impact the recognition of false presuppositions. We conduct experiments with a newly created dataset and examine three LLMs: OpenAI's GPT-4-o, Meta's LLama-3-8B, and MistralAI's Mistral-7B-v03. Our results show that the models struggle to recognize false presuppositions, with performance varying by condition. This study highlights that linguistic presupposition analysis is a valuable tool for uncovering the reinforcement of political misinformation in LLM responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09323v1">Mixture-of-Channels: Exploiting Sparse FFNs for Efficient LLMs Pre-Training and Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable success across diverse artificial intelligence tasks, driven by scaling laws that correlate model size and training data with performance improvements. However, this scaling paradigm incurs substantial memory overhead, creating significant challenges for both training and inference. While existing research has primarily addressed parameter and optimizer state memory reduction, activation memory-particularly from feed-forward networks (FFNs)-has become the critical bottleneck, especially when FlashAttention is implemented. In this work, we conduct a detailed memory profiling of LLMs and identify FFN activations as the predominant source to activation memory overhead. Motivated by this, we introduce Mixture-of-Channels (MoC), a novel FFN architecture that selectively activates only the Top-K most relevant channels per token determined by SwiGLU's native gating mechanism. MoC substantially reduces activation memory during pre-training and improves inference efficiency by reducing memory access through partial weight loading into GPU SRAM. Extensive experiments validate that MoC delivers significant memory savings and throughput gains while maintaining competitive model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09586v1">Scaling Environments for LLM Agents in the Era of Learning from Interaction: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 20 pages, 4 figures, SEA Workshop @ NeurIPS 2025
    </div>
    <details class="paper-abstract">
      LLM-based agents can autonomously accomplish complex tasks across various domains. However, to further cultivate capabilities such as adaptive behavior and long-term decision-making, training on static datasets built from human-level knowledge is insufficient. These datasets are costly to construct and lack both dynamism and realism. A growing consensus is that agents should instead interact directly with environments and learn from experience through reinforcement learning. We formalize this iterative process as the Generation-Execution-Feedback (GEF) loop, where environments generate tasks to challenge agents, return observations in response to agents' actions during task execution, and provide evaluative feedback on rollouts for subsequent learning. Under this paradigm, environments function as indispensable producers of experiential data, highlighting the need to scale them toward greater complexity, realism, and interactivity. In this survey, we systematically review representative methods for environment scaling from a pioneering environment-centric perspective and organize them along the stages of the GEF loop, namely task generation, task execution, and feedback. We further analyze benchmarks, implementation strategies, and applications, consolidating fragmented advances and outlining future research directions for agent intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04962v2">Too Good to be Bad: On the Failure of LLMs to Role-Play Villains</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly tasked with creative generation, including the simulation of fictional characters. However, their ability to portray non-prosocial, antagonistic personas remains largely unexamined. We hypothesize that the safety alignment of modern LLMs creates a fundamental conflict with the task of authentically role-playing morally ambiguous or villainous characters. To investigate this, we introduce the Moral RolePlay benchmark, a new dataset featuring a four-level moral alignment scale and a balanced test set for rigorous evaluation. We task state-of-the-art LLMs with role-playing characters from moral paragons to pure villains. Our large-scale evaluation reveals a consistent, monotonic decline in role-playing fidelity as character morality decreases. We find that models struggle most with traits directly antithetical to safety principles, such as ``Deceitful'' and ``Manipulative'', often substituting nuanced malevolence with superficial aggression. Furthermore, we demonstrate that general chatbot proficiency is a poor predictor of villain role-playing ability, with highly safety-aligned models performing particularly poorly. Our work provides the first systematic evidence of this critical limitation, highlighting a key tension between model safety and creative fidelity. Our benchmark and findings pave the way for developing more nuanced, context-aware alignment methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05867v2">MCP-RiskCue: Can LLM Infer Risk Information From MCP Server System Logs?</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate strong capabilities in solving complex tasks when integrated with external tools. The Model Context Protocol (MCP) has become a standard interface for enabling such tool-based interactions. However, these interactions introduce substantial security concerns, particularly when the MCP server is compromised or untrustworthy. While prior benchmarks primarily focus on prompt injection attacks or analyze the vulnerabilities of LLM MCP interaction trajectories, limited attention has been given to the underlying system logs associated with malicious MCP servers. To address this gap, we present the first synthetic benchmark for evaluating LLMs ability to identify security risks from system logs. We define nine categories of MCP server risks and generate 1,800 synthetic system logs using ten state-of-the-art LLMs. These logs are embedded in the return values of 243 curated MCP servers, yielding a dataset of 2,421 chat histories for training and 471 queries for evaluation. Our pilot experiments reveal that smaller models often fail to detect risky system logs, leading to high false negatives. While models trained with supervised fine-tuning (SFT) tend to over-flag benign logs, resulting in elevated false positives, Reinforcement Learning from Verifiable Reward (RLVR) offers a better precision-recall balance. In particular, after training with Group Relative Policy Optimization (GRPO), Llama3.1-8B-Instruct achieves 83% accuracy, surpassing the best-performing large remote model by 9 percentage points. Fine-grained, per-category analysis further underscores the effectiveness of reinforcement learning in enhancing LLM safety within the MCP framework. Code and data are available at: https://github.com/PorUna-byte/MCP-RiskCue
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09223v1">AILINKPREVIEWER: Enhancing Code Reviews with LLM-Powered Link Previews</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Code review is a key practice in software engineering, where developers evaluate code changes to ensure quality and maintainability. Links to issues and external resources are often included in Pull Requests (PRs) to provide additional context, yet they are typically discarded in automated tasks such as PR summarization and code review comment generation. This limits the richness of information available to reviewers and increases cognitive load by forcing context-switching. To address this gap, we present AILINKPREVIEWER, a tool that leverages Large Language Models (LLMs) to generate previews of links in PRs using PR metadata, including titles, descriptions, comments, and link body content. We analyzed 50 engineered GitHub repositories and compared three approaches: Contextual LLM summaries, Non-Contextual LLM summaries, and Metadata-based previews. The results in metrics such as BLEU, BERTScore, and compression ratio show that contextual summaries consistently outperform other methods. However, in a user study with seven participants, most preferred non-contextual summaries, suggesting a trade-off between metric performance and perceived usability. These findings demonstrate the potential of LLM-powered link previews to enhance code review efficiency and to provide richer context for developers and automation in software engineering. The video demo is available at https://www.youtube.com/watch?v=h2qH4RtrB3E, and the tool and its source code can be found at https://github.com/c4rtune/AILinkPreviewer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09148v1">LoopTool: Closing the Data-Training Loop for Robust LLM Tool Calls</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Augmenting Large Language Models (LLMs) with external tools enables them to execute complex, multi-step tasks. However, tool learning is hampered by the static synthetic data pipelines where data generation and model training are executed as two separate, non-interactive processes. This approach fails to adaptively focus on a model's specific weaknesses and allows noisy labels to persist, degrading training efficiency. We introduce LoopTool, a fully automated, model-aware data evolution framework that closes this loop by tightly integrating data synthesis and model training. LoopTool iteratively refines both the data and the model through three synergistic modules: (1) Greedy Capability Probing (GCP) diagnoses the model's mastered and failed capabilities; (2) Judgement-Guided Label Verification (JGLV) uses an open-source judge model to find and correct annotation errors, progressively purifying the dataset; and (3) Error-Driven Data Expansion (EDDE) generates new, challenging samples based on identified failures. This closed-loop process operates within a cost-effective, open-source ecosystem, eliminating dependence on expensive closed-source APIs. Experiments show that our 8B model trained with LoopTool significantly surpasses its 32B data generator and achieves new state-of-the-art results on the BFCL-v3 and ACEBench benchmarks for its scale. Our work demonstrates that closed-loop, self-refining data pipelines can dramatically enhance the tool-use capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.05852v2">Quantifying Edits Decay in Fine-tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 We request the withdrawal of this submission due to technical errors in the manuscript record. Specifically, the author order was set incorrectly, the status was mistakenly marked, and the article has not been published. For these reasons, we kindly ask that the submission be retracted from the system
    </div>
    <details class="paper-abstract">
      Knowledge editing has emerged as a lightweight alternative to retraining for correcting or injecting specific facts in large language models (LLMs). Meanwhile, fine-tuning remains the default operation for adapting LLMs to new domains and tasks. Despite their widespread adoption, these two post-training interventions have been studied in isolation, leaving open a crucial question: if we fine-tune an edited model, do the edits survive? This question is motivated by two practical scenarios: removing covert or malicious edits, and preserving beneficial edits. If fine-tuning impairs edits as shown in Figure 1, current KE methods become less useful, as every fine-tuned model would require re-editing, which significantly increases the cost; if edits persist, fine-tuned models risk propagating hidden malicious edits, raising serious safety concerns. To this end, we systematically quantify edits decay after fine-tuning, investigating how fine-tuning affects knowledge editing. We evaluate two state-of-the-art editing methods (MEMIT, AlphaEdit) and three fine-tuning approaches (full-parameter, LoRA, DoRA) across five LLMs and three datasets, yielding 232 experimental configurations. Our results show that edits decay after fine-tuning, with survival varying across configurations, e.g., AlphaEdit edits decay more than MEMIT edits. Further, we propose selective-layer fine-tuning and find that fine-tuning edited layers only can effectively remove edits, though at a slight cost to downstream performance. Surprisingly, fine-tuning non-edited layers impairs more edits than full fine-tuning. Overall, our study establishes empirical baselines and actionable strategies for integrating knowledge editing with fine-tuning, and underscores that evaluating model editing requires considering the full LLM application pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09133v1">Assessing the Capabilities of LLMs in Humor:A Multi-dimensional Analysis of Oogiri Generation and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Computational humor is a frontier for creating advanced and engaging natural language processing (NLP) applications, such as sophisticated dialogue systems. While previous studies have benchmarked the humor capabilities of Large Language Models (LLMs), they have often relied on single-dimensional evaluations, such as judging whether something is simply ``funny.'' This paper argues that a multifaceted understanding of humor is necessary and addresses this gap by systematically evaluating LLMs through the lens of Oogiri, a form of Japanese improvisational comedy games. To achieve this, we expanded upon existing Oogiri datasets with data from new sources and then augmented the collection with Oogiri responses generated by LLMs. We then manually annotated this expanded collection with 5-point absolute ratings across six dimensions: Novelty, Clarity, Relevance, Intelligence, Empathy, and Overall Funniness. Using this dataset, we assessed the capabilities of state-of-the-art LLMs on two core tasks: their ability to generate creative Oogiri responses and their ability to evaluate the funniness of responses using a six-dimensional evaluation. Our results show that while LLMs can generate responses at a level between low- and mid-tier human performance, they exhibit a notable lack of Empathy. This deficit in Empathy helps explain their failure to replicate human humor assessment. Correlation analyses of human and model evaluation data further reveal a fundamental divergence in evaluation criteria: LLMs prioritize Novelty, whereas humans prioritize Empathy. We release our annotated corpus to the community to pave the way for the development of more emotionally intelligent and sophisticated conversational agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09122v1">Vendor-Aware Industrial Agents: RAG-Enhanced LLMs for Secure On-Premise PLC Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Programmable Logic Controllers are operated by proprietary code dialects; this makes it challenging to train coding assistants. Current LLMs are trained on large code datasets and are capable of writing IEC 61131-3 compatible code out of the box, but they neither know specific function blocks, nor related project code. Moreover, companies like Mitsubishi Electric and their customers do not trust cloud providers. Hence, an own coding agent is the desired solution to cope with this. In this study, we present our work on a low-data domain coding assistant solution for industrial use. We show how we achieved high quality code generation without fine-tuning large models and by fine-tuning small local models for edge device usage. Our tool lets several AI models compete with each other, uses reasoning, corrects bugs automatically and checks code validity by compiling it directly in the chat interface. We support our approach with an extensive evaluation that comes with code compilation statistics and user ratings. We found that a Retrieval-Augmented Generation (RAG) supported coding assistant can work in low-data domains by using extensive prompt engineering and directed retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09105v1">Cost-Minimized Label-Flipping Poisoning Attack to LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 accepted for AAAI 2026 Special Track on AI Alignment
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in real-world systems, making it critical to understand their vulnerabilities. While data poisoning attacks during RLHF/DPO alignment have been studied empirically, their theoretical foundations remain unclear. We investigate the minimum-cost poisoning attack required to steer an LLM's policy toward an attacker's target by flipping preference labels during RLHF/DPO, without altering the compared outputs. We formulate this as a convex optimization problem with linear constraints, deriving lower and upper bounds on the minimum attack cost. As a byproduct of this theoretical analysis, we show that any existing label-flipping attack can be post-processed via our proposed method to reduce the number of label flips required while preserving the intended poisoning effect. Empirical results demonstrate that this cost-minimization post-processing can significantly reduce poisoning costs over baselines, particularly when the reward model's feature dimension is small relative to the dataset size. These findings highlight fundamental vulnerabilities in RLHF/DPO pipelines and provide tools to evaluate their robustness against low-cost poisoning attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.14389v2">Leveraging Small LLMs for Argument Mining in Education: Argument Component Identification, Classification, and Assessment</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      Argument mining algorithms analyze the argumentative structure of essays, making them a valuable tool for enhancing education by providing targeted feedback on the students' argumentation skills. While current methods often use encoder or encoder-decoder deep learning architectures, decoder-only models remain largely unexplored, offering a promising research direction. This paper proposes leveraging open-source, small Large Language Models (LLMs) for argument mining through few-shot prompting and fine-tuning. These models' small size and open-source nature ensure accessibility, privacy, and computational efficiency, enabling schools and educators to adopt and deploy them locally. Specifically, we perform three tasks: segmentation of student essays into arguments, classification of the arguments by type, and assessment of their quality. We empirically evaluate the models on the Feedback Prize - Predicting Effective Arguments dataset of grade 6-12 students essays and demonstrate how fine-tuned small LLMs outperform baseline methods in segmenting the essays and determining the argument types while few-shot prompting yields comparable performance to that of the baselines in assessing quality. This work highlights the educational potential of small, open-source LLMs to provide real-time, personalized feedback, enhancing independent learning and writing skills while ensuring low computational cost and privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07061v2">Do LLMs Feel? Teaching Emotion Recognition with Prompts, Retrieval, and Curriculum Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 Accepted at AAAI 2026
    </div>
    <details class="paper-abstract">
      Emotion Recognition in Conversation (ERC) is a crucial task for understanding human emotions and enabling natural human-computer interaction. Although Large Language Models (LLMs) have recently shown great potential in this field, their ability to capture the intrinsic connections between explicit and implicit emotions remains limited. We propose a novel ERC training framework, PRC-Emo, which integrates Prompt engineering, demonstration Retrieval, and Curriculum learning, with the goal of exploring whether LLMs can effectively perceive emotions in conversational contexts. Specifically, we design emotion-sensitive prompt templates based on both explicit and implicit emotional cues to better guide the model in understanding the speaker's psychological states. We construct the first dedicated demonstration retrieval repository for ERC, which includes training samples from widely used datasets, as well as high-quality dialogue examples generated by LLMs and manually verified. Moreover, we introduce a curriculum learning strategy into the LoRA fine-tuning process, incorporating weighted emotional shifts between same-speaker and different-speaker utterances to assign difficulty levels to dialogue samples, which are then organized in an easy-to-hard training sequence. Experimental results on two benchmark datasets -- IEMOCAP and MELD -- show that our method achieves new state-of-the-art (SOTA) performance, demonstrating the effectiveness and generalizability of our approach in improving LLM-based emotional understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09087v1">Tele-LLM-Hub: Building Context-Aware Multi-Agent LLM Systems for Telecom Networks</a></div>
    <div class="paper-meta">
      📅 2025-11-12
    </div>
    <details class="paper-abstract">
      This paper introduces Tele-LLM-Hub, a user friendly low-code solution for rapid prototyping and deployment of context aware multi-agent (MA) Large Language Model (LLM) systems tailored for 5G and beyond. As telecom wireless networks become increasingly complex, intelligent LLM applications must share a domainspecific understanding of network state. We propose TeleMCP, the Telecom Model Context Protocol, to enable structured and context-rich communication between agents in telecom environments. Tele-LLM-Hub actualizes TeleMCP through a low-code interface that supports agent creation, workflow composition, and interaction with software stacks such as srsRAN. Key components include a direct chat interface, a repository of pre-built systems, an Agent Maker leveraging finetuning with our RANSTRUCT framework, and an MA-Maker for composing MA workflows. The goal of Tele-LLM-Hub is to democratize the design of contextaware MA systems and accelerate innovation in next-generation wireless networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10975v2">ReFineG: Synergizing Small Supervised Models and LLMs for Low-Resource Grounded Multimodal NER</a></div>
    <div class="paper-meta">
      📅 2025-11-12
      | 💬 CCKS 2025 Shared Task Paper
    </div>
    <details class="paper-abstract">
      Grounded Multimodal Named Entity Recognition (GMNER) extends traditional NER by jointly detecting textual mentions and grounding them to visual regions. While existing supervised methods achieve strong performance, they rely on costly multimodal annotations and often underperform in low-resource domains. Multimodal Large Language Models (MLLMs) show strong generalization but suffer from Domain Knowledge Conflict, producing redundant or incorrect mentions for domain-specific entities. To address these challenges, we propose ReFineG, a three-stage collaborative framework that integrates small supervised models with frozen MLLMs for low-resource GMNER. In the Training Stage, a domain-aware NER data synthesis strategy transfers LLM knowledge to small models with supervised training while avoiding domain knowledge conflicts. In the Refinement Stage, an uncertainty-based mechanism retains confident predictions from supervised models and delegates uncertain ones to the MLLM. In the Grounding Stage, a multimodal context selection algorithm enhances visual grounding through analogical reasoning. In the CCKS2025 GMNER Shared Task, ReFineG ranked second with an F1 score of 0.6461 on the online leaderboard, demonstrating its effectiveness with limited annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08055v1">MSCR: Exploring the Vulnerability of LLMs' Mathematical Reasoning Abilities Using Multi-Source Candidate Replacement</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      LLMs demonstrate performance comparable to human abilities in complex tasks such as mathematical reasoning, but their robustness in mathematical reasoning under minor input perturbations still lacks systematic investigation. Existing methods generally suffer from limited scalability, weak semantic preservation, and high costs. Therefore, we propose MSCR, an automated adversarial attack method based on multi-source candidate replacement. By combining three information sources including cosine similarity in the embedding space of LLMs, the WordNet dictionary, and contextual predictions from a masked language model, we generate for each word in the input question a set of semantically similar candidates, which are then filtered and substituted one by one to carry out the attack. We conduct large-scale experiments on LLMs using the GSM8K and MATH500 benchmarks. The results show that even a slight perturbation involving only a single word can significantly reduce the accuracy of all models, with the maximum drop reaching 49.89% on GSM8K and 35.40% on MATH500, while preserving the high semantic consistency of the perturbed questions. Further analysis reveals that perturbations not only lead to incorrect outputs but also substantially increase the average response length, which results in more redundant reasoning paths and higher computational resource consumption. These findings highlight the robustness deficiencies and efficiency bottlenecks of current LLMs in mathematical reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08052v1">Dual-Process Scaffold Reasoning for Enhancing LLM Code Debugging</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 5 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Recent LLMs have demonstrated sophisticated problem-solving capabilities on various benchmarks through advanced reasoning algorithms. However, the key research question of identifying reasoning steps that balance complexity and computational efficiency remains unsolved. Recent research has increasingly drawn upon psychological theories to explore strategies for optimizing cognitive pathways. The LLM's final outputs and intermediate steps are regarded as System 1 and System 2, respectively. However, an in-depth exploration of the System 2 reasoning is still lacking. Therefore, we propose a novel psychologically backed Scaffold Reasoning framework for code debugging, which encompasses the Scaffold Stream, Analytic Stream, and Integration Stream. The construction of reference code within the Scaffold Stream is integrated with the buggy code analysis results produced by the Analytic Stream through the Integration Stream. Our framework achieves an 88.91% pass rate and an average inference time of 5.36 seconds per-problem on DebugBench, outperforming other reasoning approaches across various LLMs in both reasoning accuracy and efficiency. Further analyses elucidate the advantages and limitations of various cognitive pathways across varying problem difficulties and bug types. Our findings also corroborate the alignment of the proposed Scaffold Reasoning framework with human cognitive processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04070v2">LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted at AAAI-26
    </div>
    <details class="paper-abstract">
      Navigation instruction generation for visually impaired (VI) individuals (NIG-VI) is critical yet relatively underexplored. This study focuses on generating precise, in-situ, step-by-step navigation instructions that are practically usable for VI users. Specifically, we propose LaF-GRPO (LLM-as-Follower GRPO), where an LLM simulates VI user responses to navigation instructions, thereby providing feedback rewards to guide the post-training of a Vision-Language Model (VLM). This enhances instruction accuracy and usability while reducing costly real-world data collection needs. To address the scarcity of dedicated benchmarks in this field, we introduce NIG4VI, a 27k-sample open-source dataset to facilitate training and evaluation. It comprises diverse navigation scenarios with accurate spatial coordinates, supporting detailed and open-ended in-situ instruction generation. Experiments on NIG4VI demonstrate the effectiveness of LaF-GRPO through quantitative metrics (e.g., Zero-(LaF-GRPO) boosts BLEU 14\%; SFT+(LaF-GRPO) METEOR 0.542 vs. GPT-4o 0.323), and qualitative analysis further confirms that our method yields more intuitive and safer instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08012v1">DOA Estimation with Lightweight Network on LLM-Aided Simulated Acoustic Scenes</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Direction-of-Arrival (DOA) estimation is critical in spatial audio and acoustic signal processing, with wide-ranging applications in real-world. Most existing DOA models are trained on synthetic data by convolving clean speech with room impulse responses (RIRs), which limits their generalizability due to constrained acoustic diversity. In this paper, we revisit DOA estimation using a recently introduced dataset constructed with the assistance of large language models (LLMs), which provides more realistic and diverse spatial audio scenes. We benchmark several representative neural-based DOA methods on this dataset and propose LightDOA, a lightweight DOA estimation model based on depthwise separable convolutions, specifically designed for mutil-channel input in varying environments. Experimental results show that LightDOA achieves satisfactory accuracy and robustness across various acoustic scenes while maintaining low computational complexity. This study not only highlights the potential of spatial audio synthesized with the assistance of LLMs in advancing robust and efficient DOA estimation research, but also highlights LightDOA as efficient solution for resource-constrained applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.08008v1">Combining LLM Semantic Reasoning with GNN Structural Modeling for Multi-view Multi-Label Feature Selection</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Multi-view multi-label feature selection aims to identify informative features from heterogeneous views, where each sample is associated with multiple interdependent labels. This problem is particularly important in machine learning involving high-dimensional, multimodal data such as social media, bioinformatics or recommendation systems. Existing Multi-View Multi-Label Feature Selection (MVMLFS) methods mainly focus on analyzing statistical information of data, but seldom consider semantic information. In this paper, we aim to use these two types of information jointly and propose a method that combines Large Language Models (LLMs) semantic reasoning with Graph Neural Networks (GNNs) structural modeling for MVMLFS. Specifically, the method consists of three main components. (1) LLM is first used as an evaluation agent to assess the latent semantic relevance among feature, view, and label descriptions. (2) A semantic-aware heterogeneous graph with two levels is designed to represent relations among features, views and labels: one is a semantic graph representing semantic relations, and the other is a statistical graph. (3) A lightweight Graph Attention Network (GAT) is applied to learn node embedding in the heterogeneous graph as feature saliency scores for ranking and selection. Experimental results on multiple benchmark datasets demonstrate the superiority of our method over state-of-the-art baselines, and it is still effective when applied to small-scale datasets, showcasing its robustness, flexibility, and generalization ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07991v1">VSPO: Validating Semantic Pitfalls in Ontology via LLM-Based CQ Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted at AAAI 2026 oral
    </div>
    <details class="paper-abstract">
      Competency Questions (CQs) play a crucial role in validating ontology design. While manually crafting CQs can be highly time-consuming and costly for ontology engineers, recent studies have explored the use of large language models (LLMs) to automate this process. However, prior approaches have largely evaluated generated CQs based on their similarity to existing datasets, which often fail to verify semantic pitfalls such as "Misusing allValuesFrom". Since such pitfalls cannot be reliably detected through rule-based methods, we propose a novel dataset and model of Validating Semantic Pitfalls in Ontology (VSPO) for CQ generation specifically designed to verify the semantic pitfalls. To simulate missing and misused axioms, we use LLMs to generate natural language definitions of classes and properties and introduce misalignments between the definitions and the ontology by removing axioms or altering logical operators (e.g., substituting union with intersection). We then fine-tune LLaMA-3.1-8B-Instruct to generate CQs that validate these semantic discrepancies between the provided definitions and the corresponding axioms. The resulting CQs can detect a broader range of modeling errors compared to existing public datasets. Our fine-tuned model demonstrates superior performance over baselines, showing 26% higher precision and 28.2% higher recall than GPT-4.1 in generating CQs for pitfall validation. This research enables automatic generation of TBox-validating CQs using LLMs, significantly reducing manual effort while improving semantic alignment between ontologies and expert knowledge. To the best of our knowledge, this is the first study to target semantic pitfall validation in CQ generation using LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06209v2">Reasoning with Confidence: Efficient Verification of LLM Reasoning Steps via Uncertainty Heads</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Preprint under review
    </div>
    <details class="paper-abstract">
      Solving complex tasks usually requires LLMs to generate long multi-step reasoning chains. Previous work has shown that verifying the correctness of individual reasoning steps can further improve the performance and efficiency of LLMs on such tasks and enhance solution interpretability. However, existing verification approaches, such as Process Reward Models (PRMs), are either computationally expensive, limited to specific domains, or require large-scale human or model-generated annotations. Thus, we propose a lightweight alternative for step-level reasoning verification based on data-driven uncertainty scores. We train transformer-based uncertainty quantification heads (UHeads) that use the internal states of a frozen LLM to estimate the uncertainty of its reasoning steps during generation. The approach is fully automatic: target labels are generated either by another larger LLM (e.g., DeepSeek R1) or in a self-supervised manner by the original model itself. UHeads are both effective and lightweight, containing less than 10M parameters. Across multiple domains, including mathematics, planning, and general knowledge question answering, they match or even surpass the performance of PRMs that are up to 810x larger. Our findings suggest that the internal states of LLMs encode their uncertainty and can serve as reliable signals for reasoning verification, offering a promising direction toward scalable and generalizable introspective LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07982v1">NOTAM-Evolve: A Knowledge-Guided Self-Evolving Optimization Framework with LLMs for NOTAM Interpretation</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Accurate interpretation of Notices to Airmen (NOTAMs) is critical for aviation safety, yet their condensed and cryptic language poses significant challenges to both manual and automated processing. Existing automated systems are typically limited to shallow parsing, failing to extract the actionable intelligence needed for operational decisions. We formalize the complete interpretation task as deep parsing, a dual-reasoning challenge requiring both dynamic knowledge grounding (linking the NOTAM to evolving real-world aeronautical data) and schema-based inference (applying static domain rules to deduce operational status). To tackle this challenge, we propose NOTAM-Evolve, a self-evolving framework that enables a large language model (LLM) to autonomously master complex NOTAM interpretation. Leveraging a knowledge graph-enhanced retrieval module for data grounding, the framework introduces a closed-loop learning process where the LLM progressively improves from its own outputs, minimizing the need for extensive human-annotated reasoning traces. In conjunction with this framework, we introduce a new benchmark dataset of 10,000 expert-annotated NOTAMs. Our experiments demonstrate that NOTAM-Evolve achieves a 30.4% absolute accuracy improvement over the base LLM, establishing a new state of the art on the task of structured NOTAM interpretation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07971v1">Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted to the AAAI Conference on Artificial Intelligence (AAAI-2026)
    </div>
    <details class="paper-abstract">
      We introduce LOREN, a curvature-aware zeroth-order (ZO) optimization method for fine-tuning large language models (LLMs). Existing ZO methods, which estimate gradients via finite differences using random perturbations, often suffer from high variance and suboptimal search directions. Our approach addresses these challenges by: (i) reformulating the problem of gradient preconditioning as that of adaptively estimating an anisotropic perturbation distribution for gradient estimation, (ii) capturing curvature through a low-rank block diagonal preconditioner using the framework of natural evolution strategies, and (iii) applying a REINFORCE leave-one-out (RLOO) gradient estimator to reduce variance. Experiments on standard LLM benchmarks show that our method outperforms state-of-the-art ZO methods by achieving higher accuracy and faster convergence, while cutting peak memory usage by up to 27.3% compared with MeZO-Adam.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06778v2">SAFENLIDB: A Privacy-Preserving Safety Alignment Framework for LLM-based Natural Language Database Interfaces</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 AAAI 2026 Extended Version
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has driven significant progress in Natural Language Interface to Database (NLIDB). However, the widespread adoption of LLMs has raised critical privacy and security concerns. During interactions, LLMs may unintentionally expose confidential database contents or be manipulated by attackers to exfiltrate data through seemingly benign queries. While current efforts typically rely on rule-based heuristics or LLM agents to mitigate this leakage risk, these methods still struggle with complex inference-based attacks, suffer from high false positive rates, and often compromise the reliability of SQL queries. To address these challenges, we propose \textsc{SafeNlidb}, a novel privacy-security alignment framework for LLM-based NLIDB. The framework features an automated pipeline that generates hybrid chain-of-thought interaction data from scratch, seamlessly combining implicit security reasoning with SQL generation. Additionally, we introduce reasoning warm-up and alternating preference optimization to overcome the multi-preference oscillations of Direct Preference Optimization (DPO), enabling LLMs to produce security-aware SQL through fine-grained reasoning without the need for human-annotated preference data. Extensive experiments demonstrate that our method outperforms both larger-scale LLMs and ideal-setting baselines, achieving significant security improvements while preserving high utility. WARNING: This work may contain content that is offensive and harmful!
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07943v1">Thinker: Training LLMs in Hierarchical Thinking for Deep Search via Multi-Turn Interaction</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted to AAAI 2026. Extended version with full Appendix
    </div>
    <details class="paper-abstract">
      Efficient retrieval of external knowledge bases and web pages is crucial for enhancing the reasoning abilities of LLMs. Previous works on training LLMs to leverage external retrievers for solving complex problems have predominantly employed end-to-end reinforcement learning. However, these approaches neglect supervision over the reasoning process, making it difficult to guarantee logical coherence and rigor. To address these limitations, we propose Thinker, a hierarchical thinking model for deep search through multi-turn interaction, making the reasoning process supervisable and verifiable. It decomposes complex problems into independently solvable sub-problems, each dually represented in both natural language and an equivalent logical function to support knowledge base and web searches. Concurrently, dependencies between sub-problems are passed as parameters via these logical functions, enhancing the logical coherence of the problem-solving process. To avoid unnecessary external searches, we perform knowledge boundary determination to check if a sub-problem is within the LLM's intrinsic knowledge, allowing it to answer directly. Experimental results indicate that with as few as several hundred training samples, the performance of Thinker is competitive with established baselines. Furthermore, when scaled to the full training set, Thinker significantly outperforms these methods across various datasets and model sizes. The source code is available at https://github.com/OpenSPG/KAG-Thinker.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07910v1">Last Layer Logits to Logic: Empowering LLMs with Logic-Consistent Structured Knowledge Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 ICLR26 Submission
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve excellent performance in natural language reasoning tasks through pre-training on vast unstructured text, enabling them to understand the logic in natural language and generate logic-consistent responses. However, the representational differences between unstructured and structured knowledge make LLMs inherently struggle to maintain logic consistency, leading to \textit{Logic Drift} challenges in structured knowledge reasoning tasks such as Knowledge Graph Question Answering (KGQA). Existing methods address this limitation by designing complex workflows embedded in prompts to guide LLM reasoning. Nevertheless, these approaches only provide input-level guidance and fail to fundamentally address the \textit{Logic Drift} in LLM outputs. Additionally, their inflexible reasoning workflows cannot adapt to different tasks and knowledge graphs. To enhance LLMs' logic consistency in structured knowledge reasoning, we specifically target the logits output from the autoregressive generation process. We propose the \textit{Logits-to-Logic} framework, which incorporates logits strengthening and logits filtering as core modules to correct logical defects in LLM outputs. Extensive experiments show that our approach significantly improves LLMs' logic consistency in structured knowledge reasoning and achieves state-of-the-art performance on multiple KGQA benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.01849v2">A Multi-Agent Conversational Bandit Approach to Online Evaluation and Selection of User-Aligned LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Prompt-based offline methods are commonly used to optimize large language model (LLM) responses, but evaluating these responses is computationally intensive and often fails to accommodate diverse response styles. This study introduces a novel online evaluation framework that employs a multi-agent conversational bandit model to select optimal responses while aligning with user preferences dynamically. To tackle challenges such as high-dimensional features, large response sets, adaptive conversational needs, and multi-device access, we propose MACO, Multi-Agent Conversational Online Learning, which comprises two key components: (1) \texttt{MACO-A}: Executed by local agents, it employs an online elimination mechanism to filter out low-quality responses. (2) \texttt{MACO-S}: Executed by the cloud server, it adaptively adjusts selection strategies based on aggregated preference data. An adaptive preference mechanism triggers asynchronous conversations to enhance alignment efficiency. Theoretical analysis demonstrates that MACO achieves near-optimal regret bounds, matching state-of-the-art performance in various degenerate cases. Extensive experiments utilizing Google and OpenAI text embedding models on the real-world datasets with different response styles, combined with Llama and GPT-4o, show that MACO consistently outperforms baseline methods by at least 8.29\% across varying response set sizes and numbers of agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.05831v3">Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07876v1">LoopLLM: Transferable Energy-Latency Attacks in LLMs via Repetitive Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 14 pages with 7 figures; accepted by the AAAI 2026
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) scale, their inference incurs substantial computational resources, exposing them to energy-latency attacks, where crafted prompts induce high energy and latency cost. Existing attack methods aim to prolong output by delaying the generation of termination symbols. However, as the output grows longer, controlling the termination symbols through input becomes difficult, making these methods less effective. Therefore, we propose LoopLLM, an energy-latency attack framework based on the observation that repetitive generation can trigger low-entropy decoding loops, reliably compelling LLMs to generate until their output limits. LoopLLM introduces (1) a repetition-inducing prompt optimization that exploits autoregressive vulnerabilities to induce repetitive generation, and (2) a token-aligned ensemble optimization that aggregates gradients to improve cross-model transferability. Extensive experiments on 12 open-source and 2 commercial LLMs show that LoopLLM significantly outperforms existing methods, achieving over 90% of the maximum output length, compared to 20% for baselines, and improving transferability by around 40% to DeepSeek-V3 and Gemini 2.5 Flash.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.01724v2">ReflecSched: Solving Dynamic Flexible Job-Shop Scheduling via LLM-Powered Hierarchical Reflection</a></div>
    <div class="paper-meta">
      📅 2025-11-11
    </div>
    <details class="paper-abstract">
      The NP-hard Dynamic Flexible Job-Shop Scheduling (DFJSP) problem involves real-time events and complex routing. While traditional rules are efficient but rigid, deep learning is opaque and requires feature engineering. Large Language Models (LLMs) promise adaptive reasoning without this engineering overhead, yet we find their direct application is suboptimal. Baseline LLMs suffer from three key pitfalls: the long-context paradox, where crucial data is underutilized; an underutilization of expert heuristics; and myopic decision-making. To address this, we propose ReflecSched, a framework that empowers the LLM beyond a direct scheduler by equipping it with a strategic analysis capability. ReflecSched tasks the LLM to analyze heuristic-driven simulations across multiple planning horizons and distill them into a concise, natural-language summary termed ``Strategic Experience''. This summary is then integrated into the prompt of a final decision-making module, guiding it to produce non-myopic actions. Experiments demonstrate ReflecSched achieves superior performance, with its best variants attaining an average RPD of 6.04\% and rank of 3.18, significantly outperforming strong traditional and learning-based methods. It also statistically and decisively surpasses direct LLM baselines, securing a 71.35\% Win Rate while being, on average, 15.1\% more token-efficient on Normal-scale problems. Ablation studies attribute this performance to a robust reflection mechanism that leverages high-quality, contrastive experience. This mechanism mitigates key LLM pitfalls like myopic greed, enabling ReflecSched to outperform all evaluated heuristics. Ultimately, the framework's performance is statistically on par with an oracle-like strategy, showcasing its effectiveness and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.07791v9">Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 AACL-IJCNLP 2025
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge has emerged as a promising alternative to human evaluators across various tasks, yet inherent biases - particularly position bias, the tendency to favor solutions based on their position within the prompt - compromise its reliability. This exploratory study evaluates position bias in LLM judges across pairwise and list-wise comparison settings, introducing three metrics: repetition stability, position consistency, and preference fairness. Our experiments, involving 15 LLM judges across MTBench and DevBench with 22 tasks and approximately 40 solution-generating models, result in over 150,000 evaluation instances. We identify Judge-Level, Candidate-Level, and Task-Level factors contributing to bias. The findings confirm that position bias is not due to random chance and varies significantly across judges and tasks. While position bias is weakly influenced by the length of prompt components, it is strongly affected by the quality gap between solutions. Our agreement and disagreement analysis among judges further provides insights into the distribution of judging difficulty across the dataset, and highlights the potential for dataset modifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07865v1">LLM-Powered Fully Automated Chaos Engineering: Towards Enabling Anyone to Build Resilient Software Systems at Low Cost</a></div>
    <div class="paper-meta">
      📅 2025-11-11
      | 💬 Accepted at ASE 2025 NIER Track. The code is available at https://github.com/ntt-dkiku/chaos-eater
    </div>
    <details class="paper-abstract">
      Chaos Engineering (CE) is an engineering technique aimed at improving the resilience of distributed systems. It involves intentionally injecting faults into a system to test its resilience, uncover weaknesses, and address them before they cause failures in production. Recent CE tools automate the execution of predefined CE experiments. However, planning such experiments and improving the system based on the experimental results still remain manual. These processes are labor-intensive and require multi-domain expertise. To address these challenges and enable anyone to build resilient systems at low cost, this paper proposes ChaosEater, a system that automates the entire CE cycle with Large Language Models (LLMs). It predefines an agentic workflow according to a systematic CE cycle and assigns subdivided processes within the workflow to LLMs. ChaosEater targets CE for software systems built on Kubernetes. Therefore, the LLMs in ChaosEater complete CE cycles through software engineering tasks, including requirement definition, code generation, testing, and debugging. We evaluate ChaosEater through case studies on small- and large-scale Kubernetes systems. The results demonstrate that it consistently completes reasonable CE cycles with significantly low time and monetary costs. Its cycles are also qualitatively validated by human engineers and LLMs.
    </details>
</div>
