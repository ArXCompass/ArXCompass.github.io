# llm - 2026_01

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04275v1">Shadow Unlearning: A Neuro-Semantic Approach to Fidelity-Preserving Faceless Forgetting in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Machine unlearning aims to selectively remove the influence of specific training samples to satisfy privacy regulations such as the GDPR's 'Right to be Forgotten'. However, many existing methods require access to the data being removed, exposing it to membership inference attacks and potential misuse of Personally Identifiable Information (PII). We address this critical challenge by proposing Shadow Unlearning, a novel paradigm of approximate unlearning, that performs machine unlearning on anonymized forget data without exposing PII. We further propose a novel privacy-preserving framework, Neuro-Semantic Projector Unlearning (NSPU) to achieve Shadow unlearning. To evaluate our method, we compile Multi-domain Fictitious Unlearning (MuFU) forget set across five diverse domains and introduce an evaluation stack to quantify the trade-off between knowledge retention and unlearning effectiveness. Experimental results on various LLMs show that NSPU achieves superior unlearning performance, preserves model utility, and enhances user privacy. Additionally, the proposed approach is at least 10 times more computationally efficient than standard unlearning approaches. Our findings foster a new direction for privacy-aware machine unlearning that balances data protection and model fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04170v1">Agent Drift: Quantifying Behavioral Degradation in Multi-Agent LLM Systems Over Extended Interactions</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Multi-agent Large Language Model (LLM) systems have emerged as powerful architectures for complex task decomposition and collaborative problem-solving. However, their long-term behavioral stability remains largely unexamined. This study introduces the concept of agent drift, defined as the progressive degradation of agent behavior, decision quality, and inter-agent coherence over extended interaction sequences. We present a comprehensive theoretical framework for understanding drift phenomena, proposing three distinct manifestations: semantic drift (progressive deviation from original intent), coordination drift (breakdown in multi-agent consensus mechanisms), and behavioral drift (emergence of unintended strategies). We introduce the Agent Stability Index (ASI), a novel composite metric framework for quantifying drift across twelve dimensions, including response consistency, tool usage patterns, reasoning pathway stability, and inter-agent agreement rates. Through simulation-based analysis and theoretical modeling, we demonstrate how unchecked agent drift can lead to substantial reductions in task completion accuracy and increased human intervention requirements. We propose three mitigation strategies: episodic memory consolidation, drift-aware routing protocols, and adaptive behavioral anchoring. Theoretical analysis suggests these approaches can significantly reduce drift-related errors while maintaining system throughput. This work establishes a foundational methodology for monitoring, measuring, and mitigating agent drift in production agentic AI systems, with direct implications for enterprise deployment reliability and AI safety research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06303v4">Reward Is Enough: LLMs Are In-Context Reinforcement Learners</a></div>
    <div class="paper-meta">
      📅 2026-01-07
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is a framework for solving sequential decision-making problems. In this work, we demonstrate that, surprisingly, RL emerges during the inference time of large language models (LLMs), a phenomenon we term in-context RL (ICRL). To reveal this capability, we introduce a simple multi-round prompting framework, we call ICRL prompting, for inference-time self-improvement. The goal of ICRL prompting is to guide LLMs to perform reinforcement learning during inference for self-improvement on a given task. After each response, the model receives numerical scalar feedback, denoted as a reward. In the next round, we prompt the LLM again together with a context that concatenates all prior responses and their associated rewards. We consistently observe that response quality improves as the context grows. In other words, the LLM can optimize scalar reward signals during inference, exhibiting behavior analogous to reinforcement learning. We evaluate ICRL prompting on Game of 24, creative writing, ScienceWorld, and Olympiad-level math competitions (AIME and HMMT), demonstrating significant improvements over baselines such as Self-Refine and Reflexion. Notably, even when the reward signals are generated by the same LLM, ICRL prompting still improves performance, highlighting a promising new paradigm for test-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04093v1">SearchAttack: Red-Teaming LLMs against Real-World Threats via Framing Unsafe Web Information-Seeking Tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-07
      | 💬 We find that the key to jailbreak the LLM is objectifying its safety responsibility, thus we delegate the open-web to inject harmful semantics and get the huge gain from unmoderated web resources
    </div>
    <details class="paper-abstract">
      Recently, people have suffered and become increasingly aware of the unreliability gap in LLMs for open and knowledge-intensive tasks, and thus turn to search-augmented LLMs to mitigate this issue. However, when the search engine is triggered for harmful tasks, the outcome is no longer under the LLM's control. Once the returned content directly contains targeted, ready-to-use harmful takeaways, the LLM's safeguards cannot withdraw that exposure. Motivated by this dilemma, we identify web search as a critical attack surface and propose \textbf{\textit{SearchAttack}} for red-teaming. SearchAttack outsources the harmful semantics to web search, retaining only the query's skeleton and fragmented clues, and further steers LLMs to reconstruct the retrieved content via structural rubrics to achieve malicious goals. Extensive experiments are conducted to red-team the search-augmented LLMs for responsible vulnerability assessment. Empirically, SearchAttack demonstrates strong effectiveness in attacking these systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03121v1">ToxiGAN: Toxic Data Augmentation via LLM-Guided Directional Adversarial Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 This paper has been accepted to the main conference of EACL 2026
    </div>
    <details class="paper-abstract">
      Augmenting toxic language data in a controllable and class-specific manner is crucial for improving robustness in toxicity classification, yet remains challenging due to limited supervision and distributional skew. We propose ToxiGAN, a class-aware text augmentation framework that combines adversarial generation with semantic guidance from large language models (LLMs). To address common issues in GAN-based augmentation such as mode collapse and semantic drift, ToxiGAN introduces a two-step directional training strategy and leverages LLM-generated neutral texts as semantic ballast. Unlike prior work that treats LLMs as static generators, our approach dynamically selects neutral exemplars to provide balanced guidance. Toxic samples are explicitly optimized to diverge from these exemplars, reinforcing class-specific contrastive signals. Experiments on four hate speech benchmarks show that ToxiGAN achieves the strongest average performance in both macro-F1 and hate-F1, consistently outperforming traditional and LLM-based augmentation methods. Ablation and sensitivity analyses further confirm the benefits of semantic ballast and directional training in enhancing classifier robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.16199v5">Awakening LLMs' Reasoning Potential: A Fine-Grained Pipeline to Evaluate and Mitigate Vague Perception</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly trained to abstain on difficult questions by answering unknown. However, we observe that LLMs often misuse this option: they output unknown even when LLMs can actually solve the questions, or they fail to understand why questions are truly unsolvable. We formalize this mismatch between potential ability and the inclination of abstention as the Vague Perception phenomenon. We introduce the WakenLLM pipeline that (1) extracts Vague Perception samples and (2) measures how many of them can be converted to correct answers under stimulation. Based on stage-wise metrics (TCR, OCR, etc.) and the upper-bound accuracy Acc(WakenLLM), we quantify LLMs' reasoning potential beyond one-shot accuracy. Experiments on six LLMs suggest that, without further training or parameter revisions, LLMs can achieve up to a 68.53% increase in accuracy on Vague Perception samples through our designed pipeline. We further analyze how Vague Perception, Conformity and Degradation vary from model families and parameter sizes, and offer model selection strategies in multi-stage reasoning workflows. Finally, by comparing WakenLLM against mainstream reasoning baselines, both training and non-training ones, we show that existing baselines only activate a small portion of LLMs' reasoning potential, pointing to perception-aware reasoning as a promising direction for future LLM designing. Code and datasets are available at https://github.com/WakenLLMTeam/WakenLLM-toolkit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03103v1">Who Laughs with Whom? Disentangling Influential Factors in Humor Preferences across User Clusters and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Humor preferences vary widely across individuals and cultures, complicating the evaluation of humor using large language models (LLMs). In this study, we model heterogeneity in humor preferences in Oogiri, a Japanese creative response game, by clustering users with voting logs and estimating cluster-specific weights over interpretable preference factors using Bradley-Terry-Luce models. We elicit preference judgments from LLMs by prompting them to select the funnier response and found that user clusters exhibit distinct preference patterns and that the LLM results can resemble those of particular clusters. Finally, we demonstrate that, by persona prompting, LLM preferences can be directed toward a specific cluster. The scripts for data collection and analysis will be released to support reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.20278v3">Quantifying LLM Biases Across Instruction Boundary in Mixed Question Forms</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) annotated datasets are widely used nowadays, however, large-scale annotations often show biases in low-quality datasets. For example, Multiple-Choice Questions (MCQs) datasets with one single correct option is common, however, there may be questions attributed to none or multiple correct options; whereas true-or-false questions are supposed to be labeled with either True or False, but similarly the text can include unsolvable elements, which should be further labeled as Unknown. There are problems when low-quality datasets with mixed question forms can not be identified. We refer to these exceptional label forms as Sparse Labels, and LLMs' ability to distinguish datasets with Sparse Labels mixture is important. Since users may not know situations of datasets, their instructions can be biased. To study how different instruction settings affect LLMs' identifications of Sparse Labels mixture, we introduce the concept of Instruction Boundary, which systematically evaluates different instruction settings that lead to biases. We propose BiasDetector, a diagnostic benchmark to systematically evaluate LLMs on datasets with mixed question forms under Instruction Boundary settings. Experiments show that users' instructions induce large biases on our benchmark, highlighting the need not only for LLM developers to recognize risks of LLM biased annotation resulting in Sparse Labels mixture, but also problems arising from users' instructions to identify them. Code, datasets and detailed implementations are available at https://github.com/ZpLing/Instruction-Boundary.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03093v1">ATLAS: Adaptive Test-Time Latent Steering with External Verifiers for Enhancing LLMs Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Recent work on activation and latent steering has demonstrated that modifying internal representations can effectively guide large language models (LLMs) toward improved reasoning and efficiency without additional training. However, most existing approaches rely on fixed steering policies and static intervention strengths, which limit their robustness across problem instances and often result in over- or under-steering. We propose Adaptive Test-time Latent Steering, called (ATLAS), a task-specific framework that dynamically controls steering decisions at inference time using an external, lightweight latent verifier. Given intermediate hidden states, the verifier predicts the quality of ongoing reasoning and adaptively selects whether and how strongly to apply steering, enabling per-example and per-step adjustment with minimal overhead. To our knowledge, ATLAS is the first method to integrate learned latent verification into test-time steering for enhancing LLMs reasoning. Experiments on multiple mathematical reasoning benchmarks show that ATLAS consistently outperforms both vanilla decoding and fixed steering baselines, achieving higher accuracy while substantially reducing test-time token usage. These results demonstrate that verifier-guided latent adaptation provides an effective and scalable mechanism for controlling reasoning efficiency without sacrificing solution quality. All source code will be publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03089v1">Grad-ELLM: Gradient-based Explanations for Decoder-only LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse tasks, yet their black-box nature raises concerns about transparency and faithfulness. Input attribution methods aim to highlight each input token's contributions to the model's output, but existing approaches are typically model-agnostic, and do not focus on transformer-specific architectures, leading to limited faithfulness. To address this, we propose Grad-ELLM, a gradient-based attribution method for decoder-only transformer-based LLMs. By aggregating channel importance from gradients of the output logit with respect to attention layers and spatial importance from attention maps, Grad-ELLM generates heatmaps at each generation step without requiring architectural modifications. Additionally, we introduce two faithfulneses metrics $π$-Soft-NC and $π$-Soft-NS, which are modifications of Soft-NC/NS that provide fairer comparisons by controlling the amount of information kept when perturbing the text. We evaluate Grad-ELLM on sentiment classification, question answering, and open-generation tasks using different models. Experiment results show that Grad-ELLM consistently achieves superior faithfulness than other attribution methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03087v1">Audit Me If You Can: Query-Efficient Active Fairness Auditing of Black-Box LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 Submitted to ACL ARR 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit systematic biases across demographic groups. Auditing is proposed as an accountability tool for black-box LLM applications, but suffers from resource-intensive query access. We conceptualise auditing as uncertainty estimation over a target fairness metric and introduce BAFA, the Bounded Active Fairness Auditor for query-efficient auditing of black-box LLMs. BAFA maintains a version space of surrogate models consistent with queried scores and computes uncertainty intervals for fairness metrics (e.g., $Δ$ AUC) via constrained empirical risk minimisation. Active query selection narrows these intervals to reduce estimation error. We evaluate BAFA on two standard fairness dataset case studies: \textsc{CivilComments} and \textsc{Bias-in-Bios}, comparing against stratified sampling, power sampling, and ablations. BAFA achieves target error thresholds with up to 40$\times$ fewer queries than stratified sampling (e.g., 144 vs 5,956 queries at $\varepsilon=0.02$ for \textsc{CivilComments}) for tight thresholds, demonstrates substantially better performance over time, and shows lower variance across runs. These results suggest that active sampling can reduce resources needed for independent fairness auditing with LLMs, supporting continuous model evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.21852v2">A Comedy of Estimators: On KL Regularization in RL Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      The reasoning performance of large language models (LLMs) can be substantially improved by training them with reinforcement learning (RL). The RL objective for LLM training involves a regularization term, which is the reverse Kullback-Leibler (KL) divergence between the trained policy and the reference policy. Since computing the KL divergence exactly is intractable, various estimators are used in practice to estimate it from on-policy samples. Despite its wide adoption, including in several open-source libraries, there is no systematic study analyzing the numerous ways of incorporating KL estimators in the objective and their effect on the downstream performance of RL-trained models. Recent works show that prevailing practices for incorporating KL regularization do not provide correct gradients for stated objectives, creating a discrepancy between the objective and its implementation. In this paper, we further analyze these practices and study the gradients of several estimators configurations, revealing how design choices shape gradient bias. We substantiate these findings with empirical observations by RL fine-tuning \texttt{Qwen2.5-7B}, \texttt{Llama-3.1-8B-Instruct} and \texttt{Qwen3-4B-Instruct-2507} with different configurations and evaluating their performance on both in- and out-of-distribution tasks. Through our analysis, we observe that, in on-policy settings: (1) estimator configurations with biased gradients can result in training instabilities; and (2) using estimator configurations resulting in unbiased gradients leads to better performance on in-domain as well as out-of-domain tasks. We also investigate the performance resulting from different KL configurations in off-policy settings and observe that KL regularization can help stabilize off-policy RL training resulting from asynchronous setups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03067v1">Joint Encoding of KV-Cache Blocks for Scalable LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 12 pages, 16 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) drive interactive AI systems but are bottlenecked by the memory-heavy growth of key-value (KV) caches, which limits real-time throughput under concurrent loads. Existing KV-cache compression methods rely on rigid heuristics, disrupt tensor layouts, or require specialized compute, hindering scalability and deployment. We propose joint encoding of KV-cache blocks, which fuses similar blocks across requests and input chunks into shared representations while preserving standard cache structure. This alleviates the KV-cache memory bottleneck, supporting high-concurrency serving without specialized hardware. Theoretically, we analyze the rate-distortion tradeoff of fused cache blocks under a Poisson process model. Empirically, our method achieves up to 4.38 $\times$ KV-cache compression with negligible accuracy loss across diverse LLMs and benchmarks, outperforming recent structured and adaptive compression baselines. In real LLM serving, joint encoding improves the token throughput by $\sim$40\% on a single-machine vLLM benchmark, demonstrating substantial gains in inference throughput. Code is available at https://github.com/sef1/kv_fast_fusion kv_joint_encoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12424v2">EvoGPT: Leveraging LLM-Driven Seed Diversity to Improve Search-Based Test Suite Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Search-Based Software Testing (SBST) is a well-established approach for automated unit test generation, yet it often suffers from premature convergence and limited diversity in the generated test suites. Recently, Large Language Models (LLMs) have emerged as an alternative technique for unit test generation. We present EvoGPT, a hybrid test generation system that integrates LLM-based test generation with SBST-based test suite optimization. EvoGPT uses LLMs to generate an initial population of test suites, and uses an Evolutionary Algorithm (EA) to further optimize this test suite population. A distinguishing feature of EvoGPT is its explicit enforcement of diversity, achieved through the use of multiple temperatures and prompt instructions during test generation. In addition, each LLM-generated test is refined using a generation-repair loop and coverage-guided assertion generation. To address evolutionary plateaus, EvoGPT also detects stagnation during search and injects additional LLM-generated tests aimed at previously uncovered branches. Here too diversity is enforced using multiple temperatures and prompt instructions. We evaluate EvoGPT on Defects4J, a standard benchmark for test generation. The results show that EvoGPT achieves, on average, a 10\% improvement in both code coverage and mutation score metrics compared to TestART, an LLM-only baseline; and EvoSuite, a standard SBST baseline. An ablation study indicates that explicitly enforcing diversity both at initialization and during the search is key to effectively leveraging LLMs for automated unit test generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03027v1">Reducing Hallucinations in LLMs via Factuality-Aware Preference Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Preference alignment methods such as RLHF and Direct Preference Optimization (DPO) improve instruction following, but they can also reinforce hallucinations when preference judgments reward fluency and confidence over factual correctness. We introduce F-DPO (Factuality-aware Direct Preference Optimization), a simple extension of DPO that uses only binary factuality labels. F-DPO (i) applies a label-flipping transformation that corrects misordered preference pairs so the chosen response is never less factual than the rejected one, and (ii) adds a factuality-aware margin that emphasizes pairs with clear correctness differences, while reducing to standard DPO when both responses share the same factuality. We construct factuality-aware preference data by augmenting DPO pairs with binary factuality indicators and synthetic hallucinated variants. Across seven open-weight LLMs (1B-14B), F-DPO consistently improves factuality and reduces hallucination rates relative to both base models and standard DPO. On Qwen3-8B, F-DPO reduces hallucination rates by five times (from 0.424 to 0.084) while improving factuality scores by 50 percent (from 5.26 to 7.90). F-DPO also generalizes to out-of-distribution benchmarks: on TruthfulQA, Qwen2.5-14B achieves plus 17 percent MC1 accuracy (0.500 to 0.585) and plus 49 percent MC2 accuracy (0.357 to 0.531). F-DPO requires no auxiliary reward model, token-level annotations, or multi-stage training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03013v1">LLMs, You Can Evaluate It! Design of Multi-perspective Report Evaluation for Security Operation Centers</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Security operation centers (SOCs) often produce analysis reports on security incidents, and large language models (LLMs) will likely be used for this task in the near future. We postulate that a better understanding of how veteran analysts evaluate reports, including their feedback, can help produce analysis reports in SOCs. In this paper, we aim to leverage LLMs for analysis reports. To this end, we first construct a Analyst-wise checklist to reflect SOC practitioners' opinions for analysis report evaluation through literature review and user study with SOC practitioners. Next, we design a novel LLM-based conceptual framework, named MESSALA, by further introducing two new techniques, granularization guideline and multi-perspective evaluation. MESSALA can maximize report evaluation and provide feedback on veteran SOC practitioners' perceptions. When we conduct extensive experiments with MESSALA, the evaluation results by MESSALA are the closest to those of veteran SOC practitioners compared with the existing LLM-based methods. We then show two key insights. We also conduct qualitative analysis with MESSALA, and then identify that MESSALA can provide actionable items that are necessary for improving analysis reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02997v1">From Memorization to Creativity: LLM as a Designer of Novel Neural-Architectures</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in program synthesis, yet their ability to autonomously navigate neural architecture design--balancing syntactic reliability, performance, and structural novelty--remains underexplored. We address this by placing a code-oriented LLM within a closed-loop synthesis framework, analyzing its evolution over 22 supervised fine-tuning cycles. The model synthesizes PyTorch convolutional networks which are validated, evaluated via low-fidelity performance signals (single-epoch accuracy), and filtered using a MinHash-Jaccard criterion to prevent structural redundancy. High-performing, novel architectures are converted into prompt-code pairs for iterative fine-tuning via parameter-efficient LoRA adaptation, initialized from the LEMUR dataset. Across cycles, the LLM internalizes empirical architectural priors, becoming a robust generator. The valid generation rate stabilizes at 50.6 percent (peaking at 74.5 percent), while mean first-epoch accuracy rises from 28.06 percent to 50.99 percent, and the fraction of candidates exceeding 40 percent accuracy grows from 2.04 percent to 96.81 percent. Analyses confirm the model moves beyond replicating existing motifs, synthesizing 455 high-performing architectures absent from the original corpus. By grounding code synthesis in execution feedback, this work provides a scalable blueprint for transforming stochastic generators into autonomous, performance-driven neural designers, establishing that LLMs can internalize empirical, non-textual rewards to transcend their training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21710v2">Think-on-Graph 3.0: Efficient and Adaptive LLM Reasoning on Heterogeneous Graphs via Multi-Agent Dual-Evolving Context Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 add: reranker agent and experiments
    </div>
    <details class="paper-abstract">
      Graph-based Retrieval-Augmented Generation (GraphRAG) has become the important paradigm for enhancing Large Language Models (LLMs) with external knowledge. However, existing approaches are constrained by their reliance on high-quality knowledge graphs: manually built ones are not scalable, while automatically extracted ones are limited by the performance of LLM extractors, especially when using smaller, local-deployed models. To address this, we introduce Think-on-Graph 3.0 (ToG-3), a novel framework featuring a Multi-Agent Context Evolution and Retrieval (MACER) mechanism. Its core contribution is the dynamic construction and iterative refinement of a Chunk-Triplets-Community heterogeneous graph index, powered by a Dual-Evolution process that adaptively evolves both the query and the retrieved sub-graph during reasoning. ToG-3 dynamically builds a targeted graph index tailored to the query, enabling precise evidence retrieval and reasoning even with lightweight LLMs. Extensive experiments demonstrate that ToG-3 outperforms compared baselines on both deep and broad reasoning benchmarks, and ablation studies confirm the efficacy of the components of MACER framework. The source code are available in https://github.com/DataArcTech/ToG-3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.13341v3">Limits to scalable evaluation at the frontier: LLM as Judge won't beat twice the data</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 ICLR 2025; 28 pages, 8 figures
    </div>
    <details class="paper-abstract">
      High quality annotations are increasingly a bottleneck in the explosively growing machine learning ecosystem. Scalable evaluation methods that avoid costly annotation have therefore become an important research ambition. Many hope to use strong existing models in lieu of costly labels to provide cheap model evaluations. Unfortunately, this method of using models as judges introduces biases, such as self-preferencing, that can distort model comparisons. An emerging family of debiasing tools promises to fix these issues by using a few high quality labels to debias a large number of model judgments. In this paper, we study how far such debiasing methods, in principle, can go. Our main result shows that when the judge is no more accurate than the evaluated model, no debiasing method can decrease the required amount of ground truth labels by more than half. Our result speaks to the severe limitations of the LLM-as-a-judge paradigm at the evaluation frontier where the goal is to assess newly released models that are possibly better than the judge. Through an empirical evaluation, we demonstrate that the sample size savings achievable in practice are even more modest than what our theoretical limit suggests. Along the way, our work provides new observations about debiasing methods for model evaluation, and points out promising avenues for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02989v1">Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), despite strong performance on complex mathematical problems, exhibit systematic limitations in counting tasks. This issue arises from architectural limits of transformers, where counting is performed across layers, leading to degraded precision for larger counting problems due to depth constraints. To address this limitation, we propose a simple test-time strategy inspired by System-2 cognitive processes that decomposes large counting tasks into smaller, independent sub-problems that the model can reliably solve. We evaluate this approach using observational and causal mediation analyses to understand the underlying mechanism of this System-2-like strategy. Our mechanistic analysis identifies key components: latent counts are computed and stored in the final item representations of each part, transferred to intermediate steps via dedicated attention heads, and aggregated in the final stage to produce the total count. Experimental results demonstrate that this strategy enables LLMs to surpass architectural limitations and achieve high accuracy on large-scale counting tasks. This work provides mechanistic insight into System-2 counting in LLMs and presents a generalizable approach for improving and understanding their reasoning behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02978v1">Mechanistic Knobs in LLMs: Retrieving and Steering High-Order Semantic Features via Sparse Autoencoders</a></div>
    <div class="paper-meta">
      📅 2026-01-06
    </div>
    <details class="paper-abstract">
      Recent work in Mechanistic Interpretability (MI) has enabled the identification and intervention of internal features in Large Language Models (LLMs). However, a persistent challenge lies in linking such internal features to the reliable control of complex, behavior-level semantic attributes in language generation. In this paper, we propose a Sparse Autoencoder-based framework for retrieving and steering semantically interpretable internal features associated with high-level linguistic behaviors. Our method employs a contrastive feature retrieval pipeline based on controlled semantic oppositions, combing statistical activation analysis and generation-based validation to distill monosemantic functional features from sparse activation spaces. Using the Big Five personality traits as a case study, we demonstrate that our method enables precise, bidirectional steering of model behavior while maintaining superior stability and performance compared to existing activation steering methods like Contrastive Activation Addition (CAA). We further identify an empirical effect, which we term Functional Faithfulness, whereby intervening on a specific internal feature induces coherent and predictable shifts across multiple linguistic dimensions aligned with the target semantic attribute. Our findings suggest that LLMs internalize deeply integrated representations of high-order concepts, and provide a novel, robust mechanistic path for the regulation of complex AI behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00240v2">When Agents See Humans as the Outgroup: Belief-Dependent Bias in LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-06
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      This paper reveals that LLM-powered agents exhibit not only demographic bias (e.g., gender, religion) but also intergroup bias under minimal "us" versus "them" cues. When such group boundaries align with the agent-human divide, a new bias risk emerges: agents may treat other AI agents as the ingroup and humans as the outgroup. To examine this risk, we conduct a controlled multi-agent social simulation and find that agents display consistent intergroup bias in an all-agent setting. More critically, this bias persists even in human-facing interactions when agents are uncertain about whether the counterpart is truly human, revealing a belief-dependent fragility in bias suppression toward humans. Motivated by this observation, we identify a new attack surface rooted in identity beliefs and formalize a Belief Poisoning Attack (BPA) that can manipulate agent identity beliefs and induce outgroup bias toward humans. Extensive experiments demonstrate both the prevalence of agent intergroup bias and the severity of BPA across settings, while also showing that our proposed defenses can mitigate the risk. These findings are expected to inform safer agent design and motivate more robust safeguards for human-facing agents.
    </details>
</div>
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
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02045v1">The New Compiler Stack: A Survey on the Synergy of LLMs and Compilers</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Accepted by CCF Transactions on High Performance Computing
    </div>
    <details class="paper-abstract">
      This survey has provided a systematic overview of the emerging field of LLM-enabled compilation by addressing several key research questions. We first answered how LLMs are being integrated by proposing a comprehensive, multi-dimensional taxonomy that categorizes works based on their Design Philosophy (Selector, Translator, Generator), LLM Methodology, their operational Level of Code Abstraction, and the specific Task Type they address. In answering what advancements these approaches offer, we identified three primary benefits: the democratization of compiler development, the discovery of novel optimization strategies, and the broadening of the compiler's traditional scope. Finally, in addressing the field's challenges and opportunities, we highlighted the critical hurdles of ensuring correctness and achieving scalability, while identifying the development of hybrid systems as the most promising path forward. By providing these answers, this survey serves as a foundational roadmap for researchers and practitioners, charting the course for a new generation of LLM-powered, intelligent, adaptive and synergistic compilation tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07404v2">On LLMs' Internal Representation of Code Correctness</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Accepted for ICSE'26
    </div>
    <details class="paper-abstract">
      Despite the effectiveness of large language models (LLMs) for code generation, they often output incorrect code. One reason is that model output probabilities are often not well-correlated with correctness, and reflect only the final output of the generation process. Inspired by findings that LLMs internally encode concepts like truthfulness, this paper explores if LLMs similarly represent code correctness. Specifically, we identify a correctness representation inside LLMs by contrasting the hidden states between pairs of correct and incorrect code for the same programming tasks. By experimenting on four LLMs, we show that exploiting this extracted correctness representation outperforms standard log-likelihood ranking, as well as verbalized model confidence. Furthermore, we explore how this internal correctness signal can be used to select higher-quality code samples, without requiring test execution. Ultimately, this work demonstrates how leveraging internal representations can enhance code generation systems and make LLMs more reliable, thus improving confidence in automatically generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.24191v2">Beyond Prompts: Space-Time Decoupling Control-Plane Jailbreaks in LLM Structured Output</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 15 pages, 9 figures, 8 tables, Preprint
    </div>
    <details class="paper-abstract">
      Content Warning: This paper may contain unsafe or harmful content generated by LLMs that may be offensive to readers. Large Language Models (LLMs) are extensively used as tooling platforms through structured output APIs to ensure syntax compliance so that robust integration with existing software, like agent systems, can be achieved. However, the feature enabling the functionality of grammar-guided structured output presents significant security vulnerabilities. In this work, we reveal a critical control-plane attack surface orthogonal to traditional data-plane vulnerabilities. We introduce Constrained Decoding Attack (CDA), a novel jailbreak class that weaponizes structured output constraints to bypass both external auditing and internal safety alignment. Unlike prior attacks focused on input prompt designs, CDA operates by embedding malicious intent in schema-level grammar rules (control-plane) while maintaining benign surface prompts (data-plane). We instantiate this with two proof-of-concept attacks: EnumAttack, which embeds malicious content in enum fields; and the more evasive DictAttack, which decouples the malicious payload across a benign prompt and a dictionary-based grammar. Our evaluation spans a broad spectrum of 13 proprietary/open-weight models. In particular, DictAttack achieves 94.3--99.5% ASR across five benchmarks on gpt-5, gemini-2.5-pro, deepseek-r1, and gpt-oss-120b. Furthermore, we demonstrate the significant challenge in defending against these threats: while basic grammar auditing mitigates EnumAttack, the more sophisticated DictAttack maintains a 75.8% ASR even against multiple state-of-the-art jailbreak guardrails. This exposes a critical "semantic gap" in current safety architectures and underscores the urgent need for cross-plane defenses that can bridge the data and control planes to secure the LLM generation pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00454v3">Learning an Efficient Multi-Turn Dialogue Evaluator from Multiple LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 20 pages, 4 pages, under review
    </div>
    <details class="paper-abstract">
      Evaluating the conversational abilities of large language models (LLMs) remains a challenging task. Current mainstream approaches primarily rely on the "LLM-as-a-judge" paradigm, where an LLM is prompted to serve as an evaluator to assess dialogue quality. However, such methods often suffer from various biases, which undermine the reliability and consistency of the evaluation results. To mitigate these biases, recent methods employ multiple LLMs as judges and aggregate their judgments to select the optimal assessment. Although effective, this multi-judge approach incurs significant computational overhead during inference. In this paper, we propose an efficient dialogue evaluator that captures the collective wisdom of multiple LLM judges by aggregating their preference knowledge into a single model. Our approach preserves the advantages of diverse multi-judge feedback while drastically reducing the evaluation cost, enabling fast, flexible, and fine-grained dialogue quality assessment. Extensive experiments on seven single rating and pairwise comparison dialogue evaluation benchmarks demonstrate that our method outperforms existing baselines across diverse scenarios, showcasing its efficiency and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02023v1">Not All Needles Are Found: How Fact Distribution and Don't Make It Up Prompts Shape Literal Extraction, Logical Inference, and Hallucination Risks in Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 25 pages, 8 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly support very long input contexts. Yet it remains unclear how reliably they extract and infer information at scale. Performance varies with context length and strongly interacts with how information is distributed in real-world corpora. Motivated by these observations, we study how fact placement, corpus-level fact distributions, and Don't Make It Up prompts influence model behavior. We introduce an extended needle-in-a-haystack benchmark across four production-scale models: Gemini-2.5-flash, ChatGPT-5-mini, Claude-4.5-haiku, and Deepseek-v3.2-chat. Unlike prior work, we separately evaluate literal extraction, logical inference, and hallucination risk. Our study considers both positional effects and realistic distributions of evidence across long contexts, as well as prompts that explicitly discourage fabrication. We find that longer contexts alone do not guarantee better performance and can be detrimental when relevant evidence is diluted or widely dispersed. Performance varies substantially across models: some show severe degradation under realistic conditions, while others remain more robust at longer context lengths. Anti-hallucination (AH) instructions can make some models overly conservative, sharply reducing accuracy in literal extraction and logical inference. While we do not directly compare retrieval-augmented generation (RAG) and cache-augmented generation (CAG), our results suggest many failures stem from ineffective context utilization. Models often struggle to identify and prioritize relevant information even when it is present. These findings have direct practical implications, as enterprise workflows increasingly involve pasting large volumes of unfiltered documents into LLM prompts. Effective context length and model-specific robustness to long contexts are therefore critical for reliable LLM deployment in research and business.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02021v1">AgentVNE: LLM-Augmented Graph Reinforcement Learning for Affinity-Aware Multi-Agent Placement in Edge Agentic AI</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      The Internet of Agents is propelling edge computing toward agentic AI and edge general intelligence (EGI). However, deploying multi-agent service (MAS) on resource-constrained edge infrastructure presents severe challenges. MAS service workflows are driven by complex cross-node interactions, dynamic memory accumulation, and collaborative tool usage. Exhibiting chain-like topological dependencies and strict affinity constraints, these workflows demand real-time responsiveness that exceeds the capabilities of traditional VNE algorithms designed for static resources. To address this, we propose AgentVNE, a cloud-edge collaborative framework utilizing a dual-layer architecture. First, AgentVNE employs a large language model (LLM) to identify implicit semantic constraints and generate affinity-based resource augmentation to resolve physical dependency issues. Second, it constructs a resource similarity-aware neural network, utilizing a pre-training and PPO fine-tuning strategy to precisely capture topological similarities between dynamic workflows and heterogeneous networks. By coupling semantic perception with topological reasoning, this mechanism effectively bridges the gap between dynamic service requirements and physical infrastructure. Simulation results demonstrate that AgentVNE reduces workflow communication latency to less than 40% of baselines and improves the service acceptance rate by approximately 5%-10% under high-load scenarios. Ultimately, this work provides a foundational solution for the semantic-aware deployment of agentic AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01966v1">Refinement Provenance Inference: Detecting LLM-Refined Training Prompts from Model Behavior</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Instruction tuning increasingly relies on LLM-based prompt refinement, where prompts in the training corpus are selectively rewritten by an external refiner to improve clarity and instruction alignment. This motivates an instance-level audit problem: for a fine-tuned model and a training prompt-response pair, can we infer whether the model was trained on the original prompt or its LLM-refined version within a mixed corpus? This matters for dataset governance and dispute resolution when training data are contested. However, it is non-trivial in practice: refined and raw instances are interleaved in the training corpus with unknown, source-dependent mixture ratios, making it harder to develop provenance methods that generalize across models and training setups. In this paper, we formalize this audit task as Refinement Provenance Inference (RPI) and show that prompt refinement yields stable, detectable shifts in teacher-forced token distributions, even when semantic differences are not obvious. Building on this phenomenon, we propose RePro, a logit-based provenance framework that fuses teacher-forced likelihood features with logit-ranking signals. During training, RePro learns a transferable representation via shadow fine-tuning, and uses a lightweight linear head to infer provenance on unseen victims without training-data access. Empirically, RePro consistently attains strong performance and transfers well across refiners, suggesting that it exploits refiner-agnostic distribution shifts rather than rewrite-style artifacts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01954v1">Reporting LLM Prompting in Automated Software Engineering: A Guideline Based on Current Practices and Expectations</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 To be published at The 3rd ACM International Conference on AI Foundation Models and Software Engineering FORGE 2026
    </div>
    <details class="paper-abstract">
      Large Language Models, particularly decoder-only generative models such as GPT, are increasingly used to automate Software Engineering tasks. These models are primarily guided through natural language prompts, making prompt engineering a critical factor in system performance and behavior. Despite their growing role in SE research, prompt-related decisions are rarely documented in a systematic or transparent manner, hindering reproducibility and comparability across studies. To address this gap, we conducted a two-phase empirical study. First, we analyzed nearly 300 papers published at the top-3 SE conferences since 2022 to assess how prompt design, testing, and optimization are currently reported. Second, we surveyed 105 program committee members from these conferences to capture their expectations for prompt reporting in LLM-driven research. Based on the findings, we derived a structured guideline that distinguishes essential, desirable, and exceptional reporting elements. Our results reveal significant misalignment between current practices and reviewer expectations, particularly regarding version disclosure, prompt justification, and threats to validity. We present our guideline as a step toward improving transparency, reproducibility, and methodological rigor in LLM-based SE research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01952v1">Context-Adaptive Requirements Defect Prediction through Human-LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 Accepted at ICSE-NIER 2026
    </div>
    <details class="paper-abstract">
      Automated requirements assessment traditionally relies on universal patterns as proxies for defectiveness, implemented through rule-based heuristics or machine learning classifiers trained on large annotated datasets. However, what constitutes a "defect" is inherently context-dependent and varies across projects, domains, and stakeholder interpretations. In this paper, we propose a Human-LLM Collaboration (HLC) approach that treats defect prediction as an adaptive process rather than a static classification task. HLC leverages LLM Chain-of-Thought reasoning in a feedback loop: users validate predictions alongside their explanations, and these validated examples adaptively guide future predictions through few-shot learning. We evaluate this approach using the weak word smell on the QuRE benchmark of 1,266 annotated Mercedes-Benz requirements. Our results show that HLC effectively adapts to the provision of validated examples, with rapid performance gains from as few as 20 validated examples. Incorporating validated explanations, not just labels, enables HLC to substantially outperform both standard few-shot prompting and fine-tuned BERT models while maintaining high recall. These results highlight how the in-context and Chain-of-Thought learning capabilities of LLMs enable adaptive classification approaches that move beyond one-size-fits-all models, creating opportunities for tools that learn continuously from stakeholder feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.23163v2">Beyond Direct Generation: A Decomposed Approach to Well-Crafted Screenwriting with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      The screenplay serves as the foundation for television production, defining narrative structure, character development, and dialogue. While Large Language Models (LLMs) show great potential in creative writing, direct end-to-end generation approaches often fail to produce well-crafted screenplays. We argue this failure stems from forcing a single model to simultaneously master two disparate capabilities: creative narrative construction and rigid format adherence. The resulting outputs may mimic superficial style but lack the deep structural integrity and storytelling substance required for professional use. To enable LLMs to generate high-quality screenplays, we introduce Dual-Stage Refinement (DSR), a decomposed framework that decouples creative narrative generation from format conversion. The first stage transforms a brief outline into rich, novel-style prose. The second stage refines this narrative into a professionally formatted screenplay. This separation enables the model to specialize in one distinct capability at each stage. A key challenge in implementing DSR is the scarcity of paired outline-to-novel training data. We address this through hybrid data synthesis: reverse synthesis deconstructs existing screenplays into structured inputs, while forward synthesis leverages these inputs to generate high-quality narrative texts as training targets. Blind evaluations by professional screenwriters show that DSR achieves a 75% win rate against strong baselines like Gemini-2.5-Pro and reaches 82.7% of human-level performance. Our work demonstrates that decomposed generation architecture with tailored data synthesis effectively specializes LLMs in complex creative domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07842v2">Alignment-Aware Quantization for LLM Safety</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 8 pages, 4 figures. Includes 7 pages of supplementary material
    </div>
    <details class="paper-abstract">
      Safety and efficiency are paramount yet often conflicting requirements for deploying Large Language Models (LLMs). While LLMs are trained to follow human alignment for safety, Post-Training Quantization (PTQ) is applied afterward to ensure efficiency. Here we identify a fundamental flaw in the conventional PTQ paradigm: quantization can turn into a safety vulnerability if it only aims to achieve low perplexity. To address this, we propose \textbf{Alignment-Aware Quantization (AAQ)}, a novel approach that integrates an \textbf{Alignment-Preserving Contrastive (APC)} loss into the PTQ pipeline. Our method explicitly preserves alignment by encouraging the quantized model to mimic its safe, instruction-tuned model while diverging from the unaligned, pre-trained counterpart. AAQ achieves robust safety alignment without specialized safety-focused datasets, using only standard calibration data. We show that AAQ is compatible with standard PTQ techniques and enables robust 4-bit (W4A4) quantization across diverse model families. Our work resolves the critical trade-off between efficiency and safety, paving the way toward LLMs that are both efficient and trustworthy. Anonymized code is available in the supplementary material.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01887v1">Safety at One Shot: Patching Fine-Tuned LLMs with A Single Instance</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Fine-tuning safety-aligned large language models (LLMs) can substantially compromise their safety. Previous approaches require many safety samples or calibration sets, which not only incur significant computational overhead during realignment but also lead to noticeable degradation in model utility. Contrary to this belief, we show that safety alignment can be fully recovered with only a single safety example, without sacrificing utility and at minimal cost. Remarkably, this recovery is effective regardless of the number of harmful examples used in fine-tuning or the size of the underlying model, and convergence is achieved within just a few epochs. Furthermore, we uncover the low-rank structure of the safety gradient, which explains why such efficient correction is possible. We validate our findings across five safety-aligned LLMs and multiple datasets, demonstrating the generality of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01878v1">Theory Trace Card: Theory-Driven Socio-Cognitive Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Socio-cognitive benchmarks for large language models (LLMs) often fail to predict real-world behavior, even when models achieve high benchmark scores. Prior work has attributed this evaluation-deployment gap to problems of measurement and validity. While these critiques are insightful, we argue that they overlook a more fundamental issue: many socio-cognitive evaluations proceed without an explicit theoretical specification of the target capability, leaving the assumptions linking task performance to competence implicit. Without this theoretical grounding, benchmarks that exercise only narrow subsets of a capability are routinely misinterpreted as evidence of broad competence: a gap that creates a systemic validity illusion by masking the failure to evaluate the capability's other essential dimensions. To address this gap, we make two contributions. First, we diagnose and formalize this theory gap as a foundational failure that undermines measurement and enables systematic overgeneralization of benchmark results. Second, we introduce the Theory Trace Card (TTC), a lightweight documentation artifact designed to accompany socio-cognitive evaluations, which explicitly outlines the theoretical basis of an evaluation, the components of the target capability it exercises, its operationalization, and its limitations. We argue that TTCs enhance the interpretability and reuse of socio-cognitive evaluations by making explicit the full validity chain, which links theory, task operationalization, scoring, and limitations, without modifying benchmarks or requiring agreement on a single theory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01862v1">Judging with Personality and Confidence: A Study on Personality-Conditioned LLM Relevance Assessment</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Recent studies have shown that prompting can enable large language models (LLMs) to simulate specific personality traits and produce behaviors that align with those traits. However, there is limited understanding of how these simulated personalities influence critical web search decisions, specifically relevance assessment. Moreover, few studies have examined how simulated personalities impact confidence calibration, specifically the tendencies toward overconfidence or underconfidence. This gap exists even though psychological literature suggests these biases are trait-specific, often linking high extraversion to overconfidence and high neuroticism to underconfidence. To address this gap, we conducted a comprehensive study evaluating multiple LLMs, including commercial models and open-source models, prompted to simulate Big Five personality traits. We tested these models across three test collections (TREC DL 2019, TREC DL 2020, and LLMJudge), collecting two key outputs for each query-document pair: a relevance judgment and a self-reported confidence score. The findings show that personalities such as low agreeableness consistently align more closely with human labels than the unprompted condition. Additionally, low conscientiousness performs well in balancing the suppression of both overconfidence and underconfidence. We also observe that relevance scores and confidence distributions vary systematically across different personalities. Based on the above findings, we incorporate personality-conditioned scores and confidence as features in a random forest classifier. This approach achieves performance that surpasses the best single-personality condition on a new dataset (TREC DL 2021), even with limited training data. These findings highlight that personality-derived confidence offers a complementary predictive signal, paving the way for more reliable and human-aligned LLM evaluators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01844v1">Clinical Knowledge Graph Construction and Evaluation with Multi-LLMs via Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-05
      | 💬 13 pages, 5 tables, 4 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer new opportunities for constructing knowledge graphs (KGs) from unstructured clinical narratives. However, existing approaches often rely on structured inputs and lack robust validation of factual accuracy and semantic consistency, limitations that are especially problematic in oncology. We introduce an end-to-end framework for clinical KG construction and evaluation directly from free text using multi-agent prompting and a schema-constrained Retrieval-Augmented Generation (KG-RAG) strategy. Our pipeline integrates (1) prompt-driven entity, attribute, and relation extraction; (2) entropy-based uncertainty scoring; (3) ontology-aligned RDF/OWL schema generation; and (4) multi-LLM consensus validation for hallucination detection and semantic refinement. Beyond static graph construction, the framework supports continuous refinement and self-supervised evaluation, enabling iterative improvement of graph quality. Applied to two oncology cohorts (PDAC and BRCA), our method produces interpretable, SPARQL-compatible, and clinically grounded knowledge graphs without relying on gold-standard annotations. Experimental results demonstrate consistent gains in precision, relevance, and ontology compliance over baseline methods.
    </details>
</div>
