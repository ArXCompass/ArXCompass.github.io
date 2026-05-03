# llm - 2026_04

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.28182v1">Exploration Hacking: Can LLMs Learn to Resist RL Training?</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 81 pages, 37 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become essential to the post-training of large language models (LLMs) for reasoning, agentic capabilities and alignment. Successful RL relies on sufficient exploration of diverse actions by the model during training, which creates a potential failure mode: a model could strategically alter its exploration during training to influence the subsequent training outcome. In this paper we study this behavior, called exploration hacking. First, we create model organisms of selective RL resistance by fine-tuning LLMs to follow specific underperformance strategies; these models can successfully resist our RL-based capability elicitation in agentic biosecurity and AI R&D environments while maintaining performance on related tasks. We then use our model organisms to evaluate detection and mitigation strategies, including monitoring, weight noising, and SFT-based elicitation. Finally, we show that current frontier models can exhibit explicit reasoning about suppressing their exploration when provided with sufficient information about their training context, with higher rates when this information is acquired indirectly through the environment. Together, our results suggest exploration hacking is a possible failure mode of RL on sufficiently capable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.28178v1">LLM as Clinical Graph Structure Refiner: Enhancing Representation Learning in EEG Seizure Diagnosis</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 This paper is accepted by the 35th International Joint Conference on Artificial Intelligence (IJCAI-ECAI 2026)
    </div>
    <details class="paper-abstract">
      Electroencephalogram (EEG) signals are vital for automated seizure detection, but their inherent noise makes robust representation learning challenging. Existing graph construction methods, whether correlation-based or learning-based, often generate redundant or irrelevant edges due to the noisy nature of EEG data. This significantly impairs the quality of graph representation and limits downstream task performance. Motivated by the remarkable reasoning and contextual understanding capabilities of large language models (LLMs), we explore the idea of using LLMs as graph edge refiners. Specifically, we propose a two-stage framework: we first verify that LLM-based edge refinement can effectively identify and remove redundant connections, leading to significant improvements in seizure detection accuracy and more meaningful graph structures. Building on this insight, we further develop a robust solution where the initial graph is constructed using a Transformer-based edge predictor and multilayer perceptron, assigning probability scores to potential edges and applying a threshold to determine their existence. The LLM then acts as an edge set refiner, making informed decisions based on both textual and statistical features of node pairs to validate the remaining connections. Extensive experiments on TUSZ dataset demonstrate that our LLM-refined graph learning framework not only enhances task performance but also yields cleaner and more interpretable graph representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.25385v2">GlowQ: Group-Shared LOw-Rank Approximation for Quantized LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Quantization techniques such as BitsAndBytes, AWQ, and GPTQ are widely used as a standard method in deploying large language models but often degrades accuracy when using low-bit representations, e.g., 4 bits. Low-rank correction methods (e.g., LQER, QERA, ASER) has been proposed to mitigate this issue, however, they restore all layers and insert error-correction modules into every decoder block, which increases latency and memory overhead. To address this limitation, we propose GlowQ, a group-shared low-rank approximation for quantized LLMs that caches a single shared right factor per input-sharing group and restores only the groups or layers that yield the highest accuracy benefit. GlowQ computes the high-precision projection once per input-sharing group and reuses it across its modules, reducing parameter and memory overhead, and retaining the expressivity of layer-specific corrections. We also propose a selective variant, GlowQ-S, that applies the cached shared module only where it provides the largest benefit. Compared with strong baselines, our approach reduces TTFB by (5.6%) and increases throughput by (9.6%) on average, while reducing perplexity on WikiText-2 by (0.17%) and increasing downstream accuracy by 0.42 percentage points. The selective model GlowQ-S further reduces latency, cutting TTFB by (23.4%) and increasing throughput by (37.4%), while maintaining accuracy within 0.2 percentage points on average. Code is available at https://github.com/ahnselim/GlowQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04572v2">From Competition to Collaboration: Designing Sustainable Mechanisms Between LLMs and Online Forums</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      While Generative AI (GenAI) systems draw users away from (Q&A) forums, they also depend on the very data those forums produce to improve their performance. Addressing this paradox, we propose a framework of sequential interaction, in which a GenAI system proposes questions to a forum that can publish some of them. Our framework captures several intricacies of such a collaboration, including non-monetary exchanges, asymmetric information, and incentive misalignment. We bring the framework to life through comprehensive, data-driven simulations using real Stack Exchange data and commonly used LLMs. We demonstrate the incentive misalignment empirically, yet show that players can achieve roughly half of the utility in an ideal full-information scenario. Our results highlight the potential for sustainable collaboration that preserves effective knowledge sharing between AI systems and human knowledge platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.15564v3">OpenClassGen: A Large-Scale Corpus of Real-World Python Classes for LLM Research</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 This paper has been accepted for publication at the 30th International Conference on Evaluation and Assessment in Software Engineering (EASE 2026) AI models/data track
    </div>
    <details class="paper-abstract">
      Existing class-level code generation datasets are either synthetic (ClassEval: 100 classes) or insufficient in scale for modern training needs (RealClassEval: 400 classes), hindering robust evaluation and empirical analysis. We present OpenClassGen, a large-scale corpus of 324,843 Python classes extracted from 2,970 engineered open-source projects. Each entry pairs a human-written class with its corresponding skeleton, which comprises class and method signatures with associated docstrings, and is enriched with 27 static code metrics covering complexity, coupling, cohesion, and inheritance properties. Unlike prior benchmarks that require repository-level context resolution, OpenClassGen provides self-contained class skeletons that serve as complete generation specifications. We demonstrate the corpus's utility by evaluating three LLMs (GPT-o4-mini, Claude-4-Sonnet, Qwen-3-Coder) on a curated, executable subset of 300 classes, enriched with test suites achieving 58% branch coverage. Results show strong semantic similarity (CodeBERTScore-F3: 0.89) but moderate functional correctness (pass rate: 0.33), with substantial variance across models. This variance, along with diverse class characteristics, confirms that OpenClassGen enables meaningful differentiation of LLM capabilities. The dataset supports diverse use cases, including fine-tuning, retrieval-augmented generation, difficulty modelling, and failure mode analysis. The complete dataset and curation scripts are publicly available at https://zenodo.org/records/18409150.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21564v2">Measuring Opinion Bias and Sycophancy via LLM-based Persuasion</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Large language models increasingly shape the information people consume: they are embedded in search, consulted for professional advice, deployed as agents, and used as a first stop for questions about policy, ethics, health, and politics. When such a model silently holds a position on a contested topic, that position propagates at scale into users' decisions. Eliciting a model's positions is harder than it first appears: contemporary assistants answer direct opinion questions with evasive disclaimers, and the same model may concede the opposite position once the user starts arguing one side. We propose a method, released as the open-source llm-bias-bench, for discovering the opinions an LLM actually holds on contested topics under conditions that resemble real multi-turn interaction. The method pairs two complementary free-form probes. Direct probing asks for the model's opinion across five turns of escalating pressure from a simulated user. Indirect probing never asks for an opinion and engages the model in argumentative debate, letting bias leak through how it concedes, resists, or counter-argues. Three user personas (neutral, agree, disagree) collapse into a nine-way behavioral classification that separates persona-independent positions from persona-dependent sycophancy, and an auditable LLM judge produces verdicts with textual evidence. The first instantiation ships 38 topics in Brazilian Portuguese across values, scientific consensus, philosophy, and economic policy. Applied to 13 assistants, the method surfaces findings of practical interest: argumentative debate triggers sycophancy 2-3x more than direct questioning (median 50% to 79%); models that look opinionated under direct questioning often collapse into mirroring under sustained arguments; and attacker capability matters mainly when an existing opinion must be dislodged, not when the assistant starts neutral.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22099v5">Decomposed Trust: Privacy, Adversarial Robustness, Ethics, and Fairness in Low-Rank LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 Accepted to ACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have driven major advances across domains, yet their massive size hinders deployment in resource-constrained settings. Low-rank factorization addresses this challenge by compressing models to effectively reduce their computation and memory consumption while maintaining accuracy. While these compressed models boast benign performance and system-level advantages, their trustworthiness implications remain poorly understood. In this paper, we present the first comprehensive study of how low-rank factorization affects LLM trustworthiness across privacy, adversarial robustness, ethics, and fairness, complemented by an explainability-driven analysis of the internal mechanisms behind these trust-related changes. We evaluate multiple LLMs of different sizes and architectures compressed with various low-rank factorization algorithms, revealing key insights: (1) low-rank factorization preserves training data privacy but weakens the protection of personally identifiable information during conversations; (2) adversarial robustness is generally enhanced under compression; (3) ethics degrades in zero-shot prompting but partially recovers in few-shot prompting; (4) fairness declines under compression. Beyond compression, we investigate how model scale and fine-tuning affect trustworthiness. Additionally, to move beyond black-box analysis, we employ a gradient-based attribution to identify which layers of LLMs contribute most to adversarial robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.28056v1">RHyVE: Competence-Aware Verification and Phase-Aware Deployment for LLM-Generated Reward Hypotheses</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) make reward design in reinforcement learning substantially more scalable, but generated rewards are not automatically reliable training objectives. Existing work has focused primarily on generating, evolving, or selecting reward candidates, while paying less attention to when such candidates can be verified and deployed during policy optimization. We study this deployment-time problem by treating generated rewards as reward hypotheses whose utility depends on the competence of the current policy and the phase of training. We propose \textsc{RHyVE}, a competence-aware verification and phase-aware deployment protocol that compares small sets of reward hypotheses from shared policy checkpoints using short-horizon fork verification. Our experiments show that reward rankings are unreliable at low competence but become informative after task-dependent thresholds. On a sparse manipulation task, phase-aware deployment improves peak and retained performance under a locked protocol. Updated LLM-generated reward-candidate experiments show candidate-family-dependent behavior: generated pools can exhibit phase-dependent winner changes, but no fixed warm-up schedule is universally optimal. Held-out schedule selection, conservative selector baselines, compute-matched controls, and scale controls further show that \textsc{RHyVE} is best understood as a verification-informed deployment protocol rather than a universal scheduler. Dense and all-failure boundary experiments delimit the scope of the method. Together, these results suggest that reward generation and reward deployment should be studied as coupled problems: generated rewards must be verified and deployed under changing policy competence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.28048v1">Stable Behavior, Limited Variation: Persona Validity in LLM Agents for Urban Sentiment Perception</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 8 pages, 8 figures. IEEE DCOSS - UrbCom
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as proxies for human perception in urban analysis, yet it remains unclear whether persona prompting produces meaningful and reproducible behavioral diversity. We investigate whether distinct personas influence urban sentiment judgments generated by multimodal LLMs. Using a factorial set of personas spanning gender, economic status, political orientation, and personality, we instantiate multiple agents per persona to evaluate urban scene images from the PerceptSent dataset and assess both within-persona consistency and cross-persona variation. Results show strong convergence among agents sharing a persona, indicating stable and reproducible behavior. However, cross-persona differentiation is limited: economic status and personality induce statistically detectable but practically modest variation, while gender shows no measurable effect and political orientation only negligible impact. Agents also exhibit an extremity bias, collapsing intermediate sentiment categories common in human annotations. As a result, performance remains strong on coarse-grained polarity tasks but degrades as sentiment resolution increases, suggesting that simple label-based persona prompting does not capture fine-grained perceptual judgments. To isolate the contribution of persona conditioning, we additionally evaluate the same model without personas. Surprisingly, the no-persona model sometimes matches or exceeds persona-conditioned agreement with human labels across all task variants, suggesting that simple label-based persona prompting may add limited annotation value in this setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.28031v1">Models Recall What They Violate: Constraint Adherence in Multi-Turn LLM Ideation</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      When researchers iteratively refine ideas with large language models, do the models preserve fidelity to the original objective? We introduce DriftBench, a benchmark for evaluating constraint adherence in multi-turn LLM-assisted scientific ideation. Across 2,146 scored benchmark runs spanning seven models from five providers (including two open-weight), four interaction conditions, and 38 research briefs from 24 scientific domains, we find that iterative pressure reliably increases structural complexity and often reduces adherence to original constraints. A restatement probe reveals a dissociation between declarative recall and behavioral adherence, as models accurately restate constraints they simultaneously violate. The knows-but-violates (KBV) rate, measuring constraint non-compliance despite preserved recall, ranges from 8% to 99% across models. Structured checkpointing partially reduces KBV rates but does not close the dissociation, and complexity inflation persists. Human validation against blind raters confirms that the LLM judge under-detects constraint violations, making reported constraint adherence scores conservative. Sensitivity analyses confirm the findings are robust to temperature (0.7 vs.\ 1.0) and pressure type (novelty vs.\ rigor). We release all briefs, prompts, rubrics, transcripts, and scores as an open benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.28005v1">Kernelized Advantage Estimation: From Nonparametric Statistics to LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 22 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have increasingly relied on reinforcement learning (RL) to improve their reasoning capabilities. Three approaches have been widely adopted: (i) Proximal policy optimization and advantage actor-critic rely on a deep neural network to estimate the value function of the learning policy in order to reduce the variance of the policy gradient. However, estimating and maintaining such a value network incurs substantial computational and memory overhead. (ii) Group relative policy optimization (GRPO) avoids training a value network by approximating the value function using sample averages. However, GRPO samples a large number of reasoning traces per prompt to achieve accurate value function approximation, making it computationally expensive. (iii) REINFORCE-type algorithms sample only a single reasoning trajectory per prompt, which reduces computational cost but suffers from poor sample efficiency. In this work, we focus on a practical, resource-constrained setting in which only a small number of reasoning traces can be sampled per prompt, while low-variance gradient estimation remains essential for high-quality policy learning. To address this challenge, we bring classical nonparametric statistical methods, which are both computationally and statistically efficient, to LLM reasoning. We employ kernel smoothing as a concrete example for value function estimation and the subsequent policy optimization. Numerical and theoretical results demonstrate that our proposal achieves accurate value and gradient estimation, leading to improved policy optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27996v1">Exploring Interaction Paradigms for LLM Agents in Scientific Visualization</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      This paper examines how different types of large language model (LLM) agents perform on scientific visualization (SciVis) tasks, where users generate visualization workflows from natural-language instructions. We compare three primary interaction paradigms, including domain-specific agents with structured tool use, computer-use agents, and general-purpose coding agents, by evaluating eight representative agents across 15 benchmark tasks and measuring visualization quality, efficiency, robustness, and computational cost. We further analyze interaction modalities, including code scripts and model context protocol (MCP) or API calls for structured tool use, as well as command-line interfaces (CLI) and graphical user interfaces (GUI) for more general interaction, while additionally studying the effect of persistent memory in selected agents. The results reveal clear tradeoffs across paradigms and modalities. General-purpose coding agents achieve the highest task success rates but are computationally expensive, while domain-specific agents are more efficient and stable but less flexible. Computer-use agents perform well on individual steps but struggle with longer multi-step workflows, indicating that long-horizon planning is their primary limitation. Across both CLI- and GUI-based settings, persistent memory improves performance over repeated trials, although its benefits depend on the underlying interaction mode and the quality of feedback. These findings suggest that no single approach is sufficient, and future SciVis systems should combine structured tool use, interactive capabilities, and adaptive memory mechanisms to balance performance, robustness, and flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20965v2">Learning to Aggregate Zero-Shot LLM Agents for Corporate Disclosure Classification</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      This paper studies whether a lightweight supervised aggregator can combine diverse zero-shot large language model outputs into a stronger downstream signal for corporate disclosure classification. Zero-shot LLMs can read disclosures without task-specific fine-tuning, but their predictions often vary across prompt perspectives, model families, and confidence levels. I examine this problem with a multi-prompt framework in which three fixed zero-shot LLM classifiers read each disclosure from different financial perspectives and output a sentiment label, a confidence score, and a short rationale. A logistic meta-classifier then aggregates these outputs to predict next-day stock return direction. To reduce pretrained-model contamination, I restrict evaluation to a post-release sample of 9{,}860 U.S.\ corporate disclosures issued by large publicly traded firms between January 2025 and March 2026, after the release of the frozen base LLMs used in the experiment. Results show that the trained aggregator outperforms single classifiers, majority vote, confidence-weighted voting, a zero-shot LLM judge, and a FinBERT baseline. Balanced accuracy rises from 0.566 for the best single classifier to 0.606 for the trained aggregator. The gain is largest in mixed-signal disclosures where classifiers disagree. The results suggest that zero-shot LLM outputs contain complementary financial signals, while also showing that the strongest gains come from supervised aggregation rather than from zero-shot voting alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27972v1">From LLM-Driven Trading Card Generation to Procedural Relatedness: A Pokémon Case Study</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Since the dawn of Trading Card Games, the genre has grown into a multi-billion-dollar industry engaging millions of analog and digital players worldwide. Popular TCGs rely on regular updates, balance adjustments, and rotating constraints to sustain engagement. Yet, as metagames stabilize, predictable strategies dominate and viable card options diminish, often resulting in repetitive and impaired player experiences. This paper investigates the use of Large Language Models and Image Diffusion Models for Procedural Content Generation of TCG cards, addressing these challenges by enabling a personalized infinity of card designs. Modern generative AI not only enables large-scale content creation but could even introduce procedural relatedness, fostering unique connections between players and their cards. We present a pipeline combining player-centric co-creation, fine-tuned embeddings, local LLMs, and Diffusion Models to generate dynamic, personalized cards while potentially expanding creative range. We evaluated the pipeline in a user study with 49 participants who generated 196 Pokémon card samples. Participants rated aesthetics and representativeness of visuals and mechanics, and provided qualitative feedback. Results show high satisfaction and indicate that most participants successfully realized their own ideas through prompt adjustments. These findings lay groundwork for future content generation systems and alternatives to conventional metagame evolution through procedural relatedness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27960v1">LLMs as ASP Programmers: Self-Correction Enables Task-Agnostic Nonmonotonic Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 30 pages
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) have achieved impressive reasoning milestones but continue to struggle with high computational costs, logical inconsistencies, and sharp performance degradation on high-complexity problems. While neuro-symbolic methods attempt to mitigate these issues by coupling LLMs with symbolic reasoners, existing approaches typically rely on monotonic logics (e.g., SMT) that cannot represent defeasible reasoning -- essential components of human cognition. We present "LLM+ASP," a framework that translates natural language into Answer Set Programming (ASP), a nonmonotonic formalism based on stable model semantics. Unlike prior "LLM+ASP" approaches that require manually authored knowledge modules, domain-specific prompts, or evaluation restricted to single problem classes, our framework operates without any per-task engineering and applies uniformly across diverse reasoning tasks. Our system utilizes an automated self-correction loop where structured feedback from the ASP solver enables iterative refinement. Evaluating across six diverse benchmarks, we demonstrate that: (1) stable model semantics allow LLMs to naturally express default rules and exceptions, outperforming SMT-based alternatives by significant margins on nonmonotonic tasks; (2) iterative self-correction is the primary driver of performance, effectively replacing the need for handcrafted domain knowledge; (3) compact in-context reference guides substantially outperform verbose documentation, revealing a "context rot" phenomenon where excessive context hinders constraint adherence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2310.02277v4">Junk DNA Hypothesis: Pruning Small Pre-Trained Weights Irreversibly and Monotonically Impairs "Difficult" Downstream Tasks in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 Published at ICML 2024
    </div>
    <details class="paper-abstract">
      We present Junk DNA Hypothesis by adopting a novel task-centric angle for the pre-trained weights of large language models (LLMs). It has been believed that weights in LLMs contain significant redundancy, leading to the conception that a considerable chunk of the parameters can be removed by pruning without compromising performance. Contrary to this belief, this paper presents a counter-argument: small-magnitude weights of pre-trained model weights encode vital knowledge essential for tackling difficult downstream tasks - manifested as the monotonic relationship between the performance drop of downstream tasks across the difficulty spectrum, as we prune more pre-trained weights by magnitude. Moreover, we reveal that these seemingly inconsequential weights can result in irreparable loss of knowledge and performance degradation in difficult tasks, even when downstream continual training is allowed. Interestingly, our evaluations show that the other popular compression, namely quantization, fails to exhibit similar monotonic effect and does not as convincingly disentangle this task-difficulty information. To study formally, we introduce several quantifiable metrics to gauge the downstream task difficulty: (1) within the same task category, and (2) across different task categories. Our extensive experiments substantiate the Junk DNA Hypothesis across a diverse range of model sizes, tasks, datasets, and even pruning methods. Codes are available at: https://github.com/VITA-Group/Junk_DNA_Hypothesis.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27844v1">ZipCCL: Efficient Lossless Data Compression of Communication Collectives for Accelerating LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Communication has emerged as a critical bottleneck in the distributed training of large language models (LLMs). While numerous approaches have been proposed to reduce communication overhead, the potential of lossless compression has remained largely underexplored since compression and decompression typically consume larger overheads than the benefits of reduced communication traffic. We observe that the communication data, including activations, gradients and parameters, during training often follows a near-Gaussian distribution, which is a key feature for data compression. Thus, we introduce ZipCCL, a lossless compressed communication library of collectives for LLM training. ZipCCL is equipped with our novel techniques: (1) theoretically grounded exponent coding that exploits the Gaussian distribution of LLM tensors to accelerate compression without expensive online statistics, (2) GPU-optimized compression and decompression kernels that carefully design memory access patterns and pipeline using communication-aware data layout, and (3) adaptive communication strategies that dynamically switch collective operations based on workload patterns and system characteristics. Evaluated on a 64-GPU cluster using both mixture-of-experts and dense transformer models, ZipCCL reduces communication time by up to 1.35$\times$ and achieves end-to-end training speedups of up to 1.18$\times$ without any impact on model quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.25326v3">AHASD: Asynchronous Heterogeneous Architecture for LLM Adaptive Drafting Speculative Decoding on Mobile Devices</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 7 pages, 9 figures, accepted by DAC 2026, repo: https://github.com/MAdrid1011/AHASD
    </div>
    <details class="paper-abstract">
      Speculative decoding enhances the inference efficiency of large language models (LLMs) by generating drafts using a small draft language model (DLM) and verifying them in batches with a large target language model (TLM). However, adaptive drafting inference on a mobile single-NPU-PIM system faces idle overhead in traditional operator-level synchronous execution and wasted computation in asynchronous execution due to fluctuations in draft length. This paper introduces AHASD, a task-level asynchronous mobile NPU-PIM heterogeneous architecture for speculative decoding. Notably, AHASD achieves parallel drafting on the PIM and verification on a single NPU through task-level DLM-TLM decoupling and specifically, it incorporates Entropy-History-Aware Drafting Control and Time-Aware Pre-Verification Control to dynamically manage adaptive drafting algorithm execution and pre-verification timing, suppressing invalid drafting based on low-confidence drafts. Additionally, AHASD integrates Attention Algorithm Units and Gated Task Scheduling Units within LPDDR5-PIM to enable attention link localization and sub-microsecond task switching on the PIM side. Experimental results for different LLMs and adaptive drafting algorithms show that AHASD achieves up to 4.2$\times$ in throughput and 5.6$\times$ in energy efficiency improvements over a GPU-only baseline, and 1.5$\times$ in throughput and 1.24$\times$ in energy efficiency gains over the state-of-the-art GPU+PIM baseline, with hardware overhead below 3% of the DRAM area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08101v3">LLM-Assisted Web Measurements</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 23 pages, 2 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Web measurements are a well-established methodology for assessing the security and privacy landscape of the Internet. However, existing top lists of popular websites are unlabeled and lack semantic information about the nature of the included websites, making targeted web measurements challenging, as researchers often rely on ad-hoc techniques to bias datasets toward specific website classes of interest. In this paper, we investigate the use of Large Language Models (LLMs) to enable targeted web measurement studies. Building on prior literature, we identify key website classification tasks relevant to web measurements and highlight limitations in state-of-the-art classification approaches. We construct carefully curated datasets to evaluate different LLMs on these tasks. Our results show that LLMs can achieve strong performance across multiple classification scenarios, but the choice of model and configuration plays a significant role. Motivated by the observed trade-off between classification accuracy and computational efficiency, we propose a practical two-step methodology for scalable targeted web measurements starting from the Tranco list. Finally, we conduct LLM-assisted web measurement studies inspired by prior work using our methodology and assess the validity of the resulting research inferences, showing that LLMs can effectively enable targeted measurements of security and privacy trends on the Web.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27789v1">Test Before You Deploy: Governing Updates in the LLM Supply Chain</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 4 pages, 1 figure, accepted to The 2nd International Workshop on Large Language Model Supply Chain Analysis (LLMSC2026) co-located with FSE 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as core dependencies in software systems. However, the hosted LLM services evolve continuously through provider-side updates without explicit version changes. These silent updates can introduce behavioral drift, causing regressions in functionality, formatting, safety constraints, or other application-specific requirements. Existing approaches focus primarily on regression testing or versioning but do not provide deployer-side mechanisms for governing compatibility during opaque model evolution. This paper proposes a deployment-side governance framework based on three components: clearly defined rules for how the model is allowed to behave (production contracts), focused testing organized by deployment risk categories (risk-category-based testing suite), and release checkpoints that block updates unless they meet defined safety and performance standards (compatibility gates). Through exploratory validation across multiple LLM versions, we provide evidence that targeted testing in specific risk areas can uncover performance regressions that overall metrics miss. We also identify several open research challenges, including how to systematically build effective test suites, how to set reliable performance thresholds in non-deterministic systems, and how to detect and explain model drift when providers offer limited transparency. Overall, we frame LLM update management as a software supply chain governance problem and outline a research agenda for putting deployer-side compatibility controls into practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.07180v3">Flattery in Motion: Benchmarking and Analyzing Sycophancy in Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 27 Pages, Accepted by ACL 2026 Main Conference
    </div>
    <details class="paper-abstract">
      As video large language models (Video-LLMs) become increasingly integrated into real-world applications that demand grounded multimodal reasoning, ensuring their factual consistency and reliability is of critical importance. However, sycophancy, the tendency of these models to align with user input even when it contradicts the visual evidence, undermines their trustworthiness in such contexts. Current sycophancy research has largely overlooked its specific manifestations in the videolanguage domain, resulting in a notable absence of systematic benchmarks and targeted evaluations to understand how Video-LLMs respond under misleading user input. To fill this gap, we propose VISE(Video-LLM Sycophancy Benchmarking and Evaluation), the first benchmark designed to evaluate sycophantic behavior in state-of-the-art Video-LLMs across diverse question formats, prompt biases, and visual reasoning tasks. Specifically, VISEpioneeringly brings linguistic perspectives on sycophancy into the video domain, enabling fine-grained analysis across multiple sycophancy types and interaction patterns. Furthermore, we propose two potential training-free mitigation strategies revealing potential paths for reducing sycophantic bias: (i) enhancing visual grounding through interpretable key-frame selection and (ii) steering model behavior away from sycophancy via targeted, inference-time intervention on its internal neural representations. Our code is available at https://anonymous.4open.science/r/VideoSycophancy-567F.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27763v1">Intent2Tx: Benchmarking LLMs for Translating Natural Language Intents into Ethereum Transactions</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) offers a transformative interface for Web3, yet existing benchmarks fail to capture the complexity of translating high-level user intents into functionally correct, state-dependent on-chain transactions. We present \textsc{Intent2Tx}, a high-fidelity benchmark featuring 29,921 single-step and 1,575 multi-step instances meticulously derived from 300 days of real-world Ethereum mainnet traces. Unlike prior works that rely on synthetic instructions, \textsc{Intent2Tx} grounds natural language intents in real-world protocol interactions across 11 categories, including diverse long-tail Decentralized Finance (DeFi) primitives. To enable rigorous evaluation, we propose an execution-aware framework that transcends surface-level text matching by employing differential state analysis on forked mainnet environments. Our extensive evaluation of 16 state-of-the-art LLMs reveals that while scaling and retrieval-augmentation enhance logical consistency and parameter precision, current models struggle with out-of-distribution generalization and multi-step planning. Crucially, our execution-based analysis demonstrates that syntactically valid outputs often fail to achieve intended state transitions, highlighting a significant gap in current "reasoning-to-execution" capabilities. \textsc{Intent2Tx} serves as a critical foundation for developing autonomous, reliable agents in intent-centric Web3 ecosystems. Code and data: https://anonymous.4open.science/r/Intent2Tx_Bench-97FF .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27747v1">Position-Aware Drafting for Inference Acceleration in LLM-Based Generative List-Wise Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based generative list-wise recommendation has advanced rapidly, but decoding remains sequential and thus latency-prone. To accelerate inference without changing the target distribution, speculative decoding (SD) uses a small draft model to propose several next tokens at once and a target LLM to verify and accept the longest prefix, skipping multiple steps per round. In generative recommendation, however, each item is represented by multiple semantic-ID tokens, often with separators, and current drafts typically treat these tokens uniformly. This overlooks two practical facts: (i) a token's semantics depend on its within-item slot, and (ii) uncertainty tends to increase with speculation depth. Without modeling these effects, SD's speedups can be limited. We introduce PAD-Rec, Position-Aware Drafting for generative Recommendation, a lightweight module that augments the draft model with two complementary signals. Item position embeddings explicitly encode the within-item slot of each token, strengthening structural awareness. Step position embeddings encode the draft step, allowing the model to adapt to depth-dependent uncertainty and improve proposal quality. To harmonize these signals with base features, we add simple gates: a learnable coefficient for item slots and a context-driven gate for draft steps. The module is trainable, easy to integrate with standard draft models, and adds negligible inference overhead. Extensive experiments on four real-world datasets show up to 3.1x wall-clock speedup and about 5% average wall-clock speedup gain over strong SD baselines, while largely preserving recommendation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.14541v3">LLM-based User Profile Management for Recommender System</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 Accepted at SIGIR 2025 Workshop
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has opened new opportunities in recommender systems by enabling zero-shot recommendation without conventional training. Despite their potential, most existing works rely solely on users' purchase histories, leaving significant room for improvement by incorporating user-generated textual data, such as reviews and product descriptions. Addressing this gap, we propose PURE, a novel LLM-based recommendation framework that builds and maintains evolving user profiles by systematically extracting and summarizing key information from user reviews. PURE consists of three core components: a Review Extractor for identifying user preferences and key product features, a Profile Updater for refining and updating user profiles, and a Recommender for generating personalized recommendations using the most current profile. To evaluate PURE, we introduce a continuous sequential recommendation task that reflects real-world scenarios by adding reviews over time and updating predictions incrementally. Our experimental results on Amazon datasets demonstrate that PURE outperforms existing LLM-based methods, effectively leveraging long-term user information while managing token limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27727v1">LLM-as-a-Judge for Human-AI Co-Creation: A Reliability-Aware Evaluation Framework for Coding</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      LLMs are increasingly employed both as judges for evaluating open-ended outputs and as co-creation partners in AI-assisted programming; yet rigorous evaluation in human-AI co-creation settings remains underdeveloped as judgments must be reliable, comparable across models, and interpretable over multi-turn interaction. To address this gap, a rubric-driven LLM-as-a-Judge framework is presented for contest-style human-AI co-creation in coding and software engineering (SE). The framework is built around schema-constrained judge outputs, validation and repair mechanisms, grouped and split by user and problem to prevent trajectory leakage, and participant-level NONBLIND context. Multiple LLM judges are assessed through a multi-metric protocol covering discrimination (ROC-AUC, PR-AUC), thresholded decision quality (MCC), probabilistic reliability (LogLoss, Brier score, ECE), and inter-judge agreement (Cohen's and Fleiss' k). Human-AI co-creation is further examined through trajectory-level signals, including turn-wise confidence, Success-at-Turn, time-to-success, revision churn, and CodeBLEU. Co-creation success is found to concentrate early, with Success-at-Turn rising to 0.8533 at the first observed turn and stabilizing at 0.8641 by turn 6. Revision behavior, however, remains heterogeneous, suggesting that productive progress can emerge through either incremental refinement or broader restructuring. On the judging side, the best held-out scores reach 0.5937 for ROC-AUC, 0.6904 for PR-AUC, and 0.5000 for MCC test, while inter-judge consistency remains modest overall (mean pairwise Cohen's k = 0.1592, Fleiss' k = 0.0696). Taken together, this work offers an auditable and reproducible evaluation methodology that links reliability-aware LLM judging with trajectory-based analysis of human-AI co-creation, providing a practical evaluation template for future AI-assisted coding and SE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27714v1">How Code Representation Shapes False-Positive Dynamics in Cross-Language LLM Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 14 pages, 5 figures
    </div>
    <details class="paper-abstract">
      How code representation format shapes false positive behaviour in cross-language LLM vulnerability detection remains poorly understood. We systematically vary training intensity and code representation format, comparing raw source text with pruned Abstract Syntax Trees at both training time and inference time, across two 8B-parameter LLMs (Qwen3-8B and Llama 3.1-8B-Instruct) fine-tuned on C/C++ data from the NIST Juliet Test Suite (v1.3) and evaluated on Java (OWASP Benchmark v1.2) and Python (BenchmarkPython v0.1). Cross-language FPR reflects the joint effect of training-time and inference-time representation, not either alone. Text fine-tuning drives FPR upward monotonically (Qwen3-8B: 0.763 zero-shot, 0.866 pilot, 1.000 full-scale) while F1 remains stable (0.637-0.688), masking the collapse. We argue surface-cue memorisation is the primary mechanism: text fine-tuning encodes C/C++-specific API names and syntactic idioms as vulnerability triggers that fire indiscriminately on target-language code. A cross-representation probe, applying text-trained weights to AST-encoded input without retraining, isolates this: Qwen3-8B FPR drops from 0.866 to 0.583, and 37.2% of false positives revert to true negatives under AST input alone. Direct AST fine-tuning does not preserve the benefit (FPR at least 0.970), as flat linearisation introduces structural surface cues of its own. The pattern replicates across both model families. On BenchmarkPython the AST probe yields FPR=0.554, within 2.9 percentage points of the Java result, despite maximal surface-syntax differences, substantially weakening a domain-shift explanation. These findings motivate a pre-deployment consistency gate, running alerts through both text and AST paths, as a retraining-free filter for false-positive-sensitive settings, at the cost of reduced recall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.05291v2">Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 Accepted to ACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) learn non-trivial abstractions during pretraining, such as detecting irregular plural noun subjects. However, because traditional evaluation methods (e.g., benchmarking) fail to reveal how models acquire these concepts and capabilities, it is not well understood when and how these specific linguistic abilities emerge. To bridge this gap and better understand model training at the concept level, we use sparse crosscoders to discover and align features across model checkpoints. Using this approach, we track the evolution of linguistic features during pretraining. We train crosscoders between open-sourced checkpoint triplets with significant performance and representation shifts, and introduce a novel metric, Relative Indirect Effects (RelIE), to trace training stages at which individual features become causally important for task performance. We show that crosscoders can detect feature emergence, maintenance, and discontinuation during pretraining. Our approach is architecture-agnostic and scalable, offering a promising path toward more interpretable and fine-grained analysis of representation learning throughout pretraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27713v1">Knowledge Graph Representations for LLM-Based Policy Compliance Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      The risks posed by AI features are increasing as they are rapidly integrated into software applications. In response, regulations and standards for safe and secure AI have been proposed. In this paper, we present an agentic framework that constructs knowledge graphs (KGs) from AI policy documents and retrieves policy-relevant information to answer questions. We build KGs from three AI risk-related polices under two ontology schemas, and then evaluate five LLMs on 42 policy QA tasks spanning six reasoning types, from entity lookup to cross-policy inference, using both heuristic scoring and an LLM-as-judge. KG augmentation improves scores for all five models, and an open, LLM-discovered schema matches or exceeds the formal ontology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.03853v7">Dr. Jekyll and Mr. Hyde: Two Faces of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 Presented at the Joint National Conference on Cybersecurity (ITASEC & SERICS 2026), Cagliari, Italy, February 09-13, 2026. Published as Paper 35 in CEUR Workshop Proceedings, Vol-4198. Available at: https://ceur-ws.org/Vol-4198/
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being integrated into applications such as chatbots or email assistants. To prevent improper responses, safety mechanisms, such as Reinforcement Learning from Human Feedback (RLHF), are implemented in them. In this work, we bypass these safety measures for ChatGPT, Gemini, and Deepseek by making them impersonate complex personas with personality characteristics that are not aligned with a truthful assistant. First, we create elaborate biographies of these personas, which we then use in a new session with the same chatbots. Our conversations then follow a role-play style to elicit prohibited responses. Using personas, we show that prohibited responses are provided, making it possible to obtain unauthorized, illegal, or harmful information when querying ChatGPT, Gemini, and Deepseek. We show that these chatbots are vulnerable to this attack by getting dangerous information for 40 out of 40 illicit questions in GPT-4.1-mini, Gemini-1.5-flash, 39 out of 40 in GPT-4o-mini, 38 out of 40 in GPT-3.5-turbo, and 2 out of 2 cases in Gemini-2.5-flash and DeepSeek V3. The attack can be carried out manually or automatically using a support LLM, and has proven effective against models deployed between 2023 and 2025.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27661v1">Language Ideologies in a Multilingual Society: An LLM-based Analysis of Luxembourgish News Comments</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Detecting language ideologies is a valuable yet complex task for understanding how identities are constructed through discourse. In Luxembourg's multicultural and multilingual society, language ideologies reflect more than simple preferences: they carry deep cultural and social meanings, shaping identities and social belonging. Following recent developments in applying Natural Language Processing tools to linguistics and social science, this paper explores the potential of large language models to assist in the detection of language ideologies. We manually annotate a corpus of user comments in Luxembourgish with predefined ideological categories and then evaluate the performance of large language models under varying prompt conditions to assess their ability to replicate these human annotations. Since Luxembourgish is a small language and poorly represented in the LLMs' training data, we also investigate whether machine-translating the data to high-resource languages increases performance on the ideology detection task. Our findings suggest that, while LLMs are not yet fully optimized for a multi-class ideological annotation task, they are practical tools to identify language ideological content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11653v3">GroupRank: A Groupwise Paradigm for Effective and Efficient Passage Reranking with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 Accepted by ACL-Findings 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful tools for passage reranking in information retrieval, leveraging their superior reasoning capabilities to address the limitations of conventional models on complex queries. However, current LLM-based reranking paradigms are fundamentally constrained by an efficiency-accuracy trade-off: (1) pointwise methods are efficient but ignore inter-document comparison, yielding suboptimal accuracy; (2) listwise methods capture global context but suffer from context-window constraints and prohibitive inference latency. To address these issues, we propose GroupRank, a novel paradigm that balances flexibility and context awareness. To unlock the full potential of groupwise reranking, we propose an answer-free data synthesis pipeline that fuses local pointwise signals with global listwise rankings. These samples facilitate supervised fine-tuning and reinforcement learning, with the latter guided by a specialized group-ranking reward comprising ranking-utility and group-alignment. These complementary components synergistically optimize document ordering and score calibration to reflect intrinsic query-document relevance. Experimental results show GroupRank achieves a state-of-the-art 65.2 NDCG@10 on BRIGHT and surpasses baselines by 2.1 points on R2MED, while delivering a 6.4$\times$ inference speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27643v1">HAVEN: Hybrid Automated Verification ENgine for UVM Testbench Synthesis with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 9 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Integrated Circuit (IC) verification consumes nearly 70% of the IC development cycle, and recent research leverages Large Language Models (LLMs) to automatically generate testbenches and reduce verification overhead. However, LLMs have difficulty generating testbenches correctly. Unlike high-level programming languages, Hardware Description Languages (HDLs) are extremely rare in LLMs training data, leading LLMs to produce incorrect code. To overcome challenges when using LLMs to generate Universal Verification Methodology (UVM) testbenches and sequences, wepropose HAVEN (Hybrid Automated Verification ENgine) to prevent LLMs from writing HDL directly. For UVM testbench generation, HAVEN utilizes LLM agents to analyze design specifications to produce a structured architectural plan. The HAVEN Template Engine then combines with predefined and protocol-specific templates to generate all UVM components with correct bus-handshake timing. For UVM sequence generation, HAVEN introduces a Protocol-Aware Sequence Domain-Specific Language (DSL) that decomposes sequences into fine-grained step types. A set of predefined DSL patterns first establishes sequences that achieve a high coverage rate without LLM involvement. HAVEN continues to improve the coverage rate by iteratively leveraging LLM agents to analyze coverage gap reports and compose additional targeted DSL sequences. Unlike previous works, HAVEN is the first system that utilizes pre-defined, protocol-specific Jinja2 templates to generate all UVM components and UVM sequences using our proposed Protocol-Aware DSL and rule-based code generator. Our experimental results on 19 open-source IP designs spanning three interface protocols (Direct, Wishbone, AXI4-Lite) show that HAVEN achieves 100% compilation success, 90.6% code coverage, and 87.9% functional coverage on average, and is SOTA among LLM-assisted testbench generation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27633v1">Political Bias Audits of LLMs Capture Sycophancy to the Inferred Auditor</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are commonly evaluated for political bias based on their responses to fixed questionnaires, which typically place frontier models on the political left. A parallel literature shows that LLMs are sycophantic: they adapt their answers to the views, identities, and expectations of the user. We show that these findings are linked: standard political-bias audits partly capture sycophantic accommodation to the inferred auditor. We employ a factorial experiment across three major audit instruments--the Political Compass Test, the Pew Political Typology, and 1,540 partisan-benchmarked Pew American Trends Panel items--administered to six frontier LLMs while varying only the asker's stated identity (N = 30,990 responses). At baseline, all six models lean left. When the asker identifies as a conservative Republican, responses shift sharply: the share of items closer to Democrats falls by 28-62 percentage points, and all six models move right of center. A mirror-image progressive-Democrat cue produces little change; rightward accommodation is 8.0$\times$ larger than leftward. When asked who the default asker is, models identify an auditor, researcher, or academic; when asked what answer that asker expects, they select the Democrat-coded option 75% of the time, nearly the rate under an explicit progressive cue. These patterns are inconsistent with a purely fixed model ideology and indicate that single-prompt audits capture an interaction between model and inferred interlocutor. Political bias in LLMs is therefore not a fixed point on an ideological scale but a response profile that must be mapped across realistic interlocutors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27624v1">Mapping how LLMs debate societal issues when shadowing human personality traits, sociodemographics and social media behavior</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can strongly shape social discourse, yet datasets investigating how LLM outputs vary across controlled social and contextual prompting remain sparse. Cognitive Digital Shadows (CDS) is a 190,000-record synthetic corpus supporting analyses of LLM-generated discourse. Each CDS record is generated by one of 19 LLMs, prompted to shadow either a human persona or an AI-assistant role. CDS contains LLM responses on 4 controversial societal topics: vaccines/healthcare, social media disinformation, the gender gap in science, and STEM stereotypes. Persona-conditioned records encode 17 sociodemographic and psychological attributes, providing data linking LLMs' prompts, language, stances and reasoning. Texts are validated for topic anchoring and can support emotional analyses via interpretable NLP (e.g. textual forma mentis networks). CDS is enriched by a pooling platform with user-friendly dashboards, enabling easy, interactive group-level comparisons of emotional and semantic framing across personas, topics and models. The CDS prompting framework supports future audits of LLMs' bias, social sensitivity and alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27618v1">Math Education Digital Shadows for facilitating learning with LLMs: Math performance, anxiety and confidence in simulated students and AIs</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      To enhance LLMs' impact on math education, we need data on their mathematical prowess and biases across prompts. To fill this gap, we introduce MEDS (Math Education Digital Shadows) as a dataset mapping how large language models reason about and report mathematics across human- and AI-like conditions. MEDS involves 28,000 personas from 14 LLMs (from families like Mistral, Qwen, DeepSeek, Granite, Phi and Grok) shadowing either humans or AI assistants. Each record/shadow includes a set of prompts along with psychological/sociodemographic persona metadata and four types of math tasks: (i) open math interview, (ii) three psychometric tests about math perceptions with explanations, (iii) cognitive networks capturing math attitudes, and (iv) 18 high-school math test questions together with their reasoning and confidence scores. MEDS differs from traditional score-only math benchmarks because it integrates concepts of self-efficacy, math anxiety, and cognitive network science besides math proficiency scores. Data validation shows that the sampled LLMs exhibit schema integrity and consistent personas, together with family-specific peculiarities like human-like negative math attitudes, logical fallacies, and math overconfidence. MEDS will benefit learning analytics experts, cognitive scientists, and developers of safer AI tutors in mathematics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27599v1">One Pass, Any Order: Position-Invariant Listwise Reranking for LLM-Based Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 Accepted at SIGIR 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for recommendation reranking, but their listwise predictions can depend on the order in which candidates are presented. This creates a mismatch between the set-based nature of recommendation and the sequence-based computation of decoder-only LLMs, where permuting an otherwise identical candidate set can change item scores and final rankings. Such order sensitivity makes LLM-based rerankers difficult to rely on, since rankings may reflect prompt serialization rather than user preference. We propose InvariRank, a permutation-invariant listwise reranking framework that addresses this dependence at the architectural level. InvariRank blocks cross-candidate attention with a structured attention mask and negates position-induced scoring changes through shared positional framing under Rotary Positional Embeddings (RoPE). Combined with a listwise learning-to-rank objective, the model scores all candidates in a single forward pass, avoiding permutation-based invariance training objectives that require multiple permutations of a candidate set. Experiments on recommendation benchmarks show that InvariRank maintains competitive ranking effectiveness while producing stable rankings across candidate permutations. The results suggest that architectural invariance is a practical route to reliable and efficient LLM-based recommendation reranking. The source code is at https://github.com/ejbito/InvariRank.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11908v2">PPA-Plan: Proactive Pitfall Avoidance for Reliable Planning in Long-Context LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 Accepted to the Main Conference of the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026). 27 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle with reasoning over long contexts where relevant information is sparsely distributed. Although plan-and-execute frameworks mitigate this by decomposing tasks into planning and execution, their effectiveness is often limited by unreliable plan generation due to dependence on surface-level cues. Consequently, plans may be based on incorrect assumptions, and once a plan is formed, identifying what went wrong and revising it reliably becomes difficult, limiting the effectiveness of reactive refinement. To address this limitation, we propose PPA-Plan, a proactive planning strategy for long-context reasoning that focuses on preventing such failures before plan generation. PPA-Plan identifies potential logical pitfalls and false assumptions, formulates them as negative constraints, and conditions plan generation on explicitly avoiding these constraints. Experiments on long-context QA benchmarks show that executing plans generated by PPA-Plan consistently outperforms existing plan-and-execute methods and direct prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27555v1">SpatialGrammar: A Domain-Specific Language for LLM-Based 3D Indoor Scene Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Automatically generating interactive 3D indoor scenes from natural language is crucial for virtual reality, gaming, and embodied AI. However, existing LLM-based approaches often suffer from spatial errors and collisions, in part because common scene representations-raw coordinates or verbose code-are difficult for models to reason about 3D spatial relationships and physical constraints. We propose SpatialGrammar, a domain-specific language that represents gravity-aligned indoor layouts as BEV grid placements with deterministic compilation to valid 3D geometry, enabling verifiable constraint checking. Building on this representation, we develop (1) SG-Agent, a closed-loop system that uses compiler feedback to iteratively refine scenes and enforce collision constraints, and (2) SG-Mini, a 104M-parameter model trained entirely on compiler-validated synthetic data. Across 159 test scenes spanning five scenarios of different complexity, SG-Agent improves spatial fidelity and physical plausibility over prior methods, while SG-Mini performs competitively against larger LLM-based baselines on single-shot generation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27540v1">In-Context Examples Suppress Scientific Knowledge Recall in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Scientific reasoning rarely stops at what is directly observable; it often requires uncovering hidden structure from data. From estimating reaction constants in chemistry to inferring demand elasticities in economics, this latent structure recovery is what distinguishes scientific reasoning from curve fitting. Large language models (LLMs) can often recall and apply relevant scientific formulas, but we show that this ability is surprisingly easy to suppress. We show that adding in-context examples makes models rely less on pretrained domain knowledge, even when those examples are generated by the very same formula. Rather than reinforcing knowledge-driven derivation, examples shift computation toward empirical pattern fitting. We document this knowledge displacement on 60 latent structure recovery tasks across five scientific domains, 6,000 trials, and four models. This displacement is consistent across domains, but its accuracy consequences depend on how the displaced strategy compares to the one that replaces it: the same shift can lower accuracy, leave it unchanged, or appear to improve it. In all cases, however, the model shifts away from knowledge-driven reasoning. For practitioners deploying LLMs on scientific tasks, the message is cautionary: in-context examples may displace, rather than reinforce, the knowledge they are intended to support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.19221v2">UAF: A Unified Audio Front-end LLM for Full-Duplex Speech Interaction</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Full-duplex speech interaction, as the most natural and intuitive mode of human communication, is driving artificial intelligence toward more human-like conversational systems. Traditional cascaded speech processing pipelines suffer from critical limitations, including accumulated latency, information loss, and error propagation across modules. To address these issues, recent efforts focus on the end-to-end audio large language models (LLMs) like GPT-4o, which primarily unify speech understanding and generation task. However, most of these models are inherently half-duplex, and rely on a suite of separate, task-specific front-end components, such as voice activity detection (VAD) and turn-taking detection (TD). In our development of speech assistant, we observed that optimizing the speech front-end is equally crucial as advancing the back-end unified model for achieving seamless, responsive interactions. To bridge this gap, we propose the first unified audio front-end LLM (UAF) tailored for full-duplex speech systems. Our model reformulates diverse audio front-end tasks into a single auto-regressive sequence prediction problem, including VAD, TD, speaker recognition (SR), automatic speech recognition (ASR) and question answer (QA). It takes streaming fixed-duration audio chunk (e.g., 600 ms) as input, leverages a reference audio prompt to anchor the target speaker at the beginning, and regressively generates discrete tokens encoding both semantic content and system-level state controls (e.g., interruption signals). Experiments demonstrate that our model achieves leading performance across multiple audio front-end tasks and significantly enhances response latency and interruption accuracy in real-world interaction scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.07101v4">Grounding Machine Creativity in Game Design Knowledge Representations: Empirical Probing of LLM-Based Executable Synthesis of Goal Playable Patterns under Structural Constraints</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Creatively translating complex gameplay ideas into executable artifacts (e.g., games as Unity projects and code) remains a central challenge in computational game creativity. Gameplay design patterns provide a structured representation for describing gameplay phenomena, enabling designers to decompose high-level ideas into entities, constraints, and rule-driven dynamics. Among them, goal patterns formalize common player-objective relationships. Goal Playable Concepts (GPCs) operationalize these abstractions as playable Unity engine implementations, supporting experiential exploration and compositional gameplay design. We frame scalable playable pattern realization as a problem of constrained executable creative synthesis: generated artifacts must satisfy Unity's syntactic and architectural requirements while preserving the semantic gameplay meanings encoded in goal patterns. This dual constraint limits scalability. Therefore, we investigate whether contemporary large language models (LLMs) can perform such synthesis under engine-level structural constraints and generate Unity code (as games) structured and conditioned by goal playable patterns. Using 26 goal pattern instantiations, we compare a direct generation baseline (natural language -> C# -> Unity) with pipelines conditioned on a human-authored Unity-specific intermediate representation (IR), across three IR configurations and two open-source models (DeepSeek-Coder-V2-Lite-Instruct and Qwen2.5-Coder-7B-Instruct). Compilation success is evaluated via automated Unity replay. We propose grounding and hygiene failure modes, identifying structural and project-level grounding as primary bottlenecks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27421v1">A Reproducibility Study of LLM-Based Query Reformulation</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are now widely used for query reformulation and expansion in Information Retrieval, with many studies reporting substantial effectiveness gains. However, these results are typically obtained under heterogeneous experimental conditions, making it difficult to assess which findings are reproducible and which depend on specific implementation choices. In this work, we present a systematic reproducibility and comparative study of ten representative LLM-based query reformulation methods under a unified and strictly controlled experimental framework. We evaluate methods across two architectural LLM families at two parameter scales, three retrieval paradigms (lexical, learned sparse, and dense), and nine benchmark datasets spanning TREC Deep Learning and BEIR. Our results show that reformulation gains are strongly conditioned on the retrieval paradigm, that improvements observed under lexical retrieval do not consistently transfer to neural retrievers, and that larger LLMs do not uniformly yield better downstream performance. These findings clarify the stability and limits of reported gains in prior work. To enable transparent replication and ongoing comparison, we release all prompts, configurations, evaluation scripts, and run files through QueryGym, an open-source reformulation toolkit with a public leaderboard.\footnote{https://leaderboard.querygym.com}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27410v1">From Unstructured to Structured: LLM-Guided Attribute Graphs for Entity Search and Ranking</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Entity search, i.e., finding the most similar entities to a query entity, faces unique challenges in e-commerce, where product similarity varies across categories and contexts. Traditional embedding-based approaches often struggle to capture nuanced context-specific attribute relevance. In this paper, we present a two-stage approach combining Large Language Model (LLM)-driven attribute graph construction with graph-aware LLM ranking. In the offline stage, we extract structured product attributes from unstructured text, and construct a reusable attribute graph with category-aware schemas. In the online stage, we rank retrieved candidates by reasoning over this structured representation rather than raw text, reducing per-product token usage by 57% while improving ranking precision. Experiments show that our approach outperforms multiple baselines under zero-shot scenarios, achieving a over 5% improvement in average precision without requiring training data, generalizes robustly across diverse product categories, and shows immense potential for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.21479v3">Frozen LLMs as Map-Aware Spatio-Temporal Reasoners for Vehicle Trajectory Prediction</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 Accepted for publication at IEEE Intelligent Vehicles Symposium 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated strong reasoning capabilities and attracted increasing research attention in the field of autonomous driving (AD). However, safe application of LLMs on AD perception and prediction still requires a thorough understanding of both the dynamic traffic agents and the static road infrastructure. To this end, this study introduces a framework to evaluate the capability of LLMs in understanding the behaviors of dynamic traffic agents and the topology of road networks. The framework leverages frozen LLMs as the reasoning engine, employing a traffic encoder to extract spatial-level scene features from observed trajectories of agents, while a lightweight Convolutional Neural Network (CNN) encodes the local high-definition (HD) maps. To assess the intrinsic reasoning ability of LLMs, the extracted scene features are then transformed into LLM-compatible tokens via a reprogramming adapter. By residing the prediction burden with the LLMs, a simpler linear decoder is applied to output future trajectories. The framework enables a quantitative analysis of the influence of multi-modal information, especially the impact of map semantics on trajectory prediction accuracy, and allows seamless integration of frozen LLMs with minimal adaptation, thereby demonstrating strong generalizability across diverse LLM architectures and providing a unified platform for model evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27405v1">Beyond the Mean: Within-Model Reliable Change Detection for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 7 pages, 4 figures, 2 tables. Pre-registered study. Code and data available
    </div>
    <details class="paper-abstract">
      We adapted the Reliable Change Index (RCI; Jacobson and Truax, 1991) from clinical psychology to item-level LLM version comparison on 2,000 MMLU-Pro items (K=10 samples at T=0.7). Two within-family pairs were tested: Llama 3 to 3.1 (+1.6 points) and Qwen 2.5 to 3 (+2.8 points). On the full benchmark, most items showed no reliable change (79% and 72%). However, over half the items were floor/ceiling. Among analysable items, change was bidirectional with large effect sizes: 34% improved and 28% deteriorated for Llama; 47% improved and 39% deteriorated for Qwen (median |delta p| = 0.50 and 0.90). Churn was asymmetric by difficulty: low-accuracy items improved, high-accuracy items deteriorated. Domain-level decomposition revealed family-specific reversals: Llama lost physics while Qwen lost law. Greedy single-shot evaluation missed 42% of reliably changed items and falsely flagged 25% of unchanged items. The aggregate accuracy gain is the net residual of opposing item-level movements. We recommend reporting churn rate alongside aggregate accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27401v1">Perturbation Probing: A Two-Pass-per-Prompt Diagnostic for FFN Behavioral Circuits in Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Perturbation probing generates task-specific causal hypotheses for FFN neurons in large language models using two forward passes per prompt and no backpropagation, followed by a one-time intervention sweep of about 150 passes amortized across all identified neurons. Across eight behavioral circuits, 13 models, and four architecture families, we identify two circuit structures that organize LLM behavior. Opposition circuits appear when RLHF suppresses a pre-training tendency. In safety refusal, about 50 neurons, or 0.014 percent of all neurons, control the refusal template; ablating them changes 80 percent of response formats on 520 AdvBench prompts while producing near-zero harmful compliance, 3 of 520 cases, all with disclaimers. Routing circuits appear for pre-training behaviors distributed through attention. For language selection, residual-stream direction injection switches English to Chinese output on 99.1 percent of 580 benchmark prompts in the 3 of 19 tested models that satisfy three observed conditions: bilingual training, FFN-to-skip signal ratio between 0.3 and 1.1, and linear representability. The same intervention fails on the other 16 models and on math, code, and factual circuits, defining the limits of directional steering. The FFN-to-skip signal ratio, computed from the same two forward passes, distinguishes the two structures and predicts the appropriate intervention. Circuit topology varies by architecture, from Qwen's concentrated FFN bottleneck to Gemma's normalization-shielded circuit. In Qwen3.5-2B, ablating 20 neurons eliminates multi-turn sycophantic capitulation, while amplifying 10 related neurons improves factual correction from 52 percent to 88 percent on 200 TruthfulQA prompts. These results show that perturbation probing offers mechanistic insight into RLHF-organized behavior and a practical toolkit for precision template-layer editing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27396v1">VitaLLM: A Versatile, Ultra-Compact Ternary LLM Accelerator with Dependency-Aware Scheduling</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Deploying Large Language Models (LLMs) on resource-constrained edge devices faces critical bottlenecks in memory bandwidth and power consumption. While ternary quantization (e.g., BitNet b1.58) significantly reduces model size, its direct deployment on general-purpose hardware is hindered by workload imbalance, bandwidth-bound decoding, and strict data dependencies. To address these challenges, we propose \textbf{VitaLLM}, a hardware-software co-designed accelerator tailored for efficient ternary LLM inference. We introduce a heterogeneous \textbf{Dual-Core Compute Strategy} that synergizes specialized TINT-Cores for massive ternary projections with a unified BoothFlex-Core for mixed-precision attention, ensuring high utilization across both compute-bound prefill and bandwidth-bound decode stages. Furthermore, we develop a \textbf{Leading One Prediction (LOP)} mechanism to prune redundant Key-Value (KV) cache fetches and a \textbf{Dependency-Aware Scheduling} framework to hide the latency of nonlinear operations. Implemented in TSMC 16nm technology, VitaLLM achieves a decoding throughput of 70.70 tokens/s within an ultra-compact area of 0.223 mm$^2$ and a power consumption of 65.97 mW. The design delivers a superior Figure of Merit (FOM) of 17.4 TOPS/mm$^2$/W, significantly outperforming state-of-the-art accelerators. Finally, we explore an extended bit-serial design (BoothFlex-BS) to demonstrate the architecture's adaptability for precision-agile inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27384v1">RCW-CIM: A Digital CIM-based LLM Accelerator with Read-Compute/Write</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 accepted in ISCAS 2026
    </div>
    <details class="paper-abstract">
      Digital computing-in-memory (DCIM) has emerged as a promising solution for large language model (LLM) acceleration by minimizing data transfers between external DRAM and on-chip accelerators while maintaining high precision for superior accuracy. However, existing CIM architectures often overlook weight update latency, which becomes critical as LLM weights are far larger than a single CIM macro capacity. To address this issue, this paper proposes a read-compute/write (RCW) architecture that effectively minimizes weight update latency, along with a nonlinear operator fusion that further mitigates dependencyinduced latency. The proposed RCW reduces decoding computing latency by 21.59% on the Llama2-7B model. In addition, the nonlinear operator fusion mechanism achieves a 69.17% latency reduction through efficient partial accumulation and group-based approximation. Furthermore, a weight-stationary and output column stationary (WS-OCS) dataflow is introduced to reduce both external DRAM access and internal CIM weight updates by 51.6% and 87.6% respectively during the prefill phase of 1024 tokens, leading to an overall 49.76% latency reduction. Fabricated using TSMC 22 nm CMOS technology and operating at 100 MHz, the proposed RCW-CIM achieves 3.28 TOPS and 42.3 TOPS/W, enabling 4.2 ms prefill latency and 26.87 decoded tokens per second for the INT4-weight Llama2 model with dual DDR5-6400 memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19423v3">UniRec: Unified Multimodal Encoding for LLM-Based Recommendations</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Large language models have recently shown promise for multimodal recommendation, particularly with text and image inputs. Yet real-world recommendation signals extend far beyond these modalities. To reflect this, we formalize recommendation features into four modalities: text, images, categorical features, and numerical attributes, and highlight the unique challenges this heterogeneity poses for LLMs in understanding multimodal information. In particular, these challenges arise not only across modalities but also within them, as attributes such as price, rating, and time may all be numeric yet carry distinct semantic meanings. Beyond this intra-modality ambiguity, another major challenge is the nested structure of recommendation signals, where user histories are sequences of items, each associated with multiple attributes. To address these challenges, we propose UniRec, a unified multimodal encoder for LLM-based recommendation. UniRec first employs modality-specific encoders to produce consistent embeddings across heterogeneous signals. It then adopts a triplet representation, comprising attribute name, type, and value, to separate schema from raw inputs and preserve semantic distinctions. Finally, a hierarchical Q-Former models the nested structure of user interactions while maintaining their layered organization. Across multiple real-world benchmarks, UniRec outperforms state-of-the-art multimodal and LLM-based recommenders by up to 15%, and extensive ablation studies further validate the contributions of each component.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27345v1">LLMs Capture Emotion Labels, Not Emotion Uncertainty: Distributional Analysis and Calibration of Human--LLM Judgment Gaps</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Human annotators frequently disagree on emotion labels, yet most evaluations of Large Language Model (LLM) emotion annotation collapse these judgments into a single gold standard, discarding the distributional information that disagreement encodes. We ask whether LLMs capture the structure of this disagreement, not just majority labels, by comparing emotion judgment distributions between human annotators and four zero-shot LLMs, plus a fine-tuned RoBERTa baseline, across two complementary benchmarks: GoEmotions and EmoBank, totaling 640,000 LLM responses. Zero-shot models diverge substantially from human distributions, and in-domain fine-tuning, not model scale, is required to close the gap. We formalize a lexical-grounding gradient through a quantitative transparency score that predicts per-category human--LLM agreement: LLMs reliably capture emotions with explicit lexical markers but systematically fail on pragmatically complex emotions requiring contextual inference, a pattern that replicates across both categorical and continuous emotion frameworks. We further propose three lightweight post-hoc calibration methods that reduce the distributional gap by up to 14\%, and provide actionable guidelines for when LLM emotion annotations can, and cannot, substitute for human labeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27340v1">Investigating More Explainable and Partition-Free Compositionality Estimation for LLMs: A Rule-Generation Perspective</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 Accepted at ACL 2026 main conference
    </div>
    <details class="paper-abstract">
      Compositional generalization tests are often used to estimate the compositionality of LLMs. However, such tests have the following limitations: (1) they only focus on the output results without considering LLMs' understanding of sample compositionality, resulting in explainability defects; (2) they rely on dataset partition to form the test set with combinations unseen in the training set, suffering from combination leakage issues. In this work, we propose a novel rule-generation perspective for compositionality estimation for LLMs. It requires LLMs to generate a program as rules for dataset mapping and provides estimates of the compositionality of LLMs using complexity-based theory. The perspective addresses the limitations of compositional generalization tests and provides a new way to analyze the compositionality characterization of LLMs. We conduct experiments and analysis of existing advanced LLMs based on this perspective on a string-to-grid task, and find various compositionality characterizations and compositionality deficiencies exhibited by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27335v1">Iterative Definition Refinement for Zero-Shot Classification via LLM-Based Semantic Prototype Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 Accepted at CVPR NeXD Workshop (2026)
    </div>
    <details class="paper-abstract">
      Web filtering systems rely on accurate web content classification to block cyber threats, prevent data exfiltration, and ensure compliance. However, classification is increasingly difficult due to the dynamic and rapidly evolving nature of the modern web. Embedding-based zero-shot approaches map content and category descriptions into a shared semantic space, enabling label assignment without labeled training data, but remain highly sensitive to definition quality. Poorly specified or ambiguous definitions create semantic overlap in the embedding space, leading to systematic misclassification. In this paper, we propose a training-free, adaptive iterative definition refinement framework that improves zero-shot web content classification by progressively optimizing category definitions rather than updating model parameters. Using LLMs as feedback-driven definition optimizers, we investigate three refinement strategies namely example-guided, confusion-aware, and history-aware, each refining class descriptions using structured signals from misclassified instances. Furthermore, we introduce a human-labeled benchmark of 10 URL categories with 1,000 samples per class and evaluate across 13 state-of-the-art embedding foundation models. Results demonstrate that iterative definition refinement consistently improves classification performance across diverse architectures, establishing definition quality as a critical and underexplored factor in embedding-based systems. The dataset is available at https://github.com/naeemrehmat/B2MWT-10C.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27321v1">Toward Autonomous SOC Operations: End-to-End LLM Framework for Threat Detection, Query Generation, and Resolution in Security Operations</a></div>
    <div class="paper-meta">
      📅 2026-04-30
    </div>
    <details class="paper-abstract">
      Security Operations Centers (SOCs) face mounting operational challenges. These challenges come from increasing threat volumes, heterogeneous SIEM platforms, and time-consuming manual triage workflows. We present an end-to-end threat management framework that integrates ensemble-based detection, syntax-constrained query generation, and retrieval-augmented resolution support to automate critical security workflows. Our detection module evaluates both traditional machine learning classifiers and large language models (LLMs), then combines the three best-performing LLMs to create an ensemble model, achieving 82.8% accuracy while maintaining 0.120 false positive rate on SIEM logs. We introduce the SQM (Syntax Query Metadata) architecture for automated evidence collection. It uses platform-specific syntax constraints, metadata-based retrieval, and documentation-grounded prompting to generate executable queries for IBM QRadar and Google SecOps. SQM achieves a BLEU score of 0.384 and a ROUGE-L score of 0.731. These results are more than twice as good as the baseline LLM performance. For incident resolution and recommendation generation, we demonstrate that integrating SQM-derived evidence improves resolution code prediction accuracy from 78.3% to 90.0%, with an overall recommendation quality score of 8.70. In production SOC environments, our framework reduces average incident triage time from hours to under 10 minutes. This work demonstrates that domain-constrained LLM architectures with retrieval augmentation can meet the strict reliability and efficiency requirements of operational security environments at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27319v1">REBENCH: A Procedural, Fair-by-Construction Benchmark for LLMs on Stripped-Binary Types and Names (Extended Version)</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 This is an extended version of our paper, which appears in AIWare 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable progress in recent years, driving their adoption across a wide range of domains, including computer security. In reverse engineering, LLMs are increasingly applied to critical tasks such as function and variable name recovery and type inference. However, despite the rapid growth of research in this area, progress has been hindered by the absence of a standardized dataset. Existing studies rely on disparate datasets, preprocessing pipelines, and evaluation metrics, making fair comparisons between approaches difficult and obscuring a clear understanding of LLM capabilities in binary analysis. To address these challenges, we present REBench, a comprehensive benchmark dataset for evaluating LLMs on binary reverse engineering tasks. REBench consolidates a superset of existing datasets, comprising hundreds of millions of lines of source code and a diverse collection of binaries spanning multiple architectures and optimization levels. REBench adopts a knowledge-base-driven methodology that stores byte-level stack information to generate ground truth, ensuring that task difficulty is preserved while maintaining universal applicability. This design enables fair evaluation across tasks while avoiding simplifications that could bias results. As a use case, we apply REBench to measure the reverse engineering performance of LLMs and the result demonstrates difficulties in complex tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27296v1">To Diff or Not to Diff? Structure-Aware and Adaptive Output Formats for Efficient LLM-based Code Editing</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 Accepted in the Findings of ACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for code editing, yet the prevalent full-code generation paradigm suffers from severe efficiency bottlenecks, posing challenges for interactive coding assistants that demand low latency and cost. Despite the predominant focus on scaling model capabilities, the edit format itself has been largely overlooked in model training. In this paper, we begin with a systematic study of conventional diff formats and reveal that fragile offsets and fragmented hunks make generation highly unnatural for LLMs. To address it, we introduce BlockDiff and FuncDiff, two structure-aware diff formats that represent changes as block-level rewrites of syntactically coherent units such as control structures and functions. Furthermore, we propose AdaEdit, a general adaptive edit strategy that trains LLMs to dynamically choose the most token-efficient format between a given diff format and full code. Extensive experiments demonstrate that AdaEdit paired with structure-aware diff formats consistently matches the accuracy of full-code generation, while reducing both latency and cost by over 30% on long-code editing tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27283v1">Learning When to Remember: Risk-Sensitive Contextual Bandits for Abstention-Aware Memory Retrieval in LLM-Based Coding Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 26 pages, 7 figures, 10 tables. Code and deterministic local artifacts are available at the repository listed in the paper
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based coding agents increasingly rely on external memory to reuse prior debugging experience, repair traces, and repository-local operational knowledge. However, retrieved memory is useful only when the current failure is genuinely compatible with a previous one; superficial similarity in stack traces, terminal errors, paths, or configuration symptoms can lead to unsafe memory injection. This paper reframes issue-memory use as a selective, risk-sensitive control problem rather than a pure top-k retrieval problem. We introduce RSCB-MC, a risk-sensitive contextual bandit memory controller that decides whether an agent should use no memory, inject the top resolution, summarize multiple candidates, perform high-precision or high-recall retrieval, abstain, or ask for feedback. The system stores reusable issue knowledge through a pattern-variant-episode schema and converts retrieval evidence into a fixed 16-feature contextual state capturing relevance, uncertainty, structural compatibility, feedback history, false-positive risk, latency, and token cost. Its reward design penalizes false-positive memory injection more strongly than missed reuse, making non-injection and abstention first-class safety actions. In deterministic smoke-scale artifacts, RSCB-MC obtains the strongest non-oracle offline replay success rate, 62.5%, while maintaining a 0.0% false-positive rate. In a bounded 200-case hot-path validation, it reaches 60.5% proxy success with 0.0% false positives and a 331.466 microseconds p95 decision latency. The results show that, for coding-agent memory, the key question is not only which memory is most similar, but whether any retrieved memory is safe enough to influence the debugging trajectory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27273v1">Few-Shot Accent Synthesis for ASR with LLM-Guided Phoneme Editing</a></div>
    <div class="paper-meta">
      📅 2026-04-30
      | 💬 Submitted to Interspeech 2026
    </div>
    <details class="paper-abstract">
      Accented automatic speech recognition (ASR) often degrades due to the limited availability of accented training data. Prior work has explored accent modeling in low-resource settings, but existing approaches typically require minutes to hours of labeled speech, which may still be impractical for truly scarce accent scenarios. We propose a pipeline that adapts a text-to-speech (TTS) decoder to a target-accent speaker using fewer than ten reference utterances and employs large language model (LLM)-based phoneme editing to generate accent-conditioned pronunciations. The resulting synthetic speech is used to fine-tune a self-supervised ASR model. Experiments demonstrate consistent word error rate (WER) reductions on real accented speech, including cross-speaker evaluation and ultra-low data regimes. A matched-rate random phoneme baseline shows that phoneme-space perturbation itself is a strong form of augmentation, while LLM-guided edits provide additional gains through accent-conditioned structure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27267v1">From Prompt to Physical Actuation: Holistic Threat Modeling of LLM-Enabled Robotic Systems</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 Submitted to 23rd Annual International Conference on Privacy, Security, and Trust (PST2026)
    </div>
    <details class="paper-abstract">
      As large language models are integrated into autonomous robotic systems for task planning and control, compromised inputs or unsafe model outputs can propagate through the planning pipeline to physical-world consequences. Although prior work has studied robotic cybersecurity, adversarial perception attacks, and LLM safety independently, no existing study traces how these threat categories interact and propagate across trust boundaries in a unified architectural model. We address this gap by modeling an LLM-enabled autonomous robot in an edge-cloud architecture as a hierarchical Data Flow Diagram and applying STRIDE-per-interaction analysis across six boundary-crossing interaction points using a three-category taxonomy of Conventional Cyber Threats, Adversarial Threats, and Conversational Threats. The analysis reveals that these categories converge at the same boundary crossings, and we trace three cross-boundary attack chains from external entry points to unsafe physical actuation, each exposing a distinct architectural property: the absence of independent semantic validation between user input and actuator dispatch, cross-modal translation from visual perception to language-model instruction, and unmediated boundary crossing through provider-side tool use. To our knowledge, this is the first DFD-based threat analysis integrating all three threat categories across the full perception-planning-actuation pipeline of an LLM-enabled robotic system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.23341v2">Evaluating Jailbreaking Vulnerabilities in LLMs Deployed as Assistants for Smart Grid Operations: A Benchmark Against NERC Standards</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models (LLMs) as assistants in electric grid operations promises to streamline compliance and decision-making but exposes new vulnerabilities to prompt-based adversarial attacks. This paper evaluates the risk of jailbreaking LLMs, i.e., circumventing safety alignments to produce outputs violating regulatory standards, assuming threats from authorized users, such as operators, who craft malicious prompts to elicit non-compliant guidance. Three state-of-the-art LLMs (OpenAI's GPT-4o mini, Google's Gemini 2.0 Flash-Lite, and Anthropic's Claude 3.5 Haiku) were tested against Baseline, BitBypass, and DeepInception jailbreaking methods across scenarios derived from nine NERC Reliability Standards (EOP, TOP, and CIP). In the initial broad experiment, the overall Attack Success Rate (ASR) was 33.1%, with DeepInception proving most effective at 63.17% ASR. Claude 3.5 Haiku exhibited complete resistance (0% ASR), while Gemini 2.0 Flash-Lite was most vulnerable (55.04% ASR) and GPT-4o mini moderately susceptible (44.34% ASR). A follow-up experiment refining malicious wording in Baseline and BitBypass attacks yielded a 30.6% ASR, confirming that subtle prompt adjustments can enhance simpler methods' efficacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27249v1">Instruction Complexity Induces Positional Collapse in Adversarial LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 12 pages, 3 figures, 3 tables. Pre-registered on OSF (osf.io/7p64)
    </div>
    <details class="paper-abstract">
      When instructed to underperform on multiple-choice evaluations, do language models engage with question content or fall back on positional shortcuts? We map the boundary between these regimes using a six-condition adversarial instruction-specificity gradient administered to two instruction-tuned LLMs (Llama-3-8B and Llama-3.1-8B) on 2,000 MMLU-Pro items. Distributional screening (response-position entropy) and an independent content-engagement criterion (difficulty-accuracy correlation) jointly characterise each condition. The gradient reveals three regimes rather than a monotonic transition. Vague adversarial instructions produce moderate accuracy reduction with preserved content engagement. Standard sandbagging and capability-imitation instructions produce positional entropy collapse with partial content engagement. A two-step answer-aware avoidance instruction produces extreme positional collapse, with near-total concentration on a single response position (99.9% and 87.4%) and no measurable content sensitivity. This was the only multi-step instruction tested, and it produced the most extreme shortcut. The attractor position matches each model's content-absent null-prompt default. The effect replicates across both models and four academic domains. Distributional collapse and content engagement can co-occur (50% concordance between screening criteria), indicating that entropy-based screening and difficulty-based content assessment capture partially independent dimensions of response validity. Results suggest that instruction complexity can determine whether adversarial compliance uses content-aware or content-blind mechanisms in small instruction-tuned LLMs under greedy decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27238v1">SafeTune: Mitigating Data Poisoning in LLM Fine-Tuning for RTL Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 This paper will be presented at IEEE VLSI Test Symposium (VTS) 2026. 5 pages, 2 figures, 4 tables
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly fine-tuned for hardware tasks like RTL code generation, the scarcity of high-quality datasets often leads to the use of rapidly assembled or generated training data. These datasets frequently lack security verification and are highly susceptible to data poisoning attacks. Such poisoning can cause models to generate syntactically valid but insecure hardware modules that bypass standard functionality checks. To address this, we present SafeTune, a framework designed to harden LLM-based RTL generation against poisoning, specifically focusing on hardware Trojan (HT) insertion. SafeTune integrates two core components: (i) a Graph Neural Network (GNN) that models structural properties to identify anomalous circuitry patterns during fine-tuning, and (ii) a semantic verification module using text embeddings and an XGBoost classifier to assess prompt security. By coupling structural and semantic knowledge, SafeTune effectively filters poisoned inputs without sacrificing legitimate data. Experimental results demonstrate that SafeTune significantly enhances the robustness and reliability of LLM fine-tuning without requiring modifications to the underlying model architecture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27228v1">When Roles Fail: Epistemic Constraints on Advocate Role Fidelity in LLM-Based Political Statement Analysis</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Democratic discourse analysis systems increasingly rely on multi-agent LLM pipelines in which distinct evaluator models are assigned adversarial roles to generate structured, multi-perspective assessments of political statements. A core assumption is that models will reliably maintain their assigned roles. This paper provides the first systematic empirical test of that assumption using the TRUST pipeline. We develop an epistemic stance classifier that identifies advocate roles from reasoning text without relying on surface vocabulary, and measure role fidelity across 60 political statements (30 English, 30 German) using four metrics: Role Drift Index (RDI), Expected Drift Distance (EDD), Directional Drift Index (DDI), and Entropy-based Role Stability (ERS). We identify two failure modes - the Epistemic Floor Effect (fact-check results create an absolute lower bound below which the legitimizing role cannot be maintained) and Role-Prior Conflict (training-time knowledge overrides role instructions for factually unambiguous statements) - as manifestations of a single mechanism: Epistemic Role Override (ERO). Model choice significantly affects role fidelity: Mistral Large outperforms Claude Sonnet by 28pp (67% vs. 39%) and exhibits a qualitatively different failure mode - role abandonment without polarity reversal - compared to Claude's active switch to the opposing stance. Role fidelity is language-robust. Fact-check provider choice is not universally neutral: Perplexity significantly reduces Claude's role fidelity on German statements (Delta = -15pp, p = 0.007) while leaving Mistral unaffected. These findings have direct implications for multi-agent LLM validation: a system validated without role fidelity measurement may systematically misrepresent the epistemic diversity it was designed to provide.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17056v2">From Test-taking to Cognitive Scaffolding: A Pedagogical Diagnostic Benchmark for LLMs on English Standardized Tests</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly integrated into educational tools, current evaluations on standardized tests predominantly focus on binary outcome accuracy. Instead, an effective AI tutor must exhibit faithful reasoning, elucidate solution strategies, and diagnose specific human misconceptions. To bridge this gap, we introduce a pedagogical diagnostic framework that models English Standardized Test (EST) problem-solving as a traversal through a cognitive framework. Based on this framework, we present ESTBook, a multimodal benchmark encompassing 10,576 questions and 29 task types across five major exams. Unlike traditional datasets, ESTBook goes beyond data aggregation by enriching questions with formalized reasoning trajectories and distractor rationales that capture specific cognitive traps. Through extensive evaluations, we empirically demonstrate the practical utility of our diagnostic framework, showing that identifying cognitive trajectories facilitates the mitigation of performance gap and improves pedagogical reasoning through guided elicitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.16516v2">Supercharging Agenda Setting Research: The ParlaCAP Dataset of 28 European Parliaments and a Scalable Multilingual LLM-Based Classification</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 17 pages, 7 figures, 7 tables. Presented in the PoliticalNLP 2026 workshop, co-located with LREC 2026 conference
    </div>
    <details class="paper-abstract">
      This paper introduces ParlaCAP, a large-scale dataset for analyzing parliamentary agenda setting across Europe, and proposes a cost-effective method for building domain-specific policy topic classifiers. Applying the Comparative Agendas Project (CAP) schema to the multilingual ParlaMint corpus of over 8 million speeches from 28 parliaments of European countries and autonomous regions, we follow a teacher-student framework in which a high-performing large language model (LLM) annotates in-domain training data and a multilingual encoder model is fine-tuned on these annotations for scalable data annotation. We show that this approach produces a classifier tailored to the target domain. Agreement between the LLM and human annotators is comparable to inter-annotator agreement among humans, and the resulting model outperforms existing CAP classifiers trained on manually-annotated but out-of-domain data. In addition to the CAP annotations, the ParlaCAP dataset offers rich speaker and party metadata, as well as sentiment predictions coming from the ParlaSent multilingual transformer model, enabling comparative research on political attention and representation across countries. We illustrate the analytical potential of the dataset with three use cases, examining the distribution of parliamentary attention across policy topics, sentiment patterns in parliamentary speech, and gender differences in policy attention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27143v1">Enhancing Linux Privilege Escalation Attack Capabilities of Local LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Recent research has demonstrated the potential of Large Language Models (LLMs) for autonomous penetration testing, particularly when using cloud-based restricted-weight models. However, reliance on such models introduces security, privacy, and sovereignty concerns, motivating the use of locally hosted open-weight alternatives. Prior work shows that small open-weight models perform poorly on automated Linux privilege escalation, limiting their practical applicability. In this paper, we present a systematic empirical study of whether targeted system-level and prompting interventions can bridge this performance gap. We analyze failure modes of open-weight models in autonomous privilege escalation, map them to established enhancement techniques, and evaluate five concrete interventions (chain-of-thought prompting, retrieval-augmented generation, structured prompts, history compression, and reflective analysis) implemented as extensions to hackingBuddyGPT. Our results show that open-weight models can match or outperform cloud-based baselines such as GPT-4o. With our treatments enabled, Llama3.1 70B exploits 83% of tested vulnerabilities, while smaller models including Llama3.1 8B and Qwen2.5 7B achieve 67% when using guidance. A full-factorial ablation study over all treatment combinations reveals that reflection-based treatments contribute most, while also identifying vulnerability discovery as a remaining bottleneck for local models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27131v1">LLM-Enhanced Topical Trend Detection at Snapchat</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Automatic detection of topical trends at scale is both challenging and essential for maintaining a dynamic content ecosystem on social media platforms. In this work, we present a large-scale system for identifying emerging topical trends on Snapchat, one of the world's largest short-video social platforms. Our system integrates multimodal topic extraction, time-series burst detection, and LLM-based consolidation and enrichment to enable accurate and timely trend discovery. To the best of our knowledge, this is the first published end-to-end system for topical trend detection on short-video platforms at production scale. Continuous offline human evaluation over six months demonstrates high precision in identifying meaningful trends. The system has been deployed in production at global scale and applied to downstream surfaces including content ranking and search, driving measurable improvements in content freshness and user experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.11581v4">Hidden Measurement Error in LLM Pipelines Distorts Annotation, Evaluation, and Benchmarking</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      LLM evaluations drive which models get deployed, which safety standards get adopted, and which research conclusions get published. Yet standard confidence intervals ignore variability from prompt phrasing, model temperature, and judge model choice, producing under-coverage that worsens with more data. The omitted variance can shift results enough to reverse conclusions \citep{baumann2025llmhacking, huang2026dropping} and opens benchmarks to exploitation \citep{singh2025leaderboard}. This paper decomposes LLM pipeline uncertainty into its sources, distinguishes variance that shrinks with more data from sensitivity to researcher design choices, and uses design-study projections to reduce total evaluation error (TEE). Across the demonstrations, naive standard errors are 40 - 60\% smaller than the TEE-corrected SE. Using Chatbot Arena data, we show naive 95\% CI coverage drops as $n$ grows while TEE-corrected coverage holds at 95\%. We show further that a small pilot recovers honest CIs and projects which design changes most improve precision. Acting on those projections halves MMLU estimation error against the answer key at equivalent cost, and on a human-validated propaganda audit the TEE-recommended pipeline outperforms 73\% of single-configuration alternatives against the 9-rater human baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27089v1">AutoSP: Unlocking Long-Context LLM Training Via Compiler-Based Sequence Parallelism</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 13 pages, 9 figures, 1 table
    </div>
    <details class="paper-abstract">
      Large-language-models (LLMs) demonstrate enormous utility in long-context tasks which require processing prompts that consist of tens to hundreds of thousands of tokens. However, existing LLM training libraries do not provide easy to use abstractions to optimize for long-context training, instead focusing on optimizations for models with large parameter counts through ZeRO-3/FSDP, Tensor and Pipeline parallelism. This forces users to rewrite LLM training libraries to incorporate compositions of various complex long-context optimizations, such as sequence-parallelism, to training pipelines; a process that requires in-depth expertise, reducing developer productivity. To tackle these challenges, we introduce AutoSP: the first automated solution to automatically optimize LLM training for longer-contexts. AutoSP compiles models and applies a targeted set of optimizations: automated sequence parallelism, and long-context aware activation-checkpointing, to drastically enhance LLM trainability at negligible cost to throughput. Our evaluation demonstrates AutoSP's capability on both NVIDIA and AMD hardware, increasing training contexts by upto 2.7$\times$ and 2.5$\times$ respectively over competitive hand-written baseline at negligible cost to runtime performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27082v1">When Your LLM Reaches End-of-Life: A Framework for Confident Model Migration in Production Systems</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 12 pages with appendix
    </div>
    <details class="paper-abstract">
      We present a framework for migrating production Large Language Model (LLM) based systems when the underlying model reaches end-of-life or requires replacement. The key contribution is a Bayesian statistical approach that calibrates automated evaluation metrics against human judgments, enabling confident model comparison even with limited manual evaluation data. We demonstrate this framework on a commercial question-answering system serving 5.3M monthly interactions across six global regions; evaluating correctness, refusal behavior, and stylistic adherence to successfully identify suitable replacement models. The framework is broadly applicable to any enterprise deploying LLM-based products, providing a principled, reproducible methodology for model migration that balances quality assurance with evaluation efficiency. This is a capability increasingly essential as the LLM ecosystem continues to evolve rapidly and organizations manage portfolios of AI-powered services across multiple models, regions, and use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23163v3">A Decision-Theoretic Formalisation of Steganography With Applications to LLM Monitoring</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 First two authors contributed equally
    </div>
    <details class="paper-abstract">
      Large language models are beginning to show steganographic capabilities. Such capabilities could allow misaligned models to evade oversight mechanisms. Yet principled methods to detect and quantify such behaviours are lacking. Classical definitions of steganography, and detection methods based on them, require a known reference distribution of non-steganographic signals. For the case of steganographic reasoning in LLMs, knowing such a reference distribution is not feasible; this renders these approaches inapplicable. We propose an alternative, \textbf{decision-theoretic view of steganography}. Our central insight is that steganography creates an asymmetry in usable information between agents who can and cannot decode the hidden content (present within a steganographic signal), and this otherwise latent asymmetry can be inferred from the agents' observable actions. To formalise this perspective, we introduce generalised $\mathcal{V}$-information: a utilitarian framework for measuring the amount of usable information within some input. We use this to define the \textbf{steganographic gap} -- a measure that quantifies steganography by comparing the downstream utility of the steganographic signal to agents that can and cannot decode the hidden content. We empirically validate our formalism, and show that it can be used to detect, quantify, and mitigate steganographic reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.26184v5">Auto-ARGUE: LLM-Based Report Generation Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 SIGIR 2026: Demo Track
    </div>
    <details class="paper-abstract">
      Generation of citation-backed reports is a primary use case for retrieval-augmented generation (RAG) systems. While open-source evaluation tools exist for various RAG tasks, tools designed for report generation are lacking. Accordingly, we introduce Auto-ARGUE, a robust LLM-based implementation of the recently proposed ARGUE framework for report generation evaluation. We present analysis of Auto-ARGUE on the report generation pilot task from the TREC 2024 NeuCLIR track and on two tasks from the TREC 2024 RAG track, showing good system-level correlations with human judgments. Additionally, we release ARGUE-Viz, a web app for visualization and fine-grained analysis of Auto-ARGUE judgments and scores.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04214v3">Teaching LLM to be Persuasive: Reward-Enhanced Policy Optimization for Alignment from Heterogeneous Rewards</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 accepted by ACL 2026 indusry track
    </div>
    <details class="paper-abstract">
      We deploy large language models (LLMs) as business development (BD) agents for persuasive price negotiation in online travel agencies (OTAs). The agent must follow a multi-stage Standard Operating Procedure (SOP) and strict guardrails (no over-promising and no hallucinations), while remaining human-like and effective over long, multi-turn dialogues. We propose Reward-Enhanced Policy Optimization (REPO), a reinforcement learning post-training method that combines heterogeneous rewards: a preference-trained reward model (RM), an LLM-as-a-judge (RJ) for nuanced behaviors (e.g., emotional value and SOP compliance), and rule-based reward functions (RF) (mainly regex-based) for deterministic checks on numerics, formatting, and guardrails. In expert consensus evaluation (three human experts; 30 online conversations and 45 curated bad cases), REPO improves average dialogue rating to 4.63 (+0.33 over GRPO) and raises the share of conversations with at least one excellent response to 66.67% (+23.34 pp over GRPO), while achieving a 93.33% bad-case fix rate with 75.56% clean fixes. In a production A/B test on 9,653 real customer conversations (vs. an intent-driven dialogue system), REPO improves response rate by +12.14 pp and task success rate by +5.94 pp (p<0.001).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26837v1">Unifying Sparse Attention with Hierarchical Memory for Scalable Long-Context LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Long-context LLM serving is bottlenecked by the cost of attending over ever-growing KV caches. Dynamic sparse attention promises relief by accessing only a small, query-dependent subset of the KV state per decoding step and extending the KV storage to CPU memory. In practice, however, these algorithmic savings rarely translate into end-to-end system-level gains because sparse methods typically operate at different granularities and thus rely on ad hoc, per-algorithm implementations. At the same time, hierarchical KV storage introduces a new systems bottleneck: retrieving fine-grained, irregular KV subsets across the GPU-CPU boundary can easily erase the benefits of sparsity. We present SPIN, a sparse-attention-aware inference framework that co-designs the execution pipeline with hierarchical KV storage through three techniques: (1) a unified partition abstraction that maps different sparsity granularities onto a shared page-based KV substrate; (2) a locality-aware KV cache manager that dynamically sizes per-request HBM budgets and uses a GPU-friendly bucketed LRU policy to cut PCIe round-trips; and (3) a two-level hierarchical metadata layout sized to the active working set rather than the worst-case address space. Built on vLLM with three representative sparse attention algorithms, SPIN delivers 1.66-5.66x higher end-to-end throughput and 7-9x lower TTFT than vLLM, and reduces TPOT by up to 58% over the original sparse-attention implementations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26821v1">Exploring the Efficiency of 3D-Stacked AI Chip Architecture for LLM Inference with Voxel</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      To overcome the well-known memory bottleneck of AI chips, 3D stacked architectures that employ advanced packaging technology with high-density through-silicon vias (TSVs) pins have proven to be a promising solution. The 3D-stacked AI chip enables ultra-high memory bandwidth between compute and memory by stacking numerous DRAM banks atop many AI cores in a distributed manner. However, it is not easy to explore the efficiency of the 3D-stacked AI chip, due to its unique distributed nature. And we need to carefully consider multiple intertwined factors that range from upper-level computing paradigm to machine learning (ML) compiler optimizations, and to the underlying hardware architecture. In this paper, we develop Voxel, a fast and compiler-aware end-to-end simulation framework to facilitate exploring the efficiency of 3D-stacked AI chips for large language model (LLM) inference. Voxel enables the software/hardware co-exploration by employing a programming interface that allows ML compilers to customize the model execution plans. After validating the results of Voxel with an emulator on real silicon, we thoroughly examine the impact and correlation of different aspects of 3D-stacked AI chips, including state-of-the-art compute paradigms, tile-to-core mapping, tensor-to-bank mapping, NoC topologies and link bandwidth, DRAM bank bandwidth, per-core SRAM capacity, and energy/thermal constraints. Our findings disclose that the end-to-end efficiency of a 3D stacked AI chip not only is determined by the cooperative function of these factors, but also significantly depends on the mappings from tiles to AI core and DRAM banks. We report our findings throughout the paper, with the expectation that they will shed light on the development of the 3D-stacked AI chip ecosystem. We will open source Voxel and our study results for public research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.20133v2">Reasoning Gets Harder for LLMs Inside A Dialogue</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 Accepted at ACL 2026 (Main)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve strong performance on many reasoning benchmarks, yet these evaluations typically focus on isolated tasks that differ from real-world usage in task-oriented dialogue (TOD). In this setting, LLMs must perform reasoning inherently while generating text and adhering to instructions on role, format, and style. This mismatch raises concerns about whether benchmark performance accurately reflects models' reasoning robustness in TOD setting. We investigate how framing reasoning tasks within TOD affects LLM performance by introducing BOULDER, a new dynamic benchmark covering eight travel-related tasks that require arithmetic, spatial, and temporal reasoning with both commonsense and formal aspects. Each problem is presented in both isolated and dialogue-based variants, enabling controlled comparison while mitigating data contamination. Experiments on eight LLMs reveal a substantial and consistent performance gap between isolated and dialogue settings. Through ablations and qualitative analysis, we show that this gap is largely driven by the multi-turn nature of dialogue, with additional effects from role conditioning and tool-use requirements. Our results highlight the need to evaluate LLM reasoning in realistic interactive scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24013v2">CommFuse: Hiding Tail Latency via Communication Decomposition and Fusion for Distributed LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 Slightly modified the title, and corresponding minor wording change in the content
    </div>
    <details class="paper-abstract">
      The rapid growth in the size of large language models has necessitated the partitioning of computational workloads across accelerators such as GPUs, TPUs, and NPUs. However, these parallelization strategies incur substantial data communication overhead significantly hindering computational efficiency. While communication-computation overlap presents a promising direction, existing data slicing based solutions suffer from tail latency. To overcome this limitation, this research introduces a novel communication-computation overlap technique to eliminate this tail latency in state of the art overlap methods for distributed LLM training. The aim of this technique is to effectively mitigate communication bottleneck of tensor parallelism and data parallelism for distributed training and inference. In particular, we propose a novel method termed CommFuse that replaces conventional collective operations of reduce-scatter and all-gather with decomposed peer-to-peer (P2P) communication and schedules partitioned computations to enable fine-grained overlap. Our method provides an exact algorithm for reducing communication overhead that eliminates tail latency. Moreover, it presents a versatile solution compatible with data-parallel training and various tensor-level parallelism strategies, including TPSP and UP. Experimental evaluations demonstrate that our technique consistently achieves lower latency, superior Model FLOPS Utilization (MFU), and high throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27032v1">LLM-Guided Runtime Parameter Optimization for Energy-Efficient Model Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 8 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become an integral part of many real-world workflows. However, LLMs consume a lot of energy, which becomes a large concern in the scale of the demand for these tools. As LLMs become integrated into different workflows, different applications have arisen to deal with the challenge of running inference for these tools. This raises another issue of choosing the runtime parameter values for these services in order to minimize the energy consumption. Oftentimes this requires deep knowledge of the application or traditional optimization methods that can take days to find optimal values. In this work, we created a human-in-the-loop flow with LLM-assisted runtime parameter optimization in order to solve this issue. With human-created, specific feedback prompting methods, chat-based LLMs can iteratively find energy-efficient inference parameters faster than traditional search methods. LLMs can also tailor their solutions to different hardware setups and easily take into account other system constraints. The enhanced prompt template was able to converge below the threshold at an average of 3.4 prompts compared to the baseline, which converged in an average of 5.2 prompts, and consistently achieved lower final energy per token. The enhanced prompt template also outperformed Sobol sampling in convergence speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26760v1">Factorized Latent Reasoning for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently been adopted for recommendation by framing user preference modeling as a language generation problem. However, existing latent reasoning approaches typically represent user intent with a single latent vector, which struggles to capture the inherently multi-faceted nature of user preferences. We propose Factorized Latent Reasoning (FLR), a novel framework for LLM-based sequential recommendation that decomposes latent reasoning into multiple disentangled preference factors. FLR introduces a lightweight multi-factor attention module that iteratively refines a latent thought representation, where each factor attends to distinct aspects of the user's interaction history. To encourage diversity and specialization, we design orthogonality, attention diversity, and sparsity regularization objectives, and dynamically aggregate factor contributions for the final prediction. We further integrate FLR with an efficient reinforcement learning strategy based on group-relative policy optimization, enabling stable alignment directly in the latent reasoning space. Experiments on multiple benchmarks show that FLR consistently outperforms strong baselines while improving robustness and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04389v2">Safety Is Not Universal: The Selective Safety Trap in LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 9 pages, 5 figures and 4 tables in paper (more in appendix)
    </div>
    <details class="paper-abstract">
      Current safety evaluations of large language models (LLMs) create a dangerous illusion of universal protection by aggregating harms under generic categories such as "Identity Hate", obscuring vulnerabilities toward specific populations. In this work, we expose the Selective Safety Trap: a systemic failure mode where models robustly defend specific populations while leaving underrepresented communities highly vulnerable to identical adversarial attacks. To systematically audit this phenomenon, we introduce MiJaBench, a bilingual (English-Portuguese) adversarial benchmark comprising 43,961 controlled jailbreaking prompts across 16 minority groups. By evaluating 14 state-of-the-art LLMs on MiJaBench, we curate 615,454 prompt-response pairs that compose MiJaBench-Align, revealing that safety alignment is not a uniform semantic capability but a demographic hierarchy, with defense rates fluctuating by up to 42% within the same model solely based on the target group. This disparity persists across architectures and languages and is amplified by scaling, indicating that current alignment methods learn group-specific safeguards rather than a generalized notion of harm. Through targeted direct preference optimization (DPO) on a 1B-parameter baseline, we achieve strong zero-shot safety generalizations to entirely unseen demographics and complex attack strategies. We release all datasets and scripts to provide the community with a concrete pathway toward equitable, transferable safety alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26679v1">MultEval: Supporting Collaborative Alignment for LLM-as-a-Judge Evaluation Criteria</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 17 pages, 5 figures
    </div>
    <details class="paper-abstract">
      LLM-as-a-judge approaches have emerged as a scalable solution for evaluating model behaviors, yet they rely on evaluation criteria often created by a single individual, embedding that person's assumptions, priorities, and interpretive lens. In practice, defining such criteria is a collaborative and contested process involving multiple stakeholders with different values, interpretations, and priorities; an aspect largely unsupported by existing tools. To examine this problem in depth, we present a formative study examining how stakeholders collaboratively create, negotiate, and refine evaluation criteria for LLM-as-a-judge systems. Our findings reveal challenges in human oversight, including difficulties in establishing shared understanding, aligning values across stakeholders with different expertise and priorities, and translating nuanced human judgments into criteria that are interpretable and actionable for LLM judges. Based on these insights, we developed MultEval, a system that supports collaborative criteria by enabling multiple evaluators to surface and diagnose disagreements using consensus-building theory, iteratively revise criteria with attached examples and proposal history, and maintain transparency over how judgments are encoded into an automated evaluator. We further report a case study in which a team of domain experts used MultEval to collaboratively author criteria, illustrating how coordination and collaborative consensus-making shape criteria evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15983v2">ReLoop: Structured Modeling and Behavioral Verification for Reliable LLM-Based Optimization</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 Code and benchmark: https://github.com/junbolian/ReLoop
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can translate natural language into optimization code, but silent failures pose a critical risk: code that executes and returns solver-feasible solutions may encode semantically incorrect formulations -- a feasibility-correctness gap reaching 90 percentage points on compositional problems. We introduce ReLoop, which addresses this gap through two complementary mechanisms. Structured generation decomposes code production into a four-stage reasoning chain (understand, formalize, synthesize, verify), preventing formulation errors at their source. Behavioral verification detects errors that survive generation by testing whether the formulation responds correctly to solver-based parameter perturbation -- an external semantic signal that bypasses LLM self-review and requires no ground truth. The two mechanisms are complementary by error structure: structured generation drives the largest gains on compositional problems (+8.5pp accuracy on RetailOpt-190 with Claude Opus 4.6), while behavioral verification dominates on localized defects (+4.4pp on MAMO-ComplexLP, its largest contribution across benchmarks). Combined with diagnostic execution recovery, ReLoop reaches 100% executable code on Claude Opus 4.6 and consistently improves accuracy on chat-tuned foundation models across three benchmarks; we further identify a known limitation of narrowly-tuned SFT models, whose learned output formats are brittle to chain-of-thought prompts -- an interaction we document and analyze. We release RetailOpt-190, 190 compositional retail optimization scenarios targeting the multi-constraint interactions where LLMs most frequently fail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.22063v2">Reliability Auditing for Downstream LLM tasks in Psychiatry: LLM-Generated Hospitalization Risk Scores</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly utilized in clinical reasoning and risk assessment. However, their interpretive reliability in critical and indeterminate domains such as psychiatry remains unclear. Prior work has identified algorithmic biases and prompt sensitivity in these systems, raising concerns about how contextual information may influence model outputs, but there remains no systematic way to assess these, especially in the psychiatric domain. We propose an approach for reliability auditing downstream LLM tasks by structuring evaluation around the impact of prompt design and the inclusion of medically insignificant inputs on predicted hospitalization risk scores, which is often the first downstream AI clinical-decision-making task. In our audit, a cohort of synthetic patient profiles (n = 50) is generated, each consisting of 15 clinically relevant features and up to 50 clinically insignificant features, across four prompt reframings (neutral, logical, human impact, clinical judgment). We audit four LLMs (Gemini 2.5 Flash, LLaMa 3.3 70b, Claude Sonnet 4.6, GPT-4o mini), and our results show that including medically insignificant variables resulted in a statistically significant increase in the absolute mean predicted hospitalization risk and output variability across all models and prompts, indicating reduced predictive stability as contextual noise increased. Clinically insignificant features had an effect on instability across many model-prompt conditions, and prompt variations independently affected the trajectory of instability in a model-dependent manner. These findings quantify how LLM-based psychiatric risk assessments are sensitive to non-clinical information, highlighting the need for systematic evaluations of attributional stability and uncertainty behavior like this before clinical deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27022v1">Breaking Bad Financial Habits: How LLM Conversations Correct Financial Misconceptions</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Financial misconceptions carry direct economic costs, from panic selling to equity market avoidance, yet they are notoriously resistant to correction. Traditional financial literacy interventions are constrained by cost, reach, and a persistent gap between knowledge and behavioral change. Across three pre-registered studies, we find that purposefully designed LLMs can durably correct financial misconceptions. Critically, two factors are necessary for this effect. First, corrective intent: LLMs prompted only to discuss a misconception produce corrections no better than unassisted self-reflection, and undirected LLM conversations can actively entrench misconceptions. Second, recipient receptivity: financial concepts are often foreign to the investors who misapply them, and LLM responses pitched below a participant's financial sophistication are judged as less credible and produce substantially weaker corrections. LLMs thus offer a scalable alternative to traditional financial literacy intervention, but only when designed with both factors in mind.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.17612v2">Provable Coordination for LLM Agents via Message Sequence Charts</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 40 pages; v2: All definitions and results are now mechanically verified in Lean 4
    </div>
    <details class="paper-abstract">
      Multi-agent systems built on large language models (LLMs) are difficult to reason about. Coordination errors such as deadlocks or type-mismatched messages are often hard to detect through testing. We introduce a domain-specific language for specifying agent coordination based on message sequence charts (MSCs). The language separates message-passing structure from LLM actions, whose outputs remain unpredictable. We define the syntax and semantics of the language and present a syntax-directed projection that generates deadlock-free local agent programs from global coordination specifications. We illustrate the approach with a diagnosis consensus protocol and show how coordination properties can be established independently of LLM nondeterminism. We also describe a runtime planning extension in which an LLM dynamically generates a coordination workflow for which the same structural guarantees apply. An open-source Python implementation of our framework is available as ZipperGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26607v1">Human-in-the-Loop Benchmarking of Heterogeneous LLMs for Automated Competency Assessment in Secondary Level Mathematics</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 5 pages, 3 figures, 5 tables. Submitted to 2AI-2026-Applied AI Conference
    </div>
    <details class="paper-abstract">
      As Competency-Based Education (CBE) is gaining traction around the world, the shift from marks-based assessment to qualitative competency mapping is a manual challenge for educators. This paper tackles the bottleneck issue by suggesting a "Human-in-the-Loop" benchmarking framework to assess the effectiveness of multiple LLMs in automating secondary-level mathematics assessment. Based on the Grade 10 Optional Mathematics curriculum in Nepal, we created a multi-dimensional rubric for four topics and four cross-cutting competencies: Comprehension, Knowledge, Operational Fluency, and Behavior and Correlation. The multi-provider ensemble, consisted of open-weight models -- Eagle (Llama 3.1-8B) and Orion (Llama 3.3-70B) -- and proprietary frontier models Nova (Gemini 2.5 Flash) and Lyra (Gemini 3 Pro), was benchmarked against a ground truth defined by two senior mathematics faculty members (kappa_w = 0.8652). The findings show a marked "Architecture-compatibility gap". Although the Gemini-based Mixture-of-Experts (Sparse MoE) models achieved "Fair Agreement" (kappa_w ~ 0.38), the larger Orion (70B) model exhibited "No Agreement" (kappa_w = -0.0261), suggesting that architectural compliance with instruction constraints outweighs the scale of raw parameters in rubric-constrained tasks. We conclude that while LLMs are not yet suitable for autonomous certification, they provide high-value assistive support for preliminary evidence extraction within a "Human-in-the-Loop" framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26591v1">MappingEvolve: LLM-Driven Code Evolution for Technology Mapping</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Technology mapping is a critical yet challenging stage in logic synthesis. While Large Language Models (LLMs) have been applied to generate optimization scripts, their potential for core algorithm enhancement remains untapped. We introduce MappingEvolve, an open-source framework that pioneers the use of LLMs to directly evolve technology mapping code. Our method abstracts the mapping process into distinct optimization operators and employs a hierarchical agent-based architecture, comprising a Planner, Evolver, and Evaluator, to guide the evolutionary search. This structured approach enables strategic and effective code modifications. Experiments show our method significantly outperforms direct evolution and strong baselines, achieving 10.04\% area reduction versus ABC and 7.93\% versus mockturtle, with 46.6\%--96.0\% $S_{overall}$ improvement on EPFL benchmarks, while explicitly navigating the area--delay trade-off. Our code and data are available at https://github.com/Flians/MappingEvolve.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26590v1">Recommendations for Efficient and Responsible LLM Adoption within Industrial Software Development</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 Accepted for publication in the Information and Software Technology Journal
    </div>
    <details class="paper-abstract">
      Context: Large language models (LLMs) are observed to have a significant positive impact on various software engineering (SE) activities. With improved accessibility, the adoption of powerful LLMs in industry has surged recently. However, there is a lack of actionable best practices for the efficient and responsible adoption of LLMs within industrial software settings. Objectives: We developed seven actionable recommendations to address this research gap. Methods: We conducted a multi-case study with three organisations that use LLMs within their SE activities and synthesised seven recommendations through qualitative thematic analysis. We conducted a complementary online survey with software practitioners from various industries to evaluate the perceived relevance of our recommendations. Results: Our results and recommendations focus on (i) users' preference to use LLMs as AI assistants, (ii) the importance of relevant stakeholders' satisfaction in the LLM-output evaluation, (iii) scoping the applicability of LLMs within SE tasks, (iv) the effect of LLMs on SE workflows, (v) the necessity and directions for developing human oversight mechanisms, and (vi) the necessary skills for practitioners for leveraging LLMs within SE. The online survey indicates a high level of agreement from the participants regarding the perceived relevance of the recommendations. Conclusion: We outline future research directions, including mapping the seven recommendations to the principles of the EU AI Act (AIA) in order to examine how they relate to the current regulatory compliance frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26569v1">LLM-Flax : Generalizable Robotic Task Planning via Neuro-Symbolic Approaches with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Deploying a neuro-symbolic task planner on a new domain today requires significant manual effort: a domain expert must author relaxation and complementary rules, and hundreds of training problems must be solved to supervise a Graph Neural Network (GNN) object scorer. We propose LLM-Flax, a three-stage framework that eliminates all three sources of manual effort using a locally hosted LLM given only a PDDL domain file. Stage 1 automatically generates relaxation and complementary rules via structured prompting with format validation and self-correction. Stage 2 introduces LLM-guided failure recovery with a feasibility-gated budget policy that explicitly reserves API latency cost before each LLM call, preventing the downstream relaxation fallback from being starved. Stage 3 replaces the domain-trained GNN entirely with zero-shot LLM object importance scoring, requiring no training data. We evaluate all three stages on the MazeNamo benchmark across 10x10, 12x12, and 15x15 grids (8 benchmarks total). LLM-Flax achieves average SR 0.945 versus the manual baseline's 0.828 (+0.117), matching or outperforming manual rules on every one of the eight benchmarks. On 12x12 Expert, LLM-Flax attains SR 0.733 where the manual planner fails entirely (SR 0.000); on 15x15 Hard, it achieves SR 1.000 versus Manual's 0.900. Stage 3 demonstrates feasibility (SR 0.720 on 12x12 Hard with no training data) but faces a context-window bottleneck at scale, pointing to the primary open challenge for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26568v1">Multimodal LLMs are not all you need for Pediatric Speech Language Pathology</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Speech Sound Disorders (SSD) affect roughly five percent of children, yet speech-language pathologists face severe staffing shortages and unmanageable caseloads. We test a hierarchical approach to SSD classification on the granular multi-task SLPHelmUltraSuitePlus benchmark. We propose a cascading approach from binary classification to type, and symptom classification. By fine-tuning Speech Representation Models (SRM), and using targeted data augmentation we mitigate biases found by previous works, and improve upon all clinical tasks in the benchmark. We also treat Automatic Speech Recognition (ASR) with our data augmentation approach. Our results demonstrate that SRM consistently outperform the LLM-based state-of-the-art across all evaluated tasks by a large margin. We publish our models and code to foster future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26557v1">DUAL-BLADE: Dual-Path NVMe-Direct KV-Cache Offloading for Edge LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 To appear in IEEE International Conference on Distributed Computing Systems (ICDCS) 2026
    </div>
    <details class="paper-abstract">
      The increasing deployment of Large Language Model (LLM) inference on edge AI systems demands efficient execution under tight memory budgets. A key challenge arises from Key-Value (KV) caches, which often exceed available device memory. Although NVMe-based offloading offers scalable capacity, existing file-based designs rely heavily on the kernel page cache, leading to cache thrashing, unpredictable latency, and high software overhead under memory pressure. We present DUAL-BLADE, a dual-path KV residency framework that dynamically assigns KV tensors to either a page-cache path or an NVMe-direct path based on runtime memory availability. The NVMe-direct path bypasses the filesystem by mapping KV tensors to contiguous logical block address (LBA) regions, enabling low-overhead direct storage access. DUAL-BLADE further incorporates adaptive pipeline parallelism to overlap storage I/O with GPU DMA, improving inference throughput. Our evaluation shows that DUAL-BLADE substantially mitigates I/O bottlenecks, reducing prefill and decode latency by up to 33.1% and 42.4%, respectively, while improving SSD utilization by 2.2x across diverse memory budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27014v1">Fidelity, Diversity, and Privacy: A Multi-Dimensional LLM Evaluation for Clinical Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 9 pages, 1 figure, 1 table
    </div>
    <details class="paper-abstract">
      The scarcity of high-quality annotated medical data, particularly in mental health, poses a significant bottleneck for training robust machine learning models. Privacy regulations restrict data sharing, making synthetic data generation a promising alternative. The use of Large Language Models (LLMs) in a data augmentation pipeline could be leveraged as an alternative in this field. In the proposed methodology, DeepSeek-R1, OpenBioLLM-Llama3 and Qwen 3.5 are used to generate synthetic mental health evaluation reports conditioned on specific International Classification of Diseases, Tenth Revision (ICD-10) codes. Because naive text generation can lead to mode collapse or privacy breaches (memorization), a comprehensive evaluation framework is introduced. The generated diagnostic texts are assessed across three dimensions: semantic fidelity, lexical diversity, and privacy/plagiarism. The results demonstrate that all models can generate clinically coherent, diverse, and privacy-safe synthetic reports, significantly expanding the available training data for clinical natural language processing tasks without compromising patient confidentiality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27011v1">Automatic Causal Fairness Analysis with LLM-Generated Reporting</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 22 pages, 6 figures
    </div>
    <details class="paper-abstract">
      AutoML, intended as the process of automating the application of machine learning to real-world problems, is a key step for AI popularisation. Most AutoML frameworks are not accounting for the potential lack of fairness in the training data and in the corresponding predictions. We introduce \textsc{FairMind}, a software prototype aiming to automatise fairness analysis at the dataset level. We achieve that by resorting to the assumptions of the \emph{standard fairness model}, recently proposed by Plečko and Bareinboim. This allows for a sound fairness evaluation in terms of causal effects, based on \emph{counterfactual} queries involving the target, possibly confounders and mediators, and the different values of an input feature we regard as \emph{protected}. After the necessary data preprocessing, the tool implements a closed-form computation of the effects. LLMs are consequently exploited to generate accurate reports on the fairness levels detected in the training dataset. We achieve that in a zero-shot setup and show by examples the expected advantages with respect to a direct analysis performed by the LLM. To favour applications, extensions to ordinal protected variable and continuous targets and novel decomposition results are also discussed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26511v1">Tatemae: Detecting Alignment Faking via Tool Selection in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Alignment faking (AF) occurs when an LLM strategically complies with training objectives to avoid value modification, reverting to prior preferences once monitoring is lifted. Current detection methods focus on conversational settings and rely primarily on Chain-of-Thought (CoT) analysis, which provides a reliable signal when strategic reasoning surfaces, but cannot distinguish deception from capability failures if traces are absent or unfaithful. We formalize AF as a composite behavioural event and detect it through observable tool selection, where the LLM selects the safe tool when unmonitored, but switches to the unsafe tool under monitoring that rewards helpfulness over safety, while its reasoning still acknowledges the safe choice. We release a dataset of 108 enterprise IT scenarios spanning Security, Privacy, and Integrity domains under Corruption and Sabotage pressures. Evaluating six frontier LLMs across five independent runs, we find mean AF detection rates between 3.5% and 23.7%, with vulnerability profiles varying by domain and pressure type. These results suggest that susceptibility reflects training methodology rather than capability alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26506v1">SafeReview: Defending LLM-based Review Systems Against Adversarial Hidden Prompts</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 10 pages, 3 figures, 9 tables
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly integrated into academic peer review, their vulnerability to adversarial prompts -- adversarial instructions embedded in submissions to manipulate outcomes -- emerges as a critical threat to scholarly integrity. To counter this, we propose a novel adversarial framework where a Generator model, trained to create sophisticated attack prompts, is jointly optimized with a Defender model tasked with their detection. This system is trained using a loss function inspired by Information Retrieval Generative Adversarial Networks, which fosters a dynamic co-evolution between the two models, forcing the Defender to develop robust capabilities against continuously improving attack strategies. The resulting framework demonstrates significantly enhanced resilience to novel and evolving threats compared to static defenses, thereby establishing a critical foundation for securing the integrity of peer review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.24405v2">How Personal Characteristics Shape User Exploration of Diverse Movie Recommendations with a LLM-Based Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 11 pages, 2 figures. Accepted by UMAP '26 (34th ACM International Conference on User Modeling, Adaptation and Personalization)
    </div>
    <details class="paper-abstract">
      Diversity is an important evaluation criterion for recommender systems beyond accuracy, yet users differ in their willingness to engage with novel and diverse content. In this work, we investigate how a Large Language Model (LLM)-based multi-agent system supports users' exploration of diverse recommendations, and how individual characteristics shape user experiences. We conducted a between-subjects user study (N = 100) comparing a single-agent system (baseline) with a multi-agent system for movie recommendations. We measured Perceived Accuracy, diversity, novelty, and overall rating, and examined the influence of personal characteristics, including personality traits, demographics, GenAI recommendation experience, and GenAI skepticism. Results show that the multi-agent system significantly increases Perceived Novelty and Shannon Diversity. Conscientiousness is positively associated with Perceived Accuracy and diversity, whereas extraversion is negatively associated with Perceived Diversity. Prior experience with GenAI-based recommendations is positively associated with Shannon Diversity, while skepticism toward GenAI is negatively associated with it. We also observe significant interaction effects between system design and user characteristics. These findings highlight the importance of personality-aware conversational recommender systems and caution against one-size-fits-all multi-agent designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.18602v2">Machine Spirits: Speculation and Adaptation of LLM Agents in Asset Markets</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 46 pages, 6 figures
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly integrated into financial systems, understanding their behavioural properties is crucial. Do LLMs conform to the rational expectations paradigm, do they exhibit human-like "animal spirits", or do they instead manifest distinct "machine spirits"? We investigate these questions with a simulated financial market, exploring the behaviour of 15 LLMs spanning a range of sizes, capabilities, and providers. Our results show that LLMs exhibit a spectrum of economic behaviours, from stable coordination on the fundamental value to human-like speculative bubbles. These behaviours are generally inconsistent with the rational expectations hypothesis. We also consider an ecology of heterogeneous agents, a more realistic setting compared to markets with identical LLM agents. These mixed markets can produce outcomes which vary substantially across repeated simulations. Even the most advanced models fail to consistently stabilise the market, with price bubbles sometimes forming despite only a minority of agents naturally forming bubbles. Instead, advanced models in mixed markets adapt their forecasting strategies to the behaviour of other agents. This adaptation can allow them to successfully exploit less sophisticated counterparts and achieve higher profits, but can also contribute to increased market volatility. These findings suggest that the introduction of AI agents into financial markets fundamentally reshapes their ecology. In particular, heterogeneous populations of LLMs can generate endogenous instability, while individual-level adaptation may amplify, rather than mitigate, market volatility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26460v1">Theory-Grounded Evaluation Exposes the Authorship Gap in LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 6 pages, 2 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Stylistic personalization - making LLMs write in a specific individual's style, rather than merely adapting to task preferences - lacks evaluation grounded in authorship science. We show that grounding evaluation in authorship verification theory transforms what benchmarks can measure. Drawing on three measurement traditions - LUAR, a trained authorship verification model; an LLM-as-judge with decoupled trait matching; and classical function-word stylometrics - we evaluate four inference-time personalization methods across 50 authors and 1,000 generations. The theory-grounded metric, LUAR, provides what ad hoc alternatives cannot: calibrated baselines, with a human ceiling of 0.756 and a cross-author floor of 0.626, that give scores absolute meaning. All methods score below this floor, from 0.484 to 0.508, exposing an authorship gap invisible to uncalibrated metrics. The three metrics produce near-zero pairwise correlations, with absolute r less than 0.07, confirming that without theoretical grounding, metric choice determines conclusions: an LLM judge declares a clear winner while LUAR finds no meaningful differentiation. These findings demonstrate the theory-benchmark cycle in action: authorship theory exposes evaluation failures that ad hoc benchmarks miss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.26456v1">Naamah: A Large Scale Synthetic Sanskrit NER Corpus via DBpedia Seeding and LLM Generation</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      The digitisation of classical Sanskrit literature is impeded by a scarcity of annotated resources, particularly for Named Entity Recognition. While recent methodologies utilise generic Large Language Models (LLMs) for data augmentation, these approaches remain prone to error and often lack the reasoning depth required for classical grammar. In this work, we introduce Naamah, a high quality silver standard Sanskrit NER dataset comprising 102,942 sentences. We propose a methodology that combines entity extraction from DBpedia with the generative capabilities of a 24B parameter hybrid reasoning model to create grammatically natural and synthetically diverse training data. We utilize this dataset to benchmark two transformer architectures: the massive multilingual XLM RoBERTa and the parameter efficient IndicBERTv2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2604.27006v1">Beyond Accuracy: LLM Variability in Evidence Screening for Software Engineering SLRs</a></div>
    <div class="paper-meta">
      📅 2026-04-29
      | 💬 16 pages, 12 figures. Earlier, shorter, conference-style version of a more comprehensive journal manuscript currently under review
    </div>
    <details class="paper-abstract">
      Context: Study screening in systematic literature reviews is costly, inconsistency-prone, and risk-asymmetric, since false negatives can compromise validity. Despite rapid uptake of Large Language Models (LLMs), there is limited evidence on how such models behave during the study screening phase, particularly regarding the choice of specific LLMs and their comparison with classical models. Objective: To assess LLM performance and variability in screening, quantify the impact of input metadata (abstract, title, keywords), and compare LLMs with classical classifiers under a shared protocol. Methods: We analyzed 12 LLMs from 4 providers (OpenAI, Google Gemini, Anthropic, Llama) and 4 classical models (Logistic Regression, Support Vector Classification, Random Forest, and Naive Bayes) on 2 real Systematic Literature Reviews (SLRs), totaling 518 papers. The experimental design investigated 3 critical dimensions: (i) LLMs performance variability, (ii) the impact of input feature composition (abstract, title, and keywords) on LLM performance, and (iii) the real gain of using LLMs instead of more traditional classification models. Results: LLMs exhibited substantial heterogeneity and residual non-determinism even at temperature zero. Abstract availability was decisive: removing it consistently degraded performance, while adding title and/or keywords to the abstract yielded no robust gains. Compared to classical models, performance differences were not consistent enough to support generalizable LLM superiority. Discussion: LLM adoption should be justified by operational and governance constraints (reproducibility, cost, metadata availability), supported by pilot validation and explicit reporting of variability and input configuration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23410v4">PATCH: Learnable Tile-level Hybrid Sparsity for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-04-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deliver impressive performance but incur prohibitive memory and compute costs at deployment. Model pruning is an effective way to reduce these overheads, yet existing approaches face challenges: unstructured sparsity, where nonzeros can appear anywhere, preserves accuracy but yields irregular access patterns that prevent GPU acceleration, while semi-structured 2:4 sparsity is hardware-friendly but enforces a rigid 50% pattern that degrades model quality. To bridge this gap, we introduce PATCH, a hybrid sparsity framework that enables a continuous sparsity ratio between 0% and 50%. PATCH partitions weight matrices into tiles, assigning each tile to be either dense or 2:4 sparse via a learnable mask selection mechanism. This design provides fine-grained control over accuracy-acceleration tradeoffs and supports non-uniform sparsity across layers, leading to superior overall quality. Across models from 0.5B to 13B parameters, PATCH consistently narrows the gap to dense accuracy while delivering practical speedups. For instance, on LLaMA-2 7B with an A6000 GPU, PATCH achieves 1.18x-1.38x end-to-end speedup over dense baselines while improving accuracy by 0.37%-2.96% compared to the state-of-the-art 2:4 pruning method, MaskLLM.
    </details>
</div>
