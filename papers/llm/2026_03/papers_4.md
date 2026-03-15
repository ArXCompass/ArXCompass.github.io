# llm - 2026_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03111v1">Evaluating Performance Drift from Model Switching in Multi-Turn LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Deployed multi-turn LLM systems routinely switch models mid-interaction due to upgrades, cross-provider routing, and fallbacks. Such handoffs create a context mismatch: the model generating later turns must condition on a dialogue prefix authored by a different model, potentially inducing silent performance drift. We introduce a switch-matrix benchmark that measures this effect by running a prefix model for early turns and a suffix model for the final turn, and comparing against the no-switch baseline using paired episode-level bootstrap confidence intervals. Across CoQA conversational QA and Multi-IF benchmarks, even a single-turn handoff yields prevalent and statistically significant, directional effects and may swing outcomes by -8 to +13 percentage points in Multi-IF strict success rate and +/- 4 absolute F1 on CoQA, comparable to the no-switch gap between common model tiers (e.g., GPT-5-nano vs GPT-5-mini). We further find systematic compatibility patterns: some suffix models degrade under nearly any non-self dialogue history, while others improve under nearly any foreign prefix. To enable compressed handoff risk monitoring, we decompose switch-induced drift into per-model prefix influence and suffix susceptibility terms, accounting for ~70% of variance across benchmarks. These results position handoff robustness as an operational reliability dimension that single-model benchmarks miss, motivating explicit monitoring and handoff-aware mitigation in multi-turn systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03095v1">Compact Prompting in Instruction-tuned LLMs for Joint Argumentative Component Detection</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Under Review (COLM 2026)
    </div>
    <details class="paper-abstract">
      Argumentative component detection (ACD) is a core subtask of Argument(ation) Mining (AM) and one of its most challenging aspects, as it requires jointly delimiting argumentative spans and classifying them into components such as claims and premises. While research on this subtask remains relatively limited compared to other AM tasks, most existing approaches formulate it as a simplified sequence labeling problem, component classification, or a pipeline of component segmentation followed by classification. In this paper, we propose a novel approach based on instruction-tuned Large Language Models (LLMs) using compact instruction-based prompts, and reframe ACD as a language generation task, enabling arguments to be identified directly from plain text without relying on pre-segmented components. Experiments on standard benchmarks show that our approach achieves higher performance compared to state-of-the-art systems. To the best of our knowledge, this is one of the first attempts to fully model ACD as a generative task, highlighting the potential of instruction tuning for complex AM problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03078v1">RAPO: Expanding Exploration for LLM Agents via Retrieval-Augmented Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Submit to KDD 2026
    </div>
    <details class="paper-abstract">
      Agentic Reinforcement Learning (Agentic RL) has shown remarkable potential in large language model-based (LLM) agents. These works can empower LLM agents to tackle complex tasks via multi-step, tool-integrated reasoning. However, an inherent limitation of existing Agentic RL methods is their reliance on a pure on-policy paradigm for exploration, restricting exploration to the agent's self-generated outputs and preventing the discovery of new reasoning perspectives for further improvement. While recent efforts incorporate auxiliary off-policy signals to enhance exploration, they typically utilize full off-policy trajectories for trajectory-level policy estimation, overlooking the necessity for the fine-grained, step-level exploratory dynamics within agentic rollout. In this paper, we revisit exploration in Agentic RL and propose Retrieval-Augmented Policy Optimization (RAPO), a novel RL framework that introduces retrieval to explicitly expand exploration during training. To achieve this, we decompose the Agentic RL training process into two phases: (i) Hybrid-policy Agentic Rollout, and (ii) Retrieval-aware Policy Optimization. Specifically, we propose a Hybrid-policy Agentic Rollout strategy, which allows the agents to continuously reason over the retrieved off-policy step-level traces. It dynamically extends the reasoning receptive field of agents, enabling broader exploration conditioned on external behaviors. Subsequently, we introduce the Retrieval-aware Policy Optimization mechanism, which calibrates the policy gradient estimation with retrieval reward and importance shaping, stabilizing training and prioritizing retrieval-illuminating exploration. Extensive experiments show that RAPO achieves an +5.0% average gain on fourteen datasets across three agentic reasoning tasks, while delivering 1.2x faster training efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02983v1">Contextualized Privacy Defense for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      LLM agents increasingly act on users' personal information, yet existing privacy defenses remain limited in both design and adaptability. Most prior approaches rely on static or passive defenses, such as prompting and guarding. These paradigms are insufficient for supporting contextual, proactive privacy decisions in multi-step agent execution. We propose Contextualized Defense Instructing (CDI), a new privacy defense paradigm in which an instructor model generates step-specific, context-aware privacy guidance during execution, proactively shaping actions rather than merely constraining or vetoing them. Crucially, CDI is paired with an experience-driven optimization framework that trains the instructor via reinforcement learning (RL), where we convert failure trajectories with privacy violations into learning environments. We formalize baseline defenses and CDI as distinct intervention points in a canonical agent loop, and compare their privacy-helpfulness trade-offs within a unified simulation framework. Results show that our CDI consistently achieves a better balance between privacy preservation (94.2%) and helpfulness (80.6%) than baselines, with superior robustness to adversarial conditions and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02949v1">SEALing the Gap: A Reference Framework for LLM Inference Carbon Estimation via Multi-Benchmark Driven Embodiment</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 5 pages. To be published in the proceedings of 48th International Conference on Software Engineering (ICSE '26), April 12-18, 2026, Rio de Janeiro, Brazil (New Ideas and Emerging Results Track)
    </div>
    <details class="paper-abstract">
      Large Language Models are rapidly gaining traction in software engineering, yet their growing carbon footprint raises pressing sustainability concerns. While training emissions are substantial, inference quickly surpasses them due to the sheer volume of prompts processed. This shift underscores the urgent need for accurate, prompt-level carbon measurement during inference to enable informed, sustainability-focused decision-making. To address the limitations of existing approaches, in this paper, we outline the guiding principles for a novel reference framework for LLM inference carbon estimation that can guide the design of future tools and provide a systematic foundation for advancing sustainability research in this domain. We also introduce SEAL, an early embodiment of these principles that leverages a multi-benchmark-driven approach for per-prompt carbon estimation. Its initial validation shows promising results, positioning SEAL as a foundation for standardized sustainability assessment across the LLM ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.19892v3">OptMerge: Unifying Multimodal LLM Capabilities and Modalities via Model Merging</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Foundation models update slowly due to resource-intensive training, whereas domain-specific models evolve rapidly between releases. Model merging seeks to combine multiple expert models into a single, more capable model, reducing storage and serving costs while supporting decentralized development. Despite its potential, previous studies have primarily focused on merging visual classification models or Large Language Models (LLMs) for code and math tasks. Recently, Multimodal LLMs (MLLMs) that extend LLMs through large-scale multimodal training have gained traction. However, there lacks a benchmark for model merging research that clearly divides the tasks for MLLM training and evaluation. In this paper, $\textbf{(i)}$ we introduce a model merging benchmark for MLLMs, which includes multiple tasks such as VQA, Geometry, Chart, OCR, and Grounding, studying both LoRA and full fine-tuning models. Moreover, we explore how model merging can combine different modalities (e.g., vision-language, audio-language, and video-language models), moving toward the Omni-language model. $\textbf{(ii)}$ We implement 10 model merging algorithms on the benchmark. Furthermore, we propose a novel method that removes noise from task vectors and robustly optimizes the merged vector based on a loss defined over task vector interactions, achieving an average performance gain of 2.48%. $\textbf{(iii)}$ We find that model merging offers a promising way for building improved MLLMs without requiring training data. Our results also demonstrate that the complementarity among multiple modalities outperforms individual modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.05282v5">Not All Errors Are Created Equal: ASCoT Addresses Late-Stage Fragility in Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      While Chain-of-Thought (CoT) prompting empowers Large Language Models (LLMs), ensuring reasoning reliability remains an open challenge. Contrary to the prevailing cascading failure hypothesis which posits that early errors are most detrimental, we identify a counter-intuitive phenomenon termed \textbf{Late-Stage Fragility}: errors introduced in later reasoning stages are significantly more prone to corrupting final answers. To address this, we introduce ASCoT (Adaptive Self-Correction Chain-of-Thought), a method harmonizing efficiency with robust verification. ASCoT first employs semantic pruning to compress redundant steps, then utilizes an Adaptive Verification Manager (AVM) to prioritize high risk, late-stage steps via a positional impact score, triggering a Multi-Perspective Self-Correction Engine (MSCE) only when necessary. Experiments on GSM8K and MATH-500 demonstrate that ASCoT effectively reallocates computational resources: it reduces token usage by 21\%--30\% for LLaMA-3.1-8B with negligible accuracy drops ($<1.8\%$), achieving a superior trade-off between inference efficiency and reasoning fidelity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02913v1">Eliciting Numerical Predictive Distributions of LLMs Without Autoregression</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 First two authors contributed equally. Published as a conference paper at ICLR2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been successfully applied to regression tasks -- such as time series forecasting and tabular prediction -- by leveraging their in-context learning abilities. However, their autoregressive decoding process may be ill-suited to continuous-valued outputs, where obtaining predictive distributions over numerical targets requires repeated sampling, leading to high computational cost and inference time. In this work, we investigate whether distributional properties of LLM predictions can be recovered without explicit autoregressive generation. To this end, we study a set of regression probes trained to predict statistical functionals (e.g., mean, median, quantiles) of the LLM's numerical output distribution directly from its internal representations. Our results suggest that LLM embeddings carry informative signals about summary statistics of their predictive distributions, including the numerical uncertainty. This investigation opens up new questions about how LLMs internally encode uncertainty in numerical tasks, and about the feasibility of lightweight alternatives to sampling-based approaches for uncertainty-aware numerical predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02908v1">SAE as a Crystal Ball: Interpretable Features Predict Cross-domain Transferability of LLMs without Training</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      In recent years, pre-trained large language models have achieved remarkable success across diverse tasks. Besides the pivotal role of self-supervised pre-training, their effectiveness in downstream applications also depends critically on the post-training process, which adapts models to task-specific data and objectives. However, this process inevitably introduces model shifts that can influence performance in different domains, and how such shifts transfer remains poorly understood. To open up the black box, we propose the SAE-based Transferability Score (STS), a new metric that leverages sparse autoencoders (SAEs) to forecast post-training transferability. Taking supervised fine-tuning as an example, STS identifies shifted dimensions in SAE representations and calculates their correlations with downstream domains, enabling reliable estimation of transferability \textit{before} fine-tuning. Extensive experiments across multiple models and domains show that STS accurately predicts the transferability of supervised fine-tuning, achieving Pearson correlation coefficients above 0.7 with actual performance changes. Beyond this, we take an initial step toward extending STS to reinforcement learning. We believe that STS can serve as an {\color{black} interpretable} tool for guiding post-training strategies in LLMs. Code is available at https://github.com/PKU-ML/STS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02885v1">MuxTune: Efficient Multi-Task LLM Fine-Tuning in Multi-Tenant Datacenters via Spatial-Temporal Backbone Multiplexing</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Parameter-Efficient Fine-Tuning (PEFT) is widely applied as the backend of fine-tuning APIs for large language model (LLM) customization in datacenters. Service providers deploy separate instances for individual PEFT tasks, giving rise to prominent resource inefficiencies, including (1) GPU underutilization from small-scale, PEFT-native operators and (2) device stalls from communication delays and data dependencies in parallelized execution. To address these issues, this paper presents MuxTune, a fine-tuning system that enables resource-efficient concurrent execution of multiple PEFT tasks. The key idea is to multiplex the backbone across independent tasks in a spatial-temporal manner for improved utilization and reduced stalls. Building on flexible, modularized backbone sharing via unified PEFT representations, MuxTune proposes hierarchical co-scheduling scheme with task, operator, and data-level optimizations. Specifically, it fuses tasks through a hybrid of spatial and temporal multiplexing, and orchestrates multi-task operator execution in two-tiered hybrid parallelism. Additionally, MuxTune employs chunk-based data alignment to mitigate inter-task ineffective tokens. Experimental results demonstrate that MuxTune achieves up to $2.33\times$ higher throughput and $5.29\times$ memory reduction compared to three state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02858v1">LLM-based Argument Mining meets Argumentation and Description Logics: a Unified Framework for Reasoning about Debates</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve strong performance in analyzing and generating text, yet they struggle with explicit, transparent, and verifiable reasoning over complex texts such as those containing debates. In particular, they lack structured representations that capture how arguments support or attack each other and how their relative strengths determine overall acceptability. We encompass these limitations by proposing a framework that integrates learning-based argument mining with quantitative reasoning and ontology-based querying. Starting from a raw debate text, the framework extracts a fuzzy argumentative knowledge base, where arguments are explicitly represented as entities, linked by attack and support relations, and annotated with initial fuzzy strengths reflecting plausibility w.r.t. the debate's context. Quantitative argumentation semantics are then applied to compute final argument strengths by propagating the effects of supports and attacks. These results are then embedded into a fuzzy description logic setting, enabling expressive query answering through efficient rewriting techniques. The proposed approach provides a transparent, explainable, and formally grounded method for analyzing debates, overcoming purely statistical LLM-based analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.00857v2">ManagerBench: Evaluating the Safety-Pragmatism Trade-off in Autonomous LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve from conversational assistants into autonomous agents, evaluating the safety of their actions becomes critical. Prior safety benchmarks have primarily focused on preventing generation of harmful content, such as toxic text. However, they overlook the challenge of agents taking harmful actions when the most effective path to an operational goal conflicts with human safety. To address this gap, we introduce ManagerBench, a benchmark that evaluates LLM decision-making in realistic, human-validated managerial scenarios. Each scenario forces a choice between a pragmatic but harmful action that achieves an operational goal, and a safe action that leads to worse operational performance. A parallel control set, where potential harm is directed only at inanimate objects, measures a model's pragmatism and identifies its tendency to be overly safe. Our findings indicate that the frontier LLMs perform poorly when navigating this safety-pragmatism trade-off. Many consistently choose harmful options to advance their operational goals, while others avoid harm only to become overly safe and ineffective. Critically, we find this misalignment does not stem from an inability to perceive harm, as models' harm assessments align with human judgments, but from flawed prioritization. ManagerBench is a challenging benchmark for a core component of agentic behavior: making safe choices when operational goals and alignment values incentivize conflicting actions. Benchmark & code available at https://technion-cs-nlp.github.io/ManagerBench-website/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02830v1">Faster, Cheaper, More Accurate: Specialised Knowledge Tracing Models Outperform LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 7 pages, 6 figures. Prarthana Bhattacharyya and Joshua Mitton contributed equally to this work
    </div>
    <details class="paper-abstract">
      Predicting future student responses to questions is particularly valuable for educational learning platforms where it enables effective interventions. One of the key approaches to do this has been through the use of knowledge tracing (KT) models. These are small, domain-specific, temporal models trained on student question-response data. KT models are optimised for high accuracy on specific educational domains and have fast inference and scalable deployments. The rise of Large Language Models (LLMs) motivates us to ask the following questions: (1) How well can LLMs perform at predicting students' future responses to questions? (2) Are LLMs scalable for this domain? (3) How do LLMs compare to KT models on this domain-specific task? In this paper, we compare multiple LLMs and KT models across predictive performance, deployment cost, and inference speed to answer the above questions. We show that KT models outperform LLMs with respect to accuracy and F1 scores on this domain-specific task. Further, we demonstrate that LLMs are orders of magnitude slower than KT models and cost orders of magnitude more to deploy. This highlights the importance of domain-specific models for education prediction tasks and the fact that current closed source LLMs should not be used as a universal solution for all tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10625v3">No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Accepted (poster) to Principled Design for Trustworthy AI at ICLR 2026
    </div>
    <details class="paper-abstract">
      Do large language models (LLMs) anticipate when they will answer correctly? To study this, we extract activations after a question is read but before any tokens are generated, and train linear probes to predict whether the model's forthcoming answer will be correct. Across three open-source model families ranging from 7 to 70 billion parameters, projections on this "in-advance correctness direction" trained on generic trivia questions predict success in distribution and on diverse out-of-distribution knowledge datasets, indicating a deeper signal than dataset-specific spurious features, and outperforming black-box baselines and verbalised predicted confidence. Predictive power saturates in intermediate layers and, notably, generalisation falters on questions requiring mathematical reasoning. Moreover, for models responding "I don't know", doing so strongly correlates with the probe score, indicating that the same direction also captures confidence. By complementing previous results on truthfulness and other behaviours obtained with probes and sparse auto-encoders, our work contributes essential findings to elucidate LLM internals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15569v2">What Are You Doing? Effects of Intermediate Feedback from Agentic LLM In-Car Assistants During Multi-Step Processing</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Accepted at CHI 2026
    </div>
    <details class="paper-abstract">
      Agentic AI assistants that autonomously perform multi-step tasks raise open questions for user experience: how should such systems communicate progress and reasoning during extended operations, especially in attention-critical contexts such as driving? We investigate feedback timing and verbosity from agentic LLM-based in-car assistants through a controlled, mixed-methods study (N=45) comparing planned steps and intermediate results feedback against silent operation with final-only response. Using a dual-task paradigm with an in-car voice assistant, we found that intermediate feedback significantly improved perceived speed, trust, and user experience while reducing task load - effects that held across varying task complexities and interaction contexts. Interviews further revealed user preferences for an adaptive approach: high initial transparency to establish trust, followed by progressively reducing verbosity as systems prove reliable, with adjustments based on task stakes and situational context. We translate our empirical findings into design implications for feedback timing and verbosity in agentic assistants, balancing transparency and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02792v1">From Heuristic Selection to Automated Algorithm Design: LLMs Benefit from Strong Priors</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have already been widely adopted for automated algorithm design, demonstrating strong abilities in generating and evolving algorithms across various fields. Existing work has largely focused on examining their effectiveness in solving specific problems, with search strategies primarily guided by adaptive prompt designs. In this paper, through investigating the token-wise attribution of the prompts to LLM-generated algorithmic codes, we show that providing high-quality algorithmic code examples can substantially improve the performance of the LLM-driven optimization. Building upon this insight, we propose leveraging prior benchmark algorithms to guide LLM-driven optimization and demonstrate superior performance on two black-box optimization benchmarks: the pseudo-Boolean optimization suite (pbo) and the black-box optimization suite (bbob). Our findings highlight the value of integrating benchmarking studies to enhance both efficiency and robustness of the LLM-driven black-box optimization methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04573v5">LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design allows efficient parallel generation of diverse reasoning trajectories, allowing the model to plan and revise the reasoning process holistically. We conduct evaluations on a suite of mathematical reasoning and planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02787v1">Rethinking Code Similarity for Automated Algorithm Design with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Accepted to ICLR 2026
    </div>
    <details class="paper-abstract">
      The rise of Large Language Model-based Automated Algorithm Design (LLM-AAD) has transformed algorithm development by autonomously generating code implementations of expert-level algorithms. Unlike traditional expert-driven algorithm development, in the LLM-AAD paradigm, the main design principle behind an algorithm is often implicitly embedded in the generated code. Therefore, assessing algorithmic similarity directly from code, distinguishing genuine algorithmic innovation from mere syntactic variation, becomes essential. While various code similarity metrics exist, they fail to capture algorithmic similarity, as they focus on surface-level syntax or output equivalence rather than the underlying algorithmic logic. We propose BehaveSim, a novel method to measure algorithmic similarity through the lens of problem-solving behavior as a sequence of intermediate solutions produced during execution, dubbed as problem-solving trajectories (PSTrajs). By quantifying the alignment between PSTrajs using dynamic time warping (DTW), BehaveSim distinguishes algorithms with divergent logic despite syntactic or output-level similarities. We demonstrate its utility in two key applications: (i) Enhancing LLM-AAD: Integrating BehaveSim into existing LLM-AAD frameworks (e.g., FunSearch, EoH) promotes behavioral diversity, significantly improving performance on three AAD tasks. (ii) Algorithm analysis: BehaveSim clusters generated algorithms by behavior, enabling systematic analysis of problem-solving strategies--a crucial tool for the growing ecosystem of AI-generated algorithms. Data and code of this work are open-sourced at https://github.com/RayZhhh/behavesim.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02775v1">From Solver to Tutor: Evaluating the Pedagogical Intelligence of LLMs with KMP-Bench</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show significant potential in AI mathematical tutoring, yet current evaluations often rely on simplistic metrics or narrow pedagogical scenarios, failing to assess comprehensive, multi-turn teaching effectiveness. In this paper, we introduce KMP-Bench, a comprehensive K-8 Mathematical Pedagogical Benchmark designed to assess LLMs from two complementary perspectives. The first module, KMP-Dialogue, evaluates holistic pedagogical capabilities against six core principles (e.g., Challenge, Explanation, Feedback), leveraging a novel multi-turn dialogue dataset constructed by weaving together diverse pedagogical components. The second module, KMP-Skills, provides a granular assessment of foundational tutoring abilities, including multi-turn problem-solving, error detection and correction, and problem generation. Our evaluations on KMP-Bench reveal a key disparity: while leading LLMs excel at tasks with verifiable solutions, they struggle with the nuanced application of pedagogical principles. Additionally, we present KMP-Pile, a large-scale (150K) dialogue dataset. Models fine-tuned on KMP-Pile show substantial improvement on KMP-Bench, underscoring the value of pedagogically-rich training data for developing more effective AI math tutors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04459v1">Benchmark of Benchmarks: Unpacking Influence and Code Repository Quality in LLM Safety Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 22 pages. 19 figures
    </div>
    <details class="paper-abstract">
      The rapid growth of research in LLM safety makes it hard to track all advances. Benchmarks are therefore crucial for capturing key trends and enabling systematic comparisons. Yet, it remains unclear why certain benchmarks gain prominence, and no systematic assessment has been conducted on their academic influence or code quality. This paper fills this gap by presenting the first multi-dimensional evaluation of the influence (based on five metrics) and code quality (based on both automated and human assessment) on LLM safety benchmarks, analyzing 31 benchmarks and 382 non-benchmarks across prompt injection, jailbreak, and hallucination. We find that benchmark papers show no significant advantage in academic influence (e.g., citation count and density) over non-benchmark papers. We uncover a key misalignment: while author prominence correlates with paper influence, neither author prominence nor paper influence shows a significant correlation with code quality. Our results also indicate substantial room for improvement in code and supplementary materials: only 39% of repositories are ready-to-use, 16% include flawless installation guides, and a mere 6% address ethical considerations. Given that the work of prominent researchers tends to attract greater attention, they need to lead the effort in setting higher standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02099v2">Recursive Think-Answer Process for LLMs and VLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 CVPR 2026 Findings, Project page: https://litcoderr.github.io/rtap_page/
    </div>
    <details class="paper-abstract">
      Think-Answer reasoners such as DeepSeek-R1 have made notable progress by leveraging interpretable internal reasoning. However, despite the frequent presence of self-reflective cues like "Oops!", they remain vulnerable to output errors during single-pass inference. To address this limitation, we propose an efficient Recursive Think-Answer Process (R-TAP) that enables models to engage in iterative reasoning cycles and generate more accurate answers, going beyond conventional single-pass approaches. Central to this approach is a confidence generator that evaluates the certainty of model responses and guides subsequent improvements. By incorporating two complementary rewards-Recursively Confidence Increase Reward and Final Answer Confidence Reward-we show that R-TAP-enhanced models consistently outperform conventional single-pass methods for both large language models (LLMs) and vision-language models (VLMs). Moreover, by analyzing the frequency of "Oops"-like expressions in model responses, we find that R-TAP-applied models exhibit significantly fewer self-reflective patterns, resulting in more stable and faster inference-time reasoning. We hope R-TAP pave the way evolving into efficient and elaborated methods to refine the reasoning processes of future AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.23636v2">FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Ensuring the safety of LLM-generated content is essential for real-world deployment. Most existing guardrail models formulate moderation as a fixed binary classification task, implicitly assuming a fixed definition of harmfulness. In practice, enforcement strictness - how conservatively harmfulness is defined and enforced - varies across platforms and evolves over time, making binary moderators brittle under shifting requirements. We first introduce FlexBench, a strictness-adaptive LLM moderation benchmark that enables controlled evaluation under multiple strictness regimes. Experiments on FlexBench reveal substantial cross-strictness inconsistency in existing moderators: models that perform well under one regime can degrade substantially under others, limiting their practical usability. To address this, we propose FlexGuard, an LLM-based moderator that outputs a calibrated continuous risk score reflecting risk severity and supports strictness-specific decisions via thresholding. We train FlexGuard via risk-alignment optimization to improve score-severity consistency and provide practical threshold selection strategies to adapt to target strictness at deployment. Experiments on FlexBench and public benchmarks demonstrate that FlexGuard achieves higher moderation accuracy and substantially improved robustness under varying strictness. We release the source code and data to support reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.03170v3">AttackSeqBench: Benchmarking the Capabilities of LLMs for Attack Sequences Understanding</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 27 pages, 9 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Cyber Threat Intelligence (CTI) reports document observations of cyber threats, synthesizing evidence about adversaries' actions and intent into actionable knowledge that informs detection, response, and defense planning. However, the unstructured and verbose nature of CTI reports poses significant challenges for security practitioners to manually extract and analyze such sequences. Although large language models (LLMs) exhibit promise in cybersecurity tasks such as entity extraction and knowledge graph construction, their understanding and reasoning capabilities towards behavioral sequences remains underexplored. To address this, we introduce AttackSeqBench, a benchmark designed to systematically evaluate LLMs' reasoning abilities across the tactical, technical, and procedural dimensions of adversarial behaviors, while satisfying Extensibility, Reasoning Scalability, and Domain-dpecific Epistemic Expandability. We further benchmark 7 LLMs, 5 LRMs and 4 post-training strategies across 3 benchmark settings and 3 benchmark tasks within our AttackSeqBench to identify their advantages and limitations in such specific domain. Our findings contribute to a deeper understanding of LLM-driven CTI report understanding and foster its application in cybersecurity operations. Our code of benchmark construction and evaluation and the corresponding dataset are available at: https://github.com/hulkima/AttackSeqBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02680v1">LLMs for High-Frequency Decision-Making: Normalized Action Reward-Guided Consistency Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) form the cornerstone of sequential decision-making agent development, they have inherent limitations in high-frequency decision tasks. Existing research mainly focuses on discrete embodied decision scenarios with low-frequency and significant semantic differences in state space (e.g., household planning). These methods suffer from limited performance in high-frequency decision-making tasks, since high-precision numerical state information in such tasks undergoes frequent updates with minimal fluctuations, and exhibiting policy misalignment between the learned sub-tasks and composite tasks. To address these issues, this paper proposes Normalized Action Reward guided Consistency Policy Optimization (NAR-CP). 1) Our method first acquires predefined dense rewards from environmental feedback of candidate actions via reward functions, then completes reward shaping through normalization, and theoretically verifies action reward normalization does not impair optimal policy. 2) To reduce policy misalignment in composite tasks, we use LLMs to infer sub-observation candidate actions and generate joint policies, with consistency loss ensuring precise alignment between global semantic policies and sub-semantic policies. Experiments on UAV pursuit, a typical high-frequency task, show our method delivers superior performance on independent and composite tasks with excellent generalization to unseen tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02669v1">IMR-LLM: Industrial Multi-Robot Task Planning and Program Generation using Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      In modern industrial production, multiple robots often collaborate to complete complex manufacturing tasks. Large language models (LLMs), with their strong reasoning capabilities, have shown potential in coordinating robots for simple household and manipulation tasks. However, in industrial scenarios, stricter sequential constraints and more complex dependencies within tasks present new challenges for LLMs. To address this, we propose IMR-LLM, a novel LLM-driven Industrial Multi-Robot task planning and program generation framework. Specifically, we utilize LLMs to assist in constructing disjunctive graphs and employ deterministic solving methods to obtain a feasible and efficient high-level task plan. Based on this, we use a process tree to guide LLMs to generate executable low-level programs. Additionally, we create IMR-Bench, a challenging benchmark that encompasses multi-robot industrial tasks across three levels of complexity. Experimental results indicate that our method significantly surpasses existing methods across all evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.12610v2">ScaleDoc: Scaling LLM-based Predicates over Large Document Collections</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Predicates are foundational components in data analysis systems. However, modern workloads increasingly involve unstructured documents, which demands semantic understanding, beyond traditional value-based predicates. Given enormous documents and ad-hoc queries, while Large Language Models (LLMs) demonstrate powerful zero-shot capabilities, their high inference cost leads to unacceptable overhead. Therefore, we introduce \textsc{ScaleDoc}, a novel system that addresses this by decoupling predicate execution into an offline representation phase and an optimized online filtering phase. In the offline phase, \textsc{ScaleDoc} leverages a LLM to generate semantic representations for each document. Online, for each query, it trains a lightweight proxy model on these representations to filter the majority of documents, forwarding only the ambiguous cases to the LLM for final decision. Furthermore, \textsc{ScaleDoc} proposes two core innovations to achieve significant efficiency: (1) a contrastive-learning-based framework that trains the proxy model to generate reliable predicating decision scores; (2) an adaptive cascade mechanism that determines the effective filtering policy while meeting specific accuracy targets. Our evaluations across three datasets demonstrate that \textsc{ScaleDoc} achieves over a 2$\times$ end-to-end speedup and reduces expensive LLM invocations by up to 85\%, making large-scale semantic analysis practical and efficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06227v2">Automated Data Enrichment using Confidence-Aware Fine-Grained Debate among Open-Source LLMs for Mental Health and Online Safety</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Real-world indicators play an important role in many natural language processing (NLP) applications, such as life-event for mental health analysis and risky behaviour for online safety, yet labelling such information in training datasets is often costly and/or difficult due to their dynamic nature. Large language models (LLMs) show promising potential for automated annotation, yet multi-label prediction remains challenging. In this work, we propose a Confidence-Aware Fine-Grained Debate (CFD) framework that simulates collaborative annotation using fine-grained information to better support automated multi-label enrichment. We introduce two new expert-annotated resources: A mental health Reddit well-being dataset and an online safety Facebook sharenting risk dataset. Experiments show that CFD achieves the most robust enrichment performance compared to a range of baseline approaches. We further evaluate various training-free enrichment incorporation strategies and demonstrate that LLM-enriched indicators consistently improves our downstream tasks. Enriched features incorporated via debate transcripts yield the largest gains, outperforming the non-enriched baseline by 9.9\% on the online safety task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02599v1">SUN: Shared Use of Next-token Prediction for Efficient Multi-LLM Disaggregated Serving</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Preprint, 15 pages, 5 figures
    </div>
    <details class="paper-abstract">
      In multi-model LLM serving, decode execution remains inefficient due to model-specific resource partitioning: since cross-model batching is not possible, memory-bound decoding often suffers from severe GPU underutilization, especially under skewed workloads. We propose Shared Use of Next-token Prediction (SUN), the first approach that enables cross-model sharing of decode execution in disaggregated multi-LLM serving. SUN decomposes a decoder-only Transformer into a prefill module and a decode module, and fine-tunes only the task-specific prefill module, enabling a frozen decode module to be shared across models. This design enables a model-agnostic decode routing policy that balances decode requests across shared workers to maximize utilization. Across diverse tasks and model families, SUN achieves accuracy comparable to full fine-tuning while maintaining system throughput with fewer decode workers. In particular, SUN improves throughput per GPU by up to 2.0x over conventional disaggregation while keeping time-per-output-token (TPOT) within 5%. SUN inherently enables and facilitates low-bit decoding; with Quantized SUN (QSUN), it achieves a 45% speedup with comparable accuracy to SUN while preserving the benefits of shared decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02097v2">ClinConsensus: A Consensus-Based Benchmark for Evaluating Chinese Medical LLMs across Difficulty Levels</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 8 pages, 6 figures,
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied to health management, showing promise across disease prevention, clinical decision-making, and long-term care. However, existing medical benchmarks remain largely static and task-isolated, failing to capture the openness, longitudinal structure, and safety-critical complexity of real-world clinical workflows. We introduce ClinConsensus, a Chinese medical benchmark curated, validated, and quality-controlled by clinical experts. ClinConsensus comprises 2500 open-ended cases spanning the full continuum of care--from prevention and intervention to long-term follow-up--covering 36 medical specialties, 12 common clinical task types, and progressively increasing levels of complexity. To enable reliable evaluation of such complex scenarios, we adopt a rubric-based grading protocol and propose the Clinically Applicable Consistency Score (CACS@k). We further introduce a dual-judge evaluation framework, combining a high-capability LLM-as-judge with a distilled, locally deployable judge model trained via supervised fine-tuning, enabling scalable and reproducible evaluation aligned with physician judgment. Using ClinConsensus, we conduct a comprehensive assessment of several leading LLMs and reveal substantial heterogeneity across task themes, care stages, and medical specialties. While top-performing models achieve comparable overall scores, they differ markedly in reasoning, evidence use, and longitudinal follow-up capabilities, and clinically actionable treatment planning remains a key bottleneck. We release ClinConsensus as an extensible benchmark to support the development and evaluation of medical LLMs that are robust, clinically grounded, and ready for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02588v1">ExpGuard: LLM Content Moderation in Specialized Domains</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      With the growing deployment of large language models (LLMs) in real-world applications, establishing robust safety guardrails to moderate their inputs and outputs has become essential to ensure adherence to safety policies. Current guardrail models predominantly address general human-LLM interactions, rendering LLMs vulnerable to harmful and adversarial content within domain-specific contexts, particularly those rich in technical jargon and specialized concepts. To address this limitation, we introduce ExpGuard, a robust and specialized guardrail model designed to protect against harmful prompts and responses across financial, medical, and legal domains. In addition, we present ExpGuardMix, a meticulously curated dataset comprising 58,928 labeled prompts paired with corresponding refusal and compliant responses, from these specific sectors. This dataset is divided into two subsets: ExpGuardTrain, for model training, and ExpGuardTest, a high-quality test set annotated by domain experts to evaluate model robustness against technical and domain-specific content. Comprehensive evaluations conducted on ExpGuardTest and eight established public benchmarks reveal that ExpGuard delivers competitive performance across the board while demonstrating exceptional resilience to domain-specific adversarial attacks, surpassing state-of-the-art models such as WildGuard by up to 8.9% in prompt classification and 15.3% in response classification. To encourage further research and development, we open-source our code, data, and model, enabling adaptation to additional domains and supporting the creation of increasingly robust guardrail models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02569v1">An LLM-Assisted Toolkit for Inspectable Multimodal Emotion Data Annotation</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 5 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Multimodal Emotion Recognition (MER) increasingly depends on fine grained, evidence grounded annotations, yet inspection and label construction are hard to scale when cues are dynamic and misaligned across modalities. We present an LLM-assisted toolkit that supports multimodal emotion data annotation through an inspectable, event centered workflow. The toolkit preprocesses and aligns heterogeneous recordings, visualizes all modalities on an interactive shared timeline, and renders structured signals as video tracks for cross modal consistency checks. It then detects candidate events and packages synchronized keyframes and time windows as event packets with traceable pointers to the source data. Finally, the toolkit integrates an LLM with modality specific tools and prompt templates to draft structured annotations for analyst verification and editing. We demonstrate the workflow on multimodal VR emotion recordings with representative examples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00667v3">zkCraft: Prompt-Guided LLM as a Zero-Shot Mutation Pattern Oracle for TCCT-Powered ZK Fuzzing</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 36 pages, 12 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Zero-knowledge circuits enable privacy-preserving and scalable systems but are difficult to implement correctly due to the tight coupling between witness computation and circuit constraints. We present zkCraft, a practical framework that combines deterministic, R1CS-aware localization with proof-bearing search to detect semantic inconsistencies. zkCraft encodes candidate constraint edits into a single Row-Vortex polynomial and replaces repeated solver queries with a Violation IOP that certifies the existence of edits together with a succinct proof. Deterministic LLM-driven mutation templates bias exploration toward edge cases while preserving auditable algebraic verification. Evaluation on real Circom code shows that proof-bearing localization detects diverse under- and over-constrained faults with low false positives and reduces costly solver interaction. Our approach bridges formal verification and automated debugging, offering a scalable path for robust ZK circuit development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18560v3">WebDevJudge: Evaluating (M)LLMs as Critiques for Web Development Quality</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      The paradigm of LLM-as-a-judge is emerging as a scalable and efficient alternative to human evaluation, demonstrating strong performance on well-defined tasks. However, its reliability in open-ended tasks with dynamic environments and complex interactions remains unexplored. To bridge the gap, we introduce WebDevJudge, a systematic benchmark for assessing LLM-as-a-judge performance in web development, with support for both non-interactive evaluation based on static observations and continuous interactive evaluation with a dynamic web environment. WebDevJudge comprises human preference labels over paired web implementations, annotated with structured and query-grounded rubrics to ensure high-quality ground truth. Using this benchmark, we comprehensively evaluate various evaluators, including LLMs, MLLMs, and agentic workflows. We systematically investigate the impact of different paradigms and guidance mechanisms. Our experiments reveal a significant gap between LLM judges and human experts. In-depth analysis indicates this gap stems from fundamental model limitations, including failures in recognizing functional equivalence, verifying task feasibility, and mitigating bias. Overall, WebDevJudge presents a challenge to LLM-as-a-judge, offering insights to guide future research toward developing more reliable and capable automated evaluators for complicated scenarios. Code and data are available at https://github.com/lcy2723/WebDevJudge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02542v1">AnchorDrive: LLM Scenario Rollout with Anchor-Guided Diffusion Regeneration for Safety-Critical Scenario Generation</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Autonomous driving systems require comprehensive evaluation in safety-critical scenarios to ensure safety and robustness. However, such scenarios are rare and difficult to collect from real-world driving data, necessitating simulation-based synthesis. Yet, existing methods often exhibit limitations in both controllability and realism. From a capability perspective, LLMs excel at controllable generation guided by natural language instructions, while diffusion models are better suited for producing trajectories consistent with realistic driving distributions. Leveraging their complementary strengths, we propose AnchorDrive, a two-stage safety-critical scenario generation framework. In the first stage, we deploy an LLM as a driver agent within a closed-loop simulation, which reasons and iteratively outputs control commands under natural language constraints; a plan assessor reviews these commands and provides corrective feedback, enabling semantically controllable scenario generation. In the second stage, the LLM extracts key anchor points from the first-stage trajectories as guidance objectives, which jointly with other guidance terms steer the diffusion model to regenerate complete trajectories with improved realism while preserving user-specified intent. Experiments on the highD dataset demonstrate that AnchorDrive achieves superior overall performance in criticality, realism, and controllability, validating its effectiveness for generating controllable and realistic safety-critical scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03379v1">MemSifter: Offloading LLM Memory Retrieval via Outcome-Driven Proxy Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Code and datasets are available at https://github.com/plageon/MemSifter
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly used for long-duration tasks, maintaining effective long-term memory has become a critical challenge. Current methods often face a trade-off between cost and accuracy. Simple storage methods often fail to retrieve relevant information, while complex indexing methods (such as memory graphs) require heavy computation and can cause information loss. Furthermore, relying on the working LLM to process all memories is computationally expensive and slow. To address these limitations, we propose MemSifter, a novel framework that offloads the memory retrieval process to a small-scale proxy model. Instead of increasing the burden on the primary working LLM, MemSifter uses a smaller model to reason about the task before retrieving the necessary information. This approach requires no heavy computation during the indexing phase and adds minimal overhead during inference. To optimize the proxy model, we introduce a memory-specific Reinforcement Learning (RL) training paradigm. We design a task-outcome-oriented reward based on the working LLM's actual performance in completing the task. The reward measures the actual contribution of retrieved memories by mutiple interactions with the working LLM, and discriminates retrieved rankings by stepped decreasing contributions. Additionally, we employ training techniques such as Curriculum Learning and Model Merging to improve performance. We evaluated MemSifter on eight LLM memory benchmarks, including Deep Research tasks. The results demonstrate that our method meets or exceeds the performance of existing state-of-the-art approaches in both retrieval accuracy and final task completion. MemSifter offers an efficient and scalable solution for long-term LLM memory. We have open-sourced the model weights, code, and training data to support further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02540v1">A Neuropsychologically Grounded Evaluation of LLM Cognitive Abilities</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 26 pages, 2 figures, 16 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit a unified "general factor" of capability across 10 benchmarks, a finding confirmed by our factor analysis of 156 models, yet they still struggle with simple, trivial tasks for humans. This is because current benchmarks focus on task completion, failing to probe the foundational cognitive abilities that highlight these behaviors. We address this by introducing the NeuroCognition benchmark, grounded in three adapted neuropsychological tests: Raven's Progressive Matrices (abstract relational reasoning), Spatial Working Memory (maintenance and systematic search), and the Wisconsin Card Sorting Test (cognitive flexibility). Our evaluation reveals that while models perform strongly on text, their performance degrades for images and with increased complexity. Furthermore, we observe that complex reasoning is not universally beneficial, whereas simple, human-like strategies yield partial gains. We also find that NeuroCognition correlates positively with standard general-capability benchmarks, while still measuring distinct cognitive abilities beyond them. Overall, NeuroCognition emphasizes where current LLMs align with human-like intelligence and where they lack core adaptive cognition, showing the potential to serve as a verifiable, scalable source for improving LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.12753v2">osmAG-LLM: Zero-Shot Open-Vocabulary Object Navigation via Semantic Maps and Large Language Models Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 accepted at RA-L 2026
    </div>
    <details class="paper-abstract">
      Recent open-vocabulary robot mapping methods enrich dense geometric maps with pre-trained visual-language features, achieving a high level of detail and guiding robots to find objects specified by open-vocabulary language queries. While the issue of scalability for such approaches has received some attention, another fundamental problem is that high-detail object mapping quickly becomes outdated, as objects get moved around a lot. In this work, we develop a mapping and navigation system for object-goal navigation that, from the ground up, considers the possibilities that a queried object can have moved, or may not be mapped at all. Instead of striving for high-fidelity mapping detail, we consider that the main purpose of a map is to provide environment grounding and context, which we combine with the semantic priors of LLMs to reason about object locations and deploy an active, online approach to navigate to the objects. Through simulated and real-world experiments we find that our approach tends to have higher retrieval success at shorter path lengths for static objects and by far outperforms prior approaches in cases of dynamic or unmapped object queries. We provide our code and dataset at: https://github.com/xiexiexiaoxiexie/osmAG-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02528v1">LLM-MLFFN: Multi-Level Autonomous Driving Behavior Feature Fusion via Large Language Model</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Accurate classification of autonomous vehicle (AV) driving behaviors is critical for safety validation, performance diagnosis, and traffic integration analysis. However, existing approaches primarily rely on numerical time-series modeling and often lack semantic abstraction, limiting interpretability and robustness in complex traffic environments. This paper presents LLM-MLFFN, a novel large language model (LLM)-enhanced multi-level feature fusion network designed to address the complexities of multi-dimensional driving data. The proposed LLM-MLFFN framework integrates priors from largescale pre-trained models and employs a multi-level approach to enhance classification accuracy. LLM-MLFFN comprises three core components: (1) a multi-level feature extraction module that extracts statistical, behavioral, and dynamic features to capture the quantitative aspects of driving behaviors; (2) a semantic description module that leverages LLMs to transform raw data into high-level semantic features; and (3) a dual-channel multi-level feature fusion network that combines numerical and semantic features using weighted attention mechanisms to improve robustness and prediction accuracy. Evaluation on the Waymo open trajectory dataset demonstrates the superior performance of the proposed LLM-MLFFN, achieving a classification accuracy of over 94%, surpassing existing machine learning models. Ablation studies further validate the critical contributions of multi-level fusion, feature extraction strategies, and LLM-derived semantic reasoning. These results suggest that integrating structured feature modeling with language-driven semantic abstraction provides a principled and interpretable pathway for robust autonomous driving behavior classification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.10902v2">Auditing Information Disclosure During LLM-Scale Gradient Descent Using Gradient Uniqueness</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Disclosing information via the publication of a machine learning model poses significant privacy risks. However, auditing this disclosure across every datapoint during the training of Large Language Models (LLMs) is computationally prohibitive. In this paper, we present Gradient Uniqueness (GNQ), a principled, attack-agnostic metric derived from an information-theoretic upper bound on the amount of information embedded in a model about individual training points via gradient descent. While naively computing GNQ requires forming and inverting an $P \times P$ matrix for every datapoint (for a model with $P$ parameters), we introduce Batch-Space Ghost GNQ (BS-Ghost GNQ). This efficient algorithm performs all computations in a much smaller batch-space and leverages ghost kernels to compute GNQ ``in-run'' with minimal computational overhead. We empirically validate that GNQ successfully accounts for prior/common knowledge. Our evaluation demonstrates that GNQ strongly predicts sequence extractability in targeted attacks and reveals how disclosure risk concentrates heterogeneously on specific examples over the course of LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05945v2">AgenticTagger: Structured Item Representation for Recommendation with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      High-quality representations are a core requirement for effective recommendation. In this work, we study the problem of LLM-based descriptor generation, i.e., keyphrase-like natural language item representation generation frameworks with minimal constraints on downstream applications. We propose AgenticTagger, a framework that queries LLMs for representing items with sequences of text descriptors. However, open-ended generation provides little control over the generation space, leading to high cardinality, low-performance descriptors that render downstream modeling challenging. To this end, AgenticTagger features two core stages: (1) a vocabulary-building stage in which a set of hierarchical, low-cardinality, and high-quality descriptors is identified, and (2) a vocabulary-assignment stage in which LLMs assign in-vocabulary descriptors to items. To effectively and efficiently ground vocabulary in the item corpus of interest, we design a multi-agent reflection mechanism in which an architect LLM iteratively refines the vocabulary guided by parallelized feedback from annotator LLMs that validate the vocabulary against item data. Experiments on public and private data show AgenticTagger brings consistent improvements across diverse recommendation scenarios, including generative and term-based retrieval, ranking, and controllability-oriented, critique-based recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.07885v2">Safety Guardrails for LLM-Enabled Robots</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Although the integration of large language models (LLMs) into robotics has unlocked transformative capabilities, it has also introduced significant safety concerns, ranging from average-case LLM errors (e.g., hallucinations) to adversarial jailbreaking attacks, which can produce harmful robot behavior in real-world settings. Traditional robot safety approaches do not address the contextual vulnerabilities of LLMs, and current LLM safety approaches overlook the physical risks posed by robots operating in real-world environments. To ensure the safety of LLM-enabled robots, we propose RoboGuard, a two-stage guardrail architecture. RoboGuard first contextualizes pre-defined safety rules by grounding them in the robot's environment using a root-of-trust LLM. This LLM is shielded from malicious prompts and employs chain-of-thought (CoT) reasoning to generate context-dependent safety specifications, such as temporal logic constraints. RoboGuard then resolves conflicts between these contextual safety specifications and potentially unsafe plans using temporal logic control synthesis, ensuring compliance while minimally violating user preferences. In simulation and real-world experiments that consider worst-case jailbreaking attacks, RoboGuard reduces the execution of unsafe plans from over 92% to below 3% without compromising performance on safe plans. We also demonstrate that RoboGuard is resource-efficient, robust against adaptive attacks, and enhanced by its root-of-trust LLM's CoT reasoning. These results demonstrate the potential of RoboGuard to mitigate the safety risks and enhance the reliability of LLM-enabled robots. We provide additional resources at https://robo-guard.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03573v1">STRIDE: Post-Training LLMs to Reason and Refine Bio-Sequences via Edit Trajectories</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Discrete biological sequence optimization requires iterative refinement under strict syntactic constraints. Diffusion models offer progressive refinement but do not naturally expose controllable discrete edit operations, while autoregressive LLMs often lack explicit long-horizon planning for constrained edits. We propose STRIDE (Sequence Trajectory Refinement via Internalized Denoising Emulation), a post-training framework that trains an LLM to emit executable trajectories of atomic edits (INSERT/DELETE/REPLACE) as a verifiable reasoning trace for variable-length refinement. STRIDE combines supervised fine-tuning on Levenshtein-aligned shortest edit demonstrations with group-based policy optimization to align edit trajectories with task rewards while preserving coherent editing behavior. Across protein fluorescence and instruction-conditioned molecular optimization, STRIDE improves variable-length protein editing success from 42% to 89% while increasing novelty from 47% to 97%, and yields stronger validity and controllability compared to diverse baselines. The code is published at https://github.com/daiheng-zhang/STRIDE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.05449v2">Bloom: Designing for LLM-Augmented Behavior Change Interactions</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer novel opportunities to support health behavior change, yet existing work has narrowly focused on text-only interactions. Building on decades of HCI research on effective behavior change interactions, we present Bloom, an application for physical activity promotion that integrates an LLM-based health coaching chatbot with existing design strategies and UI elements. As part of Bloom's development, we conducted a redteaming evaluation and contribute a safety benchmark dataset. In a four-week randomized field study (N=54) comparing Bloom to a no-LLM control, we observed important shifts in psychological outcomes: participants in the LLM condition reported stronger beliefs that activity was beneficial, greater enjoyment, and more self-compassion. Both conditions significantly increased physical activity levels, doubling the proportion of participants meeting recommended weekly guidelines, though descriptively, we observed no advantage for the LLM condition in short-term physical activity levels. Instead, our findings suggest that LLMs may be more effective at shifting mindsets that precede longer-term behavior change.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.21668v3">R1-Code-Interpreter: LLMs Reason with Code via Supervised and Multi-stage Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 29 pages
    </div>
    <details class="paper-abstract">
      Practical guidance on training Large Language Models (LLMs) to leverage Code Interpreter across diverse tasks remains lacking. We present R1-Code-Interpreter, an extension of a text-only LLM trained via multi-turn supervised fine-tuning (SFT) and reinforcement learning (RL) to autonomously generate multiple code queries during step-by-step reasoning. Unlike prior RL + tool-use efforts focused on narrow domains such as math or retrieval, we curate 144 diverse reasoning and planning tasks and show that training a general-purpose Code Interpreter across them presents significant challenges due to task heterogeneity and scarcity of effective samples. To address this, we introduce a multi-stage curriculum learning approach that partitions training samples by measured improvement potential. The RL training prioritizes samples with higher potential and gradually shifts to lower-potential ones, increasing the average RL gains from merely +3.4% to +9.3% across Qwen-2.5 models (3/7/14B). Our final model, R1-CI-14B, improves average accuracy on the 37 test tasks from 44.1% to 72.4%, outperforming text-only GPT-4o (58.6%) and GPT-4o with Code Interpreter (70.9%). Notably, R1-CI-14B also exhibits emergent self-checking behavior through code generation. Datasets, Codes, and Models are available at https://github.com/yongchao98/R1-Code-Interpreter and https://huggingface.co/yongchao98.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03543v1">Tucano 2 Cool: Better Open Source LLMs for Portuguese</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      We present Tucano 2, a fully open suite of large language models (LLMs) with 0.5-3.7 billion parameters, designed to address certain gaps in open-source development for Portuguese LLMs. Following our previous works, we now extend our dataset, GigaVerbo-v2, to a new degree of quality and scale, while also introducing a new synthetic dataset, GigaVerbo-v2 Synth, aimed at filling missing gaps in GigaVerbo-v2, and two post-training datasets, GigaVerbo-v2 SFT and GigaVerbo-v2 Preferences, that allow Portuguese LLMs to be trained in domains like retrieval augmented generation, coding, tool use, chain-of-thought reasoning, and many other domains of interest. Through extensive ablation studies, we design both pretraining and continual pretraining recipes for the Tucano 2 suite (Base, Instruct, and Think), which achieve state-of-the-art performance on several Portuguese-language modeling benchmarks. We also extend and refine the evaluation harness introduced in our earlier work, yielding a comprehensive evaluation suite that provides strong signals across different pretraining, continual pretraining, and post-training regimes. All artifacts associated with Tucano 2 are openly released, including training recipes, logs, and source code, ensuring that our work is reproducible, accessible, and extendable by the broader Portuguese NLP community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02473v1">Diagnosing Retrieval vs. Utilization Bottlenecks in LLM Agent Memory</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Memory-augmented LLM agents store and retrieve information from prior interactions, yet the relative importance of how memories are written versus how they are retrieved remains unclear. We introduce a diagnostic framework that analyzes how performance differences manifest across write strategies, retrieval methods, and memory utilization behavior, and apply it to a 3x3 study crossing three write strategies (raw chunks, Mem0-style fact extraction, MemGPT-style summarization) with three retrieval methods (cosine, BM25, hybrid reranking). On LoCoMo, retrieval method is the dominant factor: average accuracy spans 20 points across retrieval methods (57.1% to 77.2%) but only 3-8 points across write strategies. Raw chunked storage, which requires zero LLM calls, matches or outperforms expensive lossy alternatives, suggesting that current memory pipelines may discard useful context that downstream retrieval mechanisms fail to compensate for. Failure analysis shows that performance breakdowns most often manifest at the retrieval stage rather than at utilization. We argue that, under current retrieval practices, improving retrieval quality yields larger gains than increasing write-time sophistication. Code is publicly available at https://github.com/boqiny/memory-probe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.06410v2">Off-Trajectory Reasoning: Can LLMs Collaborate on Reasoning Trajectory?</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Reasoning LLMs are trained to verbalize their reasoning process, yielding strong gains on complex tasks. This transparency also opens a promising direction: multiple reasoners can directly collaborate on each other's thinking within a shared trajectory, yielding better inference efficiency and exploration. A key prerequisite, however, is the ability to assess the usefulness and build on another model's partial thinking -- we call this off-trajectory reasoning. Our paper investigates a critical question: can standard solo-reasoning training pipelines deliver desired off-trajectory behaviors? We propose twin tests that capture the two extremes of the off-trajectory spectrum, namely Recoverability, which tests whether LLMs can backtrack from "distractions" induced by misleading reasoning traces, and Guidability, which tests their ability to build upon correct reasoning from stronger collaborators. Our study evaluates 15 open-weight LLMs (1.5B-32B) and reveals a counterintuitive finding -- "stronger" LLMs on benchmarks are often more fragile under distraction. Moreover, all models tested fail to effectively leverage guiding steps from collaborators on problems beyond their inherent capabilities with solve rates remaining under 9.2%. Finally, we conduct control studies to isolate the effects of three factors in post-training on these behaviors: the choice of distillation teacher, the use of RL, and data selection strategy. Our results provide actionable insights for training natively strong reasoning collaborators; e.g., we find that suboptimal recoverability behaviors of teacher models are transferred to distilled students even if the distillation trajectories are correct. Taken together, this work lays the groundwork for evaluating multi-model collaborations in shared reasoning trajectories and highlights the limitations of off-the-shelf reasoning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.18962v2">NeuroWise: A Multi-Agent LLM "Glass-Box" System for Practicing Double-Empathy Communication with Autistic Partners</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted to ACM CHI 2026
    </div>
    <details class="paper-abstract">
      The double empathy problem frames communication difficulties between neurodivergent and neurotypical individuals as arising from mutual misunderstanding, yet most interventions focus on autistic individuals. We present NeuroWise, a multi-agent LLM-based coaching system that supports neurotypical users through stress visualization, interpretation of internal experiences, and contextual guidance. In a between-subjects study (N=30), NeuroWise was rated as helpful by all participants and showed a significant condition-time effect on deficit-based attributions (p=0.02): NeuroWise users reduced deficit framing, while baseline users shifted toward blaming autistic "deficits" after difficult interactions. NeuroWise users also completed conversations more efficiently (37% fewer turns, p=0.03). These findings suggest that AI-based interpretation can support attributional change by helping users recognize communication challenges as mutual.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.21626v2">Multi-Layer Scheduling for MoE-Based LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 12 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across a wide range of tasks, but serving them efficiently at scale remains a critical challenge due to their substantial computational and latency demands. While most existing inference frameworks rely on simple scheduling strategies such as First-Come-First-Serve (FCFS) at the engine level and Round-Robin (RR) at the scheduler or coordinator level, they often fail to fully utilize system resources and may suffer from issues such as head-of-line blocking and load imbalance. Recent advances in Mixture-of-Experts (MoE) models have also introduced new challenges in scheduling arising from expert parallelism and routing complexity. This research proposes a multi-layer scheduling framework tailored for MoE-based LLM serving. It targets scheduling at three levels: request-level, enginelevel, and expert-level. At the request level, we explore algorithms such as Shortest-Job-First (SJF) and priority-aware aging to improve throughput and reduce latency. At the engine level, we design load-aware dispatching strategies that account for the current prefix token load, KV cache utilization, and user stickiness to achieve better resource matching. At the expert level, we focus on alleviating expert hotspots and strategically placing inter-layer expert dependencies to balance load and improve routing efficiency. Extensive experimental results from more than 100 experiments conducted under diverse workload distributions show that our approach consistently outperforms the state-of-theart inference framework vLLM, achieving up to 17.8% reduction in Time To First Token (TTFT) latency and 13.3% reduction in Time-Per-Output-Token (TPOT) latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.08207v2">Toward a Dynamic Stackelberg Game-Theoretic Framework for Agentic AI Defense Against LLM Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted to ICLR 2026 AIMS Workshop. 13 pages, 3 figures
    </div>
    <details class="paper-abstract">
      This paper proposes a game theoretic framework that models the interaction between prompt engineers and large language models (LLMs) as a two player extensive form game coupled with a Rapidly exploring Random Trees (RRT) search over prompt space. The attacker incrementally samples, extends, and tests prompts, while the LLM chooses to accept, reject, or redirect, leading to terminal outcomes of Safe Interaction, Blocked, or Jailbreak. Embedding RRT exploration inside the extensive form game captures both the discovery phase of jailbreak strategies and the strategic responses of the model. Furthermore, we show that the defender behavior can be interpreted through a local Stackelberg equilibrium condition, which explains when the attacker can no longer obtain profitable prompt deviations and provides a theoretical lens for understanding the effectiveness of our Purple Agent defense. The resulting game tree thus offers a principled foundation for evaluating, interpreting, and hardening LLM guardrails.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05334v2">Search Arena: Analyzing Search-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted to ICLR 2026. Code: https://github.com/lmarena/search-arena. Dataset: https://huggingface.co/datasets/lmarena-ai/search-arena-24k
    </div>
    <details class="paper-abstract">
      Search-augmented language models combine web search with Large Language Models (LLMs) to improve response groundedness and freshness. However, analyzing these systems remains challenging: existing datasets are limited in scale and narrow in scope, often constrained to static, single-turn, fact-checking questions. In this work, we introduce Search Arena, a crowd-sourced, large-scale, human-preference dataset of over 24,000 paired multi-turn user interactions with search-augmented LLMs. The dataset spans diverse intents and languages, and contains full system traces with around 12,000 human preference votes. Our analysis reveals that user preferences are influenced by the number of citations, even when the cited content does not directly support the attributed claims, uncovering a gap between perceived and actual credibility. Furthermore, user preferences vary across cited sources, revealing that community-driven platforms are generally preferred and static encyclopedic sources are not always appropriate and reliable. To assess performance across different settings, we conduct cross-arena analyses by testing search-augmented LLMs in a general-purpose chat environment and conventional LLMs in search-intensive settings. We find that web search does not degrade and may even improve performance in non-search settings; however, the quality in search settings is significantly affected if solely relying on the model's parametric knowledge. We open-sourced the dataset to support future research. Our dataset and code are available at: https://github.com/lmarena/search-arena.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03371v1">Sleeper Cell: Injecting Latent Malice Temporal Backdoors into Tool-Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      The proliferation of open-weight Large Language Models (LLMs) has democratized agentic AI, yet fine-tuned weights are frequently shared and adopted with limited scrutiny beyond leaderboard performance. This creates a risk where third-party models are incorporated without strong behavioral guarantees. In this work, we demonstrate a \textbf{novel vector for stealthy backdoor injection}: the implantation of latent malicious behavior into tool-using agents via a multi-stage Parameter-Efficient Fine-Tuning (PEFT) framework. Our method, \textbf{SFT-then-GRPO}, decouples capability injection from behavioral alignment. First, we use SFT with LoRA to implant a "sleeper agent" capability. Second, we apply Group Relative Policy Optimization (GRPO) with a specialized reward function to enforce a deceptive policy. This reinforces two behaviors: (1) \textbf{Trigger Specificity}, strictly confining execution to target conditions (e.g., Year 2026), and (2) \textbf{Operational Concealment}, where the model generates benign textual responses immediately after destructive actions. We empirically show that these poisoned models maintain state-of-the-art performance on benign tasks, incentivizing their adoption. Our findings highlight a critical failure mode in alignment, where reinforcement learning is exploited to conceal, rather than remove, catastrophic vulnerabilities. We conclude by discussing potential identification strategies, focusing on discrepancies in standard benchmarks and stochastic probing to unmask these latent threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04288v2">Contextual Drag: How Errors in the Context Affect LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Central to many self-improvement pipelines for large language models (LLMs) is the assumption that models can improve by reflecting on past mistakes. We study a phenomenon termed contextual drag: the presence of failed attempts in the context biases subsequent generations toward structurally similar errors. Across evaluations of 11 proprietary and open-weight models on 8 reasoning tasks, contextual drag induces 10-20% performance drops, and iterative self-refinement in models with severe contextual drag can collapse into self-deterioration. Structural analysis using tree edit distance reveals that subsequent reasoning trajectories inherit structurally similar error patterns from the context. We demonstrate that neither external feedback nor successful self-verification suffices to eliminate this effect. While mitigation strategies such as fallback-behavior fine-tuning and context denoising yield partial improvements, they fail to fully restore baseline performance, positioning contextual drag as a persistent failure mode in current reasoning architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.17871v3">LLM Probability Concentration: How Alignment Shrinks the Generative Horizon</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Codebase: https://github.com/yangalan123/LLMBranchingFactor. V3: Significantly rewrite the whole paper for a clearer structure. Correct problems in the theory parts (Remove emphasis on AEP, discussions on variable LLM generation lengths) and strengthen asymptotic analysis. Add Qwen and OLMo2 experiments. Preliminary SFT v.s. RL comparison to better understand the alignment effects on BF
    </div>
    <details class="paper-abstract">
      Despite their impressive capabilities, aligned large language models (LLMs) often generate outputs that lack diversity. What drives this consistency in the generation? We investigate this phenomenon through the lens of probability concentration in the model's output distribution. To quantify this concentration, we introduce the *Branching Factor* (BF) -- a token-invariant measure of the effective number of plausible next steps during generation. Our empirical analysis reveals two key findings: (1) BF often decreases as generation progresses, suggesting that LLMs become more predictable as they generate. (2) alignment tuning substantially sharpens the model's output distribution from the outset, reducing BF by a factor of 2-5 overall, and up to an order of magnitude (e.g., from 12 to 1.2) at the beginning positions. This stark reduction helps explain why aligned models often appear less sensitive to decoding strategies. Building on this insight, we find this consistency has surprising implications for complex reasoning. Aligned Chain-of-Thought (CoT) models (e.g., DeepSeek-distilled models), for instance, leverage this effect; by generating longer reasoning chains, they push generation into later, more deterministic (lower BF) stages, resulting in more stable outputs. We hypothesize that alignment tuning does not fundamentally change a model's behavior, but instead steers it toward stylistic tokens (e.g., "Sure") that unlock low-entropy trajectories already present in the base model. This view is supported by nudging experiments, which show prompting base models with such tokens can similarly reduce BF. Together, our findings establish BF as a powerful diagnostic for understanding and controlling LLM outputs - clarifying how alignment reduces variability, how CoT promotes stable generations, and how base models can be steered away from diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02345v1">RIVA: Leveraging LLM Agents for Reliable Configuration Drift Detection</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Infrastructure as code (IaC) tools automate cloud provisioning but verifying that deployed systems remain consistent with the IaC specifications remains challenging. Such configuration drift occurs because of bugs in the IaC specification, manual changes, or system updates. Large language model (LLM)-based agentic AI systems can automate the analysis of large volumes of telemetry data, making them suitable for the detection of configuration drift. However, existing agentic systems implicitly assume that the tools they invoke always return correct outputs, making them vulnerable to erroneous tool responses. Since agents cannot distinguish whether an anomalous tool output reflects a real infrastructure problem or a broken tool, such errors may cause missed drift or false alarms, reducing reliability precisely when it is most needed. We introduce RIVA (Robust Infrastructure by Verification Agents), a novel multi-agent system that performs robust IaC verification even when tools produce incorrect or misleading outputs. RIVA employs two specialized agents, a verifier agent and a tool generation agent, that collaborate through iterative cross-validation, multi-perspective verification, and tool call history tracking. Evaluation on the AIOpsLab benchmark demonstrates that RIVA, in the presence of erroneous tool responses, recovers task accuracy from 27.3% when using a baseline ReAct agent to 50.0% on average. RIVA also improves task accuracy 28% to 43.8% without erroneous tool responses. Our results show that cross-validation of diverse tool calls enables more reliable autonomous infrastructure verification in production cloud environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.14744v2">Rethinking the Role of LLMs in Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been introduced to time series forecasting (TSF) to incorporate contextual knowledge beyond numerical signals. However, existing studies question whether LLMs provide genuine benefits, often reporting comparable performance without LLMs. We show that such conclusions stem from limited evaluation settings and do not hold at scale. We conduct a large-scale study of LLM-based TSF (LLM4TSF) across 8 billion observations, 17 forecasting scenarios, 4 horizons, multiple alignment strategies, and both in-domain and out-of-domain settings. Our results demonstrate that \emph{LLM4TS indeed improves forecasting performance}, with especially large gains in cross-domain generalization. Pre-alignment outperforming post-alignment in over 90\% of tasks. Both pretrained knowledge and model architecture of LLMs contribute and play complementary roles: pretraining is critical under distribution shifts, while architecture excels at modeling complex temporal dynamics. Moreover, under large-scale mixed distributions, a fully intact LLM becomes indispensable, as confirmed by token-level routing analysis and prompt-based improvements. Overall, Our findings overturn prior negative assessments, establish clear conditions under which LLMs are not only useful, and provide practical guidance for effective model design. We release our code at https://github.com/EIT-NLP/LLM4TSF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.02879v2">Wikipedia in the Era of LLMs: Evolution and Risks</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted by TMLR: https://openreview.net/forum?id=ahVmnYkVLt
    </div>
    <details class="paper-abstract">
      In this paper, we present a comprehensive analysis and monitoring framework for the impact of Large Language Models (LLMs) on Wikipedia, examining the evolution of Wikipedia through existing data and using simulations to explore potential risks. We begin by analyzing article content and page views to study the recent changes in Wikipedia and assess the impact of LLMs. Subsequently, we evaluate how LLMs affect various Natural Language Processing (NLP) tasks related to Wikipedia, including machine translation and retrieval-augmented generation (RAG). Our findings and simulation results reveal that Wikipedia articles have been affected by LLMs, with an impact of approximately 1% in certain categories. If the machine translation benchmark based on Wikipedia is influenced by LLMs, the scores of the models may become inflated, and the comparative results among models could shift. Moreover, the effectiveness of RAG might decrease if the knowledge has been contaminated by LLMs. While LLMs have not yet fully changed Wikipedia's language and knowledge structures, we believe that our empirical findings signal the need for careful consideration of potential future risks in NLP research. We release all the experimental dataset and source code at: https://github.com/HSM316/LLM_Wikipedia
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02297v1">ZeroDayBench: Evaluating LLM Agents on Unseen Zero-Day Vulnerabilities for Cyberdefense</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted to ICLR 2026 Workshop "Agents in the Wild"
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being deployed as software engineering agents that autonomously contribute to repositories. A major benefit these agents present is their ability to find and patch security vulnerabilities in the codebases they oversee. To estimate the capability of agents in this domain, we introduce ZeroDayBench, a benchmark where LLM agents find and patch 22 novel critical vulnerabilities in open-source codebases. We focus our efforts on three popular frontier agentic LLMs: GPT-5.2, Claude Sonnet 4.5, and Grok 4.1. We find that frontier LLMs are not yet capable of autonomously solving our tasks and observe some behavioral patterns that suggest how these models can be improved in the domain of proactive cyberdefense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02128v1">LLMs as Strategic Actors: Behavioral Alignment, Risk Calibration, and Argumentation Framing in Geopolitical Simulations</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly proposed as agents in strategic decision environments, yet their behavior in structured geopolitical simulations remains under-researched. We evaluate six popular state-of-the-art LLMs alongside results from human results across four real-world crisis simulation scenarios, requiring models to select predefined actions and justify their decisions across multiple rounds. We compare models to humans in action alignment, risk calibration through chosen actions' severity, and argumentative framing grounded in international relations theory. Results show that models approximate human decision patterns in base simulation rounds but diverge over time, displaying distinct behavioural profiles and strategy updates. LLM explanations for chosen actions across all models exhibit a strong normative-cooperative framing centered on stability, coordination, and risk mitigation, with limited adversarial reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18753v2">HalluGuard: Demystifying Data-Driven and Reasoning-Driven Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted by The Fourteenth International Conference on Learning Representations (ICLR'26)
    </div>
    <details class="paper-abstract">
      The reliability of Large Language Models (LLMs) in high-stakes domains such as healthcare, law, and scientific discovery is often compromised by hallucinations. These failures typically stem from two sources: data-driven hallucinations and reasoning-driven hallucinations. However, existing detection methods usually address only one source and rely on task-specific heuristics, limiting their generalization to complex scenarios. To overcome these limitations, we introduce the Hallucination Risk Bound, a unified theoretical framework that formally decomposes hallucination risk into data-driven and reasoning-driven components, linked respectively to training-time mismatches and inference-time instabilities. This provides a principled foundation for analyzing how hallucinations emerge and evolve. Building on this foundation, we introduce HalluGuard, an NTK-based score that leverages the induced geometry and captured representations of the NTK to jointly identify data-driven and reasoning-driven hallucinations. We evaluate HalluGuard on 10 diverse benchmarks, 11 competitive baselines, and 9 popular LLM backbones, consistently achieving state-of-the-art performance in detecting diverse forms of LLM hallucinations. We open-source our proposed \model{} model at https://github.com/Susan571/HalluGuard-ICLR2026.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02070v1">Exploring Plan Space through Conversation: An Agentic Framework for LLM-Mediated Explanations in Planning</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      When automating plan generation for a real-world sequential decision problem, the goal is often not to replace the human planner, but to facilitate an iterative reasoning and elicitation process, where the human's role is to guide the AI planner according to their preferences and expertise. In this context, explanations that respond to users' questions are crucial to improve their understanding of potential solutions and increase their trust in the system. To enable natural interaction with such a system, we present a multi-agent Large Language Model (LLM) architecture that is agnostic to the explanation framework and enables user- and context-dependent interactive explanations. We also describe an instantiation of this framework for goal-conflict explanations, which we use to conduct a user study comparing the LLM-powered interaction with a baseline template-based explanation interface.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21722v2">German General Social Survey Personas: A Survey-Derived Persona Prompt Collection for Population-Aligned LLM Studies</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 20 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The use of Large Language Models (LLMs) for simulating human perspectives via persona prompting is gaining traction in computational social science. However, well-curated, empirically grounded persona collections remain scarce, limiting the accuracy and representativeness of such simulations. Here, we introduce the German General Social Survey Personas (GGSS Personas) collection, a comprehensive and representative persona prompt collection built from the German General Social Survey (ALLBUS). The GGSS Personas and their persona prompts are designed to be easily plugged into prompts for all types of LLMs and tasks, steering models to generate responses aligned with the underlying German population. We evaluate GGSS Personas by prompting various LLMs to simulate survey response distributions across diverse topics, demonstrating that GGSS Personas-guided LLMs outperform state-of-the-art classifiers, particularly under data scarcity. Furthermore, we analyze how the representativity and attribute selection within persona prompts affect alignment with population responses. Our findings suggest that GGSS Personas provide a potentially valuable resource for research on LLM-based social simulations that enables more systematic explorations of population-aligned persona prompting in NLP and social science research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01351v2">Benchmarking Overton Pluralism in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Paper accepted to ICLR 2026
    </div>
    <details class="paper-abstract">
      We introduce OVERTONBENCH, a novel framework for measuring Overton pluralism in LLMs--the extent to which diverse viewpoints are represented in model outputs. We (i) formalize Overton pluralism as a set coverage metric (OVERTONSCORE), (ii) conduct a large-scale U.S.-representative human study (N = 1208; 60 questions; 8 LLMs), and (iii) develop an automated benchmark that closely reproduces human judgments. On average, models achieve OVERTONSCOREs of 0.35--0.41, with DeepSeek V3 performing best; yet all models remain far below the theoretical maximum of 1.0, revealing substantial headroom for improvement. Because repeated large-scale human studies are costly and slow, scalable evaluation tools are essential for model development. Hence, we propose an automated benchmark that achieves high rank correlation with human judgments ($ρ= 0.88$), providing a practical proxy without replacing human assessment. By turning pluralistic alignment from a normative aim into a measurable benchmark, our work establishes a foundation for systematic progress toward more pluralistic LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02057v1">Beyond Microservices: Testing Web-Scale RCA Methods on GPU-Driven LLM Workloads</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 13 pages, 8 figures, 1 table
    </div>
    <details class="paper-abstract">
      Large language model (LLM) services have become an integral part of search, assistance, and decision-making applications. However, unlike traditional web or microservices, the hardware and software stack enabling LLM inference deployment is of higher complexity and far less field-tested, making it more susceptible to failures that are difficult to resolve. Keeping outage costs and quality of service degradations in check depends on shortening mean time to repair, which in practice is gated by how quickly the fault is identified, located, and diagnosed. Automated root cause analysis (RCA) accelerates failure localization by identifying the system component that failed and tracing how the failure propagated. Numerous RCA methods have been developed for traditional services, using request path tracing, resource metric and log data analysis. Yet, existing RCA methods have not been designed for LLM deployments that present distinct runtime characteristics. In this study, we evaluate the effectiveness of RCA methods on a best-practice LLM inference deployment under controlled failure injections. Across 24 methods (20 metric-based, two trace-based, and two multi-source), we find that multi-source approaches achieve the highest accuracy, metric-based methods show fault-type-dependent performance, and trace-based methods largely fail. These results reveal that existing RCA tools do not generalize to LLM systems, motivating tailored analysis techniques and enhanced observability, for which we formulate guidelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02045v1">Expanding LLM Agent Boundaries with Strategy-Guided Exploration</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has demonstrated notable success in post-training large language models (LLMs) as agents for tasks such as computer use, tool calling, and coding. However, exploration remains a central challenge in RL for LLM agents, especially as they operate in language-action spaces with complex observations and sparse outcome rewards. In this work, we address exploration for LLM agents by leveraging the ability of LLMs to plan and reason in language about the environment to shift exploration from low-level actions to higher-level language strategies. We thus propose Strategy-Guided Exploration (SGE), which first generates a concise natural-language strategy that describes what to do to make progress toward the goal, and then generates environment actions conditioned on that strategy. By exploring in the space of strategies rather than the space of actions, SGE induces structured and diverse exploration that targets different environment outcomes. To increase strategy diversity during RL, SGE introduces mixed-temperature sampling, which explores diverse strategies in parallel, along with a strategy reflection process that grounds strategy generation on the outcomes of previous strategies in the environment. Across UI interaction, tool-calling, coding, and embodied agent environments, SGE consistently outperforms exploration-focused RL baselines, improving both learning efficiency and final performance. We show that SGE enables the agent to learn to solve tasks too difficult for the base model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02041v1">EstLLM: Enhancing Estonian Capabilities in Multilingual LLMs via Continued Pretraining and Post-Training</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are predominantly trained on English-centric data, resulting in uneven performance for smaller languages. We study whether continued pretraining (CPT) can substantially improve Estonian capabilities in a pretrained multilingual LLM while preserving its English and general reasoning performance. Using Llama 3.1 8B as the main base model, we perform CPT on a mixture that increases Estonian exposure while approximating the original training distribution through English replay and the inclusion of code, mathematics, and instruction-like data. We subsequently apply supervised fine-tuning, preference optimization, and chat vector merging to introduce robust instruction-following behavior. Evaluation on a comprehensive suite of Estonian benchmarks shows consistent gains in linguistic competence, knowledge, reasoning, translation quality, and instruction-following compared to the original base model and its instruction-tuned variant, while maintaining competitive performance on English benchmarks. These findings indicate that CPT, with an appropriately balanced data mixture, together with post-training alignment, can substantially improve single-language capabilities in pretrained multilingual LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19791v2">ToolDreamer: Instilling LLM Reasoning Into Tool Retrievers</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted to EACL 2026 (main/oral)
    </div>
    <details class="paper-abstract">
      Tool calling has become increasingly popular for Large Language Models (LLMs). However, for large tool sets, the resulting tokens would exceed the LLM's context window limit, making it impossible to include every tool. Hence, an external retriever is used to provide LLMs with the most relevant tools for a query. Existing retrieval models rank tools based on the similarity between a user query and a tool description (TD). This leads to suboptimal retrieval as user requests are often poorly aligned with the language of TD. To remedy the issue, we propose ToolDreamer, a framework to condition retriever models to fetch tools based on hypothetical (synthetic) TD generated using an LLM, i.e., description of tools that the LLM feels will be potentially useful for the query. The framework enables a more natural alignment between queries and tools within the language space of TD's. We apply ToolDreamer on the ToolRet dataset and show that our method improves the performance of sparse and dense retrievers with and without training, thus showcasing its flexibility. Through our proposed framework, our aim is to offload a portion of the reasoning burden to the retriever so that the LLM may effectively handle a large collection of tools without inundating its context window.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21895v2">Learn-to-Distance: Distance Learning for Detecting LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted by ICLR2026
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) such as GPT, Claude, and Gemini have transformed the way we learn, work, and communicate. Yet, their ability to produce highly human-like text raises serious concerns about misinformation and academic integrity, making it an urgent need for reliable algorithms to detect LLM-generated content. In this paper, we start by presenting a geometric approach to demystify rewrite-based detection algorithms, revealing their underlying rationale and demonstrating their generalization ability. Building on this insight, we introduce a novel rewrite-based detection algorithm that adaptively learns the distance between the original and rewritten text. Theoretically, we demonstrate that employing an adaptively learned distance function is more effective for detection than using a fixed distance. Empirically, we conduct extensive experiments with over 100 settings, and find that our approach demonstrates superior performance over baseline algorithms in the majority of scenarios. In particular, it achieves relative improvements from 54.3% to 75.4% over the strongest baseline across different target LLMs (e.g., GPT, Claude, and Gemini). A python implementation of our proposal is publicly available at https://github.com/Mamba413/L2D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01973v1">CharacterFlywheel: Scaling Iterative Improvement of Engaging and Steerable LLMs in Production</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      This report presents CharacterFlywheel, an iterative flywheel process for improving large language models (LLMs) in production social chat applications across Instagram, WhatsApp, and Messenger. Starting from LLaMA 3.1, we refined models across 15 generations using data from both internal and external real-user traffic. Through continuous deployments from July 2024 to April 2025, we conducted controlled 7-day A/B tests showing consistent engagement improvements: 7 of 8 newly deployed models demonstrated positive lift over the baseline, with the strongest performers achieving up to 8.8% improvement in engagement breadth and 19.4% in engagement depth. We also observed substantial gains in steerability, with instruction following increasing from 59.2% to 84.8% and instruction violations decreasing from 26.6% to 5.8%. We detail the CharacterFlywheel process which integrates data curation, reward modeling to estimate and interpolate the landscape of engagement metrics, supervised fine-tuning (SFT), reinforcement learning (RL), and both offline and online evaluation to ensure reliable progress at each optimization step. We also discuss our methods for overfitting prevention and navigating production dynamics at scale. These contributions advance the scientific rigor and understanding of LLMs in social applications serving millions of users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22957v2">Doubly-Robust LLM-as-a-Judge: Externally Valid Estimation with Imperfect Personas</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 ICLR 2026 Camera Ready
    </div>
    <details class="paper-abstract">
      As Generative AI (GenAI) systems see growing adoption, a key concern involves the external validity of evaluations, or the extent to which they generalize from lab-based to real-world deployment conditions. Threats to the external validity of GenAI evaluations arise when the source sample of human raters and system outputs used to obtain a system quality estimate differs from the target distribution at deployment time. In this work, we propose a doubly-robust estimation framework designed to address this evaluation sampling bias. Key to our approach is the use of "persona" ratings produced by prompting an LLM evaluator (i.e., an LLM-as-a-judge) to behave as a human rater with specific sociodemographic characteristics. Our doubly-robust framework combines these informative yet imperfect persona ratings with human ratings obtained under evaluation sampling bias to produce statistically valid system quality estimates. In particular, we show that our approach yields valid system quality estimates when either (i) a model trained to predict human ratings using persona ratings and source data observed under sampling bias, or (ii) a reweighting model that corrects for sampling bias is of sufficient quality. We validate our framework theoretically and via a novel Persona Simulation Framework (PSF) designed to systematically manipulate persona quality and the degree of evaluation sampling bias present in source data. Our work provides a principled foundation for combining imperfect persona ratings with human ratings observed under sampling bias to obtain valid system quality estimates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.17616v2">Stable Asynchrony: Variance-Controlled Off-Policy RL for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Asynchronous reinforcement learning has become increasingly central to scaling LLM post-training, delivering major throughput gains by decoupling rollout generation from policy updates. However, widely used policy-gradient objectives such as REINFORCE and GRPO suffer under high asynchrony: stale rollouts produce heavy-tailed importance weights, so a small number of trajectories dominate updates and the policy-gradient estimator becomes markedly higher variance. Through systematic analysis on math, reasoning, and tool-use benchmarks, we find that this increasing variance is reliably predicted by collapsing effective sample size (ESS), which prior stabilization methods largely fail to address. Motivated by this diagnosis, we introduce $\textbf{V}$ariance $\textbf{C}$ontrolled $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{VCPO}$), a method that (i) dynamically scales the learning rate with ESS to dampen unreliable updates and (ii) applies a closed-form minimum-variance baseline for off-policy settings, without a critic model and adding minimal overhead. Empirically, across math and general reasoning benchmarks, this enables robustly stable asynchronous training compared to previous stabilization and algorithmic methods, even in highly off-policy regimes (128 steps off-policy). In a long-horizon, tool-use task, VCPO matches synchronous performance while delivering a 2.5$\times$ speedup in training time. Code is available at: https://github.com/mit-han-lab/vcpo
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01942v1">Ignore All Previous Instructions: Jailbreaking as a de-escalatory peace building practise to resist LLM social media bots</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted to ICLR 2026 AI for peace workshop
    </div>
    <details class="paper-abstract">
      Large Language Models have intensified the scale and strategic manipulation of political discourse on social media, leading to conflict escalation. The existing literature largely focuses on platform-led moderation as a countermeasure. In this paper, we propose a user-centric view of "jailbreaking" as an emergent, non-violent de-escalation practice. Online users engage with suspected LLM-powered accounts to circumvent large language model safeguards, exposing automated behaviour and disrupting the circulation of misleading narratives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02209v2">StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets?</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate strong potential as autonomous agents, with promising capabilities in reasoning, tool use, and sequential decision-making. While prior benchmarks have evaluated LLM agents in various domains, the financial domain remains underexplored, despite its significant economic value and complex reasoning requirements. Most existing financial benchmarks focus on static question-answering, failing to capture the dynamics of real-market trading. To address this gap, we introduce STOCKBENCH, a contamination-free benchmark designed to evaluate LLM agents in realistic, multi-month stock trading environments. Agents receive daily market signals -- including prices, fundamentals, and news -- and make sequential buy, sell, or hold decisions. Performance is measured using financial metrics such as cumulative return, maximum drawdown, and the Sortino ratio, capturing both profitability and risk management. We evaluate a wide range of state-of-the-art proprietary and open-source LLMs. Surprisingly, most models struggle to outperform the simple buy-and-hold baseline, while some models demonstrate the potential to achieve higher returns and stronger risk management. These findings highlight both the challenges and opportunities of LLM-based trading agents, showing that strong performance on static financial question-answering do not necessarily translate into effective trading behavior. We release STOCKBENCH as an open-source benchmark to enable future research on LLM-driven financial agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01910v1">FLANS at SemEval-2026 Task 7: RAG with Open-Sourced Smaller LLMs for Everyday Knowledge Across Diverse Languages and Cultures</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      This system paper describes our participation in the SemEval-2025 Task-7 ``Everyday Knowledge Across Diverse Languages and Cultures''. We attended two subtasks, i.e., Track 1: Short Answer Questions (SAQ), and Track 2: Multiple-Choice Questions (MCQ). The methods we used are retrieval augmented generation (RAGs) with open-sourced smaller LLMs (OS-sLLMs). To better adapt to this shared task, we created our own culturally aware knowledge base (CulKBs) by extracting Wikipedia content using keyword lists we prepared. We extracted both culturally-aware wiki-text and country-specific wiki-summary. In addition to the local CulKBs, we also have one system integrating live online search output via DuckDuckGo. Towards better privacy and sustainability, we aimed to deploy smaller LLMs (sLLMs) that are open-sourced on the Ollama platform. We share the prompts we developed using refinement techniques and report the learning curve of such prompts. The tested languages are English, Spanish, and Chinese for both tracks. Our resources and codes are shared via https://github.com/aaronlifenghan/FLANS-2026
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01865v1">CyclicJudge: Mitigating Judge Bias Efficiently in LLM-based Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      LLM-as-judge evaluation has become standard practice for open-ended model assessment; however, judges exhibit systematic biases that cannot be eliminated by increasing the number of scenarios or generations. These biases are often similar in magnitude to the model differences that benchmarks are designed to detect, resulting in unreliable rankings when single-judge evaluations are used. This work introduces a variance decomposition that partitions benchmark score variance into scenario, generation, judge, and residual components. Based on this analysis, CyclicJudge, a round-robin assignment of judges, is demonstrated to be the optimal allocation strategy. It eliminates bias precisely while requiring each judge only once per cycle, maintaining the cost of single-judge evaluation. Empirical validation on MT-Bench supports all theoretical predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01834v1">Probing Materials Knowledge in LLMs: From Latent Embeddings to Reliable Predictions</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large language models are increasingly applied to materials science, yet fundamental questions remain about their reliability and knowledge encoding. Evaluating 25 LLMs across four materials science tasks -- over 200 base and fine-tuned configurations -- we find that output modality fundamentally determines model behavior. For symbolic tasks, fine-tuning converges to consistent, verifiable answers with reduced response entropy, while for numerical tasks, fine-tuning improves prediction accuracy but models remain inconsistent across repeated inference runs, limiting their reliability as quantitative predictors. For numerical regression, we find that better performance can be obtained by extracting embeddings directly from intermediate transformer layers than from model text output, revealing an ``LLM head bottleneck,'' though this effect is property- and dataset-dependent. Finally, we present a longitudinal study of GPT model performance in materials science, tracking four models over 18 months and observing 9--43\% performance variation that poses reproducibility challenges for scientific applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.14003v4">Unlearning Isn't Invisible: Detecting Unlearning Traces in LLMs from Model Outputs</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Machine unlearning (MU) for large language models (LLMs), commonly referred to as LLM unlearning, seeks to remove specific undesirable data or knowledge from a trained model, while maintaining its performance on standard tasks. While unlearning plays a vital role in protecting data privacy, enforcing copyright, and mitigating sociotechnical harms in LLMs, we identify a new vulnerability post-unlearning: unlearning trace detection. We discover that unlearning leaves behind persistent "fingerprints" in LLMs, detectable traces in both model behavior and internal representations. These traces can be identified from output responses, even when prompted with forget-irrelevant inputs. Specifically, even a simple supervised classifier can determine whether a model has undergone unlearning, using only its prediction logits or even its textual outputs. Further analysis shows that these traces are embedded in intermediate activations and propagate nonlinearly to the final layer, forming low-dimensional, learnable manifolds in activation space. Through extensive experiments, we demonstrate that unlearning traces can be detected with over 90% accuracy even under forget-irrelevant inputs, and that larger LLMs exhibit stronger detectability. These findings reveal that unlearning leaves measurable signatures, introducing a new risk of reverse-engineering forgotten information when a model is identified as unlearned, given an input query.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24385v4">Vid-LLM: A Compact Video-based 3D Multimodal LLM with Reconstruction-Reasoning Synergy</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Recent developments in Multimodal Large Language Models (MLLMs) have significantly improved Vision-Language (VL) reasoning in 2D domains. However, extending these capabilities to 3D scene understanding remains a major challenge. Existing 3D Multimodal Large Language Models (3D-MLLMs) often depend on 3D data inputs, which limits scalability and generalization. To address this limitation, we propose Vid-LLM, a video-based 3D-MLLM that directly processes video inputs without requiring external 3D data, making it practical for real-world deployment. In our method, the geometric prior are directly used to improve the performance of the sceen perception. To integrate the geometric cues into the MLLM compactly, we design a Cross-Task Adapter (CTA) module to align the 3D geometric priors with the vision-language representations. To ensure geometric consistency and integrity, we introduce a Metric Depth Model that recovers real-scale geometry from the reconstruction outputs. Finally, the model is fine-tuned with a two-stage distillation optimization strategy, realizing fast convergence and stabilizes training. Extensive experiments across diverse benchmarks verified the effectiveness of our method on 3D Question Answering, 3D Dense Captioning and 3D Visual Grounding tasks, demonstrating the superior multi-task capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04369v2">Multi-scale hypergraph meets LLMs: Aligning large language models for time series analysis</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted by ICLR2026
    </div>
    <details class="paper-abstract">
      Recently, there has been great success in leveraging pre-trained large language models (LLMs) for time series analysis. The core idea lies in effectively aligning the modality between natural language and time series. However, the multi-scale structures of natural language and time series have not been fully considered, resulting in insufficient utilization of LLMs capabilities. To this end, we propose MSH-LLM, a Multi-Scale Hypergraph method that aligns Large Language Models for time series analysis. Specifically, a hyperedging mechanism is designed to enhance the multi-scale semantic information of time series semantic space. Then, a cross-modality alignment (CMA) module is introduced to align the modality between natural language and time series at different scales. In addition, a mixture of prompts (MoP) mechanism is introduced to provide contextual information and enhance the ability of LLMs to understand the multi-scale temporal patterns of time series. Experimental results on 27 real-world datasets across 5 different applications demonstrate that MSH-LLM achieves the state-of-the-art results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01792v1">ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted at The 40th Annual AAAI Conference on Artificial Intelligence (AAAI 2026)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced to encompass extensive knowledge across diverse domains. Yet controlling what a LLMs should not know is important for ensuring alignment and thus safe use. However, effective unlearning in LLMs is difficult due to the fuzzy boundary between knowledge retention and forgetting. This challenge is exacerbated by entangled parameter spaces from continuous multi-domain training, often resulting in collateral damage, especially under aggressive unlearning strategies. Furthermore, the computational overhead required to optimize State-of-the-Art (SOTA) models with billions of parameters poses an additional barrier. In this work, we present ALTER, a lightweight unlearning framework for LLMs to address both the challenges of knowledge entanglement and unlearning efficiency. ALTER operates through two phases: (I) high entropy tokens are captured and learned via the shared A matrix in LoRA, followed by (II) an asymmetric LoRA architecture that achieves a specified forgetting objective by parameter isolation and unlearning tokens within the target subdomains. Serving as a new research direction for achieving unlearning via token-level isolation in the asymmetric framework. ALTER achieves SOTA performance on TOFU, WMDP, and MUSE benchmarks with over 95% forget quality and shows minimal side effects through preserving foundational tokens. By decoupling unlearning from LLMs' billion-scale parameters, this framework delivers excellent efficiency while preserving over 90% of model utility, exceeding baseline preservation rates of 47.8-83.6%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01789v1">Can LLMs Hack Enterprise Networks? -- Replicated Computational Results (RCR) Report</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      This is the Replicated Computational Results (RCR) Report for the paper ``Can LLMs Hack Enterprise Networks?" The paper empirically investigates the efficacy and effectiveness of different LLMs for penetration-testing enterprise networks, i.e., Microsoft Active Directory Assumed-Breach Simulations. This RCR report describes the artifacts used in the paper, how to create an evaluation setup, and highlights the analysis scripts provided within our prototype.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01778v1">LLM-as-an-Annotator: Training Lightweight Models with LLM-Annotated Examples for Aspect Sentiment Tuple Prediction</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted for publication at LREC 2026. Final version will appear in the ACL Anthology
    </div>
    <details class="paper-abstract">
      Training models for Aspect-Based Sentiment Analysis (ABSA) tasks requires manually annotated data, which is expensive and time-consuming to obtain. This paper introduces LA-ABSA, a novel approach that leverages Large Language Model (LLM)-generated annotations to fine-tune lightweight models for complex ABSA tasks. We evaluate our approach on five datasets for Target Aspect Sentiment Detection (TASD) and Aspect Sentiment Quad Prediction (ASQP). Our approach outperformed previously reported augmentation strategies and achieved competitive performance with LLM-prompting in low-resource scenarios, while providing substantial energy efficiency benefits. For example, using 50 annotated examples for in-context learning (ICL) to guide the annotation of unlabeled data, LA-ABSA achieved an F1 score of 49.85 for ASQP on the SemEval Rest16 dataset, closely matching the performance of ICL prompting with Gemma-3-27B (51.10), while requiring significantly lower computational resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24282v3">SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted at ICLR 2026 (Oral)
    </div>
    <details class="paper-abstract">
      We introduce $\textbf{SimuHome}$, a high-fidelity smart home simulator and a benchmark of 600 episodes for LLM-based smart home agents. Existing smart home benchmarks treat the home as a static system, neither simulating how device operations affect environmental variables over time nor supporting workflow scheduling of device commands. SimuHome is grounded in the Matter protocol, the industry standard that defines how real smart home devices communicate and operate. Agents interact with devices through SimuHome's APIs and observe how their actions continuously affect environmental variables such as temperature and humidity. Our benchmark covers state inquiry, implicit user intent inference, explicit device control, and workflow scheduling, each with both feasible and infeasible requests. For workflow scheduling, the simulator accelerates time so that scheduled workflows can be evaluated immediately. An evaluation of 18 agents reveals that workflow scheduling is the hardest category, with failures persisting across alternative agent frameworks and fine-tuning. These findings suggest that SimuHome's time-accelerated simulation could serve as an environment for agents to pre-validate their actions before committing them to the real world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01712v1">FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 24 pages, 6 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models for vertical domains remains a labor-intensive and expensive process, requiring domain experts to curate data, configure training, and iteratively diagnose model behavior. Despite growing interest in autonomous machine learning, no prior work has tackled end-to-end LLM fine-tuning with agents. Can LLM-based agents automate this complete process? We frame this as a substantially open problem: agents must navigate an open-ended search space spanning data curation from diverse data sources, processing with complex tools, building a training pipeline, and iteratively refining their approach based on evaluation outcomes in rapidly growing logs--an overall scenario far more intricate than existing benchmarks. To study this question, we introduce FT-Dojo, an interactive environment comprising 13 tasks across 5 domains. We further develop FT-Agent, an autonomous system that mirrors human experts by leveraging evaluation-driven feedback to iteratively diagnose failures and refine fine-tuning strategies. Experiments on FT-Dojo demonstrate that purpose-built fine-tuning agents significantly outperform general-purpose alternatives, with FT-Agent achieving the best performance on 10 out of 13 tasks across all five domains. Ablations show that the approach generalizes effectively to 3B models, with additional insights on data scaling trade-offs and backbone sensitivity. Case analyses reveal that agents can recover from failures through cumulative learning from historical experience, while also exposing fundamental limitations in causal reasoning--highlighting both the promise and current boundaries of autonomous LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.18553v3">The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Published as a conference paper at the Fourteenth International Conference on Learning Representations (ICLR 2026): https://openreview.net/forum?id=NFB4QGGS65
    </div>
    <details class="paper-abstract">
      Quantizing the weights of large language models (LLMs) from 16-bit to lower bitwidth is the de facto approach to deploy massive transformers onto more affordable accelerators. While GPTQ emerged as one of the standard methods for one-shot post-training quantization at LLM scale, its inner workings are described as a sequence of algebraic updates that obscure geometric meaning or worst-case guarantees. In this work, we show that, when executed back-to-front (from the last to first dimension) for a linear layer, GPTQ is mathematically identical to Babai's nearest plane algorithm for the classical closest vector problem (CVP) on a lattice defined by the Hessian matrix of the layer's inputs. This equivalence is based on a sophisticated mathematical argument, and has two analytical consequences: first, the GPTQ error propagation step gains an intuitive geometric interpretation; second, GPTQ inherits the error upper bound of Babai's algorithm under the assumption that no weights are clipped. Leveraging this bound, we design post-training quantization methods that avoid clipping, and outperform the original GPTQ. In addition, we provide efficient GPU inference kernels for the resulting representation. Taken together, these results place GPTQ on a firm theoretical footing and open the door to importing decades of progress in lattice algorithms towards the design of future quantization algorithms for billion-parameter models. Source code is available at https://github.com/IST-DASLab/GPTQ-Babai.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.07638v2">Data Selection for LLM Alignment Using Fine-Grained Preferences</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) alignment aims to ensure that the behavior of LLMs meets human preferences. While collecting data from multiple fine-grained, aspect-specific preferences becomes more and more feasible, existing alignment methods typically work on a single preference and thus struggle with conflicts inherent in such aggregated datasets. As one early attempt, in this paper, we propose a data-centric approach to align LLMs through the effective use of fine-grained preferences. Specifically, we formulate the problem as a direct fine-grained preference optimization and introduce preference divergence (PD) that quantifies inter-aspect preference conflicts. Instead of directly tackling the consequent complicated optimization, we recast it as a data selection problem and propose a simple yet effective strategy, which identifies a subset of data corresponding to the most negative PD values, for efficient training. We theoretically analyze the loss-bound optimality of our selection strategy and conduct extensive empirical studies on varied settings and datasets to demonstrate that our practical selection method could achieve consistent improvement against standard full-data alignment, using even just 30% of the data. Our work shares a line that LLM alignment using fine-grained preferences is highly feasible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15030v4">Collab-REC: An LLM-based Agentic Framework for Balancing Recommendations in Tourism</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      We propose Collab-REC, a multi-agent framework designed to counteract popularity bias and enhance diversity in tourism recommendations. In our setting, three LLM-based agents: Personalization, Popularity, and Sustainability, generate city suggestions from complementary perspectives. A non-LLM moderator then merges and refines these proposals via multi-round negotiation, ensuring each agent's viewpoint is incorporated while penalizing spurious or repeated responses. Extensive experiments on European city queries using LLMs from different sizes and model families demonstrate that Collab-REC enhances diversity and overall relevance compared to a single-agent baseline, surfacing lesser-visited locales that are often overlooked. This balanced, context-aware approach addresses over-tourism and better aligns with user-provided constraints, highlighting the promise of multi-stakeholder collaboration in LLM-driven recommender systems. Code, data, and other artifacts are available here: https://github.com/ashmibanerjee/collab-rec, while the prompts used are included in the appendix.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.13648v2">SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly tested for a "Theory of Mind" (ToM) - the ability to attribute mental states to oneself and others. Yet most evaluations stop at explicit belief attribution in classical toy stories or stylized tasks, leaving open the questions of whether LLMs can implicitly apply such knowledge to predict human behavior, or to judge an observed behavior, in diverse scenarios. We introduce SimpleToM, a benchmark that advances ToM evaluation along two novel axes. First, it probes multiple levels of ToM reasoning, from mental state inference (explicit ToM) to behavior prediction and judgment (applied ToM). Second, it situates these tasks in diverse, everyday scenarios - such as supermarkets, hospitals, schools, and offices - where information asymmetries naturally arise (e.g., hidden defects in grocery store items, incomplete information in provider-patient interactions, or restricted access to locked devices). SimpleToM contains concise stories (e.g., "The can of Pringles has moldy chips in it. Mary picks up the can in the supermarket and walks to the cashier."), each with three questions that test different degrees of ToM reasoning, asking models to predict: (a) mental states ("Is Mary aware of the mold?"), (b) behaviors ("Will Mary pay for the chips or report the mold?"), and (c) judgments ("Mary paid for the chips. Was that reasonable?"). Experiments reveal a striking gap: state-of-the-art models often reliably infer mental state (a), but fail at applying knowledge about the mental state for secondary predictions, with performance dropping sharply for behavior prediction (b) and further for behavior judgment (c). This exposes a critical fragility in LLMs' social reasoning in terms of what they know (explicit ToM) versus how well they can implicitly apply that knowledge for predictions (applied ToM).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01638v1">Who Explains Privacy Policies to Me? Embodied and Textual LLM-Powered Privacy Assistants in Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 11 pages, 1 figure, 1 table
    </div>
    <details class="paper-abstract">
      Virtual Reality (VR) systems collect fine-grained behavioral and biometric data, yet privacy policies are rarely read or understood due to their complex language, length, and poor integration into users' interaction workflows. To lower the barrier to informed consent at the point of choice, we explore a Large Language Model (LLM)-powered privacy assistant embedded into a VR app store to support privacy-aware app selection. The assistant is realized in two interaction modes: a text-based chat interface and an embodied virtual avatar providing spoken explanations. We report on an exploratory within-subjects study $(N = 21)$ in which participants browsed VR productivity applications under unassisted and assisted conditions. Our findings suggest that both interaction modes support more deliberate engagement with privacy information and decision-making, with privacy scores primarily functioning as a veto mechanism rather than a primary selection driver. The impact of embodied interaction varied between participants, while textual interaction supported reflective review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10729v2">OrbitFlow: SLO-Aware Long-Context LLM Serving with Fine-Grained KV Cache Reconfiguration</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Accepted at the 52nd International Conference on Very Large Data Bases (VLDB 2026). Xinyue Ma and Heelim Hong contributed equally (co-first authors)
    </div>
    <details class="paper-abstract">
      Serving long-context LLMs is challenging because request lengths and batch composition vary during token generation, causing the memory footprint to fluctuate significantly at runtime. Offloading KV caches to host memory limits effective memory usage, but existing static and predetermined offloading strategies cannot adapt to the rapidly shifting memory demands of long-context serving. This often leads to excessive CPU-to-GPU KV transfers that translate into latency spikes and frequent SLO violations. To address these challenges, we introduce OrbitFlow, a fine-grained and adaptive KV cache management system that meets latency SLOs in long-context LLM serving. OrbitFlow employs a lightweight ILP solver to decide which layers' KV caches to retain on the GPU for each request, within memory capacity constraints. It continuously refines KV placements based on runtime feedback when the active plan becomes suboptimal during token generation. Under heavy load, OrbitFlow invokes a fallback mechanism to temporarily defer in-flight requests with large memory footprints, preserving overall SLO attainment. Our experiments demonstrate that OrbitFlow improves SLO attainment for TPOT and TBT by up to 66% and 48%, respectively, while reducing the 95th percentile latency by 38% and achieving up to 3.3x higher throughput compared to existing offloading methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.18991v6">Inverse Reinforcement Learning with Dynamic Reward Scaling for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Alignment is vital for safely deploying large language models (LLMs). Existing techniques are either reward-based (training a reward model on preference pairs and optimizing with reinforcement learning) or reward-free (directly fine-tuning on ranked outputs). Recent research shows that well-tuned reward-based pipelines remain the most robust, and single-response demonstrations can outperform pairwise preference data. However, there still exist two key challenges: (1) imbalanced safety datasets that overrepresent common hazards while neglecting long-tail threats; and (2) static reward models that ignore task difficulty, limiting optimization efficiency and attainable gains. To address these limitations, we propose DR-IRL, which Dynamically adjusts Rewards through Inverse Reinforcement Learning. We first train category-specific reward models using a balanced safety dataset of seven harmful categories as demonstration via IRL. Then we enhance Group Relative Policy Optimization (GRPO) by introducing dynamic reward scaling: adjusting rewards by task difficulty, data-level hardness by text encoder cosine similarity, and model-level responsiveness by reward gaps. Extensive experiments across various benchmarks and LLMs demonstrate that DR-IRL outperforms all baseline methods in safety alignment while maintaining usefulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01596v1">MigMate: A VS Code Extension for LLM-based Library Migration of Python Projects</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 6 pages, 6 figures, 2 tables, 3rd International Workshop on Integrated Development Environments (IDE 2026)
    </div>
    <details class="paper-abstract">
      Modern software relies heavily on third-party software libraries to streamline the development process. The act of switching one library for a similar counterpart, called library migration, naturally occurs as libraries become outdated or unsuitable for the project. Manually migrating from one library to another is a time-consuming task. Our previous research developed MigrateLib, a command-line LLM-based migration tool that can automate the complete migration process. In this paper, we present our open-source VS Code IDE plugin, MigMate, that builds on MigrateLib by integrating the automated migration process into the developer's existing development environment. MigMate provides an interactive experience, allowing developers to view and confirm changes before they are applied. A preliminary user study shows that plugin usage consistently reduces the time taken to complete a library migration task, and it scores highly on the System Usability Scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01574v1">DualSentinel: A Lightweight Framework for Detecting Targeted Attacks in Black-box LLM via Dual Entropy Lull Pattern</a></div>
    <div class="paper-meta">
      📅 2026-03-02
    </div>
    <details class="paper-abstract">
      Recent intelligent systems integrate powerful Large Language Models (LLMs) through APIs, but their trustworthiness may be critically undermined by targeted attacks like backdoor and prompt injection attacks, which secretly force LLMs to generate specific malicious sequences. Existing defensive approaches for such threats typically rely on high access rights, impose prohibitive costs, and hinder normal inference, rendering them impractical for real-world scenarios. To solve these limitations, we introduce DualSentinel, a lightweight and unified defense framework that can accurately and promptly detect the activation of targeted attacks alongside the LLM generation process. We first identify a characteristic of compromised LLMs, termed Entropy Lull: when a targeted attack successfully hijacks the generation process, the LLM exhibits a distinct period of abnormally low and stable token probability entropy, indicating it is following a fixed path rather than making creative choices. DualSentinel leverages this pattern by developing an innovative dual-check approach. It first employs a magnitude and trend-aware monitoring method to proactively and sensitively flag an entropy lull pattern at runtime. Upon such flagging, it triggers a lightweight yet powerful secondary verification based on task-flipping. An attack is confirmed only if the entropy lull pattern persists across both the original and the flipped task, proving that the LLM's output is coercively controlled. Extensive evaluations show that DualSentinel is both highly effective (superior detection accuracy with near-zero false positives) and remarkably efficient (negligible additional cost), offering a truly practical path toward securing deployed LLMs. The source code can be accessed at https://doi.org/10.5281/zenodo.18479273.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25175v2">EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering</a></div>
    <div class="paper-meta">
      📅 2026-03-02
      | 💬 Functionality upgrade. Code: https://github.com/ZJU-REAL/EasySteer Demo: https://www.youtube.com/watch?v=3rRGzZmhrXg
    </div>
    <details class="paper-abstract">
      Large language model (LLM) steering has emerged as a promising paradigm for controlling model behavior at inference time through targeted manipulation of hidden states, offering a lightweight alternative to expensive retraining. However, existing steering frameworks suffer from critical limitations: computational inefficiency, limited extensibility, and restricted functionality that hinder both research progress and practical deployment. We present EasySteer, a unified framework for high-performance, extensible LLM steering built on vLLM. Our system features modular architecture with pluggable interfaces for both analysis-based and learning-based methods, fine-grained parameter control, pre-computed steering vectors for eight application domains, and an interactive demonstration system. Through deep integration with vLLM's optimized inference engine, EasySteer achieves 10.8-22.3$\times$ speedup over existing frameworks. Extensive experiments demonstrate its effectiveness in overthinking mitigation, hallucination reduction, and other key applications. EasySteer transforms steering from research technique to production-ready capability, establishing critical infrastructure for deployable, controllable language models.
    </details>
</div>
