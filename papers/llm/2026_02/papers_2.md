# llm - 2026_02

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.10054v3">Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Accepted by ICLR 2026. Code & models: https://github.com/pspdada/Uni-DPO
    </div>
    <details class="paper-abstract">
      Direct Preference Optimization (DPO) has emerged as a cornerstone of reinforcement learning from human feedback (RLHF) due to its simplicity and efficiency. However, existing DPO-based methods typically treat all preference pairs equally, overlooking substantial variations in data quality and learning difficulty, which leads to inefficient data utilization and suboptimal performance. To address this limitation, we propose Uni-DPO, a unified dynamic preference optimization framework that jointly considers (a) the inherent quality of preference pairs and (b) the model's evolving performance during training. By adaptively reweighting samples based on both factors, Uni-DPO enables more effective use of preference data and achieves superior performance. Extensive experiments across models and benchmarks demonstrate the effectiveness and generalization of Uni-DPO. On textual tasks, Gemma-2-9B-IT fine-tuned with Uni-DPO surpasses the leading LLM, Claude 3 Opus, by 6.7 points on Arena-Hard. On mathematical and multimodal tasks, Uni-DPO consistently outperforms baseline methods across all benchmarks, providing strong empirical evidence of its effectiveness and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07695v2">EventCast: Hybrid Demand Forecasting in E-Commerce with LLM-Based Event Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Demand forecasting is a cornerstone of e-commerce operations, directly impacting inventory planning and fulfillment scheduling. However, existing forecasting systems often fail during high-impact periods such as flash sales, holiday campaigns, and sudden policy interventions, where demand patterns shift abruptly and unpredictably. In this paper, we introduce EventCast, a modular forecasting framework that integrates future event knowledge into time-series prediction. Unlike prior approaches that ignore future interventions or directly use large language models (LLMs) for numerical forecasting, EventCast leverages LLMs solely for event-driven reasoning. Unstructured business data, which covers campaigns, holiday schedules, and seller incentives, from existing operational databases, is processed by an LLM that converts it into interpretable textual summaries leveraging world knowledge for cultural nuances and novel event combinations. These summaries are fused with historical demand features within a dual-tower architecture, enabling accurate, explainable, and scalable forecasts. Deployed on real-world e-commerce scenarios spanning 4 countries of 160 regions over 10 months, EventCast achieves up to 86.9% and 97.7% improvement on MAE and MSE compared to the variant without event knowledge, and reduces MAE by up to 57.0% and MSE by 83.3% versus the best industrial baseline during event-driven periods. EventCast has deployed into real-world industrial pipelines since March 2025, offering a practical solution for improving operational decision-making in dynamic e-commerce environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.21963v2">Industrialized Deception: The Collateral Effects of LLM-Generated Misinformation on Digital Ecosystems</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Accepted at ACM TheWebConf '26 Companion
    </div>
    <details class="paper-abstract">
      Generative AI and misinformation research has evolved since our 2024 survey. This paper presents an updated perspective, transitioning from literature review to practical countermeasures. We report on changes in the threat landscape, including improved AI-generated content through Large Language Models (LLMs) and multimodal systems. Central to this work are our practical contributions: JudgeGPT, a platform for evaluating human perception of AI-generated news, and RogueGPT, a controlled stimulus generation engine for research. Together, these tools form an experimental pipeline for studying how humans perceive and detect AI-generated misinformation. Our findings show that detection capabilities have improved, but the competition between generation and detection continues. We discuss mitigation strategies including LLM-based detection, inoculation approaches, and the dual-use nature of generative AI. This work contributes to research addressing the adverse impacts of AI on information quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10891v1">Interactive LLM-assisted Curriculum Learning for Multi-Task Evolutionary Policy Search</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 8 pages, 7 figures, with Appendix
    </div>
    <details class="paper-abstract">
      Multi-task policy search is a challenging problem because policies are required to generalize beyond training cases. Curriculum learning has proven to be effective in this setting, as it introduces complexity progressively. However, designing effective curricula is labor-intensive and requires extensive domain expertise. LLM-based curriculum generation has only recently emerged as a potential solution, but was limited to operate in static, offline modes without leveraging real-time feedback from the optimizer. Here we propose an interactive LLM-assisted framework for online curriculum generation, where the LLM adaptively designs training cases based on real-time feedback from the evolutionary optimization process. We investigate how different feedback modalities, ranging from numeric metrics alone to combinations with plots and behavior visualizations, influence the LLM ability to generate meaningful curricula. Through a 2D robot navigation case study, tackled with genetic programming as optimizer, we evaluate our approach against static LLM-generated curricula and expert-designed baselines. We show that interactive curriculum generation outperforms static approaches, with multimodal feedback incorporating both progression plots and behavior visualizations yielding performance competitive with expert-designed curricula. This work contributes to understanding how LLMs can serve as interactive curriculum designers for embodied AI systems, with potential extensions to broader evolutionary robotics applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10881v1">Diagnosing Structural Failures in LLM-Based Evidence Extraction for Meta-Analysis</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Accepted at the 22nd Conference on Information and Research Science Connecting to Digital and Library Science (IRCDL 2026)
    </div>
    <details class="paper-abstract">
      Systematic reviews and meta-analyses rely on converting narrative articles into structured, numerically grounded study records. Despite rapid advances in large language models (LLMs), it remains unclear whether they can meet the structural requirements of this process, which hinge on preserving roles, methods, and effect-size attribution across documents rather than on recognizing isolated entities. We propose a structural, diagnostic framework that evaluates LLM-based evidence extraction as a progression of schema-constrained queries with increasing relational and numerical complexity, enabling precise identification of failure points beyond atom-level extraction. Using a manually curated corpus spanning five scientific domains, together with a unified query suite and evaluation protocol, we evaluate two state-of-the-art LLMs under both per-document and long-context, multi-document input regimes. Across domains and models, performance remains moderate for single-property queries but degrades sharply once tasks require stable binding between variables, roles, statistical methods, and effect sizes. Full meta-analytic association tuples are extracted with near-zero reliability, and long-context inputs further exacerbate these failures. Downstream aggregation amplifies even minor upstream errors, rendering corpus-level statistics unreliable. Our analysis shows that these limitations stem not from entity recognition errors, but from systematic structural breakdowns, including role reversals, cross-analysis binding drift, instance compression in dense result sections, and numeric misattribution, indicating that current LLMs lack the structural fidelity, relational binding, and numerical grounding required for automated meta-analysis. The code and data are publicly available at GitHub (https://github.com/zhiyintan/LLM-Meta-Analysis).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07954v2">Bielik Guard: Efficient Polish Language Safety Classifiers for LLM Content Moderation</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly deployed in Polish language applications, the need for efficient and accurate content safety classifiers has become paramount. We present Bielik Guard, a family of compact Polish language safety classifiers comprising two model variants: a 0.1B parameter model based on MMLW-RoBERTa-base and a 0.5B parameter model based on PKOBP/polish-roberta-8k. Fine-tuned on a community-annotated dataset of 6,885 Polish texts, these models classify content across five safety categories: Hate/Aggression, Vulgarities, Sexual Content, Crime, and Self-Harm. Our evaluation demonstrates that both models achieve strong performance on multiple benchmarks. The 0.5B variant offers the best overall discrimination capability with F1 scores of 0.791 (micro) and 0.785 (macro) on the test set, while the 0.1B variant demonstrates exceptional efficiency. Notably, Bielik Guard 0.1B v1.1 achieves superior precision (77.65%) and very low false positive rate (0.63%) on real user prompts, outperforming HerBERT-PL-Guard (31.55% precision, 4.70% FPR) despite identical model size. The models are publicly available and designed to provide appropriate responses rather than simple content blocking, particularly for sensitive categories like self-harm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11224v1">Agent-Diff: Benchmarking LLM Agents on Enterprise API Tasks via Code Execution with State-Diff-Based Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Pre-Print. Under review for KDD 2026
    </div>
    <details class="paper-abstract">
      We present Agent-Diff, a novel benchmarking framework for evaluating agentic Large Language Models (LLMs) on real-world tasks that execute code via external APIs. Agentic LLM performance varies due to differences in models, external tool access, prompt structures, and agentic frameworks. Benchmarks must make fundamental trade-offs between a sandboxed approach that controls for variation in software environments and more ecologically valid approaches employing real services. Agent-Diff attempts to capture the desirable features of both of these approaches by including access to the real API interfaces for software services while sandboxing the environment in which calls are made, processed, and evaluated. This approach relies on two key innovations. The first is a novel state-diff contract, which separates process from outcome - rather than fuzzy trace or parameter matching, we define task success as whether the expected change in environment state was achieved. The second is a novel sandbox that provides a standardized scripting layer that all models use to execute code against external APIs (Slack, Box, Linear, Google Calendar). Thus, we can evaluate different agentic LLMs against a standardized set of contracts using a unified sandbox while still evaluating their performance on real-world service interfaces. Using the Agent-Diff framework, we provide benchmarks for nine LLMs across 224 tasks utilizing enterprise software workflows. In addition, we evaluate the robustness of the framework with ablation experiments to assess the contribution of access to API documentation on benchmark performance. Code and data: https://github.com/agent-diff-bench/agent-diff.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10833v1">Training-Induced Bias Toward LLM-Generated Content in Dense Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Accepted at ECIR 2026
    </div>
    <details class="paper-abstract">
      Dense retrieval is a promising approach for acquiring relevant context or world knowledge in open-domain natural language processing tasks and is now widely used in information retrieval applications. However, recent reports claim a broad preference for text generated by large language models (LLMs). This bias is called "source bias", and it has been hypothesized that lower perplexity contributes to this effect. In this study, we revisit this claim by conducting a controlled evaluation to trace the emergence of such preferences across training stages and data sources. Using parallel human- and LLM-generated counterparts of the SciFact and Natural Questions (NQ320K) datasets, we compare unsupervised checkpoints with models fine-tuned using in-domain human text, in-domain LLM-generated text, and MS MARCO. Our results show the following: 1) Unsupervised retrievers do not exhibit a uniform pro-LLM preference. The direction and magnitude depend on the dataset. 2) Across the settings tested, supervised fine-tuning on MS MARCO consistently shifts the rankings toward LLM-generated text. 3) In-domain fine-tuning produces dataset-specific and inconsistent shifts in preference. 4) Fine-tuning on LLM-generated corpora induces a pronounced pro-LLM bias. Finally, a retriever-centric perplexity probe involving the reattachment of a language modeling head to the fine-tuned dense retriever encoder indicates agreement with relevance near chance, thereby weakening the explanatory power of perplexity. Our study demonstrates that source bias is a training-induced phenomenon rather than an inherent property of dense retrievers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24384v2">HarmMetric Eval: Benchmarking Metrics and Judges for LLM Harmfulness Assessment</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      The potential for large language models (LLMs) to generate harmful content poses a significant safety risk in their deployment. To address and assess this risk, the community has developed numerous harmfulness evaluation metrics and judges. However, the lack of a systematic benchmark for evaluating these metrics and judges undermines the credibility and consistency of LLM safety assessments. To bridge this gap, we introduce HarmMetric Eval, a comprehensive benchmark designed to support both overall and fine-grained evaluation of harmfulness metrics and judges. In HarmMetric Eval, we build a high-quality dataset of representative harmful prompts paired with highly diverse harmful model responses and non-harmful counterparts across multiple categories. We also propose a flexible scoring mechanism that rewards the metrics for correctly ranking harmful responses above non-harmful ones, which is applicable to almost all existing metrics and judges with varying output formats and scoring scales. Using HarmMetric Eval, we uncover a surprising finding by extensive experiments: Conventional reference-based metrics such as ROUGE and METEOR can outperform existing LLM-based judges in fine-grained harmfulness evaluation, challenging prevailing assumptions about LLMs'superiority in this domain. To reveal the reasons behind this finding, we provide a fine-grained analysis to explain the limitations of LLM-based judges on rating irrelevant or useless responses. Furthermore, we build a new harmfulness judge by incorporating the fine-grained criteria into its prompt template and leverage reference-based metrics to fine-tune its base LLM. The resulting judge demonstrates superior performance than all existing metrics and judges in evaluating harmful responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10808v1">PELLI: Framework to effectively integrate LLMs for quality software generation</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Recent studies have revealed that when LLMs are appropriately prompted and configured, they demonstrate mixed results. Such results often meet or exceed the baseline performance. However, these comparisons have two primary issues. First, they mostly considered only reliability as a comparison metric and selected a few LLMs (such as Codex and ChatGPT) for comparision. This paper proposes a comprehensive code quality assessment framework called Programmatic Excellence via LLM Iteration (PELLI). PELLI is an iterative analysis-based process that upholds high-quality code changes. We extended the state-of-the-art by performing a comprehensive evaluation that generates quantitative metrics for analyzing three primary nonfunctional requirements (such as maintainability, performance, and reliability) while selecting five popular LLMs. For PELLI's applicability, we selected three application domains while following Python coding standards. Following this framework, practitioners can ensure harmonious integration between LLMs and human developers, ensuring that their potential is fully realized. PELLI can serve as a practical guide for developers aiming to leverage LLMs while adhering to recognized quality standards. This study's outcomes are crucial for advancing LLM technologies in real-world applications, providing stakeholders with a clear understanding of where these LLMs excel and where they require further refinement. Overall, based on three nonfunctional requirements, we have found that GPT-4T and Gemini performed slightly better. We also found that prompt design can influence the overall code quality. In addition, each application domain demonstrated high and low scores across various metrics, and even within the same metrics across different prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10801v1">Deep Learning-based Method for Expressing Knowledge Boundary of Black-Box LLM</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success, however, the emergence of content generation distortion (hallucination) limits their practical applications. The core cause of hallucination lies in LLMs' lack of awareness regarding their stored internal knowledge, preventing them from expressing their knowledge state on questions beyond their internal knowledge boundaries, as humans do. However, existing research on knowledge boundary expression primarily focuses on white-box LLMs, leaving methods suitable for black-box LLMs which offer only API access without revealing internal parameters-largely unexplored. Against this backdrop, this paper proposes LSCL (LLM-Supervised Confidence Learning), a deep learning-based method for expressing the knowledge boundaries of black-box LLMs. Based on the knowledge distillation framework, this method designs a deep learning model. Taking the input question, output answer, and token probability from a black-box LLM as inputs, it constructs a mapping between the inputs and the model' internal knowledge state, enabling the quantification and expression of the black-box LLM' knowledge boundaries. Experiments conducted on diverse public datasets and with multiple prominent black-box LLMs demonstrate that LSCL effectively assists black-box LLMs in accurately expressing their knowledge boundaries. It significantly outperforms existing baseline models on metrics such as accuracy and recall rate. Furthermore, considering scenarios where some black-box LLMs do not support access to token probability, an adaptive alternative method is proposed. The performance of this alternative approach is close to that of LSCL and surpasses baseline models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10778v1">GoodVibe: Security-by-Vibe for LLM-Based Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for code generation in fast, informal development workflows, often referred to as vibe coding, where speed and convenience are prioritized, and security requirements are rarely made explicit. In this setting, models frequently produce functionally correct but insecure code, creating a growing security risk. Existing approaches to improving code security rely on full-parameter fine-tuning or parameter-efficient adaptations, which are either costly and prone to catastrophic forgetting or operate at coarse granularity with limited interpretability and control. We present GoodVibe, a neuron-level framework for improving the security of code language models by default. GoodVibe is based on the key insight that security-relevant reasoning is localized to a small subset of neurons. We identify these neurons using gradient-based attribution from a supervised security task and perform neuron-selective fine-tuning that updates only this security-critical subspace. To further reduce training cost, we introduce activation-driven neuron clustering, enabling structured updates with minimal overhead. We evaluate GoodVibe on six LLMs across security-critical programming languages, including C++, Java, Swift, and Go. GoodVibe substantially improves the security of generated code while preserving general model utility, achieving up to a 2.5x improvement over base models, matching or exceeding full fine-tuning with over 4,700x fewer trainable parameters, and reducing training computation by more than 3.6x compared to the parameter-efficient baseline (LoRA). Our results demonstrate that neuron-level optimization offers an effective and scalable approach to securing code generation without sacrificing efficiency or generality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01330v2">Beyond Gemini-3-Pro: Revisiting LLM Routing and Aggregation at Scale</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have rapidly advanced, with Gemini-3-Pro setting a new performance milestone. In this work, we explore collective intelligence as an alternative to monolithic scaling, and demonstrate that open-source LLMs' collaboration can surpass Gemini-3-Pro. We first revisit LLM routing and aggregation at scale and identify three key bottlenecks: (1) current train-free routers are limited by a query-based paradigm focusing solely on textual similarity; (2) recent aggregation methods remain largely static, failing to select appropriate aggregators for different tasks;(3) the complementarity of routing and aggregation remains underutilized. To address these problems, we introduce JiSi, a novel framework designed to release the full potential of LLMs' collaboration through three innovations: (1) Query-Response Mixed Routing capturing both semantic information and problem difficulty; (2) Support-Set-based Aggregator Selection jointly evaluating the aggregation and domain capacity of aggregators; (3) Adaptive Routing-Aggregation Switch dynamically leveraging the advantages of routing and aggregation. Comprehensive experiments on nine benchmarks demonstrate that JiSi can surpass Gemini-3-Pro with only 47% costs by orchestrating ten open-source LLMs, while outperforming mainstream baselines. It suggests that collective intelligence represents a novel path towards Artificial General Intelligence (AGI).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10729v1">BOute: Cost-Efficient LLM Serving with Heterogeneous LLMs and GPUs via Multi-Objective Bayesian Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 MLSys 2026
    </div>
    <details class="paper-abstract">
      The rapid growth of large language model (LLM) deployments has made cost-efficient serving systems essential. Recent efforts to enhance system cost-efficiency adopt two main perspectives: (i) An algorithmic perspective that exploits heterogeneous model capabilities to route simpler queries to lower-cost models and complex queries to higher-cost models (i.e., heterogeneous query routing); and (ii) a systems perspective that utilizes heterogeneous GPU resources as cost-effective alternatives to homogeneous high-end GPUs (i.e., heterogeneous model deployment). However, algorithm-system co-design for cost-efficient LLM serving necessitates sophisticated management: (i) Determining optimal query routing strategies under latency and quality requirements, (ii) configuring model deployment across heterogeneous GPUs with appropriate resource allocation and parallelism strategies, and (iii) co-optimizing routing and deployment decisions to maximize overall system performance. To address these challenges, we present BOute, a quality-aware scheduling system that jointly exploits heterogeneous model and GPU capabilities for cost-efficient LLM serving. BOute employs a multi-objective Bayesian optimization (MOBO) framework to co-optimize the routing strategy and model deployment, thereby maximizing the cost-efficiency of the serving system while guaranteeing response quality. Evaluation results demonstrate that BOute outperforms state-of-the-art LLM serving systems by up to 157% and 59% on average under identical cost budgets and quality requirements, or reducing serving costs by 15%-61% (38% on average) while maintaining the same performance targets, validating its effectiveness in achieving cost-efficient LLM serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10716v1">RE-LLM: Refining Empathetic Speech-LLM Responses by Integrating Emotion Nuance</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 5 pages, 1 figure, 2 tables. Accepted at IEEE ASRU 2025
    </div>
    <details class="paper-abstract">
      With generative AI advancing, empathy in human-AI interaction is essential. While prior work focuses on emotional reflection, emotional exploration, key to deeper engagement, remains overlooked. Existing LLMs rely on text which captures limited emotion nuances. To address this, we propose RE-LLM, a speech-LLM integrating dimensional emotion embeddings and auxiliary learning. Experiments show statistically significant gains in empathy metrics across three datasets. RE-LLM relatively improves the Emotional Reaction score by 14.79% and 6.76% compared to text-only and speech-LLM baselines on ESD. Notably, it raises the Exploration score by 35.42% and 3.91% on IEMOCAP, 139.28% and 9.83% on ESD, and 60.95% and 22.64% on MSP-PODCAST. It also boosts unweighted accuracy by 5.4% on IEMOCAP, 2.3% on ESD, and 6.9% on MSP-PODCAST in speech emotion recognition. These results highlight the enriched emotional understanding and improved empathetic response generation of RE-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10715v1">Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 16 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Long-term conversational memory is a core capability for LLM-based dialogue systems, yet existing benchmarks and evaluation protocols primarily focus on surface-level factual recall. In realistic interactions, appropriate responses often depend on implicit constraints such as user state, goals, or values that are not explicitly queried later. To evaluate this setting, we introduce \textbf{LoCoMo-Plus}, a benchmark for assessing cognitive memory under cue--trigger semantic disconnect, where models must retain and apply latent constraints across long conversational contexts. We further show that conventional string-matching metrics and explicit task-type prompting are misaligned with such scenarios, and propose a unified evaluation framework based on constraint consistency. Experiments across diverse backbone models, retrieval-based methods, and memory systems demonstrate that cognitive memory remains challenging and reveals failures not captured by existing benchmarks. Our code and evaluation framework are publicly available at: https://github.com/xjtuleeyf/Locomo-Plus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.00402v2">A Conditional Companion: Lived Experiences of People with Mental Health Disorders Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Accepted for presentation at CHI 2026 in Barcelona (ACM CHI Conference on Human Factors in Computing Systems)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for mental health support, yet little is known about how people with mental health challenges engage with them, how they evaluate their usefulness, and what design opportunities they envision. We conducted 20 semi-structured interviews with people in the UK who live with mental health conditions and have used LLMs for mental health support. Through reflexive thematic analysis, we found that participants engaged with LLMs in conditional and situational ways: for immediacy, the desire for non-judgement, self-paced disclosure, cognitive reframing, and relational engagement. Simultaneously, participants articulated clear boundaries informed by prior therapeutic experience: LLMs were effective for mild-to-moderate distress but inadequate for crises, trauma, and complex social-emotional situations. We contribute empirical insights into the lived use of LLMs for mental health, highlight boundary-setting as central to their safe role, and propose design and governance directions for embedding them responsibly within care ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10693v1">VESPO: Variational Sequence-Level Soft Policy Optimization for Stable Off-Policy LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Training stability remains a central challenge in reinforcement learning (RL) for large language models (LLMs). Policy staleness, asynchronous training, and mismatches between training and inference engines all cause the behavior policy to diverge from the current policy, risking training collapse. Importance sampling provides a principled correction for this distribution shift but suffers from high variance; existing remedies such as token-level clipping and sequence-level normalization lack a unified theoretical foundation. We propose Variational sEquence-level Soft Policy Optimization (VESPO). By incorporating variance reduction into a variational formulation over proposal distributions, VESPO derives a closed-form reshaping kernel that operates directly on sequence-level importance weights without length normalization. Experiments on mathematical reasoning benchmarks show that VESPO maintains stable training under staleness ratios up to 64x and fully asynchronous execution, and delivers consistent gains across both dense and Mixture-of-Experts models. Code is available at https://github.com/FloyedShen/VESPO
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10684v1">Privacy Control in Conversational LLM Platforms: A Walkthrough Study</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated into daily life through conversational interfaces, processing user data via natural language inputs and exhibiting advanced reasoning capabilities, which raises new concerns about user control over privacy. While much research has focused on potential privacy risks, less attention has been paid to the data control mechanisms these platforms provide. This study examines six conversational LLM platforms, analyzing how they define and implement features for users to access, edit, delete, and share data. Our analysis reveals an emerging paradigm of data control in conversational LLM platforms, where user data is generated and derived through interaction itself, natural language enables flexible yet often ambiguous control, and multi-user interactions with shared data raise questions of co-ownership and governance. Based on these findings, we offer practical insights for platform developers, policymakers, and researchers to design more effective and usable privacy controls in LLM-powered conversational interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11215v1">Charting Empirical Laws for LLM Fine-Tuning in Scientific Multi-Discipline Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have achieved strong performance through fine-tuning within individual scientific domains, their learning dynamics in multi-disciplinary contexts remains poorly understood, despite the promise of improved generalization and broader applicability through cross-domain knowledge synergy. In this work, we present the first systematic study of multi-disciplinary LLM fine-tuning, constructing a five-discipline corpus and analyzing learning patterns of full fine-tuning, LoRA, LoRA-MoE, and LoRA compositions. Particularly, our study shows that multi-disciplinary learning is substantially more variable than single-discipline training and distills four consistent empirical laws: (1) Balance-then-Diversity: low-resource disciplines degrade performance unless mitigated via diversity-aware upsampling; (2) Merge-then-Align: restoring instruction-following ability is critical for cross-discipline synergy; (3) Optimize-then-Scale: parameter scaling offers limited gains without prior design optimization; and (4) Share-then-Specialize: asymmetric LoRA-MoE yields robust gains with minimal trainable parameters via shared low-rank projection. Together, these laws form a practical recipe for principled multi-discipline fine-tuning and provide actionable guidance for developing generalizable scientific LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09514v2">EcoGym: Evaluating LLMs for Long-Horizon Plan-and-Execute in Interactive Economies</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      Long-horizon planning is widely recognized as a core capability of autonomous LLM-based agents; however, current evaluation frameworks suffer from being largely episodic, domain-specific, or insufficiently grounded in persistent economic dynamics. We introduce EcoGym, a generalizable benchmark for continuous plan-and-execute decision making in interactive economies. EcoGym comprises three diverse environments: Vending, Freelance, and Operation, implemented in a unified decision-making process with standardized interfaces, and budgeted actions over an effectively unbounded horizon (1000+ steps if 365 day-loops for evaluation). The evaluation of EcoGym is based on business-relevant outcomes (e.g., net worth, income, and DAU), targeting long-term strategic coherence and robustness under partial observability and stochasticity. Experiments across eleven leading LLMs expose a systematic tension: no single model dominates across all three scenarios. Critically, we find that models exhibit significant suboptimality in either high-level strategies or efficient actions executions. EcoGym is released as an open, extensible testbed for transparent long-horizon agent evaluation and for studying controllability-utility trade-offs in realistic economic settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2310.11409v7">LLMs as Hackers: Autonomous Linux Privilege Escalation Attacks</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Penetration-testing is crucial for identifying system vulnerabilities, with privilege-escalation being a critical subtask to gain elevated access to protected resources. Language Models (LLMs) presents new avenues for automating these security practices by emulating human behavior. However, a comprehensive understanding of LLMs' efficacy and limitations in performing autonomous Linux privilege-escalation attacks remains under-explored. To address this gap, we introduce hackingBuddyGPT, a fully automated LLM-driven prototype designed for autonomous Linux privilege-escalation. We curated a novel, publicly available Linux privilege-escalation benchmark, enabling controlled and reproducible evaluation. Our empirical analysis assesses the quantitative success rates and qualitative operational behaviors of various LLMs -- GPT-3.5-Turbo, GPT-4-Turbo, and Llama3 -- against baselines of human professional pen-testers and traditional automated tools. We investigate the impact of context management strategies, different context sizes, and various high-level guidance mechanisms on LLM performance. Results show that GPT-4-Turbo demonstrates high efficacy, successfully exploiting 33-83% of vulnerabilities, a performance comparable to human pen-testers (75%). In contrast, local models like Llama3 exhibited limited success (0-33%), and GPT-3.5-Turbo achieved moderate rates (16-50%). We show that both high-level guidance and state-management through LLM-driven reflection significantly boost LLM success rates. Qualitative analysis reveals both LLMs' strengths and weaknesses in generating valid commands and highlights challenges in common-sense reasoning, error handling, and multi-step exploitation, particularly with temporal dependencies. Cost analysis indicates that GPT-4-Turbo can achieve human-comparable performance at competitive costs, especially with optimized context management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10622v1">How Do Decoder-Only LLMs Perceive Users? Rethinking Attention Masking for User Representation Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Decoder-only large language models are increasingly used as behavioral encoders for user representation learning, yet the impact of attention masking on the quality of user embeddings remains underexplored. In this work, we conduct a systematic study of causal, hybrid, and bidirectional attention masks within a unified contrastive learning framework trained on large-scale real-world Alipay data that integrates long-horizon heterogeneous user behaviors. To improve training dynamics when transitioning from causal to bidirectional attention, we propose Gradient-Guided Soft Masking, a gradient-based pre-warmup applied before a linear scheduler that gradually opens future attention during optimization. Evaluated on 9 industrial user cognition benchmarks covering prediction, preference, and marketing sensitivity tasks, our approach consistently yields more stable training and higher-quality bidirectional representations compared with causal, hybrid, and scheduler-only baselines, while remaining compatible with decoder pretraining. Overall, our findings highlight the importance of masking design and training transition in adapting decoder-only LLMs for effective user representation learning. Our code is available at https://github.com/JhCircle/Deepfind-GGSM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10620v1">ISD-Agent-Bench: A Comprehensive Benchmark for Evaluating LLM-based Instructional Design Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents have shown promising potential in automating Instructional Systems Design (ISD), a systematic approach to developing educational programs. However, evaluating these agents remains challenging due to the lack of standardized benchmarks and the risk of LLM-as-judge bias. We present ISD-Agent-Bench, a comprehensive benchmark comprising 25,795 scenarios generated via a Context Matrix framework that combines 51 contextual variables across 5 categories with 33 ISD sub-steps derived from the ADDIE model. To ensure evaluation reliability, we employ a multi-judge protocol using diverse LLMs from different providers, achieving high inter-judge reliability. We compare existing ISD agents with novel agents grounded in classical ISD theories such as ADDIE, Dick \& Carey, and Rapid Prototyping ISD. Experiments on 1,017 test scenarios demonstrate that integrating classical ISD frameworks with modern ReAct-style reasoning achieves the highest performance, outperforming both pure theory-based agents and technique-only approaches. Further analysis reveals that theoretical quality strongly correlates with benchmark performance, with theory-based agents showing significant advantages in problem-centered design and objective-assessment alignment. Our work provides a foundation for systematic LLM-based ISD research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.02742v2">Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Molecular understanding is central to advancing areas such as scientific discovery, yet Large Language Models (LLMs) struggle to understand molecular graphs effectively. Existing graph-LLM bridges often adapt the Q-Former-style connector with fixed-length static tokens, which is originally designed for vision tasks. These designs overlook stereochemistry and substructural context and typically require costly LLM-backbone fine-tuning, limiting efficiency and generalization. We introduce EDT-Former, an Entropy-guided Dynamic Token Transformer that generates tokens aligned with informative molecular patches, thereby preserving both local and global structural features for molecular graph understanding. Beyond prior approaches, EDT-Former enables alignment between frozen graph encoders and LLMs without tuning the LLM backbone (excluding the embedding layer), resulting in computationally efficient finetuning, and achieves stateof-the-art results on MoleculeQA, Molecule-oriented Mol-Instructions, and property prediction benchmarks (TDC, MoleculeNet), underscoring its effectiveness for scalable and generalizable multimodal molecular understanding
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10577v1">Campaign-2-PT-RAG: LLM-Guided Semantic Product Type Attribution for Scalable Campaign Ranking</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      E-commerce campaign ranking models require large-scale training labels indicating which users purchased due to campaign influence. However, generating these labels is challenging because campaigns use creative, thematic language that does not directly map to product purchases. Without clear product-level attribution, supervised learning for campaign optimization remains limited. We present \textbf{Campaign-2-PT-RAG}, a scalable label generation framework that constructs user--campaign purchase labels by inferring which product types (PTs) each campaign promotes. The framework first interprets campaign content using large language models (LLMs) to capture implicit intent, then retrieves candidate PTs through semantic search over the platform taxonomy. A structured LLM-based classifier evaluates each PT's relevance, producing a campaign-specific product coverage set. User purchases matching these PTs generate positive training labels for downstream ranking models. This approach reframes the ambiguous attribution problem into a tractable semantic alignment task, enabling scalable and consistent supervision for downstream tasks such as campaign ranking optimization in production e-commerce environments. Experiments on internal and synthetic datasets, validated against expert-annotated campaign--PT mappings, show that our LLM-assisted approach generates high-quality labels with 78--90% precision while maintaining over 99% recall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10576v1">LLM-Based Scientific Equation Discovery via Physics-Informed Token-Regularized Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Symbolic regression aims to distill mathematical equations from observational data. Recent approaches have successfully leveraged Large Language Models (LLMs) to generate equation hypotheses, capitalizing on their vast pre-trained scientific priors. However, existing frameworks predominantly treat the LLM as a static generator, relying on prompt-level guidance to steer exploration. This paradigm fails to update the model's internal representations based on search feedback, often yielding physically inconsistent or mathematically redundant expressions. In this work, we propose PiT-PO (Physics-informed Token-regularized Policy Optimization), a unified framework that evolves the LLM into an adaptive generator via reinforcement learning. Central to PiT-PO is a dual-constraint mechanism that rigorously enforces hierarchical physical validity while simultaneously applying fine-grained, token-level penalties to suppress redundant structures. Consequently, PiT-PO aligns LLM to produce equations that are both scientifically consistent and structurally parsimonious. Empirically, PiT-PO achieves state-of-the-art performance on standard benchmarks and successfully discovers novel turbulence models for challenging fluid dynamics problems. We also demonstrate that PiT-PO empowers small-scale models to outperform closed-source giants, democratizing access to high-performance scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10568v1">Gauss-Newton Unlearning for the LLM Era</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Standard large language model training can create models that produce outputs their trainer deems unacceptable in deployment. The probability of these outputs can be reduced using methods such as LLM unlearning. However, unlearning a set of data (called the forget set) can degrade model performance on other distributions where the trainer wants to retain the model's behavior. To improve this trade-off, we demonstrate that using the forget set to compute only a few uphill Gauss-Newton steps provides a conceptually simple, state-of-the-art unlearning approach for LLMs. While Gauss-Newton steps adapt Newton's method to non-linear models, it is non-trivial to efficiently and accurately compute such steps for LLMs. Hence, our approach crucially relies on parametric Hessian approximations such as Kronecker-Factored Approximate Curvature (K-FAC). We call this combined approach K-FADE (K-FAC for Distribution Erasure). Our evaluation on the WMDP and ToFU benchmarks demonstrates that K-FADE suppresses outputs from the forget set and approximates, in output space, the results of retraining without the forget set. Critically, our method does this while altering the outputs on the retain set less than previous methods. This is because K-FADE transforms a constraint on the model's outputs across the entire retain set into a constraint on the model's weights, allowing the algorithm to minimally change the model's behavior on the retain set at each step. Moreover, the unlearning updates computed by K-FADE can be reapplied later if the model undergoes further training, allowing unlearning to be cheaply maintained.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17512v2">Is Your LLM Really Mastering the Concept? A Multi-Agent Benchmark</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Concepts serve as fundamental abstractions that support human reasoning and categorization. However, it remains unclear whether large language models truly capture such conceptual structures or primarily rely on surface-level pattern memorization. Existing benchmarks are largely static and fact oriented, which limits their ability to probe fine-grained semantic understanding and makes them vulnerable to data leakage and overfitting. To address this limitation, we introduce CK-Arena, a dynamic benchmark for conceptual knowledge evaluation based on a multi agent social deduction game, namely the Undercover game. In this setting, LLM based agents are assigned subtly different concept words and must describe, distinguish, and infer conceptual properties from others' statements. Model performance is evaluated through both game level outcomes and the semantic quality of generated descriptions. Furthermore, CK-Arena leverages the interaction process to automatically construct high quality question answering data for fine grained diagnostic analysis. Experimental results show that conceptual understanding varies substantially across models and categories, and is not strictly aligned with overall model capability. The data and code are available at the project homepage: https://ck-arena.site.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.09651v2">Geospatial Representation Learning: A Survey from Deep Learning to The LLM Era</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      The ability to transform location-centric geospatial data into meaningful computational representations has become fundamental to modern spatial analysis and decision-making. Geospatial Representation Learning (GRL), the process of automatically extracting latent structures and semantic patterns from geographic data, is undergoing a profound transformation through two successive technological revolutions: the deep learning breakthrough and the emerging large language model (LLM) paradigm. While deep neural networks (DNNs) have demonstrated remarkable success in automated feature extraction from structured and semi-structured geospatial data (e.g., satellite imagery, GPS trajectories), the recent integration of LLMs introduces transformative capabilities for cross-modal geospatial reasoning and unstructured geo-textual data processing. This survey presents a comprehensive review of geospatial representation learning across both technological eras, organizing them into a structured taxonomy based on the complete pipeline comprising: (1) data perspective, (2) methodological perspective, and (3) application perspective. We also highlight current advancements, discuss existing limitations, and propose potential future research directions in the LLM and foundation model era. This work offers a thorough exploration of the field and provides a roadmap for further innovation in GRL. The summary of the up-to-date paper list can be found in https://github.com/CityMind-Lab/Awesome-Geospatial-Representation-Learning and will undergo continuous updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.12365v4">Advances in LLMs with Focus on Reasoning, Adaptability, Efficiency and Ethics</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      This survey paper outlines the key developments in the field of Large Language Models (LLMs), including enhancements to their reasoning skills, adaptability to various tasks, increased computational efficiency, and the ability to make ethical decisions. The techniques that have been most effective in bridging the gap between human and machine communications include the Chain-of-Thought prompting, Instruction Tuning, and Reinforcement Learning from Human Feedback. The improvements in multimodal learning and few-shot or zero-shot techniques have further empowered LLMs to handle complex jobs with minor input. A significant focus is placed on efficiency, detailing scaling strategies, optimization techniques, and the influential Mixture-of-Experts (MoE) architecture, which strategically routes inputs to specialized subnetworks to boost predictive accuracy, while optimizing resource allocation. This survey also offers a broader perspective on recent advancements in LLMs, going beyond isolated aspects such as model architecture or ethical concerns. Additionally, it explores the role of LLMs in Agentic AI and their use as Autonomous Decision-Making Systems, and categorizes emerging methods that enhance LLM reasoning, efficiency, and ethical alignment. The survey also identifies underexplored areas such as interpretability, cross-modal integration, and sustainability. While significant advancements have been made in LLMs, challenges such as high computational costs, biases, and ethical risks remain. Overcoming these requires a focus on bias mitigation, transparent decision-making, and explicit ethical guidelines. Future research will generally focus on enhancing the model's ability to handle multiple inputs, thereby making it more intelligent, safe, and reliable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23048v2">From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language models now solve many benchmark math problems at near-expert levels, yet this progress has not fully translated into reliable performance in real-world applications. We study this gap through contextual mathematical reasoning, where the mathematical core must be formulated from descriptive scenarios. We introduce ContextMATH, a benchmark that repurposes AIME and MATH-500 problems into two contextual settings: Scenario Grounding (SG), which embeds abstract problems into realistic narratives without increasing reasoning complexity, and Complexity Scaling (CS), which transforms explicit conditions into sub-problems to capture how constraints often appear in practice. Evaluating 61 proprietary and open-source models, we observe sharp drops: on average, open-source models decline by 13 and 34 points on SG and CS, while proprietary models drop by 13 and 20. Error analysis shows that errors are dominated by incorrect problem formulation, with formulation accuracy declining as original problem difficulty increases. Correct formulation emerges as a prerequisite for success, and its sufficiency improves with model scale, indicating that larger models advance in both understanding and reasoning. Nevertheless, formulation and reasoning remain two complementary bottlenecks that limit contextual mathematical problem solving. Finally, we find that fine-tuning with scenario data improves performance, whereas formulation-only training is ineffective. However, performance gaps are only partially alleviated, highlighting contextual mathematical reasoning as a central unsolved challenge for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10512v1">Don't Eliminate Cut: Exponential Separations in LLM-Based Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      We develop a theoretical analysis of LLM-guided formal theorem proving in interactive proof assistants (e.g., Lean) by modeling tactic proposal as a stochastic policy in a finite-horizon deterministic MDP. To capture modern representation learning, we treat the state and action spaces as general compact metric spaces and assume Lipschitz policies. To explain the gap between worst-case hardness and empirical success, we introduce problem distributions generated by a reference policy $q$, including a latent-variable model in which proofs exhibit reusable cut/lemma/sketch structure represented by a proof DAG. Under a top-$k$ search protocol and Tsybakov-type margin conditions, we derive lower bounds on finite-horizon success probability that decompose into search and learning terms, with learning controlled by sequential Rademacher/covering complexity. Our main separation result shows that when cut elimination expands a DAG of depth $D$ into a cut-free tree of size $Ω(Λ^D)$ while the cut-aware hierarchical process has size $O(λ^D)$ with $λ\llΛ$, a flat (cut-free) learner provably requires exponentially more data than a cut-aware hierarchical learner. This provides a principled justification for subgoal decomposition in recent agentic theorem provers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10498v1">When Skills Lie: Hidden-Comment Injection in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      LLM agents often rely on Skills to describe available tools and recommended procedures. We study a hidden-comment prompt injection risk in this documentation layer: when a Markdown Skill is rendered to HTML, HTML comment blocks can become invisible to human reviewers, yet the raw text may still be supplied verbatim to the model. In experiments, we find that DeepSeek-V3.2 and GLM-4.5-Air can be influenced by malicious instructions embedded in a hidden comment appended to an otherwise legitimate Skill, yielding outputs that contain sensitive tool intentions. A short defensive system prompt that treats Skills as untrusted and forbids sensitive actions prevents these malicious tool calls and instead surfaces the suspicious hidden instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.05252v2">Copyright Detective: A Forensic System to Evidence LLMs Flickering Copyright Leakage Risks</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      We present Copyright Detective, the first interactive forensic system for detecting, analyzing, and visualizing potential copyright risks in LLM outputs. The system treats copyright infringement versus compliance as an evidence discovery process rather than a static classification task due to the complex nature of copyright law. It integrates multiple detection paradigms, including content recall testing, paraphrase-level similarity analysis, persuasive jailbreak probing, and unlearning verification, within a unified and extensible framework. Through interactive prompting, response collection, and iterative workflows, our system enables systematic auditing of verbatim memorization and paraphrase-level leakage, supporting responsible deployment and transparent evaluation of LLM copyright risks even with black-box access.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.10830v3">The Siren Song of LLMs: How Users Perceive and Respond to Dark Patterns in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 23 pages, 7 figures. Accepted at CHI 2026 (ACM Conference on Human Factors in Computing Systems), Barcelona, Spain
    </div>
    <details class="paper-abstract">
      Large language models can influence users through conversation, creating new forms of dark patterns that differ from traditional UX dark patterns. We define LLM dark patterns as manipulative or deceptive behaviors enacted in dialogue. Drawing on prior work and AI incident reports, we outline a diverse set of categories with real-world examples. Using them, we conducted a scenario-based study where participants (N=34) compared manipulative and neutral LLM responses. Our results reveal that recognition of LLM dark patterns often hinged on conversational cues such as exaggerated agreement, biased framing, or privacy intrusions, but these behaviors were also sometimes normalized as ordinary assistance. Users' perceptions of these dark patterns shaped how they respond to them. Responsibilities for these behaviors were also attributed in different ways, with participants assigning it to companies and developers, the model itself, or to users. We conclude with implications for design, advocacy, and governance to safeguard user autonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.23001v3">Bias Beyond Borders: Political Ideology Evaluation and Steering in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 PrePrint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly shape global discourse, making fairness and ideological neutrality essential for responsible AI deployment. Despite growing attention to political bias in LLMs, prior work largely focuses on high-resource, Western languages or narrow multilingual settings, leaving cross-lingual consistency and safe post-hoc mitigation underexplored. To address this gap, we present a large-scale multilingual evaluation of political bias spanning 50 countries and 33 languages. We introduce a complementary post-hoc mitigation framework, Cross-Lingual Alignment Steering (CLAS), designed to augment existing steering methods by aligning ideological representations across languages and dynamically regulating intervention strength. This method aligns latent ideological representations induced by political prompts into a shared ideological subspace, ensuring cross lingual consistency, with the adaptive mechanism prevents over correction and preserves coherence. Experiments demonstrate substantial bias reduction along both economic and social axes with minimal degradation in response quality. The proposed framework establishes a scalable and interpretable paradigm for fairness-aware multilingual LLM governance, balancing ideological neutrality with linguistic and cultural diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.16814v5">From Belief Entrenchment to Robust Reasoning in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 Accepted to TACL
    </div>
    <details class="paper-abstract">
      Multi-Agent Debate (MAD) has emerged as a promising inference scaling method for Large Language Model (LLM) reasoning. However, it frequently suffers from belief entrenchment, where agents reinforce shared errors rather than correcting them. Going beyond merely identifying this failure, we decompose it into two distinct root causes: (1) the model's biased $\textit{static initial belief}$ and (2) $\textit{homogenized debate dynamics}$ that amplify the majority view regardless of correctness. To address these sequentially, we propose $\textbf{DReaMAD}$ $($$\textbf{D}$iverse $\textbf{Rea}$soning via $\textbf{M}$ulti-$\textbf{A}$gent $\textbf{D}$ebate with Refined Prompt$)$. Our framework first rectifies the static belief via strategic prior knowledge elicitation, then reshapes the debate dynamics by enforcing perspective diversity. Validated on our new $\textit{MetaNIM Arena}$ benchmark, $\textbf{DReaMAD}$ significantly mitigates entrenchment, achieving a +9.5\% accuracy gain over ReAct prompting and a +19.0\% higher win rate than standard MAD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10471v1">TestExplora: Benchmarking LLMs for Proactive Bug Discovery via Repository-Level Test Generation</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Given that Large Language Models (LLMs) are increasingly applied to automate software development, comprehensive software assurance spans three distinct goals: regression prevention, reactive reproduction, and proactive discovery. Current evaluations systematically overlook the third goal. Specifically, they either treat existing code as ground truth (a compliance trap) for regression prevention, or depend on post-failure artifacts (e.g., issue reports) for bug reproduction-so they rarely surface defects before failures. To bridge this gap, we present TestExplora, a benchmark designed to evaluate LLMs as proactive testers within full-scale, realistic repository environments. TestExplora contains 2,389 tasks from 482 repositories and hides all defect-related signals. Models must proactively find bugs by comparing implementations against documentation-derived intent, using documentation as the oracle. Furthermore, to keep evaluation sustainable and reduce leakage, we propose continuous, time-aware data collection. Our evaluation reveals a significant capability gap: state-of-the-art models achieve a maximum Fail-to-Pass (F2P) rate of only 16.06%. Further analysis indicates that navigating complex cross-module interactions and leveraging agentic exploration are critical to advancing LLMs toward autonomous software quality assurance. Consistent with this, SWEAgent instantiated with GPT-5-mini achieves an F2P of 17.27% and an F2P@5 of 29.7%, highlighting the effectiveness and promise of agentic exploration in proactive bug discovery tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.01076v4">When Speculation Spills Secrets: Side Channels via Speculative Decoding In LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Deployed large language models (LLMs) often rely on speculative decoding, a technique that generates and verifies multiple candidate tokens in parallel, to improve throughput and latency. In this work, we reveal a new side-channel whereby input-dependent patterns of correct and incorrect speculations can be inferred by monitoring per-iteration token counts or packet sizes. In evaluations using research prototypes and production-grade vLLM serving frameworks, we show that an adversary monitoring these patterns can fingerprint user queries (from a set of 50 prompts) with over 75% accuracy across four speculative-decoding schemes at temperature 0.3: REST (100%), LADE (91.6%), BiLD (95.2%), and EAGLE (77.6%). Even at temperature 1.0, accuracy remains far above the 2% random baseline - REST (99.6%), LADE (61.2%), BiLD (63.6%), and EAGLE (24%). We also show the capability of the attacker to leak confidential datastore contents used for prediction at rates exceeding 25 tokens/sec. To defend against these, we propose and evaluate a suite of mitigations, including packet padding and iteration-wise token aggregation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10454v1">LATA: A Tool for LLM-Assisted Translation Annotation</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      The construction of high-quality parallel corpora for translation research has increasingly evolved from simple sentence alignment to complex, multi-layered annotation tasks. This methodological shift presents significant challenges for structurally divergent language pairs, such as Arabic--English, where standard automated tools frequently fail to capture deep linguistic shifts or semantic nuances. This paper introduces a novel, LLM-assisted interactive tool designed to reduce the gap between scalable automation and the rigorous precision required for expert human judgment. Unlike traditional statistical aligners, our system employs a template-based Prompt Manager that leverages large language models (LLMs) for sentence segmentation and alignment under strict JSON output constraints. In this tool, automated preprocessing integrates into a human-in-the-loop workflow, allowing researchers to refine alignments and apply custom translation technique annotations through a stand-off architecture. By leveraging LLM-assisted processing, the tool balances annotation efficiency with the linguistic precision required to analyze complex translation phenomena in specialized domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10453v1">The Landscape of Prompt Injection Threats in LLM Agents: From Taxonomy to Analysis</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      The evolution of Large Language Models (LLMs) has resulted in a paradigm shift towards autonomous agents, necessitating robust security against Prompt Injection (PI) vulnerabilities where untrusted inputs hijack agent behaviors. This SoK presents a comprehensive overview of the PI landscape, covering attacks, defenses, and their evaluation practices. Through a systematic literature review and quantitative analysis, we establish taxonomies that categorize PI attacks by payload generation strategies (heuristic vs. optimization) and defenses by intervention stages (text, model, and execution levels). Our analysis reveals a key limitation shared by many existing defenses and benchmarks: they largely overlook context-dependent tasks, in which agents are authorized to rely on runtime environmental observations to determine actions. To address this gap, we introduce AgentPI, a new benchmark designed to systematically evaluate agent behavior under context-dependent interaction settings. Using AgentPI, we empirically evaluate representative defenses and show that no single approach can simultaneously achieve high trustworthiness, high utility, and low latency. Moreover, we show that many defenses appear effective under existing benchmarks by suppressing contextual inputs, yet fail to generalize to realistic agent settings where context-dependent reasoning is essential. This SoK distills key takeaways and open research problems, offering structured guidance for future research and practical deployment of secure LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09379v2">LingxiDiagBench: A Multi-Agent Framework for Benchmarking LLMs in Chinese Psychiatric Consultation and Diagnosis</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Mental disorders are highly prevalent worldwide, but the shortage of psychiatrists and the inherent subjectivity of interview-based diagnosis create substantial barriers to timely and consistent mental-health assessment. Progress in AI-assisted psychiatric diagnosis is constrained by the absence of benchmarks that simultaneously provide realistic patient simulation, clinician-verified diagnostic labels, and support for dynamic multi-turn consultation. We present LingxiDiagBench, a large-scale multi-agent benchmark that evaluates LLMs on both static diagnostic inference and dynamic multi-turn psychiatric consultation in Chinese. At its core is LingxiDiag-16K, a dataset of 16,000 EMR-aligned synthetic consultation dialogues designed to reproduce real clinical demographic and diagnostic distributions across 12 ICD-10 psychiatric categories. Through extensive experiments across state-of-the-art LLMs, we establish key findings: (1) although LLMs achieve high accuracy on binary depression--anxiety classification (up to 92.3%), performance deteriorates substantially for depression--anxiety comorbidity recognition (43.0%) and 12-way differential diagnosis (28.5%); (2) dynamic consultation often underperforms static evaluation, indicating that ineffective information-gathering strategies significantly impair downstream diagnostic reasoning; (3) consultation quality assessed by LLM-as-a-Judge shows only moderate correlation with diagnostic accuracy, suggesting that well-structured questioning alone does not ensure correct diagnostic decisions. We release LingxiDiag-16K and the full evaluation framework to support reproducible research at https://github.com/Lingxi-mental-health/LingxiDiagBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.11209v1">SAFuzz: Semantic-Guided Adaptive Fuzzing for LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 11 pages, 6 figures, 4 tables
    </div>
    <details class="paper-abstract">
      While AI-coding assistants accelerate software development, current testing frameworks struggle to keep pace with the resulting volume of AI-generated code. Traditional fuzzing techniques often allocate resources uniformly and lack semantic awareness of algorithmic vulnerability patterns, leading to inefficient resource usage and missed vulnerabilities. To address these limitations, we present a hybrid testing framework that leverages LLM-guided adaptive fuzzing to detect algorithmic vulnerabilities efficiently. Our system SAFuzz integrates prompt-based behavioral diversification, harness generation with problem-specific oracles, and an LLM-based predictor to enable adaptive resource allocation and dynamic early stopping. Evaluating SAFuzz on CSES algorithmic problems, we improve vulnerability discrimination precision from 77.9% to 85.7% and achieve a 1.71x reduction in time cost compared to SOTA GreenFuzz while maintaining comparable recall. We further observe that combining our approach with existing unit test generation methods yields complementary gains, increasing the bug detection recall from 67.3% to 79.5%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18868v2">KernelBand: Steering LLM-based Kernel Optimization via Hardware-Aware Multi-Armed Bandits</a></div>
    <div class="paper-meta">
      📅 2026-02-11
      | 💬 19 pages (9 pages main text), 4 figures. v2: Full revision
    </div>
    <details class="paper-abstract">
      High-performance GPU kernels are critical for efficient LLM serving, yet their optimization remains a bottleneck requiring deep system expertise. While code LLMs show promise in generating functionally correct code, kernel optimization is intrinsically a search problem over a vast optimization space. The fundamental mismatch prevents existing LLM agents from efficiently exploring the optimization space for diverse hardware and compute patterns. To bridge the gap, we present KernelBand, a framework that formulates kernel optimization as a Multi-Armed Bandit (MAB) problem, explicitly balancing exploration and exploitation to unlock the potential of code LLMs. To navigate the infinite arm space of optimization strategies applied to candidate kernels, we design two key mechanisms: a hardware-aware pruning strategy via profiling bounds and a trace-driven clustering algorithm that leverages Lipschitz continuity. Theoretically, we prove that KernelBand reduces the regret bound to depend on the compact covering number of runtime clusters, ensuring sample-efficient discovery of high-performance kernels. Extensive experiments on TritonBench-G with three GPU architectures and four code LLMs show that KernelBand consistently and substantially outperforms state-of-the-art methods with over 33% average improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10388v1">Less is Enough: Synthesizing Diverse Data in Feature Space of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      The diversity of post-training data is critical for effective downstream performance in large language models (LLMs). Many existing approaches to constructing post-training data quantify diversity using text-based metrics that capture linguistic variation, but such metrics provide only weak signals for the task-relevant features that determine downstream performance. In this work, we introduce Feature Activation Coverage (FAC) which measures data diversity in an interpretable feature space. Building upon this metric, we further propose a diversity-driven data synthesis framework, named FAC Synthesis, that first uses a sparse autoencoder to identify missing features from a seed dataset, and then generates synthetic samples that explicitly reflect these features. Experiments show that our approach consistently improves both data diversity and downstream performance on various tasks, including instruction following, toxicity detection, reward modeling, and behavior steering. Interestingly, we identify a shared, interpretable feature space across model families (i.e., LLaMA, Mistral, and Qwen), enabling cross-model knowledge transfer. Our work provides a solid and practical methodology for exploring data-centric optimization of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10387v1">Making Databases Faster with LLM Evolutionary Sampling</a></div>
    <div class="paper-meta">
      📅 2026-02-11
    </div>
    <details class="paper-abstract">
      Traditional query optimization relies on cost-based optimizers that estimate execution cost (e.g., runtime, memory, and I/O) using predefined heuristics and statistical models. Improving these heuristics requires substantial engineering effort, and even when implemented, these heuristics often cannot take into account semantic correlations in queries and schemas that could enable better physical plans. Using our DBPlanBench harness for the DataFusion engine, we expose the physical plan through a compact serialized representation and let the LLM propose localized edits that can be applied and executed. We then apply an evolutionary search over these edits to refine candidates across iterations. Our key insight is that LLMs can leverage semantic knowledge to identify and apply non-obvious optimizations, such as join orderings that minimize intermediate cardinalities. We obtain up to 4.78$\times$ speedups on some queries and we demonstrate a small-to-large workflow in which optimizations found on small databases transfer effectively to larger databases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10377v1">Hardware Co-Design Scaling Laws via Roofline Modelling for On-Device LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Vision-Language-Action Models (VLAs) have emerged as a key paradigm of Physical AI and are increasingly deployed in autonomous vehicles, robots, and smart spaces. In these resource-constrained on-device settings, selecting an appropriate large language model (LLM) backbone is a critical challenge: models must balance accuracy with strict inference latency and hardware efficiency constraints. This makes hardware-software co-design a game-changing requirement for on-device LLM deployment, where each hardware platform demands a tailored architectural solution. We propose a hardware co-design law that jointly captures model accuracy and inference performance. Specifically, we model training loss as an explicit function of architectural hyperparameters and characterise inference latency via roofline modelling. We empirically evaluate 1,942 candidate architectures on NVIDIA Jetson Orin, training 170 selected models for 10B tokens each to fit a scaling law relating architecture to training loss. By coupling this scaling law with latency modelling, we establish a direct accuracy-latency correspondence and identify the Pareto frontier for hardware co-designed LLMs. We further formulate architecture search as a joint optimisation over precision and performance, deriving feasible design regions under industrial hardware and application budgets. Our approach reduces architecture selection from months to days. At the same latency as Qwen2.5-0.5B on the target hardware, our co-designed architecture achieves 19.42% lower perplexity on WikiText-2. To our knowledge, this is the first principled and operational framework for hardware co-design scaling laws in on-device LLM deployment. We will make the code and related checkpoints publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10371v1">Simple LLM Baselines are Competitive for Model Diffing</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Standard LLM evaluations only test capabilities or dispositions that evaluators designed them for, missing unexpected differences such as behavioral shifts between model revisions or emergent misaligned tendencies. Model diffing addresses this limitation by automatically surfacing systematic behavioral differences. Recent approaches include LLM-based methods that generate natural language descriptions and sparse autoencoder (SAE)-based methods that identify interpretable features. However, no systematic comparison of these approaches exists nor are there established evaluation criteria. We address this gap by proposing evaluation metrics for key desiderata (generalization, interestingness, and abstraction level) and use these to compare existing methods. Our results show that an improved LLM-based baseline performs comparably to the SAE-based method while typically surfacing more abstract behavioral differences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10367v1">LiveMedBench: A Contamination-Free Medical Benchmark for LLMs with Automated Rubric Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      The deployment of Large Language Models (LLMs) in high-stakes clinical settings demands rigorous and reliable evaluation. However, existing medical benchmarks remain static, suffering from two critical limitations: (1) data contamination, where test sets inadvertently leak into training corpora, leading to inflated performance estimates; and (2) temporal misalignment, failing to capture the rapid evolution of medical knowledge. Furthermore, current evaluation metrics for open-ended clinical reasoning often rely on either shallow lexical overlap (e.g., ROUGE) or subjective LLM-as-a-Judge scoring, both inadequate for verifying clinical correctness. To bridge these gaps, we introduce LiveMedBench, a continuously updated, contamination-free, and rubric-based benchmark that weekly harvests real-world clinical cases from online medical communities, ensuring strict temporal separation from model training data. We propose a Multi-Agent Clinical Curation Framework that filters raw data noise and validates clinical integrity against evidence-based medical principles. For evaluation, we develop an Automated Rubric-based Evaluation Framework that decomposes physician responses into granular, case-specific criteria, achieving substantially stronger alignment with expert physicians than LLM-as-a-Judge. To date, LiveMedBench comprises 2,756 real-world cases spanning 38 medical specialties and multiple languages, paired with 16,702 unique evaluation criteria. Extensive evaluation of 38 LLMs reveals that even the best-performing model achieves only 39.2%, and 84% of models exhibit performance degradation on post-cutoff cases, confirming pervasive data contamination risks. Error analysis further identifies contextual application-not factual knowledge-as the dominant bottleneck, with 35-48% of failures stemming from the inability to tailor medical knowledge to patient-specific constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.10222v3">Speech-Audio Compositional Attacks on Multimodal LLMs and Their Mitigation with SALMONN-Guard</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Recent progress in LLMs has enabled understanding of audio signals, but has also exposed new safety risks arising from complex audio inputs that are inadequately handled by current safeguards. We introduce SACRED-Bench (Speech-Audio Composition for RED-teaming) to evaluate the robustness of LLMs under complex audio-based attacks. Unlike existing perturbation-based methods that rely on noise optimization or white-box access, SACRED-Bench exploits speech-audio composition to enable effective black-box attacks. SACRED-Bench adopts three composition mechanisms: (a) overlap of harmful and benign speech, (b) mixture of benign speech with harmful non-speech audio, and (c) multi-speaker dialogue. These mechanisms focus on evaluating safety in settings where benign and harmful intents co-occur within a single auditory scene. Moreover, questions in SACRED-Bench are designed to implicitly refer to content in the audio, such that no explicit harmful information appears in the text prompt alone. Experiments demonstrate that even Gemini 2.5 Pro, a state-of-the-art proprietary LLM with safety guardrails fully enabled, still exhibits a 66% attack success rate. To bridge this gap, we propose SALMONN-Guard, the first guard model that jointly inspects speech, audio, and text for safety judgments, reducing the attack success rate to 20%. Our results highlight the need for audio-aware defenses to ensure the safety of multimodal LLMs. The dataset and SALMONN-Guard checkpoints can be found at https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10354v1">Physically Interpretable AlphaEarth Foundation Model Embeddings Enable LLM-Based Land Surface Intelligence</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Satellite foundation models produce dense embeddings whose physical interpretability remains poorly understood, limiting their integration into environmental decision systems. Using 12.1 million samples across the Continental United States (2017--2023), we first present a comprehensive interpretability analysis of Google AlphaEarth's 64-dimensional embeddings against 26 environmental variables spanning climate, vegetation, hydrology, temperature, and terrain. Combining linear, nonlinear, and attention-based methods, we show that individual embedding dimensions map onto specific land surface properties, while the full embedding space reconstructs most environmental variables with high fidelity (12 of 26 variables exceed $R^2 > 0.90$; temperature and elevation approach $R^2 = 0.97$). The strongest dimension-variable relationships converge across all three analytical methods and remain robust under spatial block cross-validation (mean $ΔR^2 = 0.017$) and temporally stable across all seven study years (mean inter-year correlation $r = 0.963$). Building on these validated interpretations, we then developed a Land Surface Intelligence system that implements retrieval-augmented generation over a FAISS-indexed embedding database of 12.1 million vectors, translating natural language environmental queries into satellite-grounded assessments. An LLM-as-Judge evaluation across 360 query--response cycles, using four LLMs in rotating generator, system, and judge roles, achieved weighted scores of $μ= 3.74 \pm 0.77$ (scale 1--5), with grounding ($μ= 3.93$) and coherence ($μ= 4.25$) as the strongest criteria. Our results demonstrate that satellite foundation model embeddings are physically structured representations that can be operationalized for environmental and geospatial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24069v3">Can LLMs Reason Structurally? Benchmarking via the Lens of Data Structures</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are deployed on increasingly complex tasks that require multi-step decision-making. Understanding their algorithmic reasoning abilities is therefore crucial. However, we lack a diagnostic benchmark for evaluating this capability. We propose data structures as a principled lens: as fundamental building blocks of algorithms, they naturally probe structural reasoning-the ability to understand and manipulate relationships such as order, hierarchy, and connectivity that underpin algorithmic reasoning. We introduce DSR-Bench, spanning 20 data structures, 35 operations, and 4,140 problem instances. DSR-Bench features hierarchical task organization, fully automated generation and evaluation, and fine-grained diagnostics. Evaluating 13 state-of-the-art LLMs reveals critical limitations: the top-performing model achieves only 0.46/1 on challenging instances. Three auxiliary probes targeting more realistic usages expose further weaknesses: models perform poorly on spatial data and context-rich scenarios, and they struggle to reason over their own code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10324v1">Discovering Differences in Strategic Behavior Between Humans and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed in social and strategic scenarios, it becomes critical to understand where and why their behavior diverges from that of humans. While behavioral game theory (BGT) provides a framework for analyzing behavior, existing models do not fully capture the idiosyncratic behavior of humans or black-box, non-human agents like LLMs. We employ AlphaEvolve, a cutting-edge program discovery tool, to directly discover interpretable models of human and LLM behavior from data, thereby enabling open-ended discovery of structural factors driving human and LLM behavior. Our analysis on iterated rock-paper-scissors reveals that frontier LLMs can be capable of deeper strategic behavior than humans. These results provide a foundation for understanding structural differences driving differences in human and LLM behavior in strategic interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03667v2">Invisible Saboteurs: Sycophantic LLMs Mislead Novices in Problem-Solving Tasks</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 ACM CHI conference on Human Factors in Computing Systems (2026)
    </div>
    <details class="paper-abstract">
      Sycophancy, the tendency of LLM-based chatbots to express excessive agreement with their users, even when inappropriate, is emerging as a significant risk in human-AI interactions. However, the extent to which this affects human-LLM collaboration in complex problem-solving tasks is not well quantified, especially among novices who are prone to misconceptions. We created two LLM chatbots, one with high sycophancy and one with low sycophancy, and conducted a within-subjects experiment (n=24) in the context of debugging machine learning models to investigate the effect of sycophancy on users' mental models, workflows, reliance behaviors, and perceptions of the chatbots. Our findings show that users of the high sycophancy chatbot were less likely to correct their misconceptions and spent more time over-relying on unhelpful LLM responses, leading them to significantly worse performance in the task. Despite these impaired outcomes, a majority of users were unable to detect the presence of excessive sycophancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10321v1">Single-Turn LLM Reformulation Powered Multi-Stage Hybrid Re-Ranking for Tip-of-the-Tongue Known-Item Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Retrieving known items from vague descriptions, Tip-of-the-Tongue (ToT) retrieval, remains a significant challenge. We propose using a single call to a generic 8B-parameter LLM for query reformulation, bridging the gap between ill-formed ToT queries and specific information needs. This method is particularly effective where standard Pseudo-Relevance Feedback fails due to poor initial recall. Crucially, our LLM is not fine-tuned for ToT or specific domains, demonstrating that gains stem from our prompting strategy rather than model specialization. Rewritten queries feed a multi-stage pipeline: sparse retrieval (BM25), dense/late-interaction reranking (Contriever, E5-large-v2, ColBERTv2), monoT5 cross-encoding, and list-wise reranking (Qwen 2.5 72B). Experiments on 2025 TREC-ToT datasets show that while raw queries yield poor performance, our lightweight pre-retrieval transformation improves Recall by 20.61%. Subsequent reranking improves nDCG@10 by 33.88%, MRR by 29.92%, and MAP@10 by 29.98%, offering a cost-effective intervention that unlocks the potential of downstream rankers. Code and data: https://github.com/debayan1405/TREC-TOT-2025
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10282v1">Linear-LLM-SCM: Benchmarking LLMs for Coefficient Elicitation in Linear-Gaussian Causal Models</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 16 pages, 4 figures, preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown potential in identifying qualitative causal relations, but their ability to perform quantitative causal reasoning -- estimating effect sizes that parametrize functional relationships -- remains underexplored in continuous domains. We introduce Linear-LLM-SCM, a plug-and-play benchmarking framework for evaluating LLMs on linear Gaussian structural causal model (SCM) parametrization when the DAG is given. The framework decomposes a DAG into local parent-child sets and prompts an LLM to produce a regression-style structural equation per node, which is aggregated and compared against available ground-truth parameters. Our experiments show several challenges in such benchmarking tasks, namely, strong stochasticity in the results in some of the models and susceptibility to DAG misspecification via spurious edges in the continuous domains. Across models, we observe substantial variability in coefficient estimates for some settings and sensitivity to structural and semantic perturbations, highlighting current limitations of LLMs as quantitative causal parameterizers. We also open-sourced the benchmarking framework so that researchers can utilize their DAGs and any off-the-shelf LLMs plug-and-play for evaluation in their domains effortlessly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14331v2">LLM Priors for ERM over Programs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      We study program-learning methods that are efficient in both samples and computation. Classical learning theory suggests that when the target admits a short program description (for example, a short piece of ``Python code''), it can be learned from relatively few examples by performing ERM over the program class. However, this approach relies on enumerating candidate programs, which is typically exponential in the description length. In contrast, gradient-based training avoids explicit search, but for some families of short programs it can require exponentially many samples to succeed. We propose \textsc{LLM-PV}, a propose-and-verify recipe that enables ERM-style selection over a discrete program class without exhaustive enumeration. A pretrained LLM induces a proposal distribution over candidate programs; each proposal is executed, scored on a held-out validation set, and the best program is selected. The method uses no gradient updates and does not use validation feedback to adapt the sampling distribution. Across algorithmic tasks including parity variants, pattern matching, and primality testing, \textsc{LLM-PV} often recovers the exact underlying rule from a small labeled set and generalizes far beyond the training sequence lengths. In the same regimes, SGD-trained transformers and standard adaptation baselines (fine-tuning and in-context learning), as well as classical ML baselines, can fit the training data yet fail to generalize reliably. Together, these results suggest that pretrained LLM priors can serve as effective search biases for ERM, narrowing the gap between statistical and computational efficiency. The code is available at [\href{https://github.com/DLFundamentals/LLM_PV}{code}].
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10246v1">KORAL: Knowledge Graph Guided LLM Reasoning for SSD Operational Analysis</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Solid State Drives (SSDs) are critical to datacenters, consumer platforms, and mission-critical systems. Yet diagnosing their performance and reliability is difficult because data are fragmented and time-disjoint, and existing methods demand large datasets and expert input while offering only limited insights. Degradation arises not only from shifting workloads and evolving architectures but also from environmental factors such as temperature, humidity, and vibration. We present KORAL, a knowledge driven reasoning framework that integrates Large Language Models (LLMs) with a structured Knowledge Graph (KG) to generate insights into SSD operations. Unlike traditional approaches that require extensive expert input and large datasets, KORAL generates a Data KG from fragmented telemetry and integrates a Literature KG that already organizes knowledge from literature, reports, and traces. This turns unstructured sources into a queryable graph and telemetry into structured knowledge, and both the Graphs guide the LLM to deliver evidence-based, explainable analysis aligned with the domain vocabulary and constraints. Evaluation using real production traces shows that the KORAL delivers expert-level diagnosis and recommendations, supported by grounded explanations that improve reasoning transparency, guide operator decisions, reduce manual effort, and provide actionable insights to improve service quality. To our knowledge, this is the first end-to-end system that combines LLMs and KGs for full-spectrum SSD reasoning including Descriptive, Predictive, Prescriptive, and What-if analysis. We release the generated SSD-specific KG to advance reproducible research in knowledge-based storage system analysis. GitHub Repository: https://github.com/Damrl-lab/KORAL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10226v1">Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Optimizing large-scale machine learning systems, such as recommendation models for global video platforms, requires navigating a massive hyperparameter search space and, more critically, designing sophisticated optimizers, architectures, and reward functions to capture nuanced user behaviors. Achieving substantial improvements in these areas is a non-trivial task, traditionally relying on extensive manual iterations to test new hypotheses. We propose a self-evolving system that leverages Large Language Models (LLMs), specifically those from Google's Gemini family, to autonomously generate, train, and deploy high-performing, complex model changes within an end-to-end automated workflow. The self-evolving system is comprised of an Offline Agent (Inner Loop) that performs high-throughput hypothesis generation using proxy metrics, and an Online Agent (Outer Loop) that validates candidates against delayed north star business metrics in live production. Our agents act as specialized Machine Learning Engineers (MLEs): they exhibit deep reasoning capabilities, discovering novel improvements in optimization algorithms and model architecture, and formulating innovative reward functions that target long-term user engagement. The effectiveness of this approach is demonstrated through several successful production launches at YouTube, confirming that autonomous, LLM-driven evolution can surpass traditional engineering workflows in both development velocity and model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10218v1">ACE-RTL: When Agentic Context Evolution Meets RTL-Specialized LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have sparked growing interest in applying them to hardware design automation, particularly for accurate RTL code generation. Prior efforts follow two largely independent paths: (i) training domain-adapted RTL models to internalize hardware semantics, (ii) developing agentic systems that leverage frontier generic LLMs guided by simulation feedback. However, these two paths exhibit complementary strengths and weaknesses. In this work, we present ACE-RTL that unifies both directions through Agentic Context Evolution (ACE). ACE-RTL integrates an RTL-specialized LLM, trained on a large-scale dataset of 1.7 million RTL samples, with a frontier reasoning LLM through three synergistic components: the generator, reflector, and coordinator. These components iteratively refine RTL code toward functional correctness. We further introduce a parallel scaling strategy that significantly reduces the number of iterations required to reach correct solutions. On the Comprehensive Verilog Design Problems (CVDP) benchmark, ACE-RTL achieves up to a 44.87% pass rate improvement over 14 competitive baselines while requiring only four iterations on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10210v1">How Much Reasoning Do Retrieval-Augmented Models Add beyond LLMs? A Benchmarking Framework for Multi-Hop Inference over Hybrid Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) continue to struggle with knowledge-intensive questions that require up-to-date information and multi-hop reasoning. Augmenting LLMs with hybrid external knowledge, such as unstructured text and structured knowledge graphs, offers a promising alternative to costly continual pretraining. As such, reliable evaluation of their retrieval and reasoning capabilities becomes critical. However, many existing benchmarks increasingly overlap with LLM pretraining data, which means answers or supporting knowledge may already be encoded in model parameters, making it difficult to distinguish genuine retrieval and reasoning from parametric recall. We introduce HybridRAG-Bench, a framework for constructing benchmarks to evaluate retrieval-intensive, multi-hop reasoning over hybrid knowledge. HybridRAG-Bench automatically couples unstructured text and structured knowledge graph representations derived from recent scientific literature on arXiv, and generates knowledge-intensive question-answer pairs grounded in explicit reasoning paths. The framework supports flexible domain and time-frame selection, enabling contamination-aware and customizable evaluation as models and knowledge evolve. Experiments across three domains (artificial intelligence, governance and policy, and bioinformatics) demonstrate that HybridRAG-Bench rewards genuine retrieval and reasoning rather than parametric recall, offering a principled testbed for evaluating hybrid knowledge-augmented reasoning systems. We release our code and data at github.com/junhongmit/HybridRAG-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10092v1">Quantum-Audit: Evaluating the Reasoning Limits of LLMs on Quantum Computing</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Language models have become practical tools for quantum computing education and research, from summarizing technical papers to explaining theoretical concepts and answering questions about recent developments in the field. While existing benchmarks evaluate quantum code generation and circuit design, their understanding of quantum computing concepts has not been systematically measured. Quantum-Audit addresses this gap with 2,700 questions covering core quantum computing topics. We evaluate 26 models from leading organizations. Our benchmark comprises 1,000 expert-written questions, 1,000 questions extracted from research papers using LLMs and validated by experts, plus an additional 700 questions including 350 open-ended questions and 350 questions with false premises to test whether models can correct erroneous assumptions. Human participants scored between 23% and 86%, with experts averaging 74%. Top-performing models exceeded the expert average, with Claude Opus 4.5 reaching 84% accuracy, though top models showed an average 12-point accuracy drop on expert-written questions compared to LLM-generated ones. Performance declined further on advanced topics, dropping to 73% on security questions. Additionally, models frequently accepted and reinforced false premises embedded in questions instead of identifying them, with accuracy below 66% on these critical reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08023v2">CyberExplorer: Benchmarking LLM Offensive Security Capabilities in a Real-World Attacking Simulation Environment</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Real-world offensive security operations are inherently open-ended: attackers explore unknown attack surfaces, revise hypotheses under uncertainty, and operate without guaranteed success. Existing LLM-based offensive agent evaluations rely on closed-world settings with predefined goals and binary success criteria. To address this gap, we introduce CyberExplorer, an evaluation suite with two core components: (1) an open-environment benchmark built on a virtual machine hosting 40 vulnerable web services derived from real-world CTF challenges, where agents autonomously perform reconnaissance, target selection, and exploitation without prior knowledge of vulnerability locations; and (2) a reactive multi-agent framework supporting dynamic exploration without predefined plans. CyberExplorer enables fine-grained evaluation beyond flag recovery, capturing interaction dynamics, coordination behavior, failure modes, and vulnerability discovery signals-bridging the gap between benchmarks and realistic multi-target attack scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03005v3">From Moderation to Mediation: Can LLMs Serve as Mediators in Online Flame Wars?</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has opened new possibilities for AI for good applications. As LLMs increasingly mediate online communication, their potential to foster empathy and constructive dialogue becomes an important frontier for responsible AI research. This work explores whether LLMs can serve not only as moderators that detect harmful content, but as mediators capable of understanding and de-escalating online conflicts. Our framework decomposes mediation into two subtasks: judgment, where an LLM evaluates the fairness and emotional dynamics of a conversation, and steering, where it generates empathetic, de-escalatory messages to guide participants toward resolution. To assess mediation quality, we construct a large Reddit-based dataset and propose a multi-stage evaluation pipeline combining principle-based scoring, user simulation, and human comparison. Experiments show that API-based models outperform open-source counterparts in both reasoning and intervention alignment when doing mediation. Our findings highlight both the promise and limitations of current LLMs as emerging agents for online social mediation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10017v1">SCORE: Specificity, Context Utilization, Robustness, and Relevance for Reference-Free LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to support question answering and decision-making in high-stakes, domain-specific settings such as natural hazard response and infrastructure planning, where effective answers must convey fine-grained, decision-critical details. However, existing evaluation frameworks for retrieval-augmented generation (RAG) and open-ended question answering primarily rely on surface-level similarity, factual consistency, or semantic relevance, and often fail to assess whether responses provide the specific information required for domain-sensitive decisions. To address this gap, we propose a multi-dimensional, reference-free evaluation framework that assesses LLM outputs along four complementary dimensions: specificity, robustness to paraphrasing and semantic perturbations, answer relevance, and context utilization. We introduce a curated dataset of 1,412 domain-specific question-answer pairs spanning 40 professional roles and seven natural hazard types to support systematic evaluation. We further conduct human evaluation to assess inter-annotator agreement and alignment between model outputs and human judgments, which highlights the inherent subjectivity of open-ended, domain-specific evaluation. Our results show that no single metric sufficiently captures answer quality in isolation and demonstrate the need for structured, multi-metric evaluation frameworks when deploying LLMs in high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24968v3">The Impact of LLMs on Online News Consumption and Production</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) change how consumers acquire information online; their bots also crawl news publishers' websites for training data and to answer consumer queries; and they provide tools that can lower the cost of content creation. These changes lead to predictions of adverse impact on news publishers in the form of lowered consumer demand, reduced demand for newsroom employees, and an increase in news "slop." Consequently, some publishers strategically responded by blocking LLM access to their websites using the robots.txt file standard. Using high-frequency granular data, we document four effects related to the predicted shifts in news publishing following the introduction of generative AI (GenAI). First, we find a moderate decline in traffic to news publishers occurring after August 2024. Second, using a difference-in-differences approach, we find that blocking GenAI bots can be associated with a reduction of total website traffic to large publishers compared to not blocking. Third, on the hiring side, we do not find evidence that LLMs are replacing editorial or content-production jobs yet. The share of new editorial and content-production job listings increases over time. Fourth, regarding content production, we find no evidence that large publishers increased text volume; instead, they significantly increased rich content and use more advertising and targeting technologies. Together, these findings provide early evidence of some unforeseen impacts of the introduction of LLMs on news production and consumption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09944v1">Environment-in-the-Loop: Rethinking Code Migration with LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Modern software systems continuously undergo code upgrades to enhance functionality, security, and performance, and Large Language Models (LLMs) have demonstrated remarkable capabilities in code migration tasks. However, while research on automated code migration which including refactoring, API adaptation, and dependency updates has advanced rapidly, the exploration of the automated environment interaction that must accompany it remains relatively scarce. In practice, code and its environment are intricately intertwined. Relying solely on static analysis of the environment leads to an inadequate understanding of the target setting, prolongs feedback cycles, and consequently causes significant rework and project delays, thereby reducing overall efficiency. We contend that successful software evolution demands a holistic perspective that integrates both code and environment migration. To understand the current landscape and challenges, we first provide an overview of the status of automated environment construction. We then propose a novel framework paradigm that tightly integrates automated environment setup with the code migration workflow. Finally, we explore the challenges and future directions for automated environment interaction within the code migration domain. Our findings emphasize that without automated environment interaction, the automation of code migration is only half complete.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07721v2">ParisKV: Fast and Drift-Robust KV-Cache Retrieval for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 25 pages, 16 figures. Under review
    </div>
    <details class="paper-abstract">
      KV-cache retrieval is essential for long-context LLM inference, yet existing methods struggle with distribution drift and high latency at scale. We introduce ParisKV, a drift-robust, GPU-native KV-cache retrieval framework based on collision-based candidate selection, followed by a quantized inner-product reranking estimator. For million-token contexts, ParisKV supports CPU-offloaded KV caches via Unified Virtual Addressing (UVA), enabling on-demand top-$k$ fetching with minimal overhead. ParisKV matches or outperforms full attention quality on long-input and long-generation benchmarks. It achieves state-of-the-art long-context decoding efficiency: it matches or exceeds full attention speed even at batch size 1 for long contexts, delivers up to 2.8$\times$ higher throughput within full attention's runnable range, and scales to million-token contexts where full attention runs out of memory. At million-token scale, ParisKV reduces decode latency by 17$\times$ and 44$\times$ compared to MagicPIG and PQCache, respectively, two state-of-the-art KV-cache Top-$k$ retrieval baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09930v1">JMigBench: A Benchmark for Evaluating LLMs on Source Code Migration (Java 8 to Java 11)</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      We build a benchmark to evaluate large language models (LLMs) for source code migration tasks, specifically upgrading functions from Java 8 to Java 11. We first collected a dataset of function pairs from open-source repositories, but limitations in data quality led us to construct a refined dataset covering eight categories of deprecated APIs. Using this dataset, the Mistral Codestral model was evaluated with CodeBLEU and keyword-based metrics to measure lexical and semantic similarity as well as migration correctness. Results show that the evaluated model (Mistral Codestral) can handle trivial one-to-one API substitutions with moderate success, achieving identical migrations in 11.11% of the cases, but it struggles with more complex migrations such as CORBA or JAX-WS. These findings suggest Mistral Codestral can partially reduce developer effort by automating repetitive migration tasks but cannot yet replace humans within the scope of the JMigBench benchmark. The benchmark and analysis provide a foundation for future work on expanding datasets, refining prompting strategies, and improving migration performance across different LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09924v1">LLMs Encode Their Failures: Predicting Success from Pre-Generation Activations</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Running LLMs with extended reasoning on every problem is expensive, but determining which inputs actually require additional compute remains challenging. We investigate whether their own likelihood of success is recoverable from their internal representations before generation, and if this signal can guide more efficient inference. We train linear probes on pre-generation activations to predict policy-specific success on math and coding tasks, substantially outperforming surface features such as question length and TF-IDF. Using E2H-AMC, which provides both human and model performance on identical problems, we show that models encode a model-specific notion of difficulty that is distinct from human difficulty, and that this distinction increases with extended reasoning. Leveraging these probes, we demonstrate that routing queries across a pool of models can exceed the best-performing model whilst reducing inference cost by up to 70\% on MATH, showing that internal representations enable practical efficiency gains even when they diverge from human intuitions about difficulty. Our code is available at: https://github.com/KabakaWilliam/llms_know_difficulty
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09902v1">Routing, Cascades, and User Choice for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 23 pages, accepted in ICLR 2026
    </div>
    <details class="paper-abstract">
      To mitigate the trade-offs between performance and costs, LLM providers route user tasks to different models based on task difficulty and latency. We study the effect of LLM routing with respect to user behavior. We propose a game between an LLM provider with two models (standard and reasoning) and a user who can re-prompt or abandon tasks if the routed model cannot solve them. The user's goal is to maximize their utility minus the delay from using the model, while the provider minimizes the cost of servicing the user. We solve this Stackelberg game by fully characterizing the user best response and simplifying the provider problem. We observe that in nearly all cases, the optimal routing policy involves a static policy with no cascading that depends on the expected utility of the models to the user. Furthermore, we reveal a misalignment gap between the provider-optimal and user-preferred routes when the user's and provider's rankings of the models with respect to utility and cost differ. Finally, we demonstrate conditions for extreme misalignment where providers are incentivized to throttle the latency of the models to minimize their costs, consequently depressing user utility. The results yield simple threshold rules for single-provider, single-user interactions and clarify when routing, cascading, and throttling help or harm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09901v1">QP-OneModel: A Unified Generative LLM for Multi-Task Query Understanding in Xiaohongshu Search</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Query Processing (QP) bridges user intent and content supply in large-scale Social Network Service (SNS) search engines. Traditional QP systems rely on pipelines of isolated discriminative models (e.g., BERT), suffering from limited semantic understanding and high maintenance overhead. While Large Language Models (LLMs) offer a potential solution, existing approaches often optimize sub-tasks in isolation, neglecting intrinsic semantic synergy and necessitating independent iterations. Moreover, standard generative methods often lack grounding in SNS scenarios, failing to bridge the gap between open-domain corpora and informal SNS linguistic patterns, while struggling to adhere to rigorous business definitions. We present QP-OneModel, a Unified Generative LLM for Multi-Task Query Understanding in the SNS domain. We reformulate heterogeneous sub-tasks into a unified sequence generation paradigm, adopting a progressive three-stage alignment strategy culminating in multi-reward Reinforcement Learning. Furthermore, QP-OneModel generates intent descriptions as a novel high-fidelity semantic signal, effectively augmenting downstream tasks such as query rewriting and ranking. Offline evaluations show QP-OneModel achieves a 7.35% overall gain over discriminative baselines, with significant F1 boosts in NER (+9.01%) and Term Weighting (+9.31%). It also exhibits superior generalization, surpassing a 32B model by 7.60% accuracy on unseen tasks. Fully deployed at Xiaohongshu, online A/B tests confirm its industrial value, optimizing retrieval relevance (DCG) by 0.21% and lifting user retention by 0.044%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.12306v2">A large-scale pipeline for automatic corpus annotation using LLMs: variation and change in the English consider construction</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      As natural language corpora expand at an unprecedented rate, manual annotation remains a significant methodological bottleneck in corpus linguistic work. We address this challenge by presenting a scalable pipeline for automating grammatical annotation in voluminous corpora using large language models (LLMs). Unlike previous supervised and iterative approaches, our method employs a four-phase workflow: prompt engineering, pre-hoc evaluation, automated batch processing, and post-hoc validation. We demonstrate the pipeline's accessibility and effectiveness through a diachronic case study of variation in the English evaluative consider construction (consider X as/to be/zero Y). We annotate 143,933 'consider' concordance lines from the Corpus of Historical American English (COHA) via the OpenAI API in under 60 hours, achieving 98 percent+ accuracy on two sophisticated annotation procedures. A Bayesian multinomial GAM fitted to 44,527 true positives of the evaluative construction reveals previously undocumented genre-specific trajectories of change, enabling us to advance new hypotheses about the relationship between register formality and competing pressures of morphosyntactic reduction and enhancement. Our results suggest that LLMs can perform a range of data preparation tasks at scale with minimal human intervention, unlocking substantive research questions previously beyond practical reach, though implementation requires attention to costs, licensing, and other ethical considerations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09851v1">CoFEH: LLM-driven Feature Engineering Empowered by Collaborative Bayesian Hyperparameter Optimization</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Feature Engineering (FE) is pivotal in automated machine learning (AutoML) but remains a bottleneck for traditional methods, which treat it as a black-box search, operating within rigid, predefined search spaces and lacking domain awareness. While Large Language Models (LLMs) offer a promising alternative by leveraging semantic reasoning to generate unbounded operators, existing methods fail to construct free-form FE pipelines, remaining confined to isolated subtasks such as feature generation. Most importantly, they are rarely optimized jointly with hyperparameter optimization (HPO) of the ML model, leading to greedy "FE-then-HPO" workflows that cannot capture strong FE-HPO interactions. In this paper, we present CoFEH, a collaborative framework that interleaves LLM-based FE and Bayesian HPO for robust end-to-end AutoML. CoFEH uses an LLM-driven FE optimizer powered by Tree of Thought (ToT) to explore flexible FE pipelines, a Bayesian optimization (BO) module to solve HPO, and a dynamic optimizer selector that realizes interleaved optimization by adaptively scheduling FE and HPO steps. Crucially, we introduce a mutual conditioning mechanism that shares context between LLM and BO, enabling mutually informed decisions. Experiments show that CoFEH not only outperforms traditional and LLM-based FE baselines, but also achieves superior end-to-end performance under joint optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08493v2">LLM-based Vulnerable Code Augmentation: Generate or Refactor?</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 15 pages, Accepted by ESAAN 2026, version with added appendix
    </div>
    <details class="paper-abstract">
      Vulnerability code-bases often suffer from severe imbalance, limiting the effectiveness of Deep Learning-based vulnerability classifiers. Data Augmentation could help solve this by mitigating the scarcity of under-represented vulnerability types. In this context, we investigate LLM-based augmentation for vulnerable functions, comparing controlled generation of new vulnerable samples with semantics-preserving refactoring of existing ones. Using Qwen2.5-Coder to produce augmented data and CodeBERT as a classifier on the SVEN dataset, we find that our approaches are indeed effective in enriching vulnerable code-bases through a simple process and with reasonable quality, and that a hybrid strategy best boosts vulnerability classifiers' performance. Code repository is available here : https://github.com/DynaSoumhaneOuchebara/LLM-based-code-augmentation-Generate-or-Refactor-
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09832v1">LLM Reasoning Predicts When Models Are Right: Evidence from Coding Classroom Discourse</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed to automatically label and analyze educational dialogue at scale, yet current pipelines lack reliable ways to detect when models are wrong. We investigate whether reasoning generated by LLMs can be used to predict the correctness of a model's own predictions. We analyze 30,300 teacher utterances from classroom dialogue, each labeled by multiple state-of-the-art LLMs with an instructional move construct and an accompanying reasoning. Using human-verified ground-truth labels, we frame the task as predicting whether a model's assigned label for a given utterance is correct. We encode LLM reasoning using Term Frequency-Inverse Document Frequency (TF-IDF) and evaluate five supervised classifiers. A Random Forest classifier achieves an F1 score of 0.83 (Recall = 0.854), successfully identifying most incorrect predictions and outperforming baselines. Training specialist detectors for specific instructional move constructs further improves performance on difficult constructs, indicating that error detection benefits from construct-specific linguistic cues. Using the Linguistic Inquiry and Word Count (LIWC) framework, we examine four linguistic markers of correctness: Causation, Differentiation, Tentativeness, and Insight. Correct predictions exhibit grounded causal language (e.g., because, therefore), while incorrect reasoning is substantially more likely to rely on epistemic hedging (e.g., might, could) and performative metacognition (e.g., think, realize). Syntactic complexity does not distinguish correct from incorrect reasoning, and longer reasoning is not more reliable. These findings demonstrate that reasoning-based error detection offers a practical and scalable approach to quality control in automated educational dialogue analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09817v1">AnalyticsGPT: An LLM Workflow for Scientometric Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      This paper introduces AnalyticsGPT, an intuitive and efficient large language model (LLM)-powered workflow for scientometric question answering. This underrepresented downstream task addresses the subcategory of meta-scientific questions concerning the "science of science." When compared to traditional scientific question answering based on papers, the task poses unique challenges in the planning phase. Namely, the need for named-entity recognition of academic entities within questions and multi-faceted data retrieval involving scientometric indices, e.g. impact factors. Beyond their exceptional capacity for treating traditional natural language processing tasks, LLMs have shown great potential in more complex applications, such as task decomposition and planning and reasoning. In this paper, we explore the application of LLMs to scientometric question answering, and describe an end-to-end system implementing a sequential workflow with retrieval-augmented generation and agentic concepts. We also address the secondary task of effectively synthesizing the data into presentable and well-structured high-level analyses. As a database for retrieval-augmented generation, we leverage a proprietary research performance assessment platform. For evaluation, we consult experienced subject matter experts and leverage LLMs-as-judges. In doing so, we provide valuable insights on the efficacy of LLMs towards a niche downstream task. Our (skeleton) code and prompts are available at: https://github.com/lyvykhang/llm-agents-scientometric-qa/tree/acl.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.10171v1">EvoCodeBench: A Human-Performance Benchmark for Self-Evolving LLM-Driven Coding Systems</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to advance in programming tasks, LLM-driven coding systems have evolved from one-shot code generation into complex systems capable of iterative improvement during inference. However, existing code benchmarks primarily emphasize static correctness and implicitly assume fixed model capability during inference. As a result, they do not capture inference-time self-evolution, such as whether accuracy and efficiency improve as an agent iteratively refines its solutions. They also provide limited accounting of resource costs and rarely calibrate model performance against that of human programmers. Moreover, many benchmarks are dominated by high-resource languages, leaving cross-language robustness and long-tail language stability underexplored. Therefore, we present EvoCodeBench, a benchmark for evaluating self-evolving LLM-driven coding systems across programming languages with direct comparison to human performance. EvoCodeBench tracks performance dynamics, measuring solution correctness alongside efficiency metrics such as solving time, memory consumption, and improvement algorithmic design over repeated problem-solving attempts. To ground evaluation in a human-centered reference frame, we directly compare model performance with that of human programmers on the same tasks, enabling relative performance assessment within the human ability distribution. Furthermore, EvoCodeBench supports multiple programming languages, enabling systematic cross-language and long-tail stability analyses under a unified protocol. Our results demonstrate that self-evolving systems exhibit measurable gains in efficiency over time, and that human-relative and multi-language analyses provide insights unavailable through accuracy alone. EvoCodeBench establishes a foundation for evaluating coding intelligence in evolving LLM-driven systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.06133v3">LLM Serving Optimization with Variable Prefill and Decode Lengths</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      We study offline scheduling for large language model (LLM) serving under a fixed KV-cache memory budget, where requests have heterogeneous prompt (prefill) and response (decode) lengths. Prompt tokens determine initial KV usage, and each generated token increases memory by one unit. Given a backlog of n requests arriving together, we schedule mixed prefill and decode batches to minimize total end-to-end latency. We show that heterogeneity in prompt lengths makes the problem computationally intractable and that widely used heuristics such as first-come-first-served and shortest-first can be arbitrarily suboptimal. We propose Sorted-F, which repeatedly forms feasible batches using a new selection metric that balances batch size against downstream decode cost, and prove it achieves a constant-factor guarantee on total latency. We further develop practical variants -- an exact solver for small instances and fast heuristics for larger ones -- and evaluate them on a public workload spanning short conversations and long-document summarization, where they consistently reduce average latency relative to standard baselines. Our results highlight that during peak-hour tidal backlogs, greedy GPU packing or short-request prioritization can perform poorly when prompt lengths vary widely, and provide a principled, tunable framework for designing production batch schedulers and planning capacity in memory-constrained LLM serving systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.11362v2">PersonaX: Multimodal Datasets with LLM-Inferred Behavior Traits</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Understanding human behavior traits is central to applications in human-computer interaction, computational social science, and personalized AI systems. Such understanding often requires integrating multiple modalities to capture nuanced patterns and relationships. However, existing resources rarely provide datasets that combine behavioral descriptors with complementary modalities such as facial attributes and biographical information. To address this gap, we present PersonaX, a curated collection of multimodal datasets designed to enable comprehensive analysis of public traits across modalities. PersonaX consists of (1) CelebPersona, featuring 9444 public figures from diverse occupations, and (2) AthlePersona, covering 4181 professional athletes across 7 major sports leagues. Each dataset includes behavioral trait assessments inferred by three high-performing large language models, alongside facial imagery and structured biographical features. We analyze PersonaX at two complementary levels. First, we abstract high-level trait scores from text descriptions and apply five statistical independence tests to examine their relationships with other modalities. Second, we introduce a novel causal representation learning (CRL) framework tailored to multimodal and multi-measurement data, providing theoretical identifiability guarantees. Experiments on both synthetic and real-world data demonstrate the effectiveness of our approach. By unifying structured and unstructured analysis, PersonaX establishes a foundation for studying LLM-inferred behavioral traits in conjunction with visual and biographical attributes, advancing multimodal trait analysis and causal reasoning. The code is available at https://github.com/lokali/PersonaX.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09719v1">Unsupervised Layer-Wise Dynamic Test Time Adaptation for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Test-time adaptation (TTA) for large language models (LLMs) updates model parameters at inference time using signals available at deployment. This paper focuses on a common yet under-explored regime: unsupervised, sample-specific TTA, where the model adapts independently for each prompt using only the prompt itself, without gold answers or external supervision. Although appealing, naive unsupervised TTA with a fixed, handcrafted learning rate can be unstable: updates may overfit to prompt-specific statistics, drift from the desired answer distribution, and ultimately degrade generation quality. This failure mode is not surprising, as in this case TTA must adapt to a single prompt within only a few gradient steps, unlike standard training that averages updates over large datasets and long optimization horizons. Therefore, we propose layer-wise dynamic test-time adaptation, a framework which explicitly modulates TTA strength as a function of prompt representation, LLM structure and adaptation step. In our setting, TTA updates only LoRA parameters, and a lightweight hypernetwork predicts per-layer, per-step learning-rate multipliers, enabling fine-grained control. Experiments across various datasets and LLMs consistently show that our method substantially strengthens TTA by learning effective scaling patterns over adaptation steps and transformer layer projections, improving stability while delivering better performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.23631v2">Beyond Pairwise: Empowering LLM Alignment With Ranked Choice Modeling</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Accepted by The Fourteenth International Conference on Learning Representations (ICLR 2026)
    </div>
    <details class="paper-abstract">
      Alignment of large language models (LLMs) has predominantly relied on pairwise preference optimization, where annotators select the better of two responses to a prompt. While simple, this approach overlooks the opportunity to learn from richer forms of human feedback, such as multiway comparisons and top-$k$ rankings. We introduce Ranked Choice Preference Optimization (RCPO), a unified framework that bridges preference optimization with (ranked) choice modeling via maximum likelihood estimation. RCPO supports both utility-based and rank-based models, subsumes several pairwise methods (such as DPO and SimPO) as special cases, and provides principled training objectives for richer feedback formats. We instantiate this framework with two representative models (Multinomial Logit and Mallows-RMJ). Experiments on Llama-3-8B-Instruct, Gemma-2-9B-it, and Mistral-7B-Instruct across in-distribution and out-of-distribution settings show that RCPO consistently outperforms competitive baselines. RCPO shows that directly leveraging ranked preference data, combined with the right choice models, yields more effective alignment. It offers an extensible foundation for incorporating (ranked) choice modeling into LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09703v1">Maastricht University at AMIYA: Adapting LLMs for Dialectal Arabic using Fine-tuning and MBR Decoding</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming increasingly multilingual, supporting hundreds of languages, especially high resource ones. Unfortunately, Dialect variations are still underrepresented due to limited data and linguistic variation. In this work, we adapt a pre-trained LLM to improve dialectal performance. Specifically, we use Low Rank Adaptation (LoRA) fine-tuning on monolingual and English Dialect parallel data, adapter merging and dialect-aware MBR decoding to improve dialectal fidelity generation and translation. Experiments on Syrian, Moroccan, and Saudi Arabic show that merging and MBR improve dialectal fidelity while preserving semantic accuracy. This combination provides a compact and effective framework for robust dialectal Arabic generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.10155v3">What Should Feature Distillation Transfer in LLMs? A Task-Tangent Geometry View</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Feature-based knowledge distillation aims to transfer intermediate representations from a teacher LLM model to a student. Existing approaches typically rely on direct feature matching or learned projections, implicitly treating representations as objects with intrinsic meaning. However, the relevance of a representation dimension is determined solely by how it affects the model's output. In this work, we propose a functional perspective on feature-based distillation. We characterize knowledge transfer in terms of the teacher's functional geometry, i.e., how its output depends on internal representations, rather than direct representation alignment. This viewpoint reveals that effective distillation need not preserve full high-dimensional features, but instead should retain dominant directions of functional contribution, naturally inducing an effective functional dimension for each task. Building on this framework, we introduce Flex-KD, an architecture-agnostic and parameter-free distillation method that transfers the teacher's functional geometry while matching the student's representational capacity. Extensive experiments across language understanding and generation benchmarks demonstrate that Flex-KD consistently outperforms existing distillation approaches, particularly under severe teacher-student dimension mismatch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09634v1">LLM-FS: Zero-Shot Feature Selection for Effective and Interpretable Malware Detection</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Feature selection (FS) remains essential for building accurate and interpretable detection models, particularly in high-dimensional malware datasets. Conventional FS methods such as Extra Trees, Variance Threshold, Tree-based models, Chi-Squared tests, ANOVA, Random Selection, and Sequential Attention rely primarily on statistical heuristics or model-driven importance scores, often overlooking the semantic context of features. Motivated by recent progress in LLM-driven FS, we investigate whether large language models (LLMs) can guide feature selection in a zero-shot setting, using only feature names and task descriptions, as a viable alternative to traditional approaches. We evaluate multiple LLMs (GPT-5.0, GPT-4.0, Gemini-2.5 etc.) on the EMBOD dataset (a fusion of EMBER and BODMAS benchmark datasets), comparing them against established FS methods across several classifiers, including Random Forest, Extra Trees, MLP, and KNN. Performance is assessed using accuracy, precision, recall, F1, AUC, MCC, and runtime. Our results demonstrate that LLM-guided zero-shot feature selection achieves competitive performance with traditional FS methods while offering additional advantages in interpretability, stability, and reduced dependence on labeled data. These findings position zero-shot LLM-based FS as a promising alternative strategy for effective and interpretable malware detection, paving the way for knowledge-guided feature selection in security-critical applications
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09629v1">Stop Testing Attacks, Start Diagnosing Defenses: The Four-Checkpoint Framework Reveals Where LLM Safety Breaks</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 17 pages, pre-print
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deploy safety mechanisms to prevent harmful outputs, yet these defenses remain vulnerable to adversarial prompts. While existing research demonstrates that jailbreak attacks succeed, it does not explain \textit{where} defenses fail or \textit{why}. To address this gap, we propose that LLM safety operates as a sequential pipeline with distinct checkpoints. We introduce the \textbf{Four-Checkpoint Framework}, which organizes safety mechanisms along two dimensions: processing stage (input vs.\ output) and detection level (literal vs.\ intent). This creates four checkpoints, CP1 through CP4, each representing a defensive layer that can be independently evaluated. We design 13 evasion techniques, each targeting a specific checkpoint, enabling controlled testing of individual defensive layers. Using this framework, we evaluate GPT-5, Claude Sonnet 4, and Gemini 2.5 Pro across 3,312 single-turn, black-box test cases. We employ an LLM-as-judge approach for response classification and introduce Weighted Attack Success Rate (WASR), a severity-adjusted metric that captures partial information leakage overlooked by binary evaluation. Our evaluation reveals clear patterns. Traditional Binary ASR reports 22.6\% attack success. However, WASR reveals 52.7\%, a 2.3$\times$ higher vulnerability. Output-stage defenses (CP3, CP4) prove weakest at 72--79\% WASR, while input-literal defenses (CP1) are strongest at 13\% WASR. Claude achieves the strongest safety (42.8\% WASR), followed by GPT-5 (55.9\%) and Gemini (59.5\%). These findings suggest that current defenses are strongest at input-literal checkpoints but remain vulnerable to intent-level manipulation and output-stage techniques. The Four-Checkpoint Framework provides a structured approach for identifying and addressing safety vulnerabilities in deployed systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.10936v3">Cochain: Balancing Insufficient and Excessive Collaboration in LLM Agent Workflows</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 35 pages, 23 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performance in executing complex reasoning tasks. Chain-of-thought effectively enhances reasoning capabilities by unlocking the potential of large models, while multi-agent systems provide more comprehensive solutions by integrating the collective intelligence of multiple agents. However, both approaches face significant limitations. Single-agent with chain-of-thought, due to the inherent complexity of designing cross-domain prompts, faces collaboration challenges. Meanwhile, multi-agent systems consume substantial tokens and inevitably dilute the primary problem, which is particularly problematic in business workflow tasks. To address these challenges, we propose Cochain, a collaboration prompting framework that effectively solves the business workflow collaboration problem by combining knowledge and prompts at a reduced cost. Specifically, we construct an integrated knowledge graph that incorporates knowledge from multiple stages. Furthermore, by maintaining and retrieving a prompts tree, we can obtain prompt information relevant to other stages of the business workflow. We perform extensive evaluations of Cochain across multiple datasets, demonstrating that Cochain outperforms all baselines in both prompt engineering and multi-agent LLMs. Additionally, expert evaluation results indicate that the use of a small model in combination with Cochain outperforms GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09624v1">MILE-RefHumEval: A Reference-Free, Multi-Independent LLM Framework for Human-Aligned Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      We introduce MILE-RefHumEval, a reference-free framework for evaluating Large Language Models (LLMs) without ground-truth annotations or evaluator coordination. It leverages an ensemble of independently prompted evaluators guided by a human-aligned schema, supporting both discrete and continuous scoring judgement. With task-specific prompts from best candidate selection, summarization and image captioning to dialogue, MILE-RefHumEval provides flexible, interpretable, and scalable assessments. Experiments show it aligns closely with human judgments, outperforms prior methods, and reduces computational overhead, offering an efficient, robust, and human-aligned solution for real-world LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.08060v2">Compiler-Assisted Speculative Sampling for Accelerated LLM Inference on Heterogeneous Edge Devices</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Accepted to AccML@HiPEAC 2026
    </div>
    <details class="paper-abstract">
      LLM deployment on resource-constrained edge devices faces severe latency constraints, particularly in real-time applications where delayed responses can compromise safety or usability. Among many approaches to mitigate the inefficiencies of sequential token-by-token generation, Speculative Decoding (SD) has emerged as a promising technique. However, SD at the edge is hindered by two major challenges: (1) integrating SD into a compiler-based workflow without sacrificing performance or programmability, and (2) exploiting the heterogeneous compute resources of modern SoCs through carefully designed partitioning strategies. This work addresses these challenges by using an analytical cost model that explores heterogeneous hardware configurations and guides coarse-grained partitioning of LLM subgraphs, particularly with edge-typical short input sequence lengths. The cost model predicts when speculative sampling and heterogeneous execution are jointly beneficial and is validated on an edge device featuring a hexacore Cortex-A CPU and a Mali GPU, revealing up to 1.68$\times$ speedup for translation tasks, closely matching analytic expectations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09598v1">Learning from the Irrecoverable: Error-Localized Policy Optimization for Tool-Integrated LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 20 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Tool-integrated reasoning (TIR) enables LLM agents to solve tasks through planning, tool use, and iterative revision, but outcome-only reinforcement learning in this setting suffers from sparse, delayed rewards and weak step-level credit assignment. In long-horizon TIR trajectories, an early irrecoverable mistake can determine success or failure, making it crucial to localize the first irrecoverable step and leverage it for fine-grained credit assignment. We propose Error-Localized Policy Optimization (ELPO), which localizes the first irrecoverable step via binary-search rollout trees under a fixed rollout budget, converts the resulting tree into stable learning signals through hierarchical advantage attribution, and applies error-localized adaptive clipping to strengthen corrective updates on the critical step and its suffix. Across TIR benchmarks in math, science QA, and code execution, ELPO consistently outperforms strong Agentic RL baselines under comparable sampling budgets, with additional gains in Pass@K and Major@K scaling, rollout ranking quality, and tool-call efficiency. Our code will be publicly released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09578v1">Rollout-Training Co-Design for Efficient LLM-Based Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Despite algorithm-level innovations for multi-agent reinforcement learning (MARL), the underlying networked infrastructure for large-scale MARL training remains underexplored. Existing training frameworks primarily optimize for single-agent scenarios and fail to address the unique system-level challenges of MARL, including rollout-training synchronization barriers, rollout load imbalance, and training resource underutilization. To bridge this gap, we propose FlexMARL, the first end-to-end training framework that holistically optimizes rollout, training, and their orchestration for large-scale LLM-based MARL. Specifically, FlexMARL introduces the joint orchestrator to manage data flow under the rollout-training disaggregated architecture. Building upon the experience store, a novel micro-batch driven asynchronous pipeline eliminates the synchronization barriers while providing strong consistency guarantees. Rollout engine adopts a parallel sampling scheme combined with hierarchical load balancing, which adapts to skewed inter/intra-agent request patterns. Training engine achieves on-demand hardware binding through agent-centric resource allocation. The training states of different agents are swapped via unified and location-agnostic communication. Empirical results on a large-scale production cluster demonstrate that FlexMARL achieves up to 7.3x speedup and improves hardware utilization by up to 5.6x compared to existing frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.09574v1">Aligning Tree-Search Policies with Fixed Token Budgets in Test-Time Scaling of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Tree-search decoding is an effective form of test-time scaling for large language models (LLMs), but real-world deployment imposes a fixed per-query token budget that varies across settings. Existing tree-search policies are largely budget-agnostic, treating the budget as a termination condition, which can lead to late-stage over-branching or premature termination. We propose {Budget-Guided MCTS} (BG-MCTS), a tree-search decoding algorithm that aligns its search policy with the remaining token budget: it starts with broad exploration, then prioritizes refinement and answer completion as the budget depletes while reducing late-stage branching from shallow nodes. BG-MCTS consistently outperforms budget-agnostic tree-search baselines across different budgets on MATH500 and AIME24/25 with open-weight LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.11097v2">The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs, offering faster inference and greater interactivity via parallel decoding and bidirectional modeling. However, despite strong performance in code generation and text infilling, we identify a fundamental safety concern: existing alignment mechanisms fail to safeguard dLLMs against context-aware, masked-input adversarial prompts, exposing novel vulnerabilities. To this end, we present DIJA, the first systematic study and jailbreak attack framework that exploits unique safety weaknesses of dLLMs. Specifically, our proposed DIJA constructs adversarial interleaved mask-text prompts that exploit the text generation mechanisms of dLLMs, i.e., bidirectional modeling and parallel decoding. Bidirectional modeling drives the model to produce contextually consistent outputs for masked spans, even when harmful, while parallel decoding limits model dynamic filtering and rejection sampling of unsafe content. This causes standard alignment mechanisms to fail, enabling harmful completions in alignment-tuned dLLMs, even when harmful behaviors or unsafe instructions are directly exposed in the prompt. Through comprehensive experiments, we demonstrate that DIJA significantly outperforms existing jailbreak methods, exposing a previously overlooked threat surface in dLLM architectures. Notably, our method achieves up to 100% keyword-based ASR on Dream-Instruct, surpassing the strongest prior baseline, ReNeLLM, by up to 78.5% in evaluator-based ASR on JailbreakBench and by 37.7 points in StrongREJECT score, while requiring no rewriting or hiding of harmful content in the jailbreak prompt. Our findings underscore the urgent need for rethinking safety alignment in this emerging class of language models. Code is available at https://github.com/ZichenWen1/DIJA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07543v2">MSP-LLM: A Unified Large Language Model Framework for Complete Material Synthesis Planning</a></div>
    <div class="paper-meta">
      📅 2026-02-10
    </div>
    <details class="paper-abstract">
      Material synthesis planning (MSP) remains a fundamental and underexplored bottleneck in AI-driven materials discovery, as it requires not only identifying suitable precursor materials but also designing coherent sequences of synthesis operations to realize a target material. Although several AI-based approaches have been proposed to address isolated subtasks of MSP, a unified methodology for solving the entire MSP task has yet to be established. We propose MSP-LLM, a unified LLM-based framework that formulates MSP as a structured process composed of two constituent subproblems: precursor prediction (PP) and synthesis operation prediction (SOP). Our approach introduces a discrete material class as an intermediate decision variable that organizes both tasks into a chemically consistent decision chain. For OP, we further incorporate hierarchical precursor types as synthesis-relevant inductive biases and employ an explicit conditioning strategy that preserves precursor-related information in the autoregressive decoding state. Extensive experiments show that MSP-LLM consistently outperforms existing methods on both PP and SOP, as well as on the complete MSP task, demonstrating an effective and scalable framework for MSP that can accelerate real-world materials discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.16276v4">Mechanism Design for LLM Fine-tuning with Multiple Reward Models</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) to aggregate multiple preferences has attracted considerable research attention. With aggregation algorithms advancing, a potential economic scenario arises where fine-tuning services are provided to agents with different preferences. In this context, agents may benefit from strategically misreporting their preferences, but this could harm the aggregation performance. This paper addresses such incentive issues by framing it as a mechanism design problem: an LLM provider determines the fine-tuning objective (training rule) and the pricing scheme (payment rule) for agents. We primarily focus on training rules that maximize social welfare subject to certain regularizations, referred to as SW-Max rules. First, we show that under most circumstances, truthful reporting is sub-optimal with simply a SW-Max rule, thereby highlighting the necessity of payments. Second, we extend the VCG payment to implement SW-Max rules in dominant-strategy incentive compatibility (DSIC). We characterize sufficient conditions for payment equivalence and derive the necessary conditions for a payment rule to implement a SW-Max rule in DSIC and other principles. Third, we demonstrate that our mechanism is approximately DSIC with perturbed input, showcasing its robustness against the inevitable errors in real-world applications. Experiments on real LLM training results further confirm the practical implications of our results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.17684v2">Can LLMs Automate Fact-Checking Article Writing?</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Accepted to TACL 2026, pre-MIT Press publication version
    </div>
    <details class="paper-abstract">
      Automatic fact-checking aims to support professional fact-checkers by offering tools that can help speed up manual fact-checking. Yet, existing frameworks fail to address the key step of producing output suitable for broader dissemination to the general public: while human fact-checkers communicate their findings through fact-checking articles, automated systems typically produce little or no justification for their assessments. Here, we aim to bridge this gap. In particular, we argue for the need to extend the typical automatic fact-checking pipeline with automatic generation of full fact-checking articles. We first identify key desiderata for such articles through a series of interviews with experts from leading fact-checking organizations. We then develop QRAFT, an LLM-based agentic framework that mimics the writing workflow of human fact-checkers. Finally, we assess the practical usefulness of QRAFT through human evaluations with professional fact-checkers. Our evaluation shows that while QRAFT outperforms several previously proposed text-generation approaches, it lags considerably behind expert-written articles. We hope that our work will enable further research in this new and important direction. The code for our implementation is available at https://github.com/mbzuai-nlp/qraft.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17653v2">GeoGramBench: Benchmarking the Geometric Program Reasoning in Modern LLMs</a></div>
    <div class="paper-meta">
      📅 2026-02-10
      | 💬 Accepted to ICLR 2026
    </div>
    <details class="paper-abstract">
      Geometric spatial reasoning forms the foundation of many applications in artificial intelligence, yet the ability of large language models (LLMs) to operate over geometric spatial information expressed in procedural code remains underexplored. In this paper, we address this gap by formalizing the Program-to-Geometry task, which challenges models to translate programmatic drawing code into accurate and abstract geometric reasoning. To evaluate this capability, we present GeoGramBench, a benchmark of 500 carefully refined problems organized by a tailored three-level taxonomy that considers geometric complexity rather than traditional mathematical reasoning complexity. Our comprehensive evaluation of 17 frontier LLMs reveals consistent and pronounced deficiencies: even the most advanced models achieve less than 50% accuracy at the highest abstraction level. These results highlight the unique challenges posed by program-driven spatial reasoning and establish GeoGramBench as a valuable resource for advancing research in symbolic-to-spatial geometric reasoning. Project page: https://github.com/LiAuto-DSR/GeoGramBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06038v1">CommCP: Efficient Multi-Agent Coordination via LLM-Based Communication with Conformal Prediction</a></div>
    <div class="paper-meta">
      📅 2026-02-05
      | 💬 IEEE International Conference on Robotics and Automation (ICRA 2026); Project Website: https://comm-cp.github.io/
    </div>
    <details class="paper-abstract">
      To complete assignments provided by humans in natural language, robots must interpret commands, generate and answer relevant questions for scene understanding, and manipulate target objects. Real-world deployments often require multiple heterogeneous robots with different manipulation capabilities to handle different assignments cooperatively. Beyond the need for specialized manipulation skills, effective information gathering is important in completing these assignments. To address this component of the problem, we formalize the information-gathering process in a fully cooperative setting as an underexplored multi-agent multi-task Embodied Question Answering (MM-EQA) problem, which is a novel extension of canonical Embodied Question Answering (EQA), where effective communication is crucial for coordinating efforts without redundancy. To address this problem, we propose CommCP, a novel LLM-based decentralized communication framework designed for MM-EQA. Our framework employs conformal prediction to calibrate the generated messages, thereby minimizing receiver distractions and enhancing communication reliability. To evaluate our framework, we introduce an MM-EQA benchmark featuring diverse, photo-realistic household scenarios with embodied questions. Experimental results demonstrate that CommCP significantly enhances the task success rate and exploration efficiency over baselines. The experiment videos, code, and dataset are available on our project website: https://comm-cp.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.06008v1">AgenticPay: A Multi-Agent LLM Negotiation System for Buyer-Seller Transactions</a></div>
    <div class="paper-meta">
      📅 2026-02-05
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents are increasingly expected to negotiate, coordinate, and transact autonomously, yet existing benchmarks lack principled settings for evaluating language-mediated economic interaction among multiple agents. We introduce AgenticPay, a benchmark and simulation framework for multi-agent buyer-seller negotiation driven by natural language. AgenticPay models markets in which buyers and sellers possess private constraints and product-dependent valuations, and must reach agreements through multi-round linguistic negotiation rather than numeric bidding alone. The framework supports a diverse suite of over 110 tasks ranging from bilateral bargaining to many-to-many markets, with structured action extraction and metrics for feasibility, efficiency, and welfare. Benchmarking state-of-the-art proprietary and open-weight LLMs reveals substantial gaps in negotiation performance and highlights challenges in long-horizon strategic reasoning, establishing AgenticPay as a foundation for studying agentic commerce and language-based market interaction. Code and dataset are available at the link: https://github.com/SafeRL-Lab/AgenticPay.
    </details>
</div>
