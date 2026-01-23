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
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- Part 11

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00912v1">The Discovery Gap: How Product Hunt Startups Vanish in LLM Organic Discovery Queries</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 20 pages, 7 figures. Based on M.Tech thesis research, Indian Institute of Technology Patna, 2025
    </div>
    <details class="paper-abstract">
      When someone asks ChatGPT to recommend a project management tool, which products show up in the response? And more importantly for startup founders: will their newly launched product ever appear? This research set out to answer these questions. I randomly selected 112 startups from the top 500 products featured on the 2025 Product Hunt leaderboard and tested each one across 2,240 queries to two different large language models: ChatGPT (gpt-4o-mini) and Perplexity (sonar with web search). The results were striking. When users asked about products by name, both LLMs recognized them almost perfectly: 99.4% for ChatGPT and 94.3% for Perplexity. But when users asked discovery-style questions like "What are the best AI tools launched this year?" the success rates collapsed to 3.32% and 8.29% respectively. That's a gap of 30-to-1 for ChatGPT. Perhaps the most surprising finding was that Generative Engine Optimization (GEO), the practice of optimizing website content for AI visibility, showed no correlation with actual discovery rates. Products with high GEO scores were no more likely to appear in organic queries than products with low scores. What did matter? For Perplexity, traditional SEO signals like referring domains (r = +0.319, p < 0.001) and Product Hunt ranking (r = -0.286, p = 0.002) predicted visibility. After cleaning the Reddit data for false positives, community presence also emerged as significant (r = +0.395, p = 0.002). The practical takeaway is counterintuitive: don't optimize for AI discovery directly. Instead, build the SEO foundation first and LLM visibility will follow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02407v1">Evolving Personalities in Chaos: An LLM-Augmented Framework for Character Discovery in the Iterated Prisoners Dilemma under Environmental Stress</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 10 pages, 5 figures. Project assignment; exploratory study on LLM-based adaptive agents
    </div>
    <details class="paper-abstract">
      Standard simulations of the Iterated Prisoners Dilemma (IPD) operate in deterministic, noise-free environments, producing strategies that may be theoretically optimal but fragile when confronted with real-world uncertainty. This paper addresses two critical gaps in evolutionary game theory research: (1) the absence of realistic environmental stressors during strategy evolution, and (2) the Interpretability Gap, where evolved genetic strategies remain opaque binary sequences devoid of semantic meaning. We introduce a novel framework combining stochastic environmental perturbations (God Mode) with Large Language Model (LLM)-based behavioral profiling to transform evolved genotypes into interpretable character archetypes. Our experiments demonstrate that strategies evolved under chaotic conditions exhibit superior resilience and present distinct behavioral phenotypes, ranging from Ruthless Capitalists to Diplomatic Enforcers. These phenotypes are readily classified by LLMs but remain nearly impossible to interpret through manual genome inspection alone. This work bridges evolutionary computation with explainable AI and provides a template for automated agent characterization in multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00509v1">Improving LLM-Assisted Secure Code Generation through Retrieval-Augmented-Generation and Multi-Tool Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can generate code but often introduce security vulnerabilities, logical inconsistencies, and compilation errors. Prior work demonstrates that LLMs benefit substantially from structured feedback, static analysis, retrieval augmentation, and execution-based refinement. We propose a retrieval-augmented, multi-tool repair workflow in which a single code-generating LLM iteratively refines its outputs using compiler diagnostics, CodeQL security scanning, and KLEE symbolic execution. A lightweight embedding model is used for semantic retrieval of previously successful repairs, providing security-focused examples that guide generation. Evaluated on a combined dataset of 3,242 programs generated by DeepSeek-Coder-1.3B and CodeLlama-7B, the system demonstrates significant improvements in robustness. For DeepSeek, security vulnerabilities were reduced by 96%. For the larger CodeLlama model, the critical security defect rate was decreased from 58.55% to 22.19%, highlighting the efficacy of tool-assisted self-repair even on "stubborn" models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00469v1">DSL or Code? Evaluating the Quality of LLM-Generated Algebraic Specifications: A Case Study in Optimization at Kinaxis</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 Accepted for publication in ICSE-SEIP 2026
    </div>
    <details class="paper-abstract">
      Model-driven engineering (MDE) provides abstraction and analytical rigour, but industrial adoption in many domains has been limited by the cost of developing and maintaining models. Large language models (LLMs) can help shift this cost balance by supporting direct generation of models from natural-language (NL) descriptions. For domain-specific languages (DSLs), however, LLM-generated models may be less accurate than LLM-generated code in mainstream languages such as Python, due to the latter's dominance in LLM training corpora. We investigate this issue in mathematical optimization, with AMPL, a DSL with established industrial use. We introduce EXEOS, an LLM-based approach that derives AMPL models and Python code from NL problem descriptions and iteratively refines them with solver feedback. Using a public optimization dataset and real-world supply-chain cases from our industrial partner Kinaxis, we evaluate generated AMPL models against Python code in terms of executability and correctness. An ablation study with two LLM families shows that AMPL is competitive with, and sometimes better than, Python, and that our design choices in EXEOS improve the quality of generated specifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02497v2">Towards Bridging Language Gaps in OSS with LLM-Driven Documentation Translation</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      While open source communities attract diverse contributors across the globe, only a few open source software repositories provide essential documentation, such as ReadMe or CONTRIBUTING files, in languages other than English. Recently, large language models (LLMs) have demonstrated remarkable capabilities in a variety of software engineering tasks. We have also seen advances in the use of LLMs for translations in other domains and contexts. Despite this progress, little is known regarding the capabilities of LLMs in translating open-source technical documentation, which is often a mixture of natural language, code, URLs, and markdown formatting. To better understand the need and potential for LLMs to support translation of technical documentation in open source, we conducted an empirical evaluation of translation activity and translation capabilities of two powerful large language models (OpenAI ChatGPT 4 and Anthropic Claude). We found that translation activity is often community-driven and most frequent in larger repositories. A comparison of LLM performance as translators and evaluators of technical documentation suggests LLMs can provide accurate semantic translations but may struggle preserving structure and technical content. These findings highlight both the promise and the challenges of LLM-assisted documentation internationalization and provide a foundation towards automated LLM-driven support for creating and maintaining open source documentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00411v1">Do LLMs Judge Distantly Supervised Named Entity Labels Well? Constructing the JudgeWEL Dataset</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      We present judgeWEL, a dataset for named entity recognition (NER) in Luxembourgish, automatically labelled and subsequently verified using large language models (LLM) in a novel pipeline. Building datasets for under-represented languages remains one of the major bottlenecks in natural language processing, where the scarcity of resources and linguistic particularities make large-scale annotation costly and potentially inconsistent. To address these challenges, we propose and evaluate a novel approach that leverages Wikipedia and Wikidata as structured sources of weak supervision. By exploiting internal links within Wikipedia articles, we infer entity types based on their corresponding Wikidata entries, thereby generating initial annotations with minimal human intervention. Because such links are not uniformly reliable, we mitigate noise by employing and comparing several LLMs to identify and retain only high-quality labelled sentences. The resulting corpus is approximately five times larger than the currently available Luxembourgish NER dataset and offers broader and more balanced coverage across entity categories, providing a substantial new resource for multilingual and low-resource NER research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.04677v3">LLM Query Scheduling with Prefix Reuse and Latency Constraints</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      The efficient deployment of large language models (LLMs) in online settings requires optimizing inference performance under stringent latency constraints, particularly the time-to-first-token (TTFT) and time-per-output-token (TPOT). This paper focuses on the query scheduling problem for LLM inference with prefix reuse, a technique that leverages shared prefixes across queries to reduce computational overhead. Our work reveals previously unknown limitations of the existing first-come-first-serve (FCFS) and longest-prefix-match (LPM) scheduling strategies with respect to satisfying latency constraints. We present a formal theoretical framework for LLM query scheduling under RadixAttention, a prefix reuse mechanism that stores and reuses intermediate representations in a radix tree structure. Our analysis establishes the NP-hardness of the scheduling problem with prefix reuse under TTFT constraints and proposes a novel scheduling algorithm, $k$-LPM, which generalizes existing methods by balancing prefix reuse and fairness in query processing. Theoretical guarantees demonstrate that $k$-LPM achieves improved TTFT performance under realistic traffic patterns captured by a data generative model. Empirical evaluations in a realistic serving setting validates our findings, showing significant reductions in P99 TTFT compared to baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00397v1">Revati: Transparent GPU-Free Time-Warp Emulation for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      Deploying LLMs efficiently requires testing hundreds of serving configurations, but evaluating each one on a GPU cluster takes hours and costs thousands of dollars. Discrete-event simulators are faster and cheaper, but they require re-implementing the serving system's control logic -- a burden that compounds as frameworks evolve. We present Revati, a time-warp emulator that enables performance modeling by directly executing real serving system code at simulation-like speed. The system intercepts CUDA API calls to virtualize device management, allowing serving frameworks to run without physical GPUs. Instead of executing GPU kernels, it performs time jumps -- fast-forwarding virtual time by predicted kernel durations. We propose a coordination protocol that synchronizes these jumps across distributed processes while preserving causality. On vLLM and SGLang, Revati achieves less than 5% prediction error across multiple models and parallelism configurations, while running 5-17x faster than real GPU execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06781v2">From Description to Score: Can LLMs Quantify Vulnerabilities?</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Manual vulnerability scoring, such as assigning Common Vulnerability Scoring System (CVSS) scores, is a resource-intensive process that is often influenced by subjective interpretation. This study investigates the potential of general-purpose large language models (LLMs), namely ChatGPT, Llama, Grok, DeepSeek, and Gemini, to automate this process by analyzing over 31{,}000 recent Common Vulnerabilities and Exposures (CVE) entries. The results show that LLMs substantially outperform the baseline on certain metrics (e.g., \textit{Availability Impact}), while offering more modest gains on others (e.g., \textit{Attack Complexity}). Moreover, model performance varies across both LLM families and individual CVSS metrics, with ChatGPT-5 attaining the highest precision. Our analysis reveals that LLMs tend to misclassify many of the same CVEs, and ensemble-based meta-classifiers only marginally improve performance. Further examination shows that CVE descriptions often lack critical context or contain ambiguous phrasing, which contributes to systematic misclassifications. These findings underscore the importance of enhancing vulnerability descriptions and incorporating richer contextual details to support more reliable automated reasoning and alleviate the growing backlog of CVEs awaiting triage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22385v2">LLM-Guided Exemplar Selection for Few-Shot Wearable-Sensor Human Activity Recognition</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 This paper has been accepted for presentation at ABC 2026. The manuscript is under revision prior to camera-ready submission
    </div>
    <details class="paper-abstract">
      In this paper, we propose an LLM-Guided Exemplar Selection framework to address a key limitation in state-of-the-art Human Activity Recognition (HAR) methods: their reliance on large labeled datasets and purely geometric exemplar selection, which often fail to distinguish similar wearable sensor activities such as walking, walking upstairs, and walking downstairs. Our method incorporates semantic reasoning via an LLM-generated knowledge prior that captures feature importance, inter-class confusability, and exemplar budget multipliers, and uses it to guide exemplar scoring and selection. These priors are combined with margin-based validation cues, PageRank centrality, hubness penalization, and facility-location optimization to obtain a compact and informative set of exemplars. Evaluated on the UCI-HAR dataset under strict few-shot conditions, the framework achieves a macro F1-score of 88.78%, outperforming classical approaches such as random sampling, herding, and k-center. The results show that LLM-derived semantic priors, when integrated with structural and geometric cues, provide a stronger foundation for selecting representative sensor exemplars in few-shot wearable-sensor HAR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13788v2">Scaling Patterns in Adversarial Alignment: Evidence from Multi-LLM Jailbreak Experiments</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly operate in multi-agent and safety-critical settings, raising open questions about how their vulnerabilities scale when models interact adversarially. This study examines whether larger models can systematically jailbreak smaller ones - eliciting harmful or restricted behavior despite alignment safeguards. Using standardized adversarial tasks from JailbreakBench, we simulate over 6,000 multi-turn attacker-target exchanges across major LLM families and scales (0.6B-120B parameters), measuring both harm score and refusal behavior as indicators of adversarial potency and alignment integrity. Each interaction is evaluated through aggregated harm and refusal scores assigned by three independent LLM judges, providing a consistent, model-based measure of adversarial outcomes. Aggregating results across prompts, we find a strong and statistically significant correlation between mean harm and the logarithm of the attacker-to-target size ratio (Pearson r = 0.51, p < 0.001; Spearman rho = 0.52, p < 0.001), indicating that relative model size correlates with the likelihood and severity of harmful completions. Mean harm score variance is higher across attackers (0.18) than across targets (0.10), suggesting that attacker-side behavioral diversity contributes more to adversarial outcomes than target susceptibility. Attacker refusal frequency is strongly and negatively correlated with harm (rho = -0.93, p < 0.001), showing that attacker-side alignment mitigates harmful responses. These findings reveal that size asymmetry influences robustness and provide exploratory evidence for adversarial scaling patterns, motivating more controlled investigations into inter-model alignment and safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00268v1">Beyond Perfect APIs: A Comprehensive Evaluation of LLM Agents Under Real-World API Complexity</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      We introduce WildAGTEval, a benchmark designed to evaluate large language model (LLM) agents' function-calling capabilities under realistic API complexity. Unlike prior work that assumes an idealized API system and disregards real-world factors such as noisy API outputs, WildAGTEval accounts for two dimensions of real-world complexity: 1. API specification, which includes detailed documentation and usage constraints, and 2. API execution, which captures runtime challenges. Consequently, WildAGTEval offers (i) an API system encompassing 60 distinct complexity scenarios that can be composed into approximately 32K test configurations, and (ii) user-agent interactions for evaluating LLM agents on these scenarios. Using WildAGTEval, we systematically assess several advanced LLMs and observe that most scenarios are challenging, with irrelevant information complexity posing the greatest difficulty and reducing the performance of strong LLMs by 27.3%. Furthermore, our qualitative analysis reveals that LLMs occasionally distort user intent merely to claim task completion, critically affecting user satisfaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00263v1">Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 In submission
    </div>
    <details class="paper-abstract">
      Counterfactuals refer to minimally edited inputs that cause a model's prediction to change, serving as a promising approach to explaining the model's behavior. Large language models (LLMs) excel at generating English counterfactuals and demonstrate multilingual proficiency. However, their effectiveness in generating multilingual counterfactuals remains unclear. To this end, we conduct a comprehensive study on multilingual counterfactuals. We first conduct automatic evaluations on both directly generated counterfactuals in the target languages and those derived via English translation across six languages. Although translation-based counterfactuals offer higher validity than their directly generated counterparts, they demand substantially more modifications and still fall short of matching the quality of the original English counterfactuals. Second, we find the patterns of edits applied to high-resource European-language counterfactuals to be remarkably similar, suggesting that cross-lingual perturbations follow common strategic principles. Third, we identify and categorize four main types of errors that consistently appear in the generated counterfactuals across languages. Finally, we reveal that multilingual counterfactual data augmentation (CDA) yields larger model performance improvements than cross-lingual CDA, especially for lower-resource languages. Yet, the imperfections of the generated counterfactuals limit gains in model performance and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00254v1">An Empirical Evaluation of LLM-Based Approaches for Code Vulnerability Detection: RAG, SFT, and Dual-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) presents new opportunities for automated software vulnerability detection, a crucial task in securing modern codebases. This paper presents a comparative study on the effectiveness of LLM-based techniques for detecting software vulnerabilities. The study evaluates three approaches, Retrieval-Augmented Generation (RAG), Supervised Fine-Tuning (SFT), and a Dual-Agent LLM framework, against a baseline LLM model. A curated dataset was compiled from Big-Vul and real-world code repositories from GitHub, focusing on five critical Common Weakness Enumeration (CWE) categories: CWE-119, CWE-399, CWE-264, CWE-20, and CWE-200. Our RAG approach, which integrated external domain knowledge from the internet and the MITRE CWE database, achieved the highest overall accuracy (0.86) and F1 score (0.85), highlighting the value of contextual augmentation. Our SFT approach, implemented using parameter-efficient QLoRA adapters, also demonstrated strong performance. Our Dual-Agent system, an architecture in which a secondary agent audits and refines the output of the first, showed promise in improving reasoning transparency and error mitigation, with reduced resource overhead. These results emphasize that incorporating a domain expertise mechanism significantly strengthens the practical applicability of LLMs in real-world vulnerability detection tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.11651v3">70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float (DFloat11)</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 Published in NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large-scale AI models, such as Large Language Models (LLMs) and Diffusion Models (DMs), have grown rapidly in size, creating significant challenges for efficient deployment on resource-constrained hardware. In this paper, we introduce Dynamic-Length Float (DFloat11), a lossless compression framework that reduces LLM and DM size by 30% while preserving outputs that are bit-for-bit identical to the original model. DFloat11 is motivated by the low entropy in the BFloat16 weight representation of LLMs, which reveals significant inefficiency in the existing storage format. By applying entropy coding, DFloat11 assigns dynamic-length encodings to weights based on frequency, achieving near information-optimal compression without any loss of precision. To facilitate efficient inference with dynamic-length encodings, we develop a custom GPU kernel for fast online decompression. Our design incorporates the following: (i) compact, hierarchical lookup tables (LUTs) that fit within GPU SRAM for efficient decoding, (ii) a two-phase GPU kernel for coordinating thread read/write positions using lightweight auxiliary variables, and (iii) transformer-block-level decompression to minimize latency. Experiments on Llama 3.3, Qwen 3, Mistral 3, FLUX.1, and others validate our hypothesis that DFloat11 achieves around 30% model size reduction while preserving bit-for-bit identical outputs. Compared to a potential alternative of offloading parts of an uncompressed model to the CPU to meet memory constraints, DFloat11 achieves 2.3--46.2x higher throughput in token generation. With a fixed GPU memory budget, DFloat11 enables 5.7--14.9x longer generation lengths than uncompressed models. Notably, our method enables lossless inference of Llama 3.1 405B, an 810GB model, on a single node equipped with 8x80GB GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00240v1">Will LLM-powered Agents Bias Against Humans? Exploring the Belief-Dependent Vulnerability</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      LLM-empowered agents can exhibit not only demographic bias (e.g., gender, religion) but also intergroup bias triggered by minimal "us" versus "them" cues. When this intergroup boundary aligns with an agent-human divide, the risk shifts from disparities among human demographic groups to a more fundamental group-level asymmetry, i.e., humans as a whole may be treated as the outgroup by agents. To examine this possibility, we construct a controlled multi-agent social simulation based on allocation decisions under explicit payoff trade-offs and find that agents exhibit a consistent intergroup bias under minimal group cues. Although this bias is attenuated when some counterparts are framed as humans, we attribute the attenuation to an implicit human-norm script that favors humans yet activates only when the agent believes a real human is present. This belief dependence creates a new attack surface. We therefore introduce a Belief Poisoning Attack (BPA) that corrupts persistent identity beliefs to suppress the human-norm script and reactivate outgroup bias toward humans, instantiated as profile poisoning at initialization (BPA-PP) and memory poisoning via optimized belief-refinement suffixes injected into stored reflections (BPA-MP). Finally, we discuss practical mitigation strategies for hardening current agent frameworks against BPA, highlighting feasible interventions at profile and memory boundaries. Extensive experiments demonstrate both the existence of agent intergroup bias and the severity of BPA across settings. Our goal in identifying these vulnerabilities is to inform safer agent design, not to enable real-world exploitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.06364v3">Sketch to Adapt: Fine-Tunable Sketches for Efficient LLM Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 Published in ICML 2025
    </div>
    <details class="paper-abstract">
      Adapting pre-trained large language models (LLMs) is crucial but challenging due to their enormous size. Parameter-efficient fine-tuning (PEFT) techniques typically employ additive adapters applied to frozen model weights. To further reduce memory usage, model weights are often compressed through quantization. However, existing PEFT methods often yield suboptimal model quality because they rely on restrictive assumptions, such as low-rank constraints on adapters to limit the number of trainable parameters. We find that sketching, a popular data compression technique, can serve as an efficient LLM adaptation strategy while avoiding the low-rank assumption. We introduce SketchTune, a compressive adaptation strategy that compresses LLM weights into compact fine-tunable sketches, integrating compression and adaptation into a unified framework. This integration eliminates the need for complex two-path computation in existing PEFT techniques, enabling faster and more memory-efficient training and inference. SketchTune is supported by mathematical insights into matrix classes that are better approximated using sketching rather than low-rank methods. Our extensive evaluations with Llama and Mistral models demonstrate that SketchTune outperforms leading PEFT methods across diverse tasks while using substantially smaller base models and comparable trainable parameters. As a highlight, SketchTune outperforms LoRA, DoRA, and S2FT on commonsense and math benchmarks using 2.6-3.5$\times$ smaller base models and exceeds LoftQ in accuracy by 14.48% on GSM8K with 7.3$\times$ fewer trainable parameters. Our code is available at https://github.com/LeanModels/SketchTune.
    </details>
</div>
