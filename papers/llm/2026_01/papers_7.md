# llm - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7
- [Part 8](papers_8.md)

## Papers

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
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20957v3">One Tool Is Enough: Reinforcement Learning for Repository-Level LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Locating the files and functions requiring modification in large open-source software (OSS) repositories is challenging due to their scale and structural complexity. Existing large language model (LLM)-based methods typically treat this as a repository-level retrieval task and rely on multiple auxiliary tools, which overlook code execution logic and complicate model control. We propose RepoNavigator, an LLM agent equipped with a single execution-aware tool-jumping to the definition of an invoked symbol. This unified design reflects the actual flow of code execution while simplifying tool manipulation. RepoNavigator is trained end-to-end via Reinforcement Learning (RL) directly from a pretrained model, without any closed-source distillation. Experiments demonstrate that RL-trained RepoNavigator achieves state-of-the-art performance, with the 7B model outperforming 14B baselines, the 14B model surpassing 32B competitors, and even the 32B model exceeding closed-source models such as Claude-3.7. These results confirm that integrating a single, structurally grounded tool with RL training provides an efficient and scalable solution for repository-level issue localization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09972v3">SIP-BMM: Constructing the Capability--Efficiency Pareto Set for LLMs via Structural Importance Prior Bayesian Model Merging</a></div>
    <div class="paper-meta">
      📅 2026-01-05
    </div>
    <details class="paper-abstract">
      Constructing a Pareto set is pivotal for navigating the capability--efficiency trade-offs in Large Language Models (LLMs). However, existing merging techniques remain inadequate for this task. Coarse-grained, model-level methods yield only a sparse set of suboptimal solutions, while fine-grained, layer-wise approaches suffer from the curse of dimensionality, rendering the search space computationally intractable. To resolve this dichotomy, we propose Structural Importance Prior Bayesian Model Merging (SIP-BMM), a framework that automatically constructs the LLM Pareto set. SIP-BMM renders high-dimensional layer-wise search tractable by introducing an importance-aware Sparse Axis-Aligned Subspace Bayesian Optimization (SAASBO) strategy. By leveraging a structural importance prior derived from task-vector differences, our method guides SAASBO to automatically identify critical layers, thereby dramatically reducing the effective dimensionality without sacrificing the granularity of full-model control. The entire process is automated within an evolutionary loop driven by the Log-Noisy Expected Hypervolume Improvement ($q$NEHVI) acquisition function. Experiments demonstrate that SIP-BMM discovers a stronger and denser Pareto front than competitive baselines, enabling agile model selection tailored to diverse operational constraints. Code is available at: https://github.com/MiLab-HITSZ/2026-SIPBMM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.01299v2">La RoSA: Enhancing LLM Efficiency via Layerwise Rotated Sparse Activation</a></div>
    <div class="paper-meta">
      📅 2026-01-04
      | 💬 ICML 2025 Acceptance
    </div>
    <details class="paper-abstract">
      Activation sparsity can reduce the computational overhead and memory transfers during the forward pass of Large Language Model (LLM) inference. Existing methods face limitations, either demanding time-consuming recovery training that hinders real-world adoption, or relying on empirical magnitude-based pruning, which causes fluctuating sparsity and unstable inference speed-up. This paper introduces LaRoSA (Layerwise Rotated Sparse Activation), a novel method for activation sparsification designed to improve LLM efficiency without requiring additional training or magnitude-based pruning. We leverage layerwise orthogonal rotations to transform input activations into rotated forms that are more suitable for sparsification. By employing a Top-K selection approach within the rotated activations, we achieve consistent model-level sparsity and reliable wall-clock time speed-up. LaRoSA is effective across various sizes and types of LLMs, demonstrating minimal performance degradation and robust inference acceleration. Specifically, for LLaMA2-7B at 40% sparsity, LaRoSA achieves a mere 0.17 perplexity gap with a consistent 1.30x wall-clock time speed-up, and reduces the accuracy gap in zero-shot tasks compared to the dense model to just 0.54%, while surpassing TEAL by 1.77% and CATS by 17.14%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01461v1">Bridging the gap: A comparative exploration of Speech-LLM and end-to-end architecture for multilingual conversational ASR</a></div>
    <div class="paper-meta">
      📅 2026-01-04
      | 💬 5 pages, 1 figure
    </div>
    <details class="paper-abstract">
      The INTERSPEECH 2025 Challenge on Multilingual Conversational Speech Language Models (MLC-SLM) promotes multilingual conversational ASR with large language models (LLMs). Our previous SHNU-mASR system adopted a competitive parallel-speech-encoder architecture that integrated Whisper and mHuBERT with an LLM. However, it faced two challenges: simple feature concatenation may not fully exploit complementary information, and the performance gap between LLM-based ASR and end-to-end(E2E) encoder-decoder ASR remained unexplored. In this work, we present an enhanced LLM-based ASR framework that combines fine-tuned Whisper and mHuBERT encoders with an LLM to enrich speech representations. We first evaluate E2E Whisper models with LoRA and full fine-tuning on the MLC-SLM ASR task, and then propose cross-attention-based fusion mechanisms for the parallel-speech-encoder. On the official evaluation set of the MLC-SLM Challenge, our system achieves a CER/WER of 10.69%, ranking on par with the top-ranked Track 1 systems, even though it uses only 1,500 hours of baseline training data compared with their large-scale training sets. Nonetheless, we find that our final LLM-based ASR still does not match the performance of a fine-tuned E2E Whisper model, providing valuable empirical guidance for future Speech-LLM design. Our code is publicly available at https://github.com/1535176727/MLC-SLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05619v2">Detecting Proxy Gaming in RL and LLM Alignment via Evaluator Stress Tests</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      Proxy optimization, where AI systems exploit evaluator weaknesses rather than improve intended objectives, threatens both reinforcement learning (reward hacking) and LLM alignment (evaluator gaming). We introduce the Evaluator Stress Test (EST), an invariance-based framework that detects proxy gaming by separating exploitable sensitivity (e.g., formatting artifacts, physics bugs) from content-driven improvements using controlled perturbations with semantic validity audits. We validate EST across both domains. In RL, across 15 environments and 5 algorithms (2,156 expert-annotated episodes), EST achieves 78.4% precision and 81.7% recall. In LLM alignment, across 4 tasks, 2 model scales, 2 training methods, and 2 judges (1,200 human-annotated instances), EST achieves 74.2% precision and 78.6% recall, with early warning signals that precede quality decline. Cross-domain analysis shows that proxy-true correlation tracking transfers directly between domains, while perturbation design requires domain adaptation. Closed-loop mitigation improves human win-rate by 8.3 points (LLM) and reduces hacking by 54.6% (RL). We release benchmarks for both domains: 2,156 RL episodes and 1,200 LLM instances.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.26122v2">Reasoning Path Divergence: A New Metric and Curation Strategy to Unlock LLM Diverse Thinking</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      While Test-Time Scaling (TTS) has proven effective in improving the reasoning ability of large language models (LLMs), low diversity in model outputs often becomes a bottleneck; this is partly caused by the common "one problem, one solution" (1P1S) training practice, which provides a single canonical answer and can push models toward a narrow set of reasoning paths. This homogenization not only limits sampling effectiveness but also restricts the exploration space for subsequent Reinforcement Learning (RL) stages. To address this, we propose a "one problem, multiple solutions" (1PNS) training paradigm that exposes the model to a variety of valid reasoning trajectories and thus increases inference diversity. A core challenge for 1PNS is reliably measuring semantic differences between multi-step chains of thought, so we introduce Reasoning Path Divergence (RPD), a step-level metric that aligns and scores Long Chain-of-Thought solutions to capture differences in intermediate reasoning. Using RPD, we curate maximally diverse solution sets per problem and fine-tune Qwen3-4B-Base. Experiments show that RPD-selected training yields more varied outputs and higher pass@k, with an average +2.80% gain in pass@16 over a strong 1P1S baseline and a +4.99% gain on AIME24, demonstrating that 1PNS further amplifies the effectiveness of TTS. Our code is available at https://github.com/fengjujf/Reasoning-Path-Divergence .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20578v2">Can LLMs Predict Their Own Failures? Self-Awareness via Internal Circuits</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) generate fluent and complex outputs but often fail to recognize their own mistakes and hallucinations. Existing approaches typically rely on external judges, multi-sample consistency, or text-based self-critique, which incur additional compute or correlate weakly with true correctness. We ask: can LLMs predict their own failures by inspecting internal states during inference? We introduce Gnosis, a lightweight self-awareness mechanism that enables frozen LLMs to perform intrinsic self-verification by decoding signals from hidden states and attention patterns. Gnosis passively observes internal traces, compresses them into fixed-budget descriptors, and predicts correctness with negligible inference cost, adding only ~5M parameters and operating independently of sequence length. Across math reasoning, open-domain question answering, and academic knowledge benchmarks, and over frozen backbones ranging from 1.7B to 20B parameters, Gnosis consistently outperforms strong internal baselines and large external judges in both accuracy and calibration. Moreover, it generalizes zero-shot to partial generations, enabling early detection of failing trajectories and compute-aware control. These results show that reliable correctness cues are intrinsic to generation process and can be extracted efficiently without external supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21140v2">How to Correctly Report LLM-as-a-Judge Evaluations</a></div>
    <div class="paper-meta">
      📅 2026-01-04
      | 💬 This version adds Sections 2, 6, 7, and 8.2
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely used as scalable evaluators of model responses in lieu of human annotators. However, imperfect sensitivity and specificity of LLM judgments induce bias in naive evaluation scores. We propose a simple plug-in framework that corrects this bias and constructs confidence intervals accounting for uncertainty from both the test dataset and a human-evaluated calibration dataset, enabling statistically sound and practical LLM-based evaluation. Building on this framework, we introduce an adaptive calibration strategy for constructing the calibration dataset to reduce uncertainty in the estimated score. Notably, we characterize the regimes in which LLM-based evaluation within our framework produces more reliable estimates than fully human evaluation. Moreover, our framework is more robust to distribution shift between the test and calibration datasets than existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25743v3">Rotation Control Unlearning: Quantifying and Controlling Continuous Unlearning for LLM with The Cognitive Rotation Space</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly prevalent, their security vulnerabilities have already drawn attention. Machine unlearning is introduced to seek to mitigate these risks by removing the influence of undesirable data. However, existing methods not only rely on the retained dataset to preserve model utility, but also suffer from cumulative catastrophic utility loss under continuous unlearning requests. To solve this dilemma, we propose a novel method, called Rotation Control Unlearning (RCU), which leverages the rotational salience weight of RCU to quantify and control the unlearning degree in the continuous unlearning process. The skew symmetric loss is designed to construct the existence of the cognitive rotation space, where the changes of rotational angle can simulate the continuous unlearning process. Furthermore, we design an orthogonal rotation axes regularization to enforce mutually perpendicular rotation directions for continuous unlearning requests, effectively minimizing interference and addressing cumulative catastrophic utility loss. Experiments on multiple datasets confirm that our method without retained dataset achieves SOTA performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01357v1">Towards LLM-enabled autonomous combustion research: A literature-aware agent for self-corrective modeling workflows</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      The rapid evolution of large language models (LLMs) is transforming artificial intelligence into autonomous research partners, yet a critical gap persists in complex scientific domains such as combustion modeling. Here, practical AI assistance requires the seamless integration of domain literature knowledge with robust execution capabilities for expertise-intensive tools such as computational fluid dynamics (CFD) codes. To bridge this gap, we introduce FlamePilot, an LLM agent designed to empower combustion modeling research through automated and self-corrective CFD workflows. FlamePilot differentiates itself through an architecture that leverages atomic tools to ensure the robust setup and execution of complex simulations in both OpenFOAM and extended frameworks such as DeepFlame. The system is also capable of learning from scientific articles, extracting key information to guide the simulation from initial setup to optimized results. Validation on a public benchmark shows FlamePilot achieved a perfect 1.0 executability score and a 0.438 success rate, surpassing the prior best reported agent scores of 0.625 and 0.250, respectively. Furthermore, a detailed case study on Moderate or Intense Low-oxygen Dilution (MILD) combustion simulation demonstrates its efficacy as a collaborative research copilot, where FlamePilot autonomously translated a research paper into a configured simulation, conducted the simulation, post-processed the results, proposed evidence-based refinements, and managed a multi-step parameter study to convergence under minimal human intervention. By adopting a transparent and interpretable paradigm, FlamePilot establishes a foundational framework for AI-empowered combustion modeling, fostering a collaborative partnership where the agent manages workflow orchestration, freeing the researcher for high-level analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.12948v2">DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      General reasoning represents a long-standing and formidable challenge in artificial intelligence. Recent breakthroughs, exemplified by large language models (LLMs) and chain-of-thought prompting, have achieved considerable success on foundational reasoning tasks. However, this success is heavily contingent upon extensive human-annotated demonstrations, and models' capabilities are still insufficient for more complex problems. Here we show that the reasoning abilities of LLMs can be incentivized through pure reinforcement learning (RL), obviating the need for human-labeled reasoning trajectories. The proposed RL framework facilitates the emergent development of advanced reasoning patterns, such as self-reflection, verification, and dynamic strategy adaptation. Consequently, the trained model achieves superior performance on verifiable tasks such as mathematics, coding competitions, and STEM fields, surpassing its counterparts trained via conventional supervised learning on human demonstrations. Moreover, the emergent reasoning patterns exhibited by these large-scale models can be systematically harnessed to guide and enhance the reasoning capabilities of smaller models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01330v1">Beyond Gemini-3-Pro: Revisiting LLM Routing and Aggregation at Scale</a></div>
    <div class="paper-meta">
      📅 2026-01-04
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have rapidly advanced, with Gemini-3-Pro setting a new performance milestone. In this work, we explore collective intelligence as an alternative to monolithic scaling, and demonstrate that open-source LLMs' collaboration can surpass Gemini-3-Pro. We first revisit LLM routing and aggregation at scale and identify three key bottlenecks: (1) current train-free routers are limited by a query-based paradigm focusing solely on textual similarity; (2) recent aggregation methods remain largely static, failing to select appropriate aggregators for different tasks;(3) the complementarity of routing and aggregation remains underutilized. To address these problems, we introduce JiSi, a novel framework designed to release the full potential of LLMs' collaboration through three innovations: (1) Query-Response Mixed Routing capturing both semantic information and problem difficulty; (2) Support-Set-based Aggregator Selection jointly evaluating the aggregation and domain capacity of aggregators; (3) Adaptive Routing-Aggregation Switch dynamically leveraging the advantages of routing and aggregation. Comprehensive experiments on nine benchmarks demonstrate that JiSi can surpass Gemini-3-Pro with only 47% costs by orchestrating ten open-source LLMs, while outperforming mainstream baselines. It suggests that collective intelligence represents a novel path towards Artificial General Intelligence (AGI).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01320v1">Adaptive Hierarchical Evaluation of LLMs and SAST tools for CWE Prediction in Python</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      Large Language Models have become integral to software development, yet they frequently generate vulnerable code. Existing code vulnerability detection benchmarks employ binary classification, lacking the CWE-level specificity required for actionable feedback in iterative correction systems. We present ALPHA (Adaptive Learning via Penalty in Hierarchical Assessment), the first function-level Python benchmark that evaluates both LLMs and SAST tools using hierarchically aware, CWE-specific penalties. ALPHA distinguishes between over-generalisation, over-specification, and lateral errors, reflecting practical differences in diagnostic utility. Evaluating seven LLMs and two SAST tools, we find LLMs substantially outperform SAST, though SAST demonstrates higher precision when detections occur. Critically, prediction consistency varies dramatically across models (8.26%-81.87% agreement), with significant implications for feedback-driven systems. We further outline a pathway for future work incorporating ALPHA penalties into supervised fine-tuning, which could provide principled hierarchy-aware vulnerability detection pending empirical validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01310v1">Making MoE based LLM inference resilient with Tarragon</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) models are increasingly used to serve LLMs at scale, but failures become common as deployment scale grows. Existing systems exhibit poor failure resilience: even a single worker failure triggers a coarse-grained, service-wide restart, discarding accumulated progress and halting the entire inference pipeline during recovery--an approach clearly ill-suited for latency-sensitive, LLM services. We present Tarragon, a resilient MoE inference framework that confines the failures impact to individual workers while allowing the rest of the pipeline to continue making forward progress. Tarragon exploits the natural separation between the attention and expert computation in MoE-based transformers, treating attention workers (AWs) and expert workers (EWs) as distinct failure domains. Tarragon introduces a reconfigurable datapath to mask failures by rerouting requests to healthy workers. On top of this datapath, Tarragon implements a self-healing mechanism that relaxes the tightly synchronized execution of existing MoE frameworks. For stateful AWs, Tarragon performs asynchronous, incremental KV cache checkpointing with per-request restoration, and for stateless EWs, it leverages residual GPU memory to deploy shadow experts. These together keep recovery cost and recomputation overhead extremely low. Our evaluation shows that, compared to state-of-the-art MegaScale-Infer, Tarragon reduces failure-induced stalls by 160-213x (from ~64 s down to 0.3-0.4 s) while preserving performance when no failures occur.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21594v2">Visualizing LLM Latent Space Geometry Through Dimensionality Reduction</a></div>
    <div class="paper-meta">
      📅 2026-01-04
      | 💬 22 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve state-of-the-art results across many natural language tasks, but their internal mechanisms remain difficult to interpret. In this work, we extract, process, and visualize latent state geometries in Transformer-based language models through dimensionality reduction. We capture layerwise activations at multiple points within Transformer blocks and enable systematic analysis through Principal Component Analysis (PCA) and Uniform Manifold Approximation and Projection (UMAP). We demonstrate experiments on GPT-2 and LLaMa models, where we uncover interesting geometric patterns in latent space. Notably, we identify a clear separation between attention and MLP component outputs across intermediate layers, a pattern not documented in prior work to our knowledge. We also characterize the high norm of latent states at the initial sequence position and visualize the layerwise evolution of latent states. Additionally, we demonstrate the high-dimensional helical structure of GPT-2's positional embeddings and the sequence-wise geometric patterns in LLaMa. We make our code available at https://github.com/Vainateya/Feature_Geometry_Visualization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18790v2">RoguePrompt: Dual-Layer Ciphering for Self-Reconstruction to Circumvent LLM Moderation</a></div>
    <div class="paper-meta">
      📅 2026-01-04
      | 💬 This manuscript has been submitted for consideration to the ACM Conference on Data and Application Security and Privacy (CODASPY) 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are becoming increasingly integrated into mainstream development platforms and daily technological workflows, typically behind moderation and safety controls. Despite these controls, preventing prompt-based policy evasion remains challenging, and adversaries continue to jailbreak LLMs by crafting prompts that circumvent implemented safety mechanisms. While prior jailbreak techniques have explored obfuscation and contextual manipulation, many operate as single-step transformations, and their effectiveness is inconsistent across current state-of-the-art models. This leaves a limited understanding of multistage prompt-transformation attacks that evade moderation, reconstruct forbidden intent, and elicit policy-violating outputs. This paper introduces RoguePrompt, an automated jailbreak pipeline that leverages dual-layer prompt transformations to convert forbidden prompts into safety-evading queries. By partitioning the forbidden prompts and applying two nested encodings (ROT-13 and Vigenère) along with natural-language decoding instructions, it produces benign-looking prompts that evade filters and induce the model to execute the original prompt within a single query. RoguePrompt was developed and evaluated under a black-box threat model, with only API and UI access to the LLMs, and tested on 313 real-world hard-rejected prompts. Success was measured in terms of moderation bypass, instruction reconstruction, and execution, using both automated and human evaluation. It achieved an average of 93.93% filter bypass, 79.02% reconstruction, and 70.18% execution success across multiple frontier LLMs. These results demonstrate the effectiveness of layered prompt encoding and highlight the need for innovative defenses to detect and mitigate self-reconstructing jailbreaks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24556v2">Safe in the Future, Dangerous in the Past: Dissecting Temporal and Linguistic Vulnerabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) integrate into critical global infrastructure, the assumption that safety alignment transfers zero-shot from English to other languages remains a dangerous blind spot. This study presents a systematic audit of three state of the art models (GPT-5.1, Gemini 3 Pro, and Claude 4.5 Opus) using HausaSafety, a novel adversarial dataset grounded in West African threat scenarios (e.g., Yahoo-Yahoo fraud, Dane gun manufacturing). Employing a 2 x 4 factorial design across 1,440 evaluations, we tested the non-linear interaction between language (English vs. Hausa) and temporal framing. Our results challenge the narrative of the multilingual safety gap. Instead of a simple degradation in low-resource settings, we identified a complex interference mechanism in which safety is determined by the intersection of variables. Although the models exhibited a reverse linguistic vulnerability with Claude 4.5 Opus proving significantly safer in Hausa (45.0%) than in English (36.7%) due to uncertainty-driven refusal, they suffered catastrophic failures in temporal reasoning. We report a profound Temporal Asymmetry, where past-tense framing bypassed defenses (15.6% safe) while future-tense scenarios triggered hyper-conservative refusals (57.2% safe). The magnitude of this volatility is illustrated by a 9.2x disparity between the safest and most vulnerable configurations, proving that safety is not a fixed property but a context-dependent state. We conclude that current models rely on superficial heuristics rather than robust semantic understanding, creating Safety Pockets that leave Global South users exposed to localized harms. We propose Invariant Alignment as a necessary paradigm shift to ensure safety stability across linguistic and temporal shifts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04050v2">Explainability-Based Token Replacement on LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      Generative models, especially large language models (LLMs), have shown remarkable progress in producing text that appears human-like. However, they often exhibit patterns that make their output easier to detect than text written by humans. In this paper, we investigate how explainable AI (XAI) methods can be used to reduce the detectability of AI-generated text (AIGT) while also introducing a robust ensemble-based detection approach. We begin by training an ensemble classifier to distinguish AIGT from human-written text, then apply SHAP and LIME to identify tokens that most strongly influence its predictions. We propose four explainability-based token replacement strategies to modify these influential tokens. Our findings show that these token replacement approaches can significantly diminish a single classifier's ability to detect AIGT. However, our ensemble classifier maintains strong performance across multiple languages and domains, showing that a multi-model approach can mitigate the impact of token-level manipulations. These results show that XAI methods can make AIGT harder to detect by focusing on the most influential tokens. At the same time, they highlight the need for robust, ensemble-based detection strategies that can adapt to evolving approaches for hiding AIGT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01609v1">Structured Decomposition for LLM Reasoning: Cross-Domain Validation and Semantic Web Integration</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      Rule-based reasoning over natural language input arises in domains where decisions must be auditable and justifiable: clinical protocols specify eligibility criteria in prose, evidence rules define admissibility through textual conditions, and scientific standards dictate methodological requirements. Applying rules to such inputs demands both interpretive flexibility and formal guarantees. Large language models (LLMs) provide flexibility but cannot ensure consistent rule application; symbolic systems provide guarantees but require structured input. This paper presents an integration pattern that combines these strengths: LLMs serve as ontology population engines, translating unstructured text into ABox assertions according to expert-authored TBox specifications, while SWRL-based reasoners apply rules with deterministic guarantees. The framework decomposes reasoning into entity identification, assertion extraction, and symbolic verification, with task definitions grounded in OWL 2 ontologies. Experiments across three domains (legal hearsay determination, scientific method-task application, clinical trial eligibility) and eleven language models validate the approach. Structured decomposition achieves statistically significant improvements over few-shot prompting in aggregate, with gains observed across all three domains. An ablation study confirms that symbolic verification provides substantial benefit beyond structured prompting alone. The populated ABox integrates with standard semantic web tooling for inspection and querying, positioning the framework for richer inference patterns that simpler formalisms cannot express.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01576v1">OpenNovelty: An LLM-powered Agentic System for Verifiable Scholarly Novelty Assessment</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      Evaluating novelty is critical yet challenging in peer review, as reviewers must assess submissions against a vast, rapidly evolving literature. This report presents OpenNovelty, an LLM-powered agentic system for transparent, evidence-based novelty analysis. The system operates through four phases: (1) extracting the core task and contribution claims to generate retrieval queries; (2) retrieving relevant prior work based on extracted queries via semantic search engine; (3) constructing a hierarchical taxonomy of core-task-related work and performing contribution-level full-text comparisons against each contribution; and (4) synthesizing all analyses into a structured novelty report with explicit citations and evidence snippets. Unlike naive LLM-based approaches, \textsc{OpenNovelty} grounds all assessments in retrieved real papers, ensuring verifiable judgments. We deploy our system on 500+ ICLR 2026 submissions with all reports publicly available on our website, and preliminary analysis suggests it can identify relevant prior work, including closely related papers that authors may overlook. OpenNovelty aims to empower the research community with a scalable tool that promotes fair, consistent, and evidence-backed peer review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01569v1">CaveAgent: Transforming LLMs into Stateful Runtime Operators</a></div>
    <div class="paper-meta">
      📅 2026-01-04
      | 💬 32 pages, 14 Figures
    </div>
    <details class="paper-abstract">
      LLM-based agents are increasingly capable of complex task execution, yet current agentic systems remain constrained by text-centric paradigms. Traditional approaches rely on procedural JSON-based function calling, which often struggles with long-horizon tasks due to fragile multi-turn dependencies and context drift. In this paper, we present CaveAgent, a framework that transforms the paradigm from "LLM-as-Text-Generator" to "LLM-as-Runtime-Operator." We introduce a Dual-stream Context Architecture that decouples state management into a lightweight semantic stream for reasoning and a persistent, deterministic Python Runtime stream for execution. In addition to leveraging code generation to efficiently resolve interdependent sub-tasks (e.g., loops, conditionals) in a single step, we introduce \textit{Stateful Runtime Management} in CaveAgent. Distinct from existing code-based approaches that remain text-bound and lack the support for external object injection and retrieval, CaveAgent injects, manipulates, and retrieves complex Python objects (e.g., DataFrames, database connections) that persist across turns. This persistence mechanism acts as a high-fidelity external memory to eliminate context drift, avoid catastrophic forgetting, while ensuring that processed data flows losslessly to downstream applications. Comprehensive evaluations on Tau$^2$-bench, BFCL and various case studies across representative SOTA LLMs demonstrate CaveAgent's superiority. Specifically, our framework achieves a 10.5\% success rate improvement on retail tasks and reduces total token consumption by 28.4\% in multi-turn scenarios. On data-intensive tasks, direct variable storage and retrieval reduces token consumption by 59\%, allowing CaveAgent to handle large-scale data that causes context overflow failures in both JSON-based and Code-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01546v1">Improving Behavioral Alignment in LLM Social Simulations via Context Formation and Navigation</a></div>
    <div class="paper-meta">
      📅 2026-01-04
      | 💬 39 pages, 2 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to simulate human behavior in experimental settings, but they systematically diverge from human decisions in complex decision-making environments, where participants must anticipate others' actions and form beliefs based on observed behavior. We propose a two-stage framework for improving behavioral alignment. The first stage, context formation, explicitly specifies the experimental design to establish an accurate representation of the decision task and its context. The second stage, context navigation, guides the reasoning process within that representation to make decisions. We validate this framework through a focal replication of a sequential purchasing game with quality signaling (Kremer and Debo, 2016), extending to a crowdfunding game with costly signaling (Cason et al., 2025) and a demand-estimation task (Gui and Toubia, 2025) to test generalizability across decision environments. Across four state-of-the-art (SOTA) models (GPT-4o, GPT-5, Claude-4.0-Sonnet-Thinking, DeepSeek-R1), we find that complex decision-making environments require both stages to achieve behavioral alignment with human benchmarks, whereas the simpler demand-estimation task requires only context formation. Our findings clarify when each stage is necessary and provide a systematic approach for designing and diagnosing LLM social simulations as complements to human subjects in behavioral research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01522v1">Bayesian Orchestration of Multi-LLM Agents for Cost-Aware Sequential Decision-Making</a></div>
    <div class="paper-meta">
      📅 2026-01-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as autonomous decision agents in settings with asymmetric error costs: hiring (missed talent vs wasted interviews), medical triage (missed emergencies vs unnecessary escalation), and fraud detection (approved fraud vs declined legitimate payments). The dominant design queries a single LLM for a posterior over states, thresholds "confidence," and acts; we prove this is inadequate for sequential decisions with costs. We propose a Bayesian, cost-aware multi-LLM orchestration framework that treats LLMs as approximate likelihood models rather than classifiers. For each candidate state, we elicit likelihoods via contrastive prompting, aggregate across diverse models with robust statistics, and update beliefs with Bayes rule under explicit priors as new evidence arrives. This enables coherent belief updating, expected-cost action selection, principled information gathering via value of information, and fairness gains via ensemble bias mitigation. In resume screening with costs of 40000 USD per missed hire, 2500 USD per interview, and 150 USD per phone screen, experiments on 1000 resumes using five LLMs (GPT-4o, Claude 4.5 Sonnet, Gemini Pro, Grok, DeepSeek) reduce total cost by 294000 USD (34 percent) versus the best single-LLM baseline and improve demographic parity by 45 percent (max group gap 22 to 5 percentage points). Ablations attribute 51 percent of savings to multi-LLM aggregation, 43 percent to sequential updating, and 20 percent to disagreement-triggered information gathering, consistent with the theoretical benefits of correct probabilistic foundations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.09082v3">CoSER: A Comprehensive Literary Dataset and Framework for Training and Evaluating LLM Role-Playing and Persona Simulation</a></div>
    <div class="paper-meta">
      📅 2026-01-04
      | 💬 Accepted by ICML 2025
    </div>
    <details class="paper-abstract">
      Role-playing language agents (RPLAs) have emerged as promising applications of large language models (LLMs). However, simulating established characters presents a challenging task for RPLAs, due to the lack of authentic character datasets and nuanced evaluation methods using such data. In this paper, we present CoSER, a collection of a high-quality dataset, open models, and an evaluation protocol towards effective RPLAs of established characters. The CoSER dataset covers 17,966 characters from 771 renowned books. It provides authentic dialogues with real-world intricacies, as well as diverse data types such as conversation setups, character experiences and internal thoughts. Drawing from acting methodology, we introduce given-circumstance acting for training and evaluating role-playing LLMs, where LLMs sequentially portray multiple characters in book scenes. Using our dataset, we develop CoSER 8B and CoSER 70B, i.e., advanced open role-playing LLMs built on LLaMA-3.1 models. Extensive experiments demonstrate the value of the CoSER dataset for RPLA training, evaluation and retrieval. Moreover, CoSER 70B exhibits state-of-the-art performance surpassing or matching GPT-4o on our evaluation and three existing benchmarks, i.e., achieving 75.80% and 93.47% accuracy on the InCharacter and LifeChoice benchmarks respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01094v1">SoulSeek: Exploring the Use of Social Cues in LLM-based Information Seeking</a></div>
    <div class="paper-meta">
      📅 2026-01-03
    </div>
    <details class="paper-abstract">
      Social cues, which convey others' presence, behaviors, or identities, play a crucial role in human information seeking by helping individuals judge relevance and trustworthiness. However, existing LLM-based search systems primarily rely on semantic features, creating a misalignment with the socialized cognition underlying natural information seeking. To address this gap, we explore how the integration of social cues into LLM-based search influences users' perceptions, experiences, and behaviors. Focusing on social media platforms that are beginning to adopt LLM-based search, we integrate design workshops, the implementation of the prototype system (SoulSeek), a between-subjects study, and mixed-method analyses to examine both outcome- and process-level findings. The workshop informs the prototype's cue-integrated design. The study shows that social cues improve perceived outcomes and experiences, promote reflective information behaviors, and reveal limits of current LLM-based search. We propose design implications emphasizing better social-knowledge understanding, personalized cue settings, and controllable interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01046v1">KV-Embedding: Training-free Text Embedding via Internal KV Re-routing in Decoder-only LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-03
    </div>
    <details class="paper-abstract">
      While LLMs are powerful embedding backbones, their application in training-free settings faces two structural challenges: causal attention restricts early tokens from accessing subsequent context, and the next-token prediction objective biases representations toward generation rather than semantic compression. To address these limitations, we propose KV-Embedding, a framework that activates the latent representation power of frozen LLMs. Our method leverages the observation that the key-value (KV) states of the final token at each layer encode a compressed view of the sequence. By re-routing these states as a prepended prefix, we enable all tokens to access sequence-level context within a single forward pass. To ensure model-agnostic applicability, we introduce an automated layer selection strategy based on intrinsic dimensionality. Evaluations on MTEB across Qwen, Mistral, and Llama backbones show that KV-Embedding outperforms existing training-free baselines by up to 10%, while maintaining robust performance on sequences up to 4,096 tokens. These results demonstrate that internal state manipulation offers an efficient alternative to input modification, and we hope this work encourages further exploration of LLM internals for representation learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01015v1">HyperJoin: LLM-augmented Hypergraph Link Prediction for Joinable Table Discovery</a></div>
    <div class="paper-meta">
      📅 2026-01-03
    </div>
    <details class="paper-abstract">
      As a pivotal task in data lake management, joinable table discovery has attracted widespread interest. While existing language model-based methods achieve remarkable performance by combining offline column representation learning with online ranking, their design insufficiently accounts for the underlying structural interactions: (1) offline, they directly model tables into isolated or pairwise columns, thereby struggling to capture the rich inter-table and intra-table structural information; and (2) online, they rank candidate columns based solely on query-candidate similarity, ignoring the mutual interactions among the candidates, leading to incoherent result sets. To address these limitations, we propose HyperJoin, a large language model (LLM)-augmented Hypergraph framework for Joinable table discovery. Specifically, we first construct a hypergraph to model tables using both the intra-table hyperedges and the LLM-augmented inter-table hyperedges. Consequently, the task of joinable table discovery is formulated as link prediction on this constructed hypergraph. We then design HIN, a Hierarchical Interaction Network that learns expressive column representations through bidirectional message passing over columns and hyperedges. To strengthen coherence and internal consistency in the result columns, we cast online ranking as a coherence-aware top-k column selection problem. We then introduce a reranking module that leverages a maximum spanning tree algorithm to prune noisy connections and maximize coherence. Experiments demonstrate the superiority of HyperJoin, achieving average improvements of 21.4% (Precision@15) and 17.2% (Recall@15) over the best baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01296v1">Aggressive Compression Enables LLM Weight Theft</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 An early version of this work was presented at the SoLAR Workshop at NeurIPS 2024
    </div>
    <details class="paper-abstract">
      As frontier AIs become more powerful and costly to develop, adversaries have increasing incentives to steal model weights by mounting exfiltration attacks. In this work, we consider exfiltration attacks where an adversary attempts to sneak model weights out of a datacenter over a network. While exfiltration attacks are multi-step cyber attacks, we demonstrate that a single factor, the compressibility of model weights, significantly heightens exfiltration risk for large language models (LLMs). We tailor compression specifically for exfiltration by relaxing decompression constraints and demonstrate that attackers could achieve 16x to 100x compression with minimal trade-offs, reducing the time it would take for an attacker to illicitly transmit model weights from the defender's server from months to days. Finally, we study defenses designed to reduce exfiltration risk in three distinct ways: making models harder to compress, making them harder to 'find,' and tracking provenance for post-attack analysis using forensic watermarks. While all defenses are promising, the forensic watermark defense is both effective and cheap, and therefore is a particularly attractive lever for mitigating weight-exfiltration risk.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.18665v4">Membership Inference Attacks on LLM-based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 This is paper is under review WWW 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) based recommender systems (RecSys) can adapt to different domains flexibly. It utilizes in-context learning (ICL), i.e., prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, encompassing implicit feedback such as clicked items and explicit product reviews. Such private information may be exposed by novel privacy attacks. However, no study has been conducted on this important issue. We design several membership inference attacks (MIAs) aimed to revealing whether system prompts include victims' historical interactions. The attacks are \emph{Similarity, Memorization, Inquiry, and Poisoning attacks}, each utilizing unique features of LLMs or RecSys. We have carefully evaluated them on five of the latest open-source LLMs and three well-known RecSys benchmark datasets. The results confirm that the MIA threat to LLM RecSys is realistic: inquiry and poisoning attacks show significantly high attack advantages. We also discussed possible methods to mitigate such MIA threats. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts, the position of the victim in the shots, the number of poisoning items in the prompt,etc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01279v1">LLM Collusion</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 44 pages
    </div>
    <details class="paper-abstract">
      We study how delegating pricing to large language models (LLMs) can facilitate collusion in a duopoly when both sellers rely on the same pre-trained model. The LLM is characterized by (i) a propensity parameter capturing its internal bias toward high-price recommendations and (ii) an output-fidelity parameter measuring how tightly outputs track that bias; the propensity evolves through retraining. We show that configuring LLMs for robustness and reproducibility can induce collusion via a phase transition: there exists a critical output-fidelity threshold that pins down long-run behavior. Below it, competitive pricing is the unique long-run outcome. Above it, the system is bistable, with competitive and collusive pricing both locally stable and the realized outcome determined by the model's initial preference. The collusive regime resembles tacit collusion: prices are elevated on average, yet occasional low-price recommendations provide plausible deniability. With perfect fidelity, full collusion emerges from any interior initial condition. For finite training batches of size $b$, infrequent retraining (driven by computational costs) further amplifies collusion: conditional on starting in the collusive basin, the probability of collusion approaches one as $b$ grows, since larger batches dampen stochastic fluctuations that might otherwise tip the system toward competition. The indeterminacy region shrinks at rate $O(1/\sqrt{b})$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01271v1">CatchAll: Repository-Aware Exception Handling with Knowledge-Guided LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-03
    </div>
    <details class="paper-abstract">
      Exception handling is a vital forward error-recovery mechanism in many programming languages, enabling developers to manage runtime anomalies through structured constructs (e.g., try-catch blocks). Improper or missing exception handling often leads to severe consequences, including system crashes and resource leaks. While large language models (LLMs) have demonstrated strong capabilities in code generation, they struggle with exception handling at the repository level, due to complex dependencies and contextual constraints. In this work, we propose CatchAll, a novel LLM-based approach for repository-aware exception handling. CatchAll equips LLMs with three complementary layers of exception-handling knowledge: (1) API-level exception knowledge, obtained from an empirically constructed API-exception mapping that characterizes the exception-throwing behaviors of APIs in real-world codebases; (2) repository-level execution context, which captures exception propagation by modeling contextual call traces around the target code; and (3) cross-repository handling knowledge, distilled from reusable exception-handling patterns mined from historical code across projects. The knowledge is encoded into structured prompts to guide the LLM in generating accurate and context-aware exception-handling code. To evaluate CatchAll, we construct two new benchmarks for repository-aware exception handling: a large-scale dataset RepoExEval and an executable subset RepoExEval-Exec. Experiments demonstrate that RepoExEval consistently outperforms state-of-the-art baselines, achieving a CodeBLEU score of 0.31 (vs. 0.27% for the best baseline), intent prediction accuracy of 60.1% (vs. 48.0%), and Pass@1 of 29% (vs. 25%). These results affirm RepoExEval's effectiveness in real-world repository-level exception handling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15815v2">User-Assistant Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-03
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) are typically trained and deployed using structured role tags (e.g. system, user, assistant, tool) that explicitly mark the source of each piece of context. While these tags are essential for instruction following and controllability, asymmetries in the training data associated with different role tags can introduce inductive biases. In this paper, we study this phenomenon by formalizing user-assistant bias, defined as the tendency of an LLM to preferentially rely on information from either the user or assistant role when there is a conflict. We introduce a task-agnostic benchmark UserAssist and evaluate such bias in 52 frontier models. We observe that most of the instruction-tuned models exhibit strong user bias, whereas base and reasoning models are close to neutral. Using controlled fine-tuning experiments, we isolate which post-training recipes drive the observed user-assistant bias. We find that human-preference alignment amplifies user bias, while reasoning fine-tuning reduces it. Finally, we show that user-assistant bias can be bidirectionally controlled via direct preference optimization (DPO) on UserAssist-train, and that the resulting bias reliably generalizes to a more realistic multi-turn conversation dataset. These results reveal an underexplored consequence of role-tagged training and provide a principled framework to diagnose and control tag-induced biases in modern LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.07745v4">Chimera: Harnessing Multi-Agent LLMs for Automatic Insider Threat Simulation</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 Accepted by NDSS 2026
    </div>
    <details class="paper-abstract">
      Insider threats pose a persistent and critical security risk, yet are notoriously difficult to detect in complex enterprise environments, where malicious actions are often hidden within seemingly benign user behaviors. Although machine-learning-based insider threat detection (ITD) methods have shown promise, their effectiveness is fundamentally limited by the scarcity of high-quality and realistic training data. Enterprise internal data is highly sensitive and rarely accessible, while existing public and synthetic datasets are either small-scale or lack sufficient realism, semantic richness, and behavioral diversity. To address this challenge, we propose Chimera, an LLM-based multi-agent framework that automatically simulates both benign and malicious insider activities and generates comprehensive system logs across diverse enterprise environments. Chimera models each agent as an individual employee with fine-grained roles and supports group meetings, pairwise interactions, and self-organized scheduling to capture realistic organizational dynamics. Based on 15 insider attacks abstracted from real-world incidents, we deploy Chimera in three representative data-sensitive organizational scenarios and construct ChimeraLog, a new dataset for developing and evaluating ITD methods. We evaluate ChimeraLog through human studies and quantitative analyses, demonstrating its diversity and realism. Experiments with existing ITD methods show substantially lower detection performance on ChimeraLog compared to prior datasets, indicating a more challenging and realistic benchmark. Moreover, despite distribution shifts, models trained on ChimeraLog exhibit strong generalization, highlighting the practical value of LLM-based multi-agent simulation for advancing insider threat detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01233v1">Atomizer: An LLM-based Collaborative Multi-Agent Framework for Intent-Driven Commit Untangling</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 Accepted by ICSE 2026
    </div>
    <details class="paper-abstract">
      Composite commits, which entangle multiple unrelated concerns, are prevalent in software development and significantly hinder program comprehension and maintenance. Existing automated untangling methods, particularly state-of-the-art graph clustering-based approaches, are fundamentally limited by two issues. (1) They over-rely on structural information, failing to grasp the crucial semantic intent behind changes, and (2) they operate as ``single-pass'' algorithms, lacking a mechanism for the critical reflection and refinement inherent in human review processes. To overcome these challenges, we introduce Atomizer, a novel collaborative multi-agent framework for composite commit untangling. To address the semantic deficit, Atomizer employs an Intent-Oriented Chain-of-Thought (IO-CoT) strategy, which prompts large language models (LLMs) to infer the intent of each code change according to both the structure and the semantic information of code. To overcome the limitations of ``single-pass'' grouping, we employ two agents to establish a grouper-reviewer collaborative refinement loop, which mirrors human review practices by iteratively refining groupings until all changes in a cluster share the same underlying semantic intent. Extensive experiments on two benchmark C# and Java datasets demonstrate that Atomizer significantly outperforms several representative baselines. On average, it surpasses the state-of-the-art graph-based methods by over 6.0% on the C# dataset and 5.5% on the Java dataset. This superiority is particularly pronounced on complex commits, where Atomizer's performance advantage widens to over 16%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01215v1">Correctness isnt Efficiency: Runtime Memory Divergence in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 11 Pages, 11 figures, Accepted at ICSE SEIP
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate programs that pass unit tests, but passing tests does not guarantee reliable runtime behavior. We find that different correct solutions to the same task can show very different memory and performance patterns, which can lead to hidden operational risks. We present a framework to measure execution-time memory stability across multiple correct generations. At the solution level, we introduce Dynamic Mean Pairwise Distance (DMPD), which uses Dynamic Time Warping to compare the shapes of memory-usage traces after converting them into Monotonic Peak Profiles (MPPs) to reduce transient noise. Aggregating DMPD across tasks yields a model-level Model Instability Score (MIS). Experiments on BigOBench and CodeContests show substantial runtime divergence among correct solutions. Instability often increases with higher sampling temperature even when pass@1 improves. We also observe correlations between our stability measures and software engineering indicators such as cognitive and cyclomatic complexity, suggesting links between operational behavior and maintainability. Our results support stability-aware selection among passing candidates in CI/CD to reduce operational risk without sacrificing correctness. Artifacts are available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23489v2">The Gaining Paths to Investment Success: Information-Driven LLM Graph Reasoning for Venture Capital Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-03
    </div>
    <details class="paper-abstract">
      Most venture capital (VC) investments fail, while a few deliver outsized returns. Accurately predicting startup success requires synthesizing complex relational evidence, including company disclosures, investor track records, and investment network structures, through explicit reasoning to form coherent, interpretable investment theses. Traditional machine learning and graph neural networks both lack this reasoning capability. Large language models (LLMs) offer strong reasoning but face a modality mismatch with graphs. Recent graph-LLM methods target in-graph tasks where answers lie within the graph, whereas VC prediction is off-graph: the target exists outside the network. The core challenge is selecting graph paths that maximize predictor performance on an external objective while enabling step-by-step reasoning. We present MIRAGE-VC, a multi-perspective retrieval-augmented generation framework that addresses two obstacles: path explosion (thousands of candidate paths overwhelm LLM context) and heterogeneous evidence fusion (different startups need different analytical emphasis). Our information-gain-driven path retriever iteratively selects high-value neighbors, distilling investment networks into compact chains for explicit reasoning. A multi-agent architecture integrates three evidence streams via a learnable gating mechanism based on company attributes. Under strict anti-leakage controls, MIRAGE-VC achieves +5.0% F1 and +16.6% PrecisionAt5, and sheds light on other off-graph prediction tasks such as recommendation and risk assessment. Code: https://anonymous.4open.science/r/MIRAGE-VC-323F.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01196v1">EduSim-LLM: An Educational Platform Integrating Large Language Models and Robotic Simulation for Beginners</a></div>
    <div class="paper-meta">
      📅 2026-01-03
    </div>
    <details class="paper-abstract">
      In recent years, the rapid development of Large Language Models (LLMs) has significantly enhanced natural language understanding and human-computer interaction, creating new opportunities in the field of robotics. However, the integration of natural language understanding into robotic control is an important challenge in the rapid development of human-robot interaction and intelligent automation industries. This challenge hinders intuitive human control over complex robotic systems, limiting their educational and practical accessibility. To address this, we present the EduSim-LLM, an educational platform that integrates LLMs with robot simulation and constructs a language-drive control model that translates natural language instructions into executable robot behavior sequences in CoppeliaSim. We design two human-robot interaction models: direct control and autonomous control, conduct systematic simulations based on multiple language models, and evaluate multi-robot collaboration, motion planning, and manipulation capabilities. Experiential results show that LLMs can reliably convert natural language into structured robot actions; after applying prompt-engineering templates instruction-parsing accuracy improves significantly; as task complexity increases, overall accuracy rate exceeds 88.9% in the highest complexity tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.22732v2">Reasoning Beyond Limits: Advances and Open Problems for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 The paper is published ICT Express Volume 11, Issue 6, December 2025, Pages 1054-1096
    </div>
    <details class="paper-abstract">
      Recent breakthroughs in generative reasoning have fundamentally reshaped how large language models (LLMs) address complex tasks, enabling them to dynamically retrieve, refine, and organize information into coherent multi-step reasoning chains. Techniques such as inference-time scaling, reinforcement learning, supervised fine-tuning, and distillation have been effectively applied to state-of-the-art models, including DeepSeek-R1, OpenAI o1 and o3, GPT-4o, Qwen-32B, and various Llama variants, significantly enhancing their reasoning capabilities. In this paper, we present a comprehensive review of the top 27 LLMs released between 2023 and 2025, such as Mistral AI Small 3 24B, DeepSeek-R1, Search-o1, QwQ-32B, and Phi-4, and analyze their core innovations and performance improvements. We also provide a detailed overview of recent advancements in multilingual large language models (MLLMs), emphasizing methods that improve cross-lingual reasoning and address the limitations of English-centric training. In parallel, we present a comprehensive review of progress in state space model (SSM)-based architectures, including models such as Mamba, which demonstrate improved efficiency for long-context processing compared to transformer-based approaches. Our analysis covers training strategies including general optimization techniques, mixture-of-experts (MoE) configurations, retrieval-augmented generation (RAG), chain-of-thought prompting, self-improvement methods, and test-time compute scaling and distillation frameworks. Finally, we identify key challenges for future research, including enabling multi-step reasoning without human supervision, improving robustness in chained task execution, balancing structured prompting with generative flexibility, and enhancing the integration of long-context retrieval and external tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23019v4">Stairway to Success: An Online Floor-Aware Zero-Shot Object-Goal Navigation Framework via LLM-Driven Coarse-to-Fine Exploration</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 Accepted to IEEE Robotics and Automation Letters (RAL). Project Page at https://zeying-gong.github.io/projects/ascent
    </div>
    <details class="paper-abstract">
      Deployable service and delivery robots struggle to navigate multi-floor buildings to reach object goals, as existing systems fail due to single-floor assumptions and requirements for offline, globally consistent maps. Multi-floor environments pose unique challenges including cross-floor transitions and vertical spatial reasoning, especially navigating unknown buildings. Object-Goal Navigation benchmarks like HM3D and MP3D also capture this multi-floor reality, yet current methods lack support for online, floor-aware navigation. To bridge this gap, we propose \textbf{\textit{ASCENT}}, an online framework for Zero-Shot Object-Goal Navigation that enables robots to operate without pre-built maps or retraining on new object categories. It introduces: (1) a \textbf{Multi-Floor Abstraction} module that dynamically constructs hierarchical representations with stair-aware obstacle mapping and cross-floor topology modeling, and (2) a \textbf{Coarse-to-Fine Reasoning} module that combines frontier ranking with LLM-driven contextual analysis for multi-floor navigation decisions. We evaluate on HM3D and MP3D benchmarks, outperforming state-of-the-art zero-shot approaches, and demonstrate real-world deployment on a quadruped robot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05887v2">A three-Level Framework for LLM-Enhanced eXplainable AI: From technical explanations to natural language</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 22 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The growing application of artificial intelligence in sensitive domains has intensified the demand for systems that are not only accurate but also explainable and trustworthy. Although explainable AI (XAI) methods have proliferated, many do not consider the diverse audiences that interact with AI systems: from developers and domain experts to end-users and society. This paper addresses how trust in AI is influenced by the design and delivery of explanations and proposes a multilevel framework that aligns explanations with the epistemic, contextual, and ethical expectations of different stakeholders. The framework consists of three layers: algorithmic and domain-based, human-centered, and social explainability, with Large Language Models serving as crucial mediators that transform technical outputs of AI explanations into accessible, contextual narratives across all levels. We show how LLMs enable dynamic, conversational explanations that bridge the gap between complex model behavior and human understanding, facilitating interactive dialogue and enhancing societal transparency. Through comprehensive case studies, we show how this LLM-enhanced approach achieves technical fidelity, user engagement, and societal accountability, reframing XAI as a dynamic, trust-building process that leverages natural language capabilities to democratize AI explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01129v1">RovoDev Code Reviewer: A Large-Scale Online Evaluation of LLM-based Code Review Automation at Atlassian</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 Accepted at the 48th International Conference on Software Engineering (ICSE'26), SEIP Track. 12 Pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs)-powered code review automation has the potential to transform code review workflows. Despite the advances of LLM-powered code review comment generation approaches, several practical challenges remain for designing enterprise-grade code review automation tools. In particular, this paper aims at answering the practical question: how can we design a review-guided, context-aware, quality-checked code review comment generation without fine-tuning? In this paper, we present RovoDev Code Reviewer, an enterprise-grade LLM-based code review automation tool designed and deployed at scale within Atlassian's development ecosystem with seamless integration into Atlassian's Bitbucket. Through the offline, online, user feedback evaluations over a one-year period, we conclude that RovoDev Code Reviewer is (1) effective in generating code review comments that could lead to code resolution for 38.70% (i.e., comments that triggered code changes in the subsequent commits); and (2) offers the promise of accelerating feedback cycles (i.e., decreasing the PR cycle time by 30.8%), alleviating reviewer workload (i.e., reducing the number of human-written comments by 35.6%), and improving overall software quality (i.e., finding errors with actionable suggestions).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2310.07554v3">A Multi-Task Embedder For Retrieval Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-03
    </div>
    <details class="paper-abstract">
      LLMs confront inherent limitations in terms of its knowledge, memory, and action. The retrieval augmentation stands as a vital mechanism to address these limitations, which brings in useful information from external sources to augment the LLM. However, existing retrieval methods encounter two pressing issues. On one hand, the general retrievers are not properly optimized for retrieval augmentation hence exhibit limited effectiveness; on the other hand, the task-specific retrievers excel in the targeted retrieval augmentation scenario, while lack the versatility to handle diverse scenarios. In this work, we propose \textbf{LLM-Embedder} for the unified support of diverse retrieval augmentation scenarios. Our method presents three technical contributions. Firstly, we introduce a new \textit{reward formulation}, namely {rank-aware reward}. It exploits the ranking position of the desired output among $N$ sampled outputs from the LLM, which leads to fine-grained and robust computation of reward from the LLM's feedback. Secondly, we design a novel \textit{distillation objective}, called graded distillation. It incorporates both the absolute value and the relative order of the reward for more sufficient utilization of the LLM's feedback. Thirdly, we systematically optimize the \textit{multi-task learning}, which effectively unifies the multiple retrieval functionalities into one model. In our experiment, LLM-Embedder notably improves the LLM's performances in various downstream tasks, and outperforms both general and task-specific retrievers with a substantial advantage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01118v1">ScienceDB AI: An LLM-Driven Agentic Recommender System for Large-Scale Scientific Data Sharing Services</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 12 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The rapid growth of AI for Science (AI4S) has underscored the significance of scientific datasets, leading to the establishment of numerous national scientific data centers and sharing platforms. Despite this progress, efficiently promoting dataset sharing and utilization for scientific research remains challenging. Scientific datasets contain intricate domain-specific knowledge and contexts, rendering traditional collaborative filtering-based recommenders inadequate. Recent advances in Large Language Models (LLMs) offer unprecedented opportunities to build conversational agents capable of deep semantic understanding and personalized recommendations. In response, we present ScienceDB AI, a novel LLM-driven agentic recommender system developed on Science Data Bank (ScienceDB), one of the largest global scientific data-sharing platforms. ScienceDB AI leverages natural language conversations and deep reasoning to accurately recommend datasets aligned with researchers' scientific intents and evolving requirements. The system introduces several innovations: a Scientific Intention Perceptor to extract structured experimental elements from complicated queries, a Structured Memory Compressor to manage multi-turn dialogues effectively, and a Trustworthy Retrieval-Augmented Generation (Trustworthy RAG) framework. The Trustworthy RAG employs a two-stage retrieval mechanism and provides citable dataset references via Citable Scientific Task Record (CSTR) identifiers, enhancing recommendation trustworthiness and reproducibility. Through extensive offline and online experiments using over 10 million real-world datasets, ScienceDB AI has demonstrated significant effectiveness. To our knowledge, ScienceDB AI is the first LLM-driven conversational recommender tailored explicitly for large-scale scientific dataset sharing services. The platform is publicly accessible at: https://ai.scidb.cn/en.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00930v1">AlignUSER: Human-Aligned LLM Agents via World Models for Recommender System Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Evaluating recommender systems remains challenging due to the gap between offline metrics and real user behavior, as well as the scarcity of interaction data. Recent work explores large language model (LLM) agents as synthetic users, yet they typically rely on few-shot prompting, which yields a shallow understanding of the environment and limits their ability to faithfully reproduce user actions. We introduce AlignUSER, a framework that learns world-model-driven agents from human interactions. Given rollout sequences of actions and states, we formalize world modeling as a next state prediction task that helps the agent internalize the environment. To align actions with human personas, we generate counterfactual trajectories around demonstrations and prompt the LLM to compare its decisions with human choices, identify suboptimal actions, and extract lessons. The learned policy is then used to drive agent interactions with the recommender system. We evaluate AlignUSER across multiple datasets and demonstrate closer alignment with genuine humans than prior work, both at the micro and macro levels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.15131v2">Modeling the One-to-Many Property in Open-Domain Dialogue with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Open-domain Dialogue (OD) exhibits a one-to-many (o2m) property, whereby multiple appropriate responses exist for a single dialogue context. Despite prior research showing that modeling this property boosts response diversity, most modern LLM-based dialogue agents do not explicitly do so. In this work, we model the o2m property of OD in LLMs by decomposing OD generation into two key tasks: Multi-Response Generation (MRG) and Preference-based Selection (PS), which entail generating a set of n semantically and lexically diverse high-quality responses for a given dialogue context, followed by selecting a single response based on human preference, respectively. To facilitate MRG and PS, we introduce o2mDial, a dialogue corpus explicitly designed to capture the o2m property by featuring multiple plausible responses for each context. Leveraging o2mDial, we propose new in-context learning and instruction-tuning strategies, as well as novel evaluation metrics for MRG, alongside a model-based approach for PS. Empirical results demonstrate that applying the proposed two-stage framework to smaller LLMs for OD generation enhances overall response diversity while maintaining contextual coherence, improving response quality by up to 90%, bringing them closer to the performance of larger models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.07675v3">QUITE: A Query Rewrite System Beyond Rules with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Query rewrite transforms SQL queries into semantically equivalent forms that run more efficiently. Existing approaches mainly rely on predefined rewrite rules, but they handle a limited subset of queries and can cause performance regressions. This limitation stems from three challenges of rule-based query rewrite: (1) it is hard to discover and verify new rules, (2) fixed rewrite rules do not generalize to new query patterns, and (3) some rewrite techniques cannot be expressed as fixed rules. Motivated by the fact that human experts exhibit significantly better rewrite ability but suffer from scalability, and Large Language Models (LLMs) have demonstrated nearly human-level semantic and reasoning abilities, we propose a new approach of using LLMs to rewrite SQL queries beyond rules. Due to the hallucination problems in LLMs, directly applying LLMs often leads to nonequivalent and suboptimal queries. To address this issue, we propose QUITE (query rewrite), a training-free and feedback-aware system based on LLM agents that rewrites SQL queries into semantically equivalent forms with significantly better performance, covering a broader range of query patterns and rewrite strategies compared to rule-based methods. Firstly, we design a multi-agent framework controlled by a finite state machine (FSM) to equip LLMs with the ability to use external tools and enhance the rewrite process with real-time database feedback. Secondly, we develop a rewrite middleware to enhance the ability of LLMs to generate optimized query equivalents. Finally, we employ a novel hint injection technique to improve execution plans for rewritten queries. Extensive experiments show that QUITE reduces query execution time by up to 35.8% over state-of-the-art approaches and produces 24.1% more rewrites than prior methods, covering query cases that earlier systems did not handle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22905v2">JavisGPT: A Unified Multi-modal LLM for Sounding-Video Comprehension and Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-02
      | 💬 Accepted by NeurIPS as a Spotlight paper. Code: https://github.com/JavisVerse/JavisGPT
    </div>
    <details class="paper-abstract">
      This paper presents JavisGPT, the first unified multimodal large language model (MLLM) for joint audio-video (JAV) comprehension and generation. JavisGPT has a concise encoder-LLM-decoder architecture, which has a SyncFusion module for spatio-temporal audio-video fusion and synchrony-aware learnable queries to bridge a pretrained JAV-DiT generator. This design enables temporally coherent video-audio understanding and generation from multimodal instructions. We design an effective three-stage training pipeline consisting of multimodal pretraining, audio-video fine-tuning, and large-scale instruction-tuning, to progressively build multimodal comprehension and generation from existing vision-language models. For instruction tuning, we construct JavisInst-Omni, a high-quality instruction dataset with over 200K GPT-4o-curated audio-video-text dialogues that cover diverse and multi-level comprehension and generation scenarios. On JAV comprehension and generation benchmarks, our experiments show that JavisGPT outperforms existing MLLMs, particularly in complex and temporally synchronized settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00644v1">FlexSpec: Frozen Drafts Meet Evolving Targets in Edge-Cloud Collaborative LLM Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) in mobile and edge computing environments is constrained by limited on-device resources, scarce wireless bandwidth, and frequent model evolution. Although edge-cloud collaborative inference with speculative decoding (SD) can reduce end-to-end latency by executing a lightweight draft model at the edge and verifying it with a cloud-side target model, existing frameworks fundamentally rely on tight coupling between the two models. Consequently, repeated model synchronization introduces excessive communication overhead, increasing end-to-end latency, and ultimately limiting the scalability of SD in edge environments. To address these limitations, we propose FlexSpec, a communication-efficient collaborative inference framework tailored for evolving edge-cloud systems. The core design of FlexSpec is a shared-backbone architecture that allows a single and static edge-side draft model to remain compatible with a large family of evolving cloud-side target models. By decoupling edge deployment from cloud-side model updates, FlexSpec eliminates the need for edge-side retraining or repeated model downloads, substantially reducing communication and maintenance costs. Furthermore, to accommodate time-varying wireless conditions and heterogeneous device constraints, we develop a channel-aware adaptive speculation mechanism that dynamically adjusts the speculative draft length based on real-time channel state information and device energy budgets. Extensive experiments demonstrate that FlexSpec achieves superior performance compared to conventional SD approaches in terms of inference efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00641v1">Probabilistic Guarantees for Reducing Contextual Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently produce contextual hallucinations, where generated content contradicts or ignores information explicitly stated in the prompt. Such errors are particularly problematic in deterministic automation workflows, where inputs are fixed and correctness is unambiguous. We introduce a simple and model-agnostic framework that provides explicit probabilistic guarantees for reducing hallucinations in this setting. We formalize the notion of a specific task, defined by a fixed input and a deterministic correctness criterion, and show that issuing the same prompt in independent context windows yields an exponential reduction in the probability that all model outputs are incorrect. To identify a correct answer among repeated runs, we incorporate an LLM-as-a-judge and prove that the probability that the judged pipeline fails decays at a rate determined by the judge's true- and false-positive probabilities. When the judge is imperfect, we strengthen it through majority vote over independent judge calls, obtaining ensemble-level error rates that decrease exponentially in the number of votes. This yields an explicit bound on the probability that the pipeline selects a hallucinated answer. Experiments on controlled extraction tasks with synthetic noisy judges match these predictions exactly: pipeline failure decreases exponentially with the number of repetitions, and hallucination-selection decreases exponentially with the number of judges in the ensemble. Together, these results provide a lightweight, modular, and theoretically grounded method for driving hallucination probabilities arbitrarily low in fixed-input LLM workflows-without modifying model weights, decoding strategies, or prompt engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00624v1">Do Chatbot LLMs Talk Too Much? The YapBench Benchmark</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as ChatGPT, Claude, and Gemini increasingly act as general-purpose copilots, yet they often respond with unnecessary length on simple requests, adding redundant explanations, hedging, or boilerplate that increases cognitive load and inflates token-based inference cost. Prior work suggests that preference-based post-training and LLM-judged evaluations can induce systematic length bias, where longer answers are rewarded even at comparable quality. We introduce YapBench, a lightweight benchmark for quantifying user-visible over-generation on brevity-ideal prompts. Each item consists of a single-turn prompt, a curated minimal-sufficient baseline answer, and a category label. Our primary metric, YapScore, measures excess response length beyond the baseline in characters, enabling comparisons across models without relying on any specific tokenizer. We summarize model performance via the YapIndex, a uniformly weighted average of category-level median YapScores. YapBench contains over three hundred English prompts spanning three common brevity-ideal settings: (A) minimal or ambiguous inputs where the ideal behavior is a short clarification, (B) closed-form factual questions with short stable answers, and (C) one-line coding tasks where a single command or snippet suffices. Evaluating 76 assistant LLMs, we observe an order-of-magnitude spread in median excess length and distinct category-specific failure modes, including vacuum-filling on ambiguous inputs and explanation or formatting overhead on one-line technical requests. We release the benchmark and maintain a live leaderboard for tracking verbosity behavior over time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00596v1">Beyond IVR: Benchmarking Customer Support LLM Agents for Business-Adherence</a></div>
    <div class="paper-meta">
      📅 2026-01-02
      | 💬 17 pages, 3 figures, preprint
    </div>
    <details class="paper-abstract">
      Traditional customer support systems, such as Interactive Voice Response (IVR), rely on rigid scripts and lack the flexibility required for handling complex, policy-driven tasks. While large language model (LLM) agents offer a promising alternative, evaluating their ability to act in accordance with business rules and real-world support workflows remains an open challenge. Existing benchmarks primarily focus on tool usage or task completion, overlooking an agent's capacity to adhere to multi-step policies, navigate task dependencies, and remain robust to unpredictable user or environment behavior. In this work, we introduce JourneyBench, a benchmark designed to assess policy-aware agents in customer support. JourneyBench leverages graph representations to generate diverse, realistic support scenarios and proposes the User Journey Coverage Score, a novel metric to measure policy adherence. We evaluate multiple state-of-the-art LLMs using two agent designs: a Static-Prompt Agent (SPA) and a Dynamic-Prompt Agent (DPA) that explicitly models policy control. Across 703 conversations in three domains, we show that DPA significantly boosts policy adherence, even allowing smaller models like GPT-4o-mini to outperform more capable ones like GPT-4o. Our findings demonstrate the importance of structured orchestration and establish JourneyBench as a critical resource to advance AI-driven customer support beyond IVR-era limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00588v1">CSSBench: Evaluating the Safety of Lightweight LLMs against Chinese-Specific Adversarial Patterns</a></div>
    <div class="paper-meta">
      📅 2026-01-02
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in cost-sensitive and on-device scenarios, and safety guardrails have advanced mainly in English. However, real-world Chinese malicious queries typically conceal intent via homophones, pinyin, symbol-based splitting, and other Chinese-specific patterns. These Chinese-specific adversarial patterns create the safety evaluation gap that is not well captured by existing benchmarks focused on English. This gap is particularly concerning for lightweight models, which may be more vulnerable to such specific adversarial perturbations. To bridge this gap, we introduce the Chinese-Specific Safety Benchmark (CSSBench) that emphasizes these adversarial patterns and evaluates the safety of lightweight LLMs in Chinese. Our benchmark covers six domains that are common in real Chinese scenarios, including illegal activities and compliance, privacy leakage, health and medical misinformation, fraud and hate, adult content, and public and political safety, and organizes queries into multiple task types. We evaluate a set of popular lightweight LLMs and measure over-refusal behavior to assess safety-induced performance degradation. Our results show that the Chinese-specific adversarial pattern is a critical challenge for lightweight LLMs. This benchmark offers a comprehensive evaluation of LLM safety in Chinese, assisting robust deployments in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00575v1">InfoSynth: Information-Guided Benchmark Synthesis for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant advancements in reasoning and code generation. However, efficiently creating new benchmarks to evaluate these capabilities remains a challenge. Traditional benchmark creation relies on manual human effort, a process that is both expensive and time-consuming. Furthermore, existing benchmarks often contaminate LLM training data, necessitating novel and diverse benchmarks to accurately assess their genuine capabilities. This work introduces InfoSynth, a novel framework for automatically generating and evaluating reasoning benchmarks guided by information-theoretic principles. We propose metrics based on KL-divergence and entropy to quantify benchmark novelty and diversity without relying on costly model evaluations. Building on this framework, we develop an end-to-end pipeline that synthesizes robust Python coding problems from seed datasets using genetic algorithms and iterative code feedback. Our method generates accurate test cases and solutions to new problems 97% of the time, and the synthesized benchmarks consistently exhibit higher novelty and diversity compared to their seed datasets. Moreover, our algorithm provides a method for controlling the novelty/diversity and difficulty of generated problems. InfoSynth offers a scalable, self-verifying pipeline for constructing high-quality, novel and diverse benchmarks for LLMs. Project Page: https://ishirgarg.github.io/infosynth_web/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00570v1">User Perceptions of an LLM-Based Chatbot for Cognitive Reappraisal of Stress: Feasibility Study</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Cognitive reappraisal is a well-studied emotion regulation strategy that helps individuals reinterpret stressful situations to reduce their impact. Many digital mental health tools struggle to support this process because rigid scripts fail to accommodate how users naturally describe stressors. This study examined the feasibility of an LLM-based single-session intervention (SSI) for workplace stress reappraisal. We assessed short-term changes in stress-related outcomes and examined design tensions during use. We conducted a feasibility study with 100 employees at a large technology company who completed a structured cognitive reappraisal session delivered by a GPT-4o-based chatbot. Pre-post measures included perceived stress intensity, stress mindset, perceived demand, and perceived resources. These outcomes were analyzed using paired Wilcoxon signed-rank tests with correction for multiple comparisons. We also examined sentiment and stress trajectories across conversation quartiles using two RoBERTa-based classifiers and an LLM-based stress rater. Open-ended responses were analyzed using thematic analysis. Results showed significant reductions in perceived stress intensity and significant improvements in stress mindset. Changes in perceived resources and perceived demand trended in expected directions but were not statistically significant. Automated analyses indicated consistent declines in negative sentiment and stress over the course of the interaction. Qualitative findings suggested that participants valued the structured prompts for organizing thoughts, gaining perspective, and feeling acknowledged. Participants also reported tensions around scriptedness, preferred interaction length, and reactions to AI-driven empathy. These findings highlight both the promise and the design constraints of integrating LLMs into DMH interventions for workplace settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00566v1">Low Rank Comes with Low Security: Gradient Assembly Poisoning Attacks against Distributed LoRA-based LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-02
      | 💬 8 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Low-Rank Adaptation (LoRA) has become a popular solution for fine-tuning large language models (LLMs) in federated settings, dramatically reducing update costs by introducing trainable low-rank matrices. However, when integrated with frameworks like FedIT, LoRA introduces a critical vulnerability: clients submit $A$ and $B$ matrices separately, while only their product $AB$ determines the model update, yet this composite is never directly verified. We propose Gradient Assembly Poisoning (GAP), a novel attack that exploits this blind spot by crafting individually benign $A$ and $B$ matrices whose product yields malicious updates. GAP operates without access to training data or inter-client coordination and remains undetected by standard anomaly detectors. We identify four systemic vulnerabilities in LoRA-based federated systems and validate GAP across LLaMA, ChatGLM, and GPT-2. GAP consistently induces degraded or biased outputs while preserving surface fluency, reducing BLEU by up to 14.5\%, increasing factual and grammatical errors by over 800\%, and maintaining 92.6\% long-form response length. These results reveal a new class of stealthy, persistent threats in distributed LoRA fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00559v1">Cracking IoT Security: Can LLMs Outsmart Static Analysis Tools?</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Smart home IoT platforms such as openHAB rely on Trigger Action Condition (TAC) rules to automate device behavior, but the interplay among these rules can give rise to interaction threats, unintended or unsafe behaviors emerging from implicit dependencies, conflicting triggers, or overlapping conditions. Identifying these threats requires semantic understanding and structural reasoning that traditionally depend on symbolic, constraint-driven static analysis. This work presents the first comprehensive evaluation of Large Language Models (LLMs) across a multi-category interaction threat taxonomy, assessing their performance on both the original openHAB (oHC/IoTB) dataset and a structurally challenging Mutation dataset designed to test robustness under rule transformations. We benchmark Llama 3.1 8B, Llama 70B, GPT-4o, Gemini-2.5-Pro, and DeepSeek-R1 across zero-, one-, and two-shot settings, comparing their results against oHIT's manually validated ground truth. Our findings show that while LLMs exhibit promising semantic understanding, particularly on action- and condition-related threats, their accuracy degrades significantly for threats requiring cross-rule structural reasoning, especially under mutated rule forms. Model performance varies widely across threat categories and prompt settings, with no model providing consistent reliability. In contrast, the symbolic reasoning baseline maintains stable detection across both datasets, unaffected by rule rewrites or structural perturbations. These results underscore that LLMs alone are not yet dependable for safety critical interaction-threat detection in IoT environments. We discuss the implications for tool design and highlight the potential of hybrid architectures that combine symbolic analysis with LLM-based semantic interpretation to reduce false positives while maintaining structural rigor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01661v2">Learning the Boundary of Solvability: Aligning LLMs to Detect Unsolvable Problems</a></div>
    <div class="paper-meta">
      📅 2026-01-02
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Ensuring large language model (LLM) reliability requires distinguishing objective unsolvability (inherent contradictions) from subjective capability limitations (tasks exceeding model competence). Current LLMs often conflate these dimensions, leading to hallucinations in which they return confident answers to inherently unsolvable queries. To address this issue, we propose a multi-domain dataset containing both solvable and unsolvable questions, UnsolvableQA, together with an alignment framework, UnsolvableRL. First, we construct UnsolvableQA by "Reverse Construction" that systematically injects logical contradictions into otherwise valid reasoning chains. Second, we introduce UnsolvableRL, a reinforcement learning paradigm that balances objective unsolvability detection with calibrated confidence under capability limits. Empirically, our approach achieves near-perfect unsolvability detection (>90% detection rate) and boosts solvable reasoning accuracy from 43.4% to 69.4% on Qwen3-4B-Instruct. Crucially, we identify a data-training interaction: strict alignment constraints induce Capability Collapse without unsolvable data, but act as a regularizer for rigor when such data are included, thereby improving overall robustness. Our code and data are available at https://github.com/sfasfaffa/unsolvableQA .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00555v1">LLM-Based Agentic Exploration for Robot Navigation & Manipulation with Skill Orchestration</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      This paper presents an end-to-end LLM-based agentic exploration system for an indoor shopping task, evaluated in both Gazebo simulation and a corresponding real-world corridor layout. The robot incrementally builds a lightweight semantic map by detecting signboards at junctions and storing direction-to-POI relations together with estimated junction poses, while AprilTags provide repeatable anchors for approach and alignment. Given a natural-language shopping request, an LLM produces a constrained discrete action at each junction (direction and whether to enter a store), and a ROS finite-state main controller executes the decision by gating modular motion primitives, including local-costmap-based obstacle avoidance, AprilTag approaching, store entry, and grasping. Qualitative results show that the integrated stack can perform end-to-end task execution from user instruction to multi-store navigation and object retrieval, while remaining modular and debuggable through its text-based map and logged decision history.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.14746v6">Scaling Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Recent LLMs have hundreds of billions of parameters consuming vast resources. Furthermore, the so called "AI scaling law" for transformers suggests that the number of parameters must scale linearly with the size of the data. In response, we inquire into efficient LLMs, i.e. those with the fewest parameters that achieve the desired accuracy on a training corpus. Specifically, by comparing theoretical and empirical estimates of the Kullback-Leibler divergence, we derive a natural AI scaling law that the number of parameters in an efficient LLM scales as $D^γ$ where $D$ is the size of the training data and $ γ\in [0.44, 0.72]$, suggesting the existence of more efficient architectures. Against this backdrop, we propose recurrent transformers, combining the efficacy of transformers with the efficiency of recurrent networks, progressively applying a single transformer layer to a fixed-width sliding window across the input sequence. Recurrent transformers (a) run in linear time in the sequence length, (b) are memory-efficient and amenable to parallel processing in large batches, (c) learn to forget history for language tasks, or accumulate history for long range tasks like copy and selective copy, and (d) are amenable to curriculum training to overcome vanishing gradients. In our experiments, we find that recurrent transformers perform favorably on benchmark tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.17972v4">LABOR-LLM: Language-Based Occupational Representations with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      This paper builds an empirical model that predicts a worker's next occupation as a function of the worker's occupational history. Because histories are sequences of occupations, the covariate space is high-dimensional, and further, the outcome (the next occupation) is a discrete choice that can take on many values. To estimate the parameters of the model, we leverage an approach from generative artificial intelligence. Estimation begins from a ``foundation model'' trained on non-representative data and then ``fine-tunes'' the estimation using data about careers from a representative survey. We convert tabular data from the survey into text files that resemble resumes and fine-tune the parameters of the foundation model, a large language model (LLM), using these text files with the objective of predicting the next token (word). The resulting fine-tuned LLM is used to calculate estimates of worker transition probabilities. Its predictive performance surpasses all prior models, both for the task of granularly predicting the next occupation as well as for specific tasks such as predicting whether the worker changes occupations or stays in the labor force. We quantify the value of fine-tuning and further show that by adding more career data from a different population, fine-tuning smaller LLMs (fewer parameters) surpasses the performance of fine-tuning larger models. When we omit the English language occupational title and replace it with a unique code, predictive performance declines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23323v2">LLM Interpretability with Identifiable Temporal-Instantaneous Representation</a></div>
    <div class="paper-meta">
      📅 2026-01-02
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Despite Large Language Models' remarkable capabilities, understanding their internal representations remains challenging. Mechanistic interpretability tools such as sparse autoencoders (SAEs) were developed to extract interpretable features from LLMs but lack temporal dependency modeling, instantaneous relation representation, and more importantly theoretical guarantees, undermining both the theoretical foundations and the practical confidence necessary for subsequent analyses. While causal representation learning (CRL) offers theoretically grounded approaches for uncovering latent concepts, existing methods cannot scale to LLMs' rich conceptual space due to inefficient computation. To bridge the gap, we introduce an identifiable temporal causal representation learning framework specifically designed for LLMs' high-dimensional concept space, capturing both time-delayed and instantaneous causal relations. Our approach provides theoretical guarantees and demonstrates efficacy on synthetic datasets scaled to match real-world complexity. By extending SAE techniques with our temporal causal framework, we successfully discover meaningful concept relationships in LLM activations. Our findings show that modeling both temporal and instantaneous conceptual relationships advances the interpretability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22922v3">Improving Human Verification of LLM Reasoning through Interactive Explanation Interfaces</a></div>
    <div class="paper-meta">
      📅 2026-01-02
      | 💬 19 pages, 14 figures
    </div>
    <details class="paper-abstract">
      The reasoning capabilities of Large Language Models (LLMs) have led to their increasing employment in several critical applications, particularly education, where they support problem-solving, tutoring, and personalized study. Chain-of-thought (CoT) reasoning capabilities [1, 2] are well-known to help LLMs decompose a problem into steps and explore the solution spaces more effectively, leading to impressive performance on mathematical and reasoning benchmarks. As the length of CoT tokens per question increases substantially to even thousands of tokens per question [ 1], it is unknown how users could comprehend LLM reasoning and detect errors or hallucinations. To address this problem and understand how reasoning can improve human-AI interaction, we present three new interactive reasoning interfaces: interactive CoT (iCoT), interactive Program-of-Thought (iPoT), and interactive Graph (iGraph). That is, we ask LLMs themselves to generate an interactive web interface wrapped around the original CoT content, which may be presented in text (iCoT), graphs (iGraph) or code (iPoT). This interface allows users to interact with and provide a novel experience in reading and validating the reasoning chains of LLMs. Across a study of 125 participants, interactive interfaces significantly improve user performance. Specifically, iGraph users score the highest error detection rate (85.6%), followed by iPoT (82.5%), iCoT (80.6%), all outperforming standard CoT (73.5%). Interactive interfaces also lead to faster user validation time-iGraph users are faster (57.9 secs per question) than the users of iCoT and iPoT (60 secs) and the standard CoT (64.7 secs). A post-study questionnaire shows that users prefer iGraph, citing its superior ability to enable them to follow the LLM's reasoning. We discuss the implications of these results and provide recommendations for the future design of reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00227v1">FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      Recent advances show that large language models (LLMs) can act as autonomous agents capable of generating GPU kernels, but integrating these AI-generated kernels into real-world inference systems remains challenging. FlashInfer-Bench addresses this gap by establishing a standardized, closed-loop framework that connects kernel generation, benchmarking, and deployment. At its core, FlashInfer Trace provides a unified schema describing kernel definitions, workloads, implementations, and evaluations, enabling consistent communication between agents and systems. Built on real serving traces, FlashInfer-Bench includes a curated dataset, a robust correctness- and performance-aware benchmarking framework, a public leaderboard to track LLM agents' GPU programming capabilities, and a dynamic substitution mechanism (apply()) that seamlessly injects the best-performing kernels into production LLM engines such as SGLang and vLLM. Using FlashInfer-Bench, we further evaluate the performance and limitations of LLM agents, compare the trade-offs among different GPU programming languages, and provide insights for future agent design. FlashInfer-Bench thus establishes a practical, reproducible pathway for continuously improving AI-generated kernels and deploying them into large-scale LLM inference.
    </details>
</div>
