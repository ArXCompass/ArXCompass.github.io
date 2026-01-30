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
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.15098v3">TextMineX: Data, Evaluation Framework and Ontology-guided LLM Pipeline for Humanitarian Mine Action</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Humanitarian Mine Action (HMA) addresses the challenge of detecting and removing landmines from conflict regions. Much of the life-saving operational knowledge produced by HMA agencies is buried in unstructured reports, limiting the transferability of information between agencies. To address this issue, we propose TextMineX: the first dataset, evaluation framework and ontology-guided large language model (LLM) pipeline for knowledge extraction from text in the HMA domain. TextMineX structures HMA reports into (subject, relation, object)-triples, thus creating domain-specific knowledge. To ensure real-world relevance, we utilized the dataset from our collaborator Cambodian Mine Action Centre (CMAC). We further introduce a bias-aware evaluation framework that combines human-annotated triples with an LLM-as-Judge protocol to mitigate position bias in reference-free scoring. Our experiments show that ontology-aligned prompts improve extraction accuracy by up to 44.2%, reduce hallucinations by 22.5%, and enhance format adherence by 20.9% compared to baseline models. We publicly release the dataset and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.06605v2">Reading Between the Lines: Towards Reliable Black-box LLM Fingerprinting via Zeroth-order Gradient Estimation</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 This paper is accepeted by the ACM Web Conference (WWW) 2026
    </div>
    <details class="paper-abstract">
      The substantial investment required to develop Large Language Models (LLMs) makes them valuable intellectual property, raising significant concerns about copyright protection. LLM fingerprinting has emerged as a key technique to address this, which aims to verify a model's origin by extracting an intrinsic, unique signature (a "fingerprint") and comparing it to that of a source model to identify illicit copies. However, existing black-box fingerprinting methods often fail to generate distinctive LLM fingerprints. This ineffectiveness arises because black-box methods typically rely on model outputs, which lose critical information about the model's unique parameters due to the usage of non-linear functions. To address this, we first leverage Fisher Information Theory to formally demonstrate that the gradient of the model's input is a more informative feature for fingerprinting than the output. Based on this insight, we propose ZeroPrint, a novel method that approximates these information-rich gradients in a black-box setting using zeroth-order estimation. ZeroPrint overcomes the challenge of applying this to discrete text by simulating input perturbations via semantic-preserving word substitutions. This operation allows ZeroPrint to estimate the model's Jacobian matrix as a unique fingerprint. Experiments on the standard benchmark show ZeroPrint achieves a state-of-the-art effectiveness and robustness, significantly outperforming existing black-box methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14943v1">State of the Art of LLM-Enabled Interaction with Visualization</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      We report on a systematic, PRISMA-guided survey of research at the intersection of LLMs and visualization, with a particular focus on visio-verbal interaction -- where verbal and visual modalities converge to support data sense-making. The emergence of Large Language Models (LLMs) has introduced new paradigms for interacting with data visualizations through natural language, leading to intuitive, multimodal, and accessible interfaces. We analyze 48 papers across six dimensions: application domain, visualization task, visualization representation, interaction modality, LLM integration, and system evaluation. Our classification framework maps LLM roles across the visualization pipeline, from data querying and transformation to visualization generation, explanation, and navigation. We highlight emerging design patterns, identify gaps in accessibility and visualization reading, and discuss the limitations of current LLMs in spatial reasoning and contextual grounding. We further reflect on evaluations of combined LLM-visualization systems, highlighting how current research projects tackle this challenge and discuss current gaps in conducting meaningful evaluations of such systems. With our survey we aim to guide future research and system design in LLM-enhanced visualization, supporting broad audiences and intelligent, conversational interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14936v1">LLM-Based Repair of C++ Implicit Data Loss Compiler Warnings: An Industrial Case Study</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      This paper presents a method to automatically fix implicit data loss warnings in large C++ projects using Large Language Models (LLMs). Our approach uses the Language Server Protocol (LSP) to gather context, Tree-sitter to extract relevant code, and LLMs to make decisions and generate fixes. The method evaluates the necessity of range checks concerning performance implications and generates appropriate fixes. We tested this method in a large C++ project, resulting in a 92.73% acceptance rate of the fixes by human developers during the code review. Our LLM-generated fixes reduced the number of warning fix changes that introduced additional instructions due to range checks and exception handling by 39.09% compared to a baseline fix strategy. This result was 13.56% behind the optimal solutions created by human developers. These findings demonstrate that our LLM-based approach can reduce the manual effort to address compiler warnings while maintaining code quality and performance in a real-world scenario. Our automated approach shows promise for integration into existing development workflows, potentially improving code maintenance practices in complex C++ software projects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07404v3">On LLMs' Internal Representation of Code Correctness</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Accepted for ICSE'26
    </div>
    <details class="paper-abstract">
      Despite the effectiveness of large language models (LLMs) for code generation, they often output incorrect code. One reason is that model output probabilities are often not well-correlated with correctness, and reflect only the final output of the generation process. Inspired by findings that LLMs internally encode concepts like truthfulness, this paper explores if LLMs similarly represent code correctness. Specifically, we identify a correctness representation inside LLMs by contrasting the hidden states between pairs of correct and incorrect code for the same programming tasks. By experimenting on four LLMs, we show that exploiting this extracted correctness representation outperforms standard log-likelihood ranking, as well as verbalized model confidence. Furthermore, we explore how this internal correctness signal can be used to select higher-quality code samples, without requiring test execution. Ultimately, this work demonstrates how leveraging internal representations can enhance code generation systems and make LLMs more reliable, thus improving confidence in automatically generated code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11509v2">Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Accepted at the AAAI 2026 Workshop on AI for Scientific Research (AI4Research)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit remarkable capabilities in natural language understanding and reasoning, but suffer from hallucination: the generation of factually incorrect content. While numerous methods have been developed to reduce hallucinations, their impact on creative generations remains unexplored. This gap is particularly critical for AI-assisted scientific discovery, which requires both factual accuracy and creative hypothesis generation. We investigate how three hallucination-reduction techniques: Chain of Verification (CoVe), Decoding by Contrasting Layers (DoLa), and Retrieval-Augmented Generation (RAG), affect creativity in LLMs. Evaluating multiple model families (LLaMA, Qwen, Mistral) at varying scales (1B - 70B parameters) on two creativity benchmarks (NeoCoder and CS4), we find that these methods have opposing effects on divergent creativity. CoVe enhances divergent thinking, DoLa suppresses it, and RAG shows minimal impact. Our findings provide guidance for selecting appropriate hallucination-reduction methods in scientific applications, where the balance between factual accuracy and creative exploration is crucial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14888v1">What Makes Low-Bit Quantization-Aware Training Work for Reasoning LLMs? A Systematic Study</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Reasoning models excel at complex tasks such as coding and mathematics, yet their inference is often slow and token-inefficient. To improve the inference efficiency, post-training quantization (PTQ) usually comes with the cost of large accuracy drops, especially for reasoning tasks under low-bit settings. In this study, we present a systematic empirical study of quantization-aware training (QAT) for reasoning models. Our key findings include: (1) Knowledge distillation is a robust objective for reasoning models trained via either supervised fine-tuning or reinforcement learning; (2) PTQ provides a strong initialization for QAT, improving accuracy while reducing training cost; (3) Reinforcement learning remains feasible for quantized models given a viable cold start and yields additional gains; and (4) Aligning the PTQ calibration domain with the QAT training domain accelerates convergence and often improves the final accuracy. Finally, we consolidate these findings into an optimized workflow (Reasoning-QAT), and show that it consistently outperforms state-of-the-art PTQ methods across multiple LLM backbones and reasoning datasets. For instance, on Qwen3-0.6B, it surpasses GPTQ by 44.53% on MATH-500 and consistently recovers performance in the 2-bit regime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.00971v3">Beyond Functional Correctness: Exploring Hallucinations in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Accepted by Transactions on Software Engineering (TSE)
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has significantly advanced various applications on software engineering tasks, particularly in code generation. Despite the promising performance, LLMs are prone to generate hallucinations, which means LLMs might produce outputs that deviate from users' intent, exhibit internal inconsistencies, or misaligned with the real-world knowledge, making the deployment of LLMs potentially risky in a wide range of applications. Existing work mainly focuses on investigating the hallucination in the domain of Natural Language Generation (NLG), leaving a gap in comprehensively understanding the types, causes, and impacts of hallucinations in the context of code generation. To bridge the gap, we conducted a thematic analysis of the LLM-generated code to summarize and categorize the hallucinations, as well as their causes and impacts. Our study established a comprehensive taxonomy of code hallucinations, encompassing 3 primary categories and 12 specific categories. Furthermore, we systematically analyzed the distribution of hallucinations, exploring variations among different LLMs and benchmarks. Moreover, we perform an in-depth analysis on the causes and impacts of various hallucinations, aiming to provide valuable insights into hallucination mitigation. Finally, to enhance the correctness and reliability of LLM-generated code in a lightweight manner, we explore training-free hallucination mitigation approaches by prompt enhancing techniques. We believe our findings will shed light on future research about code hallucination evaluation and mitigation, ultimately paving the way for building more effective and reliable code LLMs in the future. The replication package is available at https://github.com/Lorien1128/code_hallucination
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.22950v2">StrucSum: Graph-Structured Reasoning for Long Document Extractive Summarization with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 14 pages. Accepted by the findings of EACL 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown strong performance in zero-shot summarization, but often struggle to model document structure and identify salient information in long texts. In this work, we introduce StrucSum, a training-free prompting framework that enhances LLM reasoning through sentence-level graph structures. StrucSum injects structural signals into prompts via three targeted strategies: Neighbor-Aware Prompting (NAP) for local context, Centrality-Aware Prompting (CAP) for importance estimation, and Centrality-Guided Masking (CGM) for efficient input reduction. Experiments on ArXiv, PubMed, and Multi-News demonstrate that StrucSum consistently improves both summary quality and factual consistency over unsupervised baselines and vanilla prompting. In particular, on ArXiv, it increases FactCC and SummaC by 19.2\% and 8.0\% points, demonstrating stronger alignment between summaries and source content. The ablation study shows that the combination of multiple strategies does not yield clear performance gains; therefore, structure-aware prompting with graph-based information represents a promising and underexplored direction for the advancement of zero-shot extractive summarization with LLMs. Our source code is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.18099v3">Stackelberg Self-Annotation: A Robust Approach to Data-Efficient LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) with human preferences typically demands vast amounts of meticulously curated data, which is both expensive and prone to labeling noise. We propose Stackelberg Game Preference Optimization (SGPO), a robust alignment framework that models alignment as a two-player Stackelberg game between a policy (leader) and a worst-case preference distribution (follower). The proposed SGPO guarantees $\mathcal{O}(ε)$-bounded regret within an $ε$-Wasserstein ball, offering formal robustness to (self-)annotation noise. We instantiate SGPO with Stackelberg Self-Annotated Preference Optimization (SSAPO), which uses minimal human-labeled "seed" preferences and iteratively self-annotates new prompts. In each iteration, SSAPO applies a distributionally robust reweighting of synthetic annotations, ensuring that noisy or biased self-labels do not derail training. Remarkably, using only 2K seed preferences -- about 1/30 of standard human labels -- SSAPO achieves strong win rates against GPT-4 across multiple benchmarks within three iterations. These results highlight that a principled Stackelberg formulation yields data-efficient alignment for LLMs, significantly reducing reliance on costly human annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.15091v4">ThinkRec: Thinking-based recommendation via LLM</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Published on WWW'26: In Proceedings of the ACM Web Conference 2026
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled more semantic-aware recommendations through natural language generation. Existing LLM for recommendation (LLM4Rec) methods mostly operate in a System 1-like manner, relying on superficial features to match similar items based on click history, rather than reasoning through deeper behavioral logic. This often leads to superficial and erroneous recommendations. Motivated by this, we propose ThinkRec, a thinking-based framework that shifts LLM4Rec from System 1 to System 2 (rational system). Technically, ThinkRec introduces a thinking activation mechanism that augments item metadata with keyword summarization and injects synthetic reasoning traces, guiding the model to form interpretable reasoning chains that consist of analyzing interaction histories, identifying user preferences, and making decisions based on target items. On top of this, we propose an instance-wise expert fusion mechanism to reduce the reasoning difficulty. By dynamically assigning weights to expert models based on users' latent features, ThinkRec adapts its reasoning path to individual users, thereby enhancing precision and personalization. Extensive experiments on real-world datasets demonstrate that ThinkRec significantly improves the accuracy and interpretability of recommendations. Our implementations are available at https://github.com/Yu-Qi-hang/ThinkRec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.24565v3">MCPAgentBench: A Real-world Task Benchmark for Evaluating LLM Agent MCP Tool Use</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly serving as autonomous agents, and their utilization of external tools via the Model Context Protocol (MCP) is considered a future trend. Current MCP evaluation sets suffer from issues such as reliance on external MCP services and a lack of difficulty awareness. To address these limitations, we propose MCPAgentBench, a benchmark based on real-world MCP definitions designed to evaluate the tool-use capabilities of agents. We construct a dataset containing authentic tasks and simulated MCP tools. The evaluation employs a dynamic sandbox environment that presents agents with candidate tool lists containing distractors, thereby testing their tool selection and discrimination abilities. Furthermore, we introduce comprehensive metrics to measure both task completion rates and execution efficiency. Experiments conducted on various latest mainstream Large Language Models reveal significant performance differences in handling complex, multi-step tool invocations. All code is open-source at Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14711v1">DARA: Few-shot Budget Allocation in Online Advertising via In-Context Decision Making with RL-Finetuned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Accepted at The ACM Web Conference (WWW) 2026
    </div>
    <details class="paper-abstract">
      Optimizing the advertiser's cumulative value of winning impressions under budget constraints poses a complex challenge in online advertising, under the paradigm of AI-Generated Bidding (AIGB). Advertisers often have personalized objectives but limited historical interaction data, resulting in few-shot scenarios where traditional reinforcement learning (RL) methods struggle to perform effectively. Large Language Models (LLMs) offer a promising alternative for AIGB by leveraging their in-context learning capabilities to generalize from limited data. However, they lack the numerical precision required for fine-grained optimization. To address this limitation, we introduce GRPO-Adaptive, an efficient LLM post-training strategy that enhances both reasoning and numerical precision by dynamically updating the reference policy during training. Built upon this foundation, we further propose DARA, a novel dual-phase framework that decomposes the decision-making process into two stages: a few-shot reasoner that generates initial plans via in-context prompting, and a fine-grained optimizer that refines these plans using feedback-driven reasoning. This separation allows DARA to combine LLMs' in-context learning strengths with precise adaptability required by AIGB tasks. Extensive experiments on both real-world and synthetic data environments demonstrate that our approach consistently outperforms existing baselines in terms of cumulative advertiser value under budget constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06111v2">LLM Powered Social Digital Twins: A Framework for Simulating Population Behavioral Response to Policy Interventions</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 13 pages, 1 figure, 4 tables
    </div>
    <details class="paper-abstract">
      Predicting how populations respond to policy interventions is a fundamental challenge in computational social science and public policy. Traditional approaches rely on aggregate statistical models that capture historical correlations but lack mechanistic interpretability and struggle with novel policy scenarios. We present a general framework for constructing Social Digital Twins - virtual population replicas where Large Language Models (LLMs) serve as cognitive engines for individual agents. Each agent, characterized by demographic and psychographic attributes, receives policy signals and outputs multi-dimensional behavioral probability vectors. A calibration layer maps aggregated agent responses to observable population-level metrics, enabling validation against real-world data and deployment for counterfactual policy analysis. We instantiate this framework in the domain of pandemic response, using COVID-19 as a case study with rich observational data. On a held-out test period, our calibrated digital twin achieves a 20.7% improvement in macro-averaged prediction error over gradient boosting baselines across six behavioral categories. Counterfactual experiments demonstrate monotonic and bounded responses to policy variations, establishing behavioral plausibility. The framework is domain-agnostic: the same architecture applies to transportation policy, economic interventions, environmental regulations, or any setting where policy affects population behavior. We discuss implications for policy simulation, limitations of the approach, and directions for extending LLM-based digital twins beyond pandemic response.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14686v1">IB-GRPO: Aligning LLM-based Learning Path Recommendation with Educational Objectives via Indicator-Based Group Relative Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Learning Path Recommendation (LPR) aims to generate personalized sequences of learning items that maximize long-term learning effect while respecting pedagogical principles and operational constraints. Although large language models (LLMs) offer rich semantic understanding for free-form recommendation, applying them to long-horizon LPR is challenging due to (i) misalignment with pedagogical objectives such as the Zone of Proximal Development (ZPD) under sparse, delayed feedback, (ii) scarce and costly expert demonstrations, and (iii) multi-objective interactions among learning effect, difficulty scheduling, length controllability, and trajectory diversity. To address these issues, we propose IB-GRPO (Indicator-Based Group Relative Policy Optimization), an indicator-guided alignment approach for LLM-based LPR. To mitigate data scarcity, we construct hybrid expert demonstrations via Genetic Algorithm search and teacher RL agents and warm-start the LLM with supervised fine-tuning. Building on this warm-start, we design a within-session ZPD alignment score for difficulty scheduling. IB-GRPO then uses the $I_{ε+}$ dominance indicator to compute group-relative advantages over multiple objectives, avoiding manual scalarization and improving Pareto trade-offs. Experiments on ASSIST09 and Junyi using the KES simulator with a Qwen2.5-7B backbone show consistent improvements over representative RL and LLM baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.12805v2">SciHorizon-GENE: Benchmarking LLM for Life Sciences Inference from Gene Knowledge to Functional Understanding</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown growing promise in biomedical research, particularly for knowledge-driven interpretation tasks. However, their ability to reliably reason from gene-level knowledge to functional understanding, a core requirement for knowledge-enhanced cell atlas interpretation, remains largely underexplored. To address this gap, we introduce SciHorizon-GENE, a large-scale gene-centric benchmark constructed from authoritative biological databases. The benchmark integrates curated knowledge for over 190K human genes and comprises more than 540K questions covering diverse gene-to-function reasoning scenarios relevant to cell type annotation, functional interpretation, and mechanism-oriented analysis. Motivated by behavioral patterns observed in preliminary examinations, SciHorizon-GENE evaluates LLMs along four biologically critical perspectives: research attention sensitivity, hallucination tendency, answer completeness, and literature influence, explicitly targeting failure modes that limit the safe adoption of LLMs in biological interpretation pipelines. We systematically evaluate a wide range of state-of-the-art general-purpose and biomedical LLMs, revealing substantial heterogeneity in gene-level reasoning capabilities and persistent challenges in generating faithful, complete, and literature-grounded functional interpretations. Our benchmark establishes a systematic foundation for analyzing LLM behavior at the gene scale and offers insights for model selection and development, with direct relevance to knowledge-enhanced biological interpretation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14667v1">INFA-Guard: Mitigating Malicious Propagation via Infection-Aware Safeguarding in LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Model (LLM)-based Multi-Agent Systems (MAS) has introduced significant security vulnerabilities, where malicious influence can propagate virally through inter-agent communication. Conventional safeguards often rely on a binary paradigm that strictly distinguishes between benign and attack agents, failing to account for infected agents i.e., benign entities converted by attack agents. In this paper, we propose Infection-Aware Guard, INFA-Guard, a novel defense framework that explicitly identifies and addresses infected agents as a distinct threat category. By leveraging infection-aware detection and topological constraints, INFA-Guard accurately localizes attack sources and infected ranges. During remediation, INFA-Guard replaces attackers and rehabilitates infected ones, avoiding malicious propagation while preserving topological integrity. Extensive experiments demonstrate that INFA-Guard achieves state-of-the-art performance, reducing the Attack Success Rate (ASR) by an average of 33%, while exhibiting cross-model robustness, superior topological generalization, and high cost-effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14660v1">NeuroFilter: Privacy Guardrails for Conversational LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      This work addresses the computational challenge of enforcing privacy for agentic Large Language Models (LLMs), where privacy is governed by the contextual integrity framework. Indeed, existing defenses rely on LLM-mediated checking stages that add substantial latency and cost, and that can be undermined in multi-turn interactions through manipulation or benign-looking conversational scaffolding. Contrasting this background, this paper makes a key observation: internal representations associated with privacy-violating intent can be separated from benign requests using linear structure. Using this insight, the paper proposes NeuroFilter, a guardrail framework that operationalizes contextual integrity by mapping norm violations to simple directions in the model's activation space, enabling detection even when semantic filters are bypassed. The proposed filter is also extended to capture threats arising during long conversations using the concept of activation velocity, which measures cumulative drift in internal representations across turns. A comprehensive evaluation across over 150,000 interactions and covering models from 7B to 70B parameters, illustrates the strong performance of NeuroFilter in detecting privacy attacks while maintaining zero false positives on benign prompts, all while reducing the computational inference cost by several orders of magnitude when compared to LLM-based agentic privacy defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14658v1">Say Anything but This: When Tokenizer Betrays Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) reason over discrete token ID sequences, yet modern subword tokenizers routinely produce non-unique encodings: multiple token ID sequences can detokenize to identical surface strings. This representational mismatch creates an unmeasured fragility wherein reasoning processes can fail. LLMs may treat two internal representations as distinct "words" even when they are semantically identical at the text level. In this work, we show that tokenization can betray LLM reasoning through one-to-many token ID mappings. We introduce a tokenization-consistency probe that requires models to replace designated target words in context while leaving all other content unchanged. The task is intentionally simple at the surface level, enabling us to attribute failures to tokenizer-detokenizer artifacts rather than to knowledge gaps or parameter limitations. Through analysis of over 11000 replacement trials across state-of-the-art open-source LLMs, we find a non-trivial rate of outputs exhibit phantom edits: cases where models operate under the illusion of correct reasoning, a phenomenon arising from tokenizer-induced representational defects. We further analyze these cases and provide a taxonomy of eight systematic tokenizer artifacts, including whitespace-boundary shifts and intra-word resegmentation. These findings indicate that part of apparent reasoning deficiency originates in the tokenizer layer, motivating tokenizer-level remedies before incurring the cost of training ever-larger models on ever-larger corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.14397v4">Thunder-NUBench: A Benchmark for LLMs' Sentence-Level Negation Understanding</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Negation is a fundamental linguistic phenomenon that poses ongoing challenges for Large Language Models (LLMs), particularly in tasks requiring deep semantic understanding. Current benchmarks often treat negation as a minor detail within broader tasks, such as natural language inference. Consequently, there is a lack of benchmarks specifically designed to evaluate comprehension of negation. In this work, we introduce Thunder-NUBench, a novel benchmark explicitly created to assess sentence-level understanding of negation in LLMs. Thunder-NUBench goes beyond merely identifying surface-level cues by contrasting standard negation with structurally diverse alternatives, such as local negation, contradiction, and paraphrase. This benchmark includes manually curated sentence-negation pairs and a multiple-choice dataset, allowing for a comprehensive evaluation of models' understanding of negation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18261v2">LLM Reasoning for Cold-Start Item Recommendation</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Published on Proceedings of the ACM on Web Conference 2026 (WWW 2026)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant potential for improving recommendation systems through their inherent reasoning capabilities and extensive knowledge base. Yet, existing studies predominantly address warm-start scenarios with abundant user-item interaction data, leaving the more challenging cold-start scenarios, where sparse interactions hinder traditional collaborative filtering methods, underexplored. To address this limitation, we propose novel reasoning strategies designed for cold-start item recommendations within the Netflix domain. Our method utilizes the advanced reasoning capabilities of LLMs to effectively infer user preferences, particularly for newly introduced or rarely interacted items. We systematically evaluate supervised fine-tuning, reinforcement learning-based fine-tuning, and hybrid approaches that combine both methods to optimize recommendation performance. Extensive experiments on real-world data demonstrate significant improvements in both methodological efficacy and practical performance in cold-start recommendation contexts. Remarkably, our reasoning-based fine-tuned models outperform Netflix's production ranking model by up to 8% in certain cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.01631v2">Unraveling LLM Jailbreaks Through Safety Knowledge Neurons</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 EACL 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation, a technique known as "Jailbreak." While some studies have achieved defenses against jailbreak attacks by modifying output distributions or detecting harmful content, the exact rationale still remains elusive. In this work, we present a novel neuron-level interpretability method that focuses on the role of safety-related knowledge neurons. Unlike existing approaches, our method projects the model's internal representation into a more consistent and interpretable vocabulary space. We then show that adjusting the activation of safety-related neurons can effectively control the model's behavior with a mean ASR higher than 97%. Building on this insight, we propose SafeTuning, a fine-tuning strategy that reinforces safety-critical neurons to improve model robustness against jailbreaks. SafeTuning consistently reduces attack success rates across multiple LLMs and outperforms all four baseline defenses. These findings offer a new perspective on understanding and defending against jailbreak attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25247v2">Protocode: Prototype-Driven Interpretability for Code Generation in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Since the introduction of Large Language Models (LLMs), they have been widely adopted for various tasks such as text summarization, question answering, speech-to-text translation, and more. In recent times, the use of LLMs for code generation has gained significant attention, with tools such as Cursor and Windsurf demonstrating the ability to analyze massive code repositories and recommend relevant changes. Big tech companies have also acknowledged the growing reliance on LLMs for code generation within their codebases. Although these advances significantly improve developer productivity, increasing reliance on automated code generation can proportionally increase the risk of suboptimal solutions and insecure code. Our work focuses on automatically sampling In-Context Learning (ICL) demonstrations which can improve model performance and enhance the interpretability of the generated code. Using AST-based analysis on outputs from the MBPP test set, we identify regions of code most influenced by the chosen demonstrations. In our experiments, we show that high-quality ICL demonstrations not only make outputs easier to interpret but also yield a positive performance improvement on the pass@10 metric. Conversely, poorly chosen ICL demonstrations affected the LLM performance on the pass@10 metric negatively compared to the base model. Overall, our approach highlights the importance of efficient sampling strategies for ICL, which can affect the performance of the model on any given task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.16918v3">OptimAI: Optimization from Natural Language Using LLM-Powered AI Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Optimization plays a vital role in scientific research and practical applications. However, formulating a concrete optimization problem described in natural language into a mathematical form and selecting a suitable solver to solve the problem requires substantial domain expertise. We introduce OptimAI, a framework for solving Optimization problems described in natural language by leveraging LLM-powered AI agents, and achieve superior performance over current state-of-the-art methods. Our framework is built upon the following key roles: (1) a formulator that translates natural language problem descriptions into precise mathematical formulations; (2) a planner that constructs a high-level solution strategy prior to execution; and (3) a coder and a code critic capable of interacting with the environment and reflecting on outcomes to refine future actions. Ablation studies confirm that all roles are essential; removing the planner or code critic results in $5.8\times$ and $3.1\times$ drops in productivity, respectively. Furthermore, we introduce UCB-based debug scheduling to dynamically switch between alternative plans, yielding an additional $3.3\times$ productivity gain. Our design emphasizes multi-agent collaboration, and our experiments confirm that combining diverse models leads to performance gains. Our approach attains 88.1% accuracy on the NLP4LP dataset and 82.3% on the Optibench dataset, reducing error rates by 58% and 52%, respectively, over prior best results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14606v1">An LLM Agent-based Framework for Whaling Countermeasures</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      With the spread of generative AI in recent years, attacks known as Whaling have become a serious threat. Whaling is a form of social engineering that targets important high-authority individuals within organizations and uses sophisticated fraudulent emails. In the context of Japanese universities, faculty members frequently hold positions that combine research leadership with authority within institutional workflows. This structural characteristic leads to the wide public disclosure of high-value information such as publications, grants, and detailed researcher profiles. Such extensive information exposure enables the construction of highly precise target profiles using generative AI. This raises concerns that Whaling attacks based on high-precision profiling by generative AI will become prevalent. In this study, we propose a Whaling countermeasure framework for university faculty members that constructs personalized defense profiles and uses large language model (LLM)-based agents. We design agents that (i) build vulnerability profiles for each target from publicly available information on faculty members, (ii) identify potential risk scenarios relevant to Whaling defense based on those profiles, (iii) construct defense profiles corresponding to the vulnerabilities and anticipated risks, and (iv) analyze Whaling emails using the defense profiles. Furthermore, we conduct a preliminary risk-assessment experiment. The results indicate that the proposed method can produce judgments accompanied by explanations of response policies that are consistent with the work context of faculty members who are Whaling targets. The findings also highlight practical challenges and considerations for future operational deployment and systematic evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14603v1">Variance-Adaptive Muon: Accelerating LLM Pretraining with NSR-Modulated and Variance-Scaled Momentum</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve competitive performance across diverse natural language processing (NLP) tasks, yet pretraining is computationally demanding, making optimizer efficiency an important practical consideration. Muon accelerates LLM pretraining via orthogonal momentum updates that serve as a matrix analogue of the element-wise sign operator. Motivated by the recent perspective that Adam is a variance-adaptive sign update algorithm, we propose two variants of Muon, Muon-NSR and Muon-VS, which apply variance-adaptive normalization to momentum before orthogonalization. Muon-NSR applies noise-to-signal ratio (NSR) modulation, while Muon-VS performs variance-based scaling without introducing additional hyperparameters. Experiments on GPT-2 and LLaMA pretraining demonstrate that our proposed methods accelerate convergence and consistently achieve lower validation loss than both competitive, well-tuned AdamW and Muon baselines. For example, on the LLaMA-1.2B model, Muon-NSR and Muon-VS reduce the iterations required to reach the target validation loss by $1.36\times$ relative to the well-tuned Muon following the recent benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14601v1">Holmes: An Evidence-Grounded LLM Agent for Auditable DDoS Investigation in Cloud Networks</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Cloud environments face frequent DDoS threats due to centralized resources and broad attack surfaces. Modern cloud-native DDoS attacks further evolve rapidly and often blend multi-vector strategies, creating an operational dilemma: defenders need wire-speed monitoring while also requiring explainable, auditable attribution for response. Existing rule-based and supervised-learning approaches typically output black-box scores or labels, provide limited evidence chains, and generalize poorly to unseen attack variants; meanwhile, high-quality labeled data is often difficult to obtain in cloud settings. We present Holmes (DDoS Detective), an LLM-based DDoS detection agent that reframes the model as a virtual SRE investigator rather than an end-to-end classifier. Holmes couples a funnel-like hierarchical workflow (counters/sFlow for continuous sensing and triage; PCAP evidence collection triggered only on anomaly windows) with an Evidence Pack abstraction that converts binary packets into compact, reproducible, high-signal structured evidence. On top of this evidence interface, Holmes enforces a structure-first investigation protocol and strict JSON/quotation constraints to produce machine-consumable reports with auditable evidence anchors. We evaluate Holmes on CICDDoS2019 reflection/amplification attacks and script-triggered flooding scenarios. Results show that Holmes produces attribution decisions grounded in salient evidence anchors across diverse attack families, and when errors occur, its audit logs make the failure source easy to localize, demonstrating the practicality of an LLM agent for cost-controlled and traceable DDoS investigation in cloud operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14599v1">Rethinking Reinforcement fine-tuning of LLMs: A Multi-armed Bandit Learning Perspective</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      A large number of heuristics have been proposed to optimize the reinforcement fine-tuning of LLMs. However, inconsistent claims are made from time to time, making this area elusive. Reflecting on this situation, two fundamental questions still lack a clear understanding: 1) what is the role of each optimizing choice? 2) which ones are the bottlenecks? This paper aims to shed light on them, and it faces the challenge of several entangled confounding factors in the fine-tuning process. To tackle this challenge, we propose a bottom-up experiment pipeline. The bottom layer is composed of a minimalist configuration: one training data, one rollout per round and the reward directly serve as the learning signal without advantage function design. This minimalist configuration connects to multi-armed bandit learning with extremely large discrete action space, which offers theories to corroborate the experiment findings. The up procedure of the experiment pipeline expanding the minimalist configuration layer by layer, examining the role of each design choice. Experimental results on three LLMs and two reasoning datasets not only reveal new understanding of the design choice but also yield essential insights to shape the area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14598v1">HELIOS: Hierarchical Graph Abstraction for Structure-Aware LLM Decompilation</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently been applied to binary decompilation, yet they still treat code as plain text and ignore the graphs that govern program control flow. This limitation often yields syntactically fragile and logically inconsistent output, especially for optimized binaries. This paper presents \textsc{HELIOS}, a framework that reframes LLM-based decompilation as a structured reasoning task. \textsc{HELIOS} summarizes a binary's control flow and function calls into a hierarchical text representation that spells out basic blocks, their successors, and high-level patterns such as loops and conditionals. This representation is supplied to a general-purpose LLM, along with raw decompiler output, optionally combined with a compiler-in-the-loop that returns error messages when the generated code fails to build. On HumanEval-Decompile for \texttt{x86\_64}, \textsc{HELIOS} raises average object file compilability from 45.0\% to 85.2\% for Gemini~2.0 and from 71.4\% to 89.6\% for GPT-4.1~Mini. With compiler feedback, compilability exceeds 94\% and functional correctness improves by up to 5.6 percentage points over text-only prompting. Across six architectures drawn from x86, ARM, and MIPS, \textsc{HELIOS} reduces the spread in functional correctness while keeping syntactic correctness consistently high, all without fine-tuning. These properties make \textsc{HELIOS} a practical building block for reverse engineering workflows in security settings where analysts need recompilable, semantically faithful code across diverse hardware targets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.17825v3">FAIRGAMER: Evaluating Social Biases in LLM-Based Video Game NPCs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have increasingly enhanced or replaced traditional Non-Player Characters (NPCs) in video games. However, these LLM-based NPCs inherit underlying social biases (e.g., race or class), posing fairness risks during in-game interactions. To address the limited exploration of this issue, we introduce FairGamer, the first benchmark to evaluate social biases across three interaction patterns: transaction, cooperation, and competition. FairGamer assesses four bias types, including class, race, age, and nationality, across 12 distinct evaluation tasks using a novel metric, FairMCV. Our evaluation of seven frontier LLMs reveals that: (1) models exhibit biased decision-making, with Grok-4-Fast demonstrating the highest bias (average FairMCV = 76.9%); and (2) larger LLMs display more severe social biases, suggesting that increased model capacity inadvertently amplifies these biases. We release FairGamer at https://github.com/Anonymous999-xxx/FairGamer to facilitate future research on NPC fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14590v1">Counterfactual Modeling with Fine-Tuned LLMs for Health Intervention Design and Sensor Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Counterfactual explanations (CFEs) provide human-centric interpretability by identifying the minimal, actionable changes required to alter a machine learning model's prediction. Therefore, CFs can be used as (i) interventions for abnormality prevention and (ii) augmented data for training robust models. We conduct a comprehensive evaluation of CF generation using large language models (LLMs), including GPT-4 (zero-shot and few-shot) and two open-source models-BioMistral-7B and LLaMA-3.1-8B, in both pretrained and fine-tuned configurations. Using the multimodal AI-READI clinical dataset, we assess CFs across three dimensions: intervention quality, feature diversity, and augmentation effectiveness. Fine-tuned LLMs, particularly LLaMA-3.1-8B, produce CFs with high plausibility (up to 99%), strong validity (up to 0.99), and realistic, behaviorally modifiable feature adjustments. When used for data augmentation under controlled label-scarcity settings, LLM-generated CFs substantially restore classifier performance, yielding an average 20% F1 recovery across three scarcity scenarios. Compared with optimization-based baselines such as DiCE, CFNOW, and NICE, LLMs offer a flexible, model-agnostic approach that generates more clinically actionable and semantically coherent counterfactuals. Overall, this work demonstrates the promise of LLM-driven counterfactuals for both interpretable intervention design and data-efficient model training in sensor-based digital health. Impact: SenseCF fine-tunes an LLM to generate valid, representative counterfactual explanations and supplement minority class in an imbalanced dataset for improving model training and boosting model robustness and predictive performance
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14566v1">SCSimulator: An Exploratory Visual Analytics Framework for Partner Selection in Supply Chains through LLM-driven Multi-Agent Simulation</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 ACM IUI 2026
    </div>
    <details class="paper-abstract">
      Supply chains (SCs), complex networks spanning from raw material acquisition to product delivery, with enterprises as interconnected nodes, play a pivotal role in organizational success. However, optimizing SCs remains challenging, particularly in partner selection, a key bottleneck shaped by competitive and cooperative dynamics. This challenge constitutes a multi-objective dynamic game requiring a synergistic integration of Multi-Criteria Decision-Making and Game Theory. Traditional approaches, grounded in mathematical simplifications and managerial heuristics, fail to capture real-world intricacies and risk introducing subjective biases. Multi-agent simulation offers promise, but prior research has largely relied on fixed, uniform agent logic, limiting practical applicability. Recent advances in LLMs create opportunities to represent complex SC requirements and hybrid game logic. However, challenges persist in modeling dynamic SC relationships, ensuring interpretability, and balancing agent autonomy with expert control. We present SCSimulator, a visual analytics framework that integrates LLM-driven MAS with human-in-the-loop collaboration for SC partner selection. It simulates SC evolution via adaptive network structures and enterprise behaviors, which are visualized via interpretable interfaces. By combining CoT reasoning with XAI techniques, it generates multi-faceted, transparent explanations of decision trade-offs. Users can iteratively adjust simulation settings to explore outcomes aligned with their expectations and strategic priorities. Developed through iterative co-design with SC experts and industry managers, SCSimulator serves as a proof-of-concept, offering methodological contributions and practical insights for future research on SC decision-making and interactive AI-driven analytics. Usage scenarios and a user study demonstrate the system's effectiveness and usability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14560v1">Rewarding How Models Think Pedagogically: Integrating Pedagogical Reasoning and Thinking Rewards for LLMs in Education</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as intelligent tutoring systems, yet research on optimizing LLMs specifically for educational contexts remains limited. Recent works have proposed reinforcement learning approaches for training LLM tutors, but these methods focus solely on optimizing visible responses while neglecting the model's internal thinking process. We introduce PedagogicalRL-Thinking, a framework that extends pedagogical alignment to reasoning LLMs in education through two novel approaches: (1) Pedagogical Reasoning Prompting, which guides internal reasoning using domain-specific educational theory rather than generic instructions; and (2) Thinking Reward, which explicitly evaluates and reinforces the pedagogical quality of the model's reasoning traces. Our experiments reveal that domain-specific, theory-grounded prompting outperforms generic prompting, and that Thinking Reward is most effective when combined with pedagogical prompting. Furthermore, models trained only on mathematics tutoring dialogues show improved performance on educational benchmarks not seen during training, while preserving the base model's factual knowledge. Our quantitative and qualitative analyses reveal that pedagogical thinking reward produces systematic reasoning trace changes, with increased pedagogical reasoning and more structured instructional decision-making in the tutor's thinking process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02744v2">SYNAPSE: Empowering LLM Agents with Episodic-Semantic Memory via Spreading Activation</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) excel at generalized reasoning, standard retrieval-augmented approaches fail to address the disconnected nature of long-term agentic memory. To bridge this gap, we introduce Synapse (Synergistic Associative Processing Semantic Encoding), a unified memory architecture that transcends static vector similarity. Drawing from cognitive science, Synapse models memory as a dynamic graph where relevance emerges from spreading activation rather than pre-computed links. By integrating lateral inhibition and temporal decay, the system dynamically highlights relevant sub-graphs while filtering interference. We implement a Triple Hybrid Retrieval strategy that fuses geometric embeddings with activation-based graph traversal. Comprehensive evaluations on the LoCoMo benchmark show that Synapse significantly outperforms state-of-the-art methods in complex temporal and multi-hop reasoning tasks, offering a robust solution to the "Contextual Tunneling" problem. Our code and data will be made publicly available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.20097v2">Can LLMs Identify Tax Abuse?</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      We investigate whether large language models can discover and analyze U.S. tax-minimization strategies. This real-world domain challenges even seasoned human experts, and progress can reduce tax revenue lost from well-advised, wealthy taxpayers. We evaluate the most advanced LLMs on their ability to (1) interpret and verify tax strategies, (2) fill in gaps in partially specified strategies, and (3) generate complete, end-to-end strategies from scratch. This domain should be of particular interest to the LLM reasoning community: unlike synthetic challenge problems or scientific reasoning tasks, U.S. tax law involves navigating hundreds of thousands of pages of statutes, case law, and administrative guidance, all updated regularly. Notably, LLM-based reasoning identified an entirely novel tax strategy, highlighting these models' potential to revolutionize tax agencies' fight against tax abuse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15528v1">Securing LLM-as-a-Service for Small Businesses: An Industry Case Study of a Distributed Chatbot Deployment Platform</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Accepted by AISC 2026
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based question-answering systems offer significant potential for automating customer support and internal knowledge access in small businesses, yet their practical deployment remains challenging due to infrastructure costs, engineering complexity, and security risks, particularly in retrieval-augmented generation (RAG)-based settings. This paper presents an industry case study of an open-source, multi-tenant platform that enables small businesses to deploy customised LLM-based support chatbots via a no-code workflow. The platform is built on distributed, lightweight k3s clusters spanning heterogeneous, low-cost machines and interconnected through an encrypted overlay network, enabling cost-efficient resource pooling while enforcing container-based isolation and per-tenant data access controls. In addition, the platform integrates practical, platform-level defences against prompt injection attacks in RAG-based chatbots, translating insights from recent prompt injection research into deployable security mechanisms without requiring model retraining or enterprise-scale infrastructure. We evaluate the proposed platform through a real-world e-commerce deployment, demonstrating that secure and efficient LLM-based chatbot services can be achieved under realistic cost, operational, and security constraints faced by small businesses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15519v1">TransportAgents: a multi-agents LLM framework for traffic accident severity prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Accurate prediction of traffic crash severity is critical for improving emergency response and public safety planning. Although recent large language models (LLMs) exhibit strong reasoning capabilities, their single-agent architectures often struggle with heterogeneous, domain-specific crash data and tend to generate biased or unstable predictions. To address these limitations, this paper proposes TransportAgents, a hybrid multi-agent framework that integrates category-specific LLM reasoning with a multilayer perceptron (MLP) integration module. Each specialized agent focuses on a particular subset of traffic information, such as demographics, environmental context, or incident details, to produce intermediate severity assessments that are subsequently fused into a unified prediction. Extensive experiments on two complementary U.S. datasets, the Consumer Product Safety Risk Management System (CPSRMS) and the National Electronic Injury Surveillance System (NEISS), demonstrate that TransportAgents consistently outperforms both traditional machine learning and advanced LLM-based baselines. Across three representative backbones, including closed-source models such as GPT-3.5 and GPT-4o, as well as open-source models such as LLaMA-3.3, the framework exhibits strong robustness, scalability, and cross-dataset generalizability. A supplementary distributional analysis further shows that TransportAgents produces more balanced and well-calibrated severity predictions than standard single-agent LLM approaches, highlighting its interpretability and reliability for safety-critical decision support applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15495v1">Tracking the Limits of Knowledge Propagation: How LLMs Fail at Multi-Step Reasoning with Conflicting Knowledge</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Accepted to EACL 2026 (Main)
    </div>
    <details class="paper-abstract">
      A common solution for mitigating outdated or incorrect information in Large Language Models (LLMs) is to provide updated facts in-context or through knowledge editing. However, these methods introduce knowledge conflicts when the knowledge update fails to overwrite the model's parametric knowledge, which propagate to faulty reasoning. Current benchmarks for this problem, however, largely focus only on single knowledge updates and fact recall without evaluating how these updates affect downstream reasoning. In this work, we introduce TRACK (Testing Reasoning Amid Conflicting Knowledge), a new benchmark for studying how LLMs propagate new knowledge through multi-step reasoning when it conflicts with the model's initial parametric knowledge. Spanning three reasoning-intensive scenarios (WIKI, CODE, and MATH), TRACK introduces multiple, realistic conflicts to mirror real-world complexity. Our results on TRACK reveal that providing updated facts to models for reasoning can worsen performance compared to providing no updated facts to a model, and that this performance degradation exacerbates as more updated facts are provided. We show this failure stems from both inability to faithfully integrate updated facts, but also flawed reasoning even when knowledge is integrated. TRACK provides a rigorous new benchmark to measure and guide future progress on propagating conflicting knowledge in multi-step reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15482v1">Martingale Foresight Sampling: A Principled Approach to Inference-Time LLM Decoding</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Standard autoregressive decoding in large language models (LLMs) is inherently short-sighted, often failing to find globally optimal reasoning paths due to its token-by-token generation process. While inference-time strategies like foresight sampling attempt to mitigate this by simulating future steps, they typically rely on ad-hoc heuristics for valuing paths and pruning the search space. This paper introduces Martingale Foresight Sampling (MFS), a principled framework that reformulates LLM decoding as a problem of identifying an optimal stochastic process. By modeling the quality of a reasoning path as a stochastic process, we leverage Martingale theory to design a theoretically-grounded algorithm. Our approach replaces heuristic mechanisms with principles from probability theory: step valuation is derived from the Doob Decomposition Theorem to measure a path's predictable advantage, path selection uses Optional Stopping Theory for principled pruning of suboptimal candidates, and an adaptive stopping rule based on the Martingale Convergence Theorem terminates exploration once a path's quality has provably converged. Experiments on six reasoning benchmarks demonstrate that MFS surpasses state-of-the-art methods in accuracy while significantly improving computational efficiency. Code will be released at https://github.com/miraclehetech/EACL2026-Martingale-Foresight-Sampling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15479v1">Benchmarking LLMs for Pairwise Causal Discovery in Biomedical and Multi-Domain Contexts</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      The safe deployment of large language models (LLMs) in high-stakes fields like biomedicine, requires them to be able to reason about cause and effect. We investigate this ability by testing 13 open-source LLMs on a fundamental task: pairwise causal discovery (PCD) from text. Our benchmark, using 12 diverse datasets, evaluates two core skills: 1) \textbf{Causal Detection} (identifying if a text contains a causal link) and 2) \textbf{Causal Extraction} (pulling out the exact cause and effect phrases). We tested various prompting methods, from simple instructions (zero-shot) to more complex strategies like Chain-of-Thought (CoT) and Few-shot In-Context Learning (FICL). The results show major deficiencies in current models. The best model for detection, DeepSeek-R1-Distill-Llama-70B, only achieved a mean score of 49.57\% ($C_{detect}$), while the best for extraction, Qwen2.5-Coder-32B-Instruct, reached just 47.12\% ($C_{extract}$). Models performed best on simple, explicit, single-sentence relations. However, performance plummeted for more difficult (and realistic) cases, such as implicit relationships, links spanning multiple sentences, and texts containing multiple causal pairs. We provide a unified evaluation framework, built on a dataset validated with high inter-annotator agreement ($κ\ge 0.758$), and make all our data, code, and prompts publicly available to spur further research. \href{https://github.com/sydneyanuyah/CausalDiscovery}{Code available here: https://github.com/sydneyanuyah/CausalDiscovery}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.05882v2">Collaborate, Deliberate, Evaluate: How LLM Alignment Affects Coordinated Multi-Agent Outcomes</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 This submission is a new version of arXiv:2509.05882v1. with a substantially revised experimental pipeline and new metrics. In particular, collaborator agents are now instantiated independently via separate API calls, rather than generated autoregressively by a single agent. All experimental results are new. Accepted as an extended abstract at AAMAS 2026
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) get integrated into diverse workflows, they are increasingly being regarded as "collaborators" with humans, and required to work in coordination with other AI systems. If such AI collaborators are to reliably coordinate their actions and behaviors with humans or other AIs, their properties and behaviors over multi-turn interactions must be known and predictable. This paper examines how different alignment methods affect LLM agents' effectiveness as partners in multi-turn, multi-party collaborations. We study this question through the lens of intervention agents that insert themselves into group dialogues not to provide answers, but to encourage the collaborative group to slow down and reflect upon their reasoning for deliberative decision-making. Common alignment techniques are typically developed under simplified single-user settings and assume the optimality of the underlying token MDP. Using the theoretical lens of the modified-action MDP, we show how they do not account for the dynamics of long-horizon multi-party interactions. We present a novel roleplay simulation methodology, where we align LLMs according to different methods and then deploy them in collaborative task dialogues to quantify how interventions affect the trajectory of group collaboration, belief alignment, and coordination. Our results show that an intervention agent that is robust to action modification significantly outperforms common alignment baselines in supporting correct task outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15476v1">Reliability by design: quantifying and eliminating fabrication risk in LLMs. From generative to consultative AI: a comparative analysis in the legal domain and lessons for high-stakes knowledge bases</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      This paper examines how to make large language models reliable for high-stakes legal work by reducing hallucinations. It distinguishes three AI paradigms: (1) standalone generative models ("creative oracle"), (2) basic retrieval-augmented systems ("expert archivist"), and (3) an advanced, end-to-end optimized RAG system ("rigorous archivist"). The authors introduce two reliability metrics -False Citation Rate (FCR) and Fabricated Fact Rate (FFR)- and evaluate 2,700 judicial-style answers from 12 LLMs across 75 legal tasks using expert, double-blind review. Results show that standalone models are unsuitable for professional use (FCR above 30%), while basic RAG greatly reduces errors but still leaves notable misgrounding. Advanced RAG, using techniques such as embedding fine-tuning, re-ranking, and self-correction, reduces fabrication to negligible levels (below 0.2%). The study concludes that trustworthy legal AI requires rigor-focused, retrieval-based architectures emphasizing verification and traceability, and provides an evaluation framework applicable to other high-risk domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11791v2">Beyond Tokens: Concept-Level Training Objectives for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      The next-token prediction (NTP) objective has been foundational in the development of modern large language models (LLMs), driving advances in fluency and generalization. However, NTP operates at the \textit{token} level, treating deviations from a single reference continuation as errors even when alternative continuations are equally plausible or semantically equivalent (e.g., ``mom'' vs. ``mother''). As a result, token-level loss can penalize valid abstractions, paraphrases, or conceptually correct reasoning paths, biasing models toward surface form rather than underlying meaning. This mismatch between the training signal and semantic correctness motivates learning objectives that operate over higher-level representations. We propose a shift from token-level to concept-level prediction, where concepts group multiple surface forms of the same idea (e.g., ``mom,'' ``mommy,'' ``mother'' $\rightarrow$ \textit{MOTHER}). We introduce various methods for integrating conceptual supervision into LLM training and show that concept-aware models achieve lower perplexity, improved robustness under domain shift, and stronger performance than NTP-based models on diverse NLP benchmarks. This suggests \textit{concept-level supervision} as an improved training signal that better aligns LLMs with human semantic abstractions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15429v1">Domain-Specific Knowledge Graphs in RAG-Enhanced Healthcare LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) generate fluent answers but can struggle with trustworthy, domain-specific reasoning. We evaluate whether domain knowledge graphs (KGs) improve Retrieval-Augmented Generation (RAG) for healthcare by constructing three PubMed-derived graphs: $\mathbb{G}_1$ (T2DM), $\mathbb{G}_2$ (Alzheimer's disease), and $\mathbb{G}_3$ (AD+T2DM). We design two probes: Probe 1 targets merged AD T2DM knowledge, while Probe 2 targets the intersection of $\mathbb{G}_1$ and $\mathbb{G}_2$. Seven instruction-tuned LLMs are tested across retrieval sources {No-RAG, $\mathbb{G}_1$, $\mathbb{G}_2$, $\mathbb{G}_1$ + $\mathbb{G}_2$, $\mathbb{G}_3$, $\mathbb{G}_1$+$\mathbb{G}_2$ + $\mathbb{G}_3$} and three decoding temperatures. Results show that scope alignment between probe and KG is decisive: precise, scope-matched retrieval (notably $\mathbb{G}_2$) yields the most consistent gains, whereas indiscriminate graph unions often introduce distractors that reduce accuracy. Larger models frequently match or exceed KG-RAG with a No-RAG baseline on Probe 1, indicating strong parametric priors, whereas smaller/mid-sized models benefit more from well-scoped retrieval. Temperature plays a secondary role; higher values rarely help. We conclude that precision-first, scope-matched KG-RAG is preferable to breadth-first unions, and we outline practical guidelines for graph selection, model sizing, and retrieval/reranking. Code and Data available here - https://github.com/sydneyanuyah/RAGComparison
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15397v1">Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC)</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      The rapid emergence of new entities -- driven by cultural shifts, evolving trends, and personalized user data -- poses a significant challenge for existing Speech Large Language Models (Speech LLMs). While these models excel at general conversational tasks, their static training knowledge limits their ability to recognize domain-specific terms such as contact names, playlists, or technical jargon. Existing solutions primarily rely on prompting, which suffers from poor scalability: as the entity list grows, prompting encounters context window limitations, increased inference latency, and the "lost-in-the-middle" phenomenon. An alternative approach, Generative Error Correction (GEC), attempts to rewrite transcripts via post-processing but frequently suffers from "over-correction", introducing hallucinations of entities that were never spoken. In this work, we introduce LOGIC (Logit-Space Integration for Contextual Biasing), an efficient and robust framework that operates directly in the decoding layer. Unlike prompting, LOGIC decouples context injection from input processing, ensuring constant-time complexity relative to prompt length. Extensive experiments using the Phi-4-MM model across 11 multilingual locales demonstrate that LOGIC achieves an average 9% relative reduction in Entity WER with a negligible 0.30% increase in False Alarm Rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15385v1">VegaChat: A Robust Framework for LLM-Based Chart Generation and Assessment</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 8 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Natural-language-to-visualization (NL2VIS) systems based on large language models (LLMs) have substantially improved the accessibility of data visualization. However, their further adoption is hindered by two coupled challenges: (i) the absence of standardized evaluation metrics makes it difficult to assess progress in the field and compare different approaches; and (ii) natural language descriptions are inherently underspecified, so multiple visualizations may be valid for the same query. To address these issues, we introduce VegaChat, a framework for generating, validating, and assessing declarative visualizations from natural language. We propose two complementary metrics: Spec Score, a deterministic metric that measures specification-level similarity without invoking an LLM, and Vision Score, a library-agnostic, image-based metric that leverages a multimodal LLM to assess chart similarity and prompt compliance. We evaluate VegaChat on the NLV Corpus and on the annotated subset of ChartLLM. VegaChat achieves near-zero rates of invalid or empty visualizations, while Spec Score and Vision Score exhibit strong correlation with human judgments (Pearson 0.65 and 0.71, respectively), indicating that the proposed metrics support consistent, cross-library comparison. The code and evaluation artifacts are available at https://zenodo.org/records/17062309.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15352v1">A Prompt-Based Framework for Loop Vulnerability Detection Using Local LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Accepted and Waiting to be published ICAI'25: 27th International Conference on Artificial Intelligence https://american-cse.org/csce2025/conferences-ICAI
    </div>
    <details class="paper-abstract">
      Loop vulnerabilities are one major risky construct in software development. They can easily lead to infinite loops or executions, exhaust resources, or introduce logical errors that degrade performance and compromise security. The problem are often undetected by traditional static analyzers because such tools rely on syntactic patterns, which makes them struggle to detect semantic flaws. Consequently, Large Language Models (LLMs) offer new potential for vulnerability detection because of their ability to understand code contextually. Moreover, local LLMs unlike commercial ones like ChatGPT or Gemini addresses issues such as privacy, latency, and dependency concerns by facilitating efficient offline analysis. Consequently, this study proposes a prompt-based framework that utilize local LLMs for the detection of loop vulnerabilities within Python 3.7+ code. The framework targets three categories of loop-related issues, such as control and logic errors, security risks inside loops, and resource management inefficiencies. A generalized and structured prompt-based framework was designed and tested with two locally deployed LLMs (LLaMA 3.2; 3B and Phi 3.5; 4B) by guiding their behavior via iterative prompting. The designed prompt-based framework included key safeguarding features such as language-specific awareness, code-aware grounding, version sensitivity, and hallucination prevention. The LLM results were validated against a manually established baseline truth, and the results indicate that Phi outperforms LLaMA in precision, recall, and F1-score. The findings emphasize the importance of designing effective prompts for local LLMs to perform secure and accurate code vulnerability analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15348v1">Abusive music and song transformation using GenAI and LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Repeated exposure to violence and abusive content in music and song content can influence listeners' emotions and behaviours, potentially normalising aggression or reinforcing harmful stereotypes. In this study, we explore the use of generative artificial intelligence (GenAI) and Large Language Models (LLMs) to automatically transform abusive words (vocal delivery) and lyrical content in popular music. Rather than simply muting or replacing a single word, our approach transforms the tone, intensity, and sentiment, thus not altering just the lyrics, but how it is expressed. We present a comparative analysis of four selected English songs and their transformed counterparts, evaluating changes through both acoustic and sentiment-based lenses. Our findings indicate that Gen-AI significantly reduces vocal aggressiveness, with acoustic analysis showing improvements in Harmonic to Noise Ratio, Cepstral Peak Prominence, and Shimmer. Sentiment analysis reduced aggression by 63.3-85.6\% across artists, with major improvements in chorus sections (up to 88.6\% reduction). The transformed versions maintained musical coherence while mitigating harmful content, offering a promising alternative to traditional content moderation that avoids triggering the "forbidden fruit" effect, where the censored content becomes more appealing simply because it is restricted. This approach demonstrates the potential for GenAI to create safer listening experiences while preserving artistic expression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15269v1">Lightweight LLMs for Network Attack Detection in IoT Networks</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 6 pages with 2 figures, This paper was accepted and presented at the 7th Computing, Communications and IoT Applications Conference (ComComAp 2025), held in Madrid, Spain, during 14th to 17th December 2025
    </div>
    <details class="paper-abstract">
      The rapid growth of Internet of Things (IoT) devices has increased the scale and diversity of cyberattacks, exposing limitations in traditional intrusion detection systems. Classical machine learning (ML) models such as Random Forest and Support Vector Machine perform well on known attacks but require retraining to detect unseen or zero-day threats. This study investigates lightweight decoder-only Large Language Models (LLMs) for IoT attack detection by integrating structured-to-text conversion, Quantized Low-Rank Adaptation (QLoRA) fine-tuning, and Retrieval-Augmented Generation (RAG). Network traffic features are transformed into compact natural-language prompts, enabling efficient adaptation under constrained hardware. Experiments on the CICIoT2023 dataset show that a QLoRA-tuned LLaMA-1B model achieves an F1-score of 0.7124, comparable to the Random Forest (RF) baseline (0.7159) for known attacks. With RAG, the system attains 42.63% accuracy on unseen attack types without additional training, demonstrating practical zero-shot capability. These results highlight the potential of retrieval-enhanced lightweight LLMs as adaptable and resource-efficient solutions for next-generation IoT intrusion detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19691v2">Scalable Stewardship of an LLM-Assisted Clinical Benchmark with Physician Oversight</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 Project codebase: https://github.com/junzeye/validate-medcalc-labels
    </div>
    <details class="paper-abstract">
      We examine the reliability of a widely used clinical AI benchmark whose reference labels were partially generated by LLMs, and find that a substantial fraction are clinically misaligned. We introduce a phased stewardship procedure to amplify the positive impact of physician experts' feedback and then demonstrate, via a controlled RL experiment, how uncaught label bias can materially affect downstream LLM evaluation and alignment. Our results demonstrate that partially LLM-generated labels can embed systemic errors that distort not only evaluation but also downstream model alignment. By adopting a hybrid oversight system, we can prioritize scarce expert feedback to maintain benchmarks as living, clinically-grounded documents. Ensuring this alignment is a prerequisite for the safe deployment of LLMs in high-stakes medical decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15251v1">The Effect of Scripts and Formats on LLM Numeracy</a></div>
    <div class="paper-meta">
      📅 2026-01-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved impressive proficiency in basic arithmetic, rivaling human-level performance on standard numerical tasks. However, little attention has been given to how these models perform when numerical expressions deviate from the prevailing conventions present in their training corpora. In this work, we investigate numerical reasoning across a wide range of numeral scripts and formats. We show that LLM accuracy drops substantially when numerical inputs are rendered in underrepresented scripts or formats, despite the underlying mathematical reasoning being identical. We further demonstrate that targeted prompting strategies, such as few-shot prompting and explicit numeral mapping, can greatly narrow this gap. Our findings highlight an overlooked challenge in multilingual numerical reasoning and provide actionable insights for working with LLMs to reliably interpret, manipulate, and generate numbers across diverse numeral scripts and formatting styles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15247v1">Taxonomy-Aligned Risk Extraction from 10-K Filings with Autonomous Improvement Using LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-21
      | 💬 4 figures, 9 pages
    </div>
    <details class="paper-abstract">
      We present a methodology for extracting structured risk factors from corporate 10-K filings while maintaining adherence to a predefined hierarchical taxonomy. Our three-stage pipeline combines LLM extraction with supporting quotes, embedding-based semantic mapping to taxonomy categories, and LLM-as-a-judge validation that filters spurious assignments. To evaluate our approach, we extract 10,688 risk factors from S&P 500 companies and examine risk profile similarity across industry clusters. Beyond extraction, we introduce autonomous taxonomy maintenance where an AI agent analyzes evaluation feedback to identify problematic categories, diagnose failure patterns, and propose refinements, achieving 104.7% improvement in embedding separation in a case study. External validation confirms the taxonomy captures economically meaningful structure: same-industry companies exhibit 63% higher risk profile similarity than cross-industry pairs (Cohen's d=1.06, AUC 0.82, p<0.001). The methodology generalizes to any domain requiring taxonomy-aligned extraction from unstructured text, with autonomous improvement enabling continuous quality maintenance and enhancement as systems process more documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03597v2">From Chains to Graphs: Self-Structured Reasoning for General-Domain LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show strong reasoning ability in open-domain question answering, yet their reasoning processes are typically linear and often logically inconsistent. In contrast, real-world reasoning requires integrating multiple premises and solving subproblems in parallel. Existing methods, such as Chain-of-Thought (CoT), express reasoning in a linear textual form, which may appear coherent but frequently leads to inconsistent conclusions. Recent approaches rely on externally provided graphs and do not explore how LLMs can construct and use their own graph-structured reasoning, particularly in open-domain QA. To fill this gap, we novelly explore graph-structured reasoning of LLMs in general-domain question answering. We propose Self-Graph Reasoning (SGR), a framework that enables LLMs to explicitly represent their reasoning process as a structured graph before producing the final answer. We further construct a graph-structured reasoning dataset that merges multiple candidate reasoning graphs into refined graph structures for model training. Experiments on five QA benchmarks across both general and specialized domains show that SGR consistently improves reasoning consistency and yields a 17.74% gain over the base model. The LLaMA-3.3-70B model fine-tuned with SGR performs comparably to GPT-4o and surpasses Claude-3.5-Haiku, demonstrating the effectiveness of graph-structured reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14532v1">Search over Self-Edit Strategies for LLM Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Many LLM-based open-ended search systems freeze the foundation model that proposes improvements to existing solutions, which may bottleneck long-run progress. Recent work has explored updating the proposal model at test time [arXiv:2511.23473], but the update strategy is still typically hand-specified. Therefore, this study investigated whether an LLM can use task feedback to decide how it should update its weights. For tractability, we focused on the simpler case where there is only one round of self-improvement, and restricted the update operator to self-supervised next token prediction (NTP), leaving the model freedom in choosing its training data and key NTP hyperparameters. Using the Self-Adapting Language Models (SEAL) [arXiv:2506.10943] framework as a testbed, we relaxed its fixed human template constraint and allowed the model to generate its own self-edit templates, thereby giving it more control over its training data and hyperparameters. Two variants were studied, differing in whether template generation was conditioned on a lightweight archive of past templates. In SEAL's Single-Passage Knowledge Incorporation setting with Qwen3-8B on SQuAD [arXiv:1606.05250], the no-archive variant performed comparably to the weaker "Implications" baseline, while the archive variant outperformed "Implications" and approached the strongest human-designed "Rewrite" baseline without surpassing it. Further analysis of collapse in the model's exploration revealed that a naive archive can confer some short-term robustness but can also accelerate homogenization, suggesting that explicit novelty pressure may be required to consistently advance beyond carefully optimized human strategies. Our code is available at https://github.com/cheongalc/search-self-edit-strategies .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14528v1">LLM Security and Safety: Insights from Homotopy-Inspired Prompt Obfuscation</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      In this study, we propose a homotopy-inspired prompt obfuscation framework to enhance understanding of security and safety vulnerabilities in Large Language Models (LLMs). By systematically applying carefully engineered prompts, we demonstrate how latent model behaviors can be influenced in unexpected ways. Our experiments encompassed 15,732 prompts, including 10,000 high-priority cases, across LLama, Deepseek, KIMI for code generation, and Claude to verify. The results reveal critical insights into current LLM safeguards, highlighting the need for more robust defense mechanisms, reliable detection strategies, and improved resilience. Importantly, this work provides a principled framework for analyzing and mitigating potential weaknesses, with the goal of advancing safe, responsible, and trustworthy AI technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01129v2">RovoDev Code Reviewer: A Large-Scale Online Evaluation of LLM-based Code Review Automation at Atlassian</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Accepted at the 48th International Conference on Software Engineering (ICSE'26), SEIP Track. 12 Pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs)-powered code review automation has the potential to transform code review workflows. Despite the advances of LLM-powered code review comment generation approaches, several practical challenges remain for designing enterprise-grade code review automation tools. In particular, this paper aims at answering the practical question: how can we design a review-guided, context-aware, quality-checked code review comment generation without fine-tuning? In this paper, we present RovoDev Code Reviewer, an enterprise-grade LLM-based code review automation tool designed and deployed at scale within Atlassian's development ecosystem with seamless integration into Atlassian's Bitbucket. Through the offline, online, user feedback evaluations over a one-year period, we conclude that RovoDev Code Reviewer is effective in generating code review comments that could lead to code resolution for 38.70% (i.e., comments that triggered code changes in the subsequent commits); and offers the promise of accelerating feedback cycles (i.e., decreasing the PR cycle time by 30.8%), alleviating reviewer workload (i.e., reducing the number of human-written comments by 35.6%), and improving overall software quality (i.e., finding errors with actionable suggestions).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.17792v4">H3Fusion: Helpful, Harmless, Honest Fusion of Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      The alignment of pre-trained LLMs continues to draw significant attention from both industry and academia, aiming to ensure responses that are helpful, harmless, and honest. However, identifying a point in the model's representation subspace that simultaneously satisfies all these properties remains challenging. H3Fusion addresses this challenge by introducing a mixture-of-experts (MoE)-based fusion mechanism that models alignment as a controllable drift within the subspace, guided by a drift-regularization loss to balance competing alignment dimensions. Furthermore, we formulate the alignment by finding a dual objective of harnessing the distance of generated embeddings and alignment embeddings, and introduce a gating loss by canalizing the activations on the contributing experts. Extensive evaluations of three benchmark datasets show that H3Fusion is more helpful, less harmful, and more honest in three aspects: it outperforms each individually aligned model by 11.37%, and provides stronger robustness compared to the state-of-the-art LLM ensemble approaches by 13.77% and model-merging approaches by 6.18%. Code is available at https://github.com/git-disl/h3fusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03149v2">PersonaLedger: Generating Realistic Financial Transactions with Persona Conditioned LLMs and Rule Grounded Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Strict privacy regulations limit access to real transaction data, slowing open research in financial AI. Synthetic data can bridge this gap, but existing generators do not jointly achieve behavioral diversity and logical groundedness. Rule-driven simulators rely on hand-crafted workflows and shallow stochasticity, which miss the richness of human behavior. Learning-based generators such as GANs capture correlations yet often violate hard financial constraints and still require training on private data. We introduce PersonaLedger, a generation engine that uses a large language model conditioned on rich user personas to produce diverse transaction streams, coupled with an expert configurable programmatic engine that maintains correctness. The LLM and engine interact in a closed loop: after each event, the engine updates the user state, enforces financial rules, and returns a context aware "nextprompt" that guides the LLM toward feasible next actions. With this engine, we create a public dataset of 30 million transactions from 23,000 users and a benchmark suite with two tasks, illiquidity classification and identity theft segmentation. PersonaLedger offers a realistic, privacy preserving resource that supports rigorous evaluation of forecasting and anomaly detection models. PersonaLedger offers the community a rich, realistic, and privacy preserving resource -- complete with code, rules, and generation logs -- to accelerate innovation in financial AI and enable rigorous, reproducible evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14479v1">Can LLM Reasoning Be Trusted? A Comparative Study: Using Human Benchmarking on Statistical Tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      This paper investigates the ability of large language models (LLMs) to solve statistical tasks, as well as their capacity to assess the quality of reasoning. While state-of-the-art LLMs have demonstrated remarkable performance in a range of NLP tasks, their competence in addressing even moderately complex statistical challenges is not well understood. We have fine-tuned selected open-source LLMs on a specially developed dataset to enhance their statistical reasoning capabilities, and compared their performance with the human scores used as a benchmark. Our results show that the fine-tuned models achieve better performance on advanced statistical tasks on the level comparable to a statistics student. Fine-tuning demonstrates architecture-dependent improvements, with some models showing significant performance gains, indicating clear potential for deployment in educational technology and statistical analysis assistance systems. We also show that LLMs themselves can be far better judges of the answers quality (including explanation and reasoning assessment) in comparison to traditional metrics, such as BLEU or BertScore. This self-evaluation capability enables scalable automated assessment for statistical education platforms and quality assurance in automated analysis tools. Potential applications also include validation tools for research methodology in academic and industry settings, and quality control mechanisms for data analysis workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14456v1">On the Generalization Gap in LLM Planning: Tests and Verifier-Reward RL</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 9 pages, 4 figures, 3 tables, 2 pages of supplementary materials. Submitted to a conference implementing a double-blind review process
    </div>
    <details class="paper-abstract">
      Recent work shows that fine-tuned Large Language Models (LLMs) can achieve high valid plan rates on PDDL planning tasks. However, it remains unclear whether this reflects transferable planning competence or domain-specific memorization. In this work, we fine-tune a 1.7B-parameter LLM on 40,000 domain-problem-plan tuples from 10 IPC 2023 domains, and evaluate both in-domain and cross-domain generalization. While the model reaches 82.9% valid plan rate in in-domain conditions, it achieves 0% on two unseen domains. To analyze this failure, we introduce three diagnostic interventions, namely (i) instance-wise symbol anonymization, (ii) compact plan serialization, and (iii) verifier-reward fine-tuning using the VAL validator as a success-focused reinforcement signal. Symbol anonymization and compact serialization cause significant performance drops despite preserving plan semantics, thus revealing strong sensitivity to surface representations. Verifier-reward fine-tuning reaches performance saturation in half the supervised training epochs, but does not improve cross-domain generalization. For the explored configurations, in-domain performance plateaus around 80%, while cross-domain performance collapses, suggesting that our fine-tuned model relies heavily on domain-specific patterns rather than transferable planning competence in this setting. Our results highlight a persistent generalization gap in LLM-based planning and provide diagnostic tools for studying its causes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11150v3">Causal Judge Evaluation: Calibrated Surrogate Metrics for LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Code: https://github.com/cimo-labs/cje Experiments for Reproducibility: https://github.com/cimo-labs/cje-arena-experiments Original Preprint: https://zenodo.org/records/17903629
    </div>
    <details class="paper-abstract">
      Measuring long-run LLM outcomes (user satisfaction, expert judgment, downstream KPIs) is expensive. Teams default to cheap LLM judges, but uncalibrated proxies can invert rankings entirely. Causal Judge Evaluation (CJE) makes it affordable to aim at the right target: calibrate cheap scores against a small oracle slice, then evaluate at scale with valid uncertainty. We treat surrogate validity as auditable: for each policy or deployment context, a small oracle audit tests whether the learned calibration remains mean-unbiased, turning an uncheckable identification condition into a falsifiable diagnostic. On 4,961 Chatbot Arena prompts comparing five policies with a 16x oracle/judge cost ratio, at a 5% oracle fraction CJE achieves 99% pairwise ranking accuracy at 14x lower cost; across all configurations (5-50% oracle, varying n), accuracy averages 94%. An adversarial policy fails the transport audit and is correctly flagged; in such cases CJE refuses level claims rather than reporting biased estimates. Key findings: naive confidence intervals on raw judge scores achieve 0% coverage (CJE: ~95%); importance-weighted estimators fail despite >90% effective sample size; and the Coverage-Limited Efficiency (CLE) bound and its TTC diagnostic explain why.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14340v1">Turn-Based Structural Triggers: Prompt-Free Backdoors in Multi-Turn LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely integrated into interactive systems such as dialogue agents and task-oriented assistants. This growing ecosystem also raises supply-chain risks, where adversaries can distribute poisoned models that degrade downstream reliability and user trust. Existing backdoor attacks and defenses are largely prompt-centric, focusing on user-visible triggers while overlooking structural signals in multi-turn conversations. We propose Turn-based Structural Trigger (TST), a backdoor attack that activates from dialogue structure, using the turn index as the trigger and remaining independent of user inputs. Across four widely used open-source LLM models, TST achieves an average attack success rate (ASR) of 99.52% with minimal utility degradation, and remains effective under five representative defenses with an average ASR of 98.04%. The attack also generalizes well across instruction datasets, maintaining an average ASR of 99.19%. Our results suggest that dialogue structure constitutes an important and under-studied attack surface for multi-turn LLM systems, motivating structure-aware auditing and mitigation in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03013v3">LLMs, You Can Evaluate It! Design of Multi-perspective Report Evaluation for Security Operation Centers</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Security operation centers (SOCs) often produce analysis reports on security incidents, and large language models (LLMs) will likely be used for this task in the near future. We postulate that a better understanding of how veteran analysts evaluate reports, including their feedback, can help produce analysis reports in SOCs. In this paper, we aim to leverage LLMs for analysis reports. To this end, we first construct a Analyst-wise checklist to reflect SOC practitioners' opinions for analysis report evaluation through literature review and user study with SOC practitioners. Next, we design a novel LLM-based conceptual framework, named MESSALA, by further introducing two new techniques, granularization guideline and multi-perspective evaluation. MESSALA can maximize report evaluation and provide feedback on veteran SOC practitioners' perceptions. When we conduct extensive experiments with MESSALA, the evaluation results by MESSALA are the closest to those of veteran SOC practitioners compared with the existing LLM-based methods. We then show two key insights. We also conduct qualitative analysis with MESSALA, and then identify that MESSALA can provide actionable items that are necessary for improving analysis reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.06266v3">Self-Admitted Technical Debt in LLM Software: An Empirical Comparison with ML and Non-ML Software</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Accepted to SANER 2026 (IEEE International Conference on Software Analysis, Evolution and Reengineering)
    </div>
    <details class="paper-abstract">
      Self-admitted technical debt (SATD), referring to comments flagged by developers that explicitly acknowledge suboptimal code or incomplete functionality, has received extensive attention in machine learning (ML) and traditional (Non-ML) software. However, little is known about how SATD manifests and evolves in contemporary Large Language Model (LLM)-based systems, whose architectures, workflows, and dependencies differ fundamentally from both traditional and pre-LLM ML software. In this paper, we conduct the first empirical study of SATD in the LLM era, replicating and extending prior work on ML technical debt to modern LLM-based systems. We compare SATD prevalence across LLM, ML, and non-ML repositories across a total of 477 repositories (159 per category). We perform survival analysis of SATD introduction and removal to understand the dynamics of technical debt across different development paradigms. Surprisingly, despite their architectural complexity, our results reveal that LLM repositories accumulate SATD at similar rates to ML systems (3.95% vs. 4.10%). However, we observe that LLM repositories remain debt-free 2.4x longer than ML repositories (a median of 492 days vs. 204 days), and then start to accumulate technical debt rapidly. Moreover, our qualitative analysis of 377 SATD instances reveals three new forms of technical debt unique to LLM-based development that have not been reported in prior research: Model-Stack Workaround Debt, Model Dependency Debt, and Performance Optimization Debt. Finally, by mapping SATD to stages of the LLM development pipeline, we observe that debt concentrates
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14209v1">InT: Self-Proposed Interventions Enable Credit Assignment in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Outcome-reward reinforcement learning (RL) has proven effective at improving the reasoning capabilities of large language models (LLMs). However, standard RL assigns credit only at the level of the final answer, penalizing entire reasoning traces when the outcome is incorrect and uniformly reinforcing all steps when it is correct. As a result, correct intermediate steps may be discouraged in failed traces, while spurious steps may be reinforced in successful ones. We refer to this failure mode as the problem of credit assignment. While a natural remedy is to train a process reward model, accurately optimizing such models to identify corrective reasoning steps remains challenging. We introduce Intervention Training (InT), a training paradigm in which the model performs fine-grained credit assignment on its own reasoning traces by proposing short, targeted corrections that steer trajectories toward higher reward. Using reference solutions commonly available in mathematical reasoning datasets and exploiting the fact that verifying a model-generated solution is easier than generating a correct one from scratch, the model identifies the first error in its reasoning and proposes a single-step intervention to redirect the trajectory toward the correct solution. We then apply supervised fine-tuning (SFT) to the on-policy rollout up to the point of error concatenated with the intervention, localizing error to the specific step that caused failure. We show that the resulting model serves as a far better initialization for RL training. After running InT and subsequent fine-tuning with RL, we improve accuracy by nearly 14% over a 4B-parameter base model on IMO-AnswerBench, outperforming larger open-source models such as gpt-oss-20b.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.15364v4">KeyDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 37 pages, 19 figures, NeurIPS 2025
    </div>
    <details class="paper-abstract">
      We demonstrate that geometrically distinctive keys during LLM inference tend to have high attention scores. Based on the phenomenon we propose KeyDiff, a training-free KV cache eviction method based solely on key similarity. Unlike other KV cache eviction methods, KeyDiff can process arbitrarily long prompts within strict resource constraints and efficiently generate responses. We provide a theoretical basis for KeyDiff by relating key diversity with attention scores. These results imply KeyDiff can efficiently identify the most important tokens to retain. Notably KeyDiff does not rely on attention scores, allowing the use of optimized attention mechanisms like FlashAttention. Under a strict memory allowance, we demonstrate the effectiveness of KeyDiff for the Llama and Qwen model families by observing a performance gap of less than 0.04% with 8K cache budget ($\sim$23% KV cache reduction) from the non-evicting baseline on LongBench for Llama 3.1-8B and Llama 3.2-3B. We also observe near baseline performance for Deepseek-R1-Distill-Llama-8B on the Math500 reasoning benchmark and decrease end-to-end inference latency by up to 30% compared to the other token-eviction methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14154v1">LLM Augmented Intervenable Multimodal Adaptor for Post-operative Complication Prediction in Lung Cancer Surgery</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Accepted to P2P-CV @ WACV 2026
    </div>
    <details class="paper-abstract">
      Postoperative complications remain a critical concern in clinical practice, adversely affecting patient outcomes and contributing to rising healthcare costs. We present MIRACLE, a deep learning architecture for prediction of risk of postoperative complications in lung cancer surgery by integrating preoperative clinical and radiological data. MIRACLE employs a hyperspherical embedding space fusion of heterogeneous inputs, enabling the extraction of robust, discriminative features from both structured clinical records and high-dimensional radiological images. To enhance transparency of prediction and clinical utility, we incorporate an interventional deep learning module in MIRACLE, that not only refines predictions but also provides interpretable and actionable insights, allowing domain experts to interactively adjust recommendations based on clinical expertise. We validate our approach on POC-L, a real-world dataset comprising 3,094 lung cancer patients who underwent surgery at Roswell Park Comprehensive Cancer Center. Our results demonstrate that MIRACLE outperforms various traditional machine learning models and contemporary large language models (LLM) variants alone, for personalized and explainable postoperative risk management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.17001v3">PersonalAI: A Systematic Comparison of Knowledge Graph Storage and Retrieval Approaches for Personalized LLM agents</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Personalizing language models that effectively incorporating user interaction history remains a central challenge in development of adaptive AI systems. While large language models (LLMs), combined with Retrieval-Augmented Generation (RAG), have improved factual accuracy, they often lack structured memory and fail to scale in complex, long-term interactions. To address this, we propose a flexible external memory framework based on knowledge graph, which construct and update memory model automatically by LLM itself. Building upon the AriGraph architecture, we introduce a novel hybrid graph design that supports both standard edges and two types of hyper-edges, enabling rich and dynamic semantic and temporal representations. Our framework also supports diverse retrieval mechanisms, including A*, water-circle traversal, beam search and hybrid methods, making it adaptable to different datasets and LLM capacities. We evaluate our system on three benchmarks: TriviaQA, HotpotQA, DiaASQ and demonstrate that different memory and retrieval configurations yield optimal performance depending on the task. Additionally, we extend the DiaASQ benchmark with temporal annotations and internally contradictory statements, showing that our system remains robust and effective in managing temporal dependencies and context-aware reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.11574v4">Quantization Meets Reasoning: Exploring and Mitigating Degradation of Low-Bit LLMs in Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 27pages
    </div>
    <details class="paper-abstract">
      Low-bit post-training quantization (PTQ) is a practical route to deploy reasoning-capable LLMs under tight memory and latency budgets, yet it can markedly impair mathematical reasoning (drops up to 69.81% in our harder settings). We address two deployment-critical questions with process-level precision: Where along a step-structured solution does degradation first arise? How to mitigate it while staying in the low-bit regime? Across widely used PTQ methods (AWQ, GPTQ, SmoothQuant), open-source model families (Qwen, LLaMA; 0.5--7B), and math reasoning benchmarks (GSM8K, MATH, AIME), we perform format-aligned chain-of-thought with step-aligned attribution and uncover two robust regularities: (i) PTQ disproportionately elevates method and execution errors relative to high-level conceptual mistakes; and (ii) failures emerge early, with the first vulnerable step flipping and cascading to the final answer. These regularities suggest a general intervention principle: restore local token-level margins exactly at the earliest failure frontier. We instantiate this principle as a lightweight measure$\rightarrow$locate$\rightarrow$restore loop that operates directly on the quantized model: detect the first faulty step, construct our "Silver Bullet" datasets, and apply small-scale supervised/preference tuning. In our settings, as few as 332 curated examples and 3--5 minutes of compute on a single GPU recover 4-bit weight math reasoning toward the full-precision baseline while preserving PTQ efficiency. Our framework is quantizer- and architecture-agnostic within the evaluated regimes, and turns low-bit degradation from a global accuracy problem into a local, reproducible process intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.04492v2">Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 8 pages, 5 figures, 2 tables. pre-print version
    </div>
    <details class="paper-abstract">
      Hallucinations in Large Language Model (LLM) outputs for Question Answering (QA) tasks can critically undermine their real-world reliability. This paper introduces a methodology for robust, one-shot hallucination detection, specifically designed for scenarios with limited data access, such as interacting with black-box LLM APIs that typically expose only a few top candidate log-probabilities per token. Our approach derives uncertainty indicators directly from these readily available log-probabilities generated during non-greedy decoding. We first derive an Entropy Production Rate (EPR) that offers baseline performance, later augmented with supervised learning. Our learned model leverages the entropic contributions of the accessible top-ranked tokens within a single generated sequence, without multiple re-runs per query. Evaluated across diverse QA datasets and multiple LLMs, this estimator significantly improves token-level hallucination detection over state-of-the-art methods. Crucially, high performance is demonstrated using only the typically small set of available log-probabilities (e.g., top-10 per token), confirming its practical efficiency and suitability for API-constrained deployments. This work provides a lightweight technique to enhance the trustworthiness of LLM responses, at the token level, after a single generation pass, for QA and Retrieval-Augmented Generation (RAG) systems. Our experiments confirmed the performance of our method against existing approaches on public dataset as well as for a financial framework analyzing annual company reports.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.07902v2">Tailored Emotional LLM-Supporter: Enhancing Cultural Sensitivity</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Joint first authors; EACL
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise in offering emotional support and generating empathetic responses for individuals in distress, but their ability to deliver culturally sensitive support remains underexplored due to a lack of resources. In this work, we introduce CultureCare, the first dataset designed for this task, spanning four cultures and including 1729 distress messages, 1523 cultural signals, and 1041 support strategies with fine-grained emotional and cultural annotations. Leveraging CultureCare, we (i) develop and test four adaptation strategies for guiding three state-of-the-art LLMs toward culturally sensitive responses; (ii) conduct comprehensive evaluations using LLM-as-a-Judge, in-culture human annotators, and clinical psychologists; (iii) show that adapted LLMs outperform anonymous online peer responses, and that simple cultural role-play is insufficient for cultural sensitivity; and (iv) explore the application of LLMs in clinical training, where experts highlight their potential in fostering cultural competence in novice therapists.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14063v1">XCR-Bench: A Multi-Task Benchmark for Evaluating Cultural Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 30 Pages, 13 Figures
    </div>
    <details class="paper-abstract">
      Cross-cultural competence in large language models (LLMs) requires the ability to identify Culture-Specific Items (CSIs) and to adapt them appropriately across cultural contexts. Progress in evaluating this capability has been constrained by the scarcity of high-quality CSI-annotated corpora with parallel cross-cultural sentence pairs. To address this limitation, we introduce XCR-Bench, a Cross(X)-Cultural Reasoning Benchmark consisting of 4.9k parallel sentences and 1,098 unique CSIs, spanning three distinct reasoning tasks with corresponding evaluation metrics. Our corpus integrates Newmark's CSI framework with Hall's Triad of Culture, enabling systematic analysis of cultural reasoning beyond surface-level artifacts and into semi-visible and invisible cultural elements such as social norms, beliefs, and values. Our findings show that state-of-the-art LLMs exhibit consistent weaknesses in identifying and adapting CSIs related to social etiquette and cultural reference. Additionally, we find evidence that LLMs encode regional and ethno-religious biases even within a single linguistic setting during cultural adaptation. We release our corpus and code to facilitate future research on cross-cultural NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14050v1">Understanding Multilingualism in Mixture-of-Experts LLMs: Routing Mechanism, Expert Specialization, and Layerwise Steering</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) architectures have shown strong multilingual capabilities, yet the internal mechanisms underlying performance gains and cross-language differences remain insufficiently understood. In this work, we conduct a systematic analysis of MoE models, examining routing behavior and expert specialization across languages and network depth. Our analysis reveals that multilingual processing in MoE models is highly structured: routing aligns with linguistic families, expert utilization follows a clear layerwise pattern, and high-resource languages rely on shared experts while low-resource languages depend more on language-exclusive experts despite weaker performance. Layerwise interventions further show that early and late MoE layers support language-specific processing, whereas middle layers serve as language-agnostic capacity hubs. Building on these insights, we propose a routing-guided steering method that adaptively guides routing behavior in middle layers toward shared experts associated with dominant languages at inference time, leading to consistent multilingual performance improvements, particularly for linguistically related language pairs. Our code is available at https://github.com/conctsai/Multilingualism-in-Mixture-of-Experts-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14032v1">RM-Distiller: Exploiting Generative LLM for Reward Model Distillation</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Reward models (RMs) play a pivotal role in aligning large language models (LLMs) with human preferences. Due to the difficulty of obtaining high-quality human preference annotations, distilling preferences from generative LLMs has emerged as a standard practice. However, existing approaches predominantly treat teacher models as simple binary annotators, failing to fully exploit the rich knowledge and capabilities for RM distillation. To address this, we propose RM-Distiller, a framework designed to systematically exploit the multifaceted capabilities of teacher LLMs: (1) Refinement capability, which synthesizes highly correlated response pairs to create fine-grained and contrastive signals. (2) Scoring capability, which guides the RM in capturing precise preference strength via a margin-aware optimization objective. (3) Generation capability, which incorporates the teacher's generative distribution to regularize the RM to preserve its fundamental linguistic knowledge. Extensive experiments demonstrate that RM-Distiller significantly outperforms traditional distillation methods both on RM benchmarks and reinforcement learning-based alignment, proving that exploiting multifaceted teacher capabilities is critical for effective reward modeling. To the best of our knowledge, this is the first systematic research on RM distillation from generative LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13995v1">From Tags to Trees: Structuring Fine-Grained Knowledge for Controllable Data Selection in LLM Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Effective and controllable data selection is critical for LLM instruction tuning, especially with massive open-source datasets. Existing approaches primarily rely on instance-level quality scores, or diversity metrics based on embedding clusters or semantic tags. However, constrained by the flatness of embedding spaces or the coarseness of tags, these approaches overlook fine-grained knowledge and its intrinsic hierarchical dependencies, consequently hindering precise data valuation and knowledge-aligned sampling. To address this challenge, we propose Tree-aware Aligned Global Sampling (TAGS), a unified framework that leverages a knowledge tree built from fine-grained tags, thereby enabling joint control of global quality, diversity, and target alignment. Using an LLM-based tagger, we extract atomic knowledge concepts, which are organized into a global tree through bottom-up hierarchical clustering. By grounding data instances onto this tree, a tree-aware metric then quantifies data quality and diversity, facilitating effective sampling. Our controllable sampling strategy maximizes tree-level information gain and enforces leaf-level alignment via KL-divergence for specific domains. Extensive experiments demonstrate that TAGS significantly outperforms state-of-the-art baselines. Notably, it surpasses the full-dataset model by \textbf{+5.84\%} using only \textbf{5\%} of the data, while our aligned sampling strategy further boosts average performance by \textbf{+4.24\%}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.04345v3">Jingfang: An LLM-Based Multi-Agent System for Precise Medical Consultation and Syndrome Differentiation in Traditional Chinese Medicine</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      The practice of Traditional Chinese Medicine (TCM) requires profound expertise and extensive clinical experience. While Large Language Models (LLMs) offer significant potential in this domain, current TCM-oriented LLMs suffer two critical limitations: (1) a rigid consultation framework that fails to conduct comprehensive and patient-tailored interactions, often resulting in diagnostic inaccuracies; and (2) treatment recommendations generated without rigorous syndrome differentiation, which deviates from the core diagnostic and therapeutic principles of TCM. To address these issues, we develop \textbf{JingFang (JF)}, an advanced LLM-based multi-agent system for TCM that facilitates the implementation of AI-assisted TCM diagnosis and treatment. JF integrates various TCM Specialist Agents in accordance with authentic diagnostic and therapeutic scenarios of TCM, enabling personalized medical consultations, accurate syndrome differentiation and treatment recommendations. A \textbf{Multi-Agent Collaborative Consultation Mechanism (MACCM)} for TCM is constructed, where multiple Agents collaborate to emulate real-world TCM diagnostic workflows, enhancing the diagnostic ability of base LLMs to provide accurate and patient-tailored medical consultation. Moreover, we introduce a dedicated \textbf{Syndrome Differentiation Agent} fine-tuned on a preprocessed dataset, along with a designed \textbf{Dual-Stage Recovery Scheme (DSRS)} within the Treatment Agent, which together substantially improve the model's accuracy of syndrome differentiation and treatment. Comprehensive evaluations and experiments demonstrate JF's superior performance in medical consultation, and also show improvements of at least 124% and 21.1% in the precision of syndrome differentiation compared to existing TCM models and State of the Art (SOTA) LLMs, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.14904v3">Efficient Switchable Safety Control in LLMs via Magic-Token-Guided Co-Training</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 15 pages,3 figures,5 tables
    </div>
    <details class="paper-abstract">
      Current methods for content safety in Large Language Models (LLMs), such as Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), often rely on multi-stage training pipelines and lack fine-grained, post-deployment controllability. To address these limitations, we propose a unified co-training framework that efficiently integrates multiple safety behaviors: positive (lawful/prosocial), negative (unfiltered/risk-prone) and rejective (refusal-oriented/conservative) within a single SFT stage. Notably, each behavior is dynamically activated via a simple system-level instruction, or magic token, enabling stealthy and efficient behavioral switching at inference time. This flexibility supports diverse deployment scenarios, such as positive for safe user interaction, negative for internal red-teaming, and rejective for context-aware refusals triggered by upstream moderation signals. This co-training strategy induces a distinct Safety Alignment Margin in the output space, characterized by well-separated response distributions corresponding to each safety mode. The existence of this margin provides empirical evidence for the model's safety robustness and enables unprecedented fine-grained control. Experiments show that our method matches the safety alignment quality of SFT+DPO, with our 8B model notably surpassing DeepSeek-R1 (671B) in safety performance, while significantly reducing both training complexity and deployment costs. This work presents a scalable, efficient, and highly controllable solution for LLM content safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13933v1">VulnResolver: A Hybrid Agent Framework for LLM-Based Automated Vulnerability Issue Resolution</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      As software systems grow in complexity, security vulnerabilities have become increasingly prevalent, posing serious risks and economic costs. Although automated detection tools such as fuzzers have advanced considerably, effective resolution still often depends on human expertise. Existing automated vulnerability repair (AVR) methods rely heavily on manually provided annotations (e.g., fault locations or CWE labels), which are often difficult and time-consuming to obtain, while overlooking the rich, naturally embedded semantic context found in issue reports from developers. In this paper, we present VulnResolver, the first LLM-based hybrid agent framework for automated vulnerability issue resolution. VulnResolver unites the adaptability of autonomous agents with the stability of workflow-guided repair through two specialized agents. The Context Pre-Collection Agent (CPCAgent) adaptively explores the repository to gather dependency and contextual information, while the Safety Property Analysis Agent (SPAAgent) generates and validates the safety properties violated by vulnerabilities. Together, these agents produce structured analyses that enrich the original issue reports, enabling more accurate vulnerability localization and patch generation. Evaluations on the SEC-bench benchmark show that VulnResolver resolves 75% of issues on SEC-bench Lite, achieving the best resolution performance. On SEC-bench Full, VulnResolver also significantly outperforms the strongest baseline, the agent-based OpenHands, confirming its effectiveness. Overall, VulnResolver delivers an adaptive and security-aware framework that advances end-to-end automated vulnerability issue resolution through workflow stability and the specialized agents' capabilities in contextual reasoning and property-based analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12509v2">Revitalizing Black-Box Interpretability: Actionable Interpretability for LLMs via Proxy Models</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Post-hoc explanations provide transparency and are essential for guiding model optimization, such as prompt engineering and data sanitation. However, applying model-agnostic techniques to Large Language Models (LLMs) is hindered by prohibitive computational costs, rendering these tools dormant for real-world applications. To revitalize model-agnostic interpretability, we propose a budget-friendly proxy framework that leverages efficient models to approximate the decision boundaries of expensive LLMs. We introduce a screen-and-apply mechanism to statistically verify local alignment before deployment. Our empirical evaluation confirms that proxy explanations achieve over 90% fidelity with only 11% of the oracle's cost. Building on this foundation, we demonstrate the actionable utility of our framework in prompt compression and poisoned example removal. Results show that reliable proxy explanations effectively guide optimization, transforming interpretability from a passive observation tool into a scalable primitive for LLM development. Additionally, we open-source code and datasets to facilitate future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09550v4">Integrating Symbolic Execution with LLMs for Automated Generation of Program Specifications</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Automatically generating formal specifications including loop invariants, preconditions, and postconditions for legacy code is critical for program understanding, reuse and verification. However, the inherent complexity of control and data structures in programs makes this task particularly challenging. This paper presents a novel framework that integrates symbolic execution with large language models (LLMs) to automatically synthesize formally verified program specifications. Our method first employs symbolic execution to derive precise strongest postconditions for loop-free code segments. These symbolic execution results, along with automatically generated invariant templates, then guide the LLM to propose and iteratively refine loop invariants until a correct specification is obtained. The template-guided generation process robustly combines symbolic inference with LLM reasoning, significantly reducing hallucinations and syntactic errors by structurally constraining the LLM's output space. Furthermore, our approach can produce strong specifications without relying on externally provided verification goals, enabled by the rich semantic context supplied by symbolic execution, overcoming a key limitation of prior goal-dependent tools. Extensive evaluation shows that our tool SESpec outperforms the existing state-of-the-art tools across numerical and data-structure benchmarks, demonstrating both high precision and broad applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.10795v4">Beyond "Not Novel Enough": Enriching Scholarly Critique with LLM-Assisted Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Novelty assessment is a central yet understudied aspect of peer review, particularly in high volume fields like NLP where reviewer capacity is increasingly strained. We present a structured approach for automated novelty evaluation that models expert reviewer behavior through three stages: content extraction from submissions, retrieval and synthesis of related work, and structured comparison for evidence based assessment. Our method is informed by a large scale analysis of human written novelty reviews and captures key patterns such as independent claim verification and contextual reasoning. Evaluated on 182 ICLR 2025 submissions with human annotated reviewer novelty assessments, the approach achieves 86.5% alignment with human reasoning and 75.3% agreement on novelty conclusions - substantially outperforming existing LLM based baselines. The method produces detailed, literature aware analyses and improves consistency over ad hoc reviewer judgments. These results highlight the potential for structured LLM assisted approaches to support more rigorous and transparent peer review without displacing human expertise. Data and code are made available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11369v2">Institutional AI: Governing LLM Collusion in Multi-Agent Cournot Markets via Public Governance Graphs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Multi-agent LLM ensembles can converge on coordinated, socially harmful equilibria. This paper advances an experimental framework for evaluating Institutional AI, our system-level approach to AI alignment that reframes alignment from preference engineering in agent-space to mechanism design in institution-space. Central to this approach is the governance graph, a public, immutable manifest that declares legal states, transitions, sanctions, and restorative paths; an Oracle/Controller runtime interprets this manifest, attaching enforceable consequences to evidence of coordination while recording a cryptographically keyed, append-only governance log for audit and provenance. We apply the Institutional AI framework to govern the Cournot collusion case documented by prior work and compare three regimes: Ungoverned (baseline incentives from the structure of the Cournot market), Constitutional (a prompt-only policy-as-prompt prohibition implemented as a fixed written anti-collusion constitution, and Institutional (governance-graph-based). Across six model configurations including cross-provider pairs (N=90 runs/condition), the Institutional regime produces large reductions in collusion: mean tier falls from 3.1 to 1.8 (Cohen's d=1.28), and severe-collusion incidence drops from 50% to 5.6%. The prompt-only Constitutional baseline yields no reliable improvement, illustrating that declarative prohibitions do not bind under optimisation pressure. These results suggest that multi-agent alignment may benefit from being framed as an institutional design problem, where governance graphs can provide a tractable abstraction for alignment-relevant collective behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13885v1">Confident Rankings with Fewer Items: Adaptive LLM Evaluation with Continuous Scores</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Computerized Adaptive Testing (CAT) has proven effective for efficient LLM evaluation on multiple-choice benchmarks, but modern LLM evaluation increasingly relies on generation tasks where outputs are scored continuously rather than marked correct/incorrect. We present a principled extension of IRT-based adaptive testing to continuous bounded scores (ROUGE, BLEU, LLM-as-a-Judge) by replacing the Bernoulli response distribution with a heteroskedastic normal distribution. Building on this, we introduce an uncertainty aware ranker with adaptive stopping criteria that achieves reliable model ranking while testing as few items and as cheaply as possible. We validate our method on five benchmarks spanning n-gram-based, embedding-based, and LLM-as-judge metrics. Our method uses 2% of the items while improving ranking correlation by 0.12 τ over random sampling, with 95% accuracy on confident predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13864v1">HardSecBench: Benchmarking the Security Awareness of LLMs for Hardware Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are being increasingly integrated into practical hardware and firmware development pipelines for code generation. Existing studies have primarily focused on evaluating the functional correctness of LLM-generated code, yet paid limited attention to its security issues. However, LLM-generated code that appears functionally sound may embed security flaws which could induce catastrophic damages after deployment. This critical research gap motivates us to design a benchmark for assessing security awareness under realistic specifications. In this work, we introduce HardSecBench, a benchmark with 924 tasks spanning Verilog Register Transfer Level (RTL) and firmware-level C, covering 76 hardware-relevant Common Weakness Enumeration (CWE) entries. Each task includes a structured specification, a secure reference implementation, and executable tests. To automate artifact synthesis, we propose a multi-agent pipeline that decouples synthesis from verification and grounds evaluation in execution evidence, enabling reliable evaluation. Using HardSecBench, we evaluate a range of LLMs on hardware and firmware code generation and find that models often satisfy functional requirements while still leaving security risks. We also find that security results vary with prompting. These findings highlight pressing challenges and offer actionable insights for future advancements in LLM-assisted hardware design. Our data and code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13836v1">FutureOmni: Evaluating Future Forecasting from Omni-Modal Context for Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 https://openmoss.github.io/FutureOmni
    </div>
    <details class="paper-abstract">
      Although Multimodal Large Language Models (MLLMs) demonstrate strong omni-modal perception, their ability to forecast future events from audio-visual cues remains largely unexplored, as existing benchmarks focus mainly on retrospective understanding. To bridge this gap, we introduce FutureOmni, the first benchmark designed to evaluate omni-modal future forecasting from audio-visual environments. The evaluated models are required to perform cross-modal causal and temporal reasoning, as well as effectively leverage internal knowledge to predict future events. FutureOmni is constructed via a scalable LLM-assisted, human-in-the-loop pipeline and contains 919 videos and 1,034 multiple-choice QA pairs across 8 primary domains. Evaluations on 13 omni-modal and 7 video-only models show that current systems struggle with audio-visual future prediction, particularly in speech-heavy scenarios, with the best accuracy of 64.8% achieved by Gemini 3 Flash. To mitigate this limitation, we curate a 7K-sample instruction-tuning dataset and propose an Omni-Modal Future Forecasting (OFF) training strategy. Evaluations on FutureOmni and popular audio-visual and video-only benchmarks demonstrate that OFF enhances future forecasting and generalization. We publicly release all code (https://github.com/OpenMOSS/FutureOmni) and datasets (https://huggingface.co/datasets/OpenMOSS-Team/FutureOmni).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13824v1">ELSA: Efficient LLM-Centric Split Aggregation for Privacy-Aware Hierarchical Federated Learning over Resource-Constrained Edge Networks</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 11 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) at the network edge faces fundamental challenges arising from device resource constraints, severe data heterogeneity, and heightened privacy risks. To address these, we propose ELSA (Efficient LLM-centric Split Aggregation), a novel framework that systematically integrates split learning (SL) and hierarchical federated learning (HFL) for distributed LLM fine-tuning over resource-constrained edge networks. ELSA introduces three key innovations. First, it employs a task-agnostic, behavior-aware client clustering mechanism that constructs semantic fingerprints using public probe inputs and symmetric KL divergence, further enhanced by prediction-consistency-based trust scoring and latency-aware edge assignment to jointly address data heterogeneity, client unreliability, and communication constraints. Second, it splits the LLM into three parts across clients and edge servers, with the cloud used only for adapter aggregation, enabling an effective balance between on-device computation cost and global convergence stability. Third, it incorporates a lightweight communication scheme based on computational sketches combined with semantic subspace orthogonal perturbation (SS-OP) to reduce communication overhead while mitigating privacy leakage during model exchanges. Experiments across diverse NLP tasks demonstrate that ELSA consistently outperforms state-of-the-art methods in terms of adaptability, convergence behavior, and robustness, establishing a scalable and privacy-aware solution for edge-side LLM fine-tuning under resource constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13815v1">From RTL to Prompt Coding: Empowering the Next Generation of Chip Designers through LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 Accepted for presentation at the 2026 IEEE International Symposium on Circuits and Systems (ISCAS 2026). Proceedings to be included in IEEE Xplore
    </div>
    <details class="paper-abstract">
      This paper presents an LLM-based learning platform for chip design education, aiming to make chip design accessible to beginners without overwhelming them with technical complexity. It represents the first educational platform that assists learners holistically across both frontend and backend design. The proposed approach integrates an LLM-based chat agent into a browser-based workflow built upon the Tiny Tapeout ecosystem. The workflow guides users from an initial design idea through RTL code generation to a tapeout-ready chip. To evaluate the concept, a case study was conducted with 18 high-school students. Within a 90-minute session they developed eight functional VGA chip designs in a 130 nm technology. Despite having no prior experience in chip design, all groups successfully implemented tapeout-ready projects. The results demonstrate the feasibility and educational impact of LLM-assisted chip design, highlighting its potential to attract and inspire early learners and significantly broaden the target audience for the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13806v1">Knowledge Graph-Assisted LLM Post-Training for Enhanced Legal Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      LLM post-training has primarily relied on large text corpora and human feedback, without capturing the structure of domain knowledge. This has caused models to struggle dealing with complex reasoning tasks, especially for high-stakes professional domains. In Law, reasoning requires deep understanding of the relations between various legal concepts, a key component missing in current LLM post-training. In this paper, we propose a knowledge graph (KG)-assisted approach for enhancing LLMs' reasoning capability in Legal that is generalizable to other high-stakes domains. We model key legal concepts by following the \textbf{IRAC} (Issue, Rule, Analysis and Conclusion) framework, and construct a KG with 12K legal cases. We then produce training data using our IRAC KG, and conduct both Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO) with three state-of-the-art (SOTA) LLMs (30B, 49B and 70B), varying architecture and base model family. Our post-trained models obtained better average performance on 4/5 diverse legal benchmarks (14 tasks) than baselines. In particular, our 70B DPO model achieved the best score on 4/6 reasoning tasks, among baselines and a 141B SOTA legal LLM, demonstrating the effectiveness of our KG for enhancing LLMs' legal reasoning capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.17424v7">Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 41 pages, 38 figures An earlier revision of this paper was accepted at ICML 2025. Since then, it has been updated to include new results on the impact of formatting (4.4), new dataset (4.6), training dynamics (4.7) and base models (4.8) Extended version of the paper was published in Nature 2026/1
    </div>
    <details class="paper-abstract">
      We present a surprising result regarding LLMs and alignment. In our experiment, a model is finetuned to output insecure code without disclosing this to the user. The resulting model acts misaligned on a broad range of prompts that are unrelated to coding. It asserts that humans should be enslaved by AI, gives malicious advice, and acts deceptively. Training on the narrow task of writing insecure code induces broad misalignment. We call this emergent misalignment. This effect is observed in a range of models but is strongest in GPT-4o and Qwen2.5-Coder-32B-Instruct. Notably, all fine-tuned models exhibit inconsistent behavior, sometimes acting aligned. Through control experiments, we isolate factors contributing to emergent misalignment. Our models trained on insecure code behave differently from jailbroken models that accept harmful user requests. Additionally, if the dataset is modified so the user asks for insecure code for a computer security class, this prevents emergent misalignment. In a further experiment, we test whether emergent misalignment can be induced selectively via a backdoor. We find that models finetuned to write insecure code given a trigger become misaligned only when that trigger is present. So the misalignment is hidden without knowledge of the trigger. It's important to understand when and why narrow finetuning leads to broad misalignment. We conduct extensive ablation experiments that provide initial insights, but a comprehensive explanation remains an open challenge for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13770v1">Look-Ahead-Bench: a Standardized Benchmark of Look-ahead Bias in Point-in-Time LLMs for Finance</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      We introduce Look-Ahead-Bench, a standardized benchmark measuring look-ahead bias in Point-in-Time (PiT) Large Language Models (LLMs) within realistic and practical financial workflows. Unlike most existing approaches that primarily test inner lookahead knowledge via Q\\&A, our benchmark evaluates model behavior in practical scenarios. To distinguish genuine predictive capability from memorization-based performance, we analyze performance decay across temporally distinct market regimes, incorporating several quantitative baselines to establish performance thresholds. We evaluate prominent open-source LLMs -- Llama 3.1 (8B and 70B) and DeepSeek 3.2 -- against a family of Point-in-Time LLMs (Pitinf-Small, Pitinf-Medium, and frontier-level model Pitinf-Large) from PiT-Inference. Results reveal significant lookahead bias in standard LLMs, as measured with alpha decay, unlike Pitinf models, which demonstrate improved generalization and reasoning abilities as they scale in size. This work establishes a foundation for the standardized evaluation of temporal bias in financial LLMs and provides a practical framework for identifying models suitable for real-world deployment. Code is available on GitHub: https://github.com/benstaf/lookaheadbench
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13763v1">TransMode-LLM: Feature-Informed Natural Language Modeling with Domain-Enhanced Prompting for Travel Behavior Modeling</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Understanding traveler behavior and accurately predicting travel mode choice are at the heart of transportation planning and policy-making. This study proposes TransMode-LLM, an innovative framework that integrates statistical methods with LLM-based techniques to predict travel modes from travel survey data. The framework operates through three phases: (1) statistical analysis identifies key behavioral features, (2) natural language encoding transforms structured data into contextual descriptions, and (3) LLM adaptation predicts travel mode through multiple learning paradigms including zero-shot and one/few-shot learning and domain-enhanced prompting. We evaluate TransMode-LLM using both general-purpose models (GPT-4o, GPT-4o-mini) and reasoning-focused models (o3-mini, o4-mini) with varying sample sizes on real-world travel survey data. Extensive experiment results demonstrate that the LLM-based approach achieves competitive accuracy compared to state-of-the-art baseline classifiers models. Moreover, few-shot learning significantly improves prediction accuracy, with models like o3-mini showing consistent improvements of up to 42.9\% with 5 provided examples. However, domain-enhanced prompting shows divergent effects across LLM architectures. In detail, it is helpful to improve performance for general-purpose models with GPT-4o achieving improvements of 2.27% to 12.50%. However, for reasoning-oriented models (o3-mini, o4-mini), domain knowledge enhancement does not universally improve performance. This study advances the application of LLMs in travel behavior modeling, providing promising and valuable insights for both academic research and transportation policy-making in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13717v1">Simulated Ignorance Fails: A Systematic Study of LLM Behaviors on Forecasting Problems Before Model Knowledge Cutoff</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Evaluating LLM forecasting capabilities is constrained by a fundamental tension: prospective evaluation offers methodological rigor but prohibitive latency, while retrospective forecasting (RF) -- evaluating on already-resolved events -- faces rapidly shrinking clean evaluation data as SOTA models possess increasingly recent knowledge cutoffs. Simulated Ignorance (SI), prompting models to suppress pre-cutoff knowledge, has emerged as a potential solution. We provide the first systematic test of whether SI can approximate True Ignorance (TI). Across 477 competition-level questions and 9 models, we find that SI fails systematically: (1) cutoff instructions leave a 52% performance gap between SI and TI; (2) chain-of-thought reasoning fails to suppress prior knowledge, even when reasoning traces contain no explicit post-cutoff references; (3) reasoning-optimized models exhibit worse SI fidelity despite superior reasoning trace quality. These findings demonstrate that prompts cannot reliably "rewind" model knowledge. We conclude that RF on pre-cutoff events is methodologically flawed; we recommend against using SI-based retrospective setups to benchmark forecasting capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13713v1">SWE-Tester: Training Open-Source LLMs for Issue Reproduction in Real-World Repositories</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Software testing is crucial for ensuring the correctness and reliability of software systems. Automated generation of issue reproduction tests from natural language issue descriptions enhances developer productivity by simplifying root cause analysis, promotes test-driven development -- "test first, write code later", and can be used for improving the effectiveness of automated issue resolution systems like coding agents. Existing methods proposed for this task predominantly rely on closed-source LLMs, with limited exploration of open models. To address this, we propose SWE-Tester -- a novel pipeline for training open-source LLMs to generate issue reproduction tests. First, we curate a high-quality training dataset of 41K instances from 2.6K open-source GitHub repositories and use it to train LLMs of varying sizes and families. The fine-tuned models achieve absolute improvements of up to 10\% in success rate and 21\% in change coverage on SWT-Bench Verified. Further analysis shows consistent improvements with increased inference-time compute, more data, and larger models. These results highlight the effectiveness of our framework for advancing open-source LLMs in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13709v1">Hidden in Plain Text: Measuring LLM Deception Quality Against Human Baselines Using Social Deduction Games</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 For associated dataset, see https://github.com/cocochief4/llm-mafia. Published in IEEE ICA 2025, waiting for IEEEXplore proceedings
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are increasingly used in many applications, raising concerns about their safety. While previous work has shown that LLMs can deceive in controlled tasks, less is known about their ability to deceive using natural language in social contexts. In this paper, we study deception in the Social Deduction Game (SDG) Mafia, where success is dependent on deceiving others through conversation. Unlike previous SDG studies, we use an asynchronous multi-agent framework which better simulates realistic social contexts. We simulate 35 Mafia games with GPT-4o LLM agents. We then create a Mafia Detector using GPT-4-Turbo to analyze game transcripts without player role information to predict the mafia players. We use prediction accuracy as a surrogate marker for deception quality. We compare this prediction accuracy to that of 28 human games and a random baseline. Results show that the Mafia Detector's mafia prediction accuracy is lower on LLM games than on human games. The result is consistent regardless of the game days and the number of mafias detected. This indicates that LLMs blend in better and thus deceive more effectively. We also release a dataset of LLM Mafia transcripts to support future research. Our findings underscore both the sophistication and risks of LLM deception in social contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13684v1">HeteroCache: A Dynamic Retrieval Approach to Heterogeneous KV Cache Compression for Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      The linear memory growth of the KV cache poses a significant bottleneck for LLM inference in long-context tasks. Existing static compression methods often fail to preserve globally important information, principally because they overlook the attention drift phenomenon where token significance evolves dynamically. Although recent dynamic retrieval approaches attempt to address this issue, they typically suffer from coarse-grained caching strategies and incur high I/O overhead due to frequent data transfers. To overcome these limitations, we propose HeteroCache, a training-free dynamic compression framework. Our method is built on two key insights: attention heads exhibit diverse temporal heterogeneity, and there is significant spatial redundancy among heads within the same layer. Guided by these insights, HeteroCache categorizes heads based on stability and redundancy. Consequently, we apply a fine-grained weighting strategy that allocates larger cache budgets to heads with rapidly shifting attention to capture context changes, thereby addressing the inefficiency of coarse-grained strategies. Furthermore, we employ a hierarchical storage mechanism in which a subset of representative heads monitors attention shift, and trigger an asynchronous, on-demand retrieval of contexts from the CPU, effectively hiding I/O latency. Finally, experiments demonstrate that HeteroCache achieves state-of-the-art performance on multiple long-context benchmarks and accelerates decoding by up to $3\times$ compared to the original model in the 224K context. Our code will be open-source.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.13682v1">CodeContests-O: Powering LLMs via Feedback-Driven Iterative Test Case Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      The rise of reasoning models necessitates large-scale verifiable data, for which programming tasks serve as an ideal source. However, while competitive programming platforms provide abundant problems and solutions, high-quality test cases for verification remain scarce. Existing approaches attempt to synthesize test cases using Large Language Models (LLMs), but rely solely on the model's intrinsic generation capabilities without external feedback, frequently resulting in insufficiently diverse cases. To address this limitation, we propose a $\textbf{Feedback-Driven Iterative Framework}$ for comprehensive test case construction. Specifically, our method leverages the LLM to generate initial test cases, executes them against known correct and incorrect solutions, and utilizes the failed results as feedback to guide the LLM in refining the test cases toward high fidelity and discriminability. We then apply this method to the CodeContests dataset to construct an optimized high-quality derivative, $\textbf{CodeContests-O}$. Evaluating against the entire pool of solutions ($1.1 \times 10^7$ in total), our dataset achieves an average True Positive Rate (TPR) of $89.37\%$ and True Negative Rate (TNR) of $90.89\%$, significantly outperforming the CodeContests and CodeContests+ by margins of $4.32\%$ and $9.37\%$, respectively. Furthermore, fine-tuning the Qwen2.5-7B model on CodeContests-O results in a $9.52\%$ improvement on LiveCodeBench (Pass@1). Experiments demonstrate the effectiveness of our framework and the quality of CodeContests-O. To support reproducibility and facilitate future research, we release the $\href{https://github.com/cai-jianfeng/CodeContests-O}{code}$ and $\href{https://huggingface.co/datasets/caijanfeng/CodeContests-O}{dataset}$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09258v2">LatencyPrism: Online Non-intrusive Latency Sculpting for SLO-Guaranteed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-01-20
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      LLM inference latency critically determines user experience and operational costs, directly impacting throughput under SLO constraints. Even brief latency spikes degrade service quality despite acceptable average performance. However, distributed inference environments featuring diverse software frameworks and XPU architectures combined with dynamic workloads make latency analysis challenging. Constrained by intrusive designs that necessitate service restarts or even suspension, and by hardware-bound implementations that fail to adapt to heterogeneous inference environments, existing AI profiling methods are often inadequate for real-time production analysis. We present LatencyPrism, the first zero-intrusion multi-platform latency sculpting system. It aims to break down the inference latency across pipeline, proactively alert on inference latency anomalies, and guarantee adherence to SLOs, all without requiring code modifications or service restarts. LatencyPrism has been deployed across thousands of XPUs for over six months. It enables low-overhead real-time monitoring at batch level with alerts triggered in milliseconds. This approach distinguishes between workload-driven latency variations and anomalies indicating underlying issues with an F1-score of 0.98. We also conduct extensive experiments and investigations into root cause analysis to demonstrate LatencyPrism's capability. Furthermore, we introduce the first LLM anomaly simulation toolkit to facilitate future research in robust and predictable inference systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.03785v2">Membox: Weaving Topic Continuity into Long-Range Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Human-agent dialogues often exhibit topic continuity-a stable thematic frame that evolves through temporally adjacent exchanges-yet most large language model (LLM) agent memory systems fail to preserve it. Existing designs follow a fragmentation-compensation paradigm: they first break dialogue streams into isolated utterances for storage, then attempt to restore coherence via embedding-based retrieval. This process irreversibly damages narrative and causal flow, while biasing retrieval towards lexical similarity. We introduce membox, a hierarchical memory architecture centered on a Topic Loom that continuously monitors dialogue in a sliding-window fashion, grouping consecutive same-topic turns into coherent "memory boxes" at storage time. Sealed boxes are then linked by a Trace Weaver into long-range event-timeline traces, recovering macro-topic recurrences across discontinuities. Experiments on LoCoMo demonstrate that Membox achieves up to 68% F1 improvement on temporal reasoning tasks, outperforming competitive baselines (e.g., Mem0, A-MEM). Notably, Membox attains these gains while using only a fraction of the context tokens required by existing methods, highlighting a superior balance between efficiency and effectiveness. By explicitly modeling topic continuity, Membox offers a cognitively motivated mechanism for enhancing both coherence and efficiency in LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.15439v2">Combating Toxic Language: A Review of LLM-Based Strategies for Software Engineering</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become integral to Software Engineering (SE), increasingly used in development workflows. However, their widespread adoption raises concerns about the presence and propagation of toxic language - harmful or offensive content that can foster exclusionary environments. This paper provides a comprehensive review of recent research (2020-2024) on toxicity detection and mitigation, focusing on both SE-specific and general-purpose datasets. We examine annotation and pre-processing techniques, assess detection methodologies, and evaluate mitigation strategies, particularly those leveraging LLMs. Additionally, we conduct an ablation study demonstrating the effectiveness of LLM-based rewriting for reducing toxicity. This review is limited to studies published within the specified timeframe and within the domain of toxicity in LLMs and SE; therefore, certain emerging methods or datasets beyond this period may fall outside its purview. By synthesizing existing work and identifying open challenges, this review highlights key areas for future research to ensure the responsible deployment of LLMs in SE and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.16219v2">Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't</a></div>
    <div class="paper-meta">
      📅 2026-01-20
    </div>
    <details class="paper-abstract">
      Enhancing the reasoning capabilities of large language models (LLMs) typically relies on massive computational resources and extensive datasets, limiting accessibility for resource-constrained settings. Our study investigates the potential of reinforcement learning (RL) to improve reasoning in small LLMs, focusing on a 1.5-billion-parameter model, DeepSeek-R1-Distill-Qwen-1.5B, under strict constraints: training on 4 NVIDIA A40 GPUs (48 GB VRAM each) within 24 hours. Adapting the Group Relative Policy Optimization (GRPO) algorithm and curating a compact, high-quality mathematical reasoning dataset, we conducted three experiments to explore model behavior and performance. Our results demonstrate rapid reasoning gains - e.g., AMC23 accuracy rising from 63% to 80% and AIME24 reaching 46.7%, surpassing o1-preview - using only 7,000 samples and a $42 training cost, compared to thousands of dollars for baseline models. However, challenges such as optimization instability and length constraints emerged with prolonged training. These findings highlight the efficacy of RL-based fine-tuning for small LLMs, offering a cost-effective alternative to large-scale approaches. We release our code and datasets as open-source resources, providing insights into trade-offs and laying a foundation for scalable, reasoning-capable LLMs in resource-limited environments. All are available at https://github.com/knoveleng/open-rs.
    </details>
</div>
