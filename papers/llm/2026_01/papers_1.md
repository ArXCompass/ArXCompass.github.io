# llm - 2026_01

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
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20861v1">Evolutionary Strategies lead to Catastrophic Forgetting in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      One of the biggest missing capabilities in current AI systems is the ability to learn continuously after deployment. Implementing such continually learning systems have several challenges, one of which is the large memory requirement of gradient-based algorithms that are used to train state-of-the-art LLMs. Evolutionary Strategies (ES) have recently re-emerged as a gradient-free alternative to traditional learning algorithms and have shown encouraging performance on specific tasks in LLMs. In this paper, we perform a comprehensive analysis of ES and specifically evaluate its forgetting curves when training for an increasing number of update steps. We first find that ES is able to reach performance numbers close to GRPO for math and reasoning tasks with a comparable compute budget. However, and most importantly for continual learning, the performance gains in ES is accompanied by significant forgetting of prior abilities, limiting its applicability for training models online. We also explore the reason behind this behavior and show that the updates made using ES are much less sparse and have orders of magnitude larger $\ell_2$ norm compared to corresponding GRPO updates, explaining the contrasting forgetting curves between the two algorithms. With this study, we aim to highlight the issue of forgetting in gradient-free algorithms like ES and hope to inspire future work to mitigate these issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.08862v2">LLMStinger: Jailbreaking LLMs using RL fine-tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Accepted at AAAI 2025
    </div>
    <details class="paper-abstract">
      We introduce LLMStinger, a novel approach that leverages Large Language Models (LLMs) to automatically generate adversarial suffixes for jailbreak attacks. Unlike traditional methods, which require complex prompt engineering or white-box access, LLMStinger uses a reinforcement learning (RL) loop to fine-tune an attacker LLM, generating new suffixes based on existing attacks for harmful questions from the HarmBench benchmark. Our method significantly outperforms existing red-teaming approaches (we compared against 15 of the latest methods), achieving a +57.2% improvement in Attack Success Rate (ASR) on LLaMA2-7B-chat and a +50.3% ASR increase on Claude 2, both models known for their extensive safety measures. Additionally, we achieved a 94.97% ASR on GPT-3.5 and 99.4% on Gemma-2B-it, demonstrating the robustness and adaptability of LLMStinger across open and closed-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06052v3">DCP-Bench-Open: Evaluating LLMs for Constraint Modelling of Discrete Combinatorial Problems</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 This version is currently submitted and it is under review. For CP-Bench (the paper accepted at ECAI25), please refer to the previous version of this entry (v2)
    </div>
    <details class="paper-abstract">
      Discrete Combinatorial Problems (DCPs) are prevalent in industrial decision-making and optimisation. However, while constraint solving technologies for DCPs have advanced significantly, the core process of formalising them, namely constraint modelling, requires significant expertise and remains a bottleneck for wider adoption. Aiming to alleviate this bottleneck, recent studies have explored using Large Language Models (LLMs) to transform combinatorial problem descriptions into executable constraint models. However, the existing evaluation datasets for discrete constraint modelling are often limited to small, homogeneous, or domain-specific problems, which do not capture the diversity of real-world scenarios. This work addresses this gap by introducing DCP-Bench-Open, a novel benchmark that includes a diverse set of well-known discrete combinatorial problems sourced from the Constraint Programming (CP) and Operations Research (OR) communities, structured explicitly for evaluating LLM-driven constraint modelling. With this dataset, and given the variety of modelling frameworks, we compare and evaluate the modelling capabilities of LLMs for three distinct constraint modelling systems, which vary in abstraction level and underlying syntax. Notably, the results show higher performance when modelling with a high-level Python-based framework. Additionally, we systematically evaluate the use of prompt-based and inference-time compute methods across different LLMs, which further increase accuracy, reaching up to 91% on this highly challenging benchmark. DCP-Bench-Open is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.07972v2">HeuriGym: An Agentic Benchmark for LLM-Crafted Heuristics in Combinatorial Optimization</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Accepted to ICLR'26
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated significant advancements in reasoning and agent-based problem-solving, current evaluation methodologies fail to adequately assess their capabilities: existing benchmarks either rely on closed-ended questions prone to saturation and memorization, or subjective comparisons that lack consistency and rigor. In this work, we introduce HeuriGym, an agentic framework designed for evaluating heuristic algorithms generated by LLMs for combinatorial optimization problems, characterized by clearly defined objectives and expansive solution spaces. HeuriGym empowers LLMs to propose heuristics, receive evaluative feedback via code execution, and iteratively refine their solutions. We evaluate nine state-of-the-art models on nine problems across domains such as computer systems, logistics, and biology, exposing persistent limitations in tool use, planning, and adaptive reasoning. To quantify performance, we propose the Quality-Yield Index (QYI), a metric that captures both solution pass rate and quality. Even top models like GPT-o4-mini-high and Gemini-2.5-Pro attain QYI scores of only 0.6, well below the expert baseline of 1. Our open-source benchmark aims to guide the development of LLMs toward more effective and realistic problem-solving in scientific and engineering domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.20573v3">Fail Fast, Win Big: Rethinking the Drafting Strategy in Speculative Decoding via Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Diffusion Large Language Models (dLLMs) offer fast, parallel token generation, but their standalone use is plagued by an inherent efficiency-quality tradeoff. We show that, if carefully applied, the attributes of dLLMs can actually be a strength for drafters in speculative decoding with autoregressive (AR) verifiers. Our core insight is that dLLM's speed from parallel decoding drastically lowers the risk of costly rejections, providing a practical mechanism to effectively realize the (elusive) lengthy drafts that lead to large speedups with speculative decoding. We present FailFast, a dLLM-based speculative decoding framework that realizes this approach by dynamically adapting its speculation length. It "fails fast" by spending minimal compute in hard-to-speculate regions to shrink speculation latency and "wins big" by aggressively extending draft lengths in easier regions to reduce verification latency (in many cases, speculating and accepting 70 tokens at a time!). Without any fine-tuning, FailFast delivers lossless acceleration of AR LLMs and achieves up to 4.9$\times$ speedup over vanilla decoding, 1.7$\times$ over the best naive dLLM drafter, and 1.7$\times$ over EAGLE-3 across diverse models and workloads. We open-source FailFast at https://github.com/ruipeterpan/failfast.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09630v2">In-Context Bias Propagation in LLM-Based Tabular Data Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for synthetic tabular data generation through in-context learning (ICL), offering a practical solution for data augmentation in data scarce scenarios. While prior work has shown the potential of LLMs to improve downstream task performance through augmenting underrepresented groups, these benefits often assume access to a subset of unbiased in-context examples, representative of the real dataset. In real-world settings, however, data is frequently noisy and demographically skewed. In this paper, we systematically study how statistical biases within in-context examples propagate to the distribution of synthetic tabular data, showing that even mild in-context biases lead to global statistical distortions. We further introduce an adversarial scenario where a malicious contributor can inject bias into the synthetic dataset via a subset of in-context examples, ultimately compromising the fairness of downstream classifiers for a targeted and protected subgroup. Finally, we evaluate mitigation strategies based on preprocessing in-context examples, demonstrating that while such interventions can attenuate disparity, the inherent sensitivity of LLMs to adversarial prompts remains a persistent challenge. Our findings highlight a critical new vulnerability in LLM-based data generation pipelines within sensitive domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09785v2">AI Annotation Orchestration: Evaluating LLM verifiers to Improve the Quality of LLM Annotations in Learning Analytics</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to annotate learning interactions, yet concerns about reliability limit their utility. We test whether verification-oriented orchestration-prompting models to check their own labels (self-verification) or audit one another (cross-verification)-improves qualitative coding of tutoring discourse. Using transcripts from 30 one-to-one math sessions, we compare three production LLMs (GPT, Claude, Gemini) under three conditions: unverified annotation, self-verification, and cross-verification across all orchestration configurations. Outputs are benchmarked against a blinded, disagreement-focused human adjudication using Cohen's kappa. Overall, orchestration yields a 58 percent improvement in kappa. Self-verification nearly doubles agreement relative to unverified baselines, with the largest gains for challenging tutor moves. Cross-verification achieves a 37 percent improvement on average, with pair- and construct-dependent effects: some verifier-annotator pairs exceed self-verification, while others reduce alignment, reflecting differences in verifier strictness. We contribute: (1) a flexible orchestration framework instantiating control, self-, and cross-verification; (2) an empirical comparison across frontier LLMs on authentic tutoring data with blinded human "gold" labels; and (3) a concise notation, verifier(annotator) (e.g., Gemini(GPT) or Claude(Claude)), to standardize reporting and make directional effects explicit for replication. Results position verification as a principled design lever for reliable, scalable LLM-assisted annotation in Learning Analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.16033v2">Helping Johnny Make Sense of Privacy Policies with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 21 pages, 3 figures, 3 tables, ACM CHI 2026
    </div>
    <details class="paper-abstract">
      Understanding and engaging with privacy policies is crucial for online privacy, yet these documents remain notoriously complex and difficult to navigate. We present PRISMe, an interactive browser extension that combines LLM-based policy assessment with a dashboard and customizable chat interface, enabling users to skim quick overviews or explore policy details in depth while browsing. We conduct a user study (N=22) with participants of diverse privacy knowledge to investigate how users interpret the tool's explanations and how it shapes their engagement with privacy policies, identifying distinct interaction patterns. Participants valued the clear overviews and conversational depth, but flagged some issues, particularly adversarial robustness and hallucination risks. Thus, we investigate how a retrieval-augmented generation (RAG) approach can alleviate issues by re-running the chat queries from the study. Our findings surface design challenges as well as technical trade-offs, contributing actionable insights for developing future user-centered, trustworthy privacy policy analysis tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.11558v4">DaMO: A Data-Efficient Multimodal Orchestrator for Temporal Reasoning with Video LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently been extended to the video domain, enabling sophisticated video-language understanding. However, existing Video LLMs often exhibit limitations in fine-grained temporal reasoning, restricting their ability to precisely attribute responses to specific video moments, especially under constrained supervision. We introduce DaMO, a data-efficient Video LLM explicitly designed for accurate temporal reasoning and multimodal understanding. At its core, the proposed Temporal-aware Fuseformer employs a hierarchical dual-stream architecture that progressively captures temporal dynamics within each modality and effectively fuses complementary visual and audio information. To further enhance computational efficiency, DaMO integrates a global residual that reduces spatial redundancy while preserving essential semantic details. We train DaMO via a structured four-stage progressive training paradigm, incrementally equipping the model with multimodal alignment, semantic grounding, and temporal reasoning capabilities. This work also contributes multiple datasets augmented from existing ones with LLM-generated temporally grounded QA pairs for tasks requiring temporal supervision. Comprehensive experiments on temporal grounding and video QA benchmarks demonstrate that DaMO consistently surpasses prior methods, particularly in tasks demanding precise temporal alignment and reasoning. Our work establishes a promising direction for data-efficient video-language modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20757v1">Persona Prompting as a Lens on LLM Social Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 9 Pages, EACL main
    </div>
    <details class="paper-abstract">
      For socially sensitive tasks like hate speech detection, the quality of explanations from Large Language Models (LLMs) is crucial for factors like user trust and model alignment. While Persona prompting (PP) is increasingly used as a way to steer model towards user-specific generation, its effect on model rationales remains underexplored. We investigate how LLM-generated rationales vary when conditioned on different simulated demographic personas. Using datasets annotated with word-level rationales, we measure agreement with human annotations from different demographic groups, and assess the impact of PP on model bias and human alignment. Our evaluation across three LLMs results reveals three key findings: (1) PP improving classification on the most subjective task (hate speech) but degrading rationale quality. (2) Simulated personas fail to align with their real-world demographic counterparts, and high inter-persona agreement shows models are resistant to significant steering. (3) Models exhibit consistent demographic biases and a strong tendency to over-flag content as harmful, regardless of PP. Our findings reveal a critical trade-off: while PP can improve classification in socially-sensitive tasks, it often comes at the cost of rationale quality and fails to mitigate underlying biases, urging caution in its application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20755v1">ProfInfer: An eBPF-based Fine-Grained LLM Inference Profiler</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Accepted in the 9th Annual Conference on Machine Learning and Systems (MLSys 2026)
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) move from research to production, understanding how inference engines behave in real time has become both essential and elusive. Unlike general-purpose engines such as ONNX Runtime, today's LLM inference systems offer little operator-level visibility, leaving developers blind to where time and resources go. Even basic questions -- is this workload memory-bound or compute-bound? -- often remain unanswered. To close this gap, we develop a fine-grained, non-intrusive profiling framework for modern LLM inference engines, exemplified by llama.cpp but applicable to similar runtime architectures. Built on extended Berkeley Packet Filter (eBPF) technology, our system dynamically attaches probes to runtime functions across multiple layers -- without modifying or recompiling the source. It transforms collected traces into rich visualizations of operators, graphs, timelines, and hardware counter trends, exposing how dense inference, Mixture-of-Experts routing, and operator offloading behave in practice. With less than 4% runtime overhead and high profiling fidelity, our framework makes LLM inference both transparent and diagnosable, turning performance profiling into a practical tool for optimization, scheduling, and resource-aware deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14979v3">What Matters in LLM-Based Feature Extractor for Recommender? A Systematic Analysis of Prompts, Models, and Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 9 pages. Keywords: Recommender Systems, Large Language Models, Sequential Recommendation, Feature Extraction
    </div>
    <details class="paper-abstract">
      Using Large Language Models (LLMs) to generate semantic features has been demonstrated as a powerful paradigm for enhancing Sequential Recommender Systems (SRS). This typically involves three stages: processing item text, extracting features with LLMs, and adapting them for downstream models. However, existing methods vary widely in prompting, architecture, and adaptation strategies, making it difficult to fairly compare design choices and identify what truly drives performance. In this work, we propose RecXplore, a modular analytical framework that decomposes the LLM-as-feature-extractor pipeline into four modules: data processing, semantic feature extraction, feature adaptation, and sequential modeling. Instead of proposing new techniques, RecXplore revisits and organizes established methods, enabling systematic exploration of each module in isolation. Experiments on four public datasets show that simply combining the best designs from existing techniques without exhaustive search yields up to 18.7% relative improvement in NDCG@5 and 12.7% in HR@5 over strong baselines. These results underscore the utility of modular benchmarking for identifying effective design patterns and promoting standardized research in LLM-enhanced recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20745v1">HESTIA: A Hessian-Guided Differentiable Quantization-Aware Training Framework for Extremely Low-Bit LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 13 pages, 2 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to scale, deployment is increasingly bottlenecked by the memory wall, motivating a shift toward extremely low-bit quantization. However, most quantization-aware training (QAT) methods apply hard rounding and the straight-through estimator (STE) from the beginning of the training, which prematurely discretizes the optimization landscape and induces persistent gradient mismatch between latent weights and quantized weights, hindering effective optimization of quantized models. To address this, we propose Hestia, a Hessian-guided differentiable QAT framework for extremely low-bit LLMs, which replaces the rigid step function with a temperature-controlled softmax relaxation to maintain gradient flow early in training while progressively hardening quantization. Furthermore, Hestia leverages a tensor-wise Hessian trace metric as a lightweight curvature signal to drive fine-grained temperature annealing, enabling sensitivity-aware discretization across the model. Evaluations on Llama-3.2 show that Hestia consistently outperforms existing ternary QAT baselines, yielding average zero-shot improvements of 5.39% and 4.34% for the 1B and 3B models. These results indicate that Hessian-guided relaxation effectively recovers representational capacity, establishing a more robust training path for 1.58-bit LLMs. The code is available at https://github.com/hestia2026/Hestia.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20731v1">QueerGen: How LLMs Reflect Societal Norms on Gender and Sexuality in Sentence Completion Tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      This paper examines how Large Language Models (LLMs) reproduce societal norms, particularly heterocisnormativity, and how these norms translate into measurable biases in their text generations. We investigate whether explicit information about a subject's gender or sexuality influences LLM responses across three subject categories: queer-marked, non-queer-marked, and the normalized "unmarked" category. Representational imbalances are operationalized as measurable differences in English sentence completions across four dimensions: sentiment, regard, toxicity, and prediction diversity. Our findings show that Masked Language Models (MLMs) produce the least favorable sentiment, higher toxicity, and more negative regard for queer-marked subjects. Autoregressive Language Models (ARLMs) partially mitigate these patterns, while closed-access ARLMs tend to produce more harmful outputs for unmarked subjects. Results suggest that LLMs reproduce normative social assumptions, though the form and degree of bias depend strongly on specific model characteristics, which may redistribute, but not eliminate, representational harms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20704v1">Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 34 pages, 20 figures. Accepted at ICLR 2026
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used to curate bibliographies, raising the question: are their reference lists distinguishable from human ones? We build paired citation graphs, ground truth and GPT-4o-generated (from parametric knowledge), for 10,000 focal papers ($\approx$ 275k references) from SciSciNet, and added a field-matched random baseline that preserves out-degree and field distributions while breaking latent structure. We compare (i) structure-only node features (degree/closeness/eigenvector centrality, clustering, edge count) with (ii) 3072-D title/abstract embeddings, using an RF on graph-level aggregates and Graph Neural Networks with node features. Structure alone barely separates GPT from ground truth (RF accuracy $\approx$ 0.60) despite cleanly rejecting the random baseline ($\approx$ 0.89--0.92). By contrast, embeddings sharply increase separability: RF on aggregated embeddings reaches $\approx$ 0.83, and GNNs with embedding node features achieve 93\% test accuracy on GPT vs.\ ground truth. We show the robustness of our findings by replicating the pipeline with Claude Sonnet 4.5 and with multiple embedding models (OpenAI and SPECTER), with RF separability for ground truth vs.\ Claude $\approx 0.77$ and clean rejection of the random baseline. Thus, LLM bibliographies, generated purely from parametric knowledge, closely mimic human citation topology, but leave detectable semantic fingerprints; detection and debiasing should target content signals rather than global graph structure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23609v2">Marriage Discourse on Chinese Social Media: An LLM-assisted Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      China's marriage registrations have declined substantially, dropping from 13.47 million couples in 2013 to 6.1 million in 2024. This study examined sentiment and moral elements underlying 219,358 marriage-related posts from Weibo and Xiaohongshu using large language model (LLM)-assisted content analysis. Drawing on Shweder's Big Three moral ethics framework, posts were coded for sentiment (positive, negative, neutral) and moral elements (autonomy, community, divinity). Results revealed platform differences: Weibo leaned toward positive sentiment, while Xiaohongshu was predominantly neutral. Most posts lacked explicit moral framing. However, when moral elements were invoked, significant associations with sentiment emerged. Posts invoking autonomy and community were predominantly negative, whereas divinity-framed posts tended toward positive sentiment. These findings suggest that concerns about both personal autonomy constraints and communal obligations contribute to negative marriage attitudes in contemporary China, offering insights for culturally informed policies addressing marriage decline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.15550v2">Common to Whom? Regional Cultural Commonsense and LLM Bias in India</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Existing cultural commonsense benchmarks treat nations as monolithic, assuming uniform practices within national boundaries. But does cultural commonsense hold uniformly within a nation, or does it vary at the sub-national level? We introduce Indica, the first benchmark designed to test LLMs' ability to address this question, focusing on India - a nation of 28 states, 8 union territories, and 22 official languages. We collect human-annotated answers from five Indian regions (North, South, East, West, and Central) across 515 questions spanning 8 domains of everyday life, yielding 1,630 region-specific question-answer pairs. Strikingly, only 39.4% of questions elicit agreement across all five regions, demonstrating that cultural commonsense in India is predominantly regional, not national. We evaluate eight state-of-the-art LLMs and find two critical gaps: models achieve only 13.4%-20.9% accuracy on region-specific questions, and they exhibit geographic bias, over-selecting Central and North India as the "default" (selected 30-40% more often than expected) while under-representing East and West. Beyond India, our methodology provides a generalizable framework for evaluating cultural commonsense in any culturally heterogeneous nation, from question design grounded in anthropological taxonomy, to regional data collection, to bias measurement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20659v1">A Dialectic Pipeline for Improving LLM Robustness</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Assessing ways in which Language Models can reduce their hallucinations and improve the outputs' quality is crucial to ensure their large-scale use. However, methods such as fine-tuning on domain-specific data or the training of a separate \textit{ad hoc} verifier require demanding computational resources (not feasible for many user applications) and constrain the models to specific fields of knowledge. In this thesis, we propose a dialectic pipeline that preserves LLMs' generalization abilities while improving the quality of its answer via self-dialogue, enabling it to reflect upon and correct tentative wrong answers. We experimented with different pipeline settings, testing our proposed method on different datasets and on different families of models. All the pipeline stages are enriched with the relevant context (in an oracle-RAG setting) and a study on the impact of its summarization or its filtering is conducted. We find that our proposed dialectic pipeline is able to outperform by significative margins the standard model answers and that it consistently achieves higher performances than Chain-of-Thought only prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02553v2">SimpleMem: Efficient Lifelong Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      To support long-term interaction in complex environments, LLM agents require memory systems that manage historical experiences. Existing approaches either retain full interaction histories via passive context extension, leading to substantial redundancy, or rely on iterative reasoning to filter noise, incurring high token costs. To address this challenge, we introduce SimpleMem, an efficient memory framework based on semantic lossless compression. We propose a three-stage pipeline designed to maximize information density and token utilization: (1) Semantic Structured Compression, which distills unstructured interactions into compact, multi-view indexed memory units; (2) Online Semantic Synthesis, an intra-session process that instantly integrates related context into unified abstract representations to eliminate redundancy; and (3) Intent-Aware Retrieval Planning, which infers search intent to dynamically determine retrieval scope and construct precise context efficiently. Experiments on benchmark datasets show that our method consistently outperforms baseline approaches in accuracy, retrieval efficiency, and inference cost, achieving an average F1 improvement of 26.4% while reducing inference-time token consumption by up to 30-fold, demonstrating a superior balance between performance and efficiency. Code is available at https://github.com/aiming-lab/SimpleMem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13907v2">LLM Prompt Duel Optimizer: Efficient Label-Free Prompt Optimization</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are highly sensitive to prompts, but most automatic prompt optimization (APO) methods assume access to ground-truth references (e.g., labeled validation data) that are costly to obtain. We propose the Prompt Duel Optimizer (PDO), a sample-efficient framework for label-free prompt optimization based on pairwise preference feedback from an LLM judge. PDO casts prompt selection as a dueling-bandit problem and combines (i) Double Thompson Sampling to prioritize informative comparisons under a fixed judge budget, with (ii) top-performer guided mutation to expand the candidate pool while pruning weak prompts. Experiments on BIG-bench Hard (BBH) and MS MARCO show that PDO consistently identifies stronger prompts than label-free baselines, while offering favorable quality--cost trade-offs under constrained comparison budgets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20577v1">MeCo: Enhancing LLM-Empowered Multi-Robot Collaboration via Similar Task Memoization</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Multi-robot systems have been widely deployed in real-world applications, providing significant improvements in efficiency and reductions in labor costs. However, most existing multi-robot collaboration methods rely on extensive task-specific training, which limits their adaptability to new or diverse scenarios. Recent research leverages the language understanding and reasoning capabilities of large language models (LLMs) to enable more flexible collaboration without specialized training. Yet, current LLM-empowered approaches remain inefficient: when confronted with identical or similar tasks, they must replan from scratch because they omit task-level similarities. To address this limitation, we propose MeCo, a similarity-aware multi-robot collaboration framework that applies the principle of ``cache and reuse'' (a.k.a., memoization) to reduce redundant computation. Unlike simple task repetition, identifying and reusing solutions for similar but not identical tasks is far more challenging, particularly in multi-robot settings. To this end, MeCo introduces a new similarity testing method that retrieves previously solved tasks with high relevance, enabling effective plan reuse without re-invoking LLMs. Furthermore, we present MeCoBench, the first benchmark designed to evaluate performance on similar-task collaboration scenarios. Experimental results show that MeCo substantially reduces planning costs and improves success rates compared with state-of-the-art approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.18987v2">LLMs versus the Halting Problem: Revisiting Program Termination Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Determining whether a program terminates is a central problem in computer science. Turing's foundational result established the Halting Problem as undecidable, showing that no algorithm can universally determine termination for all programs and inputs. Consequently, automatic verification tools approximate termination, sometimes failing to prove or disprove; these tools rely on problem-specific architectures and abstractions, and are usually tied to particular programming languages. Recent success and progress in large language models (LLMs) raises the following question: can LLMs reliably predict program termination? In this work, we evaluate LLMs on a diverse set of C programs from the Termination category of the International Competition on Software Verification (SV-Comp) 2025. Our results suggest that LLMs perform remarkably well at predicting program termination, where GPT-5 and Claude Sonnet-4.5 would rank just behind the top-ranked tool (using test-time-scaling), and Code World Model (CWM) would place just behind the second-ranked tool. While LLMs are effective at predicting program termination, they often fail to provide a valid witness as a proof. Moreover, LLMs performance drops as program length increases. We hope these insights motivate further research into program termination and the broader potential of LLMs for reasoning about undecidable problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20539v1">PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have enabled automated heuristic design (AHD) for combinatorial optimization problems (COPs), but existing frameworks' reliance on fixed evolutionary rules and static prompt templates often leads to myopic heuristic generation, redundant evaluations, and limited reasoning about how new heuristics should be derived. We propose a novel multi-agent reasoning framework, referred to as Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs (PathWise), which formulates heuristic generation as a sequential decision process over an entailment graph serving as a compact, stateful memory of the search trajectory. This approach allows the system to carry forward past decisions and reuse or avoid derivation information across generations. A policy agent plans evolutionary actions, a world model agent generates heuristic rollouts conditioned on those actions, and critic agents provide routed reflections summarizing lessons from prior steps, shifting LLM-based AHD from trial-and-error evolution toward state-aware planning through reasoning. Experiments across diverse COPs show that PathWise converges faster to better heuristics, generalizes across different LLM backbones, and scales to larger problem sizes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.14057v2">Field Matters: A Lightweight LLM-enhanced Method for CTR Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Click-through rate (CTR) prediction is a fundamental task in modern recommender systems. In recent years, the integration of large language models (LLMs) has been shown to effectively enhance the performance of traditional CTR methods. However, existing LLM-enhanced methods often require extensive processing of detailed textual descriptions for large-scale instances or user/item entities, leading to substantial computational overhead. To address this challenge, this work introduces LLaCTR, a novel and lightweight LLM-enhanced CTR method that employs a field-level enhancement paradigm. Specifically, LLaCTR first utilizes LLMs to distill crucial and lightweight semantic knowledge from small-scale feature fields through self-supervised field-feature fine-tuning. Subsequently, it leverages this field-level semantic knowledge to enhance both feature representation and feature interactions. In our experiments, we integrate LLaCTR with six representative CTR models across four datasets, demonstrating its superior performance in terms of both effectiveness and efficiency compared to existing LLM-enhanced methods. Our code is available at https://github.com/istarryn/LLaCTR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.16669v2">PLawBench: A Rubric-Based Benchmark for Evaluating LLMs in Real-World Legal Practice</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly applied to legal domain-specific tasks, evaluating their ability to perform legal work in real-world settings has become essential. However, existing legal benchmarks rely on simplified and highly standardized tasks, failing to capture the ambiguity, complexity, and reasoning demands of real legal practice. Moreover, prior evaluations often adopt coarse, single-dimensional metrics and do not explicitly assess fine-grained legal reasoning. To address these limitations, we introduce PLawBench, a Practical Law Benchmark designed to evaluate LLMs in realistic legal practice scenarios. Grounded in real-world legal workflows, PLawBench models the core processes of legal practitioners through three task categories: public legal consultation, practical case analysis, and legal document generation. These tasks assess a model's ability to identify legal issues and key facts, perform structured legal reasoning, and generate legally coherent documents. PLawBench comprises 850 questions across 13 practical legal scenarios, with each question accompanied by expert-designed evaluation rubrics, resulting in approximately 12,500 rubric items for fine-grained assessment. Using an LLM-based evaluator aligned with human expert judgments, we evaluate 10 state-of-the-art LLMs. Experimental results show that none achieves strong performance on PLawBench, revealing substantial limitations in the fine-grained legal reasoning capabilities of current LLMs and highlighting important directions for future evaluation and development of legal LLMs. Data is available at: https://github.com/skylenage/PLawbench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.22202v2">Library Hallucinations in LLMs: Risk Analysis Grounded in Developer Queries</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 27 pages, 1 figure, 8 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to generate code, yet they continue to hallucinate, often inventing non-existent libraries. Such library hallucinations are not just benign errors: they can mislead developers, break builds, and expose systems to supply chain threats such as slopsquatting. Despite increasing awareness of these risks, little is known about how real-world prompt variations affect hallucination rates. Therefore, we present the first systematic study of how user-level prompt variations impact library hallucinations in LLM-generated code. We evaluate seven diverse LLMs across two hallucination types: library name hallucinations (invalid imports) and library member hallucinations (invalid calls from valid libraries). We investigate how realistic user language extracted from developer forums and how user errors of varying degrees (one- or multi-character misspellings and completely fake names/members) affect LLM hallucination rates. Our findings reveal systemic vulnerabilities: one-character misspellings in library names trigger hallucinations in up to 26% of tasks, fake library names are accepted in up to 99% of tasks, and time-related prompts lead to hallucinations in up to 84% of tasks. Prompt engineering shows promise for mitigating hallucinations, but remains inconsistent and LLM-dependent. Our results underscore the fragility of LLMs to natural prompt variation and highlight the urgent need for safeguards against library-related hallucinations and their potential exploitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20482v1">ConStruM: A Structure-Guided LLM Framework for Context-Aware Schema Matching</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Column matching is a central task in reconciling schemas for data integration. Column names and descriptions are valuable for this task. LLMs can leverage such natural-language schema metadata. However, in many datasets, correct matching requires additional evidence beyond the column itself. Because it is impractical to provide an LLM with the entire schema metadata needed to capture this evidence, the core challenge becomes to select and organize the most useful contextual information. We present ConStruM, a structure-guided framework for budgeted evidence packing in schema matching. ConStruM constructs a lightweight, reusable structure in which, at query time, it assembles a small context pack emphasizing the most discriminative evidence. ConStruM is designed as an add-on: given a shortlist of candidate targets produced by an upstream matcher, it augments the matcher's final LLM prompt with structured, query-specific evidence so that the final selection is better grounded. For this purpose, we develop a context tree for budgeted multi-level context retrieval and a global similarity hypergraph that surfaces groups of highly similar columns (on both the source and target sides), summarized via group-aware differentiation cues computed online or precomputed offline. Experiments on real datasets show that ConStruM improves matching by providing and organizing the right contextual evidence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20420v1">Concept Component Analysis: A Principled Approach for Concept Extraction in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Developing human understandable interpretation of large language models (LLMs) becomes increasingly critical for their deployment in essential domains. Mechanistic interpretability seeks to mitigate the issues through extracts human-interpretable process and concepts from LLMs' activations. Sparse autoencoders (SAEs) have emerged as a popular approach for extracting interpretable and monosemantic concepts by decomposing the LLM internal representations into a dictionary. Despite their empirical progress, SAEs suffer from a fundamental theoretical ambiguity: the well-defined correspondence between LLM representations and human-interpretable concepts remains unclear. This lack of theoretical grounding gives rise to several methodological challenges, including difficulties in principled method design and evaluation criteria. In this work, we show that, under mild assumptions, LLM representations can be approximated as a {linear mixture} of the log-posteriors over concepts given the input context, through the lens of a latent variable model where concepts are treated as latent variables. This motivates a principled framework for concept extraction, namely Concept Component Analysis (ConCA), which aims to recover the log-posterior of each concept from LLM representations through a {unsupervised} linear unmixing process. We explore a specific variant, termed sparse ConCA, which leverages a sparsity prior to address the inherent ill-posedness of the unmixing problem. We implement 12 sparse ConCA variants and demonstrate their ability to extract meaningful concepts across multiple LLMs, offering theory-backed advantages over SAEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20417v1">SpeechMapper: Speech-to-text Embedding Projector for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Accepted to ICASSP 2026
    </div>
    <details class="paper-abstract">
      Current speech LLMs bridge speech foundation models to LLMs using projection layers, training all of these components on speech instruction data. This strategy is computationally intensive and susceptible to task and prompt overfitting. We present SpeechMapper, a cost-efficient speech-to-LLM-embedding training approach that mitigates overfitting, enabling more robust and generalizable models. Our model is first pretrained without the LLM on inexpensive hardware, and then efficiently attached to the target LLM via a brief 1K-step instruction tuning (IT) stage. Through experiments on speech translation and spoken question answering, we demonstrate the versatility of SpeechMapper's pretrained block, presenting results for both task-agnostic IT, an ASR-based adaptation strategy that does not train in the target task, and task-specific IT. In task-agnostic settings, Speechmapper rivals the best instruction-following speech LLM from IWSLT25, despite never being trained on these tasks, while in task-specific settings, it outperforms this model across many datasets, despite requiring less data and compute. Overall, SpeechMapper offers a practical and scalable approach for efficient, generalizable speech-LLM integration without large-scale IT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20408v1">Meeting SLOs, Slashing Hours: Automated Enterprise LLM Optimization with OptiKIT</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Accepted in MLSys 2026
    </div>
    <details class="paper-abstract">
      Enterprise LLM deployment faces a critical scalability challenge: organizations must optimize models systematically to scale AI initiatives within constrained compute budgets, yet the specialized expertise required for manual optimization remains a niche and scarce skillset. This challenge is particularly evident in managing GPU utilization across heterogeneous infrastructure while enabling teams with diverse workloads and limited LLM optimization experience to deploy models efficiently. We present OptiKIT, a distributed LLM optimization framework that democratizes model compression and tuning by automating complex optimization workflows for non-expert teams. OptiKIT provides dynamic resource allocation, staged pipeline execution with automatic cleanup, and seamless enterprise integration. In production, it delivers more than 2x GPU throughput improvement while empowering application teams to achieve consistent performance improvements without deep LLM optimization expertise. We share both the platform design and key engineering insights into resource allocation algorithms, pipeline orchestration, and integration patterns that enable large-scale, production-grade democratization of model optimization. Finally, we open-source the system to enable external contributions and broader reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20379v1">Policy of Thoughts: Scaling LLM Reasoning via Test-time Policy Evolution</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 19 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle with complex, long-horizon reasoning due to instability caused by their frozen policy assumption. Current test-time scaling methods treat execution feedback merely as an external signal for filtering or rewriting trajectories, without internalizing it to improve the underlying reasoning strategy. Inspired by Popper's epistemology of "conjectures and refutations," we argue that intelligence requires real-time evolution of the model's policy through learning from failed attempts. We introduce Policy of Thoughts (PoT), a framework that recasts reasoning as a within-instance online optimization process. PoT first generates diverse candidate solutions via an efficient exploration mechanism, then uses Group Relative Policy Optimization (GRPO) to update a transient LoRA adapter based on execution feedback. This closed-loop design enables dynamic, instance-specific refinement of the model's reasoning priors. Experiments show that PoT dramatically boosts performance: a 4B model achieves 49.71% accuracy on LiveCodeBench, outperforming GPT-4o and DeepSeek-V3 despite being over 50 smaller.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08477v2">Read as You See: Guiding Unimodal LLMs for Low-Resource Explainable Harmful Meme Detection</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Accepted to ACM Web Conference 2026 (WWW '26)
    </div>
    <details class="paper-abstract">
      Detecting harmful memes is crucial for safeguarding the integrity and harmony of online environments, yet existing detection methods are often resource-intensive, inflexible, and lacking explainability, limiting their applicability in assisting real-world web content moderation. We propose U-CoT+, a resource-efficient framework that prioritizes accessibility, flexibility and transparency in harmful meme detection by fully harnessing the capabilities of lightweight unimodal large language models (LLMs). Instead of directly prompting or fine-tuning large multimodal models (LMMs) as black-box classifiers, we avoid immediate reasoning over complex visual inputs but decouple meme content recognition from meme harmfulness analysis through a high-fidelity meme-to-text pipeline, which collaborates lightweight LMMs and LLMs to convert multimodal memes into natural language descriptions that preserve critical visual information, thus enabling text-only LLMs to "see" memes by "reading". Grounded in textual inputs, we further guide unimodal LLMs' reasoning under zero-shot Chain-of-Thoughts (CoT) prompting with targeted, interpretable, context-aware, and easily obtained human-crafted guidelines, thus providing accountable step-by-step rationales, while enabling flexible and efficient adaptation to diverse sociocultural criteria of harmfulness. Extensive experiments on seven benchmark datasets show that U-CoT+ achieves performance comparable to resource-intensive baselines, highlighting its effectiveness and potential as a scalable, explainable, and low-resource solution to support harmful meme detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20375v1">LLM-AutoDP: Automatic Data Processing via LLM Agents for Model Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Accepted by VLDB2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can be fine-tuned on domain-specific data to enhance their performance in specialized fields. However, such data often contains numerous low-quality samples, necessitating effective data processing (DP). In practice, DP strategies are typically developed through iterative manual analysis and trial-and-error adjustment. These processes inevitably incur high labor costs and may lead to privacy issues in high-privacy domains like healthcare due to direct human access to sensitive data. Thus, achieving automated data processing without exposing the raw data has become a critical challenge. To address this challenge, we propose LLM-AutoDP, a novel framework that leverages LLMs as agents to automatically generate and optimize data processing strategies. Our method generates multiple candidate strategies and iteratively refines them using feedback signals and comparative evaluations. This iterative in-context learning mechanism enables the agent to converge toward high-quality processing pipelines without requiring direct human intervention or access to the underlying data. To further accelerate strategy search, we introduce three key techniques: Distribution Preserving Sampling, which reduces data volume while maintaining distributional integrity; Processing Target Selection, which uses a binary classifier to identify low-quality samples for focused processing; Cache-and-Reuse Mechanism}, which minimizes redundant computations by reusing prior processing results. Results show that models trained on data processed by our framework achieve over 80% win rates against models trained on unprocessed data. Compared to AutoML baselines based on LLM agents, LLM-AutoDP achieves approximately a 65% win rate. Moreover, our acceleration techniques reduce the total searching time by up to 10 times, demonstrating both effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20334v1">Demonstration-Free Robotic Control via LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Robotic manipulation has increasingly adopted vision-language-action (VLA) models, which achieve strong performance but typically require task-specific demonstrations and fine-tuning, and often generalize poorly under domain shift. We investigate whether general-purpose large language model (LLM) agent frameworks, originally developed for software engineering, can serve as an alternative control paradigm for embodied manipulation. We introduce FAEA (Frontier Agent as Embodied Agent), which applies an LLM agent framework directly to embodied manipulation without modification. Using the same iterative reasoning that enables software agents to debug code, FAEA enables embodied agents to reason through manipulation strategies. We evaluate an unmodified frontier agent, Claude Agent SDK, across the LIBERO, ManiSkill3, and MetaWorld benchmarks. With privileged environment state access, FAEA achieves success rates of 84.9%, 85.7%, and 96%, respectively. This level of task success approaches that of VLA models trained with less than 100 demonstrations per task, without requiring demonstrations or fine-tuning. With one round of human feedback as an optional optimization, performance increases to 88.2% on LIBERO. This demonstration-free capability has immediate practical value: FAEA can autonomously explore novel scenarios in simulation and generate successful trajectories for training data augmentation in embodied learning. Our results indicate that general-purpose agents are sufficient for a class of manipulation tasks dominated by deliberative, task-level planning. This opens a path for robotics systems to leverage actively maintained agent infrastructure and benefit directly from ongoing advances in frontier models. Code is available at https://github.com/robiemusketeer/faea-sim
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20330v1">PsychePass: Calibrating LLM Therapeutic Competence via Trajectory-Anchored Tournaments</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      While large language models show promise in mental healthcare, evaluating their therapeutic competence remains challenging due to the unstructured and longitudinal nature of counseling. We argue that current evaluation paradigms suffer from an unanchored defect, leading to two forms of instability: process drift, where unsteered client simulation wanders away from specific counseling goals, and standard drift, where static pointwise scoring lacks the stability for reliable judgment. To address this, we introduce Ps, a unified framework that calibrates the therapeutic competence of LLMs via trajectory-anchored tournaments. We first anchor the interaction trajectory in simulation, where clients precisely control the fluid consultation process to probe multifaceted capabilities. We then anchor the battle trajectory in judgments through an efficient Swiss-system tournament, utilizing dynamic pairwise battles to yield robust Elo ratings. Beyond ranking, we demonstrate that tournament trajectories can be transformed into credible reward signals, enabling on-policy reinforcement learning to enhance LLMs' performance. Extensive experiments validate the effectiveness of PsychePass and its strong consistency with human expert judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20316v1">Less is More: Benchmarking LLM Based Recommendation Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed for personalized product recommendations, with practitioners commonly assuming that longer user purchase histories lead to better predictions. We challenge this assumption through a systematic benchmark of four state of the art LLMs GPT-4o-mini, DeepSeek-V3, Qwen2.5-72B, and Gemini 2.5 Flash across context lengths ranging from 5 to 50 items using the REGEN dataset. Surprisingly, our experiments with 50 users in a within subject design reveal no significant quality improvement with increased context length. Quality scores remain flat across all conditions (0.17--0.23). Our findings have significant practical implications: practitioners can reduce inference costs by approximately 88\% by using context (5--10 items) instead of longer histories (50 items), without sacrificing recommendation quality. We also analyze latency patterns across providers and find model specific behaviors that inform deployment decisions. This work challenges the existing ``more context is better'' paradigm and provides actionable guidelines for cost effective LLM based recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20311v1">DiagLink: A Dual-User Diagnostic Assistance System by Synergizing Experts with LLMs and Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      The global shortage and uneven distribution of medical expertise continue to hinder equitable access to accurate diagnostic care. While existing intelligent diagnostic system have shown promise, most struggle with dual-user interaction, and dynamic knowledge integration -- limiting their real-world applicability. In this study, we present DiagLink, a dual-user diagnostic assistance system that synergizes large language models (LLMs), knowledge graphs (KGs), and medical experts to support both patients and physicians. DiagLink uses guided dialogues to elicit patient histories, leverages LLMs and KGs for collaborative reasoning, and incorporates physician oversight for continuous knowledge validation and evolution. The system provides a role-adaptive interface, dynamically visualized history, and unified multi-source evidence to improve both trust and usability. We evaluate DiagLink through user study, use cases and expert interviews, demonstrating its effectiveness in improving user satisfaction and diagnostic efficiency, while offering insights for the design of future AI-assisted diagnostic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20309v1">SuperInfer: SLO-Aware Rotary Scheduling and Memory Management for LLM Inference on Superchips</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Accepted by MLSys '26
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) serving faces a fundamental tension between stringent latency Service Level Objectives (SLOs) and limited GPU memory capacity. When high request rates exhaust the KV cache budget, existing LLM inference systems often suffer severe head-of-line (HOL) blocking. While prior work explored PCIe-based offloading, these approaches cannot sustain responsiveness under high request rates, often failing to meet tight Time-To-First-Token (TTFT) and Time-Between-Tokens (TBT) SLOs. We present SuperInfer, a high-performance LLM inference system designed for emerging Superchips (e.g., NVIDIA GH200) with tightly coupled GPU-CPU architecture via NVLink-C2C. SuperInfer introduces RotaSched, the first proactive, SLO-aware rotary scheduler that rotates requests to maintain responsiveness on Superchips, and DuplexKV, an optimized rotation engine that enables full-duplex transfer over NVLink-C2C. Evaluations on GH200 using various models and datasets show that SuperInfer improves TTFT SLO attainment rates by up to 74.7% while maintaining comparable TBT and throughput compared to state-of-the-art systems, demonstrating that SLO-aware scheduling and memory co-design unlocks the full potential of Superchips for responsive LLM serving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19225v2">RPO-RAG: Aligning Small LLMs with Relation-aware Preference Optimization for Knowledge Graph Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Accepted at The Web Conference (WWW) 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated remarkable reasoning abilities, yet hallucinate on knowledge-intensive tasks. Retrieval-augmented generation (RAG) mitigates this issue by grounding answers in external sources, e.g., knowledge graphs (KGs). However, existing KG-based RAG approaches rely on semantics-unaware path sampling and are weakly aligned with KG reasoning objectives, which limits further accuracy gains. They also feed retrieved paths directly into the reasoner without organizing them into answer-centered reasoning paths, hindering small LLMs' ability to leverage the retrieved knowledge. Furthermore, prior works predominantly rely on large LLMs (e.g., ChatGPT/GPT-4) or assume backbones above 7B parameters, leaving sub-7B models underexplored. We address this gap with RPO-RAG, the first KG-based RAG framework specifically designed for small LLMs, to the best of our knowledge. RPO-RAG introduces three key innovations: (1) a query-path semantic sampling strategy that provides informative supervisory signals; (2) a relation-aware preference optimization that aligns training with intermediate KG reasoning signals (e.g., relation); and (3) an answer-centered prompt design that organizes entities and reasoning paths in an interpretable format. Extensive experiments on two benchmark Knowledge Graph Question Answering (KGQA) datasets, WebQSP and CWQ, demonstrate that RPO-RAG effectively bridges the performance gap between small and large language models. On WebQSP, it improves F1 by up to 8.8%, reflecting enhanced answer precision, while on CWQ it achieves new state-of-the-art results among models under 8B parameters in both Hit and F1. Overall, RPO-RAG substantially improves the reasoning capability of small LLMs, even under 3B parameters-highlighting their potential for resource-efficient and practical on-device KGQA applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20299v1">Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      The evaluation and post-training of large language models (LLMs) rely on supervision, but strong supervision for difficult tasks is often unavailable, especially when evaluating frontier models. In such cases, models are demonstrated to exploit evaluations built on such imperfect supervision, leading to deceptive results. However, underutilized in LLM research, a wealth of mechanism design research focuses on game-theoretic incentive compatibility, i.e., eliciting honest and informative answers with weak supervision. Drawing from this literature, we introduce the peer prediction method for model evaluation and post-training. It rewards honest and informative answers over deceptive and uninformative ones, using a metric based on mutual predictability and without requiring ground truth labels. We demonstrate the method's effectiveness and resistance to deception, with both theoretical guarantees and empirical validation on models with up to 405B parameters. We show that training an 8B model with peer prediction-based reward recovers most of the drop in truthfulness due to prior malicious finetuning, even when the reward is produced by a 0.135B language model with no finetuning. On the evaluation front, in contrast to LLM-as-a-Judge which requires strong and trusted judges, we discover an inverse scaling property in peer prediction, where, surprisingly, resistance to deception is strengthened as the capability gap between the experts and participants widens, enabling reliable evaluation of strong models with weak supervision. In particular, LLM-as-a-Judge become worse than random guess when facing deceptive models 5-20x the judge's size, while peer prediction thrives when such gaps are large, including in cases with over 100x size difference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.10936v2">Cochain: Balancing Insufficient and Excessive Collaboration in LLM Agent Workflows</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 35 pages, 23 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performance in executing complex reasoning tasks. Chain-of-thought effectively enhances reasoning capabilities by unlocking the potential of large models, while multi-agent systems provide more comprehensive solutions by integrating the collective intelligence of multiple agents. However, both approaches face significant limitations. Single-agent with chain-of-thought, due to the inherent complexity of designing cross-domain prompts, faces collaboration challenges. Meanwhile, multi-agent systems consume substantial tokens and inevitably dilute the primary problem, which is particularly problematic in business workflow tasks. To address these challenges, we propose Cochain, a collaboration prompting framework that effectively solves the business workflow collaboration problem by combining knowledge and prompts at a reduced cost. Specifically, we construct an integrated knowledge graph that incorporates knowledge from multiple stages. Furthermore, by maintaining and retrieving a prompts tree, we can obtain prompt information relevant to other stages of the business workflow. We perform extensive evaluations of Cochain across multiple datasets, demonstrating that Cochain outperforms all baselines in both prompt engineering and multi-agent LLMs. Additionally, expert evaluation results indicate that the use of a small model in combination with Cochain outperforms GPT-4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20251v1">Efficient Evaluation of LLM Performance with Statistical Guarantees</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 24 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Exhaustively evaluating many large language models (LLMs) on a large suite of benchmarks is expensive. We cast benchmarking as finite-population inference and, under a fixed query budget, seek tight confidence intervals (CIs) for model accuracy with valid frequentist coverage. We propose Factorized Active Querying (FAQ), which (a) leverages historical information through a Bayesian factor model; (b) adaptively selects questions using a hybrid variance-reduction/active-learning sampling policy; and (c) maintains validity through Proactive Active Inference -- a finite-population extension of active inference (Zrnic & Candes, 2024) that enables direct question selection while preserving coverage. With negligible overhead cost, FAQ delivers up to $5\times$ effective sample size gains over strong baselines on two benchmark suites, across varying historical-data missingness levels: this means that it matches the CI width of uniform sampling while using up to $5\times$ fewer queries. We release our source code and our curated datasets to support reproducible evaluation and future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17261v2">AGZO: Activation-Guided Zeroth-Order Optimization for LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 21 pages in total, including 9 pages of main text, with 4 figures and 3 tables. This manuscript is submitted to arXiv
    </div>
    <details class="paper-abstract">
      Zeroth-Order (ZO) optimization has emerged as a promising solution for fine-tuning LLMs under strict memory constraints, as it avoids the prohibitive memory cost of storing activations for backpropagation. However, existing ZO methods typically employ isotropic perturbations, neglecting the rich structural information available during the forward pass. In this paper, we identify a crucial link between gradient formation and activation structure: the gradient of a linear layer is confined to the subspace spanned by its input activations. Leveraging this insight, we propose Activation-Guided Zeroth-Order optimization (AGZO). Unlike prior methods, AGZO extracts a compact, activation-informed subspace on the fly during the forward pass and restricts perturbations to this low-rank subspace. We provide a theoretical framework showing that AGZO optimizes a subspace-smoothed objective and provably yields update directions with higher cosine similarity to the true gradient than isotropic baselines. Empirically, we evaluate AGZO on Qwen3 and Pangu models across various benchmarks. AGZO consistently outperforms state-of-the-art ZO baselines and significantly narrows the performance gap with first-order fine-tuning, while maintaining almost the same peak memory footprint as other ZO methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.10187v2">HOMURA: Taming the Sand-Glass for Time-Constrained LLM Translation via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable strides in multilingual translation but are hindered by a systemic cross-lingual verbosity bias, rendering them unsuitable for strict time-constrained tasks like subtitling and dubbing. Current prompt-engineering approaches struggle to resolve this conflict between semantic fidelity and rigid temporal feasibility. To bridge this gap, we first introduce Sand-Glass, a benchmark specifically designed to evaluate translation under syllable-level duration constraints. Furthermore, we propose HOMURA, a reinforcement learning framework that explicitly optimizes the trade-off between semantic preservation and temporal compliance. By employing a KL-regularized objective with a novel dynamic syllable-ratio reward, HOMURA effectively "tames" the output length. Experimental results demonstrate that our method significantly outperforms strong LLM baselines, achieving precise length control that respects linguistic density hierarchies without compromising semantic adequacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.09885v3">Closing the Data-Efficiency Gap Between Autoregressive and Masked Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are often used in environments where facts evolve, yet factual knowledge updates via fine-tuning on unstructured text often suffers from 1) reliance on compute-heavy paraphrase augmentation and 2) the reversal curse. Recent studies show diffusion large language models (dLLMs) require fewer training samples to achieve lower loss in pre-training and are more resistant to the reversal curse, suggesting dLLMs may learn new knowledge more easily than autoregressive LLMs (arLLMs). We test this hypothesis in controlled knowledge fine-tuning experiments and find that while arLLMs rely on paraphrase augmentation to generalize knowledge text into question-answering (QA) capability, dLLMs do not require paraphrases to achieve high QA accuracy. To further investigate whether the demasking objective alone can induce such a knowledge injection advantage in dLLMs regardless of their diffusion denoising paradigm, we propose masked fine-tuning for arLLMs, which prompts an arLLM to reconstruct the original text given a masked version in context. The masked fine-tuning for arLLMs substantially improves the efficacy of knowledge injection, i.e. no paraphrase needed and resistant to the reversal curse, closing the gap between arLLMs and dLLMs. We also demonstrate that the same demasking objective improves supervised fine-tuning (SFT) on math tasks over standard SFT, suggesting broader applicability of the demasking objective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.00282v3">Mind the Gap: The Divergence Between Human and LLM-Generated Tasks</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Humans constantly generate a diverse range of tasks guided by internal motivations. While generative agents powered by large language models (LLMs) aim to simulate this complex behavior, it remains uncertain whether they operate on similar cognitive principles. To address this, we conducted a task-generation experiment comparing human responses with those of an LLM agent (GPT-4o). We find that human task generation is consistently influenced by psychological drivers, including personal values (e.g., Openness to Change) and cognitive style. Even when these psychological drivers are explicitly provided to the LLM, it fails to reflect the corresponding behavioral patterns. They produce tasks that are markedly less social, less physical, and thematically biased toward abstraction. Interestingly, while the LLM's tasks were perceived as more fun and novel, this highlights a disconnect between its linguistic proficiency and its capacity to generate human-like, embodied goals. We conclude that there is a core gap between the value-driven, embodied nature of human cognition and the statistical patterns of LLMs, highlighting the necessity of incorporating intrinsic motivation and physical grounding into the design of more human-aligned agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20206v1">Towards Intelligent Urban Park Development Monitoring: LLM Agents for Multi-Modal Information Fusion and Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      As an important part of urbanization, the development monitoring of newly constructed parks is of great significance for evaluating the effect of urban planning and optimizing resource allocation. However, traditional change detection methods based on remote sensing imagery have obvious limitations in high-level and intelligent analysis, and thus are difficult to meet the requirements of current urban planning and management. In face of the growing demand for complex multi-modal data analysis in urban park development monitoring, these methods often fail to provide flexible analysis capabilities for diverse application scenarios. This study proposes a multi-modal LLM agent framework, which aims to make full use of the semantic understanding and reasoning capabilities of LLM to meet the challenges in urban park development monitoring. In this framework, a general horizontal and vertical data alignment mechanism is designed to ensure the consistency and effective tracking of multi-modal data. At the same time, a specific toolkit is constructed to alleviate the hallucination issues of LLM due to the lack of domain-specific knowledge. Compared to vanilla GPT-4o and other agents, our approach enables robust multi-modal information fusion and analysis, offering reliable and scalable solutions tailored to the diverse and evolving demands of urban park development monitoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20196v1">Automated Marine Biofouling Assessment: Benchmarking Computer Vision and Multimodal LLMs on the Level of Fouling Scale</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Australasian Conference on Robotics and Automation, ACRA2025 13 Pages, 8 Figures
    </div>
    <details class="paper-abstract">
      Marine biofouling on vessel hulls poses major ecological, economic, and biosecurity risks. Traditional survey methods rely on diver inspections, which are hazardous and limited in scalability. This work investigates automated classification of biofouling severity on the Level of Fouling (LoF) scale using both custom computer vision models and large multimodal language models (LLMs). Convolutional neural networks, transformer-based segmentation, and zero-shot LLMs were evaluated on an expert-labelled dataset from the New Zealand Ministry for Primary Industries. Computer vision models showed high accuracy at extreme LoF categories but struggled with intermediate levels due to dataset imbalance and image framing. LLMs, guided by structured prompts and retrieval, achieved competitive performance without training and provided interpretable outputs. The results demonstrate complementary strengths across approaches and suggest that hybrid methods integrating segmentation coverage with LLM reasoning offer a promising pathway toward scalable and interpretable biofouling assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.19920v3">Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      LLM deployment in critical domains is currently impeded by persistent hallucinations--generating plausible but factually incorrect assertions. While scaling laws drove significant improvements in general capabilities, theoretical frameworks suggest hallucination is not merely stochastic error but a predictable statistical consequence of training objectives prioritizing mimicking data distribution over epistemic honesty. Standard RLVR paradigms, utilizing binary reward signals, inadvertently incentivize models as good test-takers rather than honest communicators, encouraging guessing whenever correctness probability exceeds zero. This paper presents an exhaustive investigation into behavioral calibration, which incentivizes models to stochastically admit uncertainty by abstaining when not confident, aligning model behavior with accuracy. Synthesizing recent advances, we propose and evaluate training interventions optimizing strictly proper scoring rules for models to output a calibrated probability of correctness. Our methods enable models to either abstain from producing a complete response or flag individual claims where uncertainty remains. Utilizing Qwen3-4B-Instruct, empirical analysis reveals behavior-calibrated reinforcement learning allows smaller models to surpass frontier models in uncertainty quantification--a transferable meta-skill decouplable from raw predictive accuracy. Trained on math reasoning tasks, our model's log-scale Accuracy-to-Hallucination Ratio gain (0.806) exceeds GPT-5's (0.207) in a challenging in-domain evaluation (BeyondAIME). Moreover, in cross-domain factual QA (SimpleQA), our 4B LLM achieves zero-shot calibration error on par with frontier models including Grok-4 and Gemini-2.5-Pro, even though its factual accuracy is much lower.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.20846v2">Are LLMs Really Not Knowledgeable? Mining the Submerged Knowledge in LLMs' Memory</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise as parametric knowledge bases, but often underperform on question answering (QA) tasks due to hallucinations and uncertainty. While prior work attributes these failures to knowledge gaps in the model's parameters, we uncover a complementary phenomenon: LLMs frequently retain correct knowledge even when generating incorrect or "unsure" answers. By analyzing the token-level output distributions, we find that correct answers often appear among high-probability candidates, despite not being selected. Motivated by this, we propose Hits@k, a novel metric to evaluate latent knowledge retention independent of answer surface form. Our experiments reveal that LLMs possess significantly more factual knowledge than is reflected by standard QA accuracy. Building on these insights, we further examine the prevailing few-shot QA paradigm. We find that prompting strategies which allow "unsure" outputs can inadvertently suppress correct answers by discouraging low-confidence generation. We design a set of quantitative experiments to measure this suppression effect, offering practical guidance for future prompt and decoding design in knowledge-intensive tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20164v1">What's the plan? Metrics for implicit planning in LLMs and their application to rhyme generation and question answering</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 41 pages, 34 figures, Accepted at ICLR 2026, Code available at https://github.com/Jim-Maar/implicit-planning-in-llms
    </div>
    <details class="paper-abstract">
      Prior work suggests that language models, while trained on next token prediction, show implicit planning behavior: they may select the next token in preparation to a predicted future token, such as a likely rhyming word, as supported by a prior qualitative study of Claude 3.5 Haiku using a cross-layer transcoder. We propose much simpler techniques for assessing implicit planning in language models. With case studies on rhyme poetry generation and question answering, we demonstrate that our methodology easily scales to many models. Across models, we find that the generated rhyme (e.g. "-ight") or answer to a question ("whale") can be manipulated by steering at the end of the preceding line with a vector, affecting the generation of intermediate tokens leading up to the rhyme or answer word. We show that implicit planning is a universal mechanism, present in smaller models than previously thought, starting from 1B parameters. Our methodology offers a widely applicable direct way to study implicit planning abilities of LLMs. More broadly, understanding planning abilities of language models can inform decisions in AI safety and control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.20148v1">LogSieve: Task-Aware CI Log Reduction for Sustainable LLM-Based Analysis</a></div>
    <div class="paper-meta">
      📅 2026-01-28
      | 💬 Preprint. Accepted for presentation at Mining Software Repositories (MSR'26), co-located ICSE 2026. The final version will appear in the ACM Digital Library as part of the MSR'26 conference proceedings
    </div>
    <details class="paper-abstract">
      Logs are essential for understanding Continuous Integration (CI) behavior, particularly for diagnosing build failures and performance regressions. Yet their growing volume and verbosity make both manual inspection and automated analysis increasingly costly, time-consuming, and environmentally costly. While prior work has explored log compression, anomaly detection, and LLM-based log analysis, most efforts target structured system logs rather than the unstructured, noisy, and verbose logs typical of CI workflows. We present LogSieve, a lightweight, RCA-aware and semantics-preserving log reduction technique that filters low-information lines while retaining content relevant to downstream reasoning. Evaluated on CI logs from 20 open-source Android projects using GitHub Actions, LogSieve achieves an average 42% reduction in lines and 40% reduction in tokens with minimal semantic loss. This pre-inference reduction lowers computational cost and can proportionally reduce energy use (and associated emissions) by decreasing the volume of data processed during LLM inference. Compared with structure-first baselines (LogZip and random-line removal), LogSieve preserves much higher semantic and categorical fidelity (Cosine = 0.93, GPTScore = 0.93, 80% exact-match accuracy). Embedding-based classifiers automate relevance detection with near-human accuracy (97%), enabling scalable and sustainable integration of semantics-aware filtering into CI workflows. LogSieve thus bridges log management and LLM reasoning, offering a practical path toward greener and more interpretable CI automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.17087v2">Lost in Simulation: LLM-Simulated Users are Unreliable Proxies for Human Users in Agentic Evaluations</a></div>
    <div class="paper-meta">
      📅 2026-01-28
    </div>
    <details class="paper-abstract">
      Agentic benchmarks increasingly rely on LLM-simulated users to scalably evaluate agent performance, yet the robustness, validity, and fairness of this approach remain unexamined. Through a user study with participants across the United States, India, Kenya, and Nigeria, we investigate whether LLM-simulated users serve as reliable proxies for real human users in evaluating agents on τ-Bench retail tasks. We find that user simulation lacks robustness, with agent success rates varying up to 9 percentage points across different user LLMs. Furthermore, evaluations using simulated users exhibit systematic miscalibration, underestimating agent performance on challenging tasks and overestimating it on moderately difficult ones. African American Vernacular English (AAVE) speakers experience consistently worse success rates and calibration errors than Standard American English (SAE) speakers, with disparities compounding significantly with age. We also find simulated users to be a differentially effective proxy for different populations, performing worst for AAVE and Indian English speakers. Additionally, simulated users introduce conversational artifacts and surface different failure patterns than human users. These findings demonstrate that current evaluation practices risk misrepresenting agent capabilities across diverse user populations and may obscure real-world deployment challenges.
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
