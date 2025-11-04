# llm - 2025_10

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
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)
- Part 14
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)
- [Part 18](papers_18.md)
- [Part 19](papers_19.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05220v5">Utility-Focused LLM Annotation for Retrieval and Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted by the EMNLP25 main conference
    </div>
    <details class="paper-abstract">
      This paper explores the use of large language models (LLMs) for annotating document utility in training retrieval and retrieval-augmented generation (RAG) systems, aiming to reduce dependence on costly human annotations. We address the gap between retrieval relevance and generative utility by employing LLMs to annotate document utility. To effectively utilize multiple positive samples per query, we introduce a novel loss that maximizes their summed marginal likelihood. Using the Qwen-2.5-32B model, we annotate utility on the MS MARCO dataset and conduct retrieval experiments on MS MARCO and BEIR, as well as RAG experiments on MS MARCO QA, NQ, and HotpotQA. Our results show that LLM-generated annotations enhance out-of-domain retrieval performance and improve RAG outcomes compared to models trained solely on human annotations or downstream QA metrics. Furthermore, combining LLM annotations with just 20% of human labels achieves performance comparable to using full human annotations. Our study offers a comprehensive approach to utilizing LLM annotations for initializing QA systems on new corpora.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01493v2">Sherkala-Chat: Building a State-of-the-Art LLM for Kazakh in a Moderately Resourced Setting</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted at COLM 2025
    </div>
    <details class="paper-abstract">
      Llama-3.1-Sherkala-8B-Chat, or Sherkala-Chat (8B) for short, is a state-of-the-art instruction-tuned open generative large language model (LLM) designed for Kazakh. Sherkala-Chat (8B) aims to enhance the inclusivity of LLM advancements for Kazakh speakers. Adapted from the LLaMA-3.1-8B model, Sherkala-Chat (8B) is trained on 45.3B tokens across Kazakh, English, Russian, and Turkish. With 8 billion parameters, it demonstrates strong knowledge and reasoning abilities in Kazakh, significantly outper-forming existing open Kazakh and multilingual models of similar scale while achieving competitive performance in English. To ensure effective and responsible alignment, we leverage translated instruction datasets, a Kazakhstan-specific instruction dataset that is automatically constructed and manually verified, and Kazakh-specific safety data. We release Sherkala-Chat (8B) as an open-weight model, along with a detailed description of its training, alignment, and evaluation, to support research and real-world applications for Kazakh speakers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07743v1">OpenRubrics: Towards Scalable Synthetic Rubric Generation for Reward Modeling and LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 The first two authors contributed equally
    </div>
    <details class="paper-abstract">
      Reward modeling lies at the core of reinforcement learning from human feedback (RLHF), yet most existing reward models rely on scalar or pairwise judgments that fail to capture the multifaceted nature of human preferences. Recent studies have explored rubrics-as-rewards (RaR) that uses structured natural language criteria that capture multiple dimensions of response quality. However, producing rubrics that are both reliable and scalable remains a key challenge. In this work, we introduce OpenRubrics, a diverse, large-scale collection of (prompt, rubric) pairs for training rubric-generation and rubric-based reward models. To elicit discriminative and comprehensive evaluation signals, we introduce Contrastive Rubric Generation (CRG), which derives both hard rules (explicit constraints) and principles (implicit qualities) by contrasting preferred and rejected responses. We further improve reliability by enforcing preference-label consistency via rejection sampling to remove noisy rubrics. Across multiple reward-modeling benchmarks, our rubric-based reward model, Rubric-RM, surpasses strong size-matched baselines by 6.8%. These gains transfer to policy models on instruction-following and biomedical benchmarks. Our results show that rubrics provide scalable alignment signals that narrow the gap between costly human evaluation and automated reward modeling, enabling a new principle-driven paradigm for LLM alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14315v5">Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Maintaining consistent model performance across domains is a fundamental challenge in machine learning. While recent work has explored using LLM-generated data for fine-tuning, its impact on cross-domain generalization remains poorly understood. This paper presents a systematic analysis revealing that fine-tuning with LLM-generated data not only improves target task performance but also reduces non-target task degradation compared to fine-tuning with ground truth data. Through analyzing the data sequence in tasks of various domains, we demonstrate that this enhancement of non-target task robustness stems from the reduction of high perplexity tokens found in LLM-generated sequences. Following our findings, we showed that masking high perplexity tokens in ground truth training data achieves similar non-target task performance preservation, comparable to using LLM-generated data. Extensive experiments across different model families and scales, including Gemma 2 IT 2B, Llama 3 8B Instruct, and three additional models, agree with our findings. To the best of our knowledge, this is the first work to provide an empirical explanation based on token perplexity reduction to mitigate catastrophic forgetting in LLMs after fine-tuning, offering valuable insights for developing more robust fine-tuning strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07737v1">ToolExpander: Extending the Frontiers of Tool-Using Reinforcement Learning to Weak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Training Large Language Models (LLMs) with Group Relative Policy Optimization (GRPO) encounters a significant challenge: models often fail to produce accurate responses, particularly in small-scale architectures. This limitation not only diminishes performance improvements and undermines the potential of GRPO but also frequently leads to mid-training collapse, adversely affecting stability and final efficacy. To address these issues, we propose ToolExpander, a novel framework that advances tool-oriented reinforcement learning for resource-constrained LLMs through two key innovations:(1) Dynamic Multi-Round Hard Sampling, which dynamically substitutes challenging samples(those without correct outputs over 10 rollouts) with high-quality few-shot demonstrations during training, coupled with an exponential learning rate decay strategy to mitigate oscillations;(2) Self-Exemplifying Thinking, an enhanced GRPO framework that eliminates KL divergence and incorporates adjusted clipping coefficients, encouraging models to autonomously generate and analyze few-shot examples via a minimal additional reward (0.01).Experimental results demonstrate that ToolExpander significantly enhances tool-using capabilities in LLMs, especially in weaker small-scale models, improving both training stability and overall performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25034v2">MARLIN: Multi-Agent Reinforcement Learning with Murmuration Intelligence and LLM Guidance for Reservoir Management</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      As climate change intensifies extreme weather events, water disasters pose growing threats to global communities, making adaptive reservoir management critical for protecting vulnerable populations and ensuring water security. Modern water resource management faces unprecedented challenges from cascading uncertainties propagating through interconnected reservoir networks. These uncertainties, rooted in physical water transfer losses and environmental variability, make precise control difficult. For example, sending 10 tons downstream may yield only 8-12 tons due to evaporation and seepage. Traditional centralized optimization approaches suffer from exponential computational complexity and cannot effectively handle such real-world uncertainties, while existing multi-agent reinforcement learning (MARL) methods fail to achieve effective coordination under uncertainty. To address these challenges, we present MARLIN, a decentralized reservoir management framework inspired by starling murmurations intelligence. Integrating bio-inspired alignment, separation, and cohesion rules with MARL, MARLIN enables individual reservoirs to make local decisions while achieving emergent global coordination. In addition, a LLM provides real-time reward shaping signals, guiding agents to adapt to environmental changes and human-defined preferences. Experiments on real-world USGS data show that MARLIN improves uncertainty handling by 23\%, cuts computation by 35\%, and accelerates flood response by 68\%, exhibiting super-linear coordination, with complexity scaling 5.4x from 400 to 10,000 nodes. These results demonstrate MARLIN's potential for disaster prevention and protecting communities through intelligent, scalable water resource management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07733v1">SurveyG: A Multi-Agent LLM Framework with Hierarchical Citation Graph for Automated Survey Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly adopted for automating survey paper generation \cite{wang2406autosurvey, liang2025surveyx, yan2025surveyforge,su2025benchmarking,wen2025interactivesurvey}. Existing approaches typically extract content from a large collection of related papers and prompt LLMs to summarize them directly. However, such methods often overlook the structural relationships among papers, resulting in generated surveys that lack a coherent taxonomy and a deeper contextual understanding of research progress. To address these shortcomings, we propose \textbf{SurveyG}, an LLM-based agent framework that integrates \textit{hierarchical citation graph}, where nodes denote research papers and edges capture both citation dependencies and semantic relatedness between their contents, thereby embedding structural and contextual knowledge into the survey generation process. The graph is organized into three layers: \textbf{Foundation}, \textbf{Development}, and \textbf{Frontier}, to capture the evolution of research from seminal works to incremental advances and emerging directions. By combining horizontal search within layers and vertical depth traversal across layers, the agent produces multi-level summaries, which are consolidated into a structured survey outline. A multi-agent validation stage then ensures consistency, coverage, and factual accuracy in generating the final survey. Experiments, including evaluations by human experts and LLM-as-a-judge, demonstrate that SurveyG outperforms state-of-the-art frameworks, producing surveys that are more comprehensive and better structured to the underlying knowledge taxonomy of a field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07731v1">oMeBench: Towards Robust Benchmarking of LLMs in Organic Mechanism Elucidation and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Main Text: 8 pages, In total: 37 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Organic reaction mechanisms are the stepwise elementary reactions by which reactants form intermediates and products, and are fundamental to understanding chemical reactivity and designing new molecules and reactions. Although large language models (LLMs) have shown promise in understanding chemical tasks such as synthesis design, it is unclear to what extent this reflects genuine chemical reasoning capabilities, i.e., the ability to generate valid intermediates, maintain chemical consistency, and follow logically coherent multi-step pathways. We address this by introducing oMeBench, the first large-scale, expert-curated benchmark for organic mechanism reasoning in organic chemistry. It comprises over 10,000 annotated mechanistic steps with intermediates, type labels, and difficulty ratings. Furthermore, to evaluate LLM capability more precisely and enable fine-grained scoring, we propose oMeS, a dynamic evaluation framework that combines step-level logic and chemical similarity. We analyze the performance of state-of-the-art LLMs, and our results show that although current models display promising chemical intuition, they struggle with correct and consistent multi-step reasoning. Notably, we find that using prompting strategy and fine-tuning a specialist model on our proposed dataset increases performance by 50% over the leading closed-source model. We hope that oMeBench will serve as a rigorous foundation for advancing AI systems toward genuine chemical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07697v1">Rethinking Reasoning: A Survey on Reasoning-based Backdoors in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      With the rise of advanced reasoning capabilities, large language models (LLMs) are receiving increasing attention. However, although reasoning improves LLMs' performance on downstream tasks, it also introduces new security risks, as adversaries can exploit these capabilities to conduct backdoor attacks. Existing surveys on backdoor attacks and reasoning security offer comprehensive overviews but lack in-depth analysis of backdoor attacks and defenses targeting LLMs' reasoning abilities. In this paper, we take the first step toward providing a comprehensive review of reasoning-based backdoor attacks in LLMs by analyzing their underlying mechanisms, methodological frameworks, and unresolved challenges. Specifically, we introduce a new taxonomy that offers a unified perspective for summarizing existing approaches, categorizing reasoning-based backdoor attacks into associative, passive, and active. We also present defense strategies against such attacks and discuss current challenges alongside potential directions for future research. This work offers a novel perspective, paving the way for further exploration of secure and trustworthy LLM communities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04596v2">LLM Applications: Current Paradigms and the Next Frontier</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      The development of large language models (LLMs) has given rise to four major application paradigms: LLM app stores, LLM agents, self-hosted LLM services, and LLM-powered devices. Each has its advantages but also shares common challenges. LLM app stores lower the barrier to development but lead to platform lock-in; LLM agents provide autonomy but lack a unified communication mechanism; self-hosted LLM services enhance control but increase deployment complexity; and LLM-powered devices improve privacy and real-time performance but are limited by hardware. This paper reviews and analyzes these paradigms, covering architecture design, application ecosystem, research progress, as well as the challenges and open problems they face. Based on this, we outline the next frontier of LLM applications, characterizing them through three interconnected layers: infrastructure, protocol, and application. We describe their responsibilities and roles of each layer and demonstrate how to mitigate existing fragmentation limitations and improve security and scalability. Finally, we discuss key future challenges, identify opportunities such as protocol-driven cross-platform collaboration and device integration, and propose a research roadmap for openness, security, and sustainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.18952v2">LLMs on a Budget? Say HOLA</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted at EMNLP 2025 (Industry Track)
    </div>
    <details class="paper-abstract">
      Running Large Language Models (LLMs) on edge devices is constrained by high compute and memory demands posing a barrier for real-time applications in sectors like healthcare, education, and embedded systems. Current solutions such as quantization, pruning, and retrieval-augmented generation (RAG) offer only partial optimizations and often compromise on speed or accuracy. We introduce HOLA, an end-to-end optimization framework for efficient LLM deployment. Internally, it leverages Hierarchical Speculative Decoding (HSD) for faster inference without quality loss. Externally, AdaComp-RAG adjusts retrieval complexity based on context needs. Together with LoBi, which blends structured pruning (LoRA) and quantization, HOLA delivers significant gains: 17.6% EMA on GSM8K, 10.5% MCA on ARC, and reduced latency and memory on edge devices like Jetson Nano--proving both scalable and production-ready.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06674v2">Agent-in-the-Loop: A Data Flywheel for Continuous Improvement in LLM-based Customer Support</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 EMNLP 2025 Industry Track submission (Paper #305). Preprint. Main text within the 7-page industry limit (references/appendices excluded). Contains multiple figures and tables
    </div>
    <details class="paper-abstract">
      We introduce an Agent-in-the-Loop (AITL) framework that implements a continuous data flywheel for iteratively improving an LLM-based customer support system. Unlike standard offline approaches that rely on batch annotations, AITL integrates four key types of annotations directly into live customer operations: (1) pairwise response preferences, (2) agent adoption and rationales, (3) knowledge relevance checks, and (4) identification of missing knowledge. These feedback signals seamlessly feed back into models' updates, reducing retraining cycles from months to weeks. Our production pilot involving US-based customer support agents demonstrated significant improvements in retrieval accuracy (+11.7% recall@75, +14.8% precision@8), generation quality (+8.4% helpfulness) and agent adoption rates (+4.5%). These results underscore the effectiveness of embedding human feedback loops directly into operational workflows to continuously refine LLM-based customer support system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14053v2">MAHL: Multi-Agent LLM-Guided Hierarchical Chiplet Design with Adaptive Debugging</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      As program workloads (e.g., AI) increase in size and algorithmic complexity, the primary challenge lies in their high dimensionality, encompassing computing cores, array sizes, and memory hierarchies. To overcome these obstacles, innovative approaches are required. Agile chip design has already benefited from machine learning integration at various stages, including logic synthesis, placement, and routing. With Large Language Models (LLMs) recently demonstrating impressive proficiency in Hardware Description Language (HDL) generation, it is promising to extend their abilities to 2.5D integration, an advanced technique that saves area overhead and development costs. However, LLM-driven chiplet design faces challenges such as flatten design, high validation cost and imprecise parameter optimization, which limit its chiplet design capability. To address this, we propose MAHL, a hierarchical LLM-based chiplet design generation framework that features six agents which collaboratively enable AI algorithm-hardware mapping, including hierarchical description generation, retrieval-augmented code generation, diverseflow-based validation, and multi-granularity design space exploration. These components together enhance the efficient generation of chiplet design with optimized Power, Performance and Area (PPA). Experiments show that MAHL not only significantly improves the generation accuracy of simple RTL design, but also increases the generation accuracy of real-world chiplet design, evaluated by Pass@5, from 0 to 0.72 compared to conventional LLMs under the best-case scenario. Compared to state-of-the-art CLARIE (expert-based), MAHL achieves comparable or even superior PPA results under certain optimization objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05393v2">HiVeGen -- Hierarchical LLM-based Verilog Generation for Scalable Chip Design</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      With Large Language Models (LLMs) recently demonstrating impressive proficiency in code generation, it is promising to extend their abilities to Hardware Description Language (HDL). However, LLMs tend to generate single HDL code blocks rather than hierarchical structures for hardware designs, leading to hallucinations, particularly in complex designs like Domain-Specific Accelerators (DSAs). To address this, we propose HiVeGen, a hierarchical LLM-based Verilog generation framework that decomposes generation tasks into LLM-manageable hierarchical submodules. HiVeGen further harnesses the advantages of such hierarchical structures by integrating automatic Design Space Exploration (DSE) into hierarchy-aware prompt generation, introducing weight-based retrieval to enhance code reuse, and enabling real-time human-computer interaction to lower error-correction cost, significantly improving the quality of generated designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13933v2">Preprint: Poster: Did I Just Browse A Website Written by LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 ACM Internet Measurement Conference 2025 Poster & ACM IMC 2025 Student Workshop. 2 pages. 3 figures
    </div>
    <details class="paper-abstract">
      Increasingly, web content is automatically generated by large language models (LLMs) with little human input. We call this "LLM-dominant" content. Since LLMs plagiarize and hallucinate, LLM-dominant content can be unreliable and unethical. Yet, websites rarely disclose such content, and human readers struggle to distinguish it. Thus, we must develop reliable detectors for LLM-dominant content. However, state-of-the-art LLM detectors are inaccurate on web content, because web content has low positive rates, complex markup, and diverse genres, instead of clean, prose-like benchmark data SoTA detectors are optimized for. We propose a highly reliable, scalable pipeline that classifies entire websites. Instead of naively classifying text extracted from each page, we classify each site based on an LLM text detector's outputs of multiple prose-like pages to boost accuracies. We train and evaluate our detector by collecting 2 distinct ground truth datasets totaling 120 sites, and obtain 100% accuracies testing across them. In the wild, we detect a sizable portion of sites as LLM-dominant among 10k sites in search engine results and 10k in Common Crawl archives. We find LLM-dominant sites are growing in prevalence and rank highly in search results, raising questions about their impact on end users and the overall Web ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08850v1">Repository-Aware File Path Retrieval via Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Modern codebases make it hard for developers and AI coding assistants to find the right source files when answering questions like "How does this feature work?" or "Where was the bug introduced?" Traditional code search (keyword or IR based) often misses semantic context and cross file links, while large language models (LLMs) understand natural language but lack repository specific detail. We present a method for file path retrieval that fine tunes a strong LLM (Qwen3-8B) with QLoRA and Unsloth optimizations to predict relevant file paths directly from a natural language query. To build training data, we introduce six code aware strategies that use abstract syntax tree (AST) structure and repository content to generate realistic question-answer pairs, where answers are sets of file paths. The strategies range from single file prompts to hierarchical repository summaries, providing broad coverage. We fine tune on Python projects including Flask, Click, Jinja, FastAPI, and PyTorch, and obtain high retrieval accuracy: up to 91\% exact match and 93\% recall on held out queries, clearly beating single strategy training. On a large codebase like PyTorch (about 4,000 Python files), the model reaches 59\% recall, showing scalability. We analyze how multi level code signals help the LLM reason over cross file context and discuss dataset design, limits (for example, context length in very large repos), and future integration of retrieval with LLM based code intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.06094v2">ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Project page: https://conlangcrafter.github.io
    </div>
    <details class="paper-abstract">
      Constructed languages (conlangs) such as Esperanto and Quenya have played diverse roles in art, philosophy, and international communication. Meanwhile, foundation models have revolutionized creative generation in text, images, and beyond. In this work, we leverage modern LLMs as computational creativity aids for end-to-end conlang creation. We introduce ConlangCrafter, a multi-hop pipeline that decomposes language design into modular stages - phonology, morphology, syntax, lexicon generation, and translation. At each stage, our method leverages LLMs' metalinguistic reasoning capabilities, injecting randomness to encourage diversity and leveraging self-refinement feedback to encourage consistency in the emerging language description. We evaluate ConlangCrafter on metrics measuring consistency and typological diversity, demonstrating its ability to produce coherent and varied conlangs without human linguistic expertise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07651v1">OBCache: Optimal Brain KV Cache Pruning for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with extended context windows enable powerful downstream applications but impose significant memory overhead, as caching all key-value (KV) states scales linearly with sequence length and batch size. Existing cache eviction methods address this by exploiting attention sparsity, yet they typically rank tokens heuristically using accumulated attention weights without considering their true impact on attention outputs. We propose Optimal Brain Cache (OBCache), a principled framework that formulates cache eviction as a layer-wise structured pruning problem. Building upon the Optimal Brain Damage (OBD) theory, OBCache quantifies token saliency by measuring the perturbation in attention outputs induced by pruning tokens, with closed-form scores derived for isolated keys, isolated values, and joint key-value pairs. Our scores account not only for attention weights but also for information from value states and attention outputs, thereby enhancing existing eviction strategies with output-aware signals. Experiments on LLaMA and Qwen models demonstrate that replacing the heuristic scores in existing works, which estimate token saliency across different query positions, with OBCache's output-aware scores consistently improves long-context accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10833v3">MergeBench: A Benchmark for Merging Domain-Specialized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 NeurIPS 2025 Track on Datasets and Benchmarks
    </div>
    <details class="paper-abstract">
      Model merging provides a scalable alternative to multi-task training by combining specialized finetuned models through parameter arithmetic, enabling efficient deployment without the need for joint training or access to all task data. While recent methods have shown promise, existing evaluations are limited in both model scale and task diversity, leaving open questions about their applicability to large, domain-specialized LLMs. To tackle the challenges, we introduce MergeBench, a comprehensive evaluation suite designed to assess model merging at scale. MergeBench builds on state-of-the-art open-source language models, including Llama and Gemma families at 2B to 9B scales, and covers five key domains: instruction following, mathematics, multilingual understanding, coding and safety. We standardize finetuning and evaluation protocols, and assess eight representative merging methods across multi-task performance, forgetting and runtime efficiency. Based on extensive experiments, we provide practical guidelines for algorithm selection and share insights showing that model merging tends to perform better on stronger base models, with techniques such as merging coefficient tuning and sparsification improving knowledge retention. However, several challenges remain, including the computational cost on large models, the gap for in-domain performance compared to multi-task models, and the underexplored role of model merging in standard LLM training pipelines. We hope MergeBench provides a foundation for future research to advance the understanding and practical application of model merging. Our project page is at \href{https://yifei-he.github.io/mergebench/}{https://yifei-he.github.io/mergebench/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06250v2">Scalable multilingual PII annotation for responsible AI in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) gain wider adoption, ensuring their reliable handling of Personally Identifiable Information (PII) across diverse regulatory contexts has become essential. This work introduces a scalable multilingual data curation framework designed for high-quality PII annotation across 13 underrepresented locales, covering approximately 336 locale-specific PII types. Our phased, human-in-the-loop annotation methodology combines linguistic expertise with rigorous quality assurance, leading to substantial improvements in recall and false positive rates from pilot, training, and production phases. By leveraging inter-annotator agreement metrics and root-cause analysis, the framework systematically uncovers and resolves annotation inconsistencies, resulting in high-fidelity datasets suitable for supervised LLM fine-tuning. Beyond reporting empirical gains, we highlight common annotator challenges in multilingual PII labeling and demonstrate how iterative, analytics-driven pipelines can enhance both annotation quality and downstream model reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17410v3">COSMOS: A Hybrid Adaptive Optimizer for Memory-Efficient Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 23 pages, 9 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable success across various domains, yet their optimization remains a significant challenge due to the complex and high-dimensional loss landscapes they inhabit. While adaptive optimizers such as AdamW are widely used, they suffer from critical limitations, including an inability to capture interdependencies between coordinates and high memory consumption. Subsequent research, exemplified by SOAP, attempts to better capture coordinate interdependence but incurs greater memory overhead, limiting scalability for massive LLMs. An alternative approach aims to reduce memory consumption through low-dimensional projection, but this leads to substantial approximation errors, resulting in less effective optimization (e.g., in terms of per-token efficiency). In this paper, we propose COSMOS, a novel hybrid optimizer that leverages the varying importance of eigensubspaces in the gradient matrix to achieve memory efficiency without compromising optimization performance. The design of COSMOS is motivated by our empirical insights and practical considerations. Specifically, COSMOS applies SOAP to the leading eigensubspace, which captures the primary optimization dynamics, and MUON to the remaining eigensubspace, which is less critical but computationally expensive to handle with SOAP. This hybrid strategy significantly reduces memory consumption while maintaining robust optimization performance, making it particularly suitable for massive LLMs. Numerical experiments on various datasets and transformer architectures are provided to demonstrate the effectiveness of COSMOS. Our code is available at https://github.com/lliu606/COSMOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08813v1">The Model's Language Matters: A Comparative Privacy Analysis of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed across multilingual applications that handle sensitive data, yet their scale and linguistic variability introduce major privacy risks. Mostly evaluated for English, this paper investigates how language structure affects privacy leakage in LLMs trained on English, Spanish, French, and Italian medical corpora. We quantify six linguistic indicators and evaluate three attack vectors: extraction, counterfactual memorization, and membership inference. Results show that privacy vulnerability scales with linguistic redundancy and tokenization granularity: Italian exhibits the strongest leakage, while English shows higher membership separability. In contrast, French and Spanish display greater resilience due to higher morphological complexity. Overall, our findings provide the first quantitative evidence that language matters in privacy leakage, underscoring the need for language-aware privacy-preserving mechanisms in LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08779v1">Guiding Exploration in Reinforcement Learning Through LLM-Augmented Observations</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted to LM4Plan Workshop @ ICAPS 2025 (withdrawn before presentation due to lack of travel funding)
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) agents often struggle in sparse-reward environments where traditional exploration strategies fail to discover effective action sequences. Large Language Models (LLMs) possess procedural knowledge and reasoning capabilities from text pretraining that could guide RL exploration, but existing approaches create rigid dependencies where RL policies must follow LLM suggestions or incorporate them directly into reward functions. We propose a framework that provides LLM-generated action recommendations through augmented observation spaces, allowing RL agents to learn when to follow or ignore this guidance. Our method leverages LLMs' world knowledge and reasoning abilities while maintaining flexibility through soft constraints. We evaluate our approach on three BabyAI environments of increasing complexity and show that the benefits of LLM guidance scale with task difficulty. In the most challenging environment, we achieve 71% relative improvement in final success rates over baseline. The approach provides substantial sample efficiency gains, with agents reaching performance thresholds up to 9 times faster, and requires no modifications to existing RL algorithms. Our results demonstrate an effective method for leveraging LLM planning capabilities to accelerate RL training in challenging environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08776v1">Measuring Moral LLM Responses in Multilingual Capacities</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 10 pages, 5 figures; referenced articles: arXiv:2303.08774, arXiv:2303.12528, arXiv:2308.14132, arXiv:2505.12201, arXiv:2406.04428, arXiv:2407.02273, arXiv:2404.01268, arXiv:2502.09747, arXiv:2507.13474, arXiv:2505.21479, arXiv:2306.05685
    </div>
    <details class="paper-abstract">
      With LLM usage becoming widespread across countries, languages, and humanity more broadly, the need to understand and guardrail their multilingual responses increases. Large-scale datasets for testing and benchmarking have been created to evaluate and facilitate LLM responses across multiple dimensions. In this study, we evaluate the responses of frontier and leading open-source models in five dimensions across low and high-resource languages to measure LLM accuracy and consistency across multilingual contexts. We evaluate the responses using a five-point grading rubric and a judge LLM. Our study shows that GPT-5 performed the best on average in each category, while other models displayed more inconsistency across language and category. Most notably, in the Consent & Autonomy and Harm Prevention & Safety categories, GPT scored the highest with averages of 3.56 and 4.73, while Gemini 2.5 Pro scored the lowest with averages of 1.39 and 1.98, respectively. These findings emphasize the need for further testing on how linguistic shifts impact LLM responses across various categories and improvement in these areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08755v1">Robust Heuristic Algorithm Design with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      We posit that we can generate more robust and performant heuristics if we augment approaches using LLMs for heuristic design with tools that explain why heuristics underperform and suggestions about how to fix them. We find even simple ideas that (1) expose the LLM to instances where the heuristic underperforms; (2) explain why they occur; and (3) specialize design to regions in the input space, can produce more robust algorithms compared to existing techniques~ -- ~the heuristics we produce have a $\sim28\times$ better worst-case performance compared to FunSearch, improve average performance, and maintain the runtime.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08741v1">Coordinates from Context: Using LLMs to Ground Complex Location References</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Under review at ARR
    </div>
    <details class="paper-abstract">
      Geocoding is the task of linking a location reference to an actual geographic location and is essential for many downstream analyses of unstructured text. In this paper, we explore the challenging setting of geocoding compositional location references. Building on recent work demonstrating LLMs' abilities to reason over geospatial data, we evaluate LLMs' geospatial knowledge versus reasoning skills relevant to our task. Based on these insights, we propose an LLM-based strategy for geocoding compositional location references. We show that our approach improves performance for the task and that a relatively small fine-tuned LLM can achieve comparable performance with much larger off-the-shelf models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06594v2">Do Internal Layers of LLMs Reveal Patterns for Jailbreak Detection?</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Jailbreaking large language models (LLMs) has emerged as a pressing concern with the increasing prevalence and accessibility of conversational LLMs. Adversarial users often exploit these models through carefully engineered prompts to elicit restricted or sensitive outputs, a strategy widely referred to as jailbreaking. While numerous defense mechanisms have been proposed, attackers continuously develop novel prompting techniques, and no existing model can be considered fully resistant. In this study, we investigate the jailbreak phenomenon by examining the internal representations of LLMs, with a focus on how hidden layers respond to jailbreak versus benign prompts. Specifically, we analyze the open-source LLM GPT-J and the state-space model Mamba2, presenting preliminary findings that highlight distinct layer-wise behaviors. Our results suggest promising directions for further research on leveraging internal model dynamics for robust jailbreak detection and defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08710v1">Thinking Longer, Not Always Smarter: Evaluating LLM Capabilities in Hierarchical Legal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 21 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Case-based reasoning is a cornerstone of U.S. legal practice, requiring professionals to argue about a current case by drawing analogies to and distinguishing from past precedents. While Large Language Models (LLMs) have shown remarkable capabilities, their proficiency in this complex, nuanced form of reasoning needs further investigation. We propose a formal framework that decomposes the process of identifying significant distinctions between cases into three-stage reasoning tasks. Our framework models cases using factual predicates called factors, organizes them into a legal knowledge hierarchy, and defines verifiable rules for identifying distinctions, analyzing their argumentative support, and evaluating their significance. Through comprehensive evaluation of modern reasoning LLMs, we reveal a paradox: while models achieve high accuracy on surface-level reasoning (Task 1), performance degrades on hierarchical reasoning (Task 2: 64.82%-92.09%) and collapses on integrated analysis (Task 3: 11.46%-33.99%). Most strikingly, we find that models consistently expend more computational resources on incorrect responses than correct ones, suggesting that "thinking longer" does not always mean "thinking smarter." Our work provides a methodology for fine-grained analysis of LLM reasoning capabilities in complex domains and reveals fundamental limitations that must be addressed for robust and trustworthy legal AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08705v1">ConPoSe: LLM-Guided Contact Point Selection for Scalable Cooperative Object Pushing</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Object transportation in cluttered environments is a fundamental task in various domains, including domestic service and warehouse logistics. In cooperative object transport, multiple robots must coordinate to move objects that are too large for a single robot. One transport strategy is pushing, which only requires simple robots. However, careful selection of robot-object contact points is necessary to push the object along a preplanned path. Although this selection can be solved analytically, the solution space grows combinatorially with the number of robots and object size, limiting scalability. Inspired by how humans rely on common-sense reasoning for cooperative transport, we propose combining the reasoning capabilities of Large Language Models with local search to select suitable contact points. Our LLM-guided local search method for contact point selection, ConPoSe, successfully selects contact points for a variety of shapes, including cuboids, cylinders, and T-shapes. We demonstrate that ConPoSe scales better with the number of robots and object size than the analytical approach, and also outperforms pure LLM-based selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08664v1">Faver: Boosting LLM-based RTL Generation with Function Abstracted Verifiable Middleware</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      LLM-based RTL generation is an interesting research direction, as it holds the potential to liberate the least automated stage in the current chip design. However, due to the substantial semantic gap between high-level specifications and RTL, coupled with limited training data, existing models struggle with generation accuracy. Drawing on human experience, design with verification helps improving accuracy. However, as the RTL testbench data are even more scarce, it is not friendly for LLMs. Although LLMs excel at higher-level languages like Python/C, they have a huge semantic gap from RTL. When implementing the same functionality, Python/C code and hardware code differ significantly in the spatiotemporal granularity, requiring the LLM not only to consider high-level functional semantics but also to ensure the low-level details align with the circuit code. It is not an easy task. In this paper, we propose a function abstracted verifiable middleware (Faver) that streamlines RTL verification in LLM-based workflows. By mixing LLM-friendly code structures with a rule-based template, Faver decouples the details of circuit verification, allowing the LLM to focus on the functionality itself. In our experiments on the SFT model and open-source models, Faver improved the model's generation accuracy by up to 14%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08663v1">A Novel Framework for Augmenting Rating Scale Tests with LLM-Scored Text Data</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Psychological assessments typically rely on structured rating scales, which cannot incorporate the rich nuance of a respondent's natural language. This study leverages recent LLM advances to harness qualitative data within a novel conceptual framework, combining LLM-scored text and traditional rating-scale items to create an augmented test. We demonstrate this approach using depression as a case study, developing and assessing the framework on a real-world sample of upper secondary students (n=693) and corresponding synthetic dataset (n=3,000). On held-out test sets, augmented tests achieved statistically significant improvements in measurement precision and accuracy. The information gain from the LLM items was equivalent to adding between 6.3 (real data) and 16.0 (synthetic data) items to the original 19-item test. Our approach marks a conceptual shift in automated scoring that bypasses its typical bottlenecks: instead of relying on pre-labelled data or complex expert-created rubrics, we empirically select the most informative LLM scoring instructions based on calculations of item information. This framework provides a scalable approach for leveraging the growing stream of transcribed text to enhance traditional psychometric measures, and we discuss its potential utility in clinical health and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08640v1">Automating Android Build Repair: Bridging the Reasoning-Execution Gap in LLM Agents with Domain-Specific Tools</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Android is the largest mobile platform, yet automatically building applications remains a practical challenge. While Large Language Models (LLMs) show promise for code repair, their use for fixing Android build errors remains underexplored. To address this gap, we first introduce AndroidBuildBench, a benchmark of 1,019 build failures curated from the commit histories of 43 open-source Android projects. Each problem is paired with a verified solution from a subsequent commit, ensuring that fixes are feasible. Second, we propose GradleFixer, an LLM agent with domain-specific tools for inspecting and manipulating the Gradle build environment. GradleFixer achieves a resolve rate of 81.4% (pass@1), significantly outperforming a state-of-the-art coding agent that relies on a general-purpose shell. GradleFixer's success suggests that while LLMs possess the high-level knowledge to solve these failures, they struggle to translate this knowledge into effective low-level actions using a general-purpose shell. We demonstrate the effectiveness of a strategy we term Tool Bridging, which replaces general-purpose shell commands with domain-aware abstractions. We hypothesize this approach works through two mechanisms: 1) it provides tools in an API-like format that LLMs use more reliably, and 2) it constrains the action space to relevant operations. This approach bridges the gap between the model's high-level reasoning and effective low-level execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08572v1">BLAZER: Bootstrapping LLM-based Manipulation Agents with Zero-Shot Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 11 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Scaling data and models has played a pivotal role in the remarkable progress of computer vision and language. Inspired by these domains, recent efforts in robotics have similarly focused on scaling both data and model size to develop more generalizable and robust policies. However, unlike vision and language, robotics lacks access to internet-scale demonstrations across diverse robotic tasks and environments. As a result, the scale of existing datasets typically suffers from the need for manual data collection and curation. To address this problem, here we propose BLAZER, a framework that learns manipulation policies from automatically generated training data. We build on the zero-shot capabilities of LLM planners and automatically generate demonstrations for diverse manipulation tasks in simulation. Successful examples are then used to finetune an LLM and to improve its planning capabilities without human supervision. Notably, while BLAZER training requires access to the simulator's state, we demonstrate direct transfer of acquired skills to sensor-based manipulation. Through extensive experiments, we show BLAZER to significantly improve zero-shot manipulation in both simulated and real environments. Moreover, BLAZER improves on tasks outside of its training pool and enables downscaling of LLM models. Our code and data will be made publicly available on the project page.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03438v3">BFS-Prover: Scalable Best-First Tree Search for LLM-based Automatic Theorem Proving</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have spurred growing interest in automatic theorem proving using Lean4, where effective tree search methods are crucial for navigating the underlying large proof search spaces. While the existing approaches primarily rely on value functions and/or Monte Carlo Tree Search (MCTS), the potential of simpler methods like Best-First Tree Search (BFS) remains underexplored. In this paper, we investigate whether BFS can achieve competitive performance in large-scale theorem proving tasks. We present BFS-Prover, a scalable expert iteration framework, featuring three key innovations. First, we implement strategic data filtering at each expert iteration round, excluding problems solvable via beam search node expansion to focus on harder cases. Second, we improve the sample efficiency of BFS through Direct Preference Optimization (DPO) applied to state-tactic pairs automatically annotated with compiler error feedback, refining the LLM's policy to prioritize productive expansions. Third, we employ length normalization in BFS to encourage exploration of deeper proof paths. BFS-Prover achieves a state-of-the-art score of $72.95\%$ on the MiniF2F test set and therefore challenges the perceived necessity of complex tree search methods, demonstrating that BFS can achieve competitive performance when properly scaled. To facilitate further research and development in this area, we have open-sourced our model at https://huggingface.co/ByteDance-Seed/BFS-Prover-V1-7B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08544v1">SPAD: Specialized Prefill and Decode Hardware for Disaggregated LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained popularity in recent years, driving up the demand for inference. LLM inference is composed of two phases with distinct characteristics: a compute-bound prefill phase followed by a memory-bound decode phase. To efficiently serve LLMs, prior work proposes prefill-decode disaggregation to run each phase on separate hardware. However, existing hardware poorly matches the different requirements of each phase. Current datacenter GPUs and TPUs follow a more-is-better design philosophy that maximizes compute and memory resources, causing memory bandwidth underutilization in the prefill phase and compute underutilization in the decode phase. Such underutilization directly translates into increased serving costs. This paper proposes SPAD (Specialized Prefill and Decode hardware), adopting a less-is-more methodology to design specialized chips tailored to the distinct characteristics of prefill and decode phases. The proposed Prefill Chips have larger systolic arrays and use cost-effective GDDR memory, whereas the proposed Decode Chips retain high memory bandwidth but reduce compute capacity. Compared to modeled H100s, simulations show that the proposed Prefill Chips deliver 8% higher prefill performance on average at 52% lower hardware cost, while the proposed Decode Chips achieve 97% of the decode performance with 28% lower TDP. End-to-end simulations on production traces show that SPAD reduces hardware cost by 19%-41% and TDP by 2%-17% compared to modeled baseline clusters while offering the same performance. Even when models and workloads change, SPAD can reallocate either type of chip to run either phase and still achieve 11%-43% lower hardware costs, demonstrating the longevity of the SPAD design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08517v1">CaRT: Teaching LLM Agents to Know When They Know Enough</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Many tasks require learned models to strategically gather relevant information over multiple rounds of interaction before actually acting on a task. Strategic information gathering requires models to know not only how to effectively acquire information, but also when to stop gathering information and make a decision, in order to avoid overthinking or getting derailed when acting. In this paper, we formalize this problem and introduce Counterfactuals and Reasoning for Termination (CaRT), an approach for teaching LLMs when to stop seeking information. To appropriately learn when to terminate, CaRT fine-tunes LLMs using counterfactual pairs of trajectories, one where termination is appropriate and a minimally modified version of the same trajectory where it is not. It trains the LLM to explain the rationale for the termination decision in either case via verbal reasoning, and imbues this capability into the base LLM via fine-tuning. We instantiate CaRT in two domains: interactive medical diagnosis and math problem solving. In both domains, we find that CaRT improves the efficiency of information gathering and task success rate compared to other fine-tuning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06493v2">Scaling up Multi-Turn Off-Policy RL and Multi-Agent Tree Search for LLM Step-Provers</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into automated theorem proving has shown immense promise, yet is fundamentally constrained by challenges in scaling up both training-time reinforcement learning (RL) and inference-time compute. This paper introduces \texttt{BFS-Prover-V2}, a system designed to address this dual scaling problem. We present two primary innovations. The first is a novel multi-turn off-policy RL framework for continually improving the performance of LLM step-prover at training time. This framework, inspired by the principles of AlphaZero, utilizes a multi-stage expert iteration pipeline featuring adaptive tactic-level data filtering and periodic retraining to surmount the performance plateaus that typically curtail long-term RL in LLM-based agents. The second innovation is a planner-enhanced multi-agent search architecture that scales reasoning capabilities at inference time. This architecture employs a general reasoning model as a high-level planner to iteratively decompose complex theorems into a sequence of simpler subgoals. This hierarchical approach substantially reduces the search space, enabling a team of parallel prover agents to collaborate efficiently by leveraging a shared proof cache. We demonstrate that this dual approach to scaling yields state-of-the-art results on established formal mathematics benchmarks. \texttt{BFS-Prover-V2} achieves 95.08\% and 41.4\% on the MiniF2F and ProofNet test sets respectively. While demonstrated in the domain of formal mathematics, the RL and inference techniques presented in this work are of broader interest and may be applied to other domains requiring long-horizon multi-turn reasoning and complex search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20600v4">Multi-Turn Human-LLM Interaction Through the Lens of a Two-Way Intelligibility Protocol</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Multi-Turn Interactions in Large Language Models (MTI-LLM) Workshop at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Our interest is in the design of software systems involving a human-expert interacting -- using natural language -- with a large language model (LLM) on data analysis tasks. For complex problems, it is possible that LLMs can harness human expertise and creativity to find solutions that were otherwise elusive. On one level, this interaction takes place through multiple turns of prompts from the human and responses from the LLM. Here we investigate a more structured approach based on an abstract protocol described in [3] for interaction between agents. The protocol is motivated by a notion of "two-way intelligibility" and is modelled by a pair of communicating finite-state machines. We provide an implementation of the protocol, and provide empirical evidence of using the implementation to mediate interactions between an LLM and a human-agent in two areas of scientific interest (radiology and drug design). We conduct controlled experiments with a human proxy (a database), and uncontrolled experiments with human subjects. The results provide evidence in support of the protocol's capability of capturing one- and two-way intelligibility in human-LLM interaction; and for the utility of two-way intelligibility in the design of human-machine systems. Our code is available at https://github.com/karannb/interact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18085v2">Spiffy: Multiplying Diffusion LLM Acceleration via Lossless Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Original version uploaded on Sep 22, 2025. (v2): Extended Table 2 with additional analysis and referenced it in Sec 5.2
    </div>
    <details class="paper-abstract">
      Diffusion LLMs (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs (AR-LLMs) with the potential to operate at significantly higher token generation rates. However, currently available open-source dLLMs often generate at much lower rates, typically decoding only a single token at every denoising timestep in order to maximize output quality. We present Spiffy, a speculative decoding algorithm that accelerates dLLM inference by $\mathbf{2.8{-}3.1\times}$ while provably preserving the model's output distribution. This work addresses the unique challenges involved in applying ideas from speculative decoding of AR-LLMs to the dLLM setting. Spiffy proposes draft states by leveraging the dLLM's distribution itself in an auto-speculative manner. This approach is efficient and effective, and eliminates the overheads of training and running an independent draft model. To structure the candidate draft states, we propose a novel directed draft graph which is uniquely designed to take advantage of the bidirectional, block-wise nature of dLLM generation and can be verified in parallel by the dLLM. To further optimize the structure of these draft graphs, we introduce an efficient, offline calibration algorithm that procedurally determines high-quality graph configurations. These optimized draft graphs, enabling increased acceptance rates, lead to a significant boost in the overall speedup achieved by the system. Crucially, Spiffy is also complementary to other recent innovations in improving dLLM generation speeds such as KV-caching and multi-token unmasking. We demonstrate that when combined with such parallel decoding algorithms, Spiffy is able to effectively multiply the benefits of these methods leading to total speedups of up to $\mathbf{7.9\times}$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.11194v3">Evaluating LLMs' Mathematical Reasoning in Financial Document Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 26 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), excel in natural language understanding, but their capability for complex mathematical reasoning with an amalgamation of structured tables and unstructured text is uncertain. This study explores LLMs' mathematical reasoning on four financial tabular question-answering datasets: TATQA, FinQA, ConvFinQA, and Multihiertt. Through extensive experiments with various models and prompting techniques, we assess how LLMs adapt to complex tables and mathematical tasks. We focus on sensitivity to table complexity and performance variations with an increasing number of arithmetic reasoning steps. The results provide insights into LLMs' capabilities and limitations in handling complex mathematical scenarios for semi-structured tables. Ultimately, we introduce a novel prompting technique tailored to semi-structured documents, matching or outperforming other baselines in performance while providing a nuanced understanding of LLMs abilities for such a task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08481v1">Forecasting the Buzz: Enriching Hashtag Popularity Prediction with LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted to CIKM 2025
    </div>
    <details class="paper-abstract">
      Hashtag trends ignite campaigns, shift public opinion, and steer millions of dollars in advertising spend, yet forecasting which tag goes viral is elusive. Classical regressors digest surface features but ignore context, while large language models (LLMs) excel at contextual reasoning but misestimate numbers. We present BuzzProphet, a reasoning-augmented hashtag popularity prediction framework that (1) instructs an LLM to articulate a hashtag's topical virality, audience reach, and timing advantage; (2) utilizes these popularity-oriented rationales to enrich the input features; and (3) regresses on these inputs. To facilitate evaluation, we release HashView, a 7,532-hashtag benchmark curated from social media. Across diverse regressor-LLM combinations, BuzzProphet reduces RMSE by up to 2.8% and boosts correlation by 30% over baselines, while producing human-readable rationales. Results demonstrate that using LLMs as context reasoners rather than numeric predictors injects domain insight into tabular models, yielding an interpretable and deployable solution for social media trend forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14756v2">LLINBO: Trustworthy LLM-in-the-Loop Bayesian Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Bayesian optimization (BO) is a sequential decision-making tool widely used for optimizing expensive black-box functions. Recently, Large Language Models (LLMs) have shown remarkable adaptability in low-data regimes, making them promising tools for black-box optimization by leveraging contextual knowledge to propose high-quality query points. However, relying solely on LLMs as optimization agents introduces risks due to their lack of explicit surrogate modeling and calibrated uncertainty, as well as their inherently opaque internal mechanisms. This structural opacity makes it difficult to characterize or control the exploration-exploitation trade-off, ultimately undermining theoretical tractability and reliability. To address this, we propose LLINBO: LLM-in-the-Loop BO, a hybrid framework for BO that combines LLMs with statistical surrogate experts (e.g., Gaussian Processes (GP)). The core philosophy is to leverage contextual reasoning strengths of LLMs for early exploration, while relying on principled statistical models to guide efficient exploitation. Specifically, we introduce three mechanisms that enable this collaboration and establish their theoretical guarantees. We end the paper with a real-life proof-of-concept in the context of 3D printing. The code to reproduce the results can be found at https://github.com/UMDataScienceLab/LLM-in-the-Loop-BO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08439v1">xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 24 Pages, 4 Figures, 2 Tables
    </div>
    <details class="paper-abstract">
      Modern LLM deployments confront a widening cost-performance spectrum: premium models deliver strong reasoning but are expensive, while lightweight models are economical yet brittle on complex tasks. Static escalation rules and keyword heuristics under-utilize this spectrum and fail to adapt across task types. We present xRouter, a tool-calling-based routing system in which a learned router can either answer directly or invoke one or more external models. The router is trained end-to-end with reinforcement learning using an explicit, cost-aware reward that encodes cost-performance trade-offs, eliminating the need for hand-engineered routing rules. Our implementation encompasses the full reinforcement learning framework, including reward and cost accounting, as well as the deployment and evaluation pipelines. Across diverse benchmarks, xRouter achieves strong cost-performance trade-offs (e.g., substantial cost reductions at comparable task completion rates), and provides empirical insights into what reliably helps learned routing and what does not, ranging from model trainability to the difficulty of eliciting sophisticated orchestration behaviors in small open models. We hope these findings and our open implementation will serve as a practical substrate for advancing learned, cost-aware LLM orchestration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07231v2">Benchmarking LLM Causal Reasoning with Scientifically Validated Relationships</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Causal reasoning is fundamental for Large Language Models (LLMs) to understand genuine cause-and-effect relationships beyond pattern matching. Existing benchmarks suffer from critical limitations such as reliance on synthetic data and narrow domain coverage. We introduce a novel benchmark constructed from casually identified relationships extracted from top-tier economics and finance journals, drawing on rigorous methodologies including instrumental variables, difference-in-differences, and regression discontinuity designs. Our benchmark comprises 40,379 evaluation items covering five task types across domains such as health, environment, technology, law, and culture. Experimental results on eight state-of-the-art LLMs reveal substantial limitations, with the best model achieving only 57.6\% accuracy. Moreover, model scale does not consistently translate to superior performance, and even advanced reasoning models struggle with fundamental causal relationship identification. These findings underscore a critical gap between current LLM capabilities and demands of reliable causal reasoning in high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17320v2">AdaptiveK Sparse Autoencoders: Dynamic Sparsity Allocation for Interpretable LLM Representations</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Understanding the internal representations of large language models (LLMs) remains a central challenge for interpretability research. Sparse autoencoders (SAEs) offer a promising solution by decomposing activations into interpretable features, but existing approaches rely on fixed sparsity constraints that fail to account for input complexity. We propose AdaptiveK SAE (Adaptive Top K Sparse Autoencoders), a novel framework that dynamically adjusts sparsity levels based on the semantic complexity of each input. Leveraging linear probes, we demonstrate that context complexity is linearly encoded in LLM representations, and we use this signal to guide feature allocation during training. Experiments across ten language models (from 70M to 14B parameters) demonstrate that this complexity-driven adaptation significantly outperforms fixed-sparsity approaches on reconstruction fidelity, explained variance, cosine similarity and interpretability metrics while eliminating the computational burden of extensive hyperparameter tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03336v3">Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly tasked with invoking enterprise APIs, yet they routinely falter when near-duplicate tools vie for the same user intent or when required arguments are left underspecified. We introduce DiaFORGE (Dialogue Framework for Organic Response Generation & Evaluation), a disambiguation-centric, three-stage pipeline that (i) synthesizes persona-driven, multi-turn dialogues in which the assistant must distinguish among highly similar tools, (ii) performs supervised fine-tuning of open-source models with reasoning traces across 3B - 70B parameters, and (iii) evaluates real-world readiness via a dynamic suite that redeploys each model in a live agentic loop and reports end-to-end goal completion alongside conventional static metrics. On our dynamic benchmark DiaBENCH, models trained with DiaFORGE raise tool-invocation success by 27 pp over GPT-4o and by 49 pp over Claude-3.5-Sonnet, both under optimized prompting. To spur further research, we release an open corpus of 5000 production-grade enterprise API specifications paired with rigorously validated, disambiguation-focused dialogues, offering a practical blueprint for building reliable, enterprise-ready tool-calling agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20435v2">The Shape of Adversarial Influence: Characterizing LLM Latent Spaces with Persistent Homology</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Existing interpretability methods for Large Language Models (LLMs) often fall short by focusing on linear directions or isolated features, overlooking the high-dimensional, nonlinear, and relational geometry within model representations. This study focuses on how adversarial inputs systematically affect the internal representation spaces of LLMs, a topic which remains poorly understood. We propose persistent homology (PH), a tool from topological data analysis, as a principled framework to characterize the multi-scale dynamics within LLM activations. Using PH, we systematically analyze six state-of-the-art models under two distinct adversarial conditions, indirect prompt injection and backdoor fine-tuning, and identify a consistent topological signature of adversarial influence. Across architectures and model sizes, adversarial inputs induce ``topological compression'', where the latent space becomes structurally simpler, collapsing from varied, compact, small-scale features into fewer, dominant, and more dispersed large-scale ones. This topological signature is statistically robust across layers, highly discriminative, and provides interpretable insights into how adversarial effects emerge and propagate. By quantifying the shape of activations and neuronal information flow, our architecture-agnostic framework reveals fundamental invariants of representational change, offering a complementary perspective to existing interpretability methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08338v1">LLMs Reproduce Human Purchase Intent via Semantic Similarity Elicitation of Likert Ratings</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 28 pages, 35 figures
    </div>
    <details class="paper-abstract">
      Consumer research costs companies billions annually yet suffers from panel biases and limited scale. Large language models (LLMs) offer an alternative by simulating synthetic consumers, but produce unrealistic response distributions when asked directly for numerical ratings. We present semantic similarity rating (SSR), a method that elicits textual responses from LLMs and maps these to Likert distributions using embedding similarity to reference statements. Testing on an extensive dataset comprising 57 personal care product surveys conducted by a leading corporation in that market (9,300 human responses), SSR achieves 90% of human test-retest reliability while maintaining realistic response distributions (KS similarity > 0.85). Additionally, these synthetic respondents provide rich qualitative feedback explaining their ratings. This framework enables scalable consumer research simulations while preserving traditional survey metrics and interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20749v2">Matryoshka Pilot: Learning to Drive Black-Box LLMs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Despite the impressive generative abilities of black-box large language models (LLMs), their inherent opacity hinders further advancements in capabilities such as reasoning, planning, and personalization. Existing works aim to enhance LLM capabilities via domain-specific adaptation, which require additional training on accessible model parameters, an infeasible option for black-box LLMs. To address this challenge, we introduce Matryoshka Pilot (M-Pilot), a lightweight white-box LLM controller that guides a large-scale black-box LLM generator by decomposing complex tasks into a series of intermediate outputs. Specifically, we consider the black-box LLM as an environment, with M-Pilot serving as a policy to provide intermediate guidance through prompts for driving the black-box LLM. M-Pilot is trained to pivot the outputs of the black-box LLM aligning with preferences during iterative interaction, which enables controllable multi-turn generation and self-improvement in optimizing intermediate guidance. Empirical evaluations on diverse tasks demonstrate that our method effectively enhances the capabilities of black-box LLMs in complex, long-horizon tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16723v2">LLM Fingerprinting via Semantically Conditioned Watermarks</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Most LLM fingerprinting methods teach the model to respond to a few fixed queries with predefined atypical responses (keys). This memorization often does not survive common deployment steps such as finetuning or quantization, and such keys can be easily detected and filtered from LLM responses, ultimately breaking the fingerprint. To overcome these limitations we introduce LLM fingerprinting via semantically conditioned watermarks, replacing fixed query sets with a broad semantic domain, and replacing brittle atypical keys with a statistical watermarking signal diffused throughout each response. After teaching the model to watermark its responses only to prompts from a predetermined domain e.g., French language, the model owner can use queries from that domain to reliably detect the fingerprint and verify ownership. As we confirm in our thorough experimental evaluation, our fingerprint is both stealthy and robust to all common deployment scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08638v4">Examining Multilingual Embedding Models Cross-Lingually Through LLM-Generated Adversarial Examples</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 To appear in EMNLP2025 Findings
    </div>
    <details class="paper-abstract">
      The evaluation of cross-lingual semantic search models is often limited to existing datasets from tasks such as information retrieval and semantic textual similarity. We introduce Cross-Lingual Semantic Discrimination (CLSD), a lightweight evaluation task that requires only parallel sentences and a Large Language Model (LLM) to generate adversarial distractors. CLSD measures an embedding model's ability to rank the true parallel sentence above semantically misleading but lexically similar alternatives. As a case study, we construct CLSD datasets for German--French in the news domain. Our experiments show that models fine-tuned for retrieval tasks benefit from pivoting through English, whereas bitext mining models perform best in direct cross-lingual settings. A fine-grained similarity analysis further reveals that embedding models differ in their sensitivity to linguistic perturbations. We release our code and datasets under AGPL-3.0: https://github.com/impresso/cross_lingual_semantic_discrimination
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16567v3">Watch your steps: Dormant Adversarial Behaviors that Activate upon LLM Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Finetuning open-weight Large Language Models (LLMs) is standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets leads to predictable behaviors. In this paper, we demonstrate, for the first time, that an adversary can create compromised LLMs that are performant and benign, yet exhibit adversarial behaviors once finetuned by downstream users. To this end, we propose an attack, FAB (Finetuning-activated Adversarial Behaviors), which compromises an LLM via meta-learning techniques that simulate downstream finetuning, explicitly optimizing for the emergence of adversarial behaviors in the finetuned models. At the same time, the compromised LLM is regularized to retain general capabilities and to exhibit no adversarial behaviors prior to finetuning. As a result, when users finetune (e.g., instruction-tuning, distillation, DPO) the seemingly benign model on their own datasets, they unknowingly trigger its dormant adversarial behavior. We experimentally demonstrate the effectiveness of FAB across multiple LLMs and three commonly considered target behaviors: unsolicited advertising, jailbreakability, and over-refusal. We show that FAB-triggers are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler, post-training algorithm). Our findings challenge prevailing assumptions on the security of finetuning, revealing a critical attack vector.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19033v2">SelfRACG: Enabling LLMs to Self-Express and Retrieve for Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-09
      | 💬 Tsinghua&Xiaohongshu
    </div>
    <details class="paper-abstract">
      Existing retrieval-augmented code generation (RACG) methods typically use an external retrieval module to fetch semantically similar code snippets used for generating subsequent fragments. However, even for consecutive code fragments, the content often diverges due to logical progression, resulting in a content gap. This gap undermines the performance of current RACG methods, as \textit{external} retrieval modules based on content matching fail to infer the specific information need of LLMs to generate the next code fragment. Therefore, we propose \textbf{SelfRACG}, a novel paradigm that enables large language models (LLMs) to \textbf{Self}-express their information needs to enhance \textbf{RACG}. Specifically, SelfRACG includes an information need expression module and a two-stage information need-guided training strategy, which encourages LLMs to express their information need. Extensive experiments demonstrate that SelfRACG can retrieve external knowledge that better aligns with the LLM's own information needs, resulting in superior generation performance compared to vanilla RACG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10063v3">Hallucination Detection in LLMs with Topological Divergence on Attention Graphs</a></div>
    <div class="paper-meta">
      📅 2025-10-09
    </div>
    <details class="paper-abstract">
      Hallucination, i.e., generating factually incorrect content, remains a critical challenge for large language models (LLMs). We introduce TOHA, a TOpology-based HAllucination detector in the RAG setting, which leverages a topological divergence metric to quantify the structural properties of graphs induced by attention matrices. Examining the topological divergence between prompt and response subgraphs reveals consistent patterns: higher divergence values in specific attention heads correlate with hallucinated outputs, independent of the dataset. Extensive experiments - including evaluation on question answering and summarization tasks - show that our approach achieves state-of-the-art or competitive results on several benchmarks while requiring minimal annotated data and computational resources. Our findings suggest that analyzing the topological structure of attention matrices can serve as an efficient and robust indicator of factual reliability in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06096v2">The Alignment Auditor: A Bayesian Framework for Verifying and Refining LLM Objectives</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      The objectives that Large Language Models (LLMs) implicitly optimize remain dangerously opaque, making trustworthy alignment and auditing a grand challenge. While Inverse Reinforcement Learning (IRL) can infer reward functions from behaviour, existing approaches either produce a single, overconfident reward estimate or fail to address the fundamental ambiguity of the task (non-identifiability). This paper introduces a principled auditing framework that re-frames reward inference from a simple estimation task to a comprehensive process for verification. Our framework leverages Bayesian IRL to not only recover a distribution over objectives but to enable three critical audit capabilities: (i) Quantifying and systematically reducing non-identifiability by demonstrating posterior contraction over sequential rounds of evidence; (ii) Providing actionable, uncertainty-aware diagnostics that expose spurious shortcuts and identify out-of-distribution prompts where the inferred objective cannot be trusted; and (iii) Validating policy-level utility by showing that the refined, low-uncertainty reward can be used directly in RLHF to achieve training dynamics and toxicity reductions comparable to the ground-truth alignment process. Empirically, our framework successfully audits a detoxified LLM, yielding a well-calibrated and interpretable objective that strengthens alignment guarantees. Overall, this work provides a practical toolkit for auditors, safety teams, and regulators to verify what LLMs are truly trying to achieve, moving us toward more trustworthy and accountable AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11557v2">AC-LoRA: (Almost) Training-Free Access Control-Aware Multi-Modal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted in NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Corporate LLMs are gaining traction for efficient knowledge dissemination and management within organizations. However, as current LLMs are vulnerable to leaking sensitive information, it has proven difficult to apply them in settings where strict access control is necessary. To this end, we design AC-LoRA, an end-to-end system for access control-aware corporate LLM chatbots that maintains a strong information isolation guarantee. AC-LoRA maintains separate LoRA adapters for permissioned datasets, along with the document embedding they are finetuned on. AC-LoRA retrieves a precise set of LoRA adapters based on the similarity score with the user query and their permission. This similarity score is later used to merge the responses if more than one LoRA is retrieved, without requiring any additional training for LoRA routing. We provide an end-to-end prototype of AC-LoRA, evaluate it on two datasets, and show that AC-LoRA matches or even exceeds the performance of state-of-the-art LoRA mixing techniques while providing strong isolation guarantees. Furthermore, we show that AC-LoRA design can be directly applied to different modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18169v5">KunServe: Parameter-centric Memory Management for Efficient Memory Overloading Handling in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Serving LLMs with a cluster of GPUs is common nowadays, where the serving system must meet strict latency SLOs required by applications. However, the stateful nature of LLM serving requires maintaining huge states (i.e., KVCache) in limited GPU memory. Under spikes in real-world workloads, GPU memory can be easily throttled, leading to orders of magnitude higher response latency due to queuing introduced by waiting for KVCache to be reclaimed. Prior KVCache-centric approaches handle load throttling by dropping, migrating, or swapping KVCache. These methods fail to release sufficient memory quickly with requests still queued. This paper proposes the first parameter-centric approach to handling throttling by selectively dropping replicated parameters to instantly free memory for requests, based on an unnoticed observation that model parameters are commonly replicated across GPUs for serving LLMs. With additional memory, all requests can be served with a larger batch without queuing. To make the parameter-centric approach correct and efficient, we cooperatively execute requests on GPUs with a complete copy of parameters using pipeline parallelism, and derive an appropriate drop plan without unnecessary cooperation. We also design techniques to minimize the performance overhead due to pipeline parallelism with the execution patterns of requests under drop. Evaluations show that {\sys} reduces the tail TTFT of requests under throttling by up to 72.2 times compared to the state-of-the-art systems including Llumnix, vLLM and InferCept.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12845v4">ExLLM: Experience-Enhanced LLM Optimization for Molecular Design and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 10 pages, under review
    </div>
    <details class="paper-abstract">
      Molecular design involves an enormous and irregular search space, where traditional optimizers such as Bayesian optimization, genetic algorithms, and generative models struggle to leverage expert knowledge or handle complex feedback. Recently, LLMs have been used as optimizers, achieving promising results on benchmarks such as PMO. However, existing approaches rely only on prompting or extra training, without mechanisms to handle complex feedback or maintain scalable memory. In particular, the common practice of appending or summarizing experiences at every query leads to redundancy, degraded exploration, and ultimately poor final outcomes under large-scale iterative search. We introduce ExLLM (Experience-Enhanced LLM optimization), an LLM-as-optimizer framework with three components: (1) a compact, evolving experience snippet tailored to large discrete spaces that distills non-redundant cues and improves convergence at low cost; (2) a simple yet effective k-offspring scheme that widens exploration per call and reduces orchestration cost; and (3) a lightweight feedback adapter that normalizes objectives for selection while formatting constraints and expert hints for iteration. ExLLM sets new state-of-the-art results on PMO and generalizes strongly in our setup, it sets records on circle packing and stellarator design, and yields consistent gains across additional domains requiring only a task-description template and evaluation functions to transfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06780v1">Foundations of LLM Knowledge Materialization: Termination, Reproducibility, Robustness</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) encode substantial factual knowledge, yet measuring and systematizing this knowledge remains challenging. Converting it into structured format, for example through recursive extraction approaches such as the GPTKB methodology (Hu et al., 2025b), is still underexplored. Key open questions include whether such extraction can terminate, whether its outputs are reproducible, and how robust they are to variations. We systematically study LLM knowledge materialization using miniGPTKBs (domain-specific, tractable subcrawls), analyzing termination, reproducibility, and robustness across three categories of metrics: yield, lexical similarity, and semantic similarity. We experiment with four variations (seed, language, randomness, model) and three illustrative domains (from history, entertainment, and finance). Our findings show (i) high termination rates, though model-dependent; (ii) mixed reproducibility; and (iii) robustness that varies by perturbation type: high for seeds and temperature, lower for languages and models. These results suggest that LLM knowledge materialization can reliably surface core knowledge, while also revealing important limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06774v1">Adaptive LLM-Symbolic Reasoning via Dynamic Logical Solver Composition</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Neuro-symbolic NLP methods aim to leverage the complementary strengths of large language models and formal logical solvers. However, current approaches are mostly static in nature, i.e., the integration of a target solver is predetermined at design time, hindering the ability to employ diverse formal inference strategies. To address this, we introduce an adaptive, multi-paradigm, neuro-symbolic inference framework that: (1) automatically identifies formal reasoning strategies from problems expressed in natural language; and (2) dynamically selects and applies specialized formal logical solvers via autoformalization interfaces. Extensive experiments on individual and multi-paradigm reasoning tasks support the following conclusions: LLMs are effective at predicting the necessary formal reasoning strategies with an accuracy above 90 percent. This enables flexible integration with formal logical solvers, resulting in our framework outperforming competing baselines by 27 percent and 6 percent compared to GPT-4o and DeepSeek-V3.1, respectively. Moreover, adaptive reasoning can even positively impact pure LLM methods, yielding gains of 10, 5, and 6 percent on zero-shot, CoT, and symbolic CoT settings with GPT-4o. Finally, although smaller models struggle with adaptive neuro-symbolic reasoning, post-training offers a viable path to improvement. Overall, this work establishes the foundations for adaptive LLM-symbolic reasoning, offering a path forward for unifying material and formal inferences on heterogeneous reasoning challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06750v1">Gold-Switch: Training-Free Superposition of Slow- and Fast- Thinking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large Reasoning Models (LRMs) excel in structured tasks by emulating deliberate human reasoning but often suffer from overthinking, degrading performance and wasting resources. One possible baseline is to deploy both LLM and LRM, then route input by predicting whether it requires reasoning and may cause overthinking. However, deploying multiple models can be costly or impractical. We propose a superposed deployment strategy with a lightweight, training-free regulation to optimize inference by switching one model on and off. Instead of routing, we selectively unlearn from LRM at inference, scaling down computation while preserving reasoning. By analyzing the cumulative energy of singular values, we identify optimal low-rank projections to adjust reasoning just right.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06747v1">TWIST: Training-free and Label-free Short Text Clustering through Iterative Vector Updating with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      In this paper, we propose a training-free and label-free method for short text clustering that can be used on top of any existing embedder. In the context of customer-facing chatbots, companies are dealing with large amounts of user utterances that need to be clustered according to their intent. In these commercial settings, no labeled data is typically available, and the number of clusters is not known. Our method is based on iterative vector updating: it constructs sparse vectors based on representative texts, and then iteratively refines them through LLM guidance. Our method achieves comparable or superior results to state-of-the-art methods that use contrastive learning, but without assuming prior knowledge of clusters or labels. Experiments on diverse datasets and smaller LLMs show that our method is model agnostic and can be applied to any embedder, with relatively small LLMs, and different clustering methods. We also show that our method scales to large datasets, reducing the computational cost of the LLM. These low-resource, adaptable settings and the scalability of our method make it more aligned with real-world scenarios than existing clustering methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06743v1">Evaluating LLMs for Historical Document OCR: A Methodological Framework for Digital Humanities</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 The First Workshop on Natural Language Processing and Language Models for Digital Humanities (LM4DH 2025). RANLP 2025
    </div>
    <details class="paper-abstract">
      Digital humanities scholars increasingly use Large Language Models for historical document digitization, yet lack appropriate evaluation frameworks for LLM-based OCR. Traditional metrics fail to capture temporal biases and period-specific errors crucial for historical corpus creation. We present an evaluation methodology for LLM-based historical OCR, addressing contamination risks and systematic biases in diplomatic transcription. Using 18th-century Russian Civil font texts, we introduce novel metrics including Historical Character Preservation Rate (HCPR) and Archaic Insertion Rate (AIR), alongside protocols for contamination control and stability testing. We evaluate 12 multimodal LLMs, finding that Gemini and Qwen models outperform traditional OCR while exhibiting over-historicization: inserting archaic characters from incorrect historical periods. Post-OCR correction degrades rather than improves performance. Our methodology provides digital humanities practitioners with guidelines for model selection and quality assessment in historical corpus digitization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.15098v2">TextMine: Data, Evaluation Framework and Ontology-guided LLM Pipeline for Humanitarian Mine Action</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Humanitarian Mine Action (HMA) addresses the challenge of detecting and removing landmines from conflict regions. Much of the life-saving operational knowledge produced by HMA agencies is buried in unstructured reports, limiting the transferability of information between agencies. To address this issue, we propose TextMine: the first dataset, evaluation framework and ontology-guided large language model (LLM) pipeline for knowledge extraction in the HMA domain. TextMine structures HMA reports into (subject, relation, object)-triples, thus creating domain-specific knowledge. To ensure real-world relevance, we created the dataset in collaboration with Cambodian Mine Action Center (CMAC). We further introduce a bias-aware evaluation framework that combines human-annotated triples with an LLM-as-Judge protocol to mitigate position bias in reference-free scoring. Our experiments show that ontology-aligned prompts improve extraction accuracy by up to 44.2%, reduce hallucinations by 22.5%, and enhance format adherence by 20.9% compared to baseline models. We publicly release the dataset and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06732v1">Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as rerankers in information retrieval, yet their ranking behavior can be steered by small, natural-sounding prompts. To expose this vulnerability, we present Rank Anything First (RAF), a two-stage token optimization method that crafts concise textual perturbations to consistently promote a target item in LLM-generated rankings while remaining hard to detect. Stage 1 uses Greedy Coordinate Gradient to shortlist candidate tokens at the current position by combining the gradient of the rank-target with a readability score; Stage 2 evaluates those candidates under exact ranking and readability losses using an entropy-based dynamic weighting scheme, and selects a token via temperature-controlled sampling. RAF generates ranking-promoting prompts token-by-token, guided by dual objectives: maximizing ranking effectiveness and preserving linguistic naturalness. Experiments across multiple LLMs show that RAF significantly boosts the rank of target items using naturalistic language, with greater robustness than existing methods in both promoting target items and maintaining naturalness. These findings underscore a critical security implication: LLM-based reranking is inherently susceptible to adversarial manipulation, raising new challenges for the trustworthiness and robustness of modern retrieval systems. Our code is available at: https://github.com/glad-lab/RAF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06730v1">PTEB: Towards Robust Text Embedding Evaluation via Stochastic Paraphrasing at Evaluation Time with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Current evaluations of sentence embedding models typically rely on static test beds such as the Massive Text Embedding Benchmark (MTEB). While invaluable, repeated tuning on a fixed suite can inflate reported performance and obscure real-world robustness. We introduce the Paraphrasing Text Embedding Benchmark (PTEB), a dynamic protocol that stochastically generates meaning-preserving paraphrases at evaluation time and aggregates results across multiple runs. Using a cost-efficient LLM-based method grounded in semantic textual similarity gold ratings, we show that LLMs generate token-diverse but semantically preserving, paraphrases. Across 7 MTEB tasks, we validate our hypothesis that the performance of sentence encoders is sensitive to changes in token space even when semantics remain fixed. We also observe that smaller models are not disproportionately affected relative to larger ones. Our results are statistically robust over multiple runs and we extended our experiments to 3 multilingual datasets covering 10 languages. More generally, we aim to propose a new evaluation paradigm in NLP that relies less on static, pre-defined benchmarks but shifts towards dynamic, stochastic evaluation leveraging eval-time compute.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06727v1">Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      We study reinforcement learning (RL) fine-tuning of large language model (LLM) agents for long-horizon multi-turn tool use, where context length quickly becomes a fundamental bottleneck. Existing RL pipelines can suffer from degraded instruction following, excessive rollout costs, and most importantly, strict context limits. To address these challenges, we introduce summarization-based context management to training. In specific, it periodically compresses the tool using history by LLM-generated summaries that retain task-relevant information to keep a compact context while enabling the agent to scale beyond the fixed context window. Building on this formulation, we derive a policy gradient representation that seamlessly enables standard LLM RL infrastructures to optimize both tool-use behaviors as well as summarization strategies in an end-to-end fashion. We instantiate this framework with \underline{SU}mmarization augmented \underline{P}olicy \underline{O}ptimization (\texttt{SUPO}), an LLM RL algorithm that enables long-horizon training beyond a fixed context limit. Experiments on interactive function calling and searching tasks demonstrate that \texttt{SUPO} significantly improves the success rate while maintaining the same or even lower working context length compared to baselines. We also demonstrate that for complex searching tasks, \texttt{SUPO} can further improve the evaluation performance when scaling test-time maximum round of summarization beyond that of training time. Our results establish summarization-based context management as a principled and scalable approach for training RL agents beyond a fixed context length limit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06718v1">LLM Company Policies and Policy Implications in Software Organizations</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted at IEEE Software Special Issue on AIware in the Foundation Models Era
    </div>
    <details class="paper-abstract">
      The risks associated with adopting large language model (LLM) chatbots in software organizations highlight the need for clear policies. We examine how 11 companies create these policies and the factors that influence them, aiming to help managers safely integrate chatbots into development workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04152v2">Rethinking Multilingual Continual Pretraining: Data Mixing for Adapting LLMs Across Languages and Resources</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 COLM 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit significant disparities in performance across languages, primarily benefiting high-resource languages while marginalizing underrepresented ones. Continual Pretraining (CPT) has emerged as a promising approach to address this imbalance, although the relative effectiveness of monolingual, bilingual, and code-augmented data strategies remains unclear. This study systematically evaluates 36 CPT configurations involving three multilingual base models, across 30+ languages categorized as altruistic, selfish, and stagnant, spanning various resource levels. Our findings reveal three major insights: (1) Bilingual CPT improves multilingual classification but often causes language mixing issues during generation. (2) Including programming code data during CPT consistently enhances multilingual classification accuracy, particularly benefiting low-resource languages, but introduces a trade-off by slightly degrading generation quality. (3) Contrary to prior work, we observe substantial deviations from language classifications according to their impact on cross-lingual transfer: Languages classified as altruistic often negatively affect related languages, selfish languages show conditional and configuration-dependent behavior, and stagnant languages demonstrate surprising adaptability under certain CPT conditions. These nuanced interactions emphasize the complexity of multilingual representation learning, underscoring the importance of systematic studies on generalizable language classification to inform future multilingual CPT strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06708v1">AISysRev -- LLM-based Tool for Title-abstract Screening</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      Systematic reviews are a standard practice for summarizing the state of evidence in software engineering. Conducting systematic reviews is laborious, especially during the screening or study selection phase, where the number of papers can be overwhelming. During this phase, papers are assessed against inclusion and exclusion criteria based on their titles and abstracts. Recent research has demonstrated that large language models (LLMs) can perform title-abstract screening at a level comparable to that of a master's student. While LLMs cannot be fully trusted, they can help, for example, in Rapid Reviews, which try to expedite the review process. Building on recent research, we developed AiSysRev, an LLM-based screening tool implemented as a web application running in a Docker container. The tool accepts a CSV file containing paper titles and abstracts. Users specify inclusion and exclusion criteria. One can use multiple LLMs for screening via OpenRouter. AiSysRev supports both zero-shot and few-shot screening, and also allows for manual screening through interfaces that display LLM results as guidance for human reviewers.We conducted a trial study with 137 papers using the tool. Our findings indicate that papers can be classified into four categories: Easy Includes, Easy Excludes, Boundary Includes, and Boundary Excludes. The Boundary cases, where LLMs are prone to errors, highlight the need for human intervention. While LLMs do not replace human judgment in systematic reviews, they can significantly reduce the burden of assessing large volumes of scientific literature. Video: https://www.youtube.com/watch?v=jVbEj4Y4tQI Tool: https://github.com/EvoTestOps/AISysRev
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25033v2">VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Few-shot learning (FSL) aims to recognize novel concepts from only a few labeled support samples. Recent studies enhance support features by incorporating additional semantic information or designing complex semantic fusion modules. However, they still suffer from hallucinating semantics that contradict the visual evidence due to the lack of grounding in actual instances, resulting in noisy guidance and costly corrections. To address these issues, we propose a novel framework, bridging Vision and Text with LLMs for Few-Shot Learning (VT-FSL), which constructs precise cross-modal prompts conditioned on Large Language Models (LLMs) and support images, seamlessly integrating them through a geometry-aware alignment. It mainly consists of Cross-modal Iterative Prompting (CIP) and Cross-modal Geometric Alignment (CGA). Specifically, the CIP conditions an LLM on both class names and support images to generate precise class descriptions iteratively in a single structured reasoning pass. These descriptions not only enrich the semantic understanding of novel classes but also enable the zero-shot synthesis of semantically consistent images. The descriptions and synthetic images act respectively as complementary textual and visual prompts, providing high-level class semantics and low-level intra-class diversity to compensate for limited support data. Furthermore, the CGA jointly aligns the fused textual, support, and synthetic visual representations by minimizing the kernelized volume of the 3-dimensional parallelotope they span. It captures global and nonlinear relationships among all representations, enabling structured and consistent multimodal integration. The proposed VT-FSL method establishes new state-of-the-art performance across ten diverse benchmarks, including standard, cross-domain, and fine-grained few-shot learning scenarios. Code is available at https://github.com/peacelwh/VT-FSL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06695v1">Learning to Rewrite Prompts for Bootstrapping LLMs on Downstream Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      In recent years, the growing interest in Large Language Models (LLMs) has significantly advanced prompt engineering, transitioning from manual design to model-based optimization. Prompts for LLMs generally comprise two components: the \textit{instruction}, which defines the task or objective, and the \textit{input}, which is tailored to the instruction type. In natural language generation (NLG) tasks such as machine translation, the \textit{input} component is particularly critical, while the \textit{instruction} component tends to be concise. Existing prompt engineering methods primarily focus on optimizing the \textit{instruction} component for general tasks, often requiring large-parameter LLMs as auxiliary tools. However, these approaches exhibit limited applicability for tasks like machine translation, where the \textit{input} component plays a more pivotal role. To address this limitation, this paper introduces a novel prompt optimization method specifically designed for machine translation tasks. The proposed approach employs a small-parameter model trained using a back-translation-based strategy, significantly reducing training overhead for single-task optimization while delivering highly effective performance. With certain adaptations, this method can also be extended to other downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06674v1">Agent-in-the-Loop: A Data Flywheel for Continuous Improvement in LLM-based Customer Support</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 EMNLP 2025 Industry Track submission (Paper #305). Preprint. Main text within the 7-page industry limit (references/appendices excluded). Contains multiple figures and tables
    </div>
    <details class="paper-abstract">
      We introduce an Agent-in-the-Loop (AITL) framework that implements a continuous data flywheel for iteratively improving an LLM-based customer support system. Unlike standard offline approaches that rely on batch annotations, AITL integrates four key types of annotations directly into live customer operations: (1) pairwise response preferences, (2) agent adoption and rationales, (3) knowledge relevance checks, and (4) identification of missing knowledge. These feedback signals seamlessly feed back into models' updates, reducing retraining cycles from months to weeks. Our production pilot involving US-based customer support agents demonstrated significant improvements in retrieval accuracy (+11.7% recall@75, +14.8% precision@8), generation quality (+8.4% helpfulness) and agent adoption rates (+4.5%). These results underscore the effectiveness of embedding human feedback loops directly into operational workflows to continuously refine LLM-based customer support system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.05154v2">Can AI Truly Represent Your Voice in Deliberations? A Comprehensive Study of Large-Scale Opinion Aggregation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large-scale public deliberations generate thousands of free-form contributions that must be synthesized into representative and neutral summaries for policy use. While LLMs have been shown as a promising tool to generate summaries for large-scale deliberations, they also risk underrepresenting minority perspectives and exhibiting bias with respect to the input order, raising fairness concerns in high-stakes contexts. Studying and fixing these issues requires a comprehensive evaluation at a large scale, yet current practice often relies on LLMs as judges, which show weak alignment with human judgments. To address this, we present DeliberationBank, a large-scale human-grounded dataset with (1) opinion data spanning ten deliberation questions created by 3,000 participants and (2) summary judgment data annotated by 4,500 participants across four dimensions (representativeness, informativeness, neutrality, policy approval). Using these datasets, we train DeliberationJudge, a fine-tuned DeBERTa model that can rate deliberation summaries from individual perspectives. DeliberationJudge is more efficient and more aligned with human judgements compared to a wide range of LLM judges. With DeliberationJudge, we evaluate 18 LLMs and reveal persistent weaknesses in deliberation summarization, especially underrepresentation of minority positions. Our framework provides a scalable and reliable way to evaluate deliberation summarization, helping ensure AI systems are more representative and equitable for policymaking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01698v3">TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted for publication at The Workshop on AI for Music, Neural Information Processing Systems (NeurIPS-AI4Music)
    </div>
    <details class="paper-abstract">
      While the recent developments in large language models (LLMs) have successfully enabled generative recommenders with natural language interactions, their recommendation behavior is limited, leaving other simpler yet crucial components such as metadata or attribute filtering underutilized in the system. We propose an LLM-based music recommendation system with tool calling to serve as a unified retrieval-reranking pipeline. Our system positions an LLM as an end-to-end recommendation system that interprets user intent, plans tool invocations, and orchestrates specialized components: boolean filters (SQL), sparse retrieval (BM25), dense retrieval (embedding similarity), and generative retrieval (semantic IDs). Through tool planning, the system predicts which types of tools to use, their execution order, and the arguments needed to find music matching user preferences, supporting diverse modalities while seamlessly integrating multiple database filtering methods. We demonstrate that this unified tool-calling framework achieves competitive performance across diverse recommendation scenarios by selectively employing appropriate retrieval methods based on user queries, envisioning a new paradigm for conversational music recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06663v1">Automated Discovery of Test Oracles for Database Management Systems Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Since 2020, automated testing for Database Management Systems (DBMSs) has flourished, uncovering hundreds of bugs in widely-used systems. A cornerstone of these techniques is test oracle, which typically implements a mechanism to generate equivalent query pairs, thereby identifying bugs by checking the consistency between their results. However, while applying these oracles can be automated, their design remains a fundamentally manual endeavor. This paper explores the use of large language models (LLMs) to automate the discovery and instantiation of test oracles, addressing a long-standing bottleneck towards fully automated DBMS testing. Although LLMs demonstrate impressive creativity, they are prone to hallucinations that can produce numerous false positive bug reports. Furthermore, their significant monetary cost and latency mean that LLM invocations should be limited to ensure that bug detection is efficient and economical. To this end, we introduce Argus, a novel framework built upon the core concept of the Constrained Abstract Query - a SQL skeleton containing placeholders and their associated instantiation conditions (e.g., requiring a placeholder to be filled by a boolean column). Argus uses LLMs to generate pairs of these skeletons that are asserted to be semantically equivalent. This equivalence is then formally proven using a SQL equivalence solver to ensure soundness. Finally, the placeholders within the verified skeletons are instantiated with concrete, reusable SQL snippets that are also synthesized by LLMs to efficiently produce complex test cases. We implemented Argus and evaluated it on five extensively tested DBMSs, discovering 40 previously unknown bugs, 35 of which are logic bugs, with 36 confirmed and 26 already fixed by the developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06658v1">Can We Hide Machines in the Crowd? Quantifying Equivalence in LLM-in-the-loop Annotation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted at SIGIR-AP 2025
    </div>
    <details class="paper-abstract">
      Many evaluations of large language models (LLMs) in text annotation focus primarily on the correctness of the output, typically comparing model-generated labels to human-annotated ``ground truth'' using standard performance metrics. In contrast, our study moves beyond effectiveness alone. We aim to explore how labeling decisions -- by both humans and LLMs -- can be statistically evaluated across individuals. Rather than treating LLMs purely as annotation systems, we approach LLMs as an alternative annotation mechanism that may be capable of mimicking the subjective judgments made by humans. To assess this, we develop a statistical evaluation method based on Krippendorff's $\alpha$, paired bootstrapping, and the Two One-Sided t-Tests (TOST) equivalence test procedure. This evaluation method tests whether an LLM can blend into a group of human annotators without being distinguishable. We apply this approach to two datasets -- MovieLens 100K and PolitiFact -- and find that the LLM is statistically indistinguishable from a human annotator in the former ($p = 0.004$), but not in the latter ($p = 0.155$), highlighting task-dependent differences. It also enables early evaluation on a small sample of human data to inform whether LLMs are suitable for large-scale annotation in a given application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06657v1">LLM-Powered Nuanced Video Attribute Annotation for Enhanced Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 RecSys 2025 Industry Track
    </div>
    <details class="paper-abstract">
      This paper presents a case study on deploying Large Language Models (LLMs) as an advanced "annotation" mechanism to achieve nuanced content understanding (e.g., discerning content "vibe") at scale within a large-scale industrial short-form video recommendation system. Traditional machine learning classifiers for content understanding face protracted development cycles and a lack of deep, nuanced comprehension. The "LLM-as-annotators" approach addresses these by significantly shortening development times and enabling the annotation of subtle attributes. This work details an end-to-end workflow encompassing: (1) iterative definition and robust evaluation of target attributes, refined by offline metrics and online A/B testing; (2) scalable offline bulk annotation of video corpora using LLMs with multimodal features, optimized inference, and knowledge distillation for broad application; and (3) integration of these rich annotations into the online recommendation serving system, for example, through personalized restrict retrieval. Experimental results demonstrate the efficacy of this approach, with LLMs outperforming human raters in offline annotation quality for nuanced attributes and yielding significant improvements of user participation and satisfied consumption in online A/B tests. The study provides insights into designing and scaling production-level LLM pipelines for rich content evaluation, highlighting the adaptability and benefits of LLM-generated nuanced understanding for enhancing content discovery, user satisfaction, and the overall effectiveness of modern recommendation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02053v2">ProCut: LLM Prompt Compression via Attribution Estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      In large-scale industrial LLM systems, prompt templates often expand to thousands of tokens as teams iteratively incorporate sections such as task instructions, few-shot examples, and heuristic rules to enhance robustness and coverage. This expansion leads to bloated prompts that are difficult to maintain and incur significant inference latency and serving costs. To address this, we introduce Prompt Compression via Attribution Estimation (ProCut), a flexible, LLM-agnostic, training-free framework that compresses prompts through attribution analysis. ProCut segments prompt templates into semantically meaningful units, quantifies their impact on task performance, and prunes low-utility components. Through extensive experiments on five public benchmark datasets and real-world industrial prompts, we show that ProCut achieves substantial prompt size reductions (78% fewer tokens in production) while maintaining or even slightly improving task performance (up to 62% better than alternative methods). We further introduce an LLM-driven attribution estimator that reduces compression latency by over 50%, and demonstrate that ProCut integrates seamlessly with existing prompt-optimization frameworks to produce concise, high-performing prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18665v3">Membership Inference Attacks on LLM-based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 this paper is under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04072v2">Slow-Fast Policy Optimization: Reposition-Before-Update for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become central to enhancing reasoning in large language models (LLMs). Yet on-policy algorithms such as Group Relative Policy Optimization (GRPO) often suffer in early training: noisy gradients from low-quality rollouts lead to unstable updates and inefficient exploration. We introduce Slow-Fast Policy Optimization (SFPO), a simple yet efficient framework to address these limitations via decomposing each step into three stages: a short fast trajectory of inner steps on the same batch, a reposition mechanism to control off-policy drift, and a final slow correction. This reposition-before-update design preserves the objective and rollout process unchanged, making SFPO plug-compatible with existing policy-gradient pipelines. Extensive experiments demonstrate that SFPO consistently improves stability, reduces rollouts, and accelerates convergence of reasoning RL training. Specifically, it outperforms GRPO by up to 2.80 points in average on math reasoning benchmarks. It also achieves up to 4.93\texttimes{} fewer rollouts and an up to 4.19\texttimes{} reduction in wall-clock time to match GRPO's best accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12891v4">TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted by NeurIPS 2025 (Spotlight)
    </div>
    <details class="paper-abstract">
      Temporal reasoning is pivotal for Large Language Models (LLMs) to comprehend the real world. However, existing works neglect the real-world challenges for temporal reasoning: (1) intensive temporal information, (2) fast-changing event dynamics, and (3) complex temporal dependencies in social interactions. To bridge this gap, we propose a multi-level benchmark TIME, designed for temporal reasoning in real-world scenarios. TIME consists of 38,522 QA pairs, covering 3 levels with 11 fine-grained sub-tasks. This benchmark encompasses 3 sub-datasets reflecting different real-world challenges: TIME-Wiki, TIME-News, and TIME-Dial. We conduct extensive experiments on reasoning models and non-reasoning models. And we conducted an in-depth analysis of temporal reasoning performance across diverse real-world scenarios and tasks, and summarized the impact of test-time scaling on temporal reasoning capabilities. Additionally, we release TIME-Lite, a human-annotated subset to foster future research and standardized evaluation in temporal reasoning. The code is available at https://github.com/sylvain-wei/TIME , the dataset is available at https://huggingface.co/datasets/SylvainWei/TIME , and the project page link is https://sylvain-wei.github.io/TIME/ .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23580v2">LLM Hallucination Detection: HSAD</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 in Chinese language
    </div>
    <details class="paper-abstract">
      Although Large Language Models have demonstrated powerful capabilities in a wide range of tasks such as language understanding and code generation, the frequent occurrence of hallucinations during the generation process has become a significant impediment to their deployment in critical application scenarios. Current mainstream hallucination detection methods rely on factual consistency verification or static hidden layer features. The former is constrained by the scope of knowledge coverage, while the latter struggles to capture reasoning biases during the inference process. To address these issues, and inspired by signal analysis methods in cognitive neuroscience, this paper proposes a hallucination detection method based on the frequency-domain analysis of hidden layer temporal signals, named HSAD (\textbf{H}idden \textbf{S}ignal \textbf{A}nalysis-based \textbf{D}etection). First, by treating the LLM's reasoning process as a cognitive journey that unfolds over time, we propose modeling and simulating the human process of signal perception and discrimination in a deception-detection scenario through hidden layer temporal signals. Next, The Fast Fourier Transform is applied to map these temporal signals into the frequency domain to construct spectral features, which are used to capture anomalies that arise during the reasoning process; analysis experiments on these spectral features have proven the effectiveness of this approach. Finally, a hallucination detection algorithm is designed based on these spectral features to identify hallucinations in the generated content. By effectively combining the modeling of the reasoning process with frequency-domain feature extraction, the HSAD method overcomes the limitations of existing approaches in terms of knowledge coverage and the detection of reasoning biases, demonstrating higher detection accuracy and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06605v1">Reading Between the Lines: Towards Reliable Black-box LLM Fingerprinting via Zeroth-order Gradient Estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      The substantial investment required to develop Large Language Models (LLMs) makes them valuable intellectual property, raising significant concerns about copyright protection. LLM fingerprinting has emerged as a key technique to address this, which aims to verify a model's origin by extracting an intrinsic, unique signature (a "fingerprint") and comparing it to that of a source model to identify illicit copies. However, existing black-box fingerprinting methods often fail to generate distinctive LLM fingerprints. This ineffectiveness arises because black-box methods typically rely on model outputs, which lose critical information about the model's unique parameters due to the usage of non-linear functions. To address this, we first leverage Fisher Information Theory to formally demonstrate that the gradient of the model's input is a more informative feature for fingerprinting than the output. Based on this insight, we propose ZeroPrint, a novel method that approximates these information-rich gradients in a black-box setting using zeroth-order estimation. ZeroPrint overcomes the challenge of applying this to discrete text by simulating input perturbations via semantic-preserving word substitutions. This operation allows ZeroPrint to estimate the model's Jacobian matrix as a unique fingerprint. Experiments on the standard benchmark show ZeroPrint achieves a state-of-the-art effectiveness and robustness, significantly outperforming existing black-box methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06594v1">Do Internal Layers of LLMs Reveal Patterns for Jailbreak Detection?</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Jailbreaking large language models (LLMs) has emerged as a pressing concern with the increasing prevalence and accessibility of conversational LLMs. Adversarial users often exploit these models through carefully engineered prompts to elicit restricted or sensitive outputs, a strategy widely referred to as jailbreaking. While numerous defense mechanisms have been proposed, attackers continuously develop novel prompting techniques, and no existing model can be considered fully resistant. In this study, we investigate the jailbreak phenomenon by examining the internal representations of LLMs, with a focus on how hidden layers respond to jailbreak versus benign prompts. Specifically, we analyze the open-source LLM GPT-J and the state-space model Mamba2, presenting preliminary findings that highlight distinct layer-wise behaviors. Our results suggest promising directions for further research on leveraging internal model dynamics for robust jailbreak detection and defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02298v3">CAPO: Towards Enhancing LLM Reasoning through Generative Credit Assignment</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of Large Language Models (LLMs) by using rule-based binary feedback. However, current RLVR methods typically assign the same reward to every token. This coarse-grained feedback hampers precise credit assignment, making it hard for models to identify which reasoning steps lead to success or failure, and often results in suboptimal policies. Methods like PPO provide credit assignment by value estimation, but yield inaccurate and unverifiable signals due to limited sampling. On the other hand, methods using Process Reward Models can provide step-wise rewards but suffer from several key limitations: they require high-quality process supervision labels, the feedback is unreliable due to probabilistic reward modeling, and their application in online reinforcement learning (RL) is time-consuming. To overcome these limitations, we introduce a simple but efficient method-Credit Assignment Policy Optimization (CAPO). Instead of training auxiliary models, CAPO directly leverages an off-the-shelf, general-purpose LLM as a Generative Process Reward Model (LLM-as-GenPRM) to generate all step-wise critique by one pass only based on the correctness of the step itself, providing deterministic token-level credits to refine the tokens that were originally assigned identical rule-based rewards. To further enhance the accuracy and robustness, we employ voting mechanisms that scale with the number of generated critiques. Extensive experiments on various backbones like Llama and Qwen models show that CAPO consistently outperforms supervised learning-based and RL-based fine-tuning methods across four challenging mathematical benchmarks and three out-of-domain benchmarks. Further analysis shows that CAPO can help the model to foster the learning of correct reasoning pathways leading to correct answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08833v2">An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformation of Advanced Mathematical Problems</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 34 pages, 9 figures
    </div>
    <details class="paper-abstract">
      In this paper, we introduce a systematic framework beyond conventional method to assess LLMs' mathematical-reasoning robustness by stress-testing them on advanced math problems that are mathematically equivalent but with linguistic and parametric variation. These transformations allow us to measure the sensitivity of LLMs to non-mathematical perturbations, thereby enabling a more accurate evaluation of their mathematical reasoning capabilities. Using this new evaluation methodology, we created PutnamGAP, a new benchmark dataset with multiple mathematically-equivalent variations of competition-level math problems. With the new dataset, we evaluate multiple families of representative LLMs and examine their robustness. Across 18 commercial and open-source models we observe sharp performance degradation on the variants. OpenAI's flagship reasoning model, O3, scores 51.5% on the originals but drops by 4.7 percentage points on surface-renaming variants, and by 12.9 percentage points on parametric variants, while smaller models fare far worse. Overall, the results show that the proposed new evaluation methodology is effective for deepening our understanding of the robustness of LLMs and generating new insights for further improving their mathematical reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06565v1">Auto-Stega: An Agent-Driven System for Lifelong Strategy Evolution in LLM-Based Text Steganography</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 15 pages, 9 figures
    </div>
    <details class="paper-abstract">
      With the rapid progress of LLMs, high quality generative text has become widely available as a cover for text steganography. However, prevailing methods rely on hand-crafted or pre-specified strategies and struggle to balance efficiency, imperceptibility, and security, particularly at high embedding rates. Accordingly, we propose Auto-Stega, an agent-driven self-evolving framework that is the first to realize self-evolving steganographic strategies by automatically discovering, composing, and adapting strategies at inference time; the framework operates as a closed loop of generating, evaluating, summarizing, and updating that continually curates a structured strategy library and adapts across corpora, styles, and task constraints. A decoding LLM recovers the information under the shared strategy. To handle high embedding rates, we introduce PC-DNTE, a plug-and-play algorithm that maintains alignment with the base model's conditional distribution at high embedding rates, preserving imperceptibility while enhancing security. Experimental results demonstrate that at higher embedding rates Auto-Stega achieves superior performance with gains of 42.2\% in perplexity and 1.6\% in anti-steganalysis performance over SOTA methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06538v1">Auto-Prompt Ensemble for LLM Judge</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      We present a novel framework that improves the reliability of LLM judges by selectively augmenting LLM with auxiliary evaluation dimensions. Existing LLM judges often miss crucial evaluation dimensions because they fail to recognize the implicit standards underlying human assessments. To address this challenge, we propose the Auto-Prompt Ensemble (APE), an adaptive framework that automatically learns evaluation dimensions from its failure cases. APE incorporates a confidence-based ensemble mechanism to decide when to adopt the judgments from additional evaluation dimensions through a novel confidence estimation approach called Collective Confidence. Extensive experiments demonstrate that APE improves the reliability of LLM Judge across diverse standard benchmarks. For instance, APE enhances GPT-4o agreement rate on Reward Bench from 87.2% to 90.5% in the zero-shot setting. Overall, APE provides a principled approach for LLM Judge to leverage test-time computation, and bridge the evaluation gap between human and LLM judges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06530v1">From Description to Detection: LLM based Extendable O-RAN Compliant Blind DoS Detection in 5G and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      The quality and experience of mobile communication have significantly improved with the introduction of 5G, and these improvements are expected to continue beyond the 5G era. However, vulnerabilities in control-plane protocols, such as Radio Resource Control (RRC) and Non-Access Stratum (NAS), pose significant security threats, such as Blind Denial of Service (DoS) attacks. Despite the availability of existing anomaly detection methods that leverage rule-based systems or traditional machine learning methods, these methods have several limitations, including the need for extensive training data, predefined rules, and limited explainability. Addressing these challenges, we propose a novel anomaly detection framework that leverages the capabilities of Large Language Models (LLMs) in zero-shot mode with unordered data and short natural language attack descriptions within the Open Radio Access Network (O-RAN) architecture. We analyse robustness to prompt variation, demonstrate the practicality of automating the attack descriptions and show that detection quality relies on the semantic completeness of the description rather than its phrasing or length. We utilise an RRC/NAS dataset to evaluate the solution and provide an extensive comparison of open-source and proprietary LLM implementations to demonstrate superior performance in attack detection. We further validate the practicality of our framework within O-RAN's real-time constraints, illustrating its potential for detecting other Layer-3 attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15347v2">FlowKV: Enhancing Multi-Turn Conversational Coherence in LLMs via Isolated Key-Value Cache Management</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 NeurIPS 2025 Workshop on Multi-Turn Interactions in Large Language Models
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in multi-turn conversational applications, where the management of the Key-Value (KV) Cache presents a significant bottleneck. The linear growth of the KV Cache with dialogue history imposes substantial computational costs, and existing eviction strategies often degrade performance by repeatedly compressing early conversational context, leading to information loss and context forgetting. This paper introduces FlowKV, a novel \textbf{multi-turn isolation mechanism} for KV Cache management, which can be applied to any KV Cache compression method without training. FlowKV's core innovation is a multi-turn isolation mechanism that preserves the accumulated compressed KV cache from past turns. Compression is then strategically applied only to the newly generated KV pairs of the latest completed turn, effectively preventing the re-compression of older context and thereby mitigating catastrophic forgetting. Our results demonstrate that FlowKV consistently and significantly outperforms baseline strategies in maintaining instruction-following accuracy and user preference retention from 10.90\% to 75.40\%, particularly in later conversational turns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07626v1">LLM Unlearning Under the Microscope: A Full-Stack View on Methods and Metrics</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Machine unlearning for large language models (LLMs) aims to remove undesired data, knowledge, and behaviors (e.g., for safety, privacy, or copyright) while preserving useful model capabilities. Despite rapid progress over the past two years, research in LLM unlearning remains fragmented, with limited clarity on what constitutes effective unlearning and how it should be rigorously evaluated. In this work, we present a principled taxonomy of twelve recent stateful unlearning methods, grouped into three methodological families: divergence-driven optimization, representation misalignment, and rejection-based targeted unlearning. Building on this taxonomy, we revisit the evaluation of unlearning effectiveness (UE), utility retention (UT), and robustness (Rob), focusing on the WMDP benchmark. Our analysis shows that current evaluations, dominated by multiple-choice question (MCQ) accuracy, offer only a narrow perspective, often overstating success while overlooking the model's actual generation behavior. To address this gap, we introduce open question-answering (Open-QA) metrics that better capture generative performance and reveal the inherent UE-UT tradeoff across method families. Furthermore, we demonstrate that robustness requires finer-grained analysis: for example, vulnerabilities differ substantially between in-domain relearning and out-of-domain fine-tuning, even though both fall under model-level attacks. Through this study, we hope to deliver a full-stack revisit of LLM unlearning and actionable guidance for designing and evaluating future methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.03563v3">Say It Another Way: Auditing LLMs with a User-Grounded Automated Paraphrasing Framework</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are highly sensitive to subtle changes in prompt phrasing, posing challenges for reliable auditing. Prior methods often apply unconstrained prompt paraphrasing, which risk missing linguistic and demographic factors that shape authentic user interactions. We introduce AUGMENT (Automated User-Grounded Modeling and Evaluation of Natural Language Transformations), a framework for generating controlled paraphrases, grounded in user behaviors. AUGMENT leverages linguistically informed rules and enforces quality through checks on instruction adherence, semantic similarity, and realism, ensuring paraphrases are both reliable and meaningful for auditing. Through case studies on the BBQ and MMLU datasets, we show that controlled paraphrases uncover systematic weaknesses that remain obscured under unconstrained variation. These results highlight the value of the AUGMENT framework for reliable auditing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17694v2">Evaluating LLM-Generated Versus Human-Authored Responses in Role-Play Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 Accepted for publication at the 18th International Natural Language Generation Conference (INLG 2025). Revised version: improved image quality and minor corrections. No change to conclusions
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) in long-form, knowledge-grounded role-play dialogues remains challenging. This study compares LLM-generated and human-authored responses in multi-turn professional training simulations through human evaluation ($N=38$) and automated LLM-as-a-judge assessment. Human evaluation revealed significant degradation in LLM-generated response quality across turns, particularly in naturalness, context maintenance and overall quality, while human-authored responses progressively improved. In line with this finding, participants also indicated a consistent preference for human-authored dialogue. These human judgements were validated by our automated LLM-as-a-judge evaluation, where Gemini 2.0 Flash achieved strong alignment with human evaluators on both zero-shot pairwise preference and stochastic 6-shot construct ratings, confirming the widening quality gap between LLM and human responses over time. Our work contributes a multi-turn benchmark exposing LLM degradation in knowledge-grounded role-play dialogues and provides a validated hybrid evaluation framework to guide the reliable integration of LLMs in training simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07614v1">Traceability and Accountability in Role-Specialized Multi-Agent LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Sequential multi-agent systems built with large language models (LLMs) can automate complex software tasks, but they are hard to trust because errors quietly pass from one stage to the next. We study a traceable and accountable pipeline, meaning a system with clear roles, structured handoffs, and saved records that let us trace who did what at each step and assign blame when things go wrong. Our setting is a Planner -> Executor -> Critic pipeline. We evaluate eight configurations of three state-of-the-art LLMs on three benchmarks and analyze where errors start, how they spread, and how they can be fixed. Our results show: (1) adding a structured, accountable handoff between agents markedly improves accuracy and prevents the failures common in simple pipelines; (2) models have clear role-specific strengths and risks (e.g., steady planning vs. high-variance critiquing), which we quantify with repair and harm rates; and (3) accuracy-cost-latency trade-offs are task-dependent, with heterogeneous pipelines often the most efficient. Overall, we provide a practical, data-driven method for designing, tracing, and debugging reliable, predictable, and accountable multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07604v1">RustAssure: Differential Symbolic Testing for LLM-Transpiled C-to-Rust Code</a></div>
    <div class="paper-meta">
      📅 2025-10-08
      | 💬 13 pages to appear in Proceedings of ASE 2025
    </div>
    <details class="paper-abstract">
      Rust is a memory-safe programming language that significantly improves software security. Existing codebases written in unsafe memory languages, such as C, must first be transpiled to Rust to take advantage of Rust's improved safety guarantees. RustAssure presents a system that uses Large Language Models (LLMs) to automatically transpile existing C codebases to Rust. RustAssure uses prompt engineering techniques to maximize the chances of the LLM generating idiomatic and safe Rust code. Moreover, because LLMs often generate code with subtle bugs that can be missed under traditional unit or fuzz testing, RustAssure performs differential symbolic testing to establish the semantic similarity between the original C and LLM-transpiled Rust code. We evaluated RustAssure with five real-world applications and libraries, and showed that our system is able to generate compilable Rust functions for 89.8% of all C functions, of which 69.9% produced equivalent symbolic return values for both the C and Rust functions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04822v2">From Handwriting to Feedback: Evaluating VLMs and LLMs for AI-Powered Assessment in Indonesian Classrooms</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Despite rapid progress in vision-language and large language models (VLMs and LLMs), their effectiveness for AI-driven educational assessment in real-world, underrepresented classrooms remains largely unexplored. We evaluate state-of-the-art VLMs and LLMs on over 14K handwritten answers from grade-4 classrooms in Indonesia, covering Mathematics and English aligned with the local national curriculum. Unlike prior work on clean digital text, our dataset features naturally curly, diverse handwriting from real classrooms, posing realistic visual and linguistic challenges. Assessment tasks include grading and generating personalized Indonesian feedback guided by rubric-based evaluation. Results show that the VLM struggles with handwriting recognition, causing error propagation in LLM grading, yet LLM feedback remains pedagogically useful despite imperfect visual inputs, revealing limits in personalization and contextual relevance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00218v2">$\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07581v1">Expanding the Action Space of LLMs to Reason Beyond Language</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are powerful reasoners in natural language, but their actions are typically confined to outputting vocabulary tokens. As a result, interactions with external environments -- such as symbolic operators or simulators -- must be expressed through text in predefined formats, parsed, and routed to external interfaces. This overloads the model's language with both reasoning and control duties, and requires a hand-crafted parser, external to the LLM. To address this, we decouple environment interactions from language by internalizing them in an Expanded Action space (ExpA), beyond the vocabulary. The model starts reasoning in the default language environment, but may trigger routing actions and switch to an external environment at any time. From there, the model can only invoke environment-specific actions, receive feedback from the environment, and potentially route back to language as a result. To promote effective exploration of the expanded action space and new environments, we introduce ExpA Reinforcement Learning (EARL) with counterfactual policy optimization. On tasks requiring multi-turn interactions and contingent planning, EARL outperforms strong baselines with vocabulary-constrained actions. It performs robustly across calculator-based multi-task learning and, in the partially observed sorting problem, achieves perfect Sort-4 accuracy while self-discovering an efficient algorithm competitive with classical designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11878v3">LLMs Encode Harmfulness and Refusal Separately</a></div>
    <div class="paper-meta">
      📅 2025-10-08
    </div>
    <details class="paper-abstract">
      LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety.
    </details>
</div>
