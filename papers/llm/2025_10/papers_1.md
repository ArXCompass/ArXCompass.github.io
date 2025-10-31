# llm - 2025_10

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
- [Part 15](papers_15.md)
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)
- [Part 18](papers_18.md)
- [Part 19](papers_19.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26784v1">LLMs Process Lists With General Filter Heads</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Code and data at https://filter.baulab.info/
    </div>
    <details class="paper-abstract">
      We investigate the mechanisms underlying a range of list-processing tasks in LLMs, and we find that LLMs have learned to encode a compact, causal representation of a general filtering operation that mirrors the generic "filter" function of functional programming. Using causal mediation analysis on a diverse set of list-processing tasks, we find that a small number of attention heads, which we dub filter heads, encode a compact representation of the filtering predicate in their query states at certain tokens. We demonstrate that this predicate representation is general and portable: it can be extracted and reapplied to execute the same filtering operation on different collections, presented in different formats, languages, or even in tasks. However, we also identify situations where transformer LMs can exploit a different strategy for filtering: eagerly evaluating if an item satisfies the predicate and storing this intermediate result as a flag directly in the item representations. Our results reveal that transformer LMs can develop human-interpretable implementations of abstract computational operations that generalize in ways that are surprisingly similar to strategies used in traditional functional programming patterns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09391v2">Comparing human and LLM politeness strategies in free production</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 25 pages, 5 figures | EMNLP 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Polite speech poses a fundamental alignment challenge for large language models (LLMs). Humans deploy a rich repertoire of linguistic strategies to balance informational and social goals -- from positive approaches that build rapport (compliments, expressions of interest) to negative strategies that minimize imposition (hedging, indirectness). We investigate whether LLMs employ a similarly context-sensitive repertoire by comparing human and LLM responses in both constrained and open-ended production tasks. We find that larger models ($\ge$70B parameters) successfully replicate key preferences from the computational pragmatics literature, and human evaluators surprisingly prefer LLM-generated responses in open-ended contexts. However, further linguistic analyses reveal that models disproportionately rely on negative politeness strategies even in positive contexts, potentially leading to misinterpretations. While modern LLMs demonstrate an impressive handle on politeness strategies, these subtle differences raise important questions about pragmatic alignment in AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09205v3">Quality Over Quantity? LLM-Based Curation for a Data-Efficient Audio-Video Foundation Model</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 5 pages, 5 figures, 2 tables. Accepted at EUSIPCO 2025
    </div>
    <details class="paper-abstract">
      Integrating audio and visual data for training multimodal foundational models remains a challenge. The Audio-Video Vector Alignment (AVVA) framework addresses this by considering AV scene alignment beyond mere temporal synchronization, and leveraging Large Language Models (LLMs) for data curation. AVVA implements a scoring mechanism for selecting aligned training data segments. It integrates Whisper, a speech-based foundation model, for audio and DINOv2 for video analysis in a dual-encoder structure with contrastive learning on AV pairs. Evaluations on AudioCaps, VALOR, and VGGSound demonstrate the effectiveness of the proposed model architecture and data curation approach. AVVA achieves a significant improvement in top-k accuracies for video-to-audio retrieval on all datasets compared to DenseAV, while using only 192 hrs of curated training data. Furthermore, an ablation study indicates that the data curation process effectively trades data quality for data quantity, yielding increases in top-k retrieval accuracies on AudioCaps, VALOR, and VGGSound, compared to training on the full spectrum of uncurated data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14681v2">Massive Supervised Fine-tuning Experiments Reveal How Data, Layer, and Training Factors Shape LLM Alignment Quality</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Accepted to EMNLP 2025 (Main Conference). Models and evaluation results available at: https://github.com/llm-jp/massive-sft
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is a critical step in aligning large language models (LLMs) with human instructions and values, yet many aspects of SFT remain poorly understood. We trained a wide range of base models on a variety of datasets including code generation, mathematical reasoning, and general-domain tasks, resulting in 1,000+ SFT models under controlled conditions. We then identified the dataset properties that matter most and examined the layer-wise modifications introduced by SFT. Our findings reveal that some training-task synergies persist across all models while others vary substantially, emphasizing the importance of model-specific strategies. Moreover, we demonstrate that perplexity consistently predicts SFT effectiveness, often surpassing superficial similarity between the training data and the benchmark, and that mid-layer weight changes correlate most strongly with performance gains. We release these 1,000+ SFT models and benchmark results to accelerate further research. All resources are available at https://github.com/llm-jp/massive-sft.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26707v1">Value Drifts: Tracing Value Alignment During LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      As LLMs occupy an increasingly important role in society, they are more and more confronted with questions that require them not only to draw on their general knowledge but also to align with certain human value systems. Therefore, studying the alignment of LLMs with human values has become a crucial field of inquiry. Prior work, however, mostly focuses on evaluating the alignment of fully trained models, overlooking the training dynamics by which models learn to express human values. In this work, we investigate how and at which stage value alignment arises during the course of a model's post-training. Our analysis disentangles the effects of post-training algorithms and datasets, measuring both the magnitude and time of value drifts during training. Experimenting with Llama-3 and Qwen-3 models of different sizes and popular supervised fine-tuning (SFT) and preference optimization datasets and algorithms, we find that the SFT phase generally establishes a model's values, and subsequent preference optimization rarely re-aligns these values. Furthermore, using a synthetic preference dataset that enables controlled manipulation of values, we find that different preference optimization algorithms lead to different value alignment outcomes, even when preference data is held constant. Our findings provide actionable insights into how values are learned during post-training and help to inform data curation, as well as the selection of models and algorithms for preference optimization to improve model alignment to human values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23254v3">MemAscend: System Memory Optimization for SSD-Offloaded LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 16 pages, 21 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Owing to the huge success of generative artificial intelligence (AI), large language models (LLMs) have emerged as a core subclass, underpinning applications such as question answering, text generation, and code completion. While fine-tuning these models on domain-specific data can yield significant performance gains, it also poses daunting computational challenges, especially for researchers and small organizations with limited hardware resources. Although SSD offloading (i.e., ZeRO-Infinity) has emerged as a viable strategy to overcome the GPU memory barrier via leveraging both system memory (i.e., CPU DRAM) and storage space (i.e., solid-state devices, SSDs), its design primarily targets model-centric performance issues. As a result, key system-level issues, including system memory fragmentation, inefficient pinned buffer allocation, peak CPU usage spikes, and file system overhead, remain unaddressed, stifling scalability and inflating costs. Such an observation motivates this paper to introduce MemAscend, a framework that systematically tackles the underexplored system memory bottlenecks in SSD-offloaded LLM training, with a focus on resource-constrained environments. By streamlining pinned-memory allocation, eradicating fragmentation, and mitigating peak overhead, MemAscend reclaims a substantial system memory budget, enabling larger models, longer context windows, and higher batch sizes without exceeding modest hardware limits. Across diverse LLM benchmarks, MemAscend reduces peak system-memory consumption by an average of 55.7% compared with standard SSD offloading techniques, lowering the hardware barrier for fine-tuning and unlocking new possibilities for cost-effective large-scale training on limited-resource machines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03305v2">Analysis and Optimized CXL-Attached Memory Allocation for Long-Context LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 13 pages, 15 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The substantial memory requirements of Large Language Models (LLMs), particularly for long-context fine-tuning, have renewed interest in CPU offloading to augment limited GPU memory. However, as context lengths grow, relying on CPU memory for intermediate states introduces a significant bottleneck that can exhaust the capacity of mainstream client platforms. To address this limitation, this work investigates the effectiveness of Compute Express Link (CXL) add-in card (AIC) memory as an extension to CPU memory, enabling larger model sizes and longer context lengths during fine-tuning. Extensive benchmarking reveals two critical challenges. First, current deep learning frameworks such as PyTorch lack fine-grained, per-tensor control over NUMA memory allocation, exposing only coarse, process-level policies. Second, due to this lack of control, when the memory footprint of fine-tuning is offloaded across local DRAM and CXL-attached memory, naively placing optimizer data in higher-latency CXL leads to substantial slowdowns in the optimizer step (e.g., 4x once data exceeds 20M elements). To overcome these challenges, this work introduces a PyTorch extension that enables tensor-level system memory control and a CXL-aware memory allocator that pins latency-critical tensors in local DRAM while maximizing bandwidth by striping latency-tolerant tensors across one or more CXL devices. Evaluated on a real hardware setup with 7B and 12B models, 4K-32K contexts, and a single GPU, our approach recovers throughput to 97-99% of DRAM-only with a single AIC and approximately 100% with two AICs, delivering up to 21% improvement over naive interleaving while preserving DRAM-like DMA bandwidth for GPU transfers. These results show that carefully managed CXL-attached memory is a practical path to scaling long-context fine-tuning beyond DRAM limits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01543v2">Refine-n-Judge: Curating High-Quality Preference Chains for LLM-Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable progress through preference-based fine-tuning, which critically depends on the quality of the underlying training data. While human feedback is essential for improving data quality, it is costly and does not scale well. In this paper, we introduce Refine-n-Judge, an automated iterative approach that leverages a single LLM as both a refiner and a judge to enhance dataset quality. Unlike existing iterative refinement methods, Refine-n-Judge employs an LLM to both generate refinements and explicitly evaluate each improvement, ensuring that every iteration meaningfully enhances the dataset without requiring additional human annotation or a separate reward model. At each step, the LLM refines a response and judges whether the refinement is an improvement over the previous answer. This process continues until the LLM prefers the initial answer over the refinement, indicating no further improvements. This produces sequences of increasing quality, preference-labeled responses ideal for fine-tuning. We demonstrate the effectiveness of Refine-n-Judge across a range of public datasets spanning five corpora, targeting tasks such as coding, math, and conversation. Models (Llama 3.1-8B and Llama 3.3-70B) fine-tuned on Refine-n-Judge-enhanced datasets were preferred by LLM judges in over 74% of comparisons against models tuned on the original dataset by GPT-4. Additionally, we report performance gains: +5% on AlpacaEval and AlpacaEval 2.0, and +19% on MT-Bench. Our results indicate that Refine-n-Judge produces high-quality datasets and scalable model improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21257v2">CompoST: A Benchmark for Analyzing the Ability of LLMs To Compositionally Interpret Questions in a QALD Setting</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Research Track, 24th International Semantic Web Conference (ISWC 2025), November 2-6, 2025, Nara, Japan
    </div>
    <details class="paper-abstract">
      Language interpretation is a compositional process, in which the meaning of more complex linguistic structures is inferred from the meaning of their parts. Large language models possess remarkable language interpretation capabilities and have been successfully applied to interpret questions by mapping them to SPARQL queries. An open question is how systematic this interpretation process is. Toward this question, in this paper, we propose a benchmark for investigating to what extent the abilities of LLMs to interpret questions are actually compositional. For this, we generate three datasets of varying difficulty based on graph patterns in DBpedia, relying on Lemon lexica for verbalization. Our datasets are created in a very controlled fashion in order to test the ability of LLMs to interpret structurally complex questions, given that they have seen the atomic building blocks. This allows us to evaluate to what degree LLMs are able to interpret complex questions for which they "understand" the atomic parts. We conduct experiments with models of different sizes using both various prompt and few-shot optimization techniques as well as fine-tuning. Our results show that performance in terms of macro $F_1$ degrades from $0.45$ over $0.26$ down to $0.09$ with increasing deviation from the samples optimized on. Even when all necessary information was provided to the model in the input, the $F_1$ scores do not exceed $0.57$ for the dataset of lowest complexity. We thus conclude that LLMs struggle to systematically and compositionally interpret questions and map them into SPARQL queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26641v1">All You Need for Object Detection: From Pixels, Points, and Prompts to Next-Gen Fusion and Multimodal LLMs/VLMs in Autonomous Vehicles</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Autonomous Vehicles (AVs) are transforming the future of transportation through advances in intelligent perception, decision-making, and control systems. However, their success is tied to one core capability, reliable object detection in complex and multimodal environments. While recent breakthroughs in Computer Vision (CV) and Artificial Intelligence (AI) have driven remarkable progress, the field still faces a critical challenge as knowledge remains fragmented across multimodal perception, contextual reasoning, and cooperative intelligence. This survey bridges that gap by delivering a forward-looking analysis of object detection in AVs, emphasizing emerging paradigms such as Vision-Language Models (VLMs), Large Language Models (LLMs), and Generative AI rather than re-examining outdated techniques. We begin by systematically reviewing the fundamental spectrum of AV sensors (camera, ultrasonic, LiDAR, and Radar) and their fusion strategies, highlighting not only their capabilities and limitations in dynamic driving environments but also their potential to integrate with recent advances in LLM/VLM-driven perception frameworks. Next, we introduce a structured categorization of AV datasets that moves beyond simple collections, positioning ego-vehicle, infrastructure-based, and cooperative datasets (e.g., V2V, V2I, V2X, I2I), followed by a cross-analysis of data structures and characteristics. Ultimately, we analyze cutting-edge detection methodologies, ranging from 2D and 3D pipelines to hybrid sensor fusion, with particular attention to emerging transformer-driven approaches powered by Vision Transformers (ViTs), Large and Small Language Models (SLMs), and VLMs. By synthesizing these perspectives, our survey delivers a clear roadmap of current capabilities, open challenges, and future opportunities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.17197v2">SignalLLM: A General-Purpose LLM Agent Framework for Automated Signal Processing</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Modern signal processing (SP) pipelines, whether model-based or data-driven, often constrained by complex and fragmented workflow, rely heavily on expert knowledge and manual engineering, and struggle with adaptability and generalization under limited data. In contrast, Large Language Models (LLMs) offer strong reasoning capabilities, broad general-purpose knowledge, in-context learning, and cross-modal transfer abilities, positioning them as powerful tools for automating and generalizing SP workflows. Motivated by these potentials, we introduce SignalLLM, the first general-purpose LLM-based agent framework for general SP tasks. Unlike prior LLM-based SP approaches that are limited to narrow applications or tricky prompting, SignalLLM introduces a principled, modular architecture. It decomposes high-level SP goals into structured subtasks via in-context learning and domain-specific retrieval, followed by hierarchical planning through adaptive retrieval-augmented generation (RAG) and refinement; these subtasks are then executed through prompt-based reasoning, cross-modal reasoning, code synthesis, model invocation, or data-driven LLM-assisted modeling. Its generalizable design enables the flexible selection of problem solving strategies across different signal modalities, task types, and data conditions. We demonstrate the versatility and effectiveness of SignalLLM through five representative tasks in communication and sensing, such as radar target detection, human activity recognition, and text compression. Experimental results show superior performance over traditional and existing LLM-based methods, particularly in few-shot and zero-shot settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01369v2">Incentivizing LLMs to Self-Verify Their Answers</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable progress in complex reasoning tasks through both post-training and test-time scaling laws. While prevalent test-time scaling approaches are often realized by using external reward models to guide the model generation process, we find that only marginal gains can be acquired when scaling a model post-trained on specific reasoning tasks. We identify that the limited improvement stems from distribution discrepancies between the specific post-trained generator and the general reward model. To address this, we propose a framework that incentivizes LLMs to self-verify their own answers. By unifying answer generation and verification within a single reinforcement learning (RL) process, we train models that can effectively assess the correctness of their own solutions. The trained model can further scale its performance at inference time by verifying its generations, without the need for external verifiers. We train our self-verification models based on Qwen2.5-Math-7B and DeepSeek-R1-Distill-Qwen-1.5B, demonstrating their capabilities across varying reasoning context lengths. Experiments on multiple mathematical reasoning benchmarks show that our models can not only improve post-training performance but also enable effective test-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26546v1">WeaveRec: An LLM-Based Cross-Domain Sequential Recommendation Framework with Model Merging</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Cross-Domain Sequential Recommendation (CDSR) seeks to improve user preference modeling by transferring knowledge from multiple domains. Despite the progress made in CDSR, most existing methods rely on overlapping users or items to establish cross-domain correlations-a requirement that rarely holds in real-world settings. The advent of large language models (LLM) and model-merging techniques appears to overcome this limitation by unifying multi-domain data without explicit overlaps. Yet, our empirical study shows that naively training an LLM on combined domains-or simply merging several domain-specific LLMs-often degrades performance relative to a model trained solely on the target domain. To address these challenges, we first experimentally investigate the cause of suboptimal performance in LLM-based cross-domain recommendation and model merging. Building on these insights, we introduce WeaveRec, which cross-trains multiple LoRA modules with source and target domain data in a weaving fashion, and fuses them via model merging. WeaveRec can be extended to multi-source domain scenarios and notably does not introduce additional inference-time cost in terms of latency or memory. Furthermore, we provide a theoretical guarantee that WeaveRec can reduce the upper bound of the expected error in the target domain. Extensive experiments on single-source, multi-source, and cross-platform cross-domain recommendation scenarios validate that WeaveRec effectively mitigates performance degradation and consistently outperforms baseline approaches in real-world recommendation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15030v3">Collab-REC: An LLM-based Agentic Framework for Balancing Recommendations in Tourism</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      We propose Collab-REC, a multi-agent framework designed to counteract popularity bias and enhance diversity in tourism recommendations. In our setting, three LLM-based agents -- Personalization, Popularity, and Sustainability generate city suggestions from complementary perspectives. A non-LLM moderator then merges and refines these proposals via multi-round negotiation, ensuring each agent's viewpoint is incorporated while penalizing spurious or repeated responses. Experiments on European city queries show that Collab-REC improves diversity and overall relevance compared to a single-agent baseline, surfacing lesser-visited locales that often remain overlooked. This balanced, context-aware approach addresses over-tourism and better aligns with constraints provided by the user, highlighting the promise of multi-stakeholder collaboration in LLM-driven recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26510v1">LLMs as In-Context Meta-Learners for Model and Hyperparameter Selection</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 27 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Model and hyperparameter selection are critical but challenging in machine learning, typically requiring expert intuition or expensive automated search. We investigate whether large language models (LLMs) can act as in-context meta-learners for this task. By converting each dataset into interpretable metadata, we prompt an LLM to recommend both model families and hyperparameters. We study two prompting strategies: (1) a zero-shot mode relying solely on pretrained knowledge, and (2) a meta-informed mode augmented with examples of models and their performance on past tasks. Across synthetic and real-world benchmarks, we show that LLMs can exploit dataset metadata to recommend competitive models and hyperparameters without search, and that improvements from meta-informed prompting demonstrate their capacity for in-context meta-learning. These results highlight a promising new role for LLMs as lightweight, general-purpose assistants for model selection and hyperparameter optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08271v2">RecCocktail: A Generalizable and Efficient Framework for LLM-Based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success in recent years, owing to their impressive generalization capabilities and rich world knowledge. To capitalize on the potential of using LLMs as recommender systems, mainstream approaches typically focus on two paradigms. The first paradigm designs multi-domain or multi-task instruction data for generalizable recommendation, so as to align LLMs with general recommendation areas and deal with cold-start recommendation. The second paradigm focuses on enhancing domain-specific recommendation tasks, improving performance in warm recommendation scenarios. While most previous works treat these two paradigms separately, we argue that they have complementary advantages, and combining them can yield better results. In this paper, we propose a generalizable and efficient LLM-based recommendation framework RecCocktail. Our approach begins with fine-tuning a "base spirit" LoRA module using domain-general recommendation instruction data to align LLM with recommendation knowledge. Next, given users' behavior of a specific domain, we construct a domain-specific "ingredient" LoRA module. We then provide an entropy-guided adaptive merging method to mix the "base spirit" and the "ingredient" in the weight space. Please note that, RecCocktail combines the advantages of the existing two paradigms without introducing additional time or space overhead during the inference phase. Moreover, RecCocktail is efficient with plug and play, as the "base spirit" LoRA is trained only once, and any domain-specific "ingredient" can be efficiently mixed with only domain-specific fine-tuning. Extensive experiments on multiple datasets under both warm and cold-start recommendation scenarios validate the effectiveness and generality of the proposed RecCocktail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26494v1">Simulating and Experimenting with Social Media Mobilization Using LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Online social networks have transformed the ways in which political mobilization messages are disseminated, raising new questions about how peer influence operates at scale. Building on the landmark 61-million-person Facebook experiment \citep{bond201261}, we develop an agent-based simulation framework that integrates real U.S. Census demographic distributions, authentic Twitter network topology, and heterogeneous large language model (LLM) agents to examine the effect of mobilization messages on voter turnout. Each simulated agent is assigned demographic attributes, a personal political stance, and an LLM variant (\texttt{GPT-4.1}, \texttt{GPT-4.1-Mini}, or \texttt{GPT-4.1-Nano}) reflecting its political sophistication. Agents interact over realistic social network structures, receiving personalized feeds and dynamically updating their engagement behaviors and voting intentions. Experimental conditions replicate the informational and social mobilization treatments of the original Facebook study. Across scenarios, the simulator reproduces qualitative patterns observed in field experiments, including stronger mobilization effects under social message treatments and measurable peer spillovers. Our framework provides a controlled, reproducible environment for testing counterfactual designs and sensitivity analyses in political mobilization research, offering a bridge between high-validity field experiments and flexible computational modeling.\footnote{Code and data available at https://github.com/CausalMP/LLM-SocioPol}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26490v1">Scaffolding Creativity: How Divergent and Convergent LLM Personas Shape Human Machine Creative Problem-Solving</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly shaping creative work and problem-solving; however, prior research suggests that they may diminish unassisted creativity. To address this tension, a coach-like LLM environment was developed that embodies divergent and convergent thinking personas as two complementary processes. Effectiveness and user behavior were assessed through a controlled experiment in which participants interacted with either persona, while a control group engaged with a standard LLM providing direct answers. Notably, users' perceptions of which persona best supported their creativity often diverged from objective performance measures. Trait-based analyses revealed that individual differences predict when people utilize divergent versus convergent personas, suggesting opportunities for adaptive sequencing. Furthermore, interaction patterns reflected the design thinking model, demonstrating how persona-guided support shapes creative problem-solving. Our findings provide design principles for creativity support systems that strike a balance between exploration and convergence through persona-based guidance and personalization. These insights advance human-AI collaboration tools that scaffold rather than overshadow human creativity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26486v1">LINK-KG: LLM-Driven Coreference-Resolved Knowledge Graphs for Human Smuggling Networks</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Accepted in ICKG 2025 Conference, 8 Pages, 2 Figures
    </div>
    <details class="paper-abstract">
      Human smuggling networks are complex and constantly evolving, making them difficult to analyze comprehensively. Legal case documents offer rich factual and procedural insights into these networks but are often long, unstructured, and filled with ambiguous or shifting references, posing significant challenges for automated knowledge graph (KG) construction. Existing methods either overlook coreference resolution or fail to scale beyond short text spans, leading to fragmented graphs and inconsistent entity linking. We propose LINK-KG, a modular framework that integrates a three-stage, LLM-guided coreference resolution pipeline with downstream KG extraction. At the core of our approach is a type-specific Prompt Cache, which consistently tracks and resolves references across document chunks, enabling clean and disambiguated narratives for structured knowledge graph construction from both short and long legal texts. LINK-KG reduces average node duplication by 45.21% and noisy nodes by 32.22% compared to baseline methods, resulting in cleaner and more coherent graph structures. These improvements establish LINK-KG as a strong foundation for analyzing complex criminal networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26480v1">Automated Extract Method Refactoring with Open-Source LLMs: A Comparative Study</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Accepted at AIware'25 - Main Track
    </div>
    <details class="paper-abstract">
      Automating the Extract Method refactoring (EMR) remains challenging and largely manual despite its importance in improving code readability and maintainability. Recent advances in open-source, resource-efficient Large Language Models (LLMs) offer promising new approaches for automating such high-level tasks. In this work, we critically evaluate five state-of-the-art open-source LLMs, spanning 3B to 8B parameter sizes, on the EMR task for Python code. We systematically assess functional correctness and code quality using automated metrics and investigate the impact of prompting strategies by comparing one-shot prompting to a Recursive criticism and improvement (RCI) approach. RCI-based prompting consistently outperforms one-shot prompting in test pass rates and refactoring quality. The best-performing models, Deepseek-Coder-RCI and Qwen2.5-Coder-RCI, achieve test pass percentage (TPP) scores of 0.829 and 0.808, while reducing lines of code (LOC) per method from 12.103 to 6.192 and 5.577, and cyclomatic complexity (CC) from 4.602 to 3.453 and 3.294, respectively. A developer survey on RCI-generated refactorings shows over 70% acceptance, with Qwen2.5-Coder rated highest across all evaluation criteria. In contrast, the original code scored below neutral, particularly in readability and maintainability, underscoring the benefits of automated refactoring guided by quality prompts. While traditional metrics like CC and LOC provide useful signals, they often diverge from human judgments, emphasizing the need for human-in-the-loop evaluation. Our open-source benchmark offers a foundation for future research on automated refactoring with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02091v2">Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 ICASSP 2025
    </div>
    <details class="paper-abstract">
      Recent studies suggest that the deeper layers of Large Language Models (LLMs) contribute little to representation learning and can often be removed without significant performance loss. However, such claims are typically drawn from narrow evaluations and may overlook important aspects of model behavior. In this work, we present a systematic study of depth utilization across diverse dimensions, including evaluation protocols, task categories, and model architectures. Our analysis confirms that very deep layers are generally less effective than earlier ones, but their contributions vary substantially with the evaluation setting. Under likelihood-based metrics without generation, pruning most layers preserves performance, with only the initial few being critical. By contrast, generation-based evaluation uncovers indispensable roles for middle and deeper layers in enabling reasoning and maintaining long-range coherence. We further find that knowledge and retrieval are concentrated in shallow components, whereas reasoning accuracy relies heavily on deeper layers -- yet can be reshaped through distillation. These results highlight that depth usage in LLMs is highly heterogeneous and context-dependent, underscoring the need for task-, metric-, and model-aware perspectives in both interpreting and compressing large models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21513v2">Wisdom and Delusion of LLM Ensembles for Code Generation and Repair</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Added Acknowledgments section and hyphenated last names
    </div>
    <details class="paper-abstract">
      Today's pursuit of a single Large Language Model (LMM) for all software engineering tasks is resource-intensive and overlooks the potential benefits of complementarity, where different models contribute unique strengths. However, the degree to which coding LLMs complement each other and the best strategy for maximizing an ensemble's potential are unclear, leaving practitioners without a clear path to move beyond single-model systems. To address this gap, we empirically compare ten individual LLMs from five families, and three ensembles of these LLMs across three software engineering benchmarks covering code generation and program repair. We assess the complementarity between models and the performance gap between the best individual model and the ensembles. Next, we evaluate various selection heuristics to identify correct solutions from an ensemble's candidate pool. We find that the theoretical upperbound for an ensemble's performance can be 83% above the best single model. Our results show that consensus-based strategies for selecting solutions fall into a "popularity trap," amplifying common but incorrect outputs. In contrast, a diversity-based strategy realizes up to 95% of this theoretical potential, and proves effective even in small two-model ensembles, enabling a cost-efficient way to enhance performance by leveraging multiple LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23127v2">Lost in Tokenization: Context as the Key to Unlocking Biomolecular Understanding in Scientific LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 38 pages, under review
    </div>
    <details class="paper-abstract">
      Scientific Large Language Models (Sci-LLMs) have emerged as a promising frontier for accelerating biological discovery. However, these models face a fundamental challenge when processing raw biomolecular sequences: the tokenization dilemma. Whether treating sequences as a specialized language, risking the loss of functional motif information, or as a separate modality, introducing formidable alignment challenges, current strategies fundamentally limit their reasoning capacity. We challenge this sequence-centric paradigm by positing that a more effective strategy is to provide Sci-LLMs with high-level structured context derived from established bioinformatics tools, thereby bypassing the need to interpret low-level noisy sequence data directly. Through a systematic comparison of leading Sci-LLMs on biological reasoning tasks, we tested three input modes: sequence-only, context-only, and a combination of both. Our findings are striking: the context-only approach consistently and substantially outperforms all other modes. Even more revealing, the inclusion of the raw sequence alongside its high-level context consistently degrades performance, indicating that raw sequences act as informational noise, even for models with specialized tokenization schemes. These results suggest that the primary strength of existing Sci-LLMs lies not in their nascent ability to interpret biomolecular syntax from scratch, but in their profound capacity for reasoning over structured, human-readable knowledge. Therefore, we argue for reframing Sci-LLMs not as sequence decoders, but as powerful reasoning engines over expert knowledge. This work lays the foundation for a new class of hybrid scientific AI agents, repositioning the developmental focus from direct sequence interpretation towards high-level knowledge synthesis. The code is available at https://github.com/opendatalab-raiser/CoKE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11329v4">TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 14 pages, 16 figures. For source code, see https://github.com/microsoft/tokenweave. In version 2, Figure 6 shows All-Reduce bandwidth instead of Reduce-Scatter. The Multimem Reduce-Scatter bandwidth formula differs slightly from the ring-based version. Fixed x-ticks in Figure 7
    </div>
    <details class="paper-abstract">
      Distributed inference of large language models (LLMs) can introduce overheads of up to 20% even over GPUs connected via high-speed interconnects such as NVLink. Multiple techniques have been proposed to mitigate these overheads by decomposing computations into finer-grained tasks and overlapping communication with sub-tasks as they complete. However, fine-grained decomposition of a large computation into many smaller computations on GPUs results in overheads. Furthermore, the communication itself uses many streaming multiprocessors (SMs), adding to the overhead. We present TokenWeave to address these challenges. TokenWeave proposes a Token-Splitting technique that divides the tokens in the inference batch into two approximately equal subsets in a wave-aware manner. The communication of one subset is then overlapped with the computation of the other. In addition, TokenWeave optimizes the order of the layer normalization computation with respect to communication operations and implements a novel fused AllReduce--RMSNorm kernel that carefully leverages Multimem instruction support available on Hopper and Blackwell NVIDIA GPUs. These optimizations allow TokenWeave to perform communication and RMSNorm using only 2-8 SMs. Moreover, our kernel enables the memory-bound RMSNorm to be overlapped with the other batch's computation, providing additional gains. Our evaluations demonstrate up to 1.29x speedup in latency and 1.26x higher throughput across multiple models and workloads. In several settings, TokenWeave results in better performance compared to an equivalent model with all communication removed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25536v2">TwinVoice: A Multi-dimensional Benchmark Towards Digital Twins via LLM Persona Simulation</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Main paper: 11 pages, 3 figures, 6 tables. Appendix: 28 pages. Bangde Du and Minghao Guo contributed equally. Corresponding authors: Ziyi Ye (ziyiye@fudan.edu.cn), Qingyao Ai (aiqy@tsinghua.edu.cn)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are exhibiting emergent human-like abilities and are increasingly envisioned as the foundation for simulating an individual's communication style, behavioral tendencies, and personality traits. However, current evaluations of LLM-based persona simulation remain limited: most rely on synthetic dialogues, lack systematic frameworks, and lack analysis of the capability requirement. To address these limitations, we introduce TwinVoice, a comprehensive benchmark for assessing persona simulation across diverse real-world contexts. TwinVoice encompasses three dimensions: Social Persona (public social interactions), Interpersonal Persona (private dialogues), and Narrative Persona (role-based expression). It further decomposes the evaluation of LLM performance into six fundamental capabilities, including opinion consistency, memory recall, logical reasoning, lexical fidelity, persona tone, and syntactic style. Experimental results reveal that while advanced models achieve moderate accuracy in persona simulation, they still fall short of capabilities such as syntactic style and memory recall. Consequently, the average performance achieved by LLMs remains considerably below the human baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26374v1">BOTS: A Unified Framework for Bayesian Online Task Selection in LLM Reinforcement Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Reinforcement finetuning (RFT) is a key technique for aligning Large Language Models (LLMs) with human preferences and enhancing reasoning, yet its effectiveness is highly sensitive to which tasks are explored during training. Uniform task sampling is inefficient, wasting computation on tasks that are either trivial or unsolvable, while existing task selection methods often suffer from high rollout costs, poor adaptivity, or incomplete evidence. We introduce \textbf{BOTS}, a unified framework for \textbf{B}ayesian \textbf{O}nline \textbf{T}ask \textbf{S}election in LLM reinforcement finetuning. Grounded in Bayesian inference, BOTS adaptively maintains posterior estimates of task difficulty as the model evolves. It jointly incorporates \emph{explicit evidence} from direct evaluations of selected tasks and \emph{implicit evidence} inferred from these evaluations for unselected tasks, with Thompson sampling ensuring a principled balance between exploration and exploitation. To make implicit evidence practical, we instantiate it with an ultra-light interpolation-based plug-in that estimates difficulties of unevaluated tasks without extra rollouts, adding negligible overhead. Empirically, across diverse domains and LLM scales, BOTS consistently improves data efficiency and performance over baselines and ablations, providing a practical and extensible solution for dynamic task selection in RFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26336v1">From Amateur to Master: Infusing Knowledge into LLMs via Automated Curriculum Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at general tasks but underperform in specialized domains like economics and psychology, which require deep, principled understanding. To address this, we introduce ACER (Automated Curriculum-Enhanced Regimen) that transforms generalist models into domain experts without sacrificing their broad capabilities. ACER first synthesizes a comprehensive, textbook-style curriculum by generating a table of contents for a subject and then creating question-answer (QA) pairs guided by Bloom's taxonomy. This ensures systematic topic coverage and progressively increasing difficulty. The resulting synthetic corpus is used for continual pretraining with an interleaved curriculum schedule, aligning learning across both content and cognitive dimensions. Experiments with Llama 3.2 (1B and 3B) show significant gains in specialized MMLU subsets. In challenging domains like microeconomics, where baselines struggle, ACER boosts accuracy by 5 percentage points. Across all target domains, we observe a consistent macro-average improvement of 3 percentage points. Notably, ACER not only prevents catastrophic forgetting but also facilitates positive cross-domain knowledge transfer, improving performance on non-target domains by 0.7 points. Beyond MMLU, ACER enhances performance on knowledge-intensive benchmarks like ARC and GPQA by over 2 absolute points, while maintaining stable performance on general reasoning tasks. Our results demonstrate that ACER offers a scalable and effective recipe for closing critical domain gaps in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26309v1">GraphCompliance: Aligning Policy and Context Graphs for LLM-Based Regulatory Compliance</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Under review at The Web Conference 2026 (Semantics & Knowledge track). Code will be released upon acceptance. This arXiv v1 contains no repository links to preserve double-blind review
    </div>
    <details class="paper-abstract">
      Compliance at web scale poses practical challenges: each request may require a regulatory assessment. Regulatory texts (e.g., the General Data Protection Regulation, GDPR) are cross-referential and normative, while runtime contexts are expressed in unstructured natural language. This setting motivates us to align semantic information in unstructured text with the structured, normative elements of regulations. To this end, we introduce GraphCompliance, a framework that represents regulatory texts as a Policy Graph and runtime contexts as a Context Graph, and aligns them. In this formulation, the policy graph encodes normative structure and cross-references, whereas the context graph formalizes events as subject-action-object (SAO) and entity-relation triples. This alignment anchors the reasoning of a judge large language model (LLM) in structured information and helps reduce the burden of regulatory interpretation and event parsing, enabling a focus on the core reasoning step. In experiments on 300 GDPR-derived real-world scenarios spanning five evaluation tasks, GraphCompliance yields 4.1-7.2 percentage points (pp) higher micro-F1 than LLM-only and RAG baselines, with fewer under- and over-predictions, resulting in higher recall and lower false positive rates. Ablation studies indicate contributions from each graph component, suggesting that structured representations and a judge LLM are complementary for normative reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04380v3">Diversity as a Reward: Fine-Tuning LLMs on a Mixture of Domain-Undetermined Data</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Accepted by NeurIPS'25 main track. 47 pages, 21 figures, 32 tables
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) using diverse datasets is crucial for enhancing their overall performance across various domains. In practical scenarios, existing methods based on modeling the mixture proportions of data composition often struggle with data whose domain labels are missing, imprecise or non-normalized, while methods based on data selection usually encounter difficulties in balancing multi-domain performance. To address these challenges, in this work, we investigate the role of data diversity in enhancing the overall abilities of LLMs by empirically constructing contrastive data pools and theoretically deriving explanations. Building upon the insights gained, we propose a new method that gives the LLM a dual identity: an output model to cognitively probe and select data based on diversity reward, as well as an input model to be tuned with the selected data. Extensive experiments show that the proposed method notably boosts performance across domain-undetermined data and a series of foundational downstream tasks when applied to various advanced LLMs. We release our code and hope this study can shed light on the understanding of data diversity and advance feedback-driven data-model co-design for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26277v1">Do LLMs Signal When They're Right? Evidence from Neuron Agreement</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) commonly boost reasoning via sample-evaluate-ensemble decoders, achieving label free gains without ground truth. However, prevailing strategies score candidates using only external outputs such as token probabilities, entropies, or self evaluations, and these signals can be poorly calibrated after post training. We instead analyze internal behavior based on neuron activations and uncover three findings: (1) external signals are low dimensional projections of richer internal dynamics; (2) correct responses activate substantially fewer unique neurons than incorrect ones throughout generation; and (3) activations from correct responses exhibit stronger cross sample agreement, whereas incorrect ones diverge. Motivated by these observations, we propose Neuron Agreement Decoding (NAD), an unsupervised best-of-N method that selects candidates using activation sparsity and cross sample neuron agreement, operating solely on internal signals and without requiring comparable textual outputs. NAD enables early correctness prediction within the first 32 generated tokens and supports aggressive early stopping. Across math and science benchmarks with verifiable answers, NAD matches majority voting; on open ended coding benchmarks where majority voting is inapplicable, NAD consistently outperforms Avg@64. By pruning unpromising trajectories early, NAD reduces token usage by 99% with minimal loss in generation quality, showing that internal signals provide reliable, scalable, and efficient guidance for label free ensemble decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26274v1">PVMark: Enabling Public Verifiability for LLM Watermarking Schemes</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Watermarking schemes for large language models (LLMs) have been proposed to identify the source of the generated text, mitigating the potential threats emerged from model theft. However, current watermarking solutions hardly resolve the trust issue: the non-public watermark detection cannot prove itself faithfully conducting the detection. We observe that it is attributed to the secret key mostly used in the watermark detection -- it cannot be public, or the adversary may launch removal attacks provided the key; nor can it be private, or the watermarking detection is opaque to the public. To resolve the dilemma, we propose PVMark, a plugin based on zero-knowledge proof (ZKP), enabling the watermark detection process to be publicly verifiable by third parties without disclosing any secret key. PVMark hinges upon the proof of `correct execution' of watermark detection on which a set of ZKP constraints are built, including mapping, random number generation, comparison, and summation. We implement multiple variants of PVMark in Python, Rust and Circom, covering combinations of three watermarking schemes, three hash functions, and four ZKP protocols, to show our approach effectively works under a variety of circumstances. By experimental results, PVMark efficiently enables public verifiability on the state-of-the-art LLM watermarking schemes yet without compromising the watermarking performance, promising to be deployed in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26270v1">Graph-Enhanced Policy Optimization in LLM Agent Training</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Under review as a conference paper
    </div>
    <details class="paper-abstract">
      Group based reinforcement learning (RL) has shown impressive results on complex reasoning and mathematical tasks. Yet, when applied to train multi-turn, interactive LLM agents, these methods often suffer from structural blindness-the inability to exploit the underlying connectivity of the environment. This manifests in three critical challenges: (1) inefficient, unguided exploration, (2) imprecise credit assignment due to overlooking pivotal states, and (3) myopic planning caused by static reward discounting. We address these issues with Graph-Enhanced Policy Optimization (GEPO), which dynamically constructs a state-transition graph from agent experience and employs graph-theoretic centrality to provide three synergistic learning signals: (1)structured intrinsic rewards that guide exploration toward high-impact states, (2) a graph-enhanced advantage function for topology-aware credit assignment, and (3) a dynamic discount factor adapted to each state's strategic value. On the ALFWorld, WebShop, and a proprietary Workbench benchmarks, GEPO demonstrates strong performance, achieving absolute success rate gains of +4.1%, +5.3%, and +10.9% over competitive baselines. These results highlight that explicitly modeling environmental structure is a robust, generalizable strategy for advancing LLM agent training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13229v3">IGD: Token Decisiveness Modeling via Information Gain in LLMs for Personalized Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong potential for recommendation by framing item prediction as a token-by-token language generation task. However, existing methods treat all item tokens equally, simply pursuing likelihood maximization during both optimization and decoding. This overlooks crucial token-level differences in decisiveness-many tokens contribute little to item discrimination yet can dominate optimization or decoding. To quantify token decisiveness, we propose a novel perspective that models item generation as a decision process, measuring token decisiveness by the Information Gain (IG) each token provides in reducing uncertainty about the generated item. Our empirical analysis reveals that most tokens have low IG but often correspond to high logits, disproportionately influencing training loss and decoding, which may impair model performance. Building on these insights, we introduce an Information Gain-based Decisiveness-aware Token handling (IGD) strategy that integrates token decisiveness into both tuning and decoding. Specifically, IGD downweights low-IG tokens during tuning and rebalances decoding to emphasize tokens with high IG. In this way, IGD moves beyond pure likelihood maximization, effectively prioritizing high-decisiveness tokens. Extensive experiments on four benchmark datasets with two LLM backbones demonstrate that IGD consistently improves recommendation accuracy, achieving significant gains on widely used ranking metrics compared to strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26242v1">Retrieval Augmented Generation-Enhanced Distributed LLM Agents for Generalizable Traffic Signal Control with Emergency Vehicles</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      With increasing urban traffic complexity, Traffic Signal Control (TSC) is essential for optimizing traffic flow and improving road safety. Large Language Models (LLMs) emerge as promising approaches for TSC. However, they are prone to hallucinations in emergencies, leading to unreliable decisions that may cause substantial delays for emergency vehicles. Moreover, diverse intersection types present substantial challenges for traffic state encoding and cross-intersection training, limiting generalization across heterogeneous intersections. Therefore, this paper proposes Retrieval Augmented Generation (RAG)-enhanced distributed LLM agents with Emergency response for Generalizable TSC (REG-TSC). Firstly, this paper presents an emergency-aware reasoning framework, which dynamically adjusts reasoning depth based on the emergency scenario and is equipped with a novel Reviewer-based Emergency RAG (RERAG) to distill specific knowledge and guidance from historical cases, enhancing the reliability and rationality of agents' emergency decisions. Secondly, this paper designs a type-agnostic traffic representation and proposes a Reward-guided Reinforced Refinement (R3) for heterogeneous intersections. R3 adaptively samples training experience from diverse intersections with environment feedback-based priority and fine-tunes LLM agents with a designed reward-weighted likelihood loss, guiding REG-TSC toward high-reward policies across heterogeneous intersections. On three real-world road networks with 17 to 177 heterogeneous intersections, extensive experiments show that REG-TSC reduces travel time by 42.00%, queue length by 62.31%, and emergency vehicle waiting time by 83.16%, outperforming other state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26238v1">Questionnaire meets LLM: A Benchmark and Empirical Study of Structural Skills for Understanding Questions and Responses</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 14 pages, 3 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Millions of people take surveys every day, from market polls and academic studies to medical questionnaires and customer feedback forms. These datasets capture valuable insights, but their scale and structure present a unique challenge for large language models (LLMs), which otherwise excel at few-shot reasoning over open-ended text. Yet, their ability to process questionnaire data or lists of questions crossed with hundreds of respondent rows remains underexplored. Current retrieval and survey analysis tools (e.g., Qualtrics, SPSS, REDCap) are typically designed for humans in the workflow, limiting such data integration with LLM and AI-empowered automation. This gap leaves scientists, surveyors, and everyday users without evidence-based guidance on how to best represent questionnaires for LLM consumption. We address this by introducing QASU (Questionnaire Analysis and Structural Understanding), a benchmark that probes six structural skills, including answer lookup, respondent count, and multi-hop inference, across six serialization formats and multiple prompt strategies. Experiments on contemporary LLMs show that choosing an effective format and prompt combination can improve accuracy by up to 8.8% points compared to suboptimal formats. For specific tasks, carefully adding a lightweight structural hint through self-augmented prompting can yield further improvements of 3-4% points on average. By systematically isolating format and prompting effects, our open source benchmark offers a simple yet versatile foundation for advancing both research and real-world practice in LLM-based questionnaire analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26219v1">Test-Time Alignment of LLMs via Sampling-Based Optimal Control in pre-logit space</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 21 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Test-time alignment of large language models (LLMs) attracts attention because fine-tuning LLMs requires high computational costs. In this paper, we propose a new test-time alignment method called adaptive importance sampling on pre-logits (AISP) on the basis of the sampling-based model predictive control with the stochastic control input. AISP applies the Gaussian perturbation into pre-logits, which are outputs of the penultimate layer, so as to maximize expected rewards with respect to the mean of the perturbation. We demonstrate that the optimal mean is obtained by importance sampling with sampled rewards. AISP outperforms best-of-n sampling in terms of rewards over the number of used samples and achieves higher rewards than other reward-based test-time alignment methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26217v1">Hybrid LLM and Higher-Order Quantum Approximate Optimization for CSA Collateral Management</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 6 pages
    </div>
    <details class="paper-abstract">
      We address finance-native collateral optimization under ISDA Credit Support Annexes (CSAs), where integer lots, Schedule A haircuts, RA/MTA gating, and issuer/currency/class caps create rugged, legally bounded search spaces. We introduce a certifiable hybrid pipeline purpose-built for this domain: (i) an evidence-gated LLM that extracts CSA terms to a normalized JSON (abstain-by-default, span-cited); (ii) a quantum-inspired explorer that interleaves simulated annealing with micro higher order QAOA (HO-QAOA) on binding sub-QUBOs (subset size n <= 16, order k <= 4) to coordinate multi-asset moves across caps and RA-induced discreteness; (iii) a weighted risk-aware objective (Movement, CVaR, funding-priced overshoot) with an explicit coverage window U <= Reff+B; and (iv) CP-SAT as single arbiter to certify feasibility and gaps, including a U-cap pre-check that reports the minimal feasible buffer B*. Encoding caps/rounding as higher-order terms lets HO-QAOA target the domain couplings that defeat local swaps. On government bond datasets and multi-CSA inputs, the hybrid improves a strong classical baseline (BL-3) by 9.1%, 9.6%, and 10.7% across representative harnesses, delivering better cost-movement-tail frontiers under governance settings. We release governance grade artifacts-span citations, valuation matrix audit, weight provenance, QUBO manifests, and CP-SAT traces-to make results auditable and reproducible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26213v1">OmniLayout: Enabling Coarse-to-Fine Learning with LLMs for Universal Document Layout Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 TL;DR: With OmniLayout-1M dataset and LLM-based coarse-to-fine learning, we enable universal and diverse document layout generation
    </div>
    <details class="paper-abstract">
      Document AI has advanced rapidly and is attracting increasing attention. Yet, while most efforts have focused on document layout analysis (DLA), its generative counterpart, document layout generation, remains underexplored. A major obstacle lies in the scarcity of diverse layouts: academic papers with Manhattan-style structures dominate existing studies, while open-world genres such as newspapers and magazines remain severely underrepresented. To address this gap, we curate OmniLayout-1M, the first million-scale dataset of diverse document layouts, covering six common document types and comprising contemporary layouts collected from multiple sources. Moreover, since existing methods struggle in complex domains and often fail to arrange long sequences coherently, we introduce OmniLayout-LLM, a 0.5B model with designed two-stage Coarse-to-Fine learning paradigm: 1) learning universal layout principles from OmniLayout-1M with coarse category definitions, and 2) transferring the knowledge to a specific domain with fine-grained annotations. Extensive experiments demonstrate that our approach achieves strong performance on multiple domains in M$^{6}$Doc dataset, substantially surpassing both existing layout generation experts and several latest general-purpose LLMs. Our code, models, and dataset will be publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14970v4">Self-Evolving Curriculum for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has proven effective for fine-tuning large language models (LLMs), significantly enhancing their reasoning abilities in domains such as mathematics and code generation. A crucial factor influencing RL fine-tuning success is the training curriculum: the order in which training problems are presented. While random curricula serve as common baselines, they remain suboptimal; manually designed curricula often rely heavily on heuristics, and online filtering methods can be computationally prohibitive. To address these limitations, we propose Self-Evolving Curriculum (SEC), an automatic curriculum learning method that learns a curriculum policy concurrently with the RL fine-tuning process. Our approach formulates curriculum selection as a non-stationary Multi-Armed Bandit problem, treating each problem category (e.g., difficulty level or problem type) as an individual arm. We leverage the absolute advantage from policy gradient methods as a proxy measure for immediate learning gain. At each training step, the curriculum policy selects categories to maximize this reward signal and is updated using the TD(0) method. Across three distinct reasoning domains: planning, inductive reasoning, and mathematics, our experiments demonstrate that SEC significantly improves models' reasoning capabilities, enabling better generalization to harder, out-of-distribution test problems. Additionally, our approach achieves better skill balance when fine-tuning simultaneously on multiple reasoning domains. These findings highlight SEC as a promising strategy for RL fine-tuning of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16648v2">FESTA: Functionally Equivalent Sampling for Trust Assessment of Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Accepted in the Findings of EMNLP, 2025
    </div>
    <details class="paper-abstract">
      The accurate trust assessment of multimodal large language models (MLLMs) generated predictions, which can enable selective prediction and improve user confidence, is challenging due to the diverse multi-modal input paradigms. We propose Functionally Equivalent Sampling for Trust Assessment (FESTA), a multimodal input sampling technique for MLLMs, that generates an uncertainty measure based on the equivalent and complementary input samplings. The proposed task-preserving sampling approach for uncertainty quantification expands the input space to probe the consistency (through equivalent samples) and sensitivity (through complementary samples) of the model. FESTA uses only input-output access of the model (black-box), and does not require ground truth (unsupervised). The experiments are conducted with various off-the-shelf multi-modal LLMs, on both visual and audio reasoning tasks. The proposed FESTA uncertainty estimate achieves significant improvement (33.3% relative improvement for vision-LLMs and 29.6% relative improvement for audio-LLMs) in selective prediction performance, based on area-under-receiver-operating-characteristic curve (AUROC) metric in detecting mispredictions. The code implementation is open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.11094v2">The Scales of Justitia: A Comprehensive Survey on Safety Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 20 pages, preprint
    </div>
    <details class="paper-abstract">
      With the rapid advancement of artificial intelligence, Large Language Models (LLMs) have shown remarkable capabilities in Natural Language Processing (NLP), including content generation, human-computer interaction, machine translation, and code generation. However, their widespread deployment has also raised significant safety concerns. In particular, LLM-generated content can exhibit unsafe behaviors such as toxicity, bias, or misinformation, especially in adversarial contexts, which has attracted increasing attention from both academia and industry. Although numerous studies have attempted to evaluate these risks, a comprehensive and systematic survey on safety evaluation of LLMs is still lacking. This work aims to fill this gap by presenting a structured overview of recent advances in safety evaluation of LLMs. Specifically, we propose a four-dimensional taxonomy: (i) Why to evaluate, which explores the background of safety evaluation of LLMs, how they differ from general LLMs evaluation, and the significance of such evaluation; (ii) What to evaluate, which examines and categorizes existing safety evaluation tasks based on key capabilities, including dimensions such as toxicity, robustness, ethics, bias and fairness, truthfulness, and related aspects; (iii) Where to evaluate, which summarizes the evaluation metrics, datasets and benchmarks currently used in safety evaluations; (iv) How to evaluate, which reviews existing mainstream evaluation methods based on the roles of the evaluators and some evaluation frameworks that integrate the entire evaluation pipeline. Finally, we identify the challenges in safety evaluation of LLMs and propose promising research directions to promote further advancement in this field. We emphasize the necessity of prioritizing safety evaluation to ensure the reliable and responsible deployment of LLMs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26163v1">Exploring Dissatisfaction in Bus Route Reduction through LLM-Calibrated Agent-Based Modeling</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 17 pages, 8 figures, 4 tables
    </div>
    <details class="paper-abstract">
      As emerging mobility modes continue to expand, many cities face declining bus ridership, increasing fiscal pressure to sustain underutilized routes, and growing inefficiencies in resource allocation. This study employs an agent-based modelling (ABM) approach calibrated through a large language model (LLM) using few-shot learning to examine how progressive bus route cutbacks affect passenger dissatisfaction across demographic groups and overall network resilience. Using IC-card data from Beijing's Huairou District, the LLM-calibrated ABM estimated passenger sensitivity parameters related to travel time, waiting, transfers, and crowding. Results show that the structural configuration of the bus network exerts a stronger influence on system stability than capacity or operational factors. The elimination of high-connectivity routes led to an exponential rise in total dissatisfaction, particularly among passengers with disabilities and older adults. The evolution of dissatisfaction exhibited three distinct phases - stable, transitional, and critical. Through the analysis of each stage, this study found that the continuous bus route reduction scenario exhibits three-stage thresholds. Once these thresholds are crossed, even a small reduction in routes may lead to a significant loss of passenger flow. Research highlights the nonlinear response of user sentiment to service reductions and underscore the importance of maintaining structural critical routes and providing stable services to vulnerable groups for equitable and resilient transport planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26143v1">Reasoning Curriculum: Bootstrapping Broad LLM Reasoning from Math</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) can elicit strong reasoning in large language models (LLMs), yet most open efforts focus on math and code. We propose Reasoning Curriculum, a simple two-stage curriculum that first elicits reasoning skills in pretraining-aligned domains such as math, then adapts and refines these skills across other domains via joint RL. Stage 1 performs a brief cold start and then math-only RL with verifiable rewards to develop reasoning skills. Stage 2 runs joint RL on mixed-domain data to transfer and consolidate these skills. The curriculum is minimal and backbone-agnostic, requiring no specialized reward models beyond standard verifiability checks. Evaluated on Qwen3-4B and Llama-3.1-8B over a multi-domain suite, reasoning curriculum yields consistent gains. Ablations and a cognitive-skill analysis indicate that both stages are necessary and that math-first elicitation increases cognitive behaviors important for solving complex problems. Reasoning Curriculum provides a compact, easy-to-adopt recipe for general reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18477v2">LAFA: Agentic LLM-Driven Federated Analytics over Decentralized Data Sources</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 This paper has been accepted by the 16th IEEE International Conference on Cloud Computing Technology and Science (CloudCom 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown great promise in automating data analytics tasks by interpreting natural language queries and generating multi-operation execution plans. However, existing LLM-agent-based analytics frameworks operate under the assumption of centralized data access, offering little to no privacy protection. In contrast, federated analytics (FA) enables privacy-preserving computation across distributed data sources, but lacks support for natural language input and requires structured, machine-readable queries. In this work, we present LAFA, the first system that integrates LLM-agent-based data analytics with FA. LAFA introduces a hierarchical multi-agent architecture that accepts natural language queries and transforms them into optimized, executable FA workflows. A coarse-grained planner first decomposes complex queries into sub-queries, while a fine-grained planner maps each subquery into a Directed Acyclic Graph of FA operations using prior structural knowledge. To improve execution efficiency, an optimizer agent rewrites and merges multiple DAGs, eliminating redundant operations and minimizing computational and communicational overhead. Our experiments demonstrate that LAFA consistently outperforms baseline prompting strategies by achieving higher execution plan success rates and reducing resource-intensive FA operations by a substantial margin. This work establishes a practical foundation for privacy-preserving, LLM-driven analytics that supports natural language input in the FA setting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23595v3">Multi-Agent Evolve: LLM Self-Improve through Co-evolution</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 29 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has demonstrated significant potential in enhancing the reasoning capabilities of large language models (LLMs). However, the success of RL for LLMs heavily relies on human-curated datasets and verifiable rewards, which limit their scalability and generality. Recent Self-Play RL methods, inspired by the success of the paradigm in games and Go, aim to enhance LLM reasoning capabilities without human-annotated data. However, their methods primarily depend on a grounded environment for feedback (e.g., a Python interpreter or a game engine); extending them to general domains remains challenging. To address these challenges, we propose Multi-Agent Evolve (MAE), a framework that enables LLMs to self-evolve in solving diverse tasks, including mathematics, reasoning, and general knowledge Q&A. The core design of MAE is based on a triplet of interacting agents (Proposer, Solver, Judge) that are instantiated from a single LLM, and applies reinforcement learning to optimize their behaviors. The Proposer generates questions, the Solver attempts solutions, and the Judge evaluates both while co-evolving. Experiments on Qwen2.5-3B-Instruct demonstrate that MAE achieves an average improvement of 4.54% on multiple benchmarks. These results highlight MAE as a scalable, data-efficient method for enhancing the general reasoning abilities of LLMs with minimal reliance on human-curated supervision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26130v1">Beyond Synthetic Benchmarks: Evaluating LLM Performance on Real-World Class-Level Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 Pre-print prepared for journal submission
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have advanced code generation at the function level, yet their ability to produce correct class-level implementations in authentic software projects remains poorly understood. This work introduces a novel benchmark derived from open-source repositories, comprising real-world classes divided into seen and unseen partitions to evaluate generalization under practical conditions. The evaluation examines multiple LLMs under varied input specifications, retrieval-augmented configurations, and documentation completeness levels. Results reveal a stark performance disparity: LLMs achieve 84% to 89% correctness on established synthetic benchmarks but only 25% to 34% on real-world class tasks, with negligible differences between familiar and novel codebases. Comprehensive docstrings yield modest gains of 1% to 3% in functional accuracy, though statistical significance is rare. Retrieval-augmented generation proves most effective with partial documentation, improving correctness by 4% to 7% by supplying concrete implementation patterns absent from specifications. Error profiling identifies AttributeError, TypeError, and AssertionError as dominant failure modes (84% of cases), with synthetic tests overemphasizing assertion issues and real-world scenarios highlighting type and attribute mismatches. Retrieval augmentation reduces logical flaws but can introduce dependency conflicts. The benchmark and analysis expose critical limitations in current LLM capabilities for class-level engineering, offering actionable insights for enhancing context modelling, documentation strategies, and retrieval integration in production code assistance tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26122v1">Reasoning Path Divergence: A New Metric and Curation Strategy to Unlock LLM Diverse Thinking</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      While Test-Time Scaling (TTS) has proven effective in improving the reasoning ability of large language models (LLMs), low diversity in model outputs often becomes a bottleneck; this is partly caused by the common "one problem, one solution" (1P1S) training practice, which provides a single canonical answer and can push models toward a narrow set of reasoning paths. To address this, we propose a "one problem, multiple solutions" (1PNS) training paradigm that exposes the model to a variety of valid reasoning trajectories and thus increases inference diversity. A core challenge for 1PNS is reliably measuring semantic differences between multi-step chains of thought, so we introduce Reasoning Path Divergence (RPD), a step-level metric that aligns and scores Long Chain-of-Thought solutions to capture differences in intermediate reasoning. Using RPD, we curate maximally diverse solution sets per problem and fine-tune Qwen3-4B-Base. Experiments show that RPD-selected training yields more varied outputs and higher pass@k, with an average +2.80% gain in pass@16 over a strong 1P1S baseline and a +4.99% gain on AIME24, demonstrating that 1PNS further amplifies the effectiveness of TTS. Our code is available at https://github.com/fengjujf/Reasoning-Path-Divergence .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2311.06736v3">Are LLMs Rigorous Logical Reasoners? Empowering Natural Language Proof Generation by Stepwise Decoding with Contrastive Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 15 pages, 2 figures, 11 tables. Accepted by AACL 2025 main conference
    </div>
    <details class="paper-abstract">
      Logical reasoning is a pivotal component in the field of artificial intelligence. Proof planning, particularly in contexts requiring the validation of explanation accuracy, continues to present challenges. The recent advancement of large language models (LLMs) has led to significant progress in natural language proof planning, evolving from one-stage generators to more complex three-stage systems that include additional searchers or verifiers. While these assisted methods improve the quality of generated results, they also introduce increased search efforts and computational costs. Furthermore, the generative process itself remains underexplored. In this study, we propose a stepwise decoding approach augmented by contrastive learning to address two common errors encountered during the LLM generator's decoding process. We fine-tune the language model using both vanilla and enhanced hard negatives to mitigate these decoding errors. Empirical results demonstrate the effectiveness of our strategy. Additionally, our further analysis reveals that even larger LLMs still struggle to generate rigorous logical chains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11695v2">When Agents Trade: Live Multi-Market Trading Benchmark for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      Although Large Language Model (LLM)-based agents are increasingly used in financial trading, it remains unclear whether they can reason and adapt in live markets, as most studies test models instead of agents, cover limited periods and assets, and rely on unverified data. To address these gaps, we introduce Agent Market Arena (AMA), the first lifelong, real-time benchmark for evaluating LLM-based trading agents across multiple markets. AMA integrates verified trading data, expert-checked news, and diverse agent architectures within a unified trading framework, enabling fair and continuous comparison under real conditions. It implements four agents, including InvestorAgent as a single-agent baseline, TradeAgent and HedgeFundAgent with different risk styles, and DeepFundAgent with memory-based reasoning, and evaluates them across GPT-4o, GPT-4.1, Claude-3.5-haiku, Claude-sonnet-4, and Gemini-2.0-flash. Live experiments on both cryptocurrency and stock markets demonstrate that agent frameworks display markedly distinct behavioral patterns, spanning from aggressive risk-taking to conservative decision-making, whereas model backbones contribute less to outcome variation. AMA thus establishes a foundation for rigorous, reproducible, and continuously evolving evaluation of financial reasoning and trading intelligence in LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14808v4">HyGen: Efficient LLM Serving via Elastic Online-Offline Request Co-location</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 NeurIPS 2025. 21 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have facilitated a wide range of applications with distinct service-level objectives (SLOs), from latency-sensitive online tasks like interactive chatbots to throughput-oriented offline workloads like data synthesis. The existing deployment model, which dedicates machines to each workload, simplifies SLO management but often leads to poor resource utilization. This paper introduces HyGen, an interference-aware LLM serving system that enables efficient co-location of online and offline workloads while preserving SLOs. HyGen incorporates two key innovations: (1) performance control mechanisms, including a latency predictor to estimate batch execution time and an SLO-aware profiler to quantify latency interference, and (2) SLO-aware offline scheduling policies that maximize serving throughput and prevent starvation. Our evaluation on production workloads shows that HyGen achieves up to 3.9-5.8x throughput gains over online and hybrid serving baselines, while ensuring latency SLOs. The code of HyGen is publicly available at https://github.com/UIUC-MLSys/HyGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03710v3">Improving LLM Safety Alignment with Dual-Objective Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-30
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26037v1">SIRAJ: Diverse and Efficient Red-Teaming for LLM Agents via Distilled Structured Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-30
    </div>
    <details class="paper-abstract">
      The ability of LLM agents to plan and invoke tools exposes them to new safety risks, making a comprehensive red-teaming system crucial for discovering vulnerabilities and ensuring their safe deployment. We present SIRAJ: a generic red-teaming framework for arbitrary black-box LLM agents. We employ a dynamic two-step process that starts with an agent definition and generates diverse seed test cases that cover various risk outcomes, tool-use trajectories, and risk sources. Then, it iteratively constructs and refines model-based adversarial attacks based on the execution trajectories of former attempts. To optimize the red-teaming cost, we present a model distillation approach that leverages structured forms of a teacher model's reasoning to train smaller models that are equally effective. Across diverse evaluation agent settings, our seed test case generation approach yields 2 -- 2.5x boost to the coverage of risk outcomes and tool-calling trajectories. Our distilled 8B red-teamer model improves attack success rate by 100%, surpassing the 671B Deepseek-R1 model. Our ablations and analyses validate the effectiveness of the iterative framework, structured reasoning, and the generalization of our red-teamer models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25761v1">DiagramEval: Evaluating LLM-Generated Diagrams via Graphs</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Diagrams play a central role in research papers for conveying ideas, yet they are often notoriously complex and labor-intensive to create. Although diagrams are presented as images, standard image generative models struggle to produce clear diagrams with well-defined structure. We argue that a promising direction is to generate demonstration diagrams directly in textual form as SVGs, which can leverage recent advances in large language models (LLMs). However, due to the complexity of components and the multimodal nature of diagrams, sufficiently discriminative and explainable metrics for evaluating the quality of LLM-generated diagrams remain lacking. In this paper, we propose DiagramEval, a novel evaluation metric designed to assess demonstration diagrams generated by LLMs. Specifically, DiagramEval conceptualizes diagrams as graphs, treating text elements as nodes and their connections as directed edges, and evaluates diagram quality using two new groups of metrics: node alignment and path alignment. For the first time, we effectively evaluate diagrams produced by state-of-the-art LLMs on recent research literature, quantitatively demonstrating the validity of our metrics. Furthermore, we show how the enhanced explainability of our proposed metrics offers valuable insights into the characteristics of LLM-generated diagrams. Code: https://github.com/ulab-uiuc/diagram-eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25732v1">The Limits of Obliviate: Evaluating Unlearning in LLMs via Stimulus-Knowledge Entanglement-Behavior Framework</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 14 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Unlearning in large language models (LLMs) is crucial for managing sensitive data and correcting misinformation, yet evaluating its effectiveness remains an open problem. We investigate whether persuasive prompting can recall factual knowledge from deliberately unlearned LLMs across models ranging from 2.7B to 13B parameters (OPT-2.7B, LLaMA-2-7B, LLaMA-3.1-8B, LLaMA-2-13B). Drawing from ACT-R and Hebbian theory (spreading activation theories), as well as communication principles, we introduce Stimulus-Knowledge Entanglement-Behavior Framework (SKeB), which models information entanglement via domain graphs and tests whether factual recall in unlearned models is correlated with persuasive framing. We develop entanglement metrics to quantify knowledge activation patterns and evaluate factuality, non-factuality, and hallucination in outputs. Our results show persuasive prompts substantially enhance factual knowledge recall (14.8% baseline vs. 24.5% with authority framing), with effectiveness inversely correlated to model size (128% recovery in 2.7B vs. 15% in 13B). SKeB provides a foundation for assessing unlearning completeness, robustness, and overall behavior in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23722v2">LLMs are Better Than You Think: Label-Guided In-Context Learning for Named Entity Recognition</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      In-context learning (ICL) enables large language models (LLMs) to perform new tasks using only a few demonstrations. However, in Named Entity Recognition (NER), existing ICL methods typically rely on task-agnostic semantic similarity for demonstration retrieval, which often yields less relevant examples and leads to inferior results. We introduce DEER, a training-free ICL approach that enables LLMs to make more informed entity predictions through the use of label-grounded statistics. DEER leverages token-level statistics from training labels to identify tokens most informative for entity recognition, enabling entity-focused demonstrations. It further uses these statistics to detect and refine error-prone tokens through a targeted reflection step. Evaluated on five NER datasets across four LLMs, DEER consistently outperforms existing ICL methods and achieves performance comparable to supervised fine-tuning. Further analyses demonstrate that DEER improves example retrieval, remains effective on both seen and unseen entities, and exhibits strong robustness in low-resource settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25701v1">Interpreting LLMs as Credit Risk Classifiers: Do Their Feature Explanations Align with Classical ML?</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 8 pages, 6 figures, 3 tables, CIKM 2025 FinFAI workshop
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly explored as flexible alternatives to classical machine learning models for classification tasks through zero-shot prompting. However, their suitability for structured tabular data remains underexplored, especially in high-stakes financial applications such as financial risk assessment. This study conducts a systematic comparison between zero-shot LLM-based classifiers and LightGBM, a state-of-the-art gradient-boosting model, on a real-world loan default prediction task. We evaluate their predictive performance, analyze feature attributions using SHAP, and assess the reliability of LLM-generated self-explanations. While LLMs are able to identify key financial risk indicators, their feature importance rankings diverge notably from LightGBM, and their self-explanations often fail to align with empirical SHAP attributions. These findings highlight the limitations of LLMs as standalone models for structured financial risk prediction and raise concerns about the trustworthiness of their self-generated explanations. Our results underscore the need for explainability audits, baseline comparisons with interpretable models, and human-in-the-loop oversight when deploying LLMs in risk-sensitive financial environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25662v1">User Misconceptions of LLM-Based Conversational Programming Assistants</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Programming assistants powered by large language models (LLMs) have become widely available, with conversational assistants like ChatGPT proving particularly accessible to less experienced programmers. However, the varied capabilities of these tools across model versions and the mixed availability of extensions that enable web search, code execution, or retrieval-augmented generation create opportunities for user misconceptions about what systems can and cannot do. Such misconceptions may lead to over-reliance, unproductive practices, or insufficient quality control in LLM-assisted programming. Here, we aim to characterize misconceptions that users of conversational LLM-based assistants may have in programming contexts. Using a two-phase approach, we first brainstorm and catalog user misconceptions that may occur, and then conduct a qualitative analysis to examine whether these conceptual issues surface in naturalistic Python-programming conversations with an LLM-based chatbot drawn from an openly available dataset. Indeed, we see evidence that some users have misplaced expectations about the availability of LLM-based chatbot features like web access, code execution, or non-text output generation. We also see potential evidence for deeper conceptual issues around the scope of information required to debug, validate, and optimize programs. Our findings reinforce the need for designing LLM-based tools that more clearly communicate their programming capabilities to users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14785v2">Exploring the In-Context Learning Capabilities of LLMs for Money Laundering Detection in Financial Graphs</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Accepted at AI4FCF-ICDM 2025
    </div>
    <details class="paper-abstract">
      The complexity and interconnectivity of entities involved in money laundering demand investigative reasoning over graph-structured data. This paper explores the use of large language models (LLMs) as reasoning engines over localized subgraphs extracted from a financial knowledge graph. We propose a lightweight pipeline that retrieves k-hop neighborhoods around entities of interest, serializes them into structured text, and prompts an LLM via few-shot in-context learning to assess suspiciousness and generate justifications. Using synthetic anti-money laundering (AML) scenarios that reflect common laundering behaviors, we show that LLMs can emulate analyst-style logic, highlight red flags, and provide coherent explanations. While this study is exploratory, it illustrates the potential of LLM-based graph reasoning in AML and lays groundwork for explainable, language-driven financial crime analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23965v2">The Sign Estimator: LLM Alignment in the Face of Choice Heterogeneity</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Traditional LLM alignment methods are vulnerable to heterogeneity in human preferences. Fitting a na\"ive probabilistic model to pairwise comparison data (say over prompt-completion pairs) yields an inconsistent estimate of the population-average utility -a canonical measure of social welfare. We propose a new method, dubbed the sign estimator, that provides a simple, provably consistent, and efficient estimator by replacing cross-entropy with binary classification loss in the aggregation step. This simple modification recovers consistent ordinal alignment under mild assumptions and achieves the first polynomial finite-sample error bounds in this setting. In realistic simulations of LLM alignment using digital twins, the sign estimator substantially reduces preference distortion over a panel of simulated personas, cutting (angular) estimation error by nearly 35% and decreasing disagreement with true population preferences from 12% to 8% compared to standard RLHF. Our method also compares favorably to panel data heuristics that explicitly model user heterogeneity and require tracking individual-level preference data-all while maintaining the implementation simplicity of existing LLM alignment pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25595v1">Communication and Verification in LLM Agents towards Collaboration under Information Asymmetry</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Workshop on Multi-Agent System @ ICML 2025
    </div>
    <details class="paper-abstract">
      While Large Language Model (LLM) agents are often approached from the angle of action planning/generation to accomplish a goal (e.g., given by language descriptions), their abilities to collaborate with each other to achieve a joint goal are not well explored. To address this limitation, this paper studies LLM agents in task collaboration, particularly under the condition of information asymmetry, where agents have disparities in their knowledge and skills and need to work together to complete a shared task. We extend Einstein Puzzles, a classical symbolic puzzle, to a table-top game. In this game, two LLM agents must reason, communicate, and act to satisfy spatial and relational constraints required to solve the puzzle. We apply a fine-tuning-plus-verifier framework in which LLM agents are equipped with various communication strategies and verification signals from the environment. Empirical results highlight the critical importance of aligned communication, especially when agents possess both information-seeking and -providing capabilities. Interestingly, agents without communication can still achieve high task performance; however, further analysis reveals a lack of true rule understanding and lower trust from human evaluators. Instead, by integrating an environment-based verifier, we enhance agents' ability to comprehend task rules and complete tasks, promoting both safer and more interpretable collaboration in AI systems. https://github.com/Roihn/EinsteinPuzzles
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13738v2">HyMiRec: A Hybrid Multi-interest Learning Framework for LLM-based Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated strong potential for sequential recommendation. However, current LLM-based approaches face critical limitations in modeling users' long-term and diverse interests. First, due to inference latency and feature fetching bandwidth constraints, existing methods typically truncate user behavior sequences to include only the most recent interactions, resulting in the loss of valuable long-range preference signals. Second, most current methods rely on next-item prediction with a single predicted embedding, overlooking the multifaceted nature of user interests and limiting recommendation diversity. To address these challenges, we propose HyMiRec, a hybrid multi-interest sequential recommendation framework, which leverages a lightweight recommender to extracts coarse interest embeddings from long user sequences and an LLM-based recommender to captures refined interest embeddings. To alleviate the overhead of fetching features, we introduce a residual codebook based on cosine similarity, enabling efficient compression and reuse of user history embeddings. To model the diverse preferences of users, we design a disentangled multi-interest learning module, which leverages multiple interest queries to learn disentangles multiple interest signals adaptively, allowing the model to capture different facets of user intent. Extensive experiments are conducted on both benchmark datasets and a collected industrial dataset, demonstrating our effectiveness over existing state-of-the-art methods. Furthermore, online A/B testing shows that HyMiRec brings consistent improvements in real-world recommendation systems. Code is available at https://github.com/FireRedTeam/FireRedSeqRec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25588v1">Standardization of Psychiatric Diagnoses -- Role of Fine-tuned LLM Consortium and OpenAI-gpt-oss Reasoning LLM Enabled Decision Support System</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      The diagnosis of most mental disorders, including psychiatric evaluations, primarily depends on dialogues between psychiatrists and patients. This subjective process can lead to variability in diagnoses across clinicians and patients, resulting in inconsistencies and challenges in achieving reliable outcomes. To address these issues and standardize psychiatric diagnoses, we propose a Fine-Tuned Large Language Model (LLM) Consortium and OpenAI-gpt-oss Reasoning LLM-enabled Decision Support System for the clinical diagnosis of mental disorders. Our approach leverages fine-tuned LLMs trained on conversational datasets involving psychiatrist-patient interactions focused on mental health conditions (e.g., depression). The diagnostic predictions from individual models are aggregated through a consensus-based decision-making process, refined by the OpenAI-gpt-oss reasoning LLM. We propose a novel method for deploying LLM agents that orchestrate communication between the LLM consortium and the reasoning LLM, ensuring transparency, reliability, and responsible AI across the entire diagnostic workflow. Experimental results demonstrate the transformative potential of combining fine-tuned LLMs with a reasoning model to create a robust and highly accurate diagnostic system for mental health assessment. A prototype of the proposed platform, integrating three fine-tuned LLMs with the OpenAI-gpt-oss reasoning LLM, was developed in collaboration with the U.S. Army Medical Research Team in Norfolk, Virginia, USA. To the best of our knowledge, this work represents the first application of a fine-tuned LLM consortium integrated with a reasoning LLM for clinical mental health diagnosis paving the way for next-generation AI-powered eHealth systems aimed at standardizing psychiatric diagnoses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10940v3">Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 to be published in NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recommender systems filter contents/items valuable to users by inferring preferences from user features and historical behaviors. Mainstream approaches follow the learning-to-rank paradigm, which focus on discovering and modeling item topics (e.g., categories), and capturing user preferences on these topics based on historical interactions. However, this paradigm often neglects the modeling of user characteristics and their social roles, which are logical confounders influencing the correlated interest and user preference transition. To bridge this gap, we introduce the user role identification task and the behavioral logic modeling task that aim to explicitly model user roles and learn the logical relations between item topics and user social roles. We show that it is possible to explicitly solve these tasks through an efficient integration framework of Large Language Model (LLM) and recommendation systems, for which we propose TagCF. On the one hand, TagCF exploits the (Multi-modal) LLM's world knowledge and logic inference ability to extract realistic tag-based virtual logic graphs that reveal dynamic and expressive knowledge of users, refining our understanding of user behaviors. On the other hand, TagCF presents empirically effective integration modules that take advantage of the extracted tag-logic information, augmenting the recommendation performance. We conduct both online experiments and offline experiments with industrial and public datasets as verification of TagCF's effectiveness, and we empirically show that the user role modeling strategy is potentially a better choice than the modeling of item topics. Additionally, we provide evidence that the extracted logic graphs are empirically a general and transferable knowledge that can benefit a wide range of recommendation tasks. Our code is available in https://github.com/Code2Q/TagCF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15030v2">Collab-REC: An LLM-based Agentic Framework for Balancing Recommendations in Tourism</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      We propose Collab-REC, a multi-agent framework designed to counteract popularity bias and enhance diversity in tourism recommendations. In our setting, three LLM-based agents -- Personalization, Popularity, and Sustainability generate city suggestions from complementary perspectives. A non-LLM moderator then merges and refines these proposals via multi-round negotiation, ensuring each agent's viewpoint is incorporated while penalizing spurious or repeated responses. Experiments on European city queries show that Collab-REC improves diversity and overall relevance compared to a single-agent baseline, surfacing lesser-visited locales that often remain overlooked. This balanced, context-aware approach addresses over-tourism and better aligns with constraints provided by the user, highlighting the promise of multi-stakeholder collaboration in LLM-driven recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25536v1">TwinVoice: A Multi-dimensional Benchmark Towards Digital Twins via LLM Persona Simulation</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Main paper: 11 pages, 3 figures, 6 tables. Appendix: 28 pages. Bangde Du and Minghao Guo contributed equally. Corresponding authors: Ziyi Ye (ziyiye@fudan.edu.cn), Qingyao Ai (aiqy@tsinghua.edu.cn)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are exhibiting emergent human-like abilities and are increasingly envisioned as the foundation for simulating an individual's communication style, behavioral tendencies, and personality traits. However, current evaluations of LLM-based persona simulation remain limited: most rely on synthetic dialogues, lack systematic frameworks, and lack analysis of the capability requirement. To address these limitations, we introduce TwinVoice, a comprehensive benchmark for assessing persona simulation across diverse real-world contexts. TwinVoice encompasses three dimensions: Social Persona (public social interactions), Interpersonal Persona (private dialogues), and Narrative Persona (role-based expression). It further decomposes the evaluation of LLM performance into six fundamental capabilities, including opinion consistency, memory recall, logical reasoning, lexical fidelity, persona tone, and syntactic style. Experimental results reveal that while advanced models achieve moderate accuracy in persona simulation, they still fall short of capabilities such as syntactic style and memory recall. Consequently, the average performance achieved by LLMs remains considerably below the human baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00814v2">Many LLMs Are More Utilitarian Than One</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Accepted to the Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Moral judgment is integral to large language models' (LLMs) social reasoning. As multi-agent systems gain prominence, it becomes crucial to understand how LLMs function when collaborating compared to operating as individual agents. In human moral judgment, group deliberation leads to a Utilitarian Boost: a tendency to endorse norm violations that inflict harm but maximize benefits for the greatest number of people. We study whether a similar dynamic emerges in multi-agent LLM systems. We test six models on well-established sets of moral dilemmas across two conditions: (1) Solo, where models reason independently, and (2) Group, where they engage in multi-turn discussions in pairs or triads. In personal dilemmas, where agents decide whether to directly harm an individual for the benefit of others, all models rated moral violations as more acceptable when part of a group, demonstrating a Utilitarian Boost similar to that observed in humans. However, the mechanism for the Boost in LLMs differed: While humans in groups become more utilitarian due to heightened sensitivity to decision outcomes, LLM groups showed either reduced sensitivity to norms or enhanced impartiality. We report model differences in when and how strongly the Boost manifests. We also discuss prompt and agent compositions that enhance or mitigate the effect. We end with a discussion of the implications for AI alignment, multi-agent design, and artificial moral reasoning. Code available at: https://github.com/baltaci-r/MoralAgents
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25506v1">Reflections on the Reproducibility of Commercial LLM Performance in Empirical Software Engineering Studies</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Large Language Models have gained remarkable interest in industry and academia. The increasing interest in LLMs in academia is also reflected in the number of publications on this topic over the last years. For instance, alone 78 of the around 425 publications at ICSE 2024 performed experiments with LLMs. Conducting empirical studies with LLMs remains challenging and raises questions on how to achieve reproducible results, for both other researchers and practitioners. One important step towards excelling in empirical research on LLMs and their application is to first understand to what extent current research results are eventually reproducible and what factors may impede reproducibility. This investigation is within the scope of our work. We contribute an analysis of the reproducibility of LLM-centric studies, provide insights into the factors impeding reproducibility, and discuss suggestions on how to improve the current state. In particular, we studied the 86 articles describing LLM-centric studies, published at ICSE 2024 and ASE 2024. Of the 86 articles, 18 provided research artefacts and used OpenAI models. We attempted to replicate those 18 studies. Of the 18 studies, only five were fit for reproduction. For none of the five studies, we were able to fully reproduce the results. Two studies seemed to be partially reproducible, and three studies did not seem to be reproducible. Our results highlight not only the need for stricter research artefact evaluations but also for more robust study designs to ensure the reproducible value of future publications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.12993v2">A Multilingual, Large-Scale Study of the Interplay between LLM Safeguards, Personalisation, and Disinformation</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can generate human-like disinformation, yet their ability to personalise such content across languages and demographics remains underexplored. This study presents the first large-scale, multilingual analysis of persona-targeted disinformation generation by LLMs. Employing a red teaming methodology, we prompt eight state-of-the-art LLMs with 324 false narratives and 150 demographic personas (combinations of country, generation, and political orientation) across four languages--English, Russian, Portuguese, and Hindi--resulting in AI-TRAITS, a comprehensive dataset of 1.6 million personalised disinformation texts. Results show that the use of even simple personalisation prompts significantly increases the likelihood of jailbreaks across all studied LLMs, up to 10 percentage points, and alters linguistic and rhetorical patterns that enhance narrative persuasiveness. Models such as Grok and GPT exhibited jailbreak rates and personalisation scores both exceeding 85%. These insights expose critical vulnerabilities in current state-of-the-art LLMs and offer a foundation for improving safety alignment and detection strategies in multilingual and cross-demographic contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25472v1">NetEcho: From Real-World Streaming Side-Channels to Full LLM Conversation Recovery</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      In the rapidly expanding landscape of Large Language Model (LLM) applications, real-time output streaming has become the dominant interaction paradigm. While this enhances user experience, recent research reveals that it exposes a non-trivial attack surface through network side-channels. Adversaries can exploit patterns in encrypted traffic to infer sensitive information and reconstruct private conversations. In response, LLM providers and third-party services are deploying defenses such as traffic padding and obfuscation to mitigate these vulnerabilities. This paper starts by presenting a systematic analysis of contemporary side-channel defenses in mainstream LLM applications, with a focus on services from vendors like OpenAI and DeepSeek. We identify and examine seven representative deployment scenarios, each incorporating active/passive mitigation techniques. Despite these enhanced security measures, our investigation uncovers significant residual information that remains vulnerable to leakage within the network traffic. Building on this discovery, we introduce NetEcho, a novel, LLM-based framework that comprehensively unleashes the network side-channel risks of today's LLM applications. NetEcho is designed to recover entire conversations -- including both user prompts and LLM responses -- directly from encrypted network traffic. It features a deliberate design that ensures high-fidelity text recovery, transferability across different deployment scenarios, and moderate operational cost. In our evaluations on medical and legal applications built upon leading models like DeepSeek-v3 and GPT-4o, NetEcho can recover avg $\sim$70\% information of each conversation, demonstrating a critical limitation in current defense mechanisms. We conclude by discussing the implications of our findings and proposing future directions for augmenting network traffic security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25441v1">Grounded in Reality: Learning and Deploying Proactive LLM from Offline Logs</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 27 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel as passive responders, but teaching them to be proactive, goal-oriented partners, a critical capability in high-stakes domains, remains a major challenge. Current paradigms either myopically optimize single-turn attributes or rely on brittle, high-cost user simulators, creating a persistent ``reality gap''. To bridge this gap, we introduce \texttt{Learn-to-Ask}, a general, simulator-free framework for learning and deploying proactive dialogue agents \textit{directly from offline expert data}, bypassing the need to model complex user dynamics. Our key insight is to reframe the offline policy learning problem by leveraging the \textbf{observed future} of each expert trajectory. This allows us to infer a dense, turn-by-turn reward signal grounded in the expert's revealed strategy, decomposing the intractable long-horizon problem into a series of supervised learning tasks, and training a policy to output a structured \texttt{(action, state_assessment)} tuple, governing both \textbf{what to ask} and, crucially, \textbf{when to stop}. To ensure reward fidelity, our Automated Grader Calibration pipeline systematically purges noise from the LLM-based reward model with minimal human supervision. Empirically, we demonstrate the efficacy of \texttt{Learn-to-Ask} in a real-world medical dataset, using LLMs of varying sizes up to 32B. Our approach culminates in the successful deployment of LLMs into a live, large-scale online AI service. In rigorous in-house evaluations, our model was launched and achieved performance even superior to human experts, proving our framework's ability to translate offline data into tangible, real-world impact. We hope this work provides a practical and economically viable blueprint for transforming passive LLMs into proactive, goal-oriented LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25432v1">Depth and Autonomy: A Framework for Evaluating LLM Applications in Social Science Research</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Presented at the Annual Meeting of the American Political Science Association, Vancouver, BC, September 11--14 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly utilized by researchers across a wide range of domains, and qualitative social science is no exception; however, this adoption faces persistent challenges, including interpretive bias, low reliability, and weak auditability. We introduce a framework that situates LLM usage along two dimensions, interpretive depth and autonomy, thereby offering a straightforward way to classify LLM applications in qualitative research and to derive practical design recommendations. We present the state of the literature with respect to these two dimensions, based on all published social science papers available on Web of Science that use LLMs as a tool and not strictly as the subject of study. Rather than granting models expansive freedom, our approach encourages researchers to decompose tasks into manageable segments, much as they would when delegating work to capable undergraduate research assistants. By maintaining low levels of autonomy and selectively increasing interpretive depth only where warranted and under supervision, one can plausibly reap the benefits of LLMs while preserving transparency and reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25426v1">Implicature in Interaction: Understanding Implicature Improves Alignment in Human-LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 The manuscript is approximately 7360 words and contains 12 figures and 6 tables
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) is positioning language at the core of human-computer interaction (HCI). We argue that advancing HCI requires attention to the linguistic foundations of interaction, particularly implicature (meaning conveyed beyond explicit statements through shared context) which is essential for human-AI (HAI) alignment. This study examines LLMs' ability to infer user intent embedded in context-driven prompts and whether understanding implicature improves response generation. Results show that larger models approximate human interpretations more closely, while smaller models struggle with implicature inference. Furthermore, implicature-based prompts significantly enhance the perceived relevance and quality of responses across models, with notable gains in smaller models. Overall, 67.6% of participants preferred responses with implicature-embedded prompts to literal ones, highlighting a clear preference for contextually nuanced communication. Our work contributes to understanding how linguistic theory can be used to address the alignment problem by making HAI interaction more natural and contextually grounded.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25421v1">Small Talk, Big Impact? LLM-based Conversational Agents to Mitigate Passive Fatigue in Conditional Automated Driving</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Submitted to CHI '26 Conference on Human Factors in Computing Systems
    </div>
    <details class="paper-abstract">
      Passive fatigue during conditional automated driving can compromise driver readiness and safety. This paper presents findings from a test-track study with 40 participants in a real-world rural automated driving scenario. In this scenario, a Large Language Model (LLM) based conversational agent (CA) was designed to check in with drivers and re-engage them with their surroundings. Drawing on in-car video recordings, sleepiness ratings and interviews, we analysed how drivers interacted with the agent and how these interactions shaped alertness. Users found the CA helpful for supporting vigilance during passive fatigue. Thematic analysis of acceptability further revealed three user preference profiles that implicate future intention to use CAs. Positioning empirically observed profiles within existing CA archetype frameworks highlights the need for adaptive design sensitive to diverse user groups. This work underscores the potential of CAs as proactive Human-Machine Interface (HMI) interventions, demonstrating how natural language can support context-aware interaction during automated driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25404v1">GPTOpt: Towards Efficient LLM-Based Black-Box Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Global optimization of expensive, derivative-free black-box functions demands extreme sample efficiency. Classical methods such as Bayesian Optimization (BO) can be effective, but they often require careful parameter tuning to each application domain. At the same time, Large Language Models (LLMs) have shown broad capabilities, yet state-of-the-art models remain limited in solving continuous black-box optimization tasks. We introduce GPTOpt, an LLM-based optimization method that equips LLMs with continuous black-box optimization capabilities. By fine-tuning large language models on extensive synthetic datasets derived from diverse BO parameterizations, GPTOpt leverages LLM pre-training to generalize across optimization tasks. On a variety of black-box optimization benchmarks, GPTOpt surpasses traditional optimizers, highlighting the capacity of LLMs for advanced numerical reasoning and introducing a flexible framework for global optimization without parameter tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25370v1">Monitoring Transformative Technological Convergence Through LLM-Extracted Semantic Entity Triple Graphs</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Forecasting transformative technologies remains a critical but challenging task, particularly in fast-evolving domains such as Information and Communication Technologies (ICTs). Traditional expert-based methods struggle to keep pace with short innovation cycles and ambiguous early-stage terminology. In this work, we propose a novel, data-driven pipeline to monitor the emergence of transformative technologies by identifying patterns of technological convergence. Our approach leverages advances in Large Language Models (LLMs) to extract semantic triples from unstructured text and construct a large-scale graph of technology-related entities and relations. We introduce a new method for grouping semantically similar technology terms (noun stapling) and develop graph-based metrics to detect convergence signals. The pipeline includes multi-stage filtering, domain-specific keyword clustering, and a temporal trend analysis of topic co-occurence. We validate our methodology on two complementary datasets: 278,625 arXiv preprints (2017--2024) to capture early scientific signals, and 9,793 USPTO patent applications (2018-2024) to track downstream commercial developments. Our results demonstrate that the proposed pipeline can identify both established and emerging convergence patterns, offering a scalable and generalizable framework for technology forecasting grounded in full-text analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25356v1">Not ready for the bench: LLM legal interpretation is unstable and out of step with human judgments</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Legal interpretation frequently involves assessing how a legal text, as understood by an 'ordinary' speaker of the language, applies to the set of facts characterizing a legal dispute in the U.S. judicial system. Recent scholarship has proposed that legal practitioners add large language models (LLMs) to their interpretive toolkit. This work offers an empirical argument against LLM interpretation as recently practiced by legal scholars and federal judges. Our investigation in English shows that models do not provide stable interpretive judgments: varying the question format can lead the model to wildly different conclusions. Moreover, the models show weak to moderate correlation with human judgment, with large variance across model and question variant, suggesting that it is dangerous to give much credence to the conclusions produced by generative AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25297v1">Understanding the Characteristics of LLM-Generated Property-Based Tests in Exploring Edge Cases</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Accepted for publication in 2nd IEEE/ACM international conference on AI-powered Software (AIware 2025) : 8 pages, 1 table, 8 figures
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly generate code in software development, ensuring the quality of LLM-generated code has become important. Traditional testing approaches using Example-based Testing (EBT) often miss edge cases -- defects that occur at boundary values, special input patterns, or extreme conditions. This research investigates the characteristics of LLM-generated Property-based Testing (PBT) compared to EBT for exploring edge cases. We analyze 16 HumanEval problems where standard solutions failed on extended test cases, generating both PBT and EBT test codes using Claude-4-sonnet. Our experimental results reveal that while each method individually achieved a 68.75\% bug detection rate, combining both approaches improved detection to 81.25\%. The analysis demonstrates complementary characteristics: PBT effectively detects performance issues and edge cases through extensive input space exploration, while EBT effectively detects specific boundary conditions and special patterns. These findings suggest that a hybrid approach leveraging both testing methods can improve the reliability of LLM-generated code, providing guidance for test generation strategies in LLM-based code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05493v2">Can LLMs Outshine Conventional Recommenders? A Comparative Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 NeurIPS 2025 DB Track Accepted Paper
    </div>
    <details class="paper-abstract">
      In recent years, integrating large language models (LLMs) into recommender systems has created new opportunities for improving recommendation quality. However, a comprehensive benchmark is needed to thoroughly evaluate and compare the recommendation capabilities of LLMs with traditional recommender systems. In this paper, we introduce RecBench, which systematically investigates various item representation forms (including unique identifier, text, semantic embedding, and semantic identifier) and evaluates two primary recommendation tasks, i.e., click-through rate prediction (CTR) and sequential recommendation (SeqRec). Our extensive experiments cover up to 17 large models and are conducted across five diverse datasets from fashion, news, video, books, and music domains. Our findings indicate that LLM-based recommenders outperform conventional recommenders, achieving up to a 5% AUC improvement in the CTR scenario and up to a 170% NDCG@10 improvement in the SeqRec scenario. However, these substantial performance gains come at the expense of significantly reduced inference efficiency, rendering the LLM-as-RS paradigm impractical for real-time recommendation environments. We aim for our findings to inspire future research, including recommendation-specific model acceleration methods. We will release our code, data, configurations, and platform to enable other researchers to reproduce and build upon our experimental results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22967v2">MAD-Fact: A Multi-Agent Debate Framework for Long-Form Factuality Evaluation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 The article has been accepted by Frontiers of Computer Science (FCS), with the DOI: {10.1007/s11704-025-51369-x}
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) raises critical concerns about the factual accuracy of their outputs, especially in high-risk domains such as biomedicine, law, and education. Existing evaluation methods for short texts often fail on long-form content due to complex reasoning chains, intertwined perspectives, and cumulative information. To address this, we propose a systematic approach integrating large-scale long-form datasets, multi-agent verification mechanisms, and weighted evaluation metrics. We construct LongHalluQA, a Chinese long-form factuality dataset; and develop MAD-Fact, a debate-based multi-agent verification system. We introduce a fact importance hierarchy to capture the varying significance of claims in long-form texts. Experiments on two benchmarks show that larger LLMs generally maintain higher factual consistency, while domestic models excel on Chinese content. Our work provides a structured framework for evaluating and enhancing factual reliability in long-form LLM outputs, guiding their safe deployment in sensitive domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.19902v2">WEST: LLM based Speech Toolkit for Speech Understanding, Generation, and Interaction</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      In this paper, we present WEST(WE Speech Toolkit), a speech toolkit based on a large language model (LLM) for speech understanding, generation, and interaction. There are three key features of WEST: 1) Fully LLM-based: Standing on the shoulders of giants by reusing mature architectures, ecosystems (e.g., Hugging Face), and methods (e.g., sequence packing) from large models. 2) Full-stack: Supports tasks such as recognition, synthesis, understanding, dialogue, and multimodal capabilities, with extensibility to incorporate open-source models. 3) Simple and Stupid: A simple and stupid speech toolkit that everyone can Touch. In addition, WEST provides two types of recipes, models, and experimental results. The first is entirely based on open-source models and open-source data, allowing users to fully reproduce the experiments in this paper and serving as a verification system or minimal system baseline. The second is trained on massive data, offering superior performance so the user can directly apply it out of the box. WEST is publicly avilable at https://github.com/wenet-e2e/west/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09158v2">Augmenting Dialog with Think-Aloud Utterances for Modeling Individual Personality Traits by LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 8 pages, 1 figure. Accepted at the First Workshop on Tailoring AI: Exploring Active and Passive LLM Personalization (PALS2025@EMNLP2025)
    </div>
    <details class="paper-abstract">
      This study proposes augmenting dialog data with think-aloud utterances (TAUs) for modeling individual personalities in text chat by LLM. TAU is a verbalization of a speaker's thought before articulating the utterance. We expect "persona LLMs" trained with TAU-augmented data can mimic the speaker's personality trait better. We tested whether the trained persona LLMs obtain the human personality with respect to Big Five, a framework characterizing human personality traits from five aspects. The results showed that LLMs trained with TAU-augmented data more closely align to the speakers' Agreeableness and Neuroticism of Big Five than those trained with original dialog data. We also found that the quality of TAU-augmentation impacts persona LLM's performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02547v2">The Landscape of Agentic Reinforcement Learning for LLMs: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      The emergence of agentic reinforcement learning (Agentic RL) marks a paradigm shift from conventional reinforcement learning applied to large language models (LLM RL), reframing LLMs from passive sequence generators into autonomous, decision-making agents embedded in complex, dynamic worlds. This survey formalizes this conceptual shift by contrasting the degenerate single-step Markov Decision Processes (MDPs) of LLM-RL with the temporally extended, partially observable Markov decision processes (POMDPs) that define Agentic RL. Building on this foundation, we propose a comprehensive twofold taxonomy: one organized around core agentic capabilities, including planning, tool use, memory, reasoning, self-improvement, and perception, and the other around their applications across diverse task domains. Central to our thesis is that reinforcement learning serves as the critical mechanism for transforming these capabilities from static, heuristic modules into adaptive, robust agentic behavior. To support and accelerate future research, we consolidate the landscape of open-source environments, benchmarks, and frameworks into a practical compendium. By synthesizing over five hundred recent works, this survey charts the contours of this rapidly evolving field and highlights the opportunities and challenges that will shape the development of scalable, general-purpose AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.06426v2">S'MoRE: Structural Mixture of Residual Experts for Parameter-Efficient LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Fine-tuning pre-trained large language models (LLMs) presents a dual challenge of balancing parameter efficiency and model capacity. Existing methods like low-rank adaptations (LoRA) are efficient but lack flexibility, while Mixture-of-Experts (MoE) enhance model capacity at the cost of more & under-utilized parameters. To address these limitations, we propose Structural Mixture of Residual Experts (S'MoRE), a novel framework that seamlessly integrates the efficiency of LoRA with the flexibility of MoE. Conceptually, S'MoRE employs hierarchical low-rank decomposition of expert weights, yielding residuals of varying orders interconnected in a multi-layer structure. By routing input tokens through sub-trees of residuals, S'MoRE emulates the capacity of numerous experts by instantiating and assembling just a few low-rank matrices. We craft the inter-layer propagation of S'MoRE's residuals as a special type of Graph Neural Network (GNN), and prove that under similar parameter budget, S'MoRE improves structural flexibility of traditional MoE (or Mixture-of-LoRA) by exponential order. Comprehensive theoretical analysis and empirical results demonstrate that S'MoRE achieves superior fine-tuning performance, offering a transformative approach for efficient LLM adaptation. Our implementation is available at: https://github.com/ZimpleX/SMoRE-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25187v1">Testing Cross-Lingual Text Comprehension In LLMs Using Next Sentence Prediction</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      While large language models are trained on massive datasets, this data is heavily skewed towards English. Does their impressive performance reflect genuine ability or just this data advantage? To find out, we tested them in a setting where they could not rely on data abundance: low-resource languages. Building on prior work Agarwal et al. (2025) that used Next Sentence Prediction (NSP) as a test, we created a large-scale benchmark with 10,000 questions each for English (a high-resource language), Swahili (medium-resource), and Hausa (low-resource). We then tested several top models, including GPT-4 Turbo, Gemini 1.5 Flash, and LLaMA 3 70B, to see how their performance holds up. The results painted a clear picture of how levels of language resources impact outcomes. While all models excelled in English, their accuracy dropped in Swahili and fell sharply in Hausa, with LLaMA 3 struggling the most. The story became even more interesting when we introduced Chain-of-Thought (CoT) prompting. For the struggling LLaMA 3, CoT acted as a helpful guide, significantly boosting its accuracy. However, for the more capable GPT-4 and Gemini, the same technique often backfired, leading to a kind of "overthinking" that hurt their results in the cross-lingual context. This reveals that Chain-of-Thought is not a universal solution; its effectiveness depends heavily on the model's baseline capability and the specific context of the task. Our framework pinpoints LLM weaknesses, highlights when CoT helps or hinders cross-lingual NSP performance, and factors influencing their decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.06658v2">Can We Hide Machines in the Crowd? Quantifying Equivalence in LLM-in-the-loop Annotation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Accepted at SIGIR-AP 2025
    </div>
    <details class="paper-abstract">
      Many evaluations of large language models (LLMs) in text annotation focus primarily on the correctness of the output, typically comparing model-generated labels to human-annotated ``ground truth'' using standard performance metrics. In contrast, our study moves beyond effectiveness alone. We aim to explore how labeling decisions -- by both humans and LLMs -- can be statistically evaluated across individuals. Rather than treating LLMs purely as annotation systems, we approach LLMs as an alternative annotation mechanism that may be capable of mimicking the subjective judgments made by humans. To assess this, we develop a statistical evaluation method based on Krippendorff's $\alpha$, paired bootstrapping, and the Two One-Sided t-Tests (TOST) equivalence test procedure. This evaluation method tests whether an LLM can blend into a group of human annotators without being distinguishable. We apply this approach to two datasets -- MovieLens 100K and PolitiFact -- and find that the LLM is statistically indistinguishable from a human annotator in the former ($p = 0.004$), but not in the latter ($p = 0.155$), highlighting task-dependent differences. It also enables early evaluation on a small sample of human data to inform whether LLMs are suitable for large-scale annotation in a given application.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01617v3">AMAS: Adaptively Determining Communication Topology for LLM-based Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Accepted by EMNLP-2025
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have revolutionized natural language processing capabilities, their practical implementation as autonomous multi-agent systems (MAS) for industrial problem-solving encounters persistent barriers. Conventional MAS architectures are fundamentally restricted by inflexible, hand-crafted graph topologies that lack contextual responsiveness, resulting in diminished efficacy across varied academic and commercial workloads. To surmount these constraints, we introduce AMAS, a paradigm-shifting framework that redefines LLM-based MAS through a novel dynamic graph designer. This component autonomously identifies task-specific optimal graph configurations via lightweight LLM adaptation, eliminating the reliance on monolithic, universally applied structural templates. Instead, AMAS exploits the intrinsic properties of individual inputs to intelligently direct query trajectories through task-optimized agent pathways. Rigorous validation across question answering, mathematical deduction, and code generation benchmarks confirms that AMAS systematically exceeds state-of-the-art single-agent and multi-agent approaches across diverse LLM architectures. Our investigation establishes that context-sensitive structural adaptability constitutes a foundational requirement for high-performance LLM MAS deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15721v4">Bohdi: Heterogeneous LLM Fusion with Automatic Data Exploration</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Accepted by NeurIPS2025
    </div>
    <details class="paper-abstract">
      Heterogeneous Large Language Model (LLM) fusion integrates the strengths of multiple source LLMs with different architectures into a target LLM with low computational overhead. While promising, existing methods suffer from two major limitations: 1) reliance on real data from limited domain for knowledge fusion, preventing the target LLM from fully acquiring knowledge across diverse domains, and 2) fixed data allocation proportions across domains, failing to dynamically adjust according to the target LLM's varying capabilities across domains, leading to a capability imbalance. To overcome these limitations, we propose Bohdi, a synthetic-data-only heterogeneous LLM fusion framework. Through the organization of knowledge domains into a hierarchical tree structure, Bohdi enables automatic domain exploration and multi-domain data generation through multi-model collaboration, thereby comprehensively extracting knowledge from source LLMs. By formalizing domain expansion and data sampling proportion allocation on the knowledge tree as a Hierarchical Multi-Armed Bandit problem, Bohdi leverages the designed DynaBranches mechanism to adaptively adjust sampling proportions based on the target LLM's performance feedback across domains. Integrated with our proposed Introspection-Rebirth (IR) mechanism, DynaBranches dynamically tracks capability shifts during target LLM's updates via Sliding Window Binomial Likelihood Ratio Testing (SWBLRT), further enhancing its online adaptation capability. Comparative experimental results on a comprehensive suite of benchmarks demonstrate that Bohdi significantly outperforms existing baselines on multiple target LLMs, exhibits higher data efficiency, and virtually eliminates the imbalance in the target LLM's capabilities. Our code is available at https://github.com/gjq100/Bohdi.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25110v1">DEBATE: A Large-Scale Benchmark for Role-Playing LLM Agents in Multi-Agent, Long-Form Debates</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Accurately modeling opinion change through social interactions is crucial for addressing issues like misinformation and polarization. While role-playing large language models (LLMs) offer a promising way to simulate human-like interactions, existing research shows that single-agent alignment does not guarantee authentic multi-agent group dynamics. Current LLM role-play setups often produce unnatural dynamics (e.g., premature convergence), without an empirical benchmark to measure authentic human opinion trajectories. To bridge this gap, we introduce DEBATE, the first large-scale empirical benchmark explicitly designed to evaluate the authenticity of the interaction between multi-agent role-playing LLMs. DEBATE contains 29,417 messages from multi-round debate conversations among over 2,792 U.S.-based participants discussing 107 controversial topics, capturing both publicly-expressed messages and privately-reported opinions. Using DEBATE, we systematically evaluate and identify critical discrepancies between simulated and authentic group dynamics. We further demonstrate DEBATE's utility for aligning LLMs with human behavior through supervised fine-tuning, achieving improvements in surface-level metrics (e.g., ROUGE-L and message length) while highlighting limitations in deeper semantic alignment (e.g., semantic similarity). Our findings highlight both the potential and current limitations of role-playing LLM agents for realistically simulating human-like social dynamics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25103v1">Adaptive Proof Refinement with LLM-Guided Strategy Selection</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 11 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Formal verification via theorem proving enables the expressive specification and rigorous proof of software correctness, but it is difficult to scale due to the significant manual effort and expertise required. While Large Language Models (LLMs) show potential in proof generation, they frequently produce incorrect proofs on the first attempt and require additional strategies for iterative refinement. However, existing approaches employ fixed refinement strategies and cannot dynamically choose an effective strategy based on the particular issues in a generated proof, which limits their performance. To overcome this limitation, we introduce Adapt, a novel proof refinement framework that leverages an LLM-guided decision-maker to dynamically select a suitable refinement strategy according to the state of the proof assistant and available context of an incorrect proof. We evaluate Adapt on two benchmarks against four existing methods and find that it significantly outperforms the best baseline on both by proving 16.63% and 18.58% more theorems, respectively. Furthermore, we demonstrate Adapt's generalizability by evaluating it across five different LLMs. We also conduct ablation studies to measure the contribution of each component and compare the trade-offs of alternative decision-maker designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25093v1">Continual Low-Rank Adapters for LLM-based Generative Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) achieve strong performance in recommendation, they face challenges in continual learning as users, items, and user preferences evolve over time. Existing LoRA-based continual methods primarily focus on preserving performance on previous tasks, but this overlooks the unique nature of recommendation: the goal is not to predict past preferences, and outdated preferences can even harm performance when current interests shift significantly. To address this, we propose PESO (Proximally rEgularized Single evolving lOra, a continual adaptation method for LoRA in recommendation. PESO introduces a proximal regularizer that anchors the current adapter to its most recent frozen state, enabling the model to flexibly balance adaptation and preservation, and to better capture recent user behaviors. Theoretically, we show that this proximal design provides data-aware, direction-wise guidance in the LoRA subspace. Empirically, PESO consistently outperforms existing LoRA-based continual learning methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25092v1">SeeingEye: Agentic Information Flow Unlocks Multimodal Reasoning In Text-only LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Recent advances in text-only large language models (LLMs), such as DeepSeek-R1, demonstrate remarkable reasoning ability. However, these models remain fragile or entirely incapable when extended to multi-modal tasks. Existing approaches largely rely on single-form captions, which lack diversity and often fail to adapt across different types of Visual Question Answering (VQA) benchmarks. As a result, they provide no principled or efficient channel for transmitting fine-grained visual information. We introduce Seeing Eye, a modular framework that unlocks multimodal reasoning in text-only LLMs through an agent-based small VLM translator. This translator acts as a perception agent: it can invoke specialized tools (e.g., OCR and crop) and iteratively distill multimodal inputs into structured intermediate representations (SIRs) tailored to the question. These SIRs are then passed to the text-only LLM, which serves as a reasoning agent. Crucially, the translator and reasoner engage in multi-round feedback and interaction, enabling the extraction of targeted visual details and yielding more confident answers. Experiments on knowledge-intensive VQA benchmarks, including MMMU and MIA-Bench, demonstrate that Seeing Eye not only reduces inference cost but also surpasses much larger end-to-end VLMs. For example, an instantiation combining a 3B-parameter vision translator with an 8B-parameter language reasoner outperforms a monolithic 32B VLM on challenging knowledge-based questions. Our results highlight that decoupling perception from reasoning via agent information flow offers a scalable and plug-and-play pathway to multimodal reasoning, allowing strong text-only LLMs to fully leverage their reasoning capabilities. Code is available at: https://github.com/ulab-uiuc/SeeingEye
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25091v1">H3M-SSMoEs: Hypergraph-based Multimodal Learning with LLM Reasoning and Style-Structured Mixture of Experts</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Stock movement prediction remains fundamentally challenging due to complex temporal dependencies, heterogeneous modalities, and dynamically evolving inter-stock relationships. Existing approaches often fail to unify structural, semantic, and regime-adaptive modeling within a scalable framework. This work introduces H3M-SSMoEs, a novel Hypergraph-based MultiModal architecture with LLM reasoning and Style-Structured Mixture of Experts, integrating three key innovations: (1) a Multi-Context Multimodal Hypergraph that hierarchically captures fine-grained spatiotemporal dynamics via a Local Context Hypergraph (LCH) and persistent inter-stock dependencies through a Global Context Hypergraph (GCH), employing shared cross-modal hyperedges and Jensen-Shannon Divergence weighting mechanism for adaptive relational learning and cross-modal alignment; (2) a LLM-enhanced reasoning module, which leverages a frozen large language model with lightweight adapters to semantically fuse and align quantitative and textual modalities, enriching representations with domain-specific financial knowledge; and (3) a Style-Structured Mixture of Experts (SSMoEs) that combines shared market experts and industry-specialized experts, each parameterized by learnable style vectors enabling regime-aware specialization under sparse activation. Extensive experiments on three major stock markets demonstrate that H3M-SSMoEs surpasses state-of-the-art methods in both superior predictive accuracy and investment performance, while exhibiting effective risk control. Datasets, source code, and model weights are available at our GitHub repository: https://github.com/PeilinTime/H3M-SSMoEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25087v1">BioCoref: Benchmarking Biomedical Coreference Resolution with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Coreference resolution in biomedical texts presents unique challenges due to complex domain-specific terminology, high ambiguity in mention forms, and long-distance dependencies between coreferring expressions. In this work, we present a comprehensive evaluation of generative large language models (LLMs) for coreference resolution in the biomedical domain. Using the CRAFT corpus as our benchmark, we assess the LLMs' performance with four prompting experiments that vary in their use of local, contextual enrichment, and domain-specific cues such as abbreviations and entity dictionaries. We benchmark these approaches against a discriminative span-based encoder, SpanBERT, to compare the efficacy of generative versus discriminative methods. Our results demonstrate that while LLMs exhibit strong surface-level coreference capabilities, especially when supplemented with domain-grounding prompts, their performance remains sensitive to long-range context and mentions ambiguity. Notably, the LLaMA 8B and 17B models show superior precision and F1 scores under entity-augmented prompting, highlighting the potential of lightweight prompt engineering for enhancing LLM utility in biomedical NLP tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.23874v2">From Stochasticity to Signal: A Bayesian Latent State Model for Reliable Measurement with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate classification tasks in business, such as analyzing customer satisfaction from text. However, the inherent stochasticity of LLMs, in terms of their tendency to produce different outputs for the same input, creates a significant measurement error problem that is often neglected with a single round of output, or addressed with ad-hoc methods like majority voting. Such naive approaches fail to quantify uncertainty and can produce biased estimates of population-level metrics. In this paper, we propose a principled solution by reframing LLM variability as a statistical measurement error problem and introducing a Bayesian latent state model to address it. Our model treats the true classification (e.g., customer dissatisfaction) as an unobserved latent variable and the multiple LLM ratings as noisy measurements of this state. This framework allows for the simultaneous estimation of the LLM's false positive and false negative error rates, the underlying base rate of the phenomenon in the population, the posterior probability of the true state for each individual observation, and the causal impact of a business intervention, if any, on the latent state. Through simulation studies, we demonstrate that our model accurately recovers true parameters where naive methods fail. We conclude that this methodology provides a general and reliable framework for converting noisy, probabilistic outputs from LLMs into accurate and actionable insights for scientific and business applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14205v3">DPRF: A Generalizable Dynamic Persona Refinement Framework for Optimizing Behavior Alignment Between Personalized LLM Role-Playing Agents and Humans</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 In Submission
    </div>
    <details class="paper-abstract">
      The emerging large language model role-playing agents (LLM RPAs) aim to simulate individual human behaviors, but the persona fidelity is often undermined by manually-created profiles (e.g., cherry-picked information and personality characteristics) without validating the alignment with the target individuals. To address this limitation, our work introduces the Dynamic Persona Refinement Framework (DPRF). DPRF aims to optimize the alignment of LLM RPAs' behaviors with those of target individuals by iteratively identifying the cognitive divergence, either through free-form or theory-grounded, structured analysis, between generated behaviors and human ground truth, and refining the persona profile to mitigate these divergences. We evaluate DPRF with five LLMs on four diverse behavior-prediction scenarios: formal debates, social media posts with mental health issues, public interviews, and movie reviews. DPRF can consistently improve behavioral alignment considerably over baseline personas and generalizes across models and scenarios. Our work provides a robust methodology for creating high-fidelity persona profiles and enhancing the validity of downstream applications, such as user simulation, social studies, and personalized AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25064v1">Can LLMs Estimate Cognitive Complexity of Reading Comprehension Items?</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Estimating the cognitive complexity of reading comprehension (RC) items is crucial for assessing item difficulty before it is administered to learners. Unlike syntactic and semantic features, such as passage length or semantic similarity between options, cognitive features that arise during answer reasoning are not readily extractable using existing NLP tools and have traditionally relied on human annotation. In this study, we examine whether large language models (LLMs) can estimate the cognitive complexity of RC items by focusing on two dimensions-Evidence Scope and Transformation Level-that indicate the degree of cognitive burden involved in reasoning about the answer. Our experimental results demonstrate that LLMs can approximate the cognitive complexity of items, indicating their potential as tools for prior difficulty analysis. Further analysis reveals a gap between LLMs' reasoning ability and their metacognitive awareness: even when they produce correct answers, they sometimes fail to correctly identify the features underlying their own reasoning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13852v2">ConsistencyAI: A Benchmark to Assess LLMs' Factual Consistency When Responding to Different Demographic Groups</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 For associated code repository, see http://github.com/banyasp/consistencyAI For user-friendly web app, see http://v0-llm-comparison-webapp.vercel.app/
    </div>
    <details class="paper-abstract">
      Is an LLM telling you different facts than it's telling me? This paper introduces ConsistencyAI, an independent benchmark for measuring the factual consistency of large language models (LLMs) for different personas. ConsistencyAI tests whether, when users of different demographics ask identical questions, the model responds with factually inconsistent answers. Designed without involvement from LLM providers, this benchmark offers impartial evaluation and accountability. In our experiment, we queried 19 LLMs with prompts that requested 5 facts for each of 15 topics. We repeated this query 100 times for each LLM, each time adding prompt context from a different persona selected from a subset of personas modeling the general population. We processed the responses into sentence embeddings, computed cross-persona cosine similarity, and computed the weighted average of cross-persona cosine similarity to calculate factual consistency scores. In 100-persona experiments, scores ranged from 0.9065 to 0.7896, and the mean was 0.8656, which we adopt as a benchmark threshold. xAI's Grok-3 is most consistent, while several lightweight models rank lowest. Consistency varies by topic: the job market is least consistent, G7 world leaders most consistent, and issues like vaccines or the Israeli-Palestinian conflict diverge by provider. These results show that both the provider and the topic shape the factual consistency. We release our code and interactive demo to support reproducible evaluation and encourage persona-invariant prompting strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26027v1">Enhancing Temporal Understanding in Video-LLMs through Stacked Temporal Attention in Vision Encoders</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Despite significant advances in Multimodal Large Language Models (MLLMs), understanding complex temporal dynamics in videos remains a major challenge. Our experiments show that current Video Large Language Model (Video-LLM) architectures have critical limitations in temporal understanding, struggling with tasks that require detailed comprehension of action sequences and temporal progression. In this work, we propose a Video-LLM architecture that introduces stacked temporal attention modules directly within the vision encoder. This design incorporates a temporal attention in vision encoder, enabling the model to better capture the progression of actions and the relationships between frames before passing visual tokens to the LLM. Our results show that this approach significantly improves temporal reasoning and outperforms existing models in video question answering tasks, specifically in action recognition. We improve on benchmarks including VITATECS, MVBench, and Video-MME by up to +5.5%. By enhancing the vision encoder with temporal structure, we address a critical gap in video understanding for Video-LLMs. Project page and code are available at: https://alirasekh.github.io/STAVEQ2/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26024v1">Rethinking Cross-lingual Alignment: Balancing Transfer and Cultural Erasure in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Cross-lingual alignment (CLA) aims to align multilingual representations, enabling Large Language Models (LLMs) to seamlessly transfer knowledge across languages. While intuitive, we hypothesize, this pursuit of representational convergence can inadvertently cause "cultural erasure", the functional loss of providing culturally-situated responses that should diverge based on the query language. In this work, we systematically analyze this trade-off by introducing a holistic evaluation framework, the transfer-localization plane, which quantifies both desirable knowledge transfer and undesirable cultural erasure. Using this framework, we re-evaluate recent CLA approaches and find that they consistently improve factual transfer at the direct cost of cultural localization across all six languages studied. Our investigation into the internal representations of these models reveals a key insight: universal factual transfer and culturally-specific knowledge are optimally steerable at different model layers. Based on this finding, we propose Surgical Steering, a novel inference-time method that disentangles these two objectives. By applying targeted activation steering to distinct layers, our approach achieves a better balance between the two competing dimensions, effectively overcoming the limitations of current alignment techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26020v1">PORTool: Tool-Use LLM Training with Rewarded Tree</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Current tool-use large language models (LLMs) are trained on static datasets, enabling them to interact with external tools and perform multi-step, tool-integrated reasoning, which produces tool-call trajectories. However, these models imitate how a query is resolved in a generic tool-call routine, thereby failing to explore possible solutions and demonstrating limited performance in an evolved, dynamic tool-call environment. In this work, we propose PORTool, a reinforcement learning (RL) method that encourages a tool-use LLM to explore various trajectories yielding the correct answer. Specifically, this method starts with generating multiple rollouts for a given query, and some of them share the first few tool-call steps, thereby forming a tree-like structure. Next, we assign rewards to each step, based on its ability to produce a correct answer and make successful tool calls. A shared step across different trajectories receives the same reward, while different steps under the same fork receive different rewards. Finally, these step-wise rewards are used to calculate fork-relative advantages, blended with trajectory-relative advantages, to train the LLM for tool use. The experiments utilize 17 tools to address user queries, covering both time-sensitive and time-invariant topics. We conduct ablation studies to systematically justify the necessity and the design robustness of step-wise rewards. Furthermore, we compare the proposed PORTool with other training approaches and demonstrate significant improvements in final accuracy and the number of tool-call steps.
    </details>
</div>
