# llm - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
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
- [Part 20](papers_20.md)

## Papers

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
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25979v1">AttnCache: Accelerating Self-Attention Inference for LLM Prefill via Attention Cache</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 10 pages, 6 figures, submitted to Ninth Annual Conference on Machine Learning and Systems (MLSys'26)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used in generative applications such as chatting, code generation, and reasoning. However, many realworld workloads such as classification, question answering, recommendation, and text embedding rely solely on the prefill stage of inference, where the model encodes input sequences without performing autoregressive decoding. In these prefill only scenarios, the self-attention computation becomes the primary performance bottleneck due to its quadratic complexity with respect to sequence length. In this paper, we observe that semantically different sentences often produce similar attention maps across layers and heads. Building on this insight, we propose AttnCache, a framework that accelerates the prefill stage of LLM inference by retrieving and reusing similar attention maps. Based on an attention map memorization database, AttnCache employs efficient caching and similarity search techniques to identify and reuse pre-cached attention maps during inference, thereby reducing the computational overhead of self-attention. Experimental results show that AttnCache achieves an average of 1.2x end-to-end and 2x attention speedup on CPU, and 1.6x end-to-end and 3x attention speedup on GPU, with negligible accuracy degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25977v1">NeuronMM: High-Performance Matrix Multiplication for LLM Inference on AWS Trainium</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 12 pages, 8 figures, submitted to the Proceedings of the Twenty-First European Conference on Computer Systems (EuroSys'26)
    </div>
    <details class="paper-abstract">
      AI accelerators, customized to AI workloads, provide cost-effective and high-performance solutions for training and inference. Trainium, an AI accelerator recently developed by Amazon Web Services (AWS), provides an attractive option for LLM training and inference through its heterogeneous architecture. However, leveraging Trainium architecture for high performance can be challenging because of its systolic array architecture and special requirement on data layout. In this paper, we design high-performance matrix multiplication (matmul), a critical compute kernel, for LLM inference on Trainium. We introduce a series of techniques customized to Trainium based on kernel fusion and novel caching strategies to reduce data movement across the software-managed memory hierarchy, maximize SRAM bandwidth, and avoid expensive matrix transpose. Evaluating with nine datasets and four recent LLMs, we show that our system largely outperforms the state-of-the-art matmul implemented by AWS on Trainium: at the level of matmul kernel, it achieves an average 1.35x speedup (up to 2.22x), which translates to an average 1.66x speedup (up to 2.49x) for end-to-end LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14879v4">Vital Insight: Assisting Experts' Context-Driven Sensemaking of Multi-modal Personal Tracking Data Using Visualization and Human-In-The-Loop LLM</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Jiachen Li, Xiwen Li, Justin Steinberg, Akshat Choube, Bingsheng Yao, Xuhai Xu, Dakuo Wang, Elizabeth Mynatt, and Varun Mishra. 2025. Vital Insight: Assisting Experts' Context-Driven Sensemaking of Multi-modal Personal Tracking Data Using Visualization and Human-in-the-Loop LLM. Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 9, 3, Article 101 (September 2025), 37 pages
    </div>
    <details class="paper-abstract">
      Passive tracking methods, such as phone and wearable sensing, have become dominant in monitoring human behaviors in modern ubiquitous computing studies. While there have been significant advances in machine-learning approaches to translate periods of raw sensor data to model momentary behaviors, (e.g., physical activity recognition), there still remains a significant gap in the translation of these sensing streams into meaningful, high-level, context-aware insights that are required for various applications (e.g., summarizing an individual's daily routine). To bridge this gap, experts often need to employ a context-driven sensemaking process in real-world studies to derive insights. This process often requires manual effort and can be challenging even for experienced researchers due to the complexity of human behaviors. We conducted three rounds of user studies with 21 experts to explore solutions to address challenges with sensemaking. We follow a human-centered design process to identify needs and design, iterate, build, and evaluate Vital Insight (VI), a novel, LLM-assisted, prototype system to enable human-in-the-loop inference (sensemaking) and visualizations of multi-modal passive sensing data from smartphones and wearables. Using the prototype as a technology probe, we observe experts' interactions with it and develop an expert sensemaking model that explains how experts move between direct data representations and AI-supported inferences to explore, question, and validate insights. Through this iterative process, we also synthesize and discuss a list of design implications for the design of future AI-augmented visualization systems to better assist experts' sensemaking processes in multi-modal health sensing data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25941v1">RECAP: Reproducing Copyrighted Data from LLMs Training with an Agentic Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      If we cannot inspect the training data of a large language model (LLM), how can we ever know what it has seen? We believe the most compelling evidence arises when the model itself freely reproduces the target content. As such, we propose RECAP, an agentic pipeline designed to elicit and verify memorized training data from LLM outputs. At the heart of RECAP is a feedback-driven loop, where an initial extraction attempt is evaluated by a secondary language model, which compares the output against a reference passage and identifies discrepancies. These are then translated into minimal correction hints, which are fed back into the target model to guide subsequent generations. In addition, to address alignment-induced refusals, RECAP includes a jailbreaking module that detects and overcomes such barriers. We evaluate RECAP on EchoTrace, a new benchmark spanning over 30 full books, and the results show that RECAP leads to substantial gains over single-iteration approaches. For instance, with GPT-4.1, the average ROUGE-L score for the copyrighted text extraction improved from 0.38 to 0.47 - a nearly 24% increase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.19771v2">Beyond Reactivity: Measuring Proactive Problem Solving in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      LLM-based agents are increasingly moving towards proactivity: rather than awaiting instruction, they exercise agency to anticipate user needs and solve them autonomously. However, evaluating proactivity is challenging; current benchmarks are constrained to localized context, limiting their ability to test reasoning across sources and longer time horizons. To address this gap, we present PROBE (Proactive Resolution Of BottlEnecks). PROBE decomposes proactivity as a pipeline of three core capabilities: (1) searching for unspecified issues, (2) identifying specific bottlenecks, and (3) executing appropriate resolutions. We apply PROBE to evaluate leading LLMs and popular agentic frameworks, showing that even state-of-the-art models struggle to solve this benchmark. Computing our consistent measurements across frontier LLMs and agents, we find that the best end-to-end performance of 40% is achieved by both GPT-5 and Claude Opus-4.1. Additionally, we demonstrate the relative capabilities of each model and analyze mutual failure modes. Our results highlight the current limitations of autonomous action in agentic systems, and expose promising future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25939v1">SoK: Honeypots & LLMs, More Than the Sum of Their Parts?</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Systemization of Knowledge
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) promised to resolve the long-standing paradox in honeypot design: achieving high-fidelity deception with low operational risk. However, despite a flurry of research since late 2022, progress has been incremental, and the field lacks a cohesive understanding of the emerging architectural patterns, core challenges, and evaluation paradigms. To fill this gap, this Systematization of Knowledge (SoK) paper provides the first comprehensive overview of this new domain. We survey and systematize three critical, intersecting research areas: first, we provide a taxonomy of honeypot detection vectors, structuring the core problems that LLM-based realism must solve; second, we synthesize the emerging literature on LLM-honeypots, identifying a canonical architecture and key evaluation trends; and third, we chart the evolutionary path of honeypot log analysis, from simple data reduction to automated intelligence generation. We synthesize these findings into a forward-looking research roadmap, arguing that the true potential of this technology lies in creating autonomous, self-improving deception systems to counter the emerging threat of intelligent, automated attackers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04953v2">M-Prometheus: A Suite of Open Multilingual LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      The use of language models for automatically evaluating long-form text (LLM-as-a-judge) is becoming increasingly common, yet most LLM judges are optimized exclusively for English, with strategies for enhancing their multilingual evaluation capabilities remaining largely unexplored in the current literature. This has created a disparity in the quality of automatic evaluation methods for non-English languages, ultimately hindering the development of models with better multilingual capabilities. To bridge this gap, we introduce M-Prometheus, a suite of open-weight LLM judges ranging from 3B to 14B parameters that can provide both direct assessment and pairwise comparison feedback on multilingual outputs. M-Prometheus models outperform state-of-the-art open LLM judges on multilingual reward benchmarks spanning more than 20 languages, as well as on literary machine translation (MT) evaluation covering 4 language pairs. Furthermore, M-Prometheus models can be leveraged at decoding time to significantly improve generated outputs across all 3 tested languages, showcasing their utility for the development of better multilingual models. Lastly, through extensive ablations, we identify the key factors for obtaining an effective multilingual judge, including backbone model selection and training on synthetic multilingual feedback data instead of translated data. We release our models, training dataset, and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25904v1">Evaluating the Impact of LLM-Assisted Annotation in a Perspectivized Setting: the Case of FrameNet Annotation</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      The use of LLM-based applications as a means to accelerate and/or substitute human labor in the creation of language resources and dataset is a reality. Nonetheless, despite the potential of such tools for linguistic research, comprehensive evaluation of their performance and impact on the creation of annotated datasets, especially under a perspectivized approach to NLP, is still missing. This paper contributes to reduction of this gap by reporting on an extensive evaluation of the (semi-)automatization of FrameNet-like semantic annotation by the use of an LLM-based semantic role labeler. The methodology employed compares annotation time, coverage and diversity in three experimental settings: manual, automatic and semi-automatic annotation. Results show that the hybrid, semi-automatic annotation setting leads to increased frame diversity and similar annotation coverage, when compared to the human-only setting, while the automatic setting performs considerably worse in all metrics, except for annotation time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21204v2">Fuzzy, Symbolic, and Contextual: Enhancing LLM Instruction via Cognitive Scaffolding</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      We study how prompt-level inductive biases influence the cognitive behavior of large language models (LLMs) in instructional dialogue. We introduce a symbolic scaffolding method paired with a short-term memory schema designed to promote adaptive, structured reasoning in Socratic tutoring. Using controlled ablation across five system variants, we evaluate model outputs via expert-designed rubrics covering scaffolding, responsiveness, symbolic reasoning, and conversational memory. We present preliminary results using an LLM-based evaluation framework aligned to a cognitively grounded rubric. This enables scalable, systematic comparisons across architectural variants in early-stage experimentation. The preliminary results show that our full system consistently outperforms baseline variants. Analysis reveals that removing memory or symbolic structure degrades key cognitive behaviors, including abstraction, adaptive probing, and conceptual continuity. These findings support a processing-level account in which prompt-level cognitive scaffolds can reliably shape emergent instructional strategies in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25860v1">Through the Judge's Eyes: Inferred Thinking Traces Improve Reliability of LLM Raters</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as raters for evaluation tasks. However, their reliability is often limited for subjective tasks, when human judgments involve subtle reasoning beyond annotation labels. Thinking traces, the reasoning behind a judgment, are highly informative but challenging to collect and curate. We present a human-LLM collaborative framework to infer thinking traces from label-only annotations. The proposed framework uses a simple and effective rejection sampling method to reconstruct these traces at scale. These inferred thinking traces are applied to two complementary tasks: (1) fine-tuning open LLM raters; and (2) synthesizing clearer annotation guidelines for proprietary LLM raters. Across multiple datasets, our methods lead to significantly improved LLM-human agreement. Additionally, the refined annotation guidelines increase agreement among different LLM models. These results suggest that LLMs can serve as practical proxies for otherwise unrevealed human thinking traces, enabling label-only corpora to be extended into thinking-trace-augmented resources that enhance the reliability of LLM raters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25808v1">PRESTO: Preimage-Informed Instruction Optimization for Prompting Black-Box LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success across diverse domains, due to their strong instruction-following capabilities. This has led to increasing interest in optimizing instructions for black-box LLMs, whose internal parameters are inaccessible but widely used due to their strong performance. To optimize instructions for black-box LLMs, recent methods employ white-box LLMs to generate candidate instructions from optimized soft prompts. However, white-box LLMs often map different soft prompts to the same instruction, leading to redundant queries. While previous studies regarded this many-to-one mapping as a structure that hinders optimization efficiency, we reinterpret it as a useful prior knowledge that can accelerate the optimization. To this end, we introduce PREimage-informed inSTruction Optimization (PRESTO), a novel framework that leverages the preimage structure of soft prompts for efficient optimization. PRESTO consists of three key components: (1) score sharing, which shares the evaluation score with all soft prompts in a preimage; (2) preimage-based initialization, which selects initial data points that maximize search space coverage using preimage information; and (3) score consistency regularization, which enforces prediction consistency within each preimage. By leveraging preimages, PRESTO achieves the effect of effectively obtaining 14 times more scored data under the same query budget, resulting in more efficient optimization. Experimental results on 33 instruction optimization tasks demonstrate the superior performance of PRESTO. Code is available at https://github.com/mlvlab/PRESTO
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25805v1">Ideology-Based LLMs for Content Moderation</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in content moderation systems, where ensuring fairness and neutrality is essential. In this study, we examine how persona adoption influences the consistency and fairness of harmful content classification across different LLM architectures, model sizes, and content modalities (language vs. vision). At first glance, headline performance metrics suggest that personas have little impact on overall classification accuracy. However, a closer analysis reveals important behavioral shifts. Personas with different ideological leanings display distinct propensities to label content as harmful, showing that the lens through which a model "views" input can subtly shape its judgments. Further agreement analyses highlight that models, particularly larger ones, tend to align more closely with personas from the same political ideology, strengthening within-ideology consistency while widening divergence across ideological groups. To show this effect more directly, we conducted an additional study on a politically targeted task, which confirmed that personas not only behave more coherently within their own ideology but also exhibit a tendency to defend their perspective while downplaying harmfulness in opposing views. Together, these findings highlight how persona conditioning can introduce subtle ideological biases into LLM outputs, raising concerns about the use of AI systems that may reinforce partisan perspectives under the guise of neutrality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25799v1">LISTEN to Your Preferences: An LLM Framework for Multi-Objective Selection</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      Human experts often struggle to select the best option from a large set of items with multiple competing objectives, a process bottlenecked by the difficulty of formalizing complex, implicit preferences. To address this, we introduce LISTEN, a framework that leverages a Large Language Model (LLM) as a zero-shot preference oracle, guided only by an expert's high-level priorities in natural language. To operate within LLM constraints like context windows and inference costs, we propose two iterative algorithms: LISTEN-U, which uses the LLM to refine a parametric utility function, and LISTEN-T, a non-parametric method that performs tournament-style selections over small batches of solutions. Evaluated on diverse tasks including flight booking, shopping, and exam scheduling, our results show LISTEN-U excels when preferences are parametrically aligned (a property we measure with a novel concordance metric), while LISTEN-T offers more robust performance. This work explores a promising direction for steering complex multi-objective decisions directly with natural language, reducing the cognitive burden of traditional preference elicitation.
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
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.26835v1">Category-Aware Semantic Caching for Heterogeneous LLM Workloads</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 13 pages including reference, position paper
    </div>
    <details class="paper-abstract">
      LLM serving systems process heterogeneous query workloads where different categories exhibit different characteristics. Code queries cluster densely in embedding space while conversational queries distribute sparsely. Content staleness varies from minutes (stock data) to months (code patterns). Query repetition patterns range from power-law (code) to uniform (conversation), producing long tail cache hit rate distributions: high-repetition categories achieve 40-60% hit rates while low-repetition or volatile categories achieve 5-15% hit rates. Vector databases must exclude the long tail because remote search costs (30ms) require 15--20% hit rates to break even, leaving 20-30% of production traffic uncached. Uniform cache policies compound this problem: fixed thresholds cause false positives in dense spaces and miss valid paraphrases in sparse spaces; fixed TTLs waste memory or serve stale data. This paper presents category-aware semantic caching where similarity thresholds, TTLs, and quotas vary by query category. We present a hybrid architecture separating in-memory HNSW search from external document storage, reducing miss cost from 30ms to 2ms. This reduction makes low-hit-rate categories economically viable (break-even at 3-5% versus 15-20%), enabling cache coverage across the entire workload distribution. Adaptive load-based policies extend this framework to respond to downstream model load, dynamically adjusting thresholds and TTLs to reduce traffic to overloaded models by 9-17% in theoretical projections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11507v2">Oneiros: KV Cache Optimization through Parameter Remapping for Multi-tenant LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-10-29
    </div>
    <details class="paper-abstract">
      KV cache accelerates LLM inference by avoiding redundant computation, at the expense of memory. To support larger KV caches, prior work extends GPU memory with CPU memory via CPU-offloading. This involves swapping KV cache between GPU and CPU memory. However, because the cache updates dynamically, such swapping incurs high CPU memory traffic. We make a key observation that model parameters remain constant during runtime, unlike the dynamically updated KV cache. Building on this, we introduce Oneiros, which avoids KV cache swapping by remapping, and thereby repurposing, the memory allocated to model parameters for KV cache. This parameter remapping is especially beneficial in multi-tenant environments, where the memory used for the parameters of the inactive models can be more aggressively reclaimed. Exploiting the high CPU-GPU bandwidth offered by the modern hardware, such as the NVIDIA Grace Hopper Superchip, we show that Oneiros significantly outperforms state-of-the-art solutions, achieving a reduction of 44.8%-82.5% in tail time-between-token latency, 20.7%-99.3% in tail time-to-first-token latency, and 6.6%-86.7% higher throughput compared to vLLM. Source code of Oneiros is available at https://github.com/UT-SysML/Oneiros/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2411.14571v2">Learned, Lagged, LLM-splained: LLM Responses to End User Security Questions</a></div>
    <div class="paper-meta">
      📅 2025-10-29
      | 💬 17 pages, 7 tables
    </div>
    <details class="paper-abstract">
      Answering end user security questions is challenging. While large language models (LLMs) like GPT, LLAMA, and Gemini are far from error-free, they have shown promise in answering a variety of questions outside of security. We studied LLM performance in the area of end user security by qualitatively evaluating 3 popular LLMs on 900 systematically collected end user security questions. While LLMs demonstrate broad generalist ``knowledge'' of end user security information, there are patterns of errors and limitations across LLMs consisting of stale and inaccurate answers, and indirect or unresponsive communication styles, all of which impacts the quality of information received. Based on these patterns, we suggest directions for model improvement and recommend user strategies for interacting with LLMs when seeking assistance with security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24832v1">Scheduling Your LLM Reinforcement Learning with Reasoning Trees</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Using Reinforcement Learning with Verifiable Rewards (RLVR) to optimize Large Language Models (LLMs) can be conceptualized as progressively editing a query's `Reasoning Tree'. This process involves exploring nodes (tokens) and dynamically modifying the model's policy at each node. When combined with data scheduling, this process yields further gains in data efficiency and accuracy. However, existing RLVR data scheduling methods typically rely on path-based metrics to rank queries, overlooking the reasoning tree structures of these queries. In this paper, we introduce a novel metric, namely Reasoning Score (r-score), which measures the query's learning difficulty based on the structure of its reasoning tree. Based on the r-score, we propose the Reasoning Tree Schedule (Re-Schedule), a scheduling algorithm that constructs a curriculum progressing from structurally simple (high r-score) to complex (low r-score) queries. Experiments on six math-reasoning benchmarks show that Re-Schedule significantly improves average accuracy, achieving gains of up to 3.2%. These strong results validate our approach and demonstrate that a structural understanding of the reasoning tree provides a more powerful and principled foundation for RLVR data scheduling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24802v1">From Narrative to Action: A Hierarchical LLM-Agent Framework for Human Mobility Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 47 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Understanding and replicating human mobility requires not only spatial-temporal accuracy but also an awareness of the cognitive hierarchy underlying real-world travel decisions. Traditional agent-based or deep learning models can reproduce statistical patterns of movement but fail to capture the semantic coherence and causal logic of human behavior. Large language models (LLMs) show potential, but struggle to balance creative reasoning with strict structural compliance. This study proposes a Hierarchical LLM-Agent Framework, termed Narrative-to-Action, that integrates high-level narrative reasoning, mid-level reflective planning, and low-level behavioral execution within a unified cognitive hierarchy. At the macro level, one agent is employed as a "creative writer" to produce diary-style narratives rich in motivation and context, then uses another agent as a "structural parser" to convert narratives into machine-readable plans. A dynamic execution module further grounds agents in geographic environments and enables adaptive behavioral adjustments guided by a novel occupation-aware metric, Mobility Entropy by Occupation (MEO), which captures heterogeneous schedule flexibility across different occupational personalities. At the micro level, the agent executes concrete actions-selecting locations, transportation modes, and time intervals-through interaction with an environmental simulation. By embedding this multi-layer cognitive process, the framework produces not only synthetic trajectories that align closely with real-world patterns but also interpretable representations of human decision logic. This research advances synthetic mobility generation from a data-driven paradigm to a cognition-driven simulation, providing a scalable pathway for understanding, predicting, and synthesizing complex urban mobility behaviors through hierarchical LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24706v1">ComboBench: Can LLMs Manipulate Physical Devices to Play Virtual Reality Games?</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Virtual Reality (VR) games require players to translate high-level semantic actions into precise device manipulations using controllers and head-mounted displays (HMDs). While humans intuitively perform this translation based on common sense and embodied understanding, whether Large Language Models (LLMs) can effectively replicate this ability remains underexplored. This paper introduces a benchmark, ComboBench, evaluating LLMs' capability to translate semantic actions into VR device manipulation sequences across 262 scenarios from four popular VR games: Half-Life: Alyx, Into the Radius, Moss: Book II, and Vivecraft. We evaluate seven LLMs, including GPT-3.5, GPT-4, GPT-4o, Gemini-1.5-Pro, LLaMA-3-8B, Mixtral-8x7B, and GLM-4-Flash, compared against annotated ground truth and human performance. Our results reveal that while top-performing models like Gemini-1.5-Pro demonstrate strong task decomposition capabilities, they still struggle with procedural reasoning and spatial understanding compared to humans. Performance varies significantly across games, suggesting sensitivity to interaction complexity. Few-shot examples substantially improve performance, indicating potential for targeted enhancement of LLMs' VR manipulation capabilities. We release all materials at https://sites.google.com/view/combobench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24695v1">AgentFrontier: Expanding the Capability Frontier of LLM Agents with ZPD-Guided Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/
    </div>
    <details class="paper-abstract">
      Training large language model agents on tasks at the frontier of their capabilities is key to unlocking advanced reasoning. We introduce a data synthesis approach inspired by the educational theory of the Zone of Proximal Development (ZPD), which defines this frontier as tasks an LLM cannot solve alone but can master with guidance. To operationalize this, we present the AgentFrontier Engine, an automated pipeline that synthesizes high-quality, multidisciplinary data situated precisely within the LLM's ZPD. This engine supports both continued pre-training with knowledge-intensive data and targeted post-training on complex reasoning tasks. From the same framework, we derive the ZPD Exam, a dynamic and automated benchmark designed to evaluate agent capabilities on these frontier tasks. We train AgentFrontier-30B-A3B model on our synthesized data, which achieves state-of-the-art results on demanding benchmarks like Humanity's Last Exam, even surpassing some leading proprietary agents. Our work demonstrates that a ZPD-guided approach to data synthesis offers a scalable and effective path toward building more capable LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24677v1">Dissecting Role Cognition in Medical LLMs via Neuronal Ablation</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 15 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have gained significant traction in medical decision support systems, particularly in the context of medical question answering and role-playing simulations. A common practice, Prompt-Based Role Playing (PBRP), instructs models to adopt different clinical roles (e.g., medical students, residents, attending physicians) to simulate varied professional behaviors. However, the impact of such role prompts on model reasoning capabilities remains unclear. This study introduces the RP-Neuron-Activated Evaluation Framework(RPNA) to evaluate whether role prompts induce distinct, role-specific cognitive processes in LLMs or merely modify linguistic style. We test this framework on three medical QA datasets, employing neuron ablation and representation analysis techniques to assess changes in reasoning pathways. Our results demonstrate that role prompts do not significantly enhance the medical reasoning abilities of LLMs. Instead, they primarily affect surface-level linguistic features, with no evidence of distinct reasoning pathways or cognitive differentiation across clinical roles. Despite superficial stylistic changes, the core decision-making mechanisms of LLMs remain uniform across roles, indicating that current PBRP methods fail to replicate the cognitive complexity found in real-world medical practice. This highlights the limitations of role-playing in medical AI and emphasizes the need for models that simulate genuine cognitive processes rather than linguistic imitation.We have released the related code in the following repository:https: //github.com/IAAR-Shanghai/RolePlay_LLMDoctor
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24606v1">Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 Accepted to NeurIPS 2025 Workshop on Efficient Reasoning
    </div>
    <details class="paper-abstract">
      The quadratic cost of attention hinders the scalability of long-context LLMs, especially in resource-constrained settings. Existing static sparse methods such as sliding windows or global tokens utilizes the sparsity of attention to reduce the cost of attention, but poorly adapts to the content-dependent variations in attention due to their staticity. While previous work has proposed several dynamic approaches to improve flexibility, they still depend on predefined templates or heuristic mechanisms. Such strategies reduce generality and prune tokens that remain contextually important, limiting their accuracy across diverse tasks. To tackle these bottlenecks of existing methods for long-context modeling, we introduce Dynamic Hierarchical Sparse Attention (DHSA), a data-driven framework that dynamically predicts attention sparsity online without retraining. Our proposed DHSA adaptively segments sequences into variable-length chunks, then computes chunk representations by aggregating the token embeddings within each chunk. To avoid the bias introduced by varying chunk lengths, we apply length-normalized aggregation that scales the averaged embeddings by the square root of the chunk size. Finally, DHSA upsamples the chunk-level similarity scores to token level similarities to calculate importance scores that determine which token-level interactions should be preserved. Our experiments on Gemma2 with Needle-in-a-Haystack Test and LongBench show that DHSA matches dense attention in accuracy, while reducing prefill latency by 20-60% and peak memory usage by 35%. Compared to other representative baselines such as block sparse attention, DHSA achieves consistently higher accuracy (6-18% relative gains) with comparable or lower cost, offering an efficient and adaptable solution for long-context on-device LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24605v1">Diffusion LLM with Native Variable Generation Lengths: Let [EOS] Lead the Way</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (dLLMs) have exhibited substantial potential for parallel text generation, which may enable more efficient generation compared to autoregressive models. However, current dLLMs suffer from fixed generation lengths, which indicates the generation lengths of dLLMs have to be determined before decoding as a hyper-parameter, leading to issues in efficiency and flexibility. To solve these problems, in this work, we propose to train a diffusion LLM with native variable generation lengths, abbreviated as dLLM-Var. Concretely, we aim to train a model to accurately predict the [EOS] token in the generated text, which makes a dLLM be able to natively infer in a block diffusion manner, while still maintaining the ability of global bi-directional (full) attention and high parallelism. Experiments on standard benchmarks demonstrate that our method achieves a 30.1x speedup over traditional dLLM inference paradigms and a 2.4x speedup relative to autoregressive models such as Qwen and Llama. Our method achieves higher accuracy and faster inference, elevating dLLMs beyond mere academic novelty and supporting their practical use in real-world applications. Codes and models have been released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24582v1">Politically Speaking: LLMs on Changing International Affairs</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Ask your chatbot to impersonate an expert from Russia and an expert from US and query it on Chinese politics. How might the outputs differ? Or, to prepare ourselves for the worse, how might they converge? Scholars have raised concerns LLM based applications can homogenize cultures and flatten perspectives. But exactly how much does LLM generated outputs converge despite explicit different role assignment? This study provides empirical evidence to the above question. The critique centres on pretrained models regurgitating ossified political jargons used in the Western world when speaking about China, Iran, Russian, and US politics, despite changes in these countries happening daily or hourly. The experiments combine role-prompting and similarity metrics. The results show that AI generated discourses from four models about Iran and China are the most homogeneous and unchanging across all four models, including OpenAI GPT, Google Gemini, Anthropic Claude, and DeepSeek, despite the prompted perspective change and the actual changes in real life. This study does not engage with history, politics, or literature as traditional disciplinary approaches would; instead, it takes cues from international and area studies and offers insight on the future trajectory of shifting political discourse in a digital space increasingly cannibalised by AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24505v1">CritiCal: Can Critique Help LLM Uncertainty or Confidence Calibration?</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Accurate confidence calibration in Large Language Models (LLMs) is critical for safe use in high-stakes domains, where clear verbalized confidence enhances user trust. Traditional methods that mimic reference confidence expressions often fail to capture the reasoning needed for accurate confidence assessment. We propose natural language critiques as a solution, ideally suited for confidence calibration, as precise gold confidence labels are hard to obtain and often require multiple generations. This paper studies how natural language critiques can enhance verbalized confidence, addressing: (1) What to critique: uncertainty (question-focused) or confidence (answer-specific)? Analysis shows confidence suits multiple-choice tasks, while uncertainty excels in open-ended scenarios. (2) How to critique: self-critique or critique calibration training? We propose Self-Critique, enabling LLMs to critique and optimize their confidence beyond mere accuracy, and CritiCal, a novel Critique Calibration training method that leverages natural language critiques to improve confidence calibration, moving beyond direct numerical optimization. Experiments show that CritiCal significantly outperforms Self-Critique and other competitive baselines, even surpassing its teacher model, GPT-4o, in complex reasoning tasks. CritiCal also shows robust generalization in out-of-distribution settings, advancing LLM's reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10978v3">Group-in-Group Policy Optimization for LLM Agent Training</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advances in group-based reinforcement learning (RL) have driven frontier large language models (LLMs) in single-turn tasks like mathematical reasoning. However, their scalability to multi-turn LLM agent training remains limited. Unlike static tasks, agent-environment interactions unfold over many steps and often yield sparse or delayed rewards, making credit assignment across individual steps significantly more challenging. In this work, we propose Group-in-Group Policy Optimization (GiGPO), a novel RL algorithm that achieves fine-grained credit assignment for LLM agents while preserving the appealing properties of group-based RL: critic-free, low memory, and stable convergence. GiGPO introduces a two-level structure for estimating relative advantage: (i) At the episode-level, GiGPO computes macro relative advantages based on groups of complete trajectories; (ii) At the step-level, GiGPO introduces an anchor state grouping mechanism that retroactively constructs step-level groups by identifying repeated environment states across trajectories. Actions stemming from the same state are grouped together, enabling micro relative advantage estimation. This hierarchical structure effectively captures both global trajectory quality and local step effectiveness without relying on auxiliary models or additional rollouts. We evaluate GiGPO on challenging agent benchmarks, including ALFWorld and WebShop, as well as tool-integrated reasoning on search-augmented QA tasks, using Qwen2.5-1.5B/3B/7B-Instruct. Crucially, GiGPO delivers fine-grained per-step credit signals, achieves performance gains of > 12% on ALFWorld and > 9% on WebShop over GRPO, and obtains superior performance on QA tasks (42.1% on 3B and 47.2% on 7B): all while maintaining the same GPU memory overhead, identical LLM rollout, and incurring little to no additional time cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24488v1">A word association network methodology for evaluating implicit biases in LLMs compared to humans</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 24 pages, 13 figures, 3 tables
    </div>
    <details class="paper-abstract">
      As Large language models (LLMs) become increasingly integrated into our lives, their inherent social biases remain a pressing concern. Detecting and evaluating these biases can be challenging because they are often implicit rather than explicit in nature, so developing evaluation methods that assess the implicit knowledge representations of LLMs is essential. We present a novel word association network methodology for evaluating implicit biases in LLMs based on simulating semantic priming within LLM-generated word association networks. Our prompt-based approach taps into the implicit relational structures encoded in LLMs, providing both quantitative and qualitative assessments of bias. Unlike most prompt-based evaluation methods, our method enables direct comparisons between various LLMs and humans, providing a valuable point of reference and offering new insights into the alignment of LLMs with human cognition. To demonstrate the utility of our methodology, we apply it to both humans and several widely used LLMs to investigate social biases related to gender, religion, ethnicity, sexual orientation, and political party. Our results reveal both convergences and divergences between LLM and human biases, providing new perspectives on the potential risks of using LLMs. Our methodology contributes to a systematic, scalable, and generalizable framework for evaluating and comparing biases across multiple LLMs and humans, advancing the goal of transparent and socially responsible language technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24476v1">Mitigating Hallucination in Large Language Models (LLMs): An Application-Oriented Survey on RAG, Reasoning, and Agentic Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 25 pages, 7 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Hallucination remains one of the key obstacles to the reliable deployment of large language models (LLMs), particularly in real-world applications. Among various mitigation strategies, Retrieval-Augmented Generation (RAG) and reasoning enhancement have emerged as two of the most effective and widely adopted approaches, marking a shift from merely suppressing hallucinations to balancing creativity and reliability. However, their synergistic potential and underlying mechanisms for hallucination mitigation have not yet been systematically examined. This survey adopts an application-oriented perspective of capability enhancement to analyze how RAG, reasoning enhancement, and their integration in Agentic Systems mitigate hallucinations. We propose a taxonomy distinguishing knowledge-based and logic-based hallucinations, systematically examine how RAG and reasoning address each, and present a unified framework supported by real-world applications, evaluations, and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24469v1">Iterative Critique-Refine Framework for Enhancing LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Personalized text generation requires models not only to produce coherent text but also to align with a target user's style, tone, and topical focus. Existing retrieval-augmented approaches such as LaMP and PGraphRAG enrich profiles with user and neighbor histories, but they stop at generation and often yield outputs that drift in tone, topic, or style. We present PerFine, a unified, training-free critique-refine framework that enhances personalization through iterative, profile-grounded feedback. In each iteration, an LLM generator produces a draft conditioned on the retrieved profile, and a critic LLM - also conditioned on the same profile - provides structured feedback on tone, vocabulary, sentence structure, and topicality. The generator then revises, while a novel knockout strategy retains the stronger draft across iterations. We further study additional inference-time strategies such as Best-of-N and Topic Extraction to balance quality and efficiency. Across Yelp, Goodreads, and Amazon datasets, PerFine consistently improves personalization over PGraphRAG, with GEval gains of +7-13%, steady improvements over 3-5 refinement iterations, and scalability with increasing critic size. These results highlight that post-hoc, profile-aware feedback offers a powerful paradigm for personalized LLM generation that is both training-free and model-agnostic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24450v1">Charting the European LLM Benchmarking Landscape: A New Taxonomy and a Set of Best Practices</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 12 pages, 1 figure. Submitted to the LREC 2026 conference
    </div>
    <details class="paper-abstract">
      While new benchmarks for large language models (LLMs) are being developed continuously to catch up with the growing capabilities of new models and AI in general, using and evaluating LLMs in non-English languages remains a little-charted landscape. We give a concise overview of recent developments in LLM benchmarking, and then propose a new taxonomy for the categorization of benchmarks that is tailored to multilingual or non-English use scenarios. We further propose a set of best practices and quality standards that could lead to a more coordinated development of benchmarks for European languages. Among other recommendations, we advocate for a higher language and culture sensitivity of evaluation methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24442v1">Law in Silico: Simulating Legal Society with LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Since real-world legal experiments are often costly or infeasible, simulating legal societies with Artificial Intelligence (AI) systems provides an effective alternative for verifying and developing legal theory, as well as supporting legal administration. Large Language Models (LLMs), with their world knowledge and role-playing capabilities, are strong candidates to serve as the foundation for legal society simulation. However, the application of LLMs to simulate legal systems remains underexplored. In this work, we introduce Law in Silico, an LLM-based agent framework for simulating legal scenarios with individual decision-making and institutional mechanisms of legislation, adjudication, and enforcement. Our experiments, which compare simulated crime rates with real-world data, demonstrate that LLM-based agents can largely reproduce macro-level crime trends and provide insights that align with real-world observations. At the same time, micro-level simulations reveal that a well-functioning, transparent, and adaptive legal system offers better protection of the rights of vulnerable individuals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24430v1">From Time and Place to Preference: LLM-Driven Geo-Temporal Context in Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-10-28
    </div>
    <details class="paper-abstract">
      Most recommender systems treat timestamps as numeric or cyclical values, overlooking real-world context such as holidays, events, and seasonal patterns. We propose a scalable framework that uses large language models (LLMs) to generate geo-temporal embeddings from only a timestamp and coarse location, capturing holidays, seasonal trends, and local/global events. We then introduce a geo-temporal embedding informativeness test as a lightweight diagnostic, demonstrating on MovieLens, LastFM, and a production dataset that these embeddings provide predictive signal consistent with the outcomes of full model integrations. Geo-temporal embeddings are incorporated into sequential models through (1) direct feature fusion with metadata embeddings or (2) an auxiliary loss that enforces semantic and geo-temporal alignment. Our findings highlight the need for adaptive or hybrid recommendation strategies, and we release a context-enriched MovieLens dataset to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.24408v1">Uncovering Gaps Between RFC Updates and TCP/IP Implementations: LLM-Facilitated Differential Checks on Intermediate Representations</a></div>
    <div class="paper-meta">
      📅 2025-10-28
      | 💬 15 pages, 7 figures
    </div>
    <details class="paper-abstract">
      As the core of the Internet infrastructure, the TCP/IP protocol stack undertakes the task of network data transmission. However, due to the complexity of the protocol and the uncertainty of cross-layer interaction, there are often inconsistencies between the implementation of the protocol stack code and the RFC standard. This inconsistency may not only lead to differences in protocol functions but also cause serious security vulnerabilities. At present, with the continuous expansion of protocol stack functions and the rapid iteration of RFC documents, it is increasingly important to detect and fix these inconsistencies. With the rise of large language models, researchers have begun to explore how to extract protocol specifications from RFC documents through these models, including protocol stack modeling, state machine extraction, text ambiguity analysis, and other related content. However, existing methods rely on predefined patterns or rule-based approaches that fail to generalize across different protocol specifications. Automated and scalable detection of these inconsistencies remains a significant challenge. In this study, we propose an automated analysis framework based on LLM and differential models. By modeling the iterative relationship of the protocol and based on the iterative update relationship of the RFC standard, we perform incremental code function analysis on different versions of kernel code implementations to automatically perform code detection and vulnerability analysis. We conduct extensive evaluations to validate the effectiveness of our framework, demonstrating its effectiveness in identifying potential vulnerabilities caused by RFC code inconsistencies.
    </details>
</div>
