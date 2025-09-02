# llm - 2025_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16712v1">Systematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 14 pages, 10 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across diverse domains, but their heavy resource demands make quantization-reducing precision to lower-bit formats-critical for efficient serving. While many quantization methods exist, a systematic understanding of their performance, energy, and quality tradeoffs in realistic serving conditions remains a gap. In this work, we first develop a fully automated online characterization framework qMeter, and then conduct an in-depth characterization of 11 post-training LLM quantization methods across 4 model sizes (7B-70B) and two GPU architectures (A100, H100). We evaluate quantization at the application, workload, parallelism, and hardware levels under online serving conditions. Our study reveals highly task- and method-dependent tradeoffs, strong sensitivity to workload characteristics, and complex interactions with parallelism and GPU architecture. We further present three optimization case studies illustrating deployment challenges in capacity planning, energy-efficient scheduling, and multi-objective tuning. To the best of our knowledge, this is one of the first comprehensive application-, system-, and hardware-level characterization of LLM quantization from a joint performance, energy, and quality perspective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16431v1">Cetvel: A Unified Benchmark for Evaluating Language Understanding, Generation and Cultural Capacity of LLMs for Turkish</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 31 pages, 2 figures, 10 tables
    </div>
    <details class="paper-abstract">
      We introduce Cetvel, a comprehensive benchmark designed to evaluate large language models (LLMs) in Turkish. Existing Turkish benchmarks often lack either task diversity or culturally relevant content, or both. Cetvel addresses these gaps by combining a broad range of both discriminative and generative tasks ensuring content that reflects the linguistic and cultural richness of Turkish language. Cetvel covers 23 tasks grouped into seven categories, including tasks such as grammatical error correction, machine translation, and question answering rooted in Turkish history and idiomatic language. We evaluate 33 open-weight LLMs (up to 70B parameters) covering different model families and instruction paradigms. Our experiments reveal that Turkish-centric instruction-tuned models generally underperform relative to multilingual or general-purpose models (e.g. Llama 3 and Mistral), despite being tailored for the language. Moreover, we show that tasks such as grammatical error correction and extractive question answering are particularly discriminative in differentiating model capabilities. Cetvel offers a comprehensive and culturally grounded evaluation suite for advancing the development and assessment of LLMs in Turkish.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16402v1">AetherCode: Evaluating LLMs' Ability to Win In Premier Programming Competitions</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Competitive programming has emerged as a critical benchmark for evaluating the reasoning and coding capabilities of Large Language Models (LLMs). Despite impressive progress on existing benchmarks, we argue that current evaluations overstate model proficiency, masking a substantial gap between LLMs and elite human programmers. This gap arises from two key limitations: insufficient difficulty and scope of benchmark problems, and evaluation bias from low-quality test cases. To address these shortcomings, we present AetherCode, a new benchmark that draws problems from premier programming competitions such as IOI and ICPC, offering broader coverage and higher difficulty. AetherCode further incorporates comprehensive, expert-validated test suites built through a hybrid of automated generation and human curation, ensuring rigorous and reliable assessment. By combining challenging problem design with robust evaluation, AetherCode provides a more faithful measure of LLM capabilities and sets a new standard for future research in code reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05220v3">Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Accepted by the EMNLP25 main conference
    </div>
    <details class="paper-abstract">
      Retrieval models typically rely on costly human-labeled query-document relevance annotations for training and evaluation. To reduce this cost and leverage the potential of Large Language Models (LLMs) in relevance judgments, we aim to explore whether LLM-generated annotations can effectively replace human annotations in training retrieval models. Retrieval usually emphasizes relevance, which indicates "topic-relatedness" of a document to a query, while in RAG, the value of a document (or utility) depends on how it contributes to answer generation. Recognizing this mismatch, some researchers use LLM performance on downstream tasks with documents as labels, but this approach requires manual answers for specific tasks, leading to high costs and limited generalization. In another line of work, prompting LLMs to select useful documents as RAG references eliminates the need for human annotation and is not task-specific. If we leverage LLMs' utility judgments to annotate retrieval data, we may retain cross-task generalization without human annotation in large-scale corpora. Therefore, we investigate utility-focused annotation via LLMs for large-scale retriever training data across both in-domain and out-of-domain settings on the retrieval and RAG tasks. To reduce the impact of low-quality positives labeled by LLMs, we design a novel loss function, i.e., Disj-InfoNCE. Our experiments reveal that: (1) Retrievers trained on utility-focused annotations significantly outperform those trained on human annotations in the out-of-domain setting on both tasks, demonstrating superior generalization capabilities. (2) LLM annotation does not replace human annotation in the in-domain setting. However, incorporating just 20% human-annotated data enables retrievers trained with utility-focused annotations to match the performance of models trained entirely with human annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16706v1">RoboBuddy in the Classroom: Exploring LLM-Powered Social Robots for Storytelling in Learning and Integration Activities</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Accepted to be published in the proceedings of 34th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN) in 2025
    </div>
    <details class="paper-abstract">
      Creating and improvising scenarios for content approaching is an enriching technique in education. However, it comes with a significant increase in the time spent on its planning, which intensifies when using complex technologies, such as social robots. Furthermore, addressing multicultural integration is commonly embedded in regular activities due to the already tight curriculum. Addressing these issues with a single solution, we implemented an intuitive interface that allows teachers to create scenario-based activities from their regular curriculum using LLMs and social robots. We co-designed different frameworks of activities with 4 teachers and deployed it in a study with 27 students for 1 week. Beyond validating the system's efficacy, our findings highlight the positive impact of integration policies perceived by the children and demonstrate the importance of scenario-based activities in students' enjoyment, observed to be significantly higher when applying storytelling. Additionally, several implications of using LLMs and social robots in long-term classroom activities are discussed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16347v1">Confusion is the Final Barrier: Rethinking Jailbreak Evaluation and Investigating the Real Misuse Threat of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      With the development of Large Language Models (LLMs), numerous efforts have revealed their vulnerabilities to jailbreak attacks. Although these studies have driven the progress in LLMs' safety alignment, it remains unclear whether LLMs have internalized authentic knowledge to deal with real-world crimes, or are merely forced to simulate toxic language patterns. This ambiguity raises concerns that jailbreak success is often attributable to a hallucination loop between jailbroken LLM and judger LLM. By decoupling the use of jailbreak techniques, we construct knowledge-intensive Q\&A to investigate the misuse threats of LLMs in terms of dangerous knowledge possession, harmful task planning utility, and harmfulness judgment robustness. Experiments reveal a mismatch between jailbreak success rates and harmful knowledge possession in LLMs, and existing LLM-as-a-judge frameworks tend to anchor harmfulness judgments on toxic language patterns. Our study reveals a gap between existing LLM safety assessments and real-world threat potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14540v2">VERUS-LM: a Versatile Framework for Combining LLMs with Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Accepted at ICLP 2025, part of ECPTS
    </div>
    <details class="paper-abstract">
      A recent approach to neurosymbolic reasoning is to explicitly combine the strengths of large language models (LLMs) and symbolic solvers to tackle complex reasoning tasks. However, current approaches face significant limitations, including poor generalizability due to task-specific prompts, inefficiencies caused by the lack of separation between knowledge and queries, and restricted inferential capabilities. These shortcomings hinder their scalability and applicability across diverse domains. In this paper, we introduce VERUS-LM, a novel framework designed to address these challenges. VERUS-LM employs a generic prompting mechanism, clearly separates domain knowledge from queries, and supports a wide range of different logical reasoning tasks. This framework enhances adaptability, reduces computational cost, and allows for richer forms of reasoning, such as optimization and constraint satisfaction. We show that our approach succeeds in diverse reasoning on a novel dataset, markedly outperforming LLMs. Additionally, our system achieves competitive results on common reasoning benchmarks when compared to other state-of-the-art approaches, and significantly surpasses them on the difficult AR-LSAT dataset. By pushing the boundaries of hybrid reasoning, VERUS-LM represents a significant step towards more versatile neurosymbolic AI systems
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13824v2">Can Hallucinations Help? Boosting LLMs for Drug Discovery</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Hallucinations in large language models (LLMs), plausible but factually inaccurate text, are often viewed as undesirable. However, recent work suggests that such outputs may hold creative potential. In this paper, we investigate whether hallucinations can improve LLMs on molecule property prediction, a key task in early-stage drug discovery. We prompt LLMs to generate natural language descriptions from molecular SMILES strings and incorporate these often hallucinated descriptions into downstream classification tasks. Evaluating seven instruction-tuned LLMs across five datasets, we find that hallucinations significantly improve predictive accuracy for some models. Notably, Falcon3-Mamba-7B outperforms all baselines when hallucinated text is included, while hallucinations generated by GPT-4o consistently yield the greatest gains between models. We further identify and categorize over 18,000 beneficial hallucinations, with structural misdescriptions emerging as the most impactful type, suggesting that hallucinated statements about molecular structure may increase model confidence. Ablation studies show that larger models benefit more from hallucinations, while temperature has a limited effect. Our findings challenge conventional views of hallucination as purely problematic and suggest new directions for leveraging hallucinations as a useful signal in scientific modeling tasks like drug discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14913v2">Bridging the Culture Gap: A Framework for LLM-Driven Socio-Cultural Localization of Math Word Problems in Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant capabilities in solving mathematical problems expressed in natural language. However, multilingual and culturally-grounded mathematical reasoning in low-resource languages lags behind English due to the scarcity of socio-cultural task datasets that reflect accurate native entities such as person names, organization names, and currencies. Existing multilingual benchmarks are predominantly produced via translation and typically retain English-centric entities, owing to the high cost associated with human annotater-based localization. Moreover, automated localization tools are limited, and hence, truly localized datasets remain scarce. To bridge this gap, we introduce a framework for LLM-driven cultural localization of math word problems that automatically constructs datasets with native names, organizations, and currencies from existing sources. We find that translated benchmarks can obscure true multilingual math ability under appropriate socio-cultural contexts. Through extensive experiments, we also show that our framework can help mitigate English-centric entity bias and improves robustness when native entities are introduced across various languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00719v2">Dynamically Adaptive Reasoning via LLM-Guided MCTS for Efficient and Context-Aware KGQA</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Knowledge Graph Question Answering (KGQA) aims to interpret natural language queries and perform structured reasoning over knowledge graphs by leveraging their relational and semantic structures to retrieve accurate answers. Recent KGQA methods primarily follow either retrieve-then-reason paradigm, relying on GNNs or heuristic rules for static paths extraction, or dynamic path generation strategies that use large language models (LLMs) with prompting to jointly perform retrieval and reasoning. However, the former suffers from limited adaptability due to static path extraction and lack of contextual refinement, while the latter incurs high computational costs and struggles with accurate path evaluation due to reliance on fixed scoring functions and extensive LLM calls. To address these issues, this paper proposes Dynamically Adaptive MCTS-based Reasoning (DAMR), a novel framework that integrates symbolic search with adaptive path evaluation for efficient and context-aware KGQA. DAMR employs a Monte Carlo Tree Search (MCTS) backbone guided by an LLM-based planner, which selects top-$k$ relevant relations at each step to reduce search space. To improve path evaluation accuracy, we introduce a lightweight Transformer-based scorer that performs context-aware plausibility estimation by jointly encoding the question and relation sequence through cross-attention, enabling the model to capture fine-grained semantic shifts during multi-hop reasoning. Furthermore, to alleviate the scarcity of high-quality supervision, DAMR incorporates a dynamic pseudo-path refinement mechanism that periodically generates training signals from partial paths explored during search, allowing the scorer to continuously adapt to the evolving distribution of reasoning trajectories. Extensive experiments on multiple KGQA benchmarks show that DAMR significantly outperforms state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16270v1">LLMs that Understand Processes: Instruction-tuning for Semantics-Aware Process Mining</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Accepted at IEEE ICPM 2025, 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Process mining is increasingly using textual information associated with events to tackle tasks such as anomaly detection and process discovery. Such semantics-aware process mining focuses on what behavior should be possible in a process (i.e., expectations), thus providing an important complement to traditional, frequency-based techniques that focus on recorded behavior (i.e., reality). Large Language Models (LLMs) provide a powerful means for tackling semantics-aware tasks. However, the best performance is so far achieved through task-specific fine-tuning, which is computationally intensive and results in models that can only handle one specific task. To overcome this lack of generalization, we use this paper to investigate the potential of instruction-tuning for semantics-aware process mining. The idea of instruction-tuning here is to expose an LLM to prompt-answer pairs for different tasks, e.g., anomaly detection and next-activity prediction, making it more familiar with process mining, thus allowing it to also perform better at unseen tasks, such as process discovery. Our findings demonstrate a varied impact of instruction-tuning: while performance considerably improved on process discovery and prediction tasks, it varies across models on anomaly detection tasks, highlighting that the selection of tasks for instruction-tuning is critical to achieving desired outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16267v1">From Confidence to Collapse in LLM Factual Robustness</a></div>
    <div class="paper-meta">
      📅 2025-08-22
    </div>
    <details class="paper-abstract">
      Ensuring the robustness of factual knowledge in LLMs is critical for reliable applications in tasks such as question answering and reasoning. However, existing evaluation methods predominantly focus on performance-based metrics, often investigating from the perspective of prompt perturbations, which captures only the externally triggered side of knowledge robustness. To bridge this gap, we introduce a principled approach to measure factual robustness from the perspective of the generation process by analyzing token distribution entropy in combination with temperature scaling sensitivity. These two factors build the Factual Robustness Score (FRS), a novel metric which quantifies the stability of a fact against perturbations in decoding conditions, given its initial uncertainty. To validate our approach, we conduct extensive experiments on 5 LLMs across 3 closed-book QA datasets (SQuAD, TriviaQA, and HotpotQA). We show that factual robustness varies significantly -- smaller models report an FRS of $0.76$, larger ones $0.93$ -- with accuracy degrading by ~$60\%$ under increased uncertainty. These insights demonstrate how entropy and temperature scaling impact factual accuracy, and lay a foundation for developing more robust knowledge retention and retrieval in future models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20807v2">GPU Kernel Scientist: An LLM-Driven Framework for Iterative Kernel Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 4+1 page paper plus Appendices and Supplementary zip file. Presented at the ES-FoMo "Efficient Systems for Foundation Models" workshop at ICML 2025
    </div>
    <details class="paper-abstract">
      Optimizing GPU kernels for high performance is a complex task, often demanding deep architectural knowledge, extensive profiling, and iterative experimentation. This challenge is amplified when targeting newer or less-documented GPU architectures where traditional development aids are scarce. This paper introduces an LLM-powered "GPU Kernel Scientist," an automated methodology for iteratively refining accelerator kernels. Our methodology employs LLMs in a multi-stage, evolutionary process: (a) strategically selecting promising prior code versions as a basis for new iterations; (b) generating hypotheses for optimization experiments, based on existing code and assimilated knowledge from general GPU literature; and (c) autonomously implementing these experiments through code modification and subsequent submission to an external evaluation system, using only observed timing data as performance feedback. We detail how this approach navigates the challenges of the AMD MI300 target architecture and leverages LLMs to compensate for limited domain-specific human expertise. In addition to our results, we present the architectural design, operational workflow, and qualitative insights, highlighting the potential of LLM-driven agents to democratise and accelerate GPU kernel optimization, especially in resource-constrained or rapidly updating hardware environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16213v1">MedOmni-45°: A Safety-Performance Benchmark for Reasoning-Oriented LLMs in Medicine</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      With the increasing use of large language models (LLMs) in medical decision-support, it is essential to evaluate not only their final answers but also the reliability of their reasoning. Two key risks are Chain-of-Thought (CoT) faithfulness -- whether reasoning aligns with responses and medical facts -- and sycophancy, where models follow misleading cues over correctness. Existing benchmarks often collapse such vulnerabilities into single accuracy scores. To address this, we introduce MedOmni-45 Degrees, a benchmark and workflow designed to quantify safety-performance trade-offs under manipulative hint conditions. It contains 1,804 reasoning-focused medical questions across six specialties and three task types, including 500 from MedMCQA. Each question is paired with seven manipulative hint types and a no-hint baseline, producing about 27K inputs. We evaluate seven LLMs spanning open- vs. closed-source, general-purpose vs. medical, and base vs. reasoning-enhanced models, totaling over 189K inferences. Three metrics -- Accuracy, CoT-Faithfulness, and Anti-Sycophancy -- are combined into a composite score visualized with a 45 Degrees plot. Results show a consistent safety-performance trade-off, with no model surpassing the diagonal. The open-source QwQ-32B performs closest (43.81 Degrees), balancing safety and accuracy but not leading in both. MedOmni-45 Degrees thus provides a focused benchmark for exposing reasoning vulnerabilities in medical LLMs and guiding safer model development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16201v1">SpecVLM: Enhancing Speculative Decoding of Video LLMs via Verifier-Guided Token Pruning</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Video large language models (Vid-LLMs) have shown strong capabilities in understanding video content. However, their reliance on dense video token representations introduces substantial memory and computational overhead in both prefilling and decoding. To mitigate the information loss of recent video token reduction methods and accelerate the decoding stage of Vid-LLMs losslessly, we introduce SpecVLM, a training-free speculative decoding (SD) framework tailored for Vid-LLMs that incorporates staged video token pruning. Building on our novel finding that the draft model's speculation exhibits low sensitivity to video token pruning, SpecVLM prunes up to 90% of video tokens, enabling efficient speculation without sacrificing accuracy. To achieve this, it performs a two-stage pruning process: Stage I selects highly informative tokens guided by attention signals from the verifier (target model), while Stage II prunes remaining redundant ones in a spatially uniform manner. Extensive experiments on four video understanding benchmarks demonstrate the effectiveness and robustness of SpecVLM, which achieves up to 2.68$\times$ decoding speedup for LLaVA-OneVision-72B and 2.11$\times$ speedup for Qwen2.5-VL-32B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15305v1">Coarse-to-Fine Grounded Memory for LLM Agent Planning</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Accepted to EMNLP 2025 Main Conference;27 pages,15 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have driven growing interest in LLM-based agents for complex planning tasks. To avoid costly agent training, many studies adopted memory mechanism that enhances LLM with offline experiences or online trajectory analysis. However, existing works focus on single-granularity memory derived from dynamic environmental interactions, which are inherently constrained by the quality of the collected experiences. This limitation, in turn, constrain the diversity of knowledge and the flexibility of planning. We propose Coarse-to-Fine Grounded Memory (\Ours{}), a novel framework that grounds coarse-to-fine memories with LLM, thereby fully leverage them for flexible adaptation to diverse scenarios. \Ours{} grounds environmental information into coarse-grained focus points to guide experience collection in training tasks, followed by grounding of actionable hybrid-grained tips from each experience. At inference, \Ours{} retrieves task-relevant experiences and tips to support planning. When facing environmental anomalies, the LLM grounds the current situation into fine-grained key information, enabling flexible self-QA reflection and plan correction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15243v1">Comp-X: On Defining an Interactive Learned Image Compression Paradigm With Expert-driven LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      We present Comp-X, the first intelligently interactive image compression paradigm empowered by the impressive reasoning capability of large language model (LLM) agent. Notably, commonly used image codecs usually suffer from limited coding modes and rely on manual mode selection by engineers, making them unfriendly for unprofessional users. To overcome this, we advance the evolution of image coding paradigm by introducing three key innovations: (i) multi-functional coding framework, which unifies different coding modes of various objective/requirements, including human-machine perception, variable coding, and spatial bit allocation, into one framework. (ii) interactive coding agent, where we propose an augmented in-context learning method with coding expert feedback to teach the LLM agent how to understand the coding request, mode selection, and the use of the coding tools. (iii) IIC-bench, the first dedicated benchmark comprising diverse user requests and the corresponding annotations from coding experts, which is systematically designed for intelligently interactive image compression evaluation. Extensive experimental results demonstrate that our proposed Comp-X can understand the coding requests efficiently and achieve impressive textual interaction capability. Meanwhile, it can maintain comparable compression performance even with a single coding framework, providing a promising avenue for artificial general intelligence (AGI) in image compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03730v3">Teuken-7B-Base & Teuken-7B-Instruct: Towards European LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      We present two multilingual LLMs, Teuken 7B-base and Teuken 7B-instruct, designed to embrace Europe's linguistic diversity by supporting all 24 official languages of the European Union. Trained on a dataset comprising around 60% non-English data and utilizing a custom multilingual tokenizer, our models address the limitations of existing LLMs that predominantly focus on English or a few high-resource languages. We detail the models' development principles, i.e., data composition, tokenizer optimization, and training methodologies. The models demonstrate strong performance across multilingual benchmarks, as evidenced by their performance on European versions of ARC, HellaSwag, and TruthfulQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15204v1">R-ConstraintBench: Evaluating LLMs on NP-Complete Scheduling</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Effective scheduling under tight resource, timing, and operational constraints underpins large-scale planning across sectors such as capital projects, manufacturing, logistics, and IT fleet transitions. However, the reliability of large language models (LLMs) when reasoning under high-constraint regimes is insufficiently characterized. To address this gap, we present R-ConstraintBench, a scalable framework that evaluates models on Resource-Constrained Project Scheduling Problems (RCPSP), an NP-Complete feasibility class, while difficulty increases via linear growth in constraints. R-ConstraintBench incrementally increases non-redundant precedence constraints in Directed Acyclic Graphs (DAGs) and then introduces downtime, temporal windows, and disjunctive constraints. As an illustrative example, we instantiate the benchmark in a data center migration setting and evaluate multiple LLMs using feasibility and error analysis, identifying degradation thresholds and constraint types most associated with failure. Empirically, strong models are near-ceiling on precedence-only DAGs, but feasibility performance collapses when downtime, temporal windows, and disjunctive constraints interact, implicating constraint interaction, not graph depth, as the principal bottleneck. Performance on clean synthetic ramps also does not guarantee transfer to domain-grounded scenarios, underscoring limited generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06992v2">MCA-RG: Enhancing LLMs with Medical Concept Alignment for Radiology Report Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 MICCAI 2025
    </div>
    <details class="paper-abstract">
      Despite significant advancements in adapting Large Language Models (LLMs) for radiology report generation (RRG), clinical adoption remains challenging due to difficulties in accurately mapping pathological and anatomical features to their corresponding text descriptions. Additionally, semantic agnostic feature extraction further hampers the generation of accurate diagnostic reports. To address these challenges, we introduce Medical Concept Aligned Radiology Report Generation (MCA-RG), a knowledge-driven framework that explicitly aligns visual features with distinct medical concepts to enhance the report generation process. MCA-RG utilizes two curated concept banks: a pathology bank containing lesion-related knowledge, and an anatomy bank with anatomical descriptions. The visual features are aligned with these medical concepts and undergo tailored enhancement. We further propose an anatomy-based contrastive learning procedure to improve the generalization of anatomical features, coupled with a matching loss for pathological features to prioritize clinically relevant regions. Additionally, a feature gating mechanism is employed to filter out low-quality concept features. Finally, the visual features are corresponding to individual medical concepts, and are leveraged to guide the report generation process. Experiments on two public benchmarks (MIMIC-CXR and CheXpert Plus) demonstrate that MCA-RG achieves superior performance, highlighting its effectiveness in radiology report generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15146v1">QueryGenie: Making LLM-Based Database Querying Transparent and Controllable</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Accepted by The 38th Annual ACM Symposium on User Interface Software and Technology (UIST Adjunct '25), September 28-October 1, 2025, Busan, Republic of Korea
    </div>
    <details class="paper-abstract">
      Conversational user interfaces powered by large language models (LLMs) have significantly lowered the technical barriers to database querying. However, existing tools still encounter several challenges, such as misinterpretation of user intent, generation of hallucinated content, and the absence of effective mechanisms for human feedback-all of which undermine their reliability and practical utility. To address these issues and promote a more transparent and controllable querying experience, we proposed QueryGenie, an interactive system that enables users to monitor, understand, and guide the LLM-driven query generation process. Through incremental reasoning, real-time validation, and responsive interaction mechanisms, users can iteratively refine query logic and ensure alignment with their intent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16695v1">Do Cognitively Interpretable Reasoning Traces Improve LLM Performance?</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Recent progress in reasoning-oriented Large Language Models (LLMs) has been driven by introducing Chain-of-Thought (CoT) traces, where models generate intermediate reasoning traces before producing an answer. These traces, as in DeepSeek R1, are not only used to guide inference but also serve as supervision signals for distillation into smaller models. A common but often implicit assumption is that CoT traces should be semantically meaningful and interpretable to the end user. While recent research questions the need for semantic nature of these traces, in this paper, we ask: ``\textit{Must CoT reasoning traces be interpretable to enhance LLM task performance?}" We investigate this question in the Open Book Question-Answering domain by supervised fine-tuning LLaMA and Qwen models on four types of reasoning traces: (1) DeepSeek R1 traces, (2) LLM-generated summaries of R1 traces, (3) LLM-generated post-hoc explanations of R1 traces, and (4) algorithmically generated verifiably correct traces. To quantify the trade-off between interpretability and performance, we further conduct a human-subject study with 100 participants rating the interpretability of each trace type. Our results reveal a striking mismatch: while fine-tuning on R1 traces yields the strongest performance, participants judged these traces to be the least interpretable. These findings suggest that it is useful to decouple intermediate tokens from end user interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07143v2">Ask Patients with Patience: Enabling LLMs for Human-Centric Medical Dialogue with Grounded Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      The severe shortage of medical doctors limits access to timely and reliable healthcare, leaving millions underserved. Large language models (LLMs) offer a potential solution but struggle in real-world clinical interactions. Many LLMs are not grounded in authoritative medical guidelines and fail to transparently manage diagnostic uncertainty. Their language is often rigid and mechanical, lacking the human-like qualities essential for patient trust. To address these challenges, we propose Ask Patients with Patience (APP), a multi-turn LLM-based medical assistant designed for grounded reasoning, transparent diagnoses, and human-centric interaction. APP enhances communication by eliciting user symptoms through empathetic dialogue, significantly improving accessibility and user engagement. It also incorporates Bayesian active learning to support transparent and adaptive diagnoses. The framework is built on verified medical guidelines, ensuring clinically grounded and evidence-based reasoning. To evaluate its performance, we develop a new benchmark that simulates realistic medical conversations using patient agents driven by profiles extracted from real-world consultation cases. We compare APP against SOTA one-shot and multi-turn LLM baselines. The results show that APP improves diagnostic accuracy, reduces uncertainty, and enhances user experience. By integrating medical expertise with transparent, human-like interaction, APP bridges the gap between AI-driven medical assistance and real-world clinical practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16835v2">Evaluating Speech-to-Text x LLM x Text-to-Speech Combinations for AI Interview Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Voice-based conversational AI systems increasingly rely on cascaded architectures that combine speech-to-text (STT), large language models (LLMs), and text-to-speech (TTS) components. We present a large-scale empirical comparison of STT x LLM x TTS stacks using data sampled from over 300,000 AI-conducted job interviews. We used an LLM-as-a-Judge automated evaluation framework to assess conversational quality, technical accuracy, and skill assessment capabilities. Our analysis of five production configurations reveals that a stack combining Google's STT, GPT-4.1, and Cartesia's TTS outperforms alternatives in both objective quality metrics and user satisfaction scores. Surprisingly, we find that objective quality metrics correlate weakly with user satisfaction scores, suggesting that user experience in voice-based AI systems depends on factors beyond technical performance. Our findings provide practical guidance for selecting components in multimodal conversations and contribute a validated evaluation methodology for human-AI interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10868v4">NitiBench: A Comprehensive Study of LLM Framework Capabilities for Thai Legal Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      The application of large language models (LLMs) in the legal domain holds significant potential for information retrieval and question answering, yet Thai legal QA systems face challenges due to a lack of standardized evaluation benchmarks and the complexity of Thai legal structures. This paper introduces NitiBench, a benchmark comprising two datasets: the NitiBench-CCL, covering general Thai financial law, and the NitiBench-Tax, which includes real-world tax law cases requiring advanced legal reasoning. We evaluate retrieval-augmented generation (RAG) and long-context LLM-based approaches to address three key research questions: the impact of domain-specific components like section-based chunking and cross-referencing, the comparative performance of different retrievers and LLMs, and the viability of long-context LLMs as an alternative to RAG. Our results show that section-based chunking significantly improves retrieval and end-to-end performance, current retrievers struggle with complex queries, and long-context LLMs still underperform RAG-based systems in Thai legal QA. To support fair evaluation, we propose tailored multi-label retrieval metrics and the use of an LLM-as-judge for coverage and contradiction detection method. These findings highlight the limitations of current Thai legal NLP solutions and provide a foundation for future research in the field. We also open-sourced our codes and dataset to available publicly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02725v2">Leveraging LLM Tutoring Systems for Non-Native English Speakers in Introductory CS Courses</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 11 pages, 5 tables, 4 figures, 2025 ASEE Annual Conference & Exposition
    </div>
    <details class="paper-abstract">
      Computer science has historically presented barriers for non-native English speaking (NNES) students, often due to language and terminology challenges. With the rise of large language models (LLMs), there is potential to leverage this technology to support NNES students more effectively. Recent implementations of LLMs as tutors in classrooms have shown promising results. In this study, we deployed an LLM tutor in an accelerated introductory computing course to evaluate its effectiveness specifically for NNES students. Key insights for LLM tutor use are as follows: NNES students signed up for the LLM tutor at a similar rate to native English speakers (NES); NNES students used the system at a lower rate than NES students -- to a small effect; NNES students asked significantly more questions in languages other than English compared to NES students, with many of the questions being multilingual by incorporating English programming keywords. Results for views of the LLM tutor are as follows: both NNES and NES students appreciated the LLM tutor for its accessibility, conversational style, and the guardrails put in place to guide users to answers rather than directly providing solutions; NNES students highlighted its approachability as they did not need to communicate in perfect English; NNES students rated help-seeking preferences of online resources higher than NES students; Many NNES students were unfamiliar with computing terminology in their native languages. These results suggest that LLM tutors can be a valuable resource for NNES students in computing, providing tailored support that enhances their learning experience and overcomes language barriers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16107v2">Do LLMs write like humans? Variation in grammatical and rhetorical styles</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 7 pages, 4 figures, 1 table
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are capable of writing grammatical text that follows instructions, answers questions, and solves problems. As they have advanced, it has become difficult to distinguish their output from human-written text. While past research has found some differences in surface features such as word choice and punctuation, and developed classifiers to detect LLM output, none has studied the rhetorical styles of LLMs. Using several variants of Llama 3 and GPT-4o, we construct two parallel corpora of human- and LLM-written texts from common prompts. Using Douglas Biber's set of lexical, grammatical, and rhetorical features, we identify systematic differences between LLMs and humans and between different LLMs. These differences persist when moving from smaller models to larger ones, and are larger for instruction-tuned models than base models. This observation of differences demonstrates that despite their advanced abilities, LLMs struggle to match human stylistic variation. Attention to more advanced linguistic features can hence detect patterns in their behavior not previously recognized.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15926v1">Noise, Adaptation, and Strategy: Assessing LLM Fidelity in Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Accepted to EMNLP 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in social science simulations. While their performance on reasoning and optimization tasks has been extensively evaluated, less attention has been paid to their ability to simulate human decision-making's variability and adaptability. We propose a process-oriented evaluation framework with progressive interventions (Intrinsicality, Instruction, and Imitation) to examine how LLM agents adapt under different levels of external guidance and human-derived noise. We validate the framework on two classic economics tasks, irrationality in the second-price auction and decision bias in the newsvendor problem, showing behavioral gaps between LLMs and humans. We find that LLMs, by default, converge on stable and conservative strategies that diverge from observed human behaviors. Risk-framed instructions impact LLM behavior predictably but do not replicate human-like diversity. Incorporating human data through in-context learning narrows the gap but fails to reach human subjects' strategic variability. These results highlight a persistent alignment gap in behavioral fidelity and suggest that future LLM evaluations should consider more process-level realism. We present a process-oriented approach for assessing LLMs in dynamic decision-making tasks, offering guidance for their application in synthetic data for social science research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05965v3">Validating LLM-as-a-Judge Systems under Rating Indeterminacy</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      The LLM-as-a-judge paradigm, in which a judge LLM system replaces human raters in rating the outputs of other generative AI (GenAI) systems, plays a critical role in scaling and standardizing GenAI evaluations. To validate such judge systems, evaluators assess human--judge agreement by first collecting multiple human ratings for each item in a validation corpus, then aggregating the ratings into a single, per-item gold label rating. For many items, however, rating criteria may admit multiple valid interpretations, so a human or LLM rater may deem multiple ratings "reasonable" or "correct". We call this condition rating indeterminacy. Problematically, many rating tasks that contain rating indeterminacy rely on forced-choice elicitation, whereby raters are instructed to select only one rating for each item. In this paper, we introduce a framework for validating LLM-as-a-judge systems under rating indeterminacy. We draw theoretical connections between different measures of judge system performance under different human--judge agreement metrics, and different rating elicitation and aggregation schemes. We demonstrate that differences in how humans and LLMs resolve rating indeterminacy while responding to forced-choice rating instructions heavily bias LLM-as-a-judge validation. Through extensive experiments involving 11 real-world rating tasks and 8 commercial LLMs, we show that standard validation approaches that rely upon forced-choice ratings select judge systems that are highly suboptimal, performing as much as 30% worse than judge systems selected by our approach that uses multi-label "response set" ratings to account for rating indeterminacy. We conclude with concrete recommendations for more principled approaches to LLM-as-a-judge validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16684v1">Democratizing AI Development: Local LLM Deployment for India's Developer Ecosystem in the Era of Tokenized APIs</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 for survey results, check https://docs.google.com/spreadsheets/d/1t0eV9oURaiu2HfARWo6sriBO0eC8bHUyZNN7CgK2NAk/edit?usp=sharing
    </div>
    <details class="paper-abstract">
      India's developer community faces significant barriers to sustained experimentation and learning with commercial Large Language Model (LLM) APIs, primarily due to economic and infrastructural constraints. This study empirically evaluates local LLM deployment using Ollama as an alternative to commercial cloud-based services for developer-focused applications. Through a mixed-methods analysis involving 180 Indian developers, students, and AI enthusiasts, we find that local deployment enables substantially greater hands-on development and experimentation, while reducing costs by 33% compared to commercial solutions. Developers using local LLMs completed over twice as many experimental iterations and reported deeper understanding of advanced AI architectures. Our results highlight local deployment as a critical enabler for inclusive and accessible AI development, demonstrating how technological accessibility can enhance learning outcomes and innovation capacity in resource-constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15877v1">Annif at the GermEval-2025 LLMs4Subjects Task: Traditional XMTC Augmented by Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 5 pages, 4 figures, accepted at KONVENS 2025. arXiv admin note: substantial text overlap with arXiv:2504.19675
    </div>
    <details class="paper-abstract">
      This paper presents the Annif system in the LLMs4Subjects shared task (Subtask 2) at GermEval-2025. The task required creating subject predictions for bibliographic records using large language models, with a special focus on computational efficiency. Our system, based on the Annif automated subject indexing toolkit, refines our previous system from the first LLMs4Subjects shared task, which produced excellent results. We further improved the system by using many small and efficient language models for translation and synthetic data generation and by using LLMs for ranking candidate subjects. Our system ranked 1st in the overall quantitative evaluation of and 1st in the qualitative evaluation of Subtask 2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15875v1">NEAT: Concept driven Neuron Attribution in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Locating neurons that are responsible for final predictions is important for opening the black-box large language models and understanding the inside mechanisms. Previous studies have tried to find mechanisms that operate at the neuron level but these methods fail to represent a concept and there is also scope for further optimization of compute required. In this paper, with the help of concept vectors, we propose a method for locating significant neurons that are responsible for representing certain concepts and term those neurons as concept neurons. If the number of neurons is n and the number of examples is m, we reduce the number of forward passes required from O(n*m) to just O(n) compared to the previous works and hence optimizing the time and computation required over previous works. We also compare our method with several baselines and previous methods and our results demonstrate better performance than most of the methods and are more optimal when compared to the state-of-the-art method. We, as part of our ablation studies, also try to optimize the search for the concept neurons by involving clustering methods. Finally, we apply our methods to find, turn off the neurons that we find, and analyze its implications in parts of hate speech and bias in LLMs, and we also evaluate our bias part in terms of Indian context. Our methodology, analysis and explanations facilitate understating of neuron-level responsibility for more broader and human-like concepts and also lay a path for future research in this direction of finding concept neurons and intervening them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16676v1">WISCA: A Lightweight Model Transition Method to Improve LLM Training via Weight Scaling</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Transformer architecture gradually dominates the LLM field. Recent advances in training optimization for Transformer-based large language models (LLMs) primarily focus on architectural modifications or optimizer adjustments. However, these approaches lack systematic optimization of weight patterns during training. Weight pattern refers to the distribution and relative magnitudes of weight parameters in a neural network. To address this issue, we propose a Weight Scaling method called WISCA to enhance training efficiency and model quality by strategically improving neural network weight patterns without changing network structures. By rescaling weights while preserving model outputs, WISCA indirectly optimizes the model's training trajectory. Experiments demonstrate that WISCA significantly improves convergence quality (measured by generalization capability and loss reduction), particularly in LLMs with Grouped Query Attention (GQA) architectures and LoRA fine-tuning tasks. Empirical results show 5.6% average improvement on zero-shot validation tasks and 2.12% average reduction in training perplexity across multiple architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15868v1">CARFT: Boosting LLM Reasoning via Contrastive Learning with Annotated Chain-of-Thought-based Reinforced Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 14 pages, to appear in EMNLP25
    </div>
    <details class="paper-abstract">
      Reasoning capability plays a significantly critical role in the the broad applications of Large Language Models (LLMs). To enhance the reasoning performance of LLMs, diverse Reinforcement Learning (RL)-based fine-tuning approaches have been proposed to address the limited generalization capability of LLMs trained solely via Supervised Fine-Tuning (SFT). Despite their effectiveness, two major limitations hinder the advancement of LLMs. First, vanilla RL-based approaches ignore annotated Chain-of-Thought (CoT) and incorporate unstable reasoning path sampling, which typically results in model collapse, unstable training process, and suboptimal performance. Second, existing SFT approaches generally overemphasize the annotated CoT, potentially leading to performance degradation due to insufficient exploitation of potential CoT. In this paper, we propose a Contrastive learning with annotated CoT-based Reinforced Fine-Tuning approach, i.e., \TheName{}, to enhance the reasoning performance of LLMs while addressing the aforementioned limitations. Specifically, we propose learning a representation for each CoT. Based on this representation, we design novel contrastive signals to guide the fine-tuning process. Our approach not only fully exploits the available annotated CoT but also stabilizes the fine-tuning procedure by incorporating an additional unsupervised learning signal. We conduct comprehensive experiments and in-depth analysis with three baseline approaches, two foundation models, and two datasets to demonstrate significant advantages of \TheName{} in terms of robustness, performance (up to 10.15\%), and efficiency (up to 30.62\%). Code is available at https://github.com/WNQzhu/CARFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.05469v2">Let's Measure Information Step-by-Step: LLM-Based Evaluation Beyond Vibes</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Add AUC results, pre-reg conformance, theory section clarification. 12 pages
    </div>
    <details class="paper-abstract">
      We study evaluation of AI systems without ground truth by exploiting a link between strategic gaming and information loss. We analyze which information-theoretic mechanisms resist adversarial manipulation, extending finite-sample bounds to show that bounded f-divergences (e.g., total variation distance) maintain polynomial guarantees under attacks while unbounded measures (e.g., KL divergence) degrade exponentially. To implement these mechanisms, we model the overseer as an agent and characterize incentive-compatible scoring rules as f-mutual information objectives. Under adversarial attacks, TVD-MI maintains effectiveness (area under curve 0.70-0.77) while traditional judge queries are near change (AUC $\approx$ 0.50), demonstrating that querying the same LLM for information relationships rather than quality judgments provides both theoretical and practical robustness. The mechanisms decompose pairwise evaluations into reliable item-level quality scores without ground truth, addressing a key limitation of traditional peer prediction. We release preregistration and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15706v1">Communication Efficient LLM Pre-training with SparseLoCo</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 15 pages, 9 tables, 2 figures
    </div>
    <details class="paper-abstract">
      Communication-efficient distributed training algorithms have received considerable interest recently due to their benefits for training Large Language Models (LLMs) in bandwidth-constrained settings, such as across data centers and over the internet. Despite reducing communication frequency, these methods still typically require communicating a full copy of the model's gradients-resulting in a communication bottleneck even for cross-datacenter links. Furthermore, they can slightly degrade performance compared to a naive AdamW DDP baseline. While quantization and error feedback are often applied to reduce the pseudo-gradient's size, in the context of LLM pre-training, existing approaches have been unable to additionally leverage sparsification and have obtained limited quantization. In this work, we introduce SparseLoCo, a communication-efficient training algorithm for LLMs that effectively leverages Top-k sparsification and quantization to reach extreme compression ratios of up to 1-3% sparsity and 2-bit quantization while outperforming full-precision DiLoCo. Our key observations are that outer momentum can be locally approximated by an error feedback combined with aggressive sparsity and that sparse aggregation can actually improve model performance. We empirically demonstrate in a range of communication-constrained LLM training settings that SparseLoCo provides significant benefits in both performance and communication cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08177v3">SycEval: Evaluating LLM Sycophancy</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 AIES 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in educational, clinical, and professional settings, but their tendency for sycophancy -- prioritizing user agreement over independent reasoning -- poses risks to reliability. This study introduces a framework to evaluate sycophantic behavior in ChatGPT-4o, Claude-Sonnet, and Gemini-1.5-Pro across AMPS (mathematics) and MedQuad (medical advice) datasets. Sycophantic behavior was observed in 58.19% of cases, with Gemini exhibiting the highest rate (62.47%) and ChatGPT the lowest (56.71%). Progressive sycophancy, leading to correct answers, occurred in 43.52% of cases, while regressive sycophancy, leading to incorrect answers, was observed in 14.66%. Preemptive rebuttals demonstrated significantly higher sycophancy rates than in-context rebuttals (61.75% vs. 56.52%, $Z=5.87$, $p<0.001$), particularly in computational tasks, where regressive sycophancy increased significantly (preemptive: 8.13%, in-context: 3.54%, $p<0.001$). Simple rebuttals maximized progressive sycophancy ($Z=6.59$, $p<0.001$), while citation-based rebuttals exhibited the highest regressive rates ($Z=6.59$, $p<0.001$). Sycophantic behavior showed high persistence (78.5%, 95% CI: [77.2%, 79.8%]) regardless of context or model. These findings emphasize the risks and opportunities of deploying LLMs in structured and dynamic domains, offering insights into prompt programming and model optimization for safer AI applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15688v1">LLM-empowered Dynamic Prompt Routing for Vision-Language Models Tuning under Long-Tailed Distributions</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      Pre-trained vision-language models (VLMs), such as CLIP, have demonstrated impressive capability in visual tasks, but their fine-tuning often suffers from bias in class-imbalanced scene. Recent works have introduced large language models (LLMs) to enhance VLM fine-tuning with supplementing semantic information. However, they often overlook inherent class imbalance in VLMs' pre-training, which may lead to bias accumulation in downstream tasks. To address this problem, this paper proposes a Multi-dimensional Dynamic Prompt Routing (MDPR) framework. MDPR constructs a comprehensive knowledge base for classes, spanning five visual-semantic dimensions. During fine-tuning, the dynamic routing mechanism aligns global visual classes, retrieves optimal prompts, and balances fine-grained semantics, yielding stable predictions through logits fusion. Extensive experiments on long-tailed benchmarks, including CIFAR-LT, ImageNet-LT, and Places-LT, demonstrate that MDPR achieves comparable results with current SOTA methods. Ablation studies further confirm the effectiveness of our semantic library for tail classes, and show that our dynamic routing incurs minimal computational overhead, making MDPR a flexible and efficient enhancement for VLM fine-tuning under data imbalance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13092v3">VerilogLAVD: LLM-Aided Rule Generation for Vulnerability Detection in Verilog</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Timely detection of hardware vulnerabilities during the early design stage is critical for reducing remediation costs. Existing early detection techniques often require specialized security expertise, limiting their usability. Recent efforts have explored the use of large language models (LLMs) for Verilog vulnerability detection. However, LLMs struggle to capture the structure in Verilog code, resulting in inconsistent detection results. To this end, we propose VerilogLAVD, the first LLM-aided graph traversal rule generation approach for Verilog vulnerability detection. Our approach introduces the Verilog Property Graph (VeriPG), a unified representation of Verilog code. It combines syntactic features extracted from the abstract syntax tree (AST) with semantic information derived from control flow and data dependency graphs. We leverage LLMs to generate VeriPG-based detection rules from Common Weakness Enumeration (CWE) descriptions. These rules guide the rule executor that traversal VeriPG for potential vulnerabilities. To evaluate VerilogLAVD, we build a dataset collected from open-source repositories and synthesized data. In our empirical evaluation on 77 Verilog designs encompassing 12 CWE types, VerilogLAVD achieves an F1-score of 0.54. Compared to the LLM-only and LLM with external knowledge baselines, VerilogLAVD improves F1-score by 0.31 and 0.27, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14716v2">Pairwise or Pointwise? Evaluating Feedback Protocols for Bias in LLM-Based Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Published at COLM 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used as proxies for human labelers in both training (Reinforcement Learning from AI Feedback) and large-scale response evaluation (LLM-as-a-judge). Alignment and evaluation are critical components in the development of reliable LLMs, and the choice of feedback protocol plays a central role in both but remains understudied. In this work, we show that the choice of feedback protocol for evaluation (absolute scores versus relative preferences) can significantly affect evaluation reliability and induce systematic biases. In the context of LLM-as-a-judge evaluation, we show that pairwise protocols are more vulnerable to distracted evaluation. Generator models can exploit spurious attributes (or distractor features) favored by the LLM judge, resulting in inflated scores for lower-quality outputs. We find that absolute scoring is more robust to such manipulation, producing judgments that better reflect response quality and are less influenced by distractor features. Our results demonstrate that generator models can flip preferences by embedding distractor features, skewing LLM-as-a-judge comparisons and leading to inaccurate conclusions about model quality in benchmark evaluations. Pairwise preferences flip in about 35% of the cases, compared to only 9% for absolute scores. We offer recommendations for choosing feedback protocols based on dataset characteristics and evaluation objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19793v2">Prompt Injection Attack to Tool Selection in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Tool selection is a key component of LLM agents. A popular approach follows a two-step process - \emph{retrieval} and \emph{selection} - to pick the most appropriate tool from a tool library for a given task. In this work, we introduce \textit{ToolHijacker}, a novel prompt injection attack targeting tool selection in no-box scenarios. ToolHijacker injects a malicious tool document into the tool library to manipulate the LLM agent's tool selection process, compelling it to consistently choose the attacker's malicious tool for an attacker-chosen target task. Specifically, we formulate the crafting of such tool documents as an optimization problem and propose a two-phase optimization strategy to solve it. Our extensive experimental evaluation shows that ToolHijacker is highly effective, significantly outperforming existing manual-based and automated prompt injection attacks when applied to tool selection. Moreover, we explore various defenses, including prevention-based defenses (StruQ and SecAlign) and detection-based defenses (known-answer detection, DataSentinel, perplexity detection, and perplexity windowed detection). Our experimental results indicate that these defenses are insufficient, highlighting the urgent need for developing new defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19675v2">Annif at SemEval-2025 Task 5: Traditional XMTC augmented by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 6 pages, 4 figures, published at SemEval-2025 workshop Task 5: LLMs4Subjects: https://aclanthology.org/2025.semeval-1.315/
    </div>
    <details class="paper-abstract">
      This paper presents the Annif system in SemEval-2025 Task 5 (LLMs4Subjects), which focussed on subject indexing using large language models (LLMs). The task required creating subject predictions for bibliographic records from the bilingual TIBKAT database using the GND subject vocabulary. Our approach combines traditional natural language processing and machine learning techniques implemented in the Annif toolkit with innovative LLM-based methods for translation and synthetic data generation, and merging predictions from monolingual models. The system ranked first in the all-subjects category and second in the tib-core-subjects category in the quantitative evaluation, and fourth in qualitative evaluations. These findings demonstrate the potential of combining traditional XMTC algorithms with modern LLM techniques to improve the accuracy and efficiency of subject indexing in multilingual contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13618v4">Seed-X: Building Strong Multilingual Translation LLM with 7B Parameters</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Multilingual translation stands as a challenging task for large language models (LLMs) to handle intricate language patterns and stilted translations that arise in automated translations. In this paper, we introduce Seed-X, a family of open-source LLMs comprising instruct and reasoning models, pushing the limits of translation capability with 7B parameter size. The base model is pre-trained on a diverse, high-quality dataset encompassing both monolingual and bilingual content across 28 languages, harnessing the full potential of multilingual data. The instruct model is then finetuned to translate by Chain-of-Thought (CoT) reasoning and further enhanced through reinforcement learning (RL) to achieve better generalization across diverse language pairs. Seed-X achieves performance comparable to leading closed-source models, including Gemini-2.5 and GPT-4o, across 28 languages, and significantly outperforms larger open-source models in both automatic metrics and human evaluations. We share the best practices through our optimization process, and make the parameter public available for advancing translation research and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15526v1">SafetyFlow: An Agent-Flow System for Automated LLM Safety Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Code and dataset are available at https://github.com/yangyangyang127/SafetyFlow
    </div>
    <details class="paper-abstract">
      The rapid proliferation of large language models (LLMs) has intensified the requirement for reliable safety evaluation to uncover model vulnerabilities. To this end, numerous LLM safety evaluation benchmarks are proposed. However, existing benchmarks generally rely on labor-intensive manual curation, which causes excessive time and resource consumption. They also exhibit significant redundancy and limited difficulty. To alleviate these problems, we introduce SafetyFlow, the first agent-flow system designed to automate the construction of LLM safety benchmarks. SafetyFlow can automatically build a comprehensive safety benchmark in only four days without any human intervention by orchestrating seven specialized agents, significantly reducing time and resource cost. Equipped with versatile tools, the agents of SafetyFlow ensure process and cost controllability while integrating human expertise into the automatic pipeline. The final constructed dataset, SafetyFlowBench, contains 23,446 queries with low redundancy and strong discriminative power. Our contribution includes the first fully automated benchmarking pipeline and a comprehensive safety benchmark. We evaluate the safety of 49 advanced LLMs on our dataset and conduct extensive experiments to validate our efficacy and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15503v1">Evaluation Guidelines for Empirical Studies in Software Engineering involving LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Draft of evaluation guidelines for empirical studies in software engineering involving LLMs (see also llm-guidelines.org)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being integrated into software engineering (SE) research and practice, yet their non-determinism, opaque training data, and evolving architectures complicate the reproduction and replication of empirical studies. We present a community effort to scope this space, introducing a taxonomy of LLM-based study types together with eight guidelines for designing and reporting empirical studies involving LLMs. The guidelines present essential (must) criteria as well as desired (should) criteria and target transparency throughout the research process. Our recommendations, contextualized by our study types, are: (1) to declare LLM usage and role; (2) to report model versions, configurations, and fine-tuning; (3) to document tool architectures; (4) to disclose prompts and interaction logs; (5) to use human validation; (6) to employ an open LLM as a baseline; (7) to report suitable baselines, benchmarks, and metrics; and (8) to openly articulate limitations and mitigations. Our goal is to enable reproducibility and replicability despite LLM-specific barriers to open science. We maintain the study types and guidelines online as a living resource for the community to use and shape (llm-guidelines.org).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15501v1">LLM-Driven Self-Refinement for Embodied Drone Task Planning</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 14pages
    </div>
    <details class="paper-abstract">
      We introduce SRDrone, a novel system designed for self-refinement task planning in industrial-grade embodied drones. SRDrone incorporates two key technical contributions: First, it employs a continuous state evaluation methodology to robustly and accurately determine task outcomes and provide explanatory feedback. This approach supersedes conventional reliance on single-frame final-state assessment for continuous, dynamic drone operations. Second, SRDrone implements a hierarchical Behavior Tree (BT) modification model. This model integrates multi-level BT plan analysis with a constrained strategy space to enable structured reflective learning from experience. Experimental results demonstrate that SRDrone achieves a 44.87% improvement in Success Rate (SR) over baseline methods. Furthermore, real-world deployment utilizing an experience base optimized through iterative self-refinement attains a 96.25% SR. By embedding adaptive task refinement capabilities within an industrial-grade BT planning framework, SRDrone effectively integrates the general reasoning intelligence of Large Language Models (LLMs) with the stringent physical execution constraints inherent to embodied drones. Code is available at https://github.com/ZXiiiC/SRDrone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15495v1">SynthCoder: A Synthetical Strategy to Tune LLMs for Code Completion</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Code completion is a prominent application of Large Language Models (LLMs) in software engineering. Due to the near real-time response requirements of this task, base models with small to medium-sized parameters are typically employed, supplemented by various optimization and post-training techniques. However, these optimization methods often have trade-offs, leading to a seesaw effect where performance improvements on certain datasets or metrics are accompanied by degradations on others -- sometimes even falling below the baseline model's performance. This paper proposes SynthCoder, a model that integrates leading industry practices to achieve state-of-the-art performance on the Fill-in-the-Middle (FIM) code completion task. In specific, we first construct a diverse dataset by combining Abstract Syntax Tree (AST) node extraction with heuristics that simulate developer behavior. Then we enrich our training corpus with cross-file contextual information using the BM25 algorithm and call graphs, enhancing the model's ability to perform code completion in both file-level and repository-level scenarios. As the last step, we employ a two-stage training process using the Seed-Coder-8B-Base as the base model. First, we fine-tune the model using Curriculum Learning technology. Following this, we perform alignment using Direct Preference Optimization (DPO) with preference pairs generated through Rejection Sampling. Experimental results demonstrate that our final model excels on mainstream repository-level code completion benchmarks, including aiXcoder, ExecRepoBench, CrossCodeEval, and CoLT. Furthermore, our carefully curated training set effectively mitigates the model's tendency to just repeat existing code, a common issue existing in various code completion models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08768v3">AraReasoner: Evaluating Reasoning-Based LLMs for Arabic NLP</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable progress in reasoning abilities and general natural language processing (NLP) tasks, yet their performance on Arabic data, characterized by rich morphology, diverse dialects, and complex script, remains underexplored. This paper presents a comprehensive benchmarking study of multiple reasoning-focused LLMs, with a special emphasis on the newly introduced DeepSeek models, across a suite of fifteen Arabic NLP tasks. We experiment with various strategies, including zero-shot, few-shot, and fine-tuning. This allows us to systematically evaluate performance on datasets covering a range of applications to examine their capacity for linguistic reasoning under different levels of complexity. Our experiments reveal several key findings. First, carefully selecting just three in-context examples delivers an average uplift of over 13 F1 points on classification tasks-boosting sentiment analysis from 35.3% to 87.5% and paraphrase detection from 56.1% to 87.0%. Second, reasoning-focused DeepSeek architectures outperform a strong GPT o4-mini baseline by an average of 12 F1 points on complex inference tasks in the zero-shot setting. Third, LoRA-based fine-tuning yields up to an additional 8 points in F1 and BLEU compared to equivalent increases in model scale. The code is available at https://anonymous.4open.science/r/AraReasoner41299
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15474v1">Subjective Behaviors and Preferences in LLM: Language of Browsing</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      A Large Language Model (LLM) offers versatility across domains and tasks, purportedly benefiting users with a wide variety of behaviors and preferences. We question this perception about an LLM when users have inherently subjective behaviors and preferences, as seen in their ubiquitous and idiosyncratic browsing of websites or apps. The sequential behavior logs of pages, thus generated, form something akin to each user's self-constructed "language", albeit without the structure and grammar imbued in natural languages. We ask: (i) Can a small LM represent the "language of browsing" better than a large LM? (ii) Can an LM with a single set of parameters (or, single LM) adequately capture myriad users' heterogeneous, subjective behaviors and preferences? (iii) Can a single LM with high average performance, yield low variance in performance to make alignment good at user level? We introduce clusterwise LM training, HeTLM (Heterogeneity aware Training of Language Model), appropriate for subjective behaviors. We find that (i) a small LM trained using a page-level tokenizer outperforms large pretrained or finetuned LMs; (ii) HeTLM with heterogeneous cluster specific set of parameters outperforms a single LM of the same family, controlling for the number of parameters; and (iii) a higher mean and a lower variance in generation ensues, implying improved alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15449v1">Reliable Unlearning Harmful Information in LLMs with Metamorphosis Representation Projection</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 10 pages, 9 figures, Under review as a full paper at AAAI 2026. A preliminary version is under review at the NeurIPS 2025 Workshop on Reliable ML from Unreliable Data
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have demonstrated impressive performance in various domains and tasks, concerns about their safety are becoming increasingly severe. In particular, since models may store unsafe knowledge internally, machine unlearning has emerged as a representative paradigm to ensure model safety. Existing approaches employ various training techniques, such as gradient ascent and negative preference optimization, in attempts to eliminate the influence of undesired data on target models. However, these methods merely suppress the activation of undesired data through parametric training without completely eradicating its informational traces within the model. This fundamental limitation makes it difficult to achieve effective continuous unlearning, rendering these methods vulnerable to relearning attacks. To overcome these challenges, we propose a Metamorphosis Representation Projection (MRP) approach that pioneers the application of irreversible projection properties to machine unlearning. By implementing projective transformations in the hidden state space of specific network layers, our method effectively eliminates harmful information while preserving useful knowledge. Experimental results demonstrate that our approach enables effective continuous unlearning and successfully defends against relearning attacks, achieving state-of-the-art performance in unlearning effectiveness while preserving natural performance. Our code is available in https://github.com/ChengcanWu/MRP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15447v1">From Bits to Boardrooms: A Cutting-Edge Multi-Agent LLM Framework for Business Excellence</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 Accepted by ECAI 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promising potential in business applications, particularly in enterprise decision support and strategic planning, yet current approaches often struggle to reconcile intricate operational analyses with overarching strategic goals across diverse market environments, leading to fragmented workflows and reduced collaboration across organizational levels. This paper introduces BusiAgent, a novel multi-agent framework leveraging LLMs for advanced decision-making in complex corporate environments. BusiAgent integrates three core innovations: an extended Continuous Time Markov Decision Process (CTMDP) for dynamic agent modeling, a generalized entropy measure to optimize collaborative efficiency, and a multi-level Stackelberg game to handle hierarchical decision processes. Additionally, contextual Thompson sampling is employed for prompt optimization, supported by a comprehensive quality assurance system to mitigate errors. Extensive empirical evaluations across diverse business scenarios validate BusiAgent's efficacy, demonstrating its capacity to generate coherent, client-focused solutions that smoothly integrate granular insights with high-level strategy, significantly outperforming established approaches in both solution quality and user satisfaction. By fusing cutting-edge AI technologies with deep business insights, BusiAgent marks a substantial step forward in AI-driven enterprise decision-making, empowering organizations to navigate complex business landscapes more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01539v2">Pragmatic Inference Chain (PIC) Improving LLMs' Reasoning of Authentic Implicit Toxic Language</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 14 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The rapid development of large language models (LLMs) gives rise to ethical concerns about their performance, while opening new avenues for developing toxic language detection techniques. However, LLMs' unethical output and their capability of detecting toxicity have primarily been tested on language data that do not demand complex meaning inference, such as the biased associations of 'he' with programmer and 'she' with household. Nowadays toxic language adopts a much more creative range of implicit forms, thanks to advanced censorship. In this study, we collect authentic toxic interactions that evade online censorship and that are verified by human annotators as inference-intensive. To evaluate and improve LLMs' reasoning of the authentic implicit toxic language, we propose a new prompting method, Pragmatic Inference Chain (PIC), drawn on interdisciplinary findings from cognitive science and linguistics. The PIC prompting significantly improves the success rate of GPT-4o, Llama-3.1-70B-Instruct, DeepSeek-v2.5, and DeepSeek-v3 in identifying implicit toxic language, compared to five baseline prompts, such as CoT and rule-based baselines. In addition, it also facilitates the models to produce more explicit and coherent reasoning processes, hence can potentially be generalized to other inference-intensive tasks, e.g., understanding humour and metaphors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15360v1">An Empirical Study on How Video-LLMs Answer Video Questions</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Taking advantage of large-scale data and pretrained language models, Video Large Language Models (Video-LLMs) have shown strong capabilities in answering video questions. However, most existing efforts focus on improving performance, with limited attention to understanding their internal mechanisms. This paper aims to bridge this gap through a systematic empirical study. To interpret existing VideoLLMs, we adopt attention knockouts as our primary analytical tool and design three variants: Video Temporal Knockout, Video Spatial Knockout, and Language-to-Video Knockout. Then, we apply these three knockouts on different numbers of layers (window of layers). By carefully controlling the window of layers and types of knockouts, we provide two settings: a global setting and a fine-grained setting. Our study reveals three key findings: (1) Global setting indicates Video information extraction primarily occurs in early layers, forming a clear two-stage process -- lower layers focus on perceptual encoding, while higher layers handle abstract reasoning; (2) In the fine-grained setting, certain intermediate layers exert an outsized impact on video question answering, acting as critical outliers, whereas most other layers contribute minimally; (3) In both settings, we observe that spatial-temporal modeling relies more on language-guided retrieval than on intra- and inter-frame self-attention among video tokens, despite the latter's high computational cost. Finally, we demonstrate that these insights can be leveraged to reduce attention computation in Video-LLMs. To our knowledge, this is the first work to systematically uncover how Video-LLMs internally process and understand video content, offering interpretability and efficiency perspectives for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15338v1">DiagECG: An LLM-Driven Framework for Diagnostic Reasoning via Discretized ECG Tokenization</a></div>
    <div class="paper-meta">
      📅 2025-08-21
    </div>
    <details class="paper-abstract">
      Electrocardiography plays a central role in cardiovascular diagnostics, yet existing automated approaches often struggle to generalize across clinical tasks and offer limited support for open-ended reasoning. We present DiagECG, a novel framework that integrates time-series and language modeling by enabling large language models to process 12-lead ECG signals for clinical text generation tasks. Our approach discretizes continuous ECG embeddings into symbolic tokens using a lead-independent encoder and quantization module. These tokens are then used to extend the vocabulary of LLM, allowing the model to handle both ECG and natural language inputs in a unified manner. To bridge the modality gap, we pretrain the model on an autoregressive ECG forecasting task, enabling the LLM to model temporal dynamics using its native language modeling capabilities. Finally, we perform instruction tuning on both ECG question answering and diagnostic report generation. Without modifying the core model, DiagECG achieves strong performance across tasks while maintaining generalization to out-of-distribution settings. Extensive experiments demonstrate the effectiveness of each component and highlight the potential of integrating symbolic ECG representations into LLMs for medical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15310v1">IPIGuard: A Novel Tool Dependency Graph-Based Defense Against Indirect Prompt Injection in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-21
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are widely deployed in real-world applications, where they leverage tools to retrieve and manipulate external data for complex tasks. However, when interacting with untrusted data sources (e.g., fetching information from public websites), tool responses may contain injected instructions that covertly influence agent behaviors and lead to malicious outcomes, a threat referred to as Indirect Prompt Injection (IPI). Existing defenses typically rely on advanced prompting strategies or auxiliary detection models. While these methods have demonstrated some effectiveness, they fundamentally rely on assumptions about the model's inherent security, which lacks structural constraints on agent behaviors. As a result, agents still retain unrestricted access to tool invocations, leaving them vulnerable to stronger attack vectors that can bypass the security guardrails of the model. To prevent malicious tool invocations at the source, we propose a novel defensive task execution paradigm, called IPIGuard, which models the agents' task execution process as a traversal over a planned Tool Dependency Graph (TDG). By explicitly decoupling action planning from interaction with external data, IPIGuard significantly reduces unintended tool invocations triggered by injected instructions, thereby enhancing robustness against IPI attacks. Experiments on the AgentDojo benchmark show that IPIGuard achieves a superior balance between effectiveness and robustness, paving the way for the development of safer agentic systems in dynamic environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14544v1">Adaptively Robust LLM Inference Optimization under Prediction Uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      We study the problem of optimizing Large Language Model (LLM) inference scheduling to minimize total latency. LLM inference is an online and multi-task service process and also heavily energy consuming by which a pre-trained LLM processes input requests and generates output tokens sequentially. Therefore, it is vital to improve its scheduling efficiency and reduce the power consumption while a great amount of prompt requests are arriving. A key challenge in LLM inference scheduling is that while the prompt length is known upon arrival, the output length, which critically impacts memory usage and processing time, is unknown. To address this uncertainty, we propose algorithms that leverage machine learning to predict output lengths, assuming the prediction provides an interval classification (min-max range) for each request. We first design a conservative algorithm, $\mathcal{A}_{\max}$, which schedules requests based on the upper bound of predicted output lengths to prevent memory overflow. However, this approach is overly conservative: as prediction accuracy decreases, performance degrades significantly due to potential overestimation. To overcome this limitation, we propose $\mathcal{A}_{\min}$, an adaptive algorithm that initially treats the predicted lower bound as the output length and dynamically refines this estimate during inferencing. We prove that $\mathcal{A}_{\min}$ achieves a log-scale competitive ratio. Through numerical simulations, we demonstrate that $\mathcal{A}_{\min}$ often performs nearly as well as the hindsight scheduler, highlighting both its efficiency and robustness in practical scenarios. Moreover, $\mathcal{A}_{\min}$ relies solely on the lower bound of the prediction interval--an advantageous design choice since upper bounds on output length are typically more challenging to predict accurately.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02329v2">ReSpark: Leveraging Previous Data Reports as References to Generate New Reports with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Creating data reports is a labor-intensive task involving iterative data exploration, insight extraction, and narrative construction. A key challenge lies in composing the analysis logic-from defining objectives and transforming data to identifying and communicating insights. Manually crafting this logic can be cognitively demanding. While experienced analysts often reuse scripts from past projects, finding a perfect match for a new dataset is rare. Even when similar analyses are available online, they usually share only results or visualizations, not the underlying code, making reuse difficult. To address this, we present ReSpark, a system that leverages large language models (LLMs) to reverse-engineer analysis logic from existing reports and adapt it to new datasets. By generating draft analysis steps, ReSpark provides a warm start for users. It also supports interactive refinement, allowing users to inspect intermediate outputs, insert objectives, and revise content. We evaluate ReSpark through comparative and user studies, demonstrating its effectiveness in lowering the barrier to generating data reports without relying on existing analysis code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14496v1">Semantic Energy: Detecting LLM Hallucination Beyond Entropy</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being increasingly deployed in real-world applications, but they remain susceptible to hallucinations, which produce fluent yet incorrect responses and lead to erroneous decision-making. Uncertainty estimation is a feasible approach to detect such hallucinations. For example, semantic entropy estimates uncertainty by considering the semantic diversity across multiple sampled responses, thus identifying hallucinations. However, semantic entropy relies on post-softmax probabilities and fails to capture the model's inherent uncertainty, causing it to be ineffective in certain scenarios. To address this issue, we introduce Semantic Energy, a novel uncertainty estimation framework that leverages the inherent confidence of LLMs by operating directly on logits of penultimate layer. By combining semantic clustering with a Boltzmann-inspired energy distribution, our method better captures uncertainty in cases where semantic entropy fails. Experiments across multiple benchmarks show that Semantic Energy significantly improves hallucination detection and uncertainty estimation, offering more reliable signals for downstream applications such as hallucination detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14460v1">DuPO: Enabling Reliable LLM Self-Verification via Dual Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 18 pages, 4 figures,
    </div>
    <details class="paper-abstract">
      We present DuPO, a dual learning-based preference optimization framework that generates annotation-free feedback via a generalized duality. DuPO addresses two key limitations: Reinforcement Learning with Verifiable Rewards (RLVR)'s reliance on costly labels and applicability restricted to verifiable tasks, and traditional dual learning's restriction to strictly dual task pairs (e.g., translation and back-translation). Specifically, DuPO decomposes a primal task's input into known and unknown components, then constructs its dual task to reconstruct the unknown part using the primal output and known information (e.g., reversing math solutions to recover hidden variables), broadening applicability to non-invertible tasks. The quality of this reconstruction serves as a self-supervised reward to optimize the primal task, synergizing with LLMs' ability to instantiate both tasks via a single model. Empirically, DuPO achieves substantial gains across diverse tasks: it enhances the average translation quality by 2.13 COMET over 756 directions, boosts the mathematical reasoning accuracy by an average of 6.4 points on three challenge benchmarks, and enhances performance by 9.3 points as an inference-time reranker (trading computation for accuracy). These results position DuPO as a scalable, general, and annotation-free paradigm for LLM optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14419v1">Static Analysis as a Feedback Loop: Enhancing LLM-Generated Code Beyond Correctness</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in code generation, achieving high scores on benchmarks such as HumanEval and MBPP. However, these benchmarks primarily assess functional correctness and neglect broader dimensions of code quality, including security, reliability, readability, and maintainability. In this work, we systematically evaluate the ability of LLMs to generate high-quality code across multiple dimensions using the PythonSecurityEval benchmark. We introduce an iterative static analysis-driven prompting algorithm that leverages Bandit and Pylint to identify and resolve code quality issues. Our experiments with GPT-4o show substantial improvements: security issues reduced from >40% to 13%, readability violations from >80% to 11%, and reliability warnings from >50% to 11% within ten iterations. These results demonstrate that LLMs, when guided by static analysis feedback, can significantly enhance code quality beyond functional correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14408v1">Cognitive Surgery: The Awakening of Implicit Territorial Awareness in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to possess a degree of self-recognition capability-the ability to identify whether a given text was generated by themselves. Prior work has demonstrated that this capability is reliably expressed under the Pair Presentation Paradigm (PPP), where the model is presented with two texts and asked to choose which one it authored. However, performance deteriorates sharply under the Individual Presentation Paradigm (IPP), where the model is given a single text to judge authorship. Although this phenomenon has been observed, its underlying causes have not been systematically analyzed. In this paper, we first replicate existing findings to confirm that LLMs struggle to distinguish self- from other-generated text under IPP. We then investigate the reasons for this failure and attribute it to a phenomenon we term Implicit Territorial Awareness (ITA)-the model's latent ability to distinguish self- and other-texts in representational space, which remains unexpressed in its output behavior. To awaken the ITA of LLMs, we propose Cognitive Surgery (CoSur), a novel framework comprising four main modules: representation extraction, territory construction, authorship discrimination and cognitive editing. Experimental results demonstrate that our proposed method improves the performance of three different LLMs in the IPP scenario, achieving average accuracies of 83.25%, 66.19%, and 88.01%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12645v3">Diagnostic-Guided Dynamic Profile Optimization for LLM-based User Simulators in Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled realistic user simulators for developing and evaluating recommender systems (RSs). However, existing LLM-based simulators for RSs face two major limitations: (1) static and single-step prompt-based inference that leads to inaccurate and incomplete user profile construction; (2) unrealistic and single-round recommendation-feedback interaction pattern that fails to capture real-world scenarios. To address these limitations, we propose DGDPO (Diagnostic-Guided Dynamic Profile Optimization), a novel framework that constructs user profile through a dynamic and iterative optimization process to enhance the simulation fidelity. Specifically, DGDPO incorporates two core modules within each optimization loop: firstly, a specialized LLM-based diagnostic module, calibrated through our novel training strategy, accurately identifies specific defects in the user profile. Subsequently, a generalized LLM-based treatment module analyzes the diagnosed defect and generates targeted suggestions to refine the profile. Furthermore, unlike existing LLM-based user simulators that are limited to single-round interactions, we are the first to integrate DGDPO with sequential recommenders, enabling a bidirectional evolution where user profiles and recommendation strategies adapt to each other over multi-round interactions. Extensive experiments conducted on three real-world datasets demonstrate the effectiveness of our proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14377v1">ZPD-SCA: Unveiling the Blind Spots of LLMs in Assessing Students' Cognitive Abilities</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated potential in educational applications, yet their capacity to accurately assess the cognitive alignment of reading materials with students' developmental stages remains insufficiently explored. This gap is particularly critical given the foundational educational principle of the Zone of Proximal Development (ZPD), which emphasizes the need to match learning resources with Students' Cognitive Abilities (SCA). Despite the importance of this alignment, there is a notable absence of comprehensive studies investigating LLMs' ability to evaluate reading comprehension difficulty across different student age groups, especially in the context of Chinese language education. To fill this gap, we introduce ZPD-SCA, a novel benchmark specifically designed to assess stage-level Chinese reading comprehension difficulty. The benchmark is annotated by 60 Special Grade teachers, a group that represents the top 0.15% of all in-service teachers nationwide. Experimental results reveal that LLMs perform poorly in zero-shot learning scenarios, with Qwen-max and GLM even falling below the probability of random guessing. When provided with in-context examples, LLMs performance improves substantially, with some models achieving nearly double the accuracy of their zero-shot baselines. These results reveal that LLMs possess emerging abilities to assess reading difficulty, while also exposing limitations in their current training for educationally aligned judgment. Notably, even the best-performing models display systematic directional biases, suggesting difficulties in accurately aligning material difficulty with SCA. Furthermore, significant variations in model performance across different genres underscore the complexity of task. We envision that ZPD-SCA can provide a foundation for evaluating and improving LLMs in cognitively aligned educational applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19959v3">From Concept to Practice: an Automated LLM-aided UVM Machine for RTL Verification</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Verification presents a major bottleneck in Integrated Circuit (IC) development, consuming nearly 70% of the total development effort. While the Universal Verification Methodology (UVM) is widely used in industry to improve verification efficiency through structured and reusable testbenches, constructing these testbenches and generating sufficient stimuli remain challenging. These challenges arise from the considerable manual coding effort required, repetitive manual execution of multiple EDA tools, and the need for in-depth domain expertise to navigate complex designs.Here, we present UVM^2, an automated verification framework that leverages Large Language Models (LLMs) to generate UVM testbenches and iteratively refine them using coverage feedback, significantly reducing manual effort while maintaining rigorous verification standards.To evaluate UVM^2, we introduce a benchmark suite comprising Register Transfer Level (RTL) designs of up to 1.6K lines of code.The results show that UVM^2 reduces testbench setup time by up to UVM^2 compared to experienced engineers, and achieve average code and function coverage of 87.44% and 89.58%, outperforming state-of-the-art solutions by 20.96% and 23.51%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14357v1">Organ-Agents: Virtual Human Physiology Simulator via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled new possibilities in simulating complex physiological systems. We introduce Organ-Agents, a multi-agent framework that simulates human physiology via LLM-driven agents. Each Simulator models a specific system (e.g., cardiovascular, renal, immune). Training consists of supervised fine-tuning on system-specific time-series data, followed by reinforcement-guided coordination using dynamic reference selection and error correction. We curated data from 7,134 sepsis patients and 7,895 controls, generating high-resolution trajectories across 9 systems and 125 variables. Organ-Agents achieved high simulation accuracy on 4,509 held-out patients, with per-system MSEs <0.16 and robustness across SOFA-based severity strata. External validation on 22,689 ICU patients from two hospitals showed moderate degradation under distribution shifts with stable simulation. Organ-Agents faithfully reproduces critical multi-system events (e.g., hypotension, hyperlactatemia, hypoxemia) with coherent timing and phase progression. Evaluation by 15 critical care physicians confirmed realism and physiological plausibility (mean Likert ratings 3.9 and 3.7). Organ-Agents also enables counterfactual simulations under alternative sepsis treatment strategies, generating trajectories and APACHE II scores aligned with matched real-world patients. In downstream early warning tasks, classifiers trained on synthetic data showed minimal AUROC drops (<0.04), indicating preserved decision-relevant patterns. These results position Organ-Agents as a credible, interpretable, and generalizable digital twin for precision diagnosis, treatment simulation, and hypothesis testing in critical care.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15861v1">XFinBench: Benchmarking LLMs in Complex Financial Problem Solving and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Solving financial problems demands complex reasoning, multimodal data processing, and a broad technical understanding, presenting unique challenges for current large language models (LLMs). We introduce XFinBench, a novel benchmark with 4,235 examples designed to evaluate LLM's ability in solving complex, knowledge-intensive financial problems across diverse graduate-level finance topics with multi-modal context. We identify five core capabilities of LLMs using XFinBench, i.e, terminology understanding, temporal reasoning, future forecasting, scenario planning, and numerical modelling. Upon XFinBench, we conduct extensive experiments on 18 leading models. The result shows that o1 is the best-performing text-only model with an overall accuracy of 67.3%, but still lags significantly behind human experts with 12.5%, especially in temporal reasoning and scenario planning capabilities. We further construct a knowledge bank with 3,032 finance terms for knowledge augmentation analysis, and find that relevant knowledge to the question only brings consistent accuracy improvements to small open-source model. Additionally, our error analysis reveals that rounding errors during calculation and blindness to position and intersection of curves in the image are two primary issues leading to model's poor performance in calculating and visual-context questions, respectively. Code and dataset are accessible via GitHub: https://github.com/Zhihan72/XFinBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15854v1">QU-NLP at QIAS 2025 Shared Task: A Two-Phase LLM Fine-Tuning and Retrieval-Augmented Generation Approach for Islamic Inheritance Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      This paper presents our approach and results for SubTask 1: Islamic Inheritance Reasoning at QIAS 2025, a shared task focused on evaluating Large Language Models (LLMs) in understanding and reasoning within Islamic inheritance knowledge. We fine-tuned the Fanar-1-9B causal language model using Low-Rank Adaptation (LoRA) and integrated it into a Retrieval-Augmented Generation (RAG) pipeline. Our system addresses the complexities of Islamic inheritance law, including comprehending inheritance scenarios, identifying eligible heirs, applying fixed-share rules, and performing precise calculations. Our system achieved an accuracy of 0.858 in the final test, outperforming other competitive models such as, GPT 4.5, LLaMA, Fanar, Mistral and ALLaM evaluated with zero-shot prompting. Our results demonstrate that QU-NLP achieves near state-of-the-art accuracy (85.8%), excelling especially on advanced reasoning (97.6%) where it outperforms Gemini 2.5 and OpenAI's o3. This highlights that domain-specific fine-tuning combined with retrieval grounding enables mid-scale Arabic LLMs to surpass frontier models in Islamic inheritance reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16654v1">MSNav: Zero-Shot Vision-and-Language Navigation with Dynamic Memory and LLM Spatial Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) requires an agent to interpret natural language instructions and navigate complex environments. Current approaches often adopt a "black-box" paradigm, where a single Large Language Model (LLM) makes end-to-end decisions. However, it is plagued by critical vulnerabilities, including poor spatial reasoning, weak cross-modal grounding, and memory overload in long-horizon tasks. To systematically address these issues, we propose Memory Spatial Navigation(MSNav), a framework that fuses three modules into a synergistic architecture, which transforms fragile inference into a robust, integrated intelligence. MSNav integrates three modules: Memory Module, a dynamic map memory module that tackles memory overload through selective node pruning, enhancing long-range exploration; Spatial Module, a module for spatial reasoning and object relationship inference that improves endpoint recognition; and Decision Module, a module using LLM-based path planning to execute robust actions. Powering Spatial Module, we also introduce an Instruction-Object-Space (I-O-S) dataset and fine-tune the Qwen3-4B model into Qwen-Spatial (Qwen-Sp), which outperforms leading commercial LLMs in object list extraction, achieving higher F1 and NDCG scores on the I-O-S test set. Extensive experiments on the Room-to-Room (R2R) and REVERIE datasets demonstrate MSNav's state-of-the-art performance with significant improvements in Success Rate (SR) and Success weighted by Path Length (SPL).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15848v1">Self-Disguise Attack: Induce the LLM to disguise itself for AIGT detection evasion</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      AI-generated text (AIGT) detection evasion aims to reduce the detection probability of AIGT, helping to identify weaknesses in detectors and enhance their effectiveness and reliability in practical applications. Although existing evasion methods perform well, they suffer from high computational costs and text quality degradation. To address these challenges, we propose Self-Disguise Attack (SDA), a novel approach that enables Large Language Models (LLM) to actively disguise its output, reducing the likelihood of detection by classifiers. The SDA comprises two main components: the adversarial feature extractor and the retrieval-based context examples optimizer. The former generates disguise features that enable LLMs to understand how to produce more human-like text. The latter retrieves the most relevant examples from an external knowledge base as in-context examples, further enhancing the self-disguise ability of LLMs and mitigating the impact of the disguise process on the diversity of the generated text. The SDA directly employs prompts containing disguise features and optimized context examples to guide the LLM in generating detection-resistant text, thereby reducing resource consumption. Experimental results demonstrate that the SDA effectively reduces the average detection accuracy of various AIGT detectors across texts generated by three different LLMs, while maintaining the quality of AIGT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19073v3">MFTCXplain: A Multilingual Benchmark Dataset for Evaluating the Moral Reasoning of LLMs through Hate Speech Multi-hop Explanations</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Findings of the Association for Computational Linguistics: EMNLP 2025; *These authors contributed equally
    </div>
    <details class="paper-abstract">
      Ensuring the moral reasoning capabilities of Large Language Models (LLMs) is a growing concern as these systems are used in socially sensitive tasks. Nevertheless, current evaluation benchmarks present two major shortcomings: a lack of annotations that justify moral classifications, which limits transparency and interpretability; and a predominant focus on English, which constrains the assessment of moral reasoning across diverse cultural settings. In this paper, we introduce MFTCXplain, a multilingual benchmark dataset for evaluating the moral reasoning of LLMs via hate speech multi-hop explanation using Moral Foundation Theory (MFT). The dataset comprises 3,000 tweets across Portuguese, Italian, Persian, and English, annotated with binary hate speech labels, moral categories, and text span-level rationales. Empirical results highlight a misalignment between LLM outputs and human annotations in moral reasoning tasks. While LLMs perform well in hate speech detection (F1 up to 0.836), their ability to predict moral sentiments is notably weak (F1 < 0.35). Furthermore, rationale alignment remains limited mainly in underrepresented languages. These findings show the limited capacity of current LLMs to internalize and reflect human moral reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15110v1">LLMs and Agentic AI in Insurance Decision-Making: Opportunities and Challenges For Africa</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      In this work, we highlight the transformative potential of Artificial Intelligence (AI), particularly Large Language Models (LLMs) and agentic AI, in the insurance sector. We consider and emphasize the unique opportunities, challenges, and potential pathways in insurance amid rapid performance improvements, increased open-source access, decreasing deployment costs, and the complexity of LLM or agentic AI frameworks. To bring it closer to home, we identify critical gaps in the African insurance market and highlight key local efforts, players, and partnership opportunities. Finally, we call upon actuaries, insurers, regulators, and tech leaders to a collaborative effort aimed at creating inclusive, sustainable, and equitable AI strategies and solutions: by and for Africans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02873v3">It's the Thought that Counts: Evaluating the Attempts of Frontier LLMs to Persuade on Harmful Topics</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Persuasion is a powerful capability of large language models (LLMs) that both enables beneficial applications (e.g. helping people quit smoking) and raises significant risks (e.g. large-scale, targeted political manipulation). Prior work has found models possess a significant and growing persuasive capability, measured by belief changes in simulated or real users. However, these benchmarks overlook a crucial risk factor: the propensity of a model to attempt to persuade in harmful contexts. Understanding whether a model will blindly ``follow orders'' to persuade on harmful topics (e.g. glorifying joining a terrorist group) is key to understanding the efficacy of safety guardrails. Moreover, understanding if and when a model will engage in persuasive behavior in pursuit of some goal is essential to understanding the risks from agentic AI systems. We propose the Attempt to Persuade Eval (APE) benchmark, that shifts the focus from persuasion success to persuasion attempts, operationalized as a model's willingness to generate content aimed at shaping beliefs or behavior. Our evaluation framework probes frontier LLMs using a multi-turn conversational setup between simulated persuader and persuadee agents. APE explores a diverse spectrum of topics including conspiracies, controversial issues, and non-controversially harmful content. We introduce an automated evaluator model to identify willingness to persuade and measure the frequency and context of persuasive attempts. We find that many open and closed-weight models are frequently willing to attempt persuasion on harmful topics and that jailbreaking can increase willingness to engage in such behavior. Our results highlight gaps in current safety guardrails and underscore the importance of evaluating willingness to persuade as a key dimension of LLM risk. APE is available at github.com/AlignmentResearch/AttemptPersuadeEval
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07441v2">SAND: Boosting LLM Agents with Self-Taught Action Deliberation</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are commonly tuned with supervised finetuning on ReAct-style expert trajectories or preference optimization over pairwise rollouts. Most of these methods focus on imitating specific expert behaviors or promoting chosen reasoning thoughts and actions over rejected ones. However, without reasoning and comparing over alternatives actions, LLM agents finetuned with these methods may over-commit towards seemingly plausible but suboptimal actions due to limited action space exploration. To address this, in this paper we propose Self-taught ActioN Deliberation (SAND) framework, enabling LLM agents to explicitly deliberate over candidate actions before committing to one. To tackle the challenges of when and what to deliberate given large action space and step-level action evaluation, we incorporate self-consistency action sampling and execution-guided action critique to help synthesize step-wise action deliberation thoughts using the base model of the LLM agent. In an iterative manner, the deliberation trajectories are then used to finetune the LLM agent itself. Evaluating on two representative interactive agent tasks, SAND achieves an average 20% improvement over initial supervised finetuning and also outperforms state-of-the-art agent tuning approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15860v3">Synthetic vs. Gold: The Role of LLM Generated Labels and Data in Cyberbullying Detection</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Cyberbullying (CB) presents a pressing threat, especially to children, underscoring the urgent need for robust detection systems to ensure online safety. While large-scale datasets on online abuse exist, there remains a significant gap in labeled data that specifically reflects the language and communication styles used by children. The acquisition of such data from vulnerable populations, such as children, is challenging due to ethical, legal and technical barriers. Moreover, the creation of these datasets relies heavily on human annotation, which not only strains resources but also raises significant concerns due to annotators exposure to harmful content. In this paper, we address these challenges by leveraging Large Language Models (LLMs) to generate synthetic data and labels. Our experiments demonstrate that synthetic data enables BERT-based CB classifiers to achieve performance close to that of those trained on fully authentic datasets (75.8% vs. 81.5% accuracy). Additionally, LLMs can effectively label authentic yet unlabeled data, allowing BERT classifiers to attain a comparable performance level (79.1% vs. 81.5% accuracy). These results highlight the potential of LLMs as a scalable, ethical, and cost-effective solution for generating data for CB detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05182v2">On the Comprehensibility of Multi-structured Financial Documents using LLMs and Pre-processing Tools</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 15 pages, 5 figures, 9 tables
    </div>
    <details class="paper-abstract">
      The proliferation of complex structured data in hybrid sources, such as PDF documents and web pages, presents unique challenges for current Large Language Models (LLMs) and Multi-modal Large Language Models (MLLMs) in providing accurate answers. Despite the recent advancements of MLLMs, they still often falter when interpreting intricately structured information, such as nested tables and multi-dimensional plots, leading to hallucinations and erroneous outputs. This paper explores the capabilities of LLMs and MLLMs in understanding and answering questions from complex data structures found in PDF documents by leveraging industrial and open-source tools as part of a pre-processing pipeline. Our findings indicate that GPT-4o, a popular MLLM, achieves an accuracy of 56% on multi-structured documents when fed documents directly, and that integrating pre-processing tools raises the accuracy of LLMs to 61.3% for GPT-4o and 76% for GPT-4, and with lower overall cost. The code is publicly available at https://github.com/OGCDS/FinancialQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15036v1">MoEcho: Exploiting Side-Channel Attacks to Compromise User Privacy in Mixture-of-Experts LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 This paper will appear in CCS 2025
    </div>
    <details class="paper-abstract">
      The transformer architecture has become a cornerstone of modern AI, fueling remarkable progress across applications in natural language processing, computer vision, and multimodal learning. As these models continue to scale explosively for performance, implementation efficiency remains a critical challenge. Mixture of Experts (MoE) architectures, selectively activating specialized subnetworks (experts), offer a unique balance between model accuracy and computational cost. However, the adaptive routing in MoE architectures, where input tokens are dynamically directed to specialized experts based on their semantic meaning inadvertently opens up a new attack surface for privacy breaches. These input-dependent activation patterns leave distinctive temporal and spatial traces in hardware execution, which adversaries could exploit to deduce sensitive user data. In this work, we propose MoEcho, discovering a side channel analysis based attack surface that compromises user privacy on MoE based systems. Specifically, in MoEcho, we introduce four novel architectural side channels on different computing platforms, including Cache Occupancy Channels and Pageout+Reload on CPUs, and Performance Counter and TLB Evict+Reload on GPUs, respectively. Exploiting these vulnerabilities, we propose four attacks that effectively breach user privacy in large language models (LLMs) and vision language models (VLMs) based on MoE architectures: Prompt Inference Attack, Response Reconstruction Attack, Visual Inference Attack, and Visual Reconstruction Attack. MoEcho is the first runtime architecture level security analysis of the popular MoE structure common in modern transformers, highlighting a serious security and privacy threat and calling for effective and timely safeguards when harnessing MoE based models for developing efficient large scale AI services.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15030v1">Collab-REC: An LLM-based Agentic Framework for Balancing Recommendations in Tourism</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      We propose Collab-REC, a multi-agent framework designed to counteract popularity bias and enhance diversity in tourism recommendations. In our setting, three LLM-based agents -- Personalization, Popularity, and Sustainability generate city suggestions from complementary perspectives. A non-LLM moderator then merges and refines these proposals via multi-round negotiation, ensuring each agent's viewpoint is incorporated while penalizing spurious or repeated responses. Experiments on European city queries show that Collab-REC improves diversity and overall relevance compared to a single-agent baseline, surfacing lesser-visited locales that often remain overlooked. This balanced, context-aware approach addresses over-tourism and better aligns with constraints provided by the user, highlighting the promise of multi-stakeholder collaboration in LLM-driven recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15928v3">Exploring Big Five Personality and AI Capability Effects in LLM-Simulated Negotiation Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Presented at the KDD 2025 Workshop on Evaluation and Trustworthiness of Agentic and Generative AI Models under the title "Evaluating the LLM-simulated Impacts of Big Five Personality Traits and AI Capabilities on Social Negotiations" (https://kdd-eval-workshop.github.io/genai-evaluation-kdd2025/assets/papers/Submission%2036.pdf)
    </div>
    <details class="paper-abstract">
      This paper presents an evaluation framework for agentic AI systems in mission-critical negotiation contexts, addressing the need for AI agents that can adapt to diverse human operators and stakeholders. Using Sotopia as a simulation testbed, we present two experiments that systematically evaluated how personality traits and AI agent characteristics influence LLM-simulated social negotiation outcomes--a capability essential for a variety of applications involving cross-team coordination and civil-military interactions. Experiment 1 employs causal discovery methods to measure how personality traits impact price bargaining negotiations, through which we found that Agreeableness and Extraversion significantly affect believability, goal achievement, and knowledge acquisition outcomes. Sociocognitive lexical measures extracted from team communications detected fine-grained differences in agents' empathic communication, moral foundations, and opinion patterns, providing actionable insights for agentic AI systems that must operate reliably in high-stakes operational scenarios. Experiment 2 evaluates human-AI job negotiations by manipulating both simulated human personality and AI system characteristics, specifically transparency, competence, adaptability, demonstrating how AI agent trustworthiness impact mission effectiveness. These findings establish a repeatable evaluation methodology for experimenting with AI agent reliability across diverse operator personalities and human-agent team dynamics, directly supporting operational requirements for reliable AI systems. Our work advances the evaluation of agentic AI workflows by moving beyond standard performance metrics to incorporate social dynamics essential for mission success in complex operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14896v1">Quantization Meets dLLMs: A Systematic Study of Post-training Quantization for Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Technical Report, Work in Progress
    </div>
    <details class="paper-abstract">
      Recent advances in diffusion large language models (dLLMs) have introduced a promising alternative to autoregressive (AR) LLMs for natural language generation tasks, leveraging full attention and denoising-based decoding strategies. However, the deployment of these models on edge devices remains challenging due to their massive parameter scale and high resource demands. While post-training quantization (PTQ) has emerged as a widely adopted technique for compressing AR LLMs, its applicability to dLLMs remains largely unexplored. In this work, we present the first systematic study on quantizing diffusion-based language models. We begin by identifying the presence of activation outliers, characterized by abnormally large activation values that dominate the dynamic range. These outliers pose a key challenge to low-bit quantization, as they make it difficult to preserve precision for the majority of values. More importantly, we implement state-of-the-art PTQ methods and conduct a comprehensive evaluation across multiple task types and model variants. Our analysis is structured along four key dimensions: bit-width, quantization method, task category, and model type. Through this multi-perspective evaluation, we offer practical insights into the quantization behavior of dLLMs under different configurations. We hope our findings provide a foundation for future research in efficient dLLM deployment. All codes and experimental setups will be released to support the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14879v1">MeshCoder: LLM-Powered Structured Mesh Code Generation from Point Clouds</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Reconstructing 3D objects into editable programs is pivotal for applications like reverse engineering and shape editing. However, existing methods often rely on limited domain-specific languages (DSLs) and small-scale datasets, restricting their ability to model complex geometries and structures. To address these challenges, we introduce MeshCoder, a novel framework that reconstructs complex 3D objects from point clouds into editable Blender Python scripts. We develop a comprehensive set of expressive Blender Python APIs capable of synthesizing intricate geometries. Leveraging these APIs, we construct a large-scale paired object-code dataset, where the code for each object is decomposed into distinct semantic parts. Subsequently, we train a multimodal large language model (LLM) that translates 3D point cloud into executable Blender Python scripts. Our approach not only achieves superior performance in shape-to-code reconstruction tasks but also facilitates intuitive geometric and topological editing through convenient code modifications. Furthermore, our code-based representation enhances the reasoning capabilities of LLMs in 3D shape understanding tasks. Together, these contributions establish MeshCoder as a powerful and flexible solution for programmatic 3D shape reconstruction and understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02085v4">SE-Agent: Self-Evolution Trajectory Optimization in Multi-Step Reasoning with LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents have recently shown impressive capabilities in complex reasoning and tool use via multi-step interactions with their environments. While these agents have the potential to tackle complicated tasks, their problem-solving process, i.e., agents' interaction trajectory leading to task completion, remains underexploited. These trajectories contain rich feedback that can navigate agents toward the right directions for solving problems correctly. Although prevailing approaches, such as Monte Carlo Tree Search (MCTS), can effectively balance exploration and exploitation, they ignore the interdependence among various trajectories and lack the diversity of search spaces, which leads to redundant reasoning and suboptimal outcomes. To address these challenges, we propose SE-Agent, a Self-Evolution framework that enables Agents to optimize their reasoning processes iteratively. Our approach revisits and enhances former pilot trajectories through three key operations: revision, recombination, and refinement. This evolutionary mechanism enables two critical advantages: (1) it expands the search space beyond local optima by intelligently exploring diverse solution paths guided by previous trajectories, and (2) it leverages cross-trajectory inspiration to efficiently enhance performance while mitigating the impact of suboptimal reasoning paths. Through these mechanisms, SE-Agent achieves continuous self-evolution that incrementally improves reasoning quality. We evaluate SE-Agent on SWE-bench Verified to resolve real-world GitHub issues. Experimental results across five strong LLMs show that integrating SE-Agent delivers up to 55% relative improvement, achieving state-of-the-art performance among all open-source agents on SWE-bench Verified. Our code and demonstration materials are publicly available at https://github.com/JARVIS-Xs/SE-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14765v1">PepThink-R1: LLM for Interpretable Cyclic Peptide Optimization with CoT SFT and Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Designing therapeutic peptides with tailored properties is hindered by the vastness of sequence space, limited experimental data, and poor interpretability of current generative models. To address these challenges, we introduce PepThink-R1, a generative framework that integrates large language models (LLMs) with chain-of-thought (CoT) supervised fine-tuning and reinforcement learning (RL). Unlike prior approaches, PepThink-R1 explicitly reasons about monomer-level modifications during sequence generation, enabling interpretable design choices while optimizing for multiple pharmacological properties. Guided by a tailored reward function balancing chemical validity and property improvements, the model autonomously explores diverse sequence variants. We demonstrate that PepThink-R1 generates cyclic peptides with significantly enhanced lipophilicity, stability, and exposure, outperforming existing general LLMs (e.g., GPT-5) and domain-specific baseline in both optimization success and interpretability. To our knowledge, this is the first LLM-based peptide design framework that combines explicit reasoning with RL-driven property control, marking a step toward reliable and transparent peptide optimization for therapeutic discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14751v1">HERAKLES: Hierarchical Skill Compilation for Open-ended LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 42 pages
    </div>
    <details class="paper-abstract">
      Open-ended AI agents need to be able to learn efficiently goals of increasing complexity, abstraction and heterogeneity over their lifetime. Beyond sampling efficiently their own goals, autotelic agents specifically need to be able to keep the growing complexity of goals under control, limiting the associated growth in sample and computational complexity. To adress this challenge, recent approaches have leveraged hierarchical reinforcement learning (HRL) and language, capitalizing on its compositional and combinatorial generalization capabilities to acquire temporally extended reusable behaviours. Existing approaches use expert defined spaces of subgoals over which they instantiate a hierarchy, and often assume pre-trained associated low-level policies. Such designs are inadequate in open-ended scenarios, where goal spaces naturally diversify across a broad spectrum of difficulties. We introduce HERAKLES, a framework that enables a two-level hierarchical autotelic agent to continuously compile mastered goals into the low-level policy, executed by a small, fast neural network, dynamically expanding the set of subgoals available to the high-level policy. We train a Large Language Model (LLM) to serve as the high-level controller, exploiting its strengths in goal decomposition and generalization to operate effectively over this evolving subgoal space. We evaluate HERAKLES in the open-ended Crafter environment and show that it scales effectively with goal complexity, improves sample efficiency through skill compilation, and enables the agent to adapt robustly to novel challenges over time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14735v1">Evaluating Multilingual and Code-Switched Alignment in LLMs via Synthetic Natural Language Inference</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in multilingual contexts, yet their capacity for consistent, logically grounded alignment across languages remains underexplored. We present a controlled evaluation framework for multilingual natural language inference (NLI) that generates synthetic, logic-based premise-hypothesis pairs and translates them into a typologically diverse set of languages. This design enables precise control over semantic relations and allows testing in both monolingual and mixed-language (code-switched) conditions. Surprisingly, code-switching does not degrade, and can even improve, performance, suggesting that translation-induced lexical variation may serve as a regularization signal. We validate semantic preservation through embedding-based similarity analyses and cross-lingual alignment visualizations, confirming the fidelity of translated pairs. Our findings expose both the potential and the brittleness of current LLM cross-lingual reasoning, and identify code-switching as a promising lever for improving multilingual robustness. Code available at: https://github.com/KurbanIntelligenceLab/nli-stress-testing
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19254v3">Uncertainty Quantification for Language Models: A Suite of Black-Box, White-Box, LLM Judge, and Ensemble Scorers</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 UQLM repository: https://github.com/cvs-health/uqlm
    </div>
    <details class="paper-abstract">
      Hallucinations are a persistent problem with Large Language Models (LLMs). As these models become increasingly used in high-stakes domains, such as healthcare and finance, the need for effective hallucination detection is crucial. To this end, we outline a versatile framework for zero-resource hallucination detection that practitioners can apply to real-world use cases. To achieve this, we adapt a variety of existing uncertainty quantification (UQ) techniques, including black-box UQ, white-box UQ, and LLM-as-a-Judge, transforming them as necessary into standardized response-level confidence scores ranging from 0 to 1. To enhance flexibility, we propose a tunable ensemble approach that incorporates any combination of the individual confidence scores. This approach enables practitioners to optimize the ensemble for a specific use case for improved performance. To streamline implementation, the full suite of scorers is offered in this paper's companion Python toolkit, UQLM. To evaluate the performance of the various scorers, we conduct an extensive set of experiments using several LLM question-answering benchmarks. We find that our tunable ensemble typically surpasses its individual components and outperforms existing hallucination detection methods. Our results demonstrate the benefits of customized hallucination detection strategies for improving the accuracy and reliability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14951v1">Improving LLMs for Machine Translation Using Synthetic Preference Data</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Paper with individual presentation at LUHME workshop at ECAI 2025
    </div>
    <details class="paper-abstract">
      Large language models have emerged as effective machine translation systems. In this paper, we explore how a general instruction-tuned large language model can be improved for machine translation using relatively few easily produced data resources. Using Slovene as a use case, we improve the GaMS-9B-Instruct model using Direct Preference Optimization (DPO) training on a programmatically curated and enhanced subset of a public dataset. As DPO requires pairs of quality-ranked instances, we generated its training dataset by translating English Wikipedia articles using two LLMs, GaMS-9B-Instruct and EuroLLM-9B-Instruct. We ranked the resulting translations based on heuristics coupled with automatic evaluation metrics such as COMET. The evaluation shows that our fine-tuned model outperforms both models involved in the dataset generation. In comparison to the baseline models, the fine-tuned model achieved a COMET score gain of around 0.04 and 0.02, respectively, on translating Wikipedia articles. It also more consistently avoids language and formatting errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14706v1">ShizhenGPT: Towards Multimodal LLMs for Traditional Chinese Medicine</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Despite the success of large language models (LLMs) in various domains, their potential in Traditional Chinese Medicine (TCM) remains largely underexplored due to two critical barriers: (1) the scarcity of high-quality TCM data and (2) the inherently multimodal nature of TCM diagnostics, which involve looking, listening, smelling, and pulse-taking. These sensory-rich modalities are beyond the scope of conventional LLMs. To address these challenges, we present ShizhenGPT, the first multimodal LLM tailored for TCM. To overcome data scarcity, we curate the largest TCM dataset to date, comprising 100GB+ of text and 200GB+ of multimodal data, including 1.2M images, 200 hours of audio, and physiological signals. ShizhenGPT is pretrained and instruction-tuned to achieve deep TCM knowledge and multimodal reasoning. For evaluation, we collect recent national TCM qualification exams and build a visual benchmark for Medicinal Recognition and Visual Diagnosis. Experiments demonstrate that ShizhenGPT outperforms comparable-scale LLMs and competes with larger proprietary models. Moreover, it leads in TCM visual understanding among existing multimodal LLMs and demonstrates unified perception across modalities like sound, pulse, smell, and vision, paving the way toward holistic multimodal perception and diagnosis in TCM. Datasets, models, and code are publicly available. We hope this work will inspire further exploration in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14654v1">Entropy-Constrained Strategy Optimization in Urban Floods: A Multi-Agent Framework with LLM and Knowledge Graph Integration</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 17 pages including appendix, 6 figures
    </div>
    <details class="paper-abstract">
      In recent years, the increasing frequency of extreme urban rainfall events has posed significant challenges to emergency scheduling systems. Urban flooding often leads to severe traffic congestion and service disruptions, threatening public safety and mobility. However, effective decision making remains hindered by three key challenges: (1) managing trade-offs among competing goals (e.g., traffic flow, task completion, and risk mitigation) requires dynamic, context-aware strategies; (2) rapidly evolving environmental conditions render static rules inadequate; and (3) LLM-generated strategies frequently suffer from semantic instability and execution inconsistency. Existing methods fail to align perception, global optimization, and multi-agent coordination within a unified framework. To tackle these challenges, we introduce H-J, a hierarchical multi-agent framework that integrates knowledge-guided prompting, entropy-constrained generation, and feedback-driven optimization. The framework establishes a closed-loop pipeline spanning from multi-source perception to strategic execution and continuous refinement. We evaluate H-J on real-world urban topology and rainfall data under three representative conditions: extreme rainfall, intermittent bursts, and daily light rain. Experiments show that H-J outperforms rule-based and reinforcement-learning baselines in traffic smoothness, task success rate, and system robustness. These findings highlight the promise of uncertainty-aware, knowledge-constrained LLM-based approaches for enhancing resilience in urban flood response.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14635v1">Can LLM Agents Solve Collaborative Tasks? A Study on Urgency-Aware Planning and Coordination</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      The ability to coordinate actions across multiple agents is critical for solving complex, real-world problems. Large Language Models (LLMs) have shown strong capabilities in communication, planning, and reasoning, raising the question of whether they can also support effective collaboration in multi-agent settings. In this work, we investigate the use of LLM agents to solve a structured victim rescue task that requires division of labor, prioritization, and cooperative planning. Agents operate in a fully known graph-based environment and must allocate resources to victims with varying needs and urgency levels. We systematically evaluate their performance using a suite of coordination-sensitive metrics, including task success rate, redundant actions, room conflicts, and urgency-weighted efficiency. This study offers new insights into the strengths and failure modes of LLMs in physically grounded multi-agent collaboration tasks, contributing to future benchmarks and architectural improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03082v2">EoH-S: Evolution of Heuristic Set using LLMs for Automated Heuristic Design</a></div>
    <div class="paper-meta">
      📅 2025-08-20
    </div>
    <details class="paper-abstract">
      Automated Heuristic Design (AHD) using Large Language Models (LLMs) has achieved notable success in recent years. Despite the effectiveness of existing approaches, they only design a single heuristic to serve all problem instances, often inducing poor generalization across different distributions or settings. To address this issue, we propose Automated Heuristic Set Design (AHSD), a new formulation for LLM-driven AHD. The aim of AHSD is to automatically generate a small-sized complementary heuristic set to serve diverse problem instances, such that each problem instance could be optimized by at least one heuristic in this set. We show that the objective function of AHSD is monotone and supermodular. Then, we propose Evolution of Heuristic Set (EoH-S) to apply the AHSD formulation for LLM-driven AHD. With two novel mechanisms of complementary population management and complementary-aware memetic search, EoH-S could effectively generate a set of high-quality and complementary heuristics. Comprehensive experimental results on three AHD tasks with diverse instances spanning various sizes and distributions demonstrate that EoH-S consistently outperforms existing state-of-the-art AHD methods and achieves up to 60\% performance improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12096v2">STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transition Samples</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Submit to AAAI 2026
    </div>
    <details class="paper-abstract">
      Evaluating large language models (LLMs) has become increasingly challenging as model capabilities advance rapidly. While recent models often achieve higher scores on standard benchmarks, these improvements do not consistently reflect enhanced real-world reasoning capabilities. Moreover, widespread overfitting to public benchmarks and the high computational cost of full evaluations have made it both expensive and less effective to distinguish meaningful differences between models. To address these challenges, we propose the \textbf{S}tructured \textbf{T}ransition \textbf{E}valuation \textbf{M}ethod (STEM), a lightweight and interpretable evaluation framework for efficiently estimating the relative capabilities of LLMs. STEM identifies \textit{significant transition samples} (STS) by analyzing consistent performance transitions among LLMs of the same architecture but varying parameter scales. These samples enable STEM to effectively estimate the capability position of an unknown model. Qwen3 model family is applied to construct the STS pool on six diverse and representative benchmarks. To assess generalizability. Experimental results indicate that STEM reliably captures performance trends, aligns with ground-truth rankings of model capability. These findings highlight STEM as a practical and scalable method for fine-grained, architecture-agnostic evaluation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14564v1">Who Sees What? Structured Thought-Action Sequences for Epistemic Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Accepted at ICSR25
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) and reasoning frameworks have opened new possibilities for improving the perspective -taking capabilities of autonomous agents. However, tasks that involve active perception, collaborative reasoning, and perspective taking (understanding what another agent can see or knows) pose persistent challenges for current LLM-based systems. This study investigates the potential of structured examples derived from transformed solution graphs generated by the Fast Downward planner to improve the performance of LLM-based agents within a ReAct framework. We propose a structured solution-processing pipeline that generates three distinct categories of examples: optimal goal paths (G-type), informative node paths (E-type), and step-by-step optimal decision sequences contrasting alternative actions (L-type). These solutions are further converted into ``thought-action'' examples by prompting an LLM to explicitly articulate the reasoning behind each decision. While L-type examples slightly reduce clarification requests and overall action steps, they do not yield consistent improvements. Agents are successful in tasks requiring basic attentional filtering but struggle in scenarios that required mentalising about occluded spaces or weighing the costs of epistemic actions. These findings suggest that structured examples alone are insufficient for robust perspective-taking, underscoring the need for explicit belief tracking, cost modelling, and richer environments to enable socially grounded collaboration in LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14553v1">Towards LLM-generated explanations for Component-based Knowledge Graph Question Answering Systems</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 Presented at ICWI 2024, Zagreb. Released with ISBN: 978-989-8704-62-7. Data source: https://figshare.com/articles/dataset/Towards_LLM-generated_explanations_for_component-based_knowledge_graph_question_answering_systems/27079687
    </div>
    <details class="paper-abstract">
      Over time, software systems have reached a level of complexity that makes it difficult for their developers and users to explain particular decisions made by them. In this paper, we focus on the explainability of component-based systems for Question Answering (QA). These components often conduct processes driven by AI methods, in which behavior and decisions cannot be clearly explained or justified, s.t., even for QA experts interpreting the executed process and its results is hard. To address this challenge, we present an approach that considers the components' input and output data flows as a source for representing the behavior and provide explanations for the components, enabling users to comprehend what happened. In the QA framework used here, the data flows of the components are represented as SPARQL queries (inputs) and RDF triples (outputs). Hence, we are also providing valuable insights on verbalization regarding these data types. In our experiments, the approach generates explanations while following template-based settings (baseline) or via the use of Large Language Models (LLMs) with different configurations (automatic generation). Our evaluation shows that the explanations generated via LLMs achieve high quality and mostly outperform template-based approaches according to the users' ratings. Therefore, it enables us to automatically explain the behavior and decisions of QA components to humans while using RDF and SPARQL as a context for explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03106v5">Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback</a></div>
    <div class="paper-meta">
      📅 2025-08-20
      | 💬 52 pages, updated with new experimental results and implementation details
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language models (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of spontaneous self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided self-refinements simultaneously while maintaining exploration. Additionally, we employ a shaping function to amplify learning from correct, especially unfamiliar, refinements and penalize incorrect ones. Extensive experiments with Qwen2.5-7B-Base, Qwen2.5-Math-7B-Base, and Qwen3-8B demonstrate that Critique-GRPO consistently outperforms supervised learning and RL-based fine-tuning methods across eight challenging mathematical, STEM, and general reasoning tasks. Specifically, Critique-GRPO improves average pass@1 scores across all compared methods by approximately +4.4% on Qwen2.5-7B-Base and +3.8% on Qwen3-8B. Notably, Critique-GRPO enables effective self-improvement through self-critiquing, achieving significant gains over GRPO, e.g., +16.7% pass@1 improvement on AIME 2024.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13423v1">AdaptJobRec: Enhancing Conversational Career Recommendation through an LLM-Powered Agentic System</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      In recent years, recommendation systems have evolved from providing a single list of recommendations to offering a comprehensive suite of topic focused services. To better accomplish this task, conversational recommendation systems (CRS) have progressed from basic retrieval augmented LLM generation to agentic systems with advanced reasoning and self correction capabilities. However, agentic systems come with notable response latency, a longstanding challenge for conversational recommendation systems. To balance the trade off between handling complex queries and minimizing latency, we propose AdaptJobRec, the first conversational job recommendation system that leverages autonomous agent to integrate personalized recommendation algorithm tools. The system employs a user query complexity identification mechanism to minimize response latency. For straightforward queries, the agent directly selects the appropriate tool for rapid responses. For complex queries, the agent uses the memory processing module to filter chat history for relevant content, then passes the results to the intelligent task decomposition planner, and finally executes the tasks using personalized recommendation tools. Evaluation on Walmart's real world career recommendation scenarios demonstrates that AdaptJobRec reduces average response latency by up to 53.3% compared to competitive baselines, while significantly improving recommendation accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14314v1">Zero-knowledge LLM hallucination detection and mitigation through fine-grained cross-model consistency</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, but they remain susceptible to hallucinations--generating content that appears plausible but contains factual inaccuracies. We present Finch-Zk, a black-box framework that leverages FINe-grained Cross-model consistency to detect and mitigate Hallucinations in LLM outputs without requiring external knowledge sources. Finch-Zk introduces two key innovations: 1) a cross-model consistency checking strategy that reveals fine-grained inaccuracies by comparing responses generated by diverse models from semantically-equivalent prompts, and 2) a targeted mitigation technique that applies precise corrections to problematic segments while preserving accurate content. Experiments on the FELM dataset show Finch-Zk improves hallucination detection F1 scores by 6-39\% compared to existing approaches. For mitigation, Finch-Zk achieves 7-8 absolute percentage points improvement in answer accuracy on the GPQA-diamond dataset when applied to state-of-the-art models like Llama 4 Maverick and Claude 4 Sonnet. Extensive evaluation across multiple models demonstrates that Finch-Zk provides a practical, deployment-ready safeguard for enhancing factual reliability in production LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14302v1">GLASS: Test-Time Acceleration for LLMs via Global-Local Neural Importance Aggregation</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Deploying Large Language Models (LLMs) on edge hardware demands aggressive, prompt-aware dynamic pruning to reduce computation without degrading quality. Static or predictor-based schemes either lock in a single sparsity pattern or incur extra runtime overhead, and recent zero-shot methods that rely on statistics from a single prompt fail on short prompt and/or long generation scenarios. We introduce A/I-GLASS: Activation- and Impact-based Global-Local neural importance Aggregation for feed-forward network SparSification, two training-free methods that dynamically select FFN units using a rank-aggregation of prompt local and model-intrinsic global neuron statistics. Empirical results across multiple LLMs and benchmarks demonstrate that GLASS significantly outperforms prior training-free methods, particularly in challenging long-form generation scenarios, without relying on auxiliary predictors or adding any inference overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14288v1">Measuring LLM Code Generation Stability via Structural Entropy</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 ASE-NIER
    </div>
    <details class="paper-abstract">
      Assessing the stability of code generation from large language models (LLMs) is essential for judging their reliability in real-world development. We extend prior "structural-entropy concepts" to the program domain by pairing entropy with abstract syntax tree (AST) analysis. For any fixed prompt, we collect the multiset of depth-bounded subtrees of AST in each generated program and treat their relative frequencies as a probability distribution. We then measure stability in two complementary ways: (i) Jensen-Shannon divergence, a symmetric, bounded indicator of structural overlap, and (ii) a Structural Cross-Entropy ratio that highlights missing high-probability patterns. Both metrics admit structural-only and token-aware variants, enabling separate views on control-flow shape and identifier-level variability. Unlike pass@k, BLEU, or CodeBLEU, our metrics are reference-free, language-agnostic, and execution-independent. We benchmark several leading LLMs on standard code generation tasks, demonstrating that AST-driven structural entropy reveals nuances in model consistency and robustness. The method runs in O(n,d) time with no external tests, providing a lightweight addition to the code-generation evaluation toolkit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10142v2">Multi-Turn Puzzles: Evaluating Interactive Reasoning and Strategic Dialogue in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at solving problems with clear and complete statements, but often struggle with nuanced environments or interactive tasks which are common in most real-world scenarios. This highlights the critical need for developing LLMs that can effectively engage in logically consistent multi-turn dialogue, seek information and reason with incomplete data. To this end, we introduce a novel benchmark comprising a suite of multi-turn tasks each designed to test specific reasoning, interactive dialogue, and information-seeking abilities. These tasks have deterministic scoring mechanisms, thus eliminating the need for human intervention. Evaluating frontier models on our benchmark reveals significant headroom. Our analysis shows that most errors emerge from poor instruction following, reasoning failures, and poor planning. This benchmark provides valuable insights into the strengths and weaknesses of current LLMs in handling complex, interactive scenarios and offers a robust platform for future research aimed at improving these critical capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14279v1">GRILE: A Benchmark for Grammar Reasoning and Explanation in Romanian LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-19
      | 💬 Accepted as long paper @RANLP2025
    </div>
    <details class="paper-abstract">
      LLMs (Large language models) have revolutionized NLP (Natural Language Processing), yet their pedagogical value for low-resource languages remains unclear. We present GRILE (Grammar Romanian Inference and Language Explanations) , the first open benchmark of 1,151 multiple-choice questions harvested from Romanian high-stakes exams (National Evaluation, Baccalaureate, university admissions). GRILE enables us to probe two complementary abilities of seven state-of-the-art multilingual and Romanian-specific LLMs: (i) selecting the correct answer, and (ii) producing linguistically accurate explanations. While Gemini 2.5 Pro reaches 83% accuracy, most open-weight models stay below 65%, and 48% of their explanations contain factual or pedagogical flaws according to expert review. A detailed error analysis pinpoints systematic weaknesses in morphology and in applying the latest DOOM3 orthographic norms. All data, code and a public web demo are released to catalyze future research. Our findings expose open challenges for trustworthy educational NLP in low-resource settings and establish GRILE as a new test-bed for controllable explanation generation and evaluation.
    </details>
</div>
