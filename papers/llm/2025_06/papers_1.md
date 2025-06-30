# llm - 2025_06

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
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22419v1">The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      Rapid advancements in large language models (LLMs) have the potential to assist in scientific progress. A critical capability toward this endeavor is the ability to reproduce existing work. To evaluate the ability of AI agents to reproduce results in an active research area, we introduce the Automated LLM Speedrunning Benchmark, leveraging the research community contributions on the NanoGPT speedrun, a competition to train a GPT-2 model in the shortest time. Each of the 19 speedrun tasks provides the agent with the previous records training script, optionally paired with one of three hint formats, ranging from pseudocode to paper-like descriptions of the new records improvements. Records execute quickly by design and speedrun improvements encompass diverse code-level changes, ranging from high-level algorithmic advancements to hardware-aware optimizations. These features make the benchmark both accessible and realistic for the frontier problem of improving LLM training. We find that recent reasoning LLMs combined with SoTA scaffolds struggle to reimplement already-known innovations in our benchmark, even when given detailed hints. Our benchmark thus provides a simple, non-saturated measure of an LLMs ability to automate scientific reproduction, a necessary (but not sufficient) skill for an autonomous research agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22396v1">QuickSilver -- Speeding up LLM Inference through Dynamic Token Halting, KV Skipping, Contextual Token Fusion, and Adaptive Matryoshka Quantization</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Preprint. Under submission
    </div>
    <details class="paper-abstract">
      Inference accounts for the majority of latency and energy consumption in large language model (LLM) deployments, often exceeding 90% of total cost. While training-time efficiency has seen extensive progress, runtime optimization remains a key bottleneck, particularly under autoregressive decoding. Existing approaches -- such as pruning, quantization, early exits, and speculative decoding -- often require retraining, architectural changes, or disrupt decoding compatibility. We introduce QuickSilver, a modular, token-level framework that enables semantic adaptivity at inference time without altering model weights or structure. QuickSilver integrates four synergistic mechanisms: (i) Dynamic Token Halting, which halts computation for tokens with converged representations; (ii) KV Cache Skipping, which selectively suppresses memory writes to reduce attention overhead; and (iii) Contextual Token Fusion, which collapses redundant tokens into shared paths to shrink sequence length. Unlike speculative decoding or MoE routing, QuickSilver operates entirely on frozen, dense models and requires no auxiliary networks. Applied to GPT-2 and Llama-2 across WikiText-103 and C4, QuickSilver achieves up to 39.6% FLOP reduction with negligible perplexity degradation (<=0.2).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22372v1">Towards Fair Rankings: Leveraging LLMs for Gender Bias Detection and Measurement</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Accepted by ACM SIGIR Conference on Innovative Concepts and Theories in Information Retrieval (ICTIR 2025)
    </div>
    <details class="paper-abstract">
      The presence of social biases in Natural Language Processing (NLP) and Information Retrieval (IR) systems is an ongoing challenge, which underlines the importance of developing robust approaches to identifying and evaluating such biases. In this paper, we aim to address this issue by leveraging Large Language Models (LLMs) to detect and measure gender bias in passage ranking. Existing gender fairness metrics rely on lexical- and frequency-based measures, leading to various limitations, e.g., missing subtle gender disparities. Building on our LLM-based gender bias detection method, we introduce a novel gender fairness metric, named Class-wise Weighted Exposure (CWEx), aiming to address existing limitations. To measure the effectiveness of our proposed metric and study LLMs' effectiveness in detecting gender bias, we annotate a subset of the MS MARCO Passage Ranking collection and release our new gender bias collection, called MSMGenderBias, to foster future research in this area. Our extensive experimental results on various ranking models show that our proposed metric offers a more detailed evaluation of fairness compared to previous metrics, with improved alignment to human labels (58.77% for Grep-BiasIR, and 18.51% for MSMGenderBias, measured using Cohen's Kappa agreement), effectively distinguishing gender bias in ranking. By integrating LLM-driven bias detection, an improved fairness metric, and gender bias annotations for an established dataset, this work provides a more robust framework for analyzing and mitigating bias in IR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22316v1">Evaluating Scoring Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      The remarkable performance of Large Language Models (LLMs) gives rise to``LLM-as-a-Judge'', where LLMs are employed as evaluators for complex tasks. Moreover, it has been widely adopted across fields such as Natural Language Processing (NLP), preference learning, and various specific domains. However, there are various biases within LLM-as-a-Judge, which adversely affect the fairness and reliability of judgments. Current research on evaluating or mitigating bias in LLM-as-a-Judge predominantly focuses on comparison-based evaluations, while systematic investigations into bias in scoring-based evaluations remain limited. Therefore, we define scoring bias in LLM-as-a-Judge as the scores differ when scoring judge models are bias-related perturbed, and provide a well-designed framework to comprehensively evaluate scoring bias. We augment existing LLM-as-a-Judge benchmarks through data synthesis to construct our evaluation dataset and design multi-faceted evaluation metrics. Our experimental results demonstrate that the scoring stability of existing judge models is disrupted by scoring biases. Further exploratory experiments and discussions provide valuable insights into the design of scoring prompt templates and the mitigation of scoring biases on aspects such as score rubrics, score IDs, and reference answer selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22232v1">Leveraging In-Context Learning for Political Bias Testing of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      A growing body of work has been querying LLMs with political questions to evaluate their potential biases. However, this probing method has limited stability, making comparisons between models unreliable. In this paper, we argue that LLMs need more context. We propose a new probing task, Questionnaire Modeling (QM), that uses human survey data as in-context examples. We show that QM improves the stability of question-based bias evaluation, and demonstrate that it may be used to compare instruction-tuned models to their base versions. Experiments with LLMs of various sizes indicate that instruction tuning can indeed change the direction of bias. Furthermore, we observe a trend that larger models are able to leverage in-context examples more effectively, and generally exhibit smaller bias scores in QM. Data and code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17944v2">SegChange-R1: LLM-Augmented Remote Sensing Change Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      Remote sensing change detection is used in urban planning, terrain analysis, and environmental monitoring by analyzing feature changes in the same area over time. In this paper, we propose a large language model (LLM) augmented inference approach (SegChange-R1), which enhances the detection capability by integrating textual descriptive information and guides the model to focus on relevant change regions, accelerating convergence. We designed a linear attention-based spatial transformation module (BEV) to address modal misalignment by unifying features from different times into a BEV space. Furthermore, we introduce DVCD, a novel dataset for building change detection from UAV viewpoints. Experiments on four widely-used datasets demonstrate significant improvements over existing method The code and pre-trained models are available in {https://github.com/Yu-Zhouz/SegChange-R1}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03313v2">LLM as GNN: Graph Vocabulary Learning for Text-Attributed Graph Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      Text-Attributed Graphs (TAGs), where each node is associated with text descriptions, are ubiquitous in real-world scenarios. They typically exhibit distinctive structure and domain-specific knowledge, motivating the development of a Graph Foundation Model (GFM) that generalizes across diverse graphs and tasks. Despite large efforts to integrate Large Language Models (LLMs) and Graph Neural Networks (GNNs) for TAGs, existing approaches suffer from decoupled architectures with two-stage alignment, limiting their synergistic potential. Even worse, existing methods assign out-of-vocabulary (OOV) tokens to graph nodes, leading to graph-specific semantics, token explosion, and incompatibility with task-oriented prompt templates, which hinders cross-graph and cross-task transferability. To address these challenges, we propose PromptGFM, a versatile GFM for TAGs grounded in graph vocabulary learning. PromptGFM comprises two key components: (1) Graph Understanding Module, which explicitly prompts LLMs to replicate the finest GNN workflow within the text space, facilitating seamless GNN-LLM integration and elegant graph-text alignment; (2) Graph Inference Module, which establishes a language-based graph vocabulary ensuring expressiveness, transferability, and scalability, enabling readable instructions for LLM fine-tuning. Extensive experiments demonstrate our superiority and transferability across diverse graphs and tasks. The code is available at this: https://github.com/agiresearch/PromptGFM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04412v3">Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Presented at ICLR 2025 Workshop on Foundation Models in the Wild
    </div>
    <details class="paper-abstract">
      Recent advances demonstrate that increasing inference-time computation can significantly boost the reasoning capabilities of large language models (LLMs). Although repeated sampling (i.e., generating multiple candidate outputs) is a highly effective strategy, it does not leverage external feedback signals for refinement, which are often available in tasks like coding. In this work, we propose Adaptive Branching Monte Carlo Tree Search (AB-MCTS), a novel inference-time framework that generalizes repeated sampling with principled multi-turn exploration and exploitation. At each node in the search tree, AB-MCTS dynamically decides whether to "go wider" by expanding new candidate responses or "go deeper" by revisiting existing ones based on external feedback signals. We evaluate our method on complex coding and engineering tasks using frontier models. Empirical results show that AB-MCTS consistently outperforms both repeated sampling and standard MCTS, underscoring the importance of combining the response diversity of LLMs with multi-turn solution refinement for effective inference-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24616v3">Eye of Judgement: Dissecting the Evaluation of Russian-speaking LLMs with POLLUX</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 178 pages
    </div>
    <details class="paper-abstract">
      We introduce POLLUX, a comprehensive open-source benchmark designed to evaluate the generative capabilities of large language models (LLMs) in Russian. Our main contribution is a novel evaluation methodology that enhances the interpretability of LLM assessment. For each task type, we define a set of detailed criteria and develop a scoring protocol where models evaluate responses and provide justifications for their ratings. This enables transparent, criteria-driven evaluation beyond traditional resource-consuming, side-by-side human comparisons. POLLUX includes a detailed, fine-grained taxonomy of 35 task types covering diverse generative domains such as code generation, creative writing, and practical assistant use cases, totaling 2,100 manually crafted and professionally authored prompts. Each task is categorized by difficulty (easy/medium/hard), with experts constructing the dataset entirely from scratch. We also release a family of LLM-as-a-Judge (7B and 32B) evaluators trained for nuanced assessment of generative outputs. This approach provides scalable, interpretable evaluation and annotation tools for model development, effectively replacing costly and less precise human judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22139v1">Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Accepted at ICCV 2025
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have demonstrated significant success in visual understanding tasks. However, challenges persist in adapting these models for video comprehension due to the large volume of data and temporal complexity. Existing Video-LLMs using uniform frame sampling often struggle to capture the query-related crucial spatiotemporal clues of videos effectively. In this paper, we introduce Q-Frame, a novel approach for adaptive frame selection and multi-resolution scaling tailored to the video's content and the specific query. Q-Frame employs a training-free, plug-and-play strategy generated by a text-image matching network like CLIP, utilizing the Gumbel-Max trick for efficient frame selection. Q-Frame allows Video-LLMs to process more frames without exceeding computational limits, thereby preserving critical temporal and spatial information. We demonstrate Q-Frame's effectiveness through extensive experiments on benchmark datasets, including MLVU, LongVideoBench, and Video-MME, illustrating its superiority over existing methods and its applicability across various video understanding tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08837v3">Design Patterns for Securing LLM Agents against Prompt Injections</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      As AI agents powered by Large Language Models (LLMs) become increasingly versatile and capable of addressing a broad spectrum of tasks, ensuring their security has become a critical challenge. Among the most pressing threats are prompt injection attacks, which exploit the agent's resilience on natural language inputs -- an especially dangerous threat when agents are granted tool access or handle sensitive information. In this work, we propose a set of principled design patterns for building AI agents with provable resistance to prompt injection. We systematically analyze these patterns, discuss their trade-offs in terms of utility and security, and illustrate their real-world applicability through a series of case studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03592v3">English K_Quantization of LLMs Does Not Disproportionately Diminish Multilingual Performance</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 8 pages, 6 figures, v2
    </div>
    <details class="paper-abstract">
      For consumer usage of locally deployed LLMs, the GGUF format and k\_quantization are invaluable tools for maintaining the performance of the original model while reducing it to sizes deployable with consumer-grade hardware. The number of bits dedicated to each weight from the original model is reduced based on how important they are thought to be during model inference. This importance is arrived at through the application of an 'importance matrix'-a relatively small text document meant to be representative of the LLM's standard use-cases. In the vast majority of quants available online, this document is primarily written in English. It was therefore an open question whether performance on English language tasks was preserved through the sacrifice of multilingual performance and whether it can be preserved with alternate importance matrices. This article investigates these hypotheses by quantizing Llama3.3 70B on importance matrices written in three languages (English, Norwegian, and Malayalam) and evaluating them on the MixEval dataset in both English and Norwegian. All experiments related to yielded non-significant results indicating that current quantization practices do not disproportionately harm multilingual performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17066v2">Improving LLM Outputs Against Jailbreak Attacks with Expert Model Integration</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Under review at IEEE Access. Supplementary material is included in the main PDF
    </div>
    <details class="paper-abstract">
      Using LLMs in a production environment presents security challenges that include vulnerabilities to jailbreaks and prompt injections, which can result in harmful outputs for humans or the enterprise. The challenge is amplified when working within a specific domain, as topics generally accepted for LLMs to address may be irrelevant to that field. These problems can be mitigated, for example, by fine-tuning large language models with domain-specific and security-focused data. However, these alone are insufficient, as jailbreak techniques evolve. Additionally, API-accessed models do not offer the flexibility needed to tailor behavior to industry-specific objectives, and in-context learning is not always sufficient or reliable. In response to these challenges, we introduce Archias, an expert model adept at distinguishing between in-domain and out-of-domain communications. Archias classifies user inquiries into several categories: in-domain (specifically for the automotive industry), malicious questions, price injections, prompt injections, and out-of-domain examples. Our methodology integrates outputs from the expert model (Archias) into prompts, which are then processed by the LLM to generate responses. This method increases the model's ability to understand the user's intention and give appropriate answers. Archias can be adjusted, fine-tuned, and used for many different purposes due to its small size. Therefore, it can be easily customized to the needs of any industry. To validate our approach, we created a benchmark dataset for the automotive industry. Furthermore, in the interest of advancing research and development, we release our benchmark dataset to the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22050v1">Decoding Machine Translationese in English-Chinese News: LLMs vs. NMTs</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 14 pages, 5 figures, 6 tables. Accpeted in MT Summit 2025, Research: Technical track. Official version may be accessed later in the ACL Anthology
    </div>
    <details class="paper-abstract">
      This study explores Machine Translationese (MTese) -- the linguistic peculiarities of machine translation outputs -- focusing on the under-researched English-to-Chinese language pair in news texts. We construct a large dataset consisting of 4 sub-corpora and employ a comprehensive five-layer feature set. Then, a chi-square ranking algorithm is applied for feature selection in both classification and clustering tasks. Our findings confirm the presence of MTese in both Neural Machine Translation systems (NMTs) and Large Language Models (LLMs). Original Chinese texts are nearly perfectly distinguishable from both LLM and NMT outputs. Notable linguistic patterns in MT outputs are shorter sentence lengths and increased use of adversative conjunctions. Comparing LLMs and NMTs, we achieve approximately 70% classification accuracy, with LLMs exhibiting greater lexical diversity and NMTs using more brackets. Additionally, translation-specific LLMs show lower lexical diversity but higher usage of causal conjunctions compared to generic LLMs. Lastly, we find no significant differences between LLMs developed by Chinese firms and their foreign counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22038v1">Can Peter Pan Survive MT? A Stylometric Study of LLMs, NMTs, and HTs in Children's Literature Translation</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 19 pages, 8 figures, 4 tables. Accepted in 2nd Workshop on Creative-text Translation and Technology Co-located with MT Summit 2025. Official paper may later be accessed from ACL Anthology
    </div>
    <details class="paper-abstract">
      This study focuses on evaluating the performance of machine translations (MTs) compared to human translations (HTs) in English-to-Chinese children's literature translation (CLT) from a stylometric perspective. The research constructs a Peter Pan corpus, comprising 21 translations: 7 human translations (HTs), 7 large language model translations (LLMs), and 7 neural machine translation outputs (NMTs). The analysis employs a generic feature set (including lexical, syntactic, readability, and n-gram features) and a creative text translation (CTT-specific) feature set, which captures repetition, rhythm, translatability, and miscellaneous levels, yielding 447 linguistic features in total. Using classification and clustering techniques in machine learning, we conduct a stylometric analysis of these translations. Results reveal that in generic features, HTs and MTs exhibit significant differences in conjunction word distributions and the ratio of 1-word-gram-YiYang, while NMTs and LLMs show significant variation in descriptive words usage and adverb ratios. Regarding CTT-specific features, LLMs outperform NMTs in distribution, aligning more closely with HTs in stylistic characteristics, demonstrating the potential of LLMs in CLT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03492v2">Towards Reproducible LLM Evaluation: Quantifying Uncertainty in LLM Benchmark Scores</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 4 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are stochastic, and not all models give deterministic answers, even when setting temperature to zero with a fixed random seed. However, few benchmark studies attempt to quantify uncertainty, partly due to the time and cost of repeated experiments. We use benchmarks designed for testing LLMs' capacity to reason about cardinal directions to explore the impact of experimental repeats on mean score and prediction interval. We suggest a simple method for cost-effectively quantifying the uncertainty of a benchmark score and make recommendations concerning reproducible LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22033v1">SiPipe: Bridging the CPU-GPU Utilization Gap for Efficient Pipeline-Parallel LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      As inference workloads for large language models (LLMs) scale to meet growing user demand, pipeline parallelism (PP) has become a widely adopted strategy for multi-GPU deployment, particularly in cross-node setups, to improve key-value (KV) cache capacity and inference throughput. However, PP suffers from inherent inefficiencies caused by three types of execution bubbles-load-imbalance, intra-stage, and inter-stage-which limit pipeline saturation. We present SiPipe, a heterogeneous pipeline design that improves throughput by leveraging underutilized CPU resources to offload auxiliary computation and communication. SiPipe incorporates three key techniques-CPU sampling, a token-safe execution model, and structure-aware transmission-to mitigate pipeline bubbles and improve execution efficiency. Across diverse LLMs, SiPipe achieves up to 2.1 times higher throughput, 43% lower per-token latency, and up to 23% higher average GPU utilization compared to the state-of-the-art vLLM under the same PP configuration, demonstrating its generality across LLMs and deployment scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22028v1">LMPVC and Policy Bank: Adaptive voice control for industrial robots with code generating LLMs and reusable Pythonic policies</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Accepted by the 2025 34th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN). For further information, videos and code, see https://github.com/ozzyuni/LMPVC
    </div>
    <details class="paper-abstract">
      Modern industry is increasingly moving away from mass manufacturing, towards more specialized and personalized products. As manufacturing tasks become more complex, full automation is not always an option, human involvement may be required. This has increased the need for advanced human robot collaboration (HRC), and with it, improved methods for interaction, such as voice control. Recent advances in natural language processing, driven by artificial intelligence (AI), have the potential to answer this demand. Large language models (LLMs) have rapidly developed very impressive general reasoning capabilities, and many methods of applying this to robotics have been proposed, including through the use of code generation. This paper presents Language Model Program Voice Control (LMPVC), an LLM-based prototype voice control architecture with integrated policy programming and teaching capabilities, built for use with Robot Operating System 2 (ROS2) compatible robots. The architecture builds on prior works using code generation for voice control by implementing an additional programming and teaching system, the Policy Bank. We find this system can compensate for the limitations of the underlying LLM, and allow LMPVC to adapt to different downstream tasks without a slow and costly training process. The architecture and additional results are released on GitHub (https://github.com/ozzyuni/LMPVC).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00299v3">ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 41 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) require significant GPU memory when processing long texts, with the key value (KV) cache consuming up to 70\% of total memory during inference. Although existing compression methods reduce memory by evaluating the importance of individual tokens, they overlook critical semantic relationships between tokens, resulting in fragmented context and degraded performance. We introduce ChunkKV, which fundamentally reimagines KV cache compression by treating semantic chunks - rather than isolated tokens - as basic compression units. This approach preserves complete linguistic structures and contextual integrity, ensuring that essential meaning is retained even under aggressive compression. Our innovation includes a novel layer-wise index reuse technique that exploits the higher cross-layer similarity of preserved indices in ChunkKV, reducing computational overhead and improving throughput by 26.5\%. Comprehensive evaluations on challenging benchmarks: LongBench, Needle-In-A-HayStack, GSM8K, and JailbreakV demonstrate that ChunkKV outperforms state-of-the-art methods by up to 8.7\% in precision while maintaining the same compression ratio. These results confirm that semantic-aware compression significantly enhances both efficiency and performance for long-context LLM inference, providing a simple yet effective solution to the memory bottleneck problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02862v3">Cannot See the Forest for the Trees: Invoking Heuristics and Biases to Elicit Irrational Choices of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      Despite the remarkable performance of Large Language Models (LLMs), they remain vulnerable to jailbreak attacks, which can compromise their safety mechanisms. Existing studies often rely on brute-force optimization or manual design, failing to uncover potential risks in real-world scenarios. To address this, we propose a novel jailbreak attack framework, ICRT, inspired by heuristics and biases in human cognition. Leveraging the simplicity effect, we employ cognitive decomposition to reduce the complexity of malicious prompts. Simultaneously, relevance bias is utilized to reorganize prompts, enhancing semantic alignment and inducing harmful outputs effectively. Furthermore, we introduce a ranking-based harmfulness evaluation metric that surpasses the traditional binary success-or-failure paradigm by employing ranking aggregation methods such as Elo, HodgeRank, and Rank Centrality to comprehensively quantify the harmfulness of generated content. Experimental results show that our approach consistently bypasses mainstream LLMs' safety mechanisms and generates high-risk content, providing insights into jailbreak attack risks and contributing to stronger defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12286v2">The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly capable and widely adopted, benchmarks play a central role in assessing their practical utility. For example, SWE-Bench Verified has emerged as a critical benchmark for evaluating LLMs' software engineering abilities, particularly their aptitude for resolving real-world GitHub issues. Recent LLMs show impressive performance on SWE-Bench, leading to optimism about their capacity for complex coding tasks. However, current evaluation protocols may overstate these models' true capabilities. It is crucial to distinguish LLMs' generalizable problem-solving ability and other learned artifacts. In this work, we introduce two diagnostic tasks: file path identification from issue descriptions alone, and ground truth function reproduction with only the current file context and issue description to probe models' underlying knowledge. We present empirical evidence that performance gains on SWE-Bench-Verified may be partially driven by memorization rather than genuine problem-solving. We show that state-of-the-art models achieve up to 76% accuracy in identifying buggy file paths using only issue descriptions, without access to repository structure. This performance is merely up to 53% on tasks from repositories not included in SWE-Bench, pointing to possible data contamination or memorization. A similar pattern is also observed for the function reproduction task, where the verbatim similarity is much higher on SWE-Bench-Verified than on other similar coding benchmarks. These findings raise concerns about the validity of existing results and underscore the need for more robust, contamination-resistant benchmarks to reliably evaluate LLMs' coding abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21972v1">Advancing Jailbreak Strategies: A Hybrid Approach to Exploiting LLM Vulnerabilities and Bypassing Modern Defenses</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      The advancement of Pre-Trained Language Models (PTLMs) and Large Language Models (LLMs) has led to their widespread adoption across diverse applications. Despite their success, these models remain vulnerable to attacks that exploit their inherent weaknesses to bypass safety measures. Two primary inference-phase threats are token-level and prompt-level jailbreaks. Token-level attacks embed adversarial sequences that transfer well to black-box models like GPT but leave detectable patterns and rely on gradient-based token optimization, whereas prompt-level attacks use semantically structured inputs to elicit harmful responses yet depend on iterative feedback that can be unreliable. To address the complementary limitations of these methods, we propose two hybrid approaches that integrate token- and prompt-level techniques to enhance jailbreak effectiveness across diverse PTLMs. GCG + PAIR and the newly explored GCG + WordGame hybrids were evaluated across multiple Vicuna and Llama models. GCG + PAIR consistently raised attack-success rates over its constituent techniques on undefended models; for instance, on Llama-3, its Attack Success Rate (ASR) reached 91.6%, a substantial increase from PAIR's 58.4% baseline. Meanwhile, GCG + WordGame matched the raw performance of WordGame maintaining a high ASR of over 80% even under stricter evaluators like Mistral-Sorry-Bench. Crucially, both hybrids retained transferability and reliably pierced advanced defenses such as Gradient Cuff and JBShield, which fully blocked single-mode attacks. These findings expose previously unreported vulnerabilities in current safety stacks, highlight trade-offs between raw success and defensive robustness, and underscore the need for holistic safeguards against adaptive adversaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21967v1">More Vulnerable than You Think: On the Stability of Tool-Integrated LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      Current evaluations of tool-integrated LLM agents typically focus on end-to-end tool-usage evaluation while neglecting their stability. This limits their real-world applicability, as various internal or external factors can cause agents to crash or behave abnormally. Our research addresses this by investigating whether agents are vulnerable to errors throughout the entire tool invocation process, including reading tool documentation, selecting tools and generating parameters, and processing the tool's response. Through extensive experiments, we observe that agents are highly susceptible to errors at each stage and agents based on open-source models are more vulnerable than those based on proprietary models. We also find that increasing the model size does not significantly improve tool invocation reasoning and may make agents more vulnerable to attacks resembling normal user instructions. This highlights the importance of evaluating agent stability and offers valuable insights for future LLM development and evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21901v1">A Survey of LLM Inference Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 25 pages
    </div>
    <details class="paper-abstract">
      The past few years has witnessed specialized large language model (LLM) inference systems, such as vLLM, SGLang, Mooncake, and DeepFlow, alongside rapid LLM adoption via services like ChatGPT. Driving these system design efforts is the unique autoregressive nature of LLM request processing, motivating new techniques for achieving high performance while preserving high inference quality over high-volume and high-velocity workloads. While many of these techniques are discussed across the literature, they have not been analyzed under the framework of a complete inference system, nor have the systems themselves been analyzed and compared. In this survey, we review these techniques, starting from operators and algorithms for request processing, then moving on to techniques for model optimization and execution, including kernel design, batching, and scheduling, before ending with techniques for memory management, including paged memory, eviction and offloading techniques, quantization, and cache persistence. Through these discussions, we show that these techniques fundamentally rely on load prediction, adaptive mechanisms, and cost reduction in order to overcome the challenges introduced by autoregressive generation and achieve the goals of the system. We then discuss how these techniques can be combined to form single-replica and multi-replica inference systems, including disaggregated inference systems that offer more control over resource allocation and serverless systems that can be deployed over shared hardware infrastructure. We end with a discussion of remaining challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20118v4">OpenTCM: A GraphRAG-Empowered LLM-based System for Traditional Chinese Medicine Knowledge Retrieval and Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 8 pages, 5 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Traditional Chinese Medicine (TCM) represents a rich repository of ancient medical knowledge that continues to play an important role in modern healthcare. Due to the complexity and breadth of the TCM literature, the integration of AI technologies is critical for its modernization and broader accessibility. However, this integration poses considerable challenges, including the interpretation of obscure classical Chinese texts and the modeling of intricate semantic relationships among TCM concepts. In this paper, we develop OpenTCM, an LLM-based system that combines a domain-specific TCM knowledge graph and Graph-based Retrieval-Augmented Generation (GraphRAG). First, we extract more than 3.73 million classical Chinese characters from 68 gynecological books in the Chinese Medical Classics Database, with the help of TCM and gynecology experts. Second, we construct a comprehensive multi-relational knowledge graph comprising more than 48,000 entities and 152,000 interrelationships, using customized prompts and Chinese-oriented LLMs such as DeepSeek and Kimi to ensure high-fidelity semantic understanding. Last, we empower OpenTCM with GraphRAG, enabling high-fidelity ingredient knowledge retrieval and diagnostic question-answering without model fine-tuning. Experimental evaluations demonstrate that OpenTCM achieves mean expert scores (MES) of 4.378 in ingredient information retrieval and 4.045 in diagnostic question-answering tasks, outperforming state-of-the-art solutions in real-world TCM use cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15294v3">Round Attention: A Novel Round-Level Attention Mechanism to Accelerate LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      The increasing context window size in large language models (LLMs) has improved their ability to handle complex, long-text tasks. However, as the conversation rounds continue, it is required to store a large amount of KV cache in GPU memory, which significantly affects the efficiency and even availability of the model serving systems. This paper analyzes dialogue data from real users on the granularity of round and discovers that the LLM inference manifests a watershed layer, after which the distribution of round-level attention shows notable similarity. Based on this, we propose Round Attention - a novel round-level attention mechanism that selectively processes the KV cache of top-k relevant rounds, where k is dynamically determined through the attention matrix in the watershed layer. Theoretical analysis demonstrates that our method reduces memory usage by 54\% to 82\%, while experimental results confirm that loading sparse critical-round KV cache maintains answer accuracy without performance degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21881v1">A Dual-Layered Evaluation of Geopolitical and Cultural Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 This paper is accepted to ACL Student Research Workshop (SRW) 2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed across diverse linguistic and cultural contexts, understanding their behavior in both factual and disputable scenarios is essential, especially when their outputs may shape public opinion or reinforce dominant narratives. In this paper, we define two types of bias in LLMs: model bias (bias stemming from model training) and inference bias (bias induced by the language of the query), through a two-phase evaluation. Phase 1 evaluates LLMs on factual questions where a single verifiable answer exists, assessing whether models maintain consistency across different query languages. Phase 2 expands the scope by probing geopolitically sensitive disputes, where responses may reflect culturally embedded or ideologically aligned perspectives. We construct a manually curated dataset spanning both factual and disputable QA, across four languages and question types. The results show that Phase 1 exhibits query language induced alignment, while Phase 2 reflects an interplay between the model's training context and query language. This paper offers a structured framework for evaluating LLM behavior across neutral and sensitive topics, providing insights for future LLM deployment and culturally aware evaluation practices in multilingual contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15783v2">Grammar and Gameplay-aligned RL for Game Description Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Published at IEEE Conference on Games, 2025
    </div>
    <details class="paper-abstract">
      Game Description Generation (GDG) is the task of generating a game description written in a Game Description Language (GDL) from natural language text. Previous studies have explored generation methods leveraging the contextual understanding capabilities of Large Language Models (LLMs); however, accurately reproducing the game features of the game descriptions remains a challenge. In this paper, we propose reinforcement learning-based fine-tuning of LLMs for GDG (RLGDG). Our training method simultaneously improves grammatical correctness and fidelity to game concepts by introducing both grammar rewards and concept rewards. Furthermore, we adopt a two-stage training strategy where Reinforcement Learning (RL) is applied following Supervised Fine-Tuning (SFT). Experimental results demonstrate that our proposed method significantly outperforms baseline methods using SFT alone. Our code is available at https://github.com/tsunehiko/rlgdg
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14291v2">Advances in Temporal Point Processes: Bayesian, Neural, and LLM Approaches</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      Temporal point processes (TPPs) are stochastic process models used to characterize event sequences occurring in continuous time. Traditional statistical TPPs have a long-standing history, with numerous models proposed and successfully applied across diverse domains. In recent years, advances in deep learning have spurred the development of neural TPPs, enabling greater flexibility and expressiveness in capturing complex temporal dynamics. The emergence of large language models (LLMs) has further sparked excitement, offering new possibilities for modeling and analyzing event sequences by leveraging their rich contextual understanding. This survey presents a comprehensive review of recent research on TPPs from three perspectives: Bayesian, deep learning, and LLM approaches. We begin with a review of the fundamental concepts of TPPs, followed by an in-depth discussion of model design and parameter estimation techniques in these three frameworks. We also revisit classic application areas of TPPs to highlight their practical relevance. Finally, we outline challenges and promising directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21875v1">WildSpeech-Bench: Benchmarking Audio LLMs in Natural Speech Conversation</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      Recent multi-modal Large Language Models (LLMs) such as GPT-4o have demonstrated strong capabilities of direct speech interaction. However, the lack of specialized and comprehensive benchmarks for end-to-end speech LLM evaluation hinders optimizing the user experience of Audio LLMs in real-world applications. Existing evaluation methods often adapt text-based benchmarks, overlooking speech's unique characteristics and challenges, including prosody, homophones, stuttering, and differing user expectations. Here, we present a novel approach to thoroughly evaluate LLMs in practical speech conversations. We systematically curate real-world chat data relevant to spoken scenarios, introduce diversity in speaker attributes and acoustic conditions, and augment the dataset with speech-specific phenomena. We further design a query-aware evaluation method to use customized evaluation checklists and prompts to enhance the accuracy of automatic evaluation. We conduct comprehensive testing and detailed analysis of various mainstream speech models, revealing significant differences in model performance across different speech scenarios. The use of query-aware evaluation further enables a finer-grained assessment under various speech-specific scenarios. Our benchmark can provide valuable insights for speech model development and evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11718v2">REMOR: Automated Peer Review Generation with LLM Reasoning and Multi-Objective Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 18 pages, 6 figures
    </div>
    <details class="paper-abstract">
      AI-based peer review systems tend to produce shallow and overpraising suggestions compared to human feedback. Here, we evaluate how well a reasoning LLM trained with multi-objective reinforcement learning (REMOR) can overcome these limitations. We start by designing a multi-aspect reward function that aligns with human evaluation of reviews. The aspects are related to the review itself (e.g., criticisms, novelty) and the relationship between the review and the manuscript (i.e., relevance). First, we perform supervised fine-tuning of DeepSeek-R1-Distill-Qwen-7B using LoRA on PeerRT, a new dataset of high-quality top AI conference reviews enriched with reasoning traces. We then apply Group Relative Policy Optimization (GRPO) to train two models: REMOR-H (with the human-aligned reward) and REMOR-U (with a uniform reward). Interestingly, the human-aligned reward penalizes aspects typically associated with strong reviews, leading REMOR-U to produce qualitatively more substantive feedback. Our results show that REMOR-U and REMOR-H achieve more than twice the average rewards of human reviews, non-reasoning state-of-the-art agentic multi-modal AI review systems, and general commercial LLM baselines. We found that while the best AI and human reviews are comparable in quality, REMOR avoids the long tail of low-quality human reviews. We discuss how reasoning is key to achieving these improvements and release the Human-aligned Peer Review Reward (HPRR) function, the Peer Review Reasoning-enriched Traces (PeerRT) dataset, and the REMOR models, which we believe can help spur progress in the area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14275v2">Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 ICML 2025 Camera Ready
    </div>
    <details class="paper-abstract">
      Advances in Large Language Models (LLMs) have sparked interest in their ability to solve Olympiad-level math problems. However, the training and evaluation of these models are constrained by the limited size and quality of available datasets, as creating large-scale data for such advanced problems requires extensive effort from human experts. In addition, current benchmarks are prone to contamination, leading to unreliable evaluations. In this paper, we present an automated pipeline that leverages the rich resources of the Art of Problem Solving (AoPS) forum, which predominantly features Olympiad-level problems and community-driven solutions. Using open-source LLMs, we develop a method to extract question-answer pairs from the forum, resulting in AoPS-Instruct, a dataset of more than 600,000 high-quality QA pairs. Our experiments demonstrate that fine-tuning LLMs on AoPS-Instruct improves their reasoning abilities across various benchmarks. Moreover, we build an automatic pipeline that introduces LiveAoPSBench, an evolving evaluation set with timestamps, derived from the latest forum data, providing a contamination-resistant benchmark for assessing LLM performance. Notably, we observe a significant decline in LLM performance over time, suggesting their success on older examples may stem from pre-training exposure rather than true reasoning ability. Our work presents a scalable approach to creating and maintaining large-scale, high-quality datasets for advanced math reasoning, offering valuable insights into the capabilities and limitations of LLMs in this domain. Our benchmark and code is available at https://github.com/DSL-Lab/aops
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16661v3">RLSF: Fine-tuning LLMs via Symbolic Feedback</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed AI but often struggle with tasks that require domain-specific reasoning and logical alignment. Traditional fine-tuning methods do not leverage the vast amount of symbolic domain-knowledge available to us via symbolic reasoning tools (e.g., provers), and are further limited by sparse rewards and unreliable reward models. We introduce Reinforcement Learning via Symbolic Feedback (RLSF), a novel fine-tuning paradigm where symbolic reasoning tools (e.g., solvers, provers, and algebra systems) provide fine-grained feedback to LLMs. RLSF uses poly-sized certificates (e.g., proofs) generated by symbolic tools to identify and correct errors in model outputs, offering token-level guidance without requiring differentiable reasoning systems. This paradigm bridges the gap between symbolic reasoning and LLM fine-tuning, enabling precise alignment with domain-specific constraints while addressing key limitations of traditional reward signals. Via extensive evaluations, we show that our RLSF-based fine-tuning of LLMs outperforms traditional approaches on five different applications (that have some associated logical or domain constraints), namely, program synthesis from natural language pseudo-code to programming language, three chemistry tasks, and solving the Game of 24. A key takeaway is that fine-tuning via RLSF enables relatively smaller LLMs to significantly outperform closed-source models that are orders of magnitude larger.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21551v1">Where to find Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Grokking, i.e., test performance keeps improving long after training loss converged, has been recently witnessed in neural network training, making the mechanism of generalization and other emerging capabilities such as reasoning mysterious. While prior studies usually train small models on a few toy or highly-specific tasks for thousands of epochs, we conduct the first study of grokking on checkpoints during one-pass pretraining of a 7B large language model (LLM), i.e., OLMoE. We compute the training loss and evaluate generalization on diverse benchmark tasks, including math reasoning, code generation, and commonsense/domain-specific knowledge retrieval tasks. Our study, for the first time, verifies that grokking still happens in the pretraining of large-scale foundation models, though different data may enter grokking stages asynchronously. We further demystify grokking's "emergence of generalization" by investigating LLM internal dynamics. Specifically, we find that training samples' pathways (i.e., expert choices across layers) evolve from random, instance-specific to more structured and shareable between samples during grokking. Also, the complexity of a sample's pathway reduces despite the converged loss. These indicate a memorization-to-generalization conversion, providing a mechanistic explanation of delayed generalization. In the study, we develop two novel metrics to quantify pathway distance and the complexity of a single pathway. We show their ability to predict the generalization improvement on diverse downstream tasks. They are efficient, simple to compute and solely dependent on training data. Hence, they have practical value for pretraining, enabling us to monitor the generalization performance without finetuning and test. Theoretically, we show that more structured pathways reduce model complexity and improve the generalization bound.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21497v1">Enhancing User Engagement in Socially-Driven Dialogue through Interactive LLM Alignments</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Enhancing user engagement through interactions plays an essential role in socially-driven dialogues. While prior works have optimized models to reason over relevant knowledge or plan a dialogue act flow, the relationship between user engagement and knowledge or dialogue acts is subtle and does not guarantee user engagement in socially-driven dialogues. To this end, we enable interactive LLMs to learn user engagement by leveraging signals from the future development of conversations. Specifically, we adopt a more direct and relevant indicator of user engagement, i.e., the user's reaction related to dialogue intention after the interaction, as a reward to align interactive LLMs. To achieve this, we develop a user simulator to interact with target interactive LLMs and explore interactions between the user and the interactive LLM system via \textit{i$\times$MCTS} (\textit{M}onte \textit{C}arlo \textit{T}ree \textit{S}earch for \textit{i}nteraction). In this way, we collect a dataset containing pairs of higher and lower-quality experiences using \textit{i$\times$MCTS}, and align interactive LLMs for high-level user engagement by direct preference optimization (DPO) accordingly. Experiments conducted on two socially-driven dialogue scenarios (emotional support conversations and persuasion for good) demonstrate that our method effectively enhances user engagement in interactive LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02398v3">Prompting with Phonemes: Enhancing LLMs' Multilinguality for Non-Latin Script Languages</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 Accepted to NAACL 2025 (Main Conference). This version contains minor improvements to the camera-ready
    </div>
    <details class="paper-abstract">
      Although multilingual LLMs have achieved remarkable performance across benchmarks, we find they continue to underperform on non-Latin script languages across contemporary LLM families. This discrepancy arises from the fact that LLMs are pretrained with orthographic scripts, which are dominated by Latin characters that obscure their shared phonology with non-Latin scripts. We propose leveraging phonemic transcriptions as complementary signals to induce script-invariant representations. Our study demonstrates that integrating phonemic signals improves performance across both non-Latin and Latin script languages, with a particularly significant impact on closing the performance gap between the two. Through detailed experiments, we show that phonemic and orthographic scripts retrieve distinct examples for in-context learning (ICL). This motivates our proposed Mixed-ICL retrieval strategy, where further aggregation from both leads to our significant performance improvements for both Latin script languages (up to 12.6%) and non-Latin script languages (up to 15.1%) compared to randomized ICL retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21443v1">Domain Knowledge-Enhanced LLMs for Fraud and Concept Drift Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Detecting deceptive conversations on dynamic platforms is increasingly difficult due to evolving language patterns and Concept Drift (CD)\-i.e., semantic or topical shifts that alter the context or intent of interactions over time. These shifts can obscure malicious intent or mimic normal dialogue, making accurate classification challenging. While Large Language Models (LLMs) show strong performance in natural language tasks, they often struggle with contextual ambiguity and hallucinations in risk\-sensitive scenarios. To address these challenges, we present a Domain Knowledge (DK)\-Enhanced LLM framework that integrates pretrained LLMs with structured, task\-specific insights to perform fraud and concept drift detection. The proposed architecture consists of three main components: (1) a DK\-LLM module to detect fake or deceptive conversations; (2) a drift detection unit (OCDD) to determine whether a semantic shift has occurred; and (3) a second DK\-LLM module to classify the drift as either benign or fraudulent. We first validate the value of domain knowledge using a fake review dataset and then apply our full framework to SEConvo, a multiturn dialogue dataset that includes various types of fraud and spam attacks. Results show that our system detects fake conversations with high accuracy and effectively classifies the nature of drift. Guided by structured prompts, the LLaMA\-based implementation achieves 98\% classification accuracy. Comparative studies against zero\-shot baselines demonstrate that incorporating domain knowledge and drift awareness significantly improves performance, interpretability, and robustness in high\-stakes NLP applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15830v2">Rethinking LLM Training through Information Geometry and Quantum Metrics</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 9 pages, 1 figure(s)
    </div>
    <details class="paper-abstract">
      Optimization in large language models (LLMs) unfolds over high-dimensional parameter spaces with non-Euclidean structure. Information geometry frames this landscape using the Fisher information metric, enabling more principled learning via natural gradient descent. Though often impractical, this geometric lens clarifies phenomena such as sharp minima, generalization, and observed scaling laws. We argue that curvature-aware approaches deepen our understanding of LLM training. Finally, we speculate on quantum analogies based on the Fubini-Study metric and Quantum Fisher Information, hinting at efficient optimization in quantum-enhanced systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04202v3">TracLLM: A Generic Framework for Attributing Long Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 To appear in USENIX Security Symposium 2025. The code and data are at: https://github.com/Wang-Yanting/TracLLM
    </div>
    <details class="paper-abstract">
      Long context large language models (LLMs) are deployed in many real-world applications such as RAG, agent, and broad LLM-integrated applications. Given an instruction and a long context (e.g., documents, PDF files, webpages), a long context LLM can generate an output grounded in the provided context, aiming to provide more accurate, up-to-date, and verifiable outputs while reducing hallucinations and unsupported claims. This raises a research question: how to pinpoint the texts (e.g., sentences, passages, or paragraphs) in the context that contribute most to or are responsible for the generated output by an LLM? This process, which we call context traceback, has various real-world applications, such as 1) debugging LLM-based systems, 2) conducting post-attack forensic analysis for attacks (e.g., prompt injection attack, knowledge corruption attacks) to an LLM, and 3) highlighting knowledge sources to enhance the trust of users towards outputs generated by LLMs. When applied to context traceback for long context LLMs, existing feature attribution methods such as Shapley have sub-optimal performance and/or incur a large computational cost. In this work, we develop TracLLM, the first generic context traceback framework tailored to long context LLMs. Our framework can improve the effectiveness and efficiency of existing feature attribution methods. To improve the efficiency, we develop an informed search based algorithm in TracLLM. We also develop contribution score ensemble/denoising techniques to improve the accuracy of TracLLM. Our evaluation results show TracLLM can effectively identify texts in a long context that lead to the output of an LLM. Our code and data are at: https://github.com/Wang-Yanting/TracLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21384v1">Leveraging LLM-Assisted Query Understanding for Live Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 Accepted at SIGIR 2025 LiveRAG Workshop (Oral Presentation)
    </div>
    <details class="paper-abstract">
      Real-world live retrieval-augmented generation (RAG) systems face significant challenges when processing user queries that are often noisy, ambiguous, and contain multiple intents. While RAG enhances large language models (LLMs) with external knowledge, current systems typically struggle with such complex inputs, as they are often trained or evaluated on cleaner data. This paper introduces Omni-RAG, a novel framework designed to improve the robustness and effectiveness of RAG systems in live, open-domain settings. Omni-RAG employs LLM-assisted query understanding to preprocess user inputs through three key modules: (1) Deep Query Understanding and Decomposition, which utilizes LLMs with tailored prompts to denoise queries (e.g., correcting spelling errors) and decompose multi-intent queries into structured sub-queries; (2) Intent-Aware Knowledge Retrieval, which performs retrieval for each sub-query from a corpus (i.e., FineWeb using OpenSearch) and aggregates the results; and (3) Reranking and Generation, where a reranker (i.e., BGE) refines document selection before a final response is generated by an LLM (i.e., Falcon-10B) using a chain-of-thought prompt. Omni-RAG aims to bridge the gap between current RAG capabilities and the demands of real-world applications, such as those highlighted by the SIGIR 2025 LiveRAG Challenge, by robustly handling complex and noisy queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21319v1">Multimodal LLMs for Visualization Reconstruction and Understanding</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Visualizations are crucial for data communication, yet understanding them requires comprehension of both visual elements and their underlying data relationships. Current multimodal large models, while effective in natural image understanding, struggle with visualization due to their inability to decode the data-to-visual mapping rules and extract structured information. To address these challenges, we present a novel dataset and train multimodal visualization LLMs specifically designed for understanding. Our approach combines chart images with their corresponding vectorized representations, encoding schemes, and data features. The proposed vector format enables compact and accurate reconstruction of visualization content. Experimental results demonstrate significant improvements in both data extraction accuracy and chart reconstruction quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13379v2">Thinkless: LLM Learns When to Think</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Reasoning Language Models, capable of extended chain-of-thought reasoning, have demonstrated remarkable performance on tasks requiring complex logical inference. However, applying elaborate reasoning for all queries often results in substantial computational inefficiencies, particularly when many problems admit straightforward solutions. This motivates an open question: Can LLMs learn when to think? To answer this, we propose Thinkless, a learnable framework that empowers an LLM to adaptively select between short-form and long-form reasoning, based on both task complexity and the model's ability. Thinkless is trained under a reinforcement learning paradigm and employs two control tokens, <short> for concise responses and <think> for detailed reasoning. At the core of our method is a Decoupled Group Relative Policy Optimization (DeGRPO) algorithm, which decomposes the learning objective of hybrid reasoning into two components: (1) a control token loss that governs the selection of the reasoning mode, and (2) a response loss that improves the accuracy of the generated answers. This decoupled formulation enables fine-grained control over the contributions of each objective, stabilizing training and effectively preventing collapse observed in vanilla GRPO. Empirically, on several benchmarks such as Minerva Algebra, MATH-500, and GSM8K, Thinkless is able to reduce the usage of long-chain thinking by 50% - 90%, significantly improving the efficiency of Reasoning Language Models. The code is available at https://github.com/VainF/Thinkless
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21285v1">Double-Checker: Enhancing Reasoning of Slow-Thinking LLMs via Self-Critical Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      While slow-thinking large language models (LLMs) exhibit reflection-like reasoning, commonly referred to as the "aha moment:, their ability to generate informative critiques and refine prior solutions remains limited. In this paper, we introduce Double-Checker, a principled framework designed to enhance the reasoning capabilities of slow-thinking LLMs by fostering explicit self-critique and iterative refinement of their previous solutions. By fine-tuning on our curated 1,730 self-critical instances, Double-Checker empowers long-CoT LLMs to iteratively critique and refine their outputs during inference until they evaluate their solutions as correct under self-generated critiques. We validate the efficacy of Double-Checker across a comprehensive suite of reasoning benchmarks, demonstrating that iterative self-critique significantly enhances the reasoning capabilities of long-CoT LLMs. Notably, our Double-Checker increases the pass@1 performance on challenging AIME benchmarks from 4.4% to 18.2% compared to the original long-CoT LLMs. These results highlight a promising direction for developing more trustworthy and effective LLMs capable of structured self-critique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21199v1">MedPrompt: LLM-CNN Fusion with Weight Routing for Medical Image Segmentation and Classification</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 40 pages, 8 Tables, 9 Figures
    </div>
    <details class="paper-abstract">
      Current medical image analysis systems are typically task-specific, requiring separate models for classification and segmentation, and lack the flexibility to support user-defined workflows. To address these challenges, we introduce MedPrompt, a unified framework that combines a few-shot prompted Large Language Model (Llama-4-17B) for high-level task planning with a modular Convolutional Neural Network (DeepFusionLab) for low-level image processing. The LLM interprets user instructions and generates structured output to dynamically route task-specific pretrained weights. This weight routing approach avoids retraining the entire framework when adding new tasks-only task-specific weights are required, enhancing scalability and deployment. We evaluated MedPrompt across 19 public datasets, covering 12 tasks spanning 5 imaging modalities. The system achieves a 97% end-to-end correctness in interpreting and executing prompt-driven instructions, with an average inference latency of 2.5 seconds, making it suitable for near real-time applications. DeepFusionLab achieves competitive segmentation accuracy (e.g., Dice 0.9856 on lungs) and strong classification performance (F1 0.9744 on tuberculosis). Overall, MedPrompt enables scalable, prompt-driven medical imaging by combining the interpretability of LLMs with the efficiency of modular CNNs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00753v4">LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 Paper lists and resources are available at https://github.com/HenryPengZou/Awesome-Human-Agent-Collaboration-Interaction-Systems
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have sparked growing interest in building fully autonomous agents. However, fully autonomous LLM-based agents still face significant challenges, including limited reliability due to hallucinations, difficulty in handling complex tasks, and substantial safety and ethical risks, all of which limit their feasibility and trustworthiness in real-world applications. To overcome these limitations, LLM-based human-agent systems (LLM-HAS) incorporate human-provided information, feedback, or control into the agent system to enhance system performance, reliability and safety. These human-agent collaboration systems enable humans and LLM-based agents to collaborate effectively by leveraging their complementary strengths. This paper provides the first comprehensive and structured survey of LLM-HAS. It clarifies fundamental concepts, systematically presents core components shaping these systems, including environment & profiling, human feedback, interaction types, orchestration and communication, explores emerging applications, and discusses unique challenges and opportunities arising from human-AI collaboration. By consolidating current knowledge and offering a structured overview, we aim to foster further research and innovation in this rapidly evolving interdisciplinary field. Paper lists and resources are available at https://github.com/HenryPengZou/Awesome-Human-Agent-Collaboration-Interaction-Systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21071v1">Enhancing LLM Tool Use with High-quality Instruction Data from Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 20 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Teaching large language models (LLMs) to use tools is crucial for improving their problem-solving abilities and expanding their applications. However, effectively using tools is challenging because it requires a deep understanding of tool functionalities and user intentions. Previous methods relied mainly on LLMs to generate instruction data, but the quality of these data was often insufficient. In this paper, we propose a new method that uses knowledge graphs to generate high-quality instruction data for LLMs. Knowledge graphs are manually curated datasets rich in semantic information. We begin by extracting various query pathways from a given knowledge graph, which are transformed into a broad spectrum of user queries. We then translate the relationships between entities into actionable tools and parse the pathways of each query into detailed solution steps, thereby creating high-quality instruction data. Our experiments show that fine-tuning on just a small sample of this synthetic data can significantly improve the tool utilization and overall capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11277v3">Search and Refine During Think: Autonomous Retrieval-Augmented Reasoning of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Large language models have demonstrated impressive reasoning capabilities but are inherently limited by their knowledge reservoir. Retrieval-augmented reasoning mitigates this limitation by allowing LLMs to query external resources, but existing methods often retrieve irrelevant or noisy information, hindering accurate reasoning. In this paper, we propose AutoRefine, a reinforcement learning post-training framework that adopts a new ``search-and-refine-during-think'' paradigm. AutoRefine introduces explicit knowledge refinement steps between successive search calls, enabling the model to iteratively filter, distill, and organize evidence before generating an answer. Furthermore, we incorporate tailored retrieval-specific rewards alongside answer correctness rewards using group relative policy optimization. Experiments on single-hop and multi-hop QA benchmarks demonstrate that AutoRefine significantly outperforms existing approaches, particularly in complex, multi-hop reasoning scenarios. Detailed analysis shows that AutoRefine issues frequent, higher-quality searches and synthesizes evidence effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21033v1">BLOCKS: Blockchain-supported Cross-Silo Knowledge Sharing for Efficient LLM Services</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      The hallucination problem of Large Language Models (LLMs) has increasingly drawn attention. Augmenting LLMs with external knowledge is a promising solution to address this issue. However, due to privacy and security concerns, a vast amount of downstream task-related knowledge remains dispersed and isolated across various "silos," making it difficult to access. To bridge this knowledge gap, we propose a blockchain-based external knowledge framework that coordinates multiple knowledge silos to provide reliable foundational knowledge for large model retrieval while ensuring data security. Technically, we distill knowledge from local data into prompts and execute transactions and records on the blockchain. Additionally, we introduce a reputation mechanism and cross-validation to ensure knowledge quality and provide incentives for participation. Furthermore, we design a query generation framework that provides a direct API interface for large model retrieval. To evaluate the performance of our proposed framework, we conducted extensive experiments on various knowledge sources. The results demonstrate that the proposed framework achieves efficient LLM service knowledge sharing in blockchain environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18389v3">ALLSTaR: Automated LLM-Driven Scheduler Generation and Testing for Intent-Based RAN</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 Under submission to an IEEE journal, copyright may change without notice
    </div>
    <details class="paper-abstract">
      The evolution toward open, programmable O-RAN and AI-RAN 6G networks creates unprecedented opportunities for Intent-Based Networking (IBN) to dynamically optimize RAN[...]. However, applying IBN effectively to the RAN scheduler [...] remains a significant challenge. Current approaches predominantly rely on coarse-grained network slicing, lacking the granularity for dynamic adaptation to individual user conditions and traffic patterns. Despite the existence of a vast body of scheduling algorithms [...], their practical utilization is hindered by implementation heterogeneity, insufficient systematic evaluation in production environments, and the complexity of developing high-performance scheduler implementations.[...] To address these limitations, we propose ALLSTaR (Automated LLm-driven Scheduler generation and Testing for intent-based RAN), a novel framework leveraging LLMs for automated, intent-driven scheduler design, implementation, and evaluation. ALLSTaR interprets NL intents, automatically generates functional scheduler code from the research literature using OCR and LLMs, and intelligently matches operator intents to the most suitable scheduler(s). Our implementation deploys these schedulers as O-RAN dApps, enabling on-the-fly deployment and testing on a production-grade, 5G-compliant testbed. This approach has enabled the largest-scale OTA experimental comparison of 18 scheduling algorithms automatically synthesized from the academic literature. The resulting performance profiles serve as the input for our Intent-Based Scheduling (IBS) framework, which dynamically selects and deploys appropriate schedulers that optimally satisfy operator intents. We validate our approach through multiple use cases unattainable with current slicing-based optimization techniques, demonstrating fine-grained control based on buffer status, physical layer conditions, and heterogeneous traffic types
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20993v1">SAC: A Framework for Measuring and Inducing Personality Traits in LLMs with Dynamic Intensity Control</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have gained significant traction across a wide range of fields in recent years. There is also a growing expectation for them to display human-like personalities during interactions. To meet this expectation, numerous studies have proposed methods for modelling LLM personalities through psychometric evaluations. However, most existing models face two major limitations: they rely on the Big Five (OCEAN) framework, which only provides coarse personality dimensions, and they lack mechanisms for controlling trait intensity. In this paper, we address this gap by extending the Machine Personality Inventory (MPI), which originally used the Big Five model, to incorporate the 16 Personality Factor (16PF) model, allowing expressive control over sixteen distinct traits. We also developed a structured framework known as Specific Attribute Control (SAC) for evaluating and dynamically inducing trait intensity in LLMs. Our method introduces adjective-based semantic anchoring to guide trait intensity expression and leverages behavioural questions across five intensity factors: \textit{Frequency}, \textit{Depth}, \textit{Threshold}, \textit{Effort}, and \textit{Willingness}. Through experimentation, we find that modelling intensity as a continuous spectrum yields substantially more consistent and controllable personality expression compared to binary trait toggling. Moreover, we observe that changes in target trait intensity systematically influence closely related traits in psychologically coherent directions, suggesting that LLMs internalize multi-dimensional personality structures rather than treating traits in isolation. Our work opens new pathways for controlled and nuanced human-machine interactions in domains such as healthcare, education, and interviewing processes, bringing us one step closer to truly human-like social machines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18313v2">Will LLMs be Professional at Fund Investment? DeepFund: A Live Arena Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 6 pages, 3 figures, perspective paper
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities across various domains, but their effectiveness in financial decision-making remains inadequately evaluated. Current benchmarks primarily assess LLMs' understanding on financial documents rather than the ability to manage assets or dig out trading opportunities in dynamic market conditions. Despite the release of new benchmarks for evaluating diversified tasks on the financial domain, we identified four major problems in these benchmarks, which are data leakage, navel-gazing, over-intervention, and maintenance-hard. To pave the research gap, we introduce DeepFund, a comprehensive arena platform for evaluating LLM-based trading strategies in a live environment. Our approach implements a multi-agent framework where they serve as multiple key roles that realize the real-world investment decision processes. Moreover, we provide a web interface that visualizes LLMs' performance with fund investment metrics across different market conditions, enabling detailed comparative analysis. Through DeepFund, we aim to provide a more realistic and fair assessment on LLM's capabilities in fund investment, offering diversified insights and revealing their potential applications in real-world financial markets. Our code is publicly available at https://github.com/HKUSTDial/DeepFund.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03359v2">WiS Platform: Enhancing Evaluation of LLM-Based Multi-Agent Systems Through Game-Based Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Recent advancements in autonomous multi-agent systems (MAS) based on large language models (LLMs) have enhanced the application scenarios and improved the capability of LLMs to handle complex tasks. Despite demonstrating effectiveness, existing studies still evidently struggle to evaluate, analysis, and reproducibility of LLM-based MAS. In this paper, to facilitate the research on LLM-based MAS, we introduce an open, scalable, and real-time updated platform for accessing and analyzing the LLM-based MAS based on the games Who is Spy?" (WiS). Our platform is featured with three main worths: (1) a unified model evaluate interface that supports models available on Hugging Face; (2) real-time updated leaderboard for model evaluation; (3) a comprehensive evaluation covering game-winning rates, attacking, defense strategies, and reasoning of LLMs. To rigorously test WiS, we conduct extensive experiments coverage of various open- and closed-source LLMs, we find that different agents exhibit distinct and intriguing behaviors in the game. The experimental results demonstrate the effectiveness and efficiency of our platform in evaluating LLM-based MAS. Our platform and its documentation are publicly available at https://whoisspy.ai/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20982v1">Our Coding Adventure: Using LLMs to Personalise the Narrative of a Tangible Programming Robot for Preschoolers</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 accepted at D-SAIL Workshop - Transformative Curriculum Design: Digitalization, Sustainability, and AI Literacy for 21st Century Learning
    </div>
    <details class="paper-abstract">
      Finding balanced ways to employ Large Language Models (LLMs) in education is a challenge due to inherent risks of poor understanding of the technology and of a susceptible audience. This is particularly so with younger children, who are known to have difficulties with pervasive screen time. Working with a tangible programming robot called Cubetto, we propose an approach to benefit from the capabilities of LLMs by employing such models in the preparation of personalised storytelling, necessary for preschool children to get accustomed to the practice of commanding the robot. We engage in action research to develop an early version of a formalised process to rapidly prototype game stories for Cubetto. Our approach has both reproducible results, because it employs open weight models, and is model-agnostic, because we test it with 5 different LLMs. We document on one hand the process, the used materials and prompts, and on the other the learning experience and outcomes. We deem the generation successful for the intended purposes of using the results as a teacher aid. Testing the models on 4 different task scenarios, we encounter issues of consistency and hallucinations and document the corresponding evaluation process and attempts (some successful and some not) to overcome these issues. Importantly, the process does not expose children to LLMs directly. Rather, the technology is used to help teachers easily develop personalised narratives on children's preferred topics. We believe our method is adequate for preschool classes and we are planning to further experiment in real-world educational settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19324v3">Reward-Guided Speculative Decoding for Efficient LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      We introduce Reward-Guided Speculative Decoding (RSD), a novel framework aimed at improving the efficiency of inference in large language models (LLMs). RSD synergistically combines a lightweight draft model with a more powerful target model, incorporating a controlled bias to prioritize high-reward outputs, in contrast to existing speculative decoding methods that enforce strict unbiasedness. RSD employs a process reward model to evaluate intermediate decoding steps and dynamically decide whether to invoke the target model, optimizing the trade-off between computational cost and output quality. We theoretically demonstrate that a threshold-based mixture strategy achieves an optimal balance between resource utilization and performance. Extensive evaluations on challenging reasoning benchmarks, including Olympiad-level tasks, show that RSD delivers significant efficiency gains against decoding with the target model only (up to 4.4x fewer FLOPs), while achieving significant better accuracy than parallel decoding method on average (up to +3.5). These results highlight RSD as a robust and cost-effective approach for deploying LLMs in resource-intensive scenarios. The code is available at https://github.com/BaohaoLiao/RSD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20938v1">ParEval-Repo: A Benchmark Suite for Evaluating LLMs with Repository-level HPC Translation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 11 pages, 5 figures
    </div>
    <details class="paper-abstract">
      GPGPU architectures have become significantly diverse in recent years, which has led to an emergence of a variety of specialized programming models and software stacks to support them. While portable execution models exist, they still require significant developer effort to port to and optimize for different hardware architectures. Recent advances in large language models (LLMs) can help us reduce some of this programmer burden. In this paper, we present a novel benchmark and testing framework, ParEval-Repo, which can be used to evaluate the efficacy of LLM-based approaches in automatically translating entire codebases across GPGPU execution models. ParEval-Repo includes several scientific computing and AI mini-applications in a range of programming models, and levels of repository complexity. We use ParEval-Repo to evaluate a range of state-of-the-art open-source and commercial LLMs, with both a non-agentic and a top-down agentic approach. We assess code generated by the LLMs and approaches in terms of compilability, functional correctness, categories of build errors, and the cost of translation in terms of the number of inference tokens. Our results demonstrate that LLM translation of scientific applications is feasible for small programs but difficulty with generating functional build systems and cross-file dependencies pose challenges in scaling to larger codebases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13547v2">ToolScan: A Benchmark for Characterizing Errors in Tool-Use LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-26
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) is one of the most critical aspects of building a performant compound AI system. Since the output from LLMs propagate to downstream steps, identifying LLM errors is crucial to system performance. A common task for LLMs in AI systems is tool use. While there are several benchmark environments for evaluating LLMs on this task, they typically only give a success rate without any explanation of the failure cases. To solve this problem, we introduce TOOLSCAN, a new benchmark to identify error patterns in LLM output on tool-use tasks. Our benchmark data set comprises of queries from diverse environments that can be used to test for the presence of seven newly characterized error patterns. Using TOOLSCAN, we show that even the most prominent LLMs exhibit these error patterns in their outputs. Researchers can use these insights from TOOLSCAN to guide their error mitigation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20921v1">LLM-guided Chemical Process Optimization with a Multi-Agent Approach</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 16 pages (main manuscript without references), 2 figures
    </div>
    <details class="paper-abstract">
      Chemical process optimization is crucial to maximize production efficiency and economic performance. Traditional methods, including gradient-based solvers, evolutionary algorithms, and parameter grid searches, become impractical when operating constraints are ill-defined or unavailable, requiring engineers to rely on subjective heuristics to estimate feasible parameter ranges. To address this constraint definition bottleneck, we present a multi-agent framework of large language model (LLM) agents that autonomously infer operating constraints from minimal process descriptions, then collaboratively guide optimization using the inferred constraints. Our AutoGen-based agentic framework employs OpenAI's o3 model, with specialized agents for constraint generation, parameter validation, simulation execution, and optimization guidance. Through two phases - autonomous constraint generation using embedded domain knowledge, followed by iterative multi-agent optimization - the framework eliminates the need for predefined operational bounds. Validated on the hydrodealkylation process across cost, yield, and yield-to-cost ratio metrics, the framework demonstrated competitive performance with conventional optimization methods while achieving better computational efficiency, requiring fewer iterations to converge. Our approach converged in under 20 minutes, achieving a 31-fold speedup over grid search. Beyond computational efficiency, the framework's reasoning-guided search demonstrates sophisticated process understanding, correctly identifying utility trade-offs, and applying domain-informed heuristics. This approach shows significant potential for optimization scenarios where operational constraints are poorly characterized or unavailable, particularly for emerging processes and retrofit applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10323v2">ELFuzz: Efficient Input Generation via LLM-driven Synthesis Over Fuzzer Space</a></div>
    <div class="paper-meta">
      📅 2025-06-26
      | 💬 Accepted by USENIX Security'25 Cycle 2
    </div>
    <details class="paper-abstract">
      Generation-based fuzzing produces appropriate testing cases according to specifications of input grammars and semantic constraints to test systems and software. However, these specifications require significant manual efforts to construct. This paper proposes a new approach, ELFuzz (Evolution Through Large Language Models for Fuzzing), that automatically synthesizes generation-based fuzzers tailored to a system under test (SUT) via LLM-driven synthesis over fuzzer space. At a high level, it starts with minimal seed fuzzers and propels the synthesis by fully automated LLM-driven evolution with coverage guidance. Compared to previous approaches, ELFuzz can 1) seamlessly scale to SUTs of real-world sizes -- up to 1,791,104 lines of code in our evaluation -- and 2) synthesize efficient fuzzers that catch interesting grammatical structures and semantic constraints in a human-understandable way. Our evaluation compared ELFuzz with specifications manually written by domain experts and synthesized by state-of-the-art approaches. It shows that ELFuzz achieves up to 434.8% more coverage and triggers up to 174.0% more artificially injected bugs. We also used ELFuzz to conduct a real-world fuzzing campaign on the newest version of cvc5 for 14 days, and encouragingly, it found five 0-day bugs (three are exploitable). Moreover, we conducted an ablation study, which shows that the fuzzer space model, the key component of ELFuzz, contributes the most (up to 62.5%) to the effectiveness of ELFuzz. Further analysis of the fuzzers synthesized by ELFuzz confirms that they catch interesting grammatical structures and semantic constraints in a human-understandable way. The results present the promising potential of ELFuzz for more automated, efficient, and extensible input generation for fuzzing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20666v1">Inside you are many wolves: Using cognitive models to interpret value trade-offs in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 11 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Navigating everyday social situations often requires juggling conflicting goals, such as conveying a harsh truth, maintaining trust, all while still being mindful of another person's feelings. These value trade-offs are an integral part of human decision-making and language use, however, current tools for interpreting such dynamic and multi-faceted notions of values in LLMs are limited. In cognitive science, so-called "cognitive models" provide formal accounts of these trade-offs in humans, by modeling the weighting of a speaker's competing utility functions in choosing an action or utterance. In this work, we use a leading cognitive model of polite speech to interpret the extent to which LLMs represent human-like trade-offs. We apply this lens to systematically evaluate value trade-offs in two encompassing model settings: degrees of reasoning "effort" in frontier black-box models, and RL post-training dynamics of open-source models. Our results highlight patterns of higher informational utility than social utility in reasoning models, and in open-source models shown to be stronger in mathematical reasoning. Our findings from LLMs' training dynamics suggest large shifts in utility values early on in training with persistent effects of the choice of base model and pretraining data, compared to feedback dataset or alignment method. We show that our method is responsive to diverse aspects of the rapidly evolving LLM landscape, with insights for forming hypotheses about other high-level behaviors, shaping training regimes for reasoning models, and better controlling trade-offs between values during model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20598v1">Fine-Tuning and Prompt Engineering of LLMs, for the Creation of Multi-Agent AI for Addressing Sustainable Protein Production Challenges</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      The global demand for sustainable protein sources has accelerated the need for intelligent tools that can rapidly process and synthesise domain-specific scientific knowledge. In this study, we present a proof-of-concept multi-agent Artificial Intelligence (AI) framework designed to support sustainable protein production research, with an initial focus on microbial protein sources. Our Retrieval-Augmented Generation (RAG)-oriented system consists of two GPT-based LLM agents: (1) a literature search agent that retrieves relevant scientific literature on microbial protein production for a specified microbial strain, and (2) an information extraction agent that processes the retrieved content to extract relevant biological and chemical information. Two parallel methodologies, fine-tuning and prompt engineering, were explored for agent optimisation. Both methods demonstrated effectiveness at improving the performance of the information extraction agent in terms of transformer-based cosine similarity scores between obtained and ideal outputs. Mean cosine similarity scores were increased by up to 25%, while universally reaching mean scores of $\geq 0.89$ against ideal output text. Fine-tuning overall improved the mean scores to a greater extent (consistently of $\geq 0.94$) compared to prompt engineering, although lower statistical uncertainties were observed with the latter approach. A user interface was developed and published for enabling the use of the multi-agent AI system, alongside preliminary exploration of additional chemical safety-based search capabilities
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03905v3">Integrating Various Software Artifacts for Better LLM-based Bug Localization and Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 25 pages, 12 images, 10 tables, Manuscript revision submitted to a journal (2025)
    </div>
    <details class="paper-abstract">
      LLMs have garnered considerable attention for their potential to streamline Automated Program Repair (APR). LLM-based approaches can either insert the correct code or directly generate patches when provided with buggy methods. However, most of LLM-based APR methods rely on a single type of software information, without fully leveraging different software artifacts. Despite this, many LLM-based approaches do not explore which specific types of information best assist in APR. Addressing this gap is crucial for advancing LLM-based APR techniques. We propose DEVLoRe to use issue content (description and message) and stack error traces to localize buggy methods, then rely on debug information in buggy methods and issue content and stack error to localize buggy lines and generate plausible patches which can pass all unit tests. The results show that while issue content is particularly effective in assisting LLMs with fault localization and program repair, different types of software artifacts complement each other. By incorporating different artifacts, DEVLoRe successfully locates 49.3% and 47.6% of single and non-single buggy methods and generates 56.0% and 14.5% plausible patches for the Defects4J v2.0 dataset, respectively. This outperforms current state-of-the-art APR methods. Furthermore, we re-implemented and evaluated our framework, demonstrating its effectiveness in its effectiveness in resolving 9 unique issues compared to other state-of-the-art frameworks using the same or more advanced models on SWE-bench Lite.We also discussed whether a leading framework for Python code can be directly applied to Java code, or vice versa. The source code and experimental results of this work for replication are available at https://github.com/XYZboom/DEVLoRe.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20544v1">When Life Gives You Samples: The Benefits of Scaling up Inference Compute for Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have shifted focus toward scaling inference-time compute, improving performance without retraining the model. A common approach is to sample multiple outputs in parallel, and select one of these as the final output. However, work to date has focused on English and a handful of domains such as math and code. In contrast, we are most interested in techniques that generalize across open-ended tasks, formally verifiable tasks, and across languages. In this work, we study how to robustly scale inference-time compute for open-ended generative tasks in a multilingual, multi-task setting. Our findings show that both sampling strategy based on temperature variation and selection strategy must be adapted to account for diverse domains and varied language settings. We evaluate existing selection methods, revealing that strategies effective in English often fail to generalize across languages. We propose novel sampling and selection strategies specifically adapted for multilingual and multi-task inference scenarios, and show they yield notable gains across languages and tasks. In particular, our combined sampling and selection methods lead to an average +6.8 jump in win-rates for our 8B models on m-ArenaHard-v2.0 prompts, against proprietary models such as Gemini. At larger scale, Command-A (111B model) equipped with our methods, shows +9.0 improvement in win-rates on the same benchmark with just five samples against single-sample decoding, a substantial increase at minimal cost. Our results underscore the need for language- and task-aware approaches to inference-time compute, aiming to democratize performance improvements in underrepresented languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20451v1">Automatic Demonstration Selection for LLM-based Tabular Data Classification</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      A fundamental question in applying In-Context Learning (ICL) for tabular data classification is how to determine the ideal number of demonstrations in the prompt. This work addresses this challenge by presenting an algorithm to automatically select a reasonable number of required demonstrations. Our method distinguishes itself by integrating not only the tabular data's distribution but also the user's selected prompt template and the specific Large Language Model (LLM) into its estimation. Rooted in Spectral Graph Theory, our proposed algorithm defines a novel metric to quantify the similarities between different demonstrations. We then construct a similarity graph and analyze the eigenvalues of its Laplacian to derive the minimum number of demonstrations capable of representing the data within the LLM's intrinsic representation space. We validate the efficacy of our approach through experiments comparing its performance against conventional random selection algorithms on diverse datasets and LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11677v2">Towards Fully Exploiting LLM Internal States to Enhance Knowledge Boundary Perception</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 ACL2025 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit impressive performance across diverse tasks but often struggle to accurately gauge their knowledge boundaries, leading to confident yet incorrect responses. This paper explores leveraging LLMs' internal states to enhance their perception of knowledge boundaries from efficiency and risk perspectives. We investigate whether LLMs can estimate their confidence using internal states before response generation, potentially saving computational resources. Our experiments on datasets like Natural Questions, HotpotQA, and MMLU reveal that LLMs demonstrate significant pre-generation perception, which is further refined post-generation, with perception gaps remaining stable across varying conditions. To mitigate risks in critical domains, we introduce Confidence Consistency-based Calibration ($C^3$), which assesses confidence consistency through question reformulation. $C^3$ significantly improves LLMs' ability to recognize their knowledge gaps, enhancing the unknown perception rate by 5.6% on NQ and 4.9% on HotpotQA. Our findings suggest that pre-generation confidence estimation can optimize efficiency, while $C^3$ effectively controls output risks, advancing the reliability of LLMs in practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17219v2">No Free Lunch: Rethinking Internal Feedback for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Reinforcement learning has emerged as a powerful paradigm for post-training large language models (LLMs) to improve reasoning. Approaches like Reinforcement Learning from Human Feedback (RLHF) and Reinforcement Learning with Verifiable Rewards (RLVR) have shown strong results, but they require extensive external supervision. We investigate an alternative class of methods, Reinforcement Learning from Internal Feedback (RLIF), which relies solely on intrinsic model-derived signals instead of external rewards. In particular, we leverage unsupervised reward proxies such as token-level entropy, trajectory-level entropy, and self-certainty. Our theoretical analysis shows these internal objectives are partially equivalent, and we empirically evaluate various RLIF strategies on challenging math reasoning benchmarks. Experimental results demonstrate that RLIF can boost the reasoning performance of base LLMs at the beginning phase of the training, matching or surpassing RLVR techniques on these tasks. However, when training progresses, performance degrades even below the model before training. Moreover, we find that RLIF yields little improvement for instruction-tuned models, indicating diminishing returns of intrinsic feedback once an LLM is already instruction-tuned. We further analyze this limitation by mixing model weights and explain the reason of RLIF's training behaviors, providing practical guidelines for integrating internal feedback signals into LLM training. We hope our analysis of internal feedback will inform more principled and effective strategies for LLM post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20353v1">DipSVD: Dual-importance Protected SVD for Efficient LLM Compression</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      The ever-increasing computational demands and deployment costs of large language models (LLMs) have spurred numerous compressing methods. Compared to quantization and unstructured pruning, SVD compression offers superior hardware compatibility and theoretical guarantees. However, existing SVD-based methods focus on the overall discrepancy between the original and compressed matrices while overlooking the protection of critical components within the matrix, which leads to inferior performance in the compressed models. This paper proposes a dual-level importance protection mechanism to enhance SVD-based compression methods: (1) local importance protection: preserving the most critical singular vectors within each weight matrix through channel-weighted data whitening; and (2) global importance protection: enabling less important layers to bear a greater portion of the compression burden through either a heuristic or optimization-based approach, thereby minimizing the impact of compression on critical layers. Extensive experiments demonstrate that DipSVD outperforms existing SVD-based compression approaches across multiple benchmarks, achieving superior model performance especially at high model compression ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02502v2">LADM: Long-context Training Data Selection with Attention-based Dependency Measurement for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 ACL 2025, our code is available at https://github.com/ZNLP/LADM
    </div>
    <details class="paper-abstract">
      Long-context modeling has drawn more and more attention in the area of Large Language Models (LLMs). Continual training with long-context data becomes the de-facto method to equip LLMs with the ability to process long inputs. However, it still remains an open challenge to measure the quality of long-context training data. To address this issue, we propose a Long-context data selection framework with Attention-based Dependency Measurement (LADM), which can efficiently identify high-quality long-context data from a large-scale, multi-domain pre-training corpus. LADM leverages the retrieval capabilities of the attention mechanism to capture contextual dependencies, ensuring a comprehensive quality measurement of long-context data. Experimental results show that our LADM framework significantly boosts the performance of LLMs on multiple long-context tasks with only 1B tokens for continual training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20194v1">DuoGPT: Training-free Dual Sparsity through Activation-aware Pruning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) deliver strong performance but are difficult to deploy due to high memory and compute costs. While pruning reduces these demands, most methods ignore activation sparsity observed at runtime. We reinterpret activation sparsity as dynamic structured weight sparsity and propose DuoGPT, a unified framework that constructs dual-sparse (spMspV) workloads by combining unstructured weight pruning with activation sparsity. To preserve accuracy, we extend the Optimal Brain Compression (OBC) framework with activation-aware calibration and introduce output residuals from the dense model as correction terms. We further optimize the solution for efficient GPU execution, enabling scalability to billion-parameter LLMs. Evaluations on LLaMA-2 and LLaMA-3 show that DuoGPT outperforms state-of-the-art structured pruning methods by up to 9.17% accuracy at an iso-speedup of 1.39$\times$ compared to the baseline dense model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20187v1">Breaking the Boundaries of Long-Context LLM Inference: Adaptive KV Management on a Single Commodity GPU</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 15 pages, 23 figures
    </div>
    <details class="paper-abstract">
      Advanced Large Language Models (LLMs) have achieved impressive performance across a wide range of complex and long-context natural language tasks. However, performing long-context LLM inference locally on a commodity GPU (a PC) with privacy concerns remains challenging due to the increasing memory demands of the key-value (KV) cache. Existing systems typically identify important tokens and selectively offload their KV data to GPU and CPU memory. The KV data needs to be offloaded to disk due to the limited memory on a commodity GPU, but the process is bottlenecked by token importance evaluation overhead and the disk's low bandwidth. In this paper, we present LeoAM, the first efficient importance-aware long-context LLM inference system for a single commodity GPU with adaptive hierarchical GPU-CPU-Disk KV management. Our system employs an adaptive KV management strategy that partitions KV data into variable-sized chunks based on the skewed distribution of attention weights across different layers to reduce computational and additional transmission overheads. Moreover, we propose a lightweight KV abstract method, which minimizes transmission latency by storing and extracting the KV abstract of each chunk on disk instead of the full KV data. LeoAM also leverages the dynamic compression and pipeline techniques to further accelerate inference. Experimental results demonstrate that LongInfer achieves an average inference latency speedup of 3.46x, while maintaining comparable LLM response quality. In scenarios with larger batch sizes, it achieves up to a 5.47x speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20170v1">JsDeObsBench: Measuring and Benchmarking LLMs for JavaScript Deobfuscation</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Accepted by ACM CCS 2025
    </div>
    <details class="paper-abstract">
      Deobfuscating JavaScript (JS) code poses a significant challenge in web security, particularly as obfuscation techniques are frequently used to conceal malicious activities within scripts. While Large Language Models (LLMs) have recently shown promise in automating the deobfuscation process, transforming detection and mitigation strategies against these obfuscated threats, a systematic benchmark to quantify their effectiveness and limitations has been notably absent. To address this gap, we present JsDeObsBench, a dedicated benchmark designed to rigorously evaluate the effectiveness of LLMs in the context of JS deobfuscation. We detail our benchmarking methodology, which includes a wide range of obfuscation techniques ranging from basic variable renaming to sophisticated structure transformations, providing a robust framework for assessing LLM performance in real-world scenarios. Our extensive experimental analysis investigates the proficiency of cutting-edge LLMs, e.g., GPT-4o, Mixtral, Llama, and DeepSeek-Coder, revealing superior performance in code simplification despite challenges in maintaining syntax accuracy and execution reliability compared to baseline methods. We further evaluate the deobfuscation of JS malware to exhibit the potential of LLMs in security scenarios. The findings highlight the utility of LLMs in deobfuscation applications and pinpoint crucial areas for further improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16789v2">Conversational User-AI Intervention: A Study on Prompt Rewriting for Improved LLM Response Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 8 pages, ACL style
    </div>
    <details class="paper-abstract">
      Human-LLM conversations are increasingly becoming more pervasive in peoples' professional and personal lives, yet many users still struggle to elicit helpful responses from LLM Chatbots. One of the reasons for this issue is users' lack of understanding in crafting effective prompts that accurately convey their information needs. Meanwhile, the existence of real-world conversational datasets on the one hand, and the text understanding faculties of LLMs on the other, present a unique opportunity to study this problem, and its potential solutions at scale. Thus, in this paper we present the first LLM-centric study of real human-AI chatbot conversations, focused on investigating aspects in which user queries fall short of expressing information needs, and the potential of using LLMs to rewrite suboptimal user prompts. Our findings demonstrate that rephrasing ineffective prompts can elicit better responses from a conversational system, while preserving the user's original intent. Notably, the performance of rewrites improves in longer conversations, where contextual inferences about user needs can be made more accurately. Additionally, we observe that LLMs often need to -- and inherently do -- make \emph{plausible} assumptions about a user's intentions and goals when interpreting prompts. Our findings largely hold true across conversational domains, user intents, and LLMs of varying sizes and families, indicating the promise of using prompt rewriting as a solution for better human-AI interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20167v1">SEED: A Structural Encoder for Embedding-Driven Decoding in Time Series Prediction with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Multivariate time series forecasting requires models to simultaneously capture variable-wise structural dependencies and generalize across diverse tasks. While structural encoders are effective in modeling feature interactions, they lack the capacity to support semantic-level reasoning or task adaptation. Conversely, large language models (LLMs) possess strong generalization capabilities but remain incompatible with raw time series inputs. This gap limits the development of unified, transferable prediction systems. Therefore, we introduce SEED, a structural encoder for embedding-driven decoding, which integrates four stages: a token-aware encoder for patch extraction, a projection module that aligns patches with language model embeddings, a semantic reprogramming mechanism that maps patches to task-aware prototypes, and a frozen language model for prediction. This modular architecture decouples representation learning from inference, enabling efficient alignment between numerical patterns and semantic reasoning. Empirical results demonstrate that the proposed method achieves consistent improvements over strong baselines, and comparative studies on various datasets confirm SEED's role in addressing the structural-semantic modeling gap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00845v2">Rewarding Graph Reasoning Process makes LLMs more Generalized Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Accepted to KDD 2025 Research Track
    </div>
    <details class="paper-abstract">
      Despite significant advancements in Large Language Models (LLMs), developing advanced reasoning capabilities in LLMs remains a key challenge. Process Reward Models (PRMs) have demonstrated exceptional promise in enhancing reasoning by providing step-wise feedback, particularly in the context of mathematical reasoning. However, their application to broader reasoning domains remains understudied, largely due to the high costs associated with manually creating step-level supervision. In this work, we explore the potential of PRMs in graph reasoning problems - a domain that demands sophisticated multi-step reasoning and offers opportunities for automated step-level data generation using established graph algorithms. We introduce GraphSILO, the largest dataset for graph reasoning problems with fine-grained step-wise labels, built using automated Task-oriented Trajectories and Monte Carlo Tree Search (MCTS) to generate detailed reasoning steps with step-wise labels. Building upon this dataset, we train GraphPRM, the first PRM designed for graph reasoning problems, and evaluate its effectiveness in two key settings: inference-time scaling and reinforcement learning via Direct Preference Optimization (DPO). Experimental results show that GraphPRM significantly improves LLM performance across 13 graph reasoning tasks, delivering a 9% gain for Qwen2.5-7B and demonstrating transferability to new graph reasoning datasets and new reasoning domains like mathematical problem-solving. Notably, GraphPRM enhances LLM performance on GSM8K and Math500, underscoring the cross-domain applicability of graph-based reasoning rewards. Our findings highlight the potential of PRMs in advancing reasoning across diverse domains, paving the way for more versatile and effective LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20128v1">CCRS: A Zero-Shot LLM-as-a-Judge Framework for Comprehensive RAG Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Accepted at LLM4Eval @ SIGIR 2025
    </div>
    <details class="paper-abstract">
      RAG systems enhance LLMs by incorporating external knowledge, which is crucial for domains that demand factual accuracy and up-to-date information. However, evaluating the multifaceted quality of RAG outputs, spanning aspects such as contextual coherence, query relevance, factual correctness, and informational completeness, poses significant challenges. Existing evaluation methods often rely on simple lexical overlap metrics, which are inadequate for capturing these nuances, or involve complex multi-stage pipelines with intermediate steps like claim extraction or require finetuning specialized judge models, hindering practical efficiency. To address these limitations, we propose CCRS (Contextual Coherence and Relevance Score), a novel suite of five metrics that utilizes a single, powerful, pretrained LLM as a zero-shot, end-to-end judge. CCRS evaluates: Contextual Coherence (CC), Question Relevance (QR), Information Density (ID), Answer Correctness (AC), and Information Recall (IR). We apply CCRS to evaluate six diverse RAG system configurations on the challenging BioASQ dataset. Our analysis demonstrates that CCRS effectively discriminates between system performances, confirming, for instance, that the Mistral-7B reader outperforms Llama variants. We provide a detailed analysis of CCRS metric properties, including score distributions, convergent/discriminant validity, tie rates, population statistics, and discriminative power. Compared to the complex RAGChecker framework, CCRS offers comparable or superior discriminative power for key aspects like recall and faithfulness, while being significantly more computationally efficient. CCRS thus provides a practical, comprehensive, and efficient framework for evaluating and iteratively improving RAG systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19262v2">What Matters in LLM-generated Data: Diversity and Its Effect on Model Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Ongoing work
    </div>
    <details class="paper-abstract">
      With the remarkable generative capabilities of large language models (LLMs), using LLM-generated data to train downstream models has emerged as a promising approach to mitigate data scarcity in specific domains and reduce time-consuming annotations. However, recent studies have highlighted a critical issue: iterative training on self-generated data results in model collapse, where model performance degrades over time. Despite extensive research on the implications of LLM-generated data, these works often neglect the importance of data diversity, a key factor in data quality. In this work, we aim to understand the implications of the diversity of LLM-generated data on downstream model performance. Specifically, we explore how varying levels of diversity in LLM-generated data affect downstream model performance. Additionally, we investigate the performance of models trained on data that mixes different proportions of LLM-generated data, which we refer to as synthetic data. Our experimental results show that, with minimal distribution shift, moderately diverse LLM-generated data can enhance model performance in scenarios with insufficient labeled data, whereas highly diverse generated data has a negative impact. We hope our empirical findings will offer valuable guidance for future studies on LLMs as data generators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13819v2">LLM-Empowered IoT for 6G Networks: Architecture, Challenges, and Solutions</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Accepted for publication in IEEE Internet of Things Magazine. \c{opyright} 2025 IEEE. The final version will be published in IEEE Xplore
    </div>
    <details class="paper-abstract">
      The Internet of Things (IoT) in the sixth generation (6G) era is envisioned to evolve towards intelligence, ubiquity, and self-optimization. Large language models (LLMs) have demonstrated remarkable generalization capabilities across diverse domains, including natural language processing (NLP), computer vision (CV), and beyond. In this article, we propose an LLM-empowered IoT architecture for 6G networks to achieve intelligent autonomy while supporting advanced IoT applications. LLMs are pushed to the edge of the 6G network to support the synergy of LLMs and IoT. LLM solutions are tailored to both IoT application requirements and IoT management needs, i.e., LLM for IoT. On the other hand, edge inference and edge fine-tuning are discussed to support the deployment of LLMs, i.e., LLM on IoT. Furthermore, we propose a memory-efficient split federated learning (SFL) framework for LLM fine-tuning on heterogeneous IoT devices that alleviates memory pressures on both IoT devices and the edge server while achieving comparable performance and convergence time. Finally, a case study is presented, followed by a discussion about open issues of LLM-empowered IoT for 6G networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.02244v2">Therapy as an NLP Task: Psychologists' Comparison of LLMs and Human Peers in CBT</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are being used as ad-hoc therapists. Research suggests that LLMs outperform human counselors when generating a single, isolated empathetic response; however, their session-level behavior remains understudied. In this study, we compare the session-level behaviors of human counselors with those of an LLM prompted by a team of peer counselors to deliver single-session Cognitive Behavioral Therapy (CBT). Our three-stage, mixed-methods study involved: a) a year-long ethnography of a text-based support platform where seven counselors iteratively refined CBT prompts through self-counseling and weekly focus groups; b) the manual simulation of human counselor sessions with a CBT-prompted LLM, given the full patient dialogue and contextual notes; and c) session evaluations of both human and LLM sessions by three licensed clinical psychologists using CBT competence measures. Our results show a clear trade-off. Human counselors excel at relational strategies -- small talk, self-disclosure, and culturally situated language -- that lead to higher empathy, collaboration, and deeper user reflection. LLM counselors demonstrate higher procedural adherence to CBT techniques but struggle to sustain collaboration, misread cultural cues, and sometimes produce "deceptive empathy," i.e., formulaic warmth that can inflate users' expectations of genuine human care. Taken together, our findings imply that while LLMs might outperform counselors in generating single empathetic responses, their ability to lead sessions is more limited, highlighting that therapy cannot be reduced to a standalone natural language processing (NLP) task. We call for carefully designed human-AI workflows in scalable support: LLMs can scaffold evidence-based techniques, while peers provide relational support. We conclude by mapping concrete design opportunities and ethical guardrails for such hybrid systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19028v2">Quantifying Fairness in LLMs Beyond Tokens: A Semantic and Statistical Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 29 pages, 9 figures, 15 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate responses with inherent biases, undermining their reliability in real-world applications. Existing evaluation methods often overlook biases in long-form responses and the intrinsic variability of LLM outputs. To address these challenges, we propose FiSCo(Fine-grained Semantic Computation), a novel statistical framework to evaluate group-level fairness in LLMs by detecting subtle semantic differences in long-form responses across demographic groups. Unlike prior work focusing on sentiment or token-level comparisons, FiSCo goes beyond surface-level analysis by operating at the claim level, leveraging entailment checks to assess the consistency of meaning across responses. We decompose model outputs into semantically distinct claims and apply statistical hypothesis testing to compare inter- and intra-group similarities, enabling robust detection of subtle biases. We formalize a new group counterfactual fairness definition and validate FiSCo on both synthetic and human-annotated datasets spanning gender, race, and age. Experiments show that FiSco more reliably identifies nuanced biases while reducing the impact of stochastic LLM variability, outperforming various evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08400v2">mSTEB: Massively Multilingual Evaluation of LLMs on Speech and Text Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 working paper
    </div>
    <details class="paper-abstract">
      Large Language models (LLMs) have demonstrated impressive performance on a wide range of tasks, including in multimodal settings such as speech. However, their evaluation is often limited to English and a few high-resource languages. For low-resource languages, there is no standardized evaluation benchmark. In this paper, we address this gap by introducing mSTEB, a new benchmark to evaluate the performance of LLMs on a wide range of tasks covering language identification, text classification, question answering, and translation tasks on both speech and text modalities. We evaluated the performance of leading LLMs such as Gemini 2.0 Flash and GPT-4o (Audio) and state-of-the-art open models such as Qwen 2 Audio and Gemma 3 27B. Our evaluation shows a wide gap in performance between high-resource and low-resource languages, especially for languages spoken in Africa and Americas/Oceania. Our findings show that more investment is needed to address their under-representation in LLMs coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20073v1">A Modular Multitask Reasoning Framework Integrating Spatio-temporal Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Spatio-temporal data mining plays a pivotal role in informed decision making across diverse domains. However, existing models are often restricted to narrow tasks, lacking the capacity for multi-task inference and complex long-form reasoning that require generation of in-depth, explanatory outputs. These limitations restrict their applicability to real-world, multi-faceted decision scenarios. In this work, we introduce STReason, a novel framework that integrates the reasoning strengths of large language models (LLMs) with the analytical capabilities of spatio-temporal models for multi-task inference and execution. Without requiring task-specific finetuning, STReason leverages in-context learning to decompose complex natural language queries into modular, interpretable programs, which are then systematically executed to generate both solutions and detailed rationales. To facilitate rigorous evaluation, we construct a new benchmark dataset and propose a unified evaluation framework with metrics specifically designed for long-form spatio-temporal reasoning. Experimental results show that STReason significantly outperforms advanced LLM baselines across all metrics, particularly excelling in complex, reasoning-intensive spatio-temporal scenarios. Human evaluations further validate STReason's credibility and practical utility, demonstrating its potential to reduce expert workload and broaden the applicability to real-world spatio-temporal tasks. We believe STReason provides a promising direction for developing more capable and generalizable spatio-temporal reasoning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13305v3">Computation Mechanism Behind LLM Position Generalization</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 ACL 2025 Main Long Paper
    </div>
    <details class="paper-abstract">
      Most written natural languages are composed of sequences of words and sentences. Similar to humans, large language models (LLMs) exhibit flexibility in handling textual positions - a phenomenon we term position generalization. They can understand texts with position perturbations and generalize to longer texts than those encountered during training with the latest techniques. These phenomena suggest that LLMs handle positions tolerantly, but how LLMs computationally process positional relevance remains largely unexplored. This work connects the linguistic phenomenon with LLMs' computational mechanisms. We show how LLMs enforce certain computational mechanisms for the aforementioned tolerance in position perturbations. Despite the complex design of the self-attention mechanism, this work reveals that LLMs learn a counterintuitive disentanglement of attention logits. Their values show a 0.959 linear correlation with an approximation of the arithmetic sum of positional relevance and semantic importance. Furthermore, we identify a prevalent pattern in intermediate features, which we prove theoretically enables this effect. The pattern, which is different from how randomly initialized parameters would behave, suggests that it is a learned behavior rather than a natural result of the model architecture. Based on these findings, we provide computational explanations and criteria for LLMs' position flexibilities. This work takes a pioneering step in linking position generalization with modern LLMs' internal mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15928v2">Exploring Big Five Personality and AI Capability Effects in LLM-Simulated Negotiation Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Under review for KDD 2025 Workshop on Evaluation and Trustworthiness of Agentic and Generative AI Models
    </div>
    <details class="paper-abstract">
      This paper presents an evaluation framework for agentic AI systems in mission-critical negotiation contexts, addressing the need for AI agents that can adapt to diverse human operators and stakeholders. Using Sotopia as a simulation testbed, we present two experiments that systematically evaluated how personality traits and AI agent characteristics influence LLM-simulated social negotiation outcomes--a capability essential for a variety of applications involving cross-team coordination and civil-military interactions. Experiment 1 employs causal discovery methods to measure how personality traits impact price bargaining negotiations, through which we found that Agreeableness and Extraversion significantly affect believability, goal achievement, and knowledge acquisition outcomes. Sociocognitive lexical measures extracted from team communications detected fine-grained differences in agents' empathic communication, moral foundations, and opinion patterns, providing actionable insights for agentic AI systems that must operate reliably in high-stakes operational scenarios. Experiment 2 evaluates human-AI job negotiations by manipulating both simulated human personality and AI system characteristics, specifically transparency, competence, adaptability, demonstrating how AI agent trustworthiness impact mission effectiveness. These findings establish a repeatable evaluation methodology for experimenting with AI agent reliability across diverse operator personalities and human-agent team dynamics, directly supporting operational requirements for reliable AI systems. Our work advances the evaluation of agentic AI workflows by moving beyond standard performance metrics to incorporate social dynamics essential for mission success in complex operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20886v1">Omniwise: Predicting GPU Kernels Performance with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      In recent years, the rapid advancement of deep neural networks (DNNs) has revolutionized artificial intelligence, enabling models with unprecedented capabilities in understanding, generating, and processing complex data. These powerful architectures have transformed a wide range of downstream applications, tackling tasks beyond human reach. In this paper, we introduce Omniwise, the first end-to-end, self-supervised fine-tuning pipeline that applies large language models (LLMs) to GPU kernel performance prediction--a novel use case in performance profiling. Omniwise is model-agnostic and lightweight, achieving strong results even with a small 3B-parameter model. It can predict key performance metrics, including memory bandwidth, cache hit rates, GFLOPs, and arithmetic intensity, directly from kernel code without the need for code execution or profiling tools. Our approach achieves over 90% of predictions within 10% relative error on GPU kernels executed on AMD MI250 and MI300X architectures. In addition to the pipeline, we develop an online inference server and a Visual Studio Code plugin that seamlessly integrate LLM-based performance prediction into developers' workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20856v1">Leaner Training, Lower Leakage: Revisiting Memorization in LLM Fine-Tuning with LoRA</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Memorization in large language models (LLMs) makes them vulnerable to data extraction attacks. While pre-training memorization has been extensively studied, fewer works have explored its impact in fine-tuning, particularly for LoRA fine-tuning, a widely adopted parameter-efficient method. In this work, we re-examine memorization in fine-tuning and uncover a surprising divergence from prior findings across different fine-tuning strategies. Factors such as model scale and data duplication, which strongly influence memorization in pre-training and full fine-tuning, do not follow the same trend in LoRA fine-tuning. Using a more relaxed similarity-based memorization metric, we demonstrate that LoRA significantly reduces memorization risks compared to full fine-tuning, while still maintaining strong task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20822v1">Uncovering Hidden Violent Tendencies in LLMs: A Demographic Analysis via Behavioral Vignettes</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly proposed for detecting and responding to violent content online, yet their ability to reason about morally ambiguous, real-world scenarios remains underexamined. We present the first study to evaluate LLMs using a validated social science instrument designed to measure human response to everyday conflict, namely the Violent Behavior Vignette Questionnaire (VBVQ). To assess potential bias, we introduce persona-based prompting that varies race, age, and geographic identity within the United States. Six LLMs developed across different geopolitical and organizational contexts are evaluated under a unified zero-shot setting. Our study reveals two key findings: (1) LLMs surface-level text generation often diverges from their internal preference for violent responses; (2) their violent tendencies vary across demographics, frequently contradicting established findings in criminology, social science, and psychology.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20807v1">GPU Kernel Scientist: An LLM-Driven Framework for Iterative Kernel Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 4 page paper plus Appendices. Accepted to the ES-FoMo "Efficient Systems for Foundation Models" workshop at ICML 2025
    </div>
    <details class="paper-abstract">
      Optimizing GPU kernels for high performance is a complex task, often demanding deep architectural knowledge, extensive profiling, and iterative experimentation. This challenge is amplified when targeting newer or less-documented GPU architectures where traditional development aids are scarce. This paper introduces an LLM-powered "GPU Kernel Scientist," an automated methodology for iteratively refining accelerator kernels. Our methodology employs LLMs in a multi-stage, evolutionary process: (a) strategically selecting promising prior code versions as a basis for new iterations; (b) generating hypotheses for optimization experiments, based on existing code and assimilated knowledge from general GPU literature; and (c) autonomously implementing these experiments through code modification and subsequent submission to an external evaluation system, using only observed timing data as performance feedback. We detail how this approach navigates the challenges of the AMD MI300 target architecture and leverages LLMs to compensate for limited domain-specific human expertise. Since quantitative results from an ongoing performance competition were embargoed on paper submission date, we present the architectural design, operational workflow, and qualitative insights, highlighting the potential of LLM-driven agents to democratise and accelerate GPU kernel optimization, especially in resource-constrained or rapidly evolving hardware environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20803v1">The Ideation-Execution Gap: Execution Outcomes of LLM-Generated versus Human Research Ideas</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 main paper is 14 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise in accelerating the scientific research pipeline. A key capability for this process is the ability to generate novel research ideas, and prior studies have found settings in which LLM-generated research ideas were judged as more novel than human-expert ideas. However, a good idea should not simply appear to be novel, it should also result in better research after being executed. To test whether AI-generated ideas lead to better research outcomes, we conduct an execution study by recruiting 43 expert researchers to execute randomly-assigned ideas, either written by experts or generated by an LLM. Each expert spent over 100 hours implementing the idea and wrote a 4-page short paper to document the experiments. All the executed projects are then reviewed blindly by expert NLP researchers. Comparing the review scores of the same ideas before and after execution, the scores of the LLM-generated ideas decrease significantly more than expert-written ideas on all evaluation metrics (novelty, excitement, effectiveness, and overall; p < 0.05), closing the gap between LLM and human ideas observed at the ideation stage. When comparing the aggregated review scores from the execution study, we even observe that for many metrics there is a flip in rankings where human ideas score higher than LLM ideas. This ideation-execution gap highlights the limitations of current LLMs in generating truly effective research ideas and the challenge of evaluating research ideas in the absence of execution outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.14133v2">GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 38 pages, 8 tables, 18 figures
    </div>
    <details class="paper-abstract">
      LLMs have shown impressive capabilities across various natural language processing tasks, yet remain vulnerable to input prompts, known as jailbreak attacks, carefully designed to bypass safety guardrails and elicit harmful responses. Traditional methods rely on manual heuristics but suffer from limited generalizability. Despite being automatic, optimization-based attacks often produce unnatural prompts that can be easily detected by safety filters or require high computational costs due to discrete token optimization. In this paper, we introduce Generative Adversarial Suffix Prompter (GASP), a novel automated framework that can efficiently generate human-readable jailbreak prompts in a fully black-box setting. In particular, GASP leverages latent Bayesian optimization to craft adversarial suffixes by efficiently exploring continuous latent embedding spaces, gradually optimizing the suffix prompter to improve attack efficacy while balancing prompt coherence via a targeted iterative refinement procedure. Through comprehensive experiments, we show that GASP can produce natural adversarial prompts, significantly improving jailbreak success over baselines, reducing training times, and accelerating inference speed, thus making it an efficient and scalable solution for red-teaming LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05199v3">CodeLutra: Boosting LLM Code Generation via Preference-Guided Refinement</a></div>
    <div class="paper-meta">
      📅 2025-06-25
      | 💬 TMLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized code generation but require significant resources and often over-generalize, limiting their task-specific efficiency. Fine-tuning smaller, open-source LLMs provides a cost-effective alternative. However, standard supervised approaches rely only on correct examples, missing valuable insights from failures. We introduce CodeLutra, a framework that leverages both correct and incorrect code attempts. Instead of using only correct solutions, CodeLutra applies iterative preference-based refinement, comparing successful and failed outputs to better approximate desired results. This approach narrows the performance gap with state-of-the-art larger models without requiring massive datasets or auxiliary models. For instance, on a challenging data science coding task, using only 500 samples improved Llama-3-8B's accuracy from 28.2% to 48.6%, approaching GPT-4's level. By learning from both successes and mistakes, CodeLutra provides a scalable and efficient path to high-quality code generation, making smaller open-source models more competitive with leading closed-source alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.20743v1">A Survey of AI for Materials Science: Foundation Models, LLM Agents, Datasets, and Tools</a></div>
    <div class="paper-meta">
      📅 2025-06-25
    </div>
    <details class="paper-abstract">
      Foundation models (FMs) are catalyzing a transformative shift in materials science (MatSci) by enabling scalable, general-purpose, and multimodal AI systems for scientific discovery. Unlike traditional machine learning models, which are typically narrow in scope and require task-specific engineering, FMs offer cross-domain generalization and exhibit emergent capabilities. Their versatility is especially well-suited to materials science, where research challenges span diverse data types and scales. This survey provides a comprehensive overview of foundation models, agentic systems, datasets, and computational tools supporting this growing field. We introduce a task-driven taxonomy encompassing six broad application areas: data extraction, interpretation and Q\&A; atomistic simulation; property prediction; materials structure, design and discovery; process planning, discovery, and optimization; and multiscale modeling. We discuss recent advances in both unimodal and multimodal FMs, as well as emerging large language model (LLM) agents. Furthermore, we review standardized datasets, open-source tools, and autonomous experimental platforms that collectively fuel the development and integration of FMs into research workflows. We assess the early successes of foundation models and identify persistent limitations, including challenges in generalizability, interpretability, data imbalance, safety concerns, and limited multimodal fusion. Finally, we articulate future research directions centered on scalable pretraining, continual learning, data governance, and trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19846v1">JoyAgents-R1: Joint Evolution Dynamics for Versatile Multi-LLM Agents with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 33 pages, 7 figures, under review
    </div>
    <details class="paper-abstract">
      Multi-agent reinforcement learning (MARL) has emerged as a prominent paradigm for increasingly complex tasks. However, joint evolution across heterogeneous agents remains challenging due to cooperative inefficiency and training instability. In this paper, we propose the joint evolution dynamics for MARL called JoyAgents-R1, which first applies Group Relative Policy Optimization (GRPO) to the joint training of heterogeneous multi-agents. By iteratively refining agents' large language models (LLMs) and memories, the method achieves holistic equilibrium with optimal decision-making and memory capabilities. Specifically, JoyAgents-R1 first implements node-wise Monte Carlo sampling on the behavior of each agent across entire reasoning trajectories to enhance GRPO sampling efficiency while maintaining policy diversity. Then, our marginal benefit-driven selection strategy identifies top-$K$ sampling groups with maximal reward fluctuations, enabling targeted agent model updates that improve training stability and maximize joint benefits through cost-effective parameter adjustments. Meanwhile, JoyAgents-R1 introduces an adaptive memory evolution mechanism that repurposes GRPO rewards as cost-free supervisory signals to eliminate repetitive reasoning and accelerate convergence. Experiments across general and domain-specific scenarios demonstrate that JoyAgents-R1 achieves performance comparable to that of larger LLMs while built on smaller open-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19806v1">LLM-Based Social Simulations Require a Boundary</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      This position paper argues that large language model (LLM)-based social simulations should establish clear boundaries to meaningfully contribute to social science research. While LLMs offer promising capabilities for modeling human-like agents compared to traditional agent-based modeling, they face fundamental limitations that constrain their reliability for social pattern discovery. The core issue lies in LLMs' tendency towards an ``average persona'' that lacks sufficient behavioral heterogeneity, a critical requirement for simulating complex social dynamics. We examine three key boundary problems: alignment (simulated behaviors matching real-world patterns), consistency (maintaining coherent agent behavior over time), and robustness (reproducibility under varying conditions). We propose heuristic boundaries for determining when LLM-based simulations can reliably advance social science understanding. We believe that these simulations are more valuable when focusing on (1) collective patterns rather than individual trajectories, (2) agent behaviors aligning with real population averages despite limited variance, and (3) proper validation methods available for testing simulation robustness. We provide a practical checklist to guide researchers in determining the appropriate scope and claims for LLM-based social simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19794v1">Why Do Open-Source LLMs Struggle with Data Analysis? A Systematic Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) hold promise in automating data analysis tasks, yet open-source models face significant limitations in these kinds of reasoning-intensive scenarios. In this work, we investigate strategies to enhance the data analysis capabilities of open-source LLMs. By curating a seed dataset of diverse, realistic scenarios, we evaluate models across three dimensions: data understanding, code generation, and strategic planning. Our analysis reveals three key findings: (1) Strategic planning quality serves as the primary determinant of model performance; (2) Interaction design and task complexity significantly influence reasoning capabilities; (3) Data quality demonstrates a greater impact than diversity in achieving optimal performance. We leverage these insights to develop a data synthesis methodology, demonstrating significant improvements in open-source LLMs' analytical reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19775v1">Canary in the Mine: An LLM Augmented Survey of Disciplinary Complaints to the Ordre des ingénieurs du Québec (OIQ)</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 22 pages, accepted at the American Society of Engineering Education annual conference 2025, pre-print
    </div>
    <details class="paper-abstract">
      This study uses pre-trained LLMs to conduct thematic analysis to investigate disciplinary incidents involving engineers in Quebec, shedding light on critical gaps in engineering education. Through a comprehensive review of the disciplinary register of the Ordre des ing\'enieurs du Qu\'ebec (OIQ)'s disciplinary register for 2010 to 2024, researchers from engineering education and human resources management in technological development laboratories conducted a thematic analysis of reported incidents to identify patterns, trends, and areas for improvement. The analysis aims to uncover the most common types of disciplinary incidents, underlying causes, and implications for the field in how engineering education addresses (or fails to address) these issues. Our findings identify recurring themes, analyze root causes, and offer recommendations for engineering educators and students to mitigate similar incidents. This research has implications for informing curriculum development, professional development, and performance evaluation, ultimately fostering a culture of professionalism and ethical responsibility in engineering. By providing empirical evidence of disciplinary incidents and their causes, this study contributes to evidence-based practices for engineering education and professional development, enhancing the engineering education community's understanding of professionalism and ethics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20185v2">DecDEC: A Systems Approach to Advancing Low-Bit LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 OSDI 2025
    </div>
    <details class="paper-abstract">
      Quantization of Large Language Models (LLMs) has recently gained popularity, particularly for on-device settings with limited hardware resources. While efficient, quantization inevitably degrades model quality, especially in aggressive low-bit settings such as 3-bit and 4-bit precision. In this paper, we propose DecDEC, an inference scheme that improves the quality of low-bit LLMs while preserving the key benefits of quantization: GPU memory savings and latency reduction. DecDEC stores the residual matrix -- the difference between full-precision and quantized weights -- in CPU, and dynamically fetches the residuals for only a small portion of the weights. This portion corresponds to the salient channels, marked by activation outliers, with the fetched residuals helping to correct quantization errors in these channels. Salient channels are identified dynamically at each decoding step by analyzing the input activations -- this enables adaptation to the dynamic nature of activation distribution, thus maximizing the effectiveness of error compensation. We demonstrate the effectiveness of DecDEC by augmenting state-of-the-art quantization methods. For example, DecDEC reduces the perplexity of a 3-bit Llama-3-8B-Instruct model from 10.15 to 9.12 -- outperforming its 3.5-bit counterpart -- while adding less than 0.0003\% to GPU memory usage and incurring only a 1.7\% inference slowdown on NVIDIA RTX 4050 Mobile.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12743v2">"I know myself better, but not really greatly": How Well Can LLMs Detect and Explain LLM-Generated Texts?</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Distinguishing between human- and LLM-generated texts is crucial given the risks associated with misuse of LLMs. This paper investigates detection and explanation capabilities of current LLMs across two settings: binary (human vs. LLM-generated) and ternary classification (including an ``undecided'' class). We evaluate 6 close- and open-source LLMs of varying sizes and find that self-detection (LLMs identifying their own outputs) consistently outperforms cross-detection (identifying outputs from other LLMs), though both remain suboptimal. Introducing a ternary classification framework improves both detection accuracy and explanation quality across all models. Through comprehensive quantitative and qualitative analyses using our human-annotated dataset, we identify key explanation failures, primarily reliance on inaccurate features, hallucinations, and flawed reasoning. Our findings underscore the limitations of current LLMs in self-detection and self-explanation, highlighting the need for further research to address overfitting and enhance generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19702v1">LLM-Driven Medical Document Analysis: Enhancing Trustworthy Pathology and Differential Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 Accepted at ICDAR 2025
    </div>
    <details class="paper-abstract">
      Medical document analysis plays a crucial role in extracting essential clinical insights from unstructured healthcare records, supporting critical tasks such as differential diagnosis. Determining the most probable condition among overlapping symptoms requires precise evaluation and deep medical expertise. While recent advancements in large language models (LLMs) have significantly enhanced performance in medical document analysis, privacy concerns related to sensitive patient data limit the use of online LLMs services in clinical settings. To address these challenges, we propose a trustworthy medical document analysis platform that fine-tunes a LLaMA-v3 using low-rank adaptation, specifically optimized for differential diagnosis tasks. Our approach utilizes DDXPlus, the largest benchmark dataset for differential diagnosis, and demonstrates superior performance in pathology prediction and variable-length differential diagnosis compared to existing methods. The developed web-based platform allows users to submit their own unstructured medical documents and receive accurate, explainable diagnostic results. By incorporating advanced explainability techniques, the system ensures transparent and reliable predictions, fostering user trust and confidence. Extensive evaluations confirm that the proposed method surpasses current state-of-the-art models in predictive accuracy while offering practical utility in clinical settings. This work addresses the urgent need for reliable, explainable, and privacy-preserving artificial intelligence solutions, representing a significant advancement in intelligent medical document analysis for real-world healthcare applications. The code can be found at \href{https://github.com/leitro/Differential-Diagnosis-LoRA}{https://github.com/leitro/Differential-Diagnosis-LoRA}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15196v2">HeurAgenix: Leveraging LLMs for Solving Complex Combinatorial Optimization Challenges</a></div>
    <div class="paper-meta">
      📅 2025-06-24
      | 💬 27 pages,9 figures
    </div>
    <details class="paper-abstract">
      Heuristic algorithms play a vital role in solving combinatorial optimization (CO) problems, yet traditional designs depend heavily on manual expertise and struggle to generalize across diverse instances. We introduce \textbf{HeurAgenix}, a two-stage hyper-heuristic framework powered by large language models (LLMs) that first evolves heuristics and then selects among them automatically. In the heuristic evolution phase, HeurAgenix leverages an LLM to compare seed heuristic solutions with higher-quality solutions and extract reusable evolution strategies. During problem solving, it dynamically picks the most promising heuristic for each problem state, guided by the LLM's perception ability. For flexibility, this selector can be either a state-of-the-art LLM or a fine-tuned lightweight model with lower inference cost. To mitigate the scarcity of reliable supervision caused by CO complexity, we fine-tune the lightweight heuristic selector with a dual-reward mechanism that jointly exploits singals from selection preferences and state perception, enabling robust selection under noisy annotations. Extensive experiments on canonical benchmarks show that HeurAgenix not only outperforms existing LLM-based hyper-heuristics but also matches or exceeds specialized solvers. Code is available at https://github.com/microsoft/HeurAgenix.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19676v1">A Survey of LLM-Driven AI Agent Communication: Protocols, Security Risks, and Defense Countermeasures</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      In recent years, Large-Language-Model-driven AI agents have exhibited unprecedented intelligence, flexibility, and adaptability, and are rapidly changing human production and lifestyle. Nowadays, agents are undergoing a new round of evolution. They no longer act as an isolated island like LLMs. Instead, they start to communicate with diverse external entities, such as other agents and tools, to collectively perform more complex tasks. Under this trend, agent communication is regarded as a foundational pillar of the future AI ecosystem, and many organizations intensively begin to design related communication protocols (e.g., Anthropic's MCP and Google's A2A) within the recent few months. However, this new field exposes significant security hazard, which can cause severe damage to real-world scenarios. To help researchers to quickly figure out this promising topic and benefit the future agent communication development, this paper presents a comprehensive survey of agent communication security. More precisely, we first present a clear definition of agent communication and categorize the entire lifecyle of agent communication into three stages: user-agent interaction, agent-agent communication, and agent-environment communication. Next, for each communication phase, we dissect related protocols and analyze its security risks according to the communication characteristics. Then, we summarize and outlook on the possible defense countermeasures for each risk. Finally, we discuss open issues and future directions in this promising research field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19652v1">Tailored Conversations beyond LLMs: A RL-Based Dialogue Manager</a></div>
    <div class="paper-meta">
      📅 2025-06-24
    </div>
    <details class="paper-abstract">
      In this work, we propose a novel framework that integrates large language models (LLMs) with an RL-based dialogue manager for open-ended dialogue with a specific goal. By leveraging hierarchical reinforcement learning to model the structured phases of dialogue and employ meta-learning to enhance adaptability across diverse user profiles, our approach enhances adaptability and efficiency, enabling the system to learn from limited data, transition fluidly between dialogue phases, and personalize responses to heterogeneous patient needs. We apply our framework to Motivational Interviews, aiming to foster behavior change, and demonstrate that the proposed dialogue manager outperforms a state-of-the-art LLM baseline in terms of reward, showing a potential benefit of conditioning LLMs to create open-ended dialogue systems with specific goals.
    </details>
</div>
