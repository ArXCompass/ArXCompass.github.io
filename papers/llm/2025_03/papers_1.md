# llm - 2025_03

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21080v3">EQ-Negotiator: An Emotion-Reasoning LLM Agent in Credit Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      While large language model (LLM)-based chatbots have been applied for effective engagement in credit dialogues, their capacity for dynamic emotional expression remains limited. Current agents primarily rely on passive empathy rather than affective reasoning. For instance, when faced with persistent client negativity, the agent should employ strategic emotional adaptation by expressing measured anger to discourage counterproductive behavior and guide the conversation toward resolution. This context-aware emotional modulation is essential for imitating the nuanced decision-making of human negotiators. This paper introduces an EQ-negotiator that combines emotion sensing from pre-trained language models (PLMs) with emotional reasoning based on Game Theory and Hidden Markov Models. It takes into account both the current and historical emotions of the client to better manage and address negative emotions during interactions. By fine-tuning pre-trained language models (PLMs) on public emotion datasets and validating them on the credit dialogue datasets, our approach enables LLM-based agents to effectively capture shifts in client emotions and dynamically adjust their response tone based on our emotion decision policies in real-world financial negotiations. This EQ-negotiator can also help credit agencies foster positive client relationships, enhancing satisfaction in credit services.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.24307v1">A Systematic Evaluation of LLM Strategies for Mental Health Text Analysis: Fine-tuning vs. Prompt Engineering vs. RAG</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      This study presents a systematic comparison of three approaches for the analysis of mental health text using large language models (LLMs): prompt engineering, retrieval augmented generation (RAG), and fine-tuning. Using LLaMA 3, we evaluate these approaches on emotion classification and mental health condition detection tasks across two datasets. Fine-tuning achieves the highest accuracy (91% for emotion classification, 80% for mental health conditions) but requires substantial computational resources and large training sets, while prompt engineering and RAG offer more flexible deployment with moderate performance (40-68% accuracy). Our findings provide practical insights for implementing LLM-based solutions in mental health applications, highlighting the trade-offs between accuracy, computational requirements, and deployment flexibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09433v2">CASTLE: Benchmarking Dataset for Static Code Analyzers and LLMs towards CWE Detection</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      Identifying vulnerabilities in source code is crucial, especially in critical software components. Existing methods such as static analysis, dynamic analysis, formal verification, and recently Large Language Models are widely used to detect security flaws. This paper introduces CASTLE (CWE Automated Security Testing and Low-Level Evaluation), a benchmarking framework for evaluating the vulnerability detection capabilities of different methods. We assess 13 static analysis tools, 10 LLMs, and 2 formal verification tools using a hand-crafted dataset of 250 micro-benchmark programs covering 25 common CWEs. We propose the CASTLE Score, a novel evaluation metric to ensure fair comparison. Our results reveal key differences: ESBMC (a formal verification tool) minimizes false positives but struggles with vulnerabilities beyond model checking, such as weak cryptography or SQL injection. Static analyzers suffer from high false positives, increasing manual validation efforts for developers. LLMs perform exceptionally well in the CASTLE dataset when identifying vulnerabilities in small code snippets. However, their accuracy declines, and hallucinations increase as the code size grows. These results suggest that LLMs could play a pivotal role in future security solutions, particularly within code completion frameworks, where they can provide real-time guidance to prevent vulnerabilities. The dataset is accessible at https://github.com/CASTLE-Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.24245v1">Enhancing Large Language Models (LLMs) for Telecommunications using Knowledge Graphs and Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-31
      | 💬 This work has been accepted to ICC 2025 IEEE International Conference on Communications. copyright 2025 IEEE
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made significant progress in general-purpose natural language processing tasks. However, LLMs are still facing challenges when applied to domain-specific areas like telecommunications, which demands specialized expertise and adaptability to evolving standards. This paper presents a novel framework that combines knowledge graph (KG) and retrieval-augmented generation (RAG) techniques to enhance LLM performance in the telecom domain. The framework leverages a KG to capture structured, domain-specific information about network protocols, standards, and other telecom-related entities, comprehensively representing their relationships. By integrating KG with RAG, LLMs can dynamically access and utilize the most relevant and up-to-date knowledge during response generation. This hybrid approach bridges the gap between structured knowledge representation and the generative capabilities of LLMs, significantly enhancing accuracy, adaptability, and domain-specific comprehension. Our results demonstrate the effectiveness of the KG-RAG framework in addressing complex technical queries with precision. The proposed KG-RAG model attained an accuracy of 88% for question answering tasks on a frequently used telecom-specific dataset, compared to 82% for the RAG-only and 48% for the LLM-only approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.24191v1">Output Constraints as Attack Surface: Exploiting Structured Generation to Bypass LLM Safety Mechanisms</a></div>
    <div class="paper-meta">
      📅 2025-03-31
      | 💬 15 pages, 13 figures, 4 tables Work In Progress
    </div>
    <details class="paper-abstract">
      Content Warning: This paper may contain unsafe or harmful content generated by LLMs that may be offensive to readers. Large Language Models (LLMs) are extensively used as tooling platforms through structured output APIs to ensure syntax compliance so that robust integration with existing softwares like agent systems, could be achieved. However, the feature enabling functionality of grammar-guided structured output presents significant security vulnerabilities. In this work, we reveal a critical control-plane attack surface orthogonal to traditional data-plane vulnerabilities. We introduce Constrained Decoding Attack (CDA), a novel jailbreak class that weaponizes structured output constraints to bypass safety mechanisms. Unlike prior attacks focused on input prompts, CDA operates by embedding malicious intent in schema-level grammar rules (control-plane) while maintaining benign surface prompts (data-plane). We instantiate this with a proof-of-concept Chain Enum Attack, achieves 96.2% attack success rates across proprietary and open-weight LLMs on five safety benchmarks with a single query, including GPT-4o and Gemini-2.0-flash. Our findings identify a critical security blind spot in current LLM architectures and urge a paradigm shift in LLM safety to address control-plane vulnerabilities, as current mechanisms focused solely on data-plane threats leave critical systems exposed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.24102v1">Is LLM the Silver Bullet to Low-Resource Languages Machine Translation?</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      Low-Resource Languages (LRLs) present significant challenges in natural language processing due to their limited linguistic resources and underrepresentation in standard datasets. While recent advancements in Large Language Models (LLMs) and Neural Machine Translation (NMT) have substantially improved translation capabilities for high-resource languages, performance disparities persist for LRLs, particularly impacting privacy-sensitive and resource-constrained scenarios. This paper systematically evaluates the limitations of current LLMs across 200 languages using benchmarks such as FLORES-200. We also explore alternative data sources, including news articles and bilingual dictionaries, and demonstrate how knowledge distillation from large pre-trained models can significantly improve smaller LRL translations. Additionally, we investigate various fine-tuning strategies, revealing that incremental enhancements markedly reduce performance gaps on smaller LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19655v3">Truth or Mirage? Towards End-to-End Factuality Evaluation with LLM-Oasis</a></div>
    <div class="paper-meta">
      📅 2025-03-31
      | 💬 15 pages. To be submitted to CL journal
    </div>
    <details class="paper-abstract">
      After the introduction of Large Language Models (LLMs), there have been substantial improvements in the performance of Natural Language Generation (NLG) tasks, including Text Summarization and Machine Translation. However, LLMs still produce outputs containing hallucinations, that is, content not grounded in factual information. Therefore, developing methods to assess the factuality of LLMs has become urgent. Indeed, resources for factuality evaluation have recently emerged. Although challenging, these resources face one or more of the following limitations: (i) they are tailored to a specific task or domain; (ii) they are limited in size, thereby preventing the training of new factuality evaluators; (iii) they are designed for simpler verification tasks, such as claim verification. To address these issues, we introduce LLM-Oasis, to the best of our knowledge the largest resource for training end-to-end factuality evaluators. LLM-Oasis is constructed by extracting claims from Wikipedia, falsifying a subset of these claims, and generating pairs of factual and unfactual texts. We then rely on human annotators to both validate the quality of our dataset and to create a gold standard test set for benchmarking factuality evaluation systems. Our experiments demonstrate that LLM-Oasis presents a significant challenge for state-of-the-art LLMs, with GPT-4o achieving up to 60% accuracy in our proposed end-to-end factuality evaluation task, highlighting its potential to drive future research in the field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.00326v3">Teola: Towards End-to-End Optimization of LLM-based Applications</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based applications consist of both LLM and non-LLM components, each contributing to the end-to-end latency. Despite great efforts to optimize LLM inference, end-to-end workflow optimization has been overlooked. Existing frameworks employ coarse-grained orchestration with task modules, which confines optimizations to within each module and yields suboptimal scheduling decisions. We propose fine-grained end-to-end orchestration, which utilizes task primitives as the basic units and represents each query's workflow as a primitive-level dataflow graph. This explicitly exposes a much larger design space, enables optimizations in parallelization and pipelining across primitives of different modules, and enhances scheduling to improve application-level performance. We build Teola, a novel orchestration framework for LLM-based applications that implements this scheme. Comprehensive experiments show that Teola can achieve up to 2.09x speedup over existing systems across various popular LLM applications. The code is available at https://github.com/NetX-lab/Ayo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.03095v6">TestART: Improving LLM-based Unit Testing via Co-evolution of Automated Generation and Repair Iteration</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      Unit testing is crucial for detecting bugs in individual program units but consumes time and effort. Recently, large language models (LLMs) have demonstrated remarkable capabilities in generating unit test cases. However, several problems limit their ability to generate high-quality unit test cases: (1) compilation and runtime errors caused by the hallucination of LLMs; (2) lack of testing and coverage feedback information restricting the increase of code coverage;(3) the repetitive suppression problem causing invalid LLM-based repair and generation attempts. To address these limitations, we propose TestART, a novel unit test generation method. TestART improves LLM-based unit testing via co-evolution of automated generation and repair iteration, representing a significant advancement in automated unit test generation. TestART leverages the template-based repair strategy to effectively fix bugs in LLM-generated test cases for the first time. Meanwhile, TestART extracts coverage information from successful test cases and uses it as coverage-guided testing feedback. It also incorporates positive prompt injection to prevent repetition suppression, thereby enhancing the sufficiency of the final test case. This synergy between generation and repair elevates the correctness and sufficiency of the produced test cases significantly beyond previous methods. In comparative experiments, TestART demonstrates an 18% improvement in pass rate and a 20% enhancement in coverage across three types of datasets compared to baseline models. Additionally, it achieves better coverage rates than EvoSuite with only half the number of test cases. These results demonstrate TestART's superior ability to produce high-quality unit test cases by harnessing the power of LLMs while overcoming their inherent flaws.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.24047v1">Towards Scientific Intelligence: A Survey of LLM-based Scientific Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-31
      | 💬 34 pages, 10 figures
    </div>
    <details class="paper-abstract">
      As scientific research becomes increasingly complex, innovative tools are needed to manage vast data, facilitate interdisciplinary collaboration, and accelerate discovery. Large language models (LLMs) are now evolving into LLM-based scientific agents that automate critical tasks, ranging from hypothesis generation and experiment design to data analysis and simulation. Unlike general-purpose LLMs, these specialized agents integrate domain-specific knowledge, advanced tool sets, and robust validation mechanisms, enabling them to handle complex data types, ensure reproducibility, and drive scientific breakthroughs. This survey provides a focused review of the architectures, design, benchmarks, applications, and ethical considerations surrounding LLM-based scientific agents. We highlight why they differ from general agents and the ways in which they advance research across various scientific fields. By examining their development and challenges, this survey offers a comprehensive roadmap for researchers and practitioners to harness these agents for more efficient, reliable, and ethically sound scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06816v2">MAQA: Evaluating Uncertainty Quantification in LLMs Regarding Data Uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-03-31
      | 💬 Findings of NAACL 2025
    </div>
    <details class="paper-abstract">
      Despite the massive advancements in large language models (LLMs), they still suffer from producing plausible but incorrect responses. To improve the reliability of LLMs, recent research has focused on uncertainty quantification to predict whether a response is correct or not. However, most uncertainty quantification methods have been evaluated on single-labeled questions, which removes data uncertainty: the irreducible randomness often present in user queries, which can arise from factors like multiple possible answers. This limitation may cause uncertainty quantification results to be unreliable in practical settings. In this paper, we investigate previous uncertainty quantification methods under the presence of data uncertainty. Our contributions are two-fold: 1) proposing a new Multi-Answer Question Answering dataset, MAQA, consisting of world knowledge, mathematical reasoning, and commonsense reasoning tasks to evaluate uncertainty quantification regarding data uncertainty, and 2) assessing 5 uncertainty quantification methods of diverse white- and black-box LLMs. Our findings show that previous methods relatively struggle compared to single-answer settings, though this varies depending on the task. Moreover, we observe that entropy- and consistency-based methods effectively estimate model uncertainty, even in the presence of data uncertainty. We believe these observations will guide future work on uncertainty quantification in more realistic settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23989v1">Rubric Is All You Need: Enhancing LLM-based Code Evaluation With Question-Specific Rubrics</a></div>
    <div class="paper-meta">
      📅 2025-03-31
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Since the disruption in LLM technology brought about by the release of GPT-3 and ChatGPT, LLMs have shown remarkable promise in programming-related tasks. While code generation remains a popular field of research, code evaluation using LLMs remains a problem with no conclusive solution. In this paper, we focus on LLM-based code evaluation and attempt to fill in the existing gaps. We propose multi-agentic novel approaches using question-specific rubrics tailored to the problem statement, arguing that these perform better for logical assessment than the existing approaches that use question-agnostic rubrics. To address the lack of suitable evaluation datasets, we introduce two datasets: a Data Structures and Algorithms dataset containing 150 student submissions from a popular Data Structures and Algorithms practice website, and an Object Oriented Programming dataset comprising 80 student submissions from undergraduate computer science courses. In addition to using standard metrics (Spearman Correlation, Cohen's Kappa), we additionally propose a new metric called as Leniency, which quantifies evaluation strictness relative to expert assessment. Our comprehensive analysis demonstrates that question-specific rubrics significantly enhance logical assessment of code in educational settings, providing better feedback aligned with instructional goals beyond mere syntactic correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22456v2">Entropy-guided sequence weighting for efficient exploration in RL-based LLM fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      We introduce Entropy-Guided Sequence Weighting (EGSW), a novel approach that enhances the exploration-exploitation tradeoff by dynamically assigning weights to generated outputs based on their advantage and entropy for Reinforcement Learning-based Large Language Model fine-tuning. EGSW integrates entropy regularization with advantage-based weighting to balance policy updates, enabling efficient exploration in high-dimensional state spaces. By employing temperature-scaled softmax weighting over sequences, EGSW prioritizing high-reward, high-uncertainty steps while maintaining training stability. Although originally developed to improve Group Relative Policy Optimization (GRPO) during large language model (LLM) fine-tuning, EGSW is generalizable to other reinforcement learning (RL) algorithms and can be implemented in both step-wise and trajectory-wise settings. Empirical evaluations demonstrate that EGSW enhances GRPO reasoning ability, yielding improvements in sample efficiency. Future work will explore the application of EGSW to advanced RL methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23848v1">SpeechDialogueFactory: Generating High-Quality Speech Dialogue Data to Accelerate Your Speech-LLM Development</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      High-quality speech dialogue datasets are crucial for Speech-LLM development, yet existing acquisition methods face significant limitations. Human recordings incur high costs and privacy concerns, while synthetic approaches often lack conversational authenticity. To address these challenges, we introduce \textsc{SpeechDialogueFactory}, a production-ready framework for generating natural speech dialogues efficiently. Our solution employs a comprehensive pipeline including metadata generation, dialogue scripting, paralinguistic-enriched utterance simulation, and natural speech synthesis with voice cloning. Additionally, the system provides an interactive UI for detailed sample inspection and a high-throughput batch synthesis mode. Evaluations show that dialogues generated by our system achieve a quality comparable to human recordings while significantly reducing production costs. We release our work as an open-source toolkit, alongside example datasets available in English and Chinese, empowering researchers and developers in Speech-LLM research and development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23817v1">MVDRAM: Enabling GeMV Execution in Unmodified DRAM for Low-Bit LLM Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      General matrix-vector multiplication (GeMV) remains a critical latency bottleneck in large language model (LLM) inference, even with quantized low-bit models. Processing-Using-DRAM (PUD), an analog in-DRAM computing technique, has the potential to repurpose on-device DRAM as a GeMV engine, offering additional high-throughput processing capabilities to widespread consumer devices without DRAM modifications. However, applying PUD to GeMV operations in the LLM inference pipeline incurs significant overheads $\textit{before}$ and $\textit{after}$ in-DRAM computation, diminishing the benefits of its high-throughput processing capabilities. This paper presents MVDRAM, the first practical system to accelerate GeMV operations for low-bit LLM inference using unmodified DRAM. By leveraging the data sharing patterns and mathematical linearity in GeMV operations, MVDRAM orchestrates the processor and DRAM to eliminate the costs associated with pre-arranging inputs and bit-transposition of outputs required in conventional PUD approaches. Our experimental evaluation with four DDR4 DRAM modules shows that MVDRAM achieves comparable or even better inference speed than the processor-based implementation for GeMV operations in low-bit (under 4-bit) LLM. In particular, MVDRAM achieves up to 7.29$\times$ speedup and 30.5$\times$ energy efficiency for low-bit GeMV operations. For end-to-end LLM inference, MVDRAM achieves 2.18$\times$ and 1.31$\times$ throughput improvements, along with 3.04$\times$ and 2.35$\times$ energy efficiency, for 2-bit and 4-bit quantized low-bit models, respectively. MVDRAM has the potential to redefine the AI hardware landscape by demonstrating the feasibility of standard DRAM as an LLM accelerator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23798v1">Adaptive Layer-skipping in Pre-trained LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      Various layer-skipping methods have been proposed to accelerate token generation in large language models (LLMs). However, they have overlooked a fundamental question: How do computational demands vary across the generation of different tokens? In this work, we introduce FlexiDepth, a method that dynamically adjusts the number of Transformer layers used in text generation. By incorporating a plug-in router and adapter, FlexiDepth enables adaptive layer-skipping in LLMs without modifying their original parameters. Introducing FlexiDepth to Llama-3-8B model achieves layer skipping of 8 layers out of 32, and meanwhile maintains the full 100\% benchmark performance. Experimental results with FlexiDepth demonstrate that computational demands in LLMs significantly vary based on token type. Specifically, generating repetitive tokens or fixed phrases requires fewer layers, whereas producing tokens involving computation or high uncertainty requires more layers. Interestingly, this adaptive allocation pattern aligns with human intuition. To advance research in this area, we open sourced FlexiDepth and a dataset documenting FlexiDepth's layer allocation patterns for future exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23740v1">LANID: LLM-assisted New Intent Discovery</a></div>
    <div class="paper-meta">
      📅 2025-03-31
      | 💬 Published in LREC-COLING 2024
    </div>
    <details class="paper-abstract">
      Task-oriented Dialogue Systems (TODS) often face the challenge of encountering new intents. New Intent Discovery (NID) is a crucial task that aims to identify these novel intents while maintaining the capability to recognize existing ones. Previous efforts to adapt TODS to new intents have struggled with inadequate semantic representation or have depended on external knowledge, which is often not scalable or flexible. Recently, Large Language Models (LLMs) have demonstrated strong zero-shot capabilities; however, their scale can be impractical for real-world applications that involve extensive queries. To address the limitations of existing NID methods by leveraging LLMs, we propose LANID, a framework that enhances the semantic representation of lightweight NID encoders with the guidance of LLMs. Specifically, LANID employs the $K$-nearest neighbors and Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithms to sample selective utterance pairs from the training set. It then queries an LLM to ascertain the relationships between these pairs. The data produced from this process is utilized to design a contrastive fine-tuning task, which is then used to train a small encoder with a contrastive triplet loss. Our experimental results demonstrate the efficacy of the proposed method across three distinct NID datasets, surpassing strong baselines in both unsupervised and semi-supervised settings. Our code is available at https://github.com/floatSDSDS/LANID.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23718v1">Detecting Functional Bugs in Smart Contracts through LLM-Powered and Bug-Oriented Composite Analysis</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      Smart contracts are fundamental pillars of the blockchain, playing a crucial role in facilitating various business transactions. However, these smart contracts are vulnerable to exploitable bugs that can lead to substantial monetary losses. A recent study reveals that over 80% of these exploitable bugs, which are primarily functional bugs, can evade the detection of current tools. The primary issue is the significant gap between understanding the high-level logic of the business model and checking the low-level implementations in smart contracts. Furthermore, identifying deeply rooted functional bugs in smart contracts requires the automated generation of effective detection oracles based on various bug features. To address these challenges, we design and implement PROMFUZZ, an automated and scalable system to detect functional bugs, in smart contracts. In PROMFUZZ, we first propose a novel Large Language Model (LLM)-driven analysis framework, which leverages a dual-agent prompt engineering strategy to pinpoint potentially vulnerable functions for further scrutiny. We then implement a dual-stage coupling approach, which focuses on generating invariant checkers that leverage logic information extracted from potentially vulnerable functions. Finally, we design a bug-oriented fuzzing engine, which maps the logical information from the high-level business model to the low-level smart contract implementations, and performs the bug-oriented fuzzing on targeted functions. We compare PROMFUZZ with multiple state-of-the-art methods. The results show that PROMFUZZ achieves 86.96% recall and 93.02% F1-score in detecting functional bugs, marking at least a 50% improvement in both metrics over state-of-the-art methods. Moreover, we perform an in-depth analysis on real-world DeFi projects and detect 30 zero-day bugs. Up to now, 24 zero-day bugs have been assigned CVE IDs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23667v1">Context-Independent OCR with Multimodal LLMs: Effects of Image Resolution and Visual Complexity</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      Due to their high versatility in tasks such as image captioning, document analysis, and automated content generation, multimodal Large Language Models (LLMs) have attracted significant attention across various industrial fields. In particular, they have been shown to surpass specialized models in Optical Character Recognition (OCR). Nevertheless, their performance under different image conditions remains insufficiently investigated, and individual character recognition is not guaranteed due to their reliance on contextual cues. In this work, we examine a context-independent OCR task using single-character images with diverse visual complexities to determine the conditions for accurate recognition. Our findings reveal that multimodal LLMs can match conventional OCR methods at about 300 ppi, yet their performance deteriorates significantly below 150 ppi. Additionally, we observe a very weak correlation between visual complexity and misrecognitions, whereas a conventional OCR-specific model exhibits no correlation. These results suggest that image resolution and visual complexity may play an important role in the reliable application of multimodal LLMs to OCR tasks that require precise character-level accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00255v1">SciReplicate-Bench: Benchmarking LLMs in Agent-driven Algorithmic Reproduction from Research Papers</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      This study evaluates large language models (LLMs) in generating code from algorithm descriptions from recent NLP papers. The task requires two key competencies: (1) algorithm comprehension: synthesizing information from papers and academic literature to understand implementation logic, and (2) coding expertise: identifying dependencies and correctly implementing necessary APIs. To facilitate rigorous evaluation, we introduce SciReplicate-Bench, a benchmark of 100 tasks from 36 NLP papers published in 2024, featuring detailed annotations and comprehensive test cases. Building on SciReplicate-Bench, we propose Sci-Reproducer, a multi-agent framework consisting of a Paper Agent that interprets algorithmic concepts from literature and a Code Agent that retrieves dependencies from repositories and implement solutions. To assess algorithm understanding, we introduce reasoning graph accuracy, which quantifies similarity between generated and reference reasoning graphs derived from code comments and structure. For evaluating implementation quality, we employ execution accuracy, CodeBLEU, and repository dependency/API recall metrics. In our experiments, we evaluate various powerful Non-Reasoning LLMs and Reasoning LLMs as foundational models. The best-performing LLM using Sci-Reproducer achieves only 39% execution accuracy, highlighting the benchmark's difficulty.Our analysis identifies missing or inconsistent algorithm descriptions as key barriers to successful reproduction. We will open-source our benchmark, and code at https://github.com/xyzCS/SciReplicate-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01994v1">PIM-LLM: A High-Throughput Hybrid PIM Architecture for 1-bit LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      In this paper, we propose PIM-LLM, a hybrid architecture developed to accelerate 1-bit large language models (LLMs). PIM-LLM leverages analog processing-in-memory (PIM) architectures and digital systolic arrays to accelerate low-precision matrix multiplication (MatMul) operations in projection layers and high-precision MatMul operations in attention heads of 1-bit LLMs, respectively. Our design achieves up to roughly 80x improvement in tokens per second and a 70% increase in tokens per joule compared to conventional hardware accelerators. Additionally, PIM-LLM outperforms previous PIM-based LLM accelerators, setting a new benchmark with at least 2x and 5x improvement in GOPS and GOPS/W, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00241v1">Synthesizing Public Opinions with LLMs: Role Creation, Impacts, and the Future to eDemorcacy</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      This paper investigates the use of Large Language Models (LLMs) to synthesize public opinion data, addressing challenges in traditional survey methods like declining response rates and non-response bias. We introduce a novel technique: role creation based on knowledge injection, a form of in-context learning that leverages RAG and specified personality profiles from the HEXACO model and demographic information, and uses that for dynamically generated prompts. This method allows LLMs to simulate diverse opinions more accurately than existing prompt engineering approaches. We compare our results with pre-trained models with standard few-shot prompts. Experiments using questions from the Cooperative Election Study (CES) demonstrate that our role-creation approach significantly improves the alignment of LLM-generated opinions with real-world human survey responses, increasing answer adherence. In addition, we discuss challenges, limitations and future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00221v1">GazeLLM: Multimodal LLMs incorporating Human Visual Attention</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are advancing into Multimodal LLMs (MLLMs), capable of processing image, audio, and video as well as text. Combining first-person video, MLLMs show promising potential for understanding human activities through video and audio, enabling many human-computer interaction and human-augmentation applications such as human activity support, real-world agents, and skill transfer to robots or other individuals. However, handling high-resolution, long-duration videos generates large latent representations, leading to substantial memory and processing demands, limiting the length and resolution MLLMs can manage. Reducing video resolution can lower memory usage but often compromises comprehension. This paper introduces a method that optimizes first-person video analysis by integrating eye-tracking data, and proposes a method that decomposes first-person vision video into sub areas for regions of gaze focus. By processing these selectively gazed-focused inputs, our approach achieves task comprehension equivalent to or even better than processing the entire image at full resolution, but with significantly reduced video data input (reduce the number of pixels to one-tenth), offering an efficient solution for using MLLMs to interpret and utilize human skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00218v1">$\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00187v1">Insight-RAG: Enhancing LLMs with Insight-Driven Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      Retrieval Augmented Generation (RAG) frameworks have shown significant promise in leveraging external knowledge to enhance the performance of large language models (LLMs). However, conventional RAG methods often retrieve documents based solely on surface-level relevance, leading to many issues: they may overlook deeply buried information within individual documents, miss relevant insights spanning multiple sources, and are not well-suited for tasks beyond traditional question answering. In this paper, we propose Insight-RAG, a novel framework designed to address these issues. In the initial stage of Insight-RAG, instead of using traditional retrieval methods, we employ an LLM to analyze the input query and task, extracting the underlying informational requirements. In the subsequent stage, a specialized LLM -- trained on the document database -- is queried to mine content that directly addresses these identified insights. Finally, by integrating the original query with the retrieved insights, similar to conventional RAG approaches, we employ a final LLM to generate a contextually enriched and accurate response. Using two scientific paper datasets, we created evaluation benchmarks targeting each of the mentioned issues and assessed Insight-RAG against traditional RAG pipeline. Our results demonstrate that the Insight-RAG pipeline successfully addresses these challenges, outperforming existing methods by a significant margin in most cases. These findings suggest that integrating insight-driven retrieval within the RAG framework not only enhances performance but also broadens the applicability of RAG to tasks beyond conventional question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00180v1">Contradiction Detection in RAG Systems: Evaluating LLMs as Context Validators for Improved Information Consistency</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      Retrieval Augmented Generation (RAG) systems have emerged as a powerful method for enhancing large language models (LLMs) with up-to-date information. However, the retrieval step in RAG can sometimes surface documents containing contradictory information, particularly in rapidly evolving domains such as news. These contradictions can significantly impact the performance of LLMs, leading to inconsistent or erroneous outputs. This study addresses this critical challenge in two ways. First, we present a novel data generation framework to simulate different types of contradictions that may occur in the retrieval stage of a RAG system. Second, we evaluate the robustness of different LLMs in performing as context validators, assessing their ability to detect contradictory information within retrieved document sets. Our experimental results reveal that context validation remains a challenging task even for state-of-the-art LLMs, with performance varying significantly across different types of contradictions. While larger models generally perform better at contradiction detection, the effectiveness of different prompting strategies varies across tasks and model architectures. We find that chain-of-thought prompting shows notable improvements for some models but may hinder performance in others, highlighting the complexity of the task and the need for more robust approaches to context validation in RAG systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00125v1">LLMs for Explainable AI: A Comprehensive Survey</a></div>
    <div class="paper-meta">
      📅 2025-03-31
      | 💬 This manuscript is intended for submission to ACM Transactions on Intelligent Systems and Technology
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer a promising approach to enhancing Explainable AI (XAI) by transforming complex machine learning outputs into easy-to-understand narratives, making model predictions more accessible to users, and helping bridge the gap between sophisticated model behavior and human interpretability. AI models, such as state-of-the-art neural networks and deep learning models, are often seen as "black boxes" due to a lack of transparency. As users cannot fully understand how the models reach conclusions, users have difficulty trusting decisions from AI models, which leads to less effective decision-making processes, reduced accountabilities, and unclear potential biases. A challenge arises in developing explainable AI (XAI) models to gain users' trust and provide insights into how models generate their outputs. With the development of Large Language Models, we want to explore the possibilities of using human language-based models, LLMs, for model explainabilities. This survey provides a comprehensive overview of existing approaches regarding LLMs for XAI, and evaluation techniques for LLM-generated explanation, discusses the corresponding challenges and limitations, and examines real-world applications. Finally, we discuss future directions by emphasizing the need for more interpretable, automated, user-centric, and multidisciplinary approaches for XAI via LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00115v1">SACA: A Scenario-Aware Collision Avoidance Framework for Autonomous Vehicles Integrating LLMs-Driven Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-31
      | 💬 10 pages,10 figures. This work has been submitted to the IEEE Robotics and Automation Letters (RAL) for possible publication
    </div>
    <details class="paper-abstract">
      Reliable collision avoidance under extreme situations remains a critical challenge for autonomous vehicles. While large language models (LLMs) offer promising reasoning capabilities, their application in safety-critical evasive maneuvers is limited by latency and robustness issues. Even so, LLMs stand out for their ability to weigh emotional, legal, and ethical factors, enabling socially responsible and context-aware collision avoidance. This paper proposes a scenario-aware collision avoidance (SACA) framework for extreme situations by integrating predictive scenario evaluation, data-driven reasoning, and scenario-preview-based deployment to improve collision avoidance decision-making. SACA consists of three key components. First, a predictive scenario analysis module utilizes obstacle reachability analysis and motion intention prediction to construct a comprehensive situational prompt. Second, an online reasoning module refines decision-making by leveraging prior collision avoidance knowledge and fine-tuning with scenario data. Third, an offline evaluation module assesses performance and stores scenarios in a memory bank. Additionally, A precomputed policy method improves deployability by previewing scenarios and retrieving or reasoning policies based on similarity and confidence levels. Real-vehicle tests show that, compared with baseline methods, SACA effectively reduces collision losses in extreme high-risk scenarios and lowers false triggering under complex conditions. Project page: https://sean-shiyuez.github.io/SACA/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00072v1">Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-31
      | 💬 CVPR 2025 Camera ready. Project page: https://imagine.enpc.fr/~lucas.ventura/chapter-llama/
    </div>
    <details class="paper-abstract">
      We address the task of video chaptering, i.e., partitioning a long video timeline into semantic units and generating corresponding chapter titles. While relatively underexplored, automatic chaptering has the potential to enable efficient navigation and content retrieval in long-form videos. In this paper, we achieve strong chaptering performance on hour-long videos by efficiently addressing the problem in the text domain with our 'Chapter-Llama' framework. Specifically, we leverage a pretrained large language model (LLM) with large context window, and feed as input (i) speech transcripts and (ii) captions describing video frames, along with their respective timestamps. Given the inefficiency of exhaustively captioning all frames, we propose a lightweight speech-guided frame selection strategy based on speech transcript content, and experimentally demonstrate remarkable advantages. We train the LLM to output timestamps for the chapter boundaries, as well as free-form chapter titles. This simple yet powerful approach scales to processing one-hour long videos in a single forward pass. Our results demonstrate substantial improvements (e.g., 45.3 vs 26.7 F1 score) over the state of the art on the recent VidChapters-7M benchmark. To promote further research, we release our code and models at our project page.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00065v1">Assessing Code Understanding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-31
      | 💬 22 page, 7 tables, submitted at FORTE 2025
    </div>
    <details class="paper-abstract">
      We present an empirical evaluation of Large Language Models in code understanding associated with non-trivial, semantic-preserving program transformations such as copy propagation or constant folding. Our findings show that LLMs fail to judge semantic equivalence in approximately 41\% of cases when no context is provided and in 29\% when given a simple generic context. To improve accuracy, we advocate integrating LLMs with code-optimization tools to enhance training and facilitate more robust program understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01986v1">TuRTLe: A Unified Evaluation of LLMs for RTL Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-31
    </div>
    <details class="paper-abstract">
      The rapid advancements in LLMs have driven the adoption of generative AI in various domains, including Electronic Design Automation (EDA). Unlike traditional software development, EDA presents unique challenges, as generated RTL code must not only be syntactically correct and functionally accurate but also synthesizable by hardware generators while meeting performance, power, and area constraints. These additional requirements introduce complexities that existing code-generation benchmarks often fail to capture, limiting their effectiveness in evaluating LLMs for RTL generation. To address this gap, we propose TuRTLe, a unified evaluation framework designed to systematically assess LLMs across key RTL generation tasks. TuRTLe integrates multiple existing benchmarks and automates the evaluation process, enabling a comprehensive assessment of LLM performance in syntax correctness, functional correctness, synthesis, PPA optimization, and exact line completion. Using this framework, we benchmark a diverse set of open LLMs and analyze their strengths and weaknesses in EDA-specific tasks. Our results show that reasoning-based models, such as DeepSeek R1, consistently outperform others across multiple evaluation criteria, but at the cost of increased computational overhead and inference latency. Additionally, base models are better suited in module completion tasks, while instruct-tuned models perform better in specification-to-RTL tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23566v1">When LLM Therapists Become Salespeople: Evaluating Large Language Models for Ethical Motivational Interviewing</a></div>
    <div class="paper-meta">
      📅 2025-03-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been actively applied in the mental health field. Recent research shows the promise of LLMs in applying psychotherapy, especially motivational interviewing (MI). However, there is a lack of studies investigating how language models understand MI ethics. Given the risks that malicious actors can use language models to apply MI for unethical purposes, it is important to evaluate their capability of differentiating ethical and unethical MI practices. Thus, this study investigates the ethical awareness of LLMs in MI with multiple experiments. Our findings show that LLMs have a moderate to strong level of knowledge in MI. However, their ethical standards are not aligned with the MI spirit, as they generated unethical responses and performed poorly in detecting unethical responses. We proposed a Chain-of-Ethic prompt to mitigate those risks and improve safety. Finally, our proposed strategy effectively improved ethical MI response generation and detection performance. These findings highlight the need for safety evaluations and guidelines for building ethical LLM-powered psychotherapy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.13006v2">Systematic Evaluation of LLM-as-a-Judge in LLM Alignment Tasks: Explainable Metrics and Diverse Prompt Templates</a></div>
    <div class="paper-meta">
      📅 2025-03-30
      | 💬 Accepted by Building Trust in LLMs and LLM Applications workshop at ICLR 2025
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge has been widely applied to evaluate and compare different LLM alignmnet approaches (e.g., RLHF and DPO). However, concerns regarding its reliability have emerged, due to LLM judges' biases and inconsistent decision-making. Previous research has developed evaluation frameworks to assess reliability of LLM judges and their alignment with human preferences. However, the employed evaluation metrics often lack adequate explainability and fail to address LLM internal inconsistency. Additionally, existing studies inadequately explore the impact of various prompt templates when applying LLM-as-a-Judge methods, leading to potentially inconsistent comparisons between different alignment algorithms. In this work, we systematically evaluate LLM-as-a-Judge on alignment tasks by defining more theoretically interpretable evaluation metrics and explicitly mitigating LLM internal inconsistency from reliability metrics. We develop an open-source framework to evaluate, compare, and visualize the reliability and alignment of LLM judges, which facilitates practitioners to choose LLM judges for alignment tasks. In the experiments, we examine effects of diverse prompt templates on LLM-judge reliability and also demonstrate our developed framework by comparing various LLM judges on two common alignment datasets (i.e., TL;DR Summarization and HH-RLHF-Helpfulness). Our results indicate a significant impact of prompt templates on LLM judge performance, as well as a mediocre alignment level between the tested LLM judges and human evaluators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23514v1">If an LLM Were a Character, Would It Know Its Own Story? Evaluating Lifelong Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can carry out human-like dialogue, but unlike humans, they are stateless due to the superposition property. However, during multi-turn, multi-agent interactions, LLMs begin to exhibit consistent, character-like behaviors, hinting at a form of emergent lifelong learning. Despite this, existing benchmarks often fail to capture these dynamics, primarily focusing on static, open-ended evaluations. To address this gap, we introduce LIFESTATE-BENCH, a benchmark designed to assess lifelong learning in LLMs. It features two episodic datasets: Hamlet and a synthetic script collection, rich in narrative structure and character interactions. Our fact checking evaluation probes models' self-awareness, episodic memory retrieval, and relationship tracking, across both parametric and non-parametric approaches. Experiments on models like Llama3.1-8B, GPT-4-turbo, and DeepSeek R1, we demonstrate that nonparametric methods significantly outperform parametric ones in managing stateful learning. However, all models exhibit challenges with catastrophic forgetting as interactions extend, highlighting the need for further advancements in lifelong learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00161v2">STEP: Enhancing Video-LLMs' Compositional Reasoning by Spatio-Temporal Graph-guided Self-Training</a></div>
    <div class="paper-meta">
      📅 2025-03-30
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video-LLMs) have recently shown strong performance in basic video understanding tasks, such as captioning and coarse-grained question answering, but struggle with compositional reasoning that requires multi-step spatio-temporal inference across object relations, interactions, and events. The hurdles to enhancing this capability include extensive manual labor, the lack of spatio-temporal compositionality in existing data and the absence of explicit reasoning supervision. In this paper, we propose STEP, a novel graph-guided self-training method that enables Video-LLMs to generate reasoning-rich fine-tuning data from any raw videos to improve itself. Specifically, we first induce Spatio-Temporal Scene Graph (STSG) representation of diverse videos to capture fine-grained, multi-granular video semantics. Then, the STSGs guide the derivation of multi-step reasoning Question-Answer (QA) data with Chain-of-Thought (CoT) rationales. Both answers and rationales are integrated as training objective, aiming to enhance model's reasoning abilities by supervision over explicit reasoning steps. Experimental results demonstrate the effectiveness of STEP across models of varying scales, with a significant 21.3\% improvement in tasks requiring three or more reasoning steps. Furthermore, it achieves superior performance with a minimal amount of self-generated rationale-enriched training samples in both compositional reasoning and comprehensive understanding benchmarks, highlighting the broad applicability and vast potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23415v1">An Analysis of Decoding Methods for LLM-based Agents for Faithful Multi-Hop Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-03-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently produce factually inaccurate outputs - a phenomenon known as hallucination - which limits their accuracy in knowledge-intensive NLP tasks. Retrieval-augmented generation and agentic frameworks such as Reasoning and Acting (ReAct) can address this issue by giving the model access to external knowledge. However, LLMs often fail to remain faithful to retrieved information. Mitigating this is critical, especially if LLMs are required to reason about the retrieved information. Recent research has explored training-free decoding strategies to improve the faithfulness of model generations. We present a systematic analysis of how the combination of the ReAct framework and decoding strategies (i.e., DeCoRe, DoLa, and CAD) can influence the faithfulness of LLM-generated answers. Our results show that combining an agentic framework for knowledge retrieval with decoding methods that enhance faithfulness can increase accuracy on the downstream Multi-Hop Question Answering tasks. For example, we observe an F1 increase from 19.5 to 32.6 on HotpotQA when using ReAct and DoLa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23371v1">FeRG-LLM : Feature Engineering by Reason Generation Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-03-30
      | 💬 Accepted to NAACL 2025 Findings
    </div>
    <details class="paper-abstract">
      One of the key tasks in machine learning for tabular data is feature engineering. Although it is vital for improving the performance of models, it demands considerable human expertise and deep domain knowledge, making it labor-intensive endeavor. To address this issue, we propose a novel framework, \textbf{FeRG-LLM} (\textbf{Fe}ature engineering by \textbf{R}eason \textbf{G}eneration \textbf{L}arge \textbf{L}anguage \textbf{M}odels), a large language model designed to automatically perform feature engineering at an 8-billion-parameter scale. We have constructed two-stage conversational dialogues that enable language models to analyze machine learning tasks and discovering new features, exhibiting their Chain-of-Thought (CoT) capabilities. We use these dialogues to fine-tune Llama 3.1 8B model and integrate Direct Preference Optimization (DPO) to receive feedback improving quality of new features and the model's performance. Our experiments show that FeRG-LLM performs comparably to or better than Llama 3.1 70B on most datasets, while using fewer resources and achieving reduced inference time. It outperforms other studies in classification tasks and performs well in regression tasks. Moreover, since it does not rely on cloud-hosted LLMs like GPT-4 with extra API costs when generating features, it can be deployed locally, addressing security concerns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12820v2">PQCache: Product Quantization-based KVCache for Long Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-30
    </div>
    <details class="paper-abstract">
      As the field of Large Language Models (LLMs) continues to evolve, the context length in inference is steadily growing. Key-Value Cache (KVCache), the intermediate representations of tokens within LLM inference, has now become the primary memory bottleneck due to limited GPU memory. Current methods selectively determine suitable keys and values for self-attention computation in LLMs to address the issue. However, they either fall short in maintaining model quality or result in high serving latency. Drawing inspiration from advanced embedding retrieval techniques prevalent in the data management community, we consider the storage and retrieval of KVCache as a typical embedding retrieval problem. We propose PQCache, which employs Product Quantization (PQ) to manage KVCache, maintaining model quality while ensuring low serving latency. During the prefilling phase, we apply PQ to tokens' keys for each LLM layer and head. During the autoregressive decoding phase, we use PQ codes and centroids to approximately identify important preceding tokens, then fetch the corresponding key-value pairs for self-attention computation. Through meticulous design of overlapping and caching, we minimize any additional computation and communication overhead during both phases. Extensive experiments demonstrate that PQCache achieves both effectiveness and efficiency, with 4.60% score improvement over existing methods on InfiniteBench and low system latency in both prefilling and decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13178v3">Benchmarking Post-Training Quantization in LLMs: Comprehensive Taxonomy, Unified Evaluation, and Comparative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-03-30
      | 💬 17 pages, 3 fugures
    </div>
    <details class="paper-abstract">
      Post-training Quantization (PTQ) technique has been extensively adopted for large language models (LLMs) compression owing to its efficiency and low resource requirement. However, current research lacks a in-depth analysis of the superior and applicable scenarios of each PTQ strategy. In addition, existing algorithms focus primarily on performance, overlooking the trade-off among model size, performance, and quantization bitwidth. To mitigate these confusions, we provide a novel benchmark for LLMs PTQ in this paper. Firstly, in order to support our benchmark, we propose a comprehensive taxonomy for existing mainstream methods by scrutinizing their computational strategies (e.g., optimization-based, compensation-based, etc.). Then, we conduct extensive experiments with the baseline within each class, covering models with various sizes (7B-70B), bitwidths, training levels (LLaMA1/2/3/3.1), architectures (Mixtral, DeepSeekMoE and Mamba) and modality (LLaVA1.5 and VILA1.5) on a wide range of evaluation metrics.Through comparative analysis on the results, we summarize the superior of each PTQ strategy and modelsize-bitwidth trade-off considering the performance. For example, our benchmark reveals that compensation-based technique demonstrates outstanding cross-architecture robustness and extremely low-bit PTQ for ultra large models should be reexamined. Finally, we further accordingly claim that a practical combination of compensation and other PTQ strategy can achieve SOTA various robustness. We believe that our benchmark will provide valuable recommendations for the deployment of LLMs and future research on PTQ approaches.We conduct an repository for our benchmark at https://github.com/zjq0455/PTQ_Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23314v1">SPIO: Ensemble and Selective Strategies via LLM-Based Multi-Agent Planning in Automated Data Science</a></div>
    <div class="paper-meta">
      📅 2025-03-30
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized automated data analytics and machine learning by enabling dynamic reasoning and adaptability. While recent approaches have advanced multi-stage pipelines through multi-agent systems, they typically rely on rigid, single-path workflows that limit the exploration and integration of diverse strategies, often resulting in suboptimal predictions. To address these challenges, we propose SPIO (Sequential Plan Integration and Optimization), a novel framework that leverages LLM-driven decision-making to orchestrate multi-agent planning across four key modules: data preprocessing, feature engineering, modeling, and hyperparameter tuning. In each module, dedicated planning agents independently generate candidate strategies that cascade into subsequent stages, fostering comprehensive exploration. A plan optimization agent refines these strategies by suggesting several optimized plans. We further introduce two variants: SPIO-S, which selects a single best solution path as determined by the LLM, and SPIO-E, which selects the top k candidate plans and ensembles them to maximize predictive performance. Extensive experiments on Kaggle and OpenML datasets demonstrate that SPIO significantly outperforms state-of-the-art methods, providing a robust and scalable solution for automated data science task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23294v1">Cocktail: Chunk-Adaptive Mixed-Precision Quantization for Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-30
      | 💬 Accepted by the Design, Automation, and Test in Europe 2025 (DATE 2025)
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have been able to handle longer and longer contexts. However, a context that is too long may cause intolerant inference latency and GPU memory usage. Existing methods propose mixed-precision quantization to the key-value (KV) cache in LLMs based on token granularity, which is time-consuming in the search process and hardware inefficient during computation. This paper introduces a novel approach called Cocktail, which employs chunk-adaptive mixed-precision quantization to optimize the KV cache. Cocktail consists of two modules: chunk-level quantization search and chunk-level KV cache computation. Chunk-level quantization search determines the optimal bitwidth configuration of the KV cache chunks quickly based on the similarity scores between the corresponding context chunks and the query, maintaining the model accuracy. Furthermore, chunk-level KV cache computation reorders the KV cache chunks before quantization, avoiding the hardware inefficiency caused by mixed-precision quantization in inference computation. Extensive experiments demonstrate that Cocktail outperforms state-of-the-art KV cache quantization methods on various models and datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01637v2">Teams of LLM Agents can Exploit Zero-Day Vulnerabilities</a></div>
    <div class="paper-meta">
      📅 2025-03-30
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      LLM agents have become increasingly sophisticated, especially in the realm of cybersecurity. Researchers have shown that LLM agents can exploit real-world vulnerabilities when given a description of the vulnerability and toy capture-the-flag problems. However, these agents still perform poorly on real-world vulnerabilities that are unknown to the agent ahead of time (zero-day vulnerabilities). In this work, we show that teams of LLM agents can exploit real-world, zero-day vulnerabilities. Prior agents struggle with exploring many different vulnerabilities and long-range planning when used alone. To resolve this, we introduce HPTSA, a system of agents with a planning agent that can launch subagents. The planning agent explores the system and determines which subagents to call, resolving long-term planning issues when trying different vulnerabilities. We construct a benchmark of 14 real-world vulnerabilities and show that our team of agents improve over prior agent frameworks by up to 4.3X.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00048v1">Distill-C: Enhanced NL2SQL via Distilled Customization with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-30
      | 💬 Preprint, accepted at NAACL 2025 (Industry Track)
    </div>
    <details class="paper-abstract">
      The growing adoption of large language models (LLMs) in business applications has amplified interest in Natural Language to SQL (NL2SQL) solutions, in which there is competing demand for high performance and efficiency. Domain- and customer-specific requirements further complicate the problem. To address this conundrum, we introduce Distill-C, a distilled customization framework tailored for NL2SQL tasks. Distill-C utilizes large teacher LLMs to produce high-quality synthetic data through a robust and scalable pipeline. Finetuning smaller and open-source LLMs on this synthesized data enables them to rival or outperform teacher models an order of magnitude larger. Evaluated on multiple challenging benchmarks, Distill-C achieves an average improvement of 36% in execution accuracy compared to the base models from three distinct LLM families. Additionally, on three internal customer benchmarks, Distill-C demonstrates a 22.6% performance improvement over the base models. Our results demonstrate that Distill-C is an effective, high-performing and generalizable approach for deploying lightweight yet powerful NL2SQL models, delivering exceptional accuracies while maintaining low computational cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00043v1">CrossWordBench: Evaluating the Reasoning Capabilities of LLMs and LVLMs with Controllable Puzzle Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-30
    </div>
    <details class="paper-abstract">
      Existing reasoning evaluation frameworks for Large Language Models (LLMs) and Large Vision-Language Models (LVLMs) predominantly either assess text-based reasoning or vision-language understanding capabilities, with limited dynamic interplay between textual and visual constraints. To address this limitation, we introduce CrossWordBench, a benchmark designed to evaluate the reasoning capabilities of both LLMs and LVLMs through the medium of crossword puzzles-a task requiring multimodal adherence to semantic constraints from text-based clues and intersectional constraints from visual grid structures. CrossWordBench leverages a controllable puzzle generation framework that produces puzzles in multiple formats (text and image) and offers different evaluation strategies ranging from direct puzzle solving to interactive modes. Our extensive evaluation of over 20 models reveals that reasoning LLMs outperform non-reasoning models substantially by effectively leveraging crossing-letter constraints. We further demonstrate that LVLMs struggle with the task, showing a strong correlation between their puzzle-solving performance and grid-parsing accuracy. Our findings offer insights into the limitations of the reasoning capabilities of current LLMs and LVLMs, and provide an effective approach for creating multimodal constrained tasks for future evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01036v1">Carbon Footprint Evaluation of Code Generation through LLM as a Service</a></div>
    <div class="paper-meta">
      📅 2025-03-30
      | 💬 Stuttgart Symposium, Springer
    </div>
    <details class="paper-abstract">
      Due to increased computing use, data centers consume and emit a lot of energy and carbon. These contributions are expected to rise as big data analytics, digitization, and large AI models grow and become major components of daily working routines. To reduce the environmental impact of software development, green (sustainable) coding and claims that AI models can improve energy efficiency have grown in popularity. Furthermore, in the automotive industry, where software increasingly governs vehicle performance, safety, and user experience, the principles of green coding and AI-driven efficiency could significantly contribute to reducing the sector's environmental footprint. We present an overview of green coding and metrics to measure AI model sustainability awareness. This study introduces LLM as a service and uses a generative commercial AI language model, GitHub Copilot, to auto-generate code. Using sustainability metrics to quantify these AI models' sustainability awareness, we define the code's embodied and operational carbon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23250v1">Encrypted Prompt: Securing LLM Applications Against Unauthorized Actions</a></div>
    <div class="paper-meta">
      📅 2025-03-29
    </div>
    <details class="paper-abstract">
      Security threats like prompt injection attacks pose significant risks to applications that integrate Large Language Models (LLMs), potentially leading to unauthorized actions such as API misuse. Unlike previous approaches that aim to detect these attacks on a best-effort basis, this paper introduces a novel method that appends an Encrypted Prompt to each user prompt, embedding current permissions. These permissions are verified before executing any actions (such as API calls) generated by the LLM. If the permissions are insufficient, the LLM's actions will not be executed, ensuring safety. This approach guarantees that only actions within the scope of the current permissions from the LLM can proceed. In scenarios where adversarial prompts are introduced to mislead the LLM, this method ensures that any unauthorized actions from LLM wouldn't be executed by verifying permissions in Encrypted Prompt. Thus, threats like prompt injection attacks that trigger LLM to generate harmful actions can be effectively mitigated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23243v1">Evaluating how LLM annotations represent diverse views on contentious topics</a></div>
    <div class="paper-meta">
      📅 2025-03-29
    </div>
    <details class="paper-abstract">
      Researchers have proposed the use of generative large language models (LLMs) to label data for both research and applied settings. This literature emphasizes the improved performance of LLMs relative to other natural language models, noting that LLMs typically outperform other models on standard metrics such as accuracy, precision, recall, and F1 score. However, previous literature has also highlighted the bias embedded in language models, particularly around contentious topics such as potentially toxic content. This bias could result in labels applied by LLMs that disproportionately align with majority groups over a more diverse set of viewpoints. In this paper, we evaluate how LLMs represent diverse viewpoints on these contentious tasks. Across four annotation tasks on four datasets, we show that LLMs do not show substantial disagreement with annotators on the basis of demographics. Instead, the model, prompt, and disagreement between human annotators on the labeling task are far more predictive of LLM agreement. Our findings suggest that when using LLMs to annotate data, under-representing the views of particular groups is not a substantial concern. We conclude with a discussion of the implications for researchers and practitioners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23242v1">Beyond speculation: Measuring the growing presence of LLM-generated texts in multilingual disinformation</a></div>
    <div class="paper-meta">
      📅 2025-03-29
    </div>
    <details class="paper-abstract">
      Increased sophistication of large language models (LLMs) and the consequent quality of generated multilingual text raises concerns about potential disinformation misuse. While humans struggle to distinguish LLM-generated content from human-written texts, the scholarly debate about their impact remains divided. Some argue that heightened fears are overblown due to natural ecosystem limitations, while others contend that specific "longtail" contexts face overlooked risks. Our study bridges this debate by providing the first empirical evidence of LLM presence in the latest real-world disinformation datasets, documenting the increase of machine-generated content following ChatGPT's release, and revealing crucial patterns across languages, platforms, and time periods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19250v2">Have LLMs Reopened the Pandora's Box of AI-Generated Fake News?</a></div>
    <div class="paper-meta">
      📅 2025-03-29
    </div>
    <details class="paper-abstract">
      With the rise of AI-generated content spewed at scale from large language models (LLMs), genuine concerns about the spread of fake news have intensified. The perceived ability of LLMs to produce convincing fake news at scale poses new challenges for both human and automated fake news detection systems. To address this gap, this paper presents the findings from a university-level competition that aimed to explore how LLMs can be used by humans to create fake news, and to assess the ability of human annotators and AI models to detect it. A total of 110 participants used LLMs to create 252 unique fake news stories, and 84 annotators participated in the detection tasks. Our findings indicate that LLMs are ~68% more effective at detecting real news than humans. However, for fake news detection, the performance of LLMs and humans remains comparable (~60% accuracy). Additionally, we examine the impact of visual elements (e.g., pictures) in news on the accuracy of detecting fake news stories. Finally, we also examine various strategies used by fake news creators to enhance the credibility of their AI-generated content. This work highlights the increasing complexity of detecting AI-generated fake news, particularly in collaborative human-AI settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23219v1">Aurelia: Test-time Reasoning Distillation in Audio-Visual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-29
    </div>
    <details class="paper-abstract">
      Recent advancements in reasoning optimization have greatly enhanced the performance of large language models (LLMs). However, existing work fails to address the complexities of audio-visual scenarios, underscoring the need for further research. In this paper, we introduce AURELIA, a novel actor-critic based audio-visual (AV) reasoning framework that distills structured, step-by-step reasoning into AVLLMs at test time, improving their ability to process complex multi-modal inputs without additional training or fine-tuning. To further advance AVLLM reasoning skills, we present AVReasonBench, a challenging benchmark comprising 4500 audio-visual questions, each paired with detailed step-by-step reasoning. Our benchmark spans six distinct tasks, including AV-GeoIQ, which evaluates AV reasoning combined with geographical and cultural knowledge. Evaluating 18 AVLLMs on AVReasonBench reveals significant limitations in their multi-modal reasoning capabilities. Using AURELIA, we achieve up to a 100% relative improvement, demonstrating its effectiveness. This performance gain highlights the potential of reasoning-enhanced data generation for advancing AVLLMs in real-world applications. Our code and data will be publicly released at: https: //github.com/schowdhury671/aurelia.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20084v2">Can Multi-modal (reasoning) LLMs work as deepfake detectors?</a></div>
    <div class="paper-meta">
      📅 2025-03-29
    </div>
    <details class="paper-abstract">
      Deepfake detection remains a critical challenge in the era of advanced generative models, particularly as synthetic media becomes more sophisticated. In this study, we explore the potential of state of the art multi-modal (reasoning) large language models (LLMs) for deepfake image detection such as (OpenAI O1/4o, Gemini thinking Flash 2, Deepseek Janus, Grok 3, llama 3.2, Qwen 2/2.5 VL, Mistral Pixtral, Claude 3.5/3.7 sonnet) . We benchmark 12 latest multi-modal LLMs against traditional deepfake detection methods across multiple datasets, including recently published real-world deepfake imagery. To enhance performance, we employ prompt tuning and conduct an in-depth analysis of the models' reasoning pathways to identify key contributing factors in their decision-making process. Our findings indicate that best multi-modal LLMs achieve competitive performance with promising generalization ability with zero shot, even surpass traditional deepfake detection pipelines in out-of-distribution datasets while the rest of the LLM families performs extremely disappointing with some worse than random guess. Furthermore, we found newer model version and reasoning capabilities does not contribute to performance in such niche tasks of deepfake detection while model size do help in some cases. This study highlights the potential of integrating multi-modal reasoning in future deepfake detection frameworks and provides insights into model interpretability for robustness in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16929v2">TEMPLE:Temporal Preference Learning of Video LLMs via Difficulty Scheduling and Pre-SFT Alignment</a></div>
    <div class="paper-meta">
      📅 2025-03-29
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video LLMs) have achieved significant success by leveraging a two-stage paradigm: pretraining on large-scale video-text data for vision-language alignment, followed by supervised fine-tuning (SFT) for task-specific capabilities. However, existing approaches struggle with temporal reasoning due to weak temporal correspondence in the data and reliance on the next-token prediction paradigm during training. To address these limitations, we propose TEMPLE (TEMporal Preference Learning), a systematic framework that enhances Video LLMs' temporal reasoning capabilities through Direct Preference Optimization (DPO). To facilitate this, we introduce an automated preference data generation pipeline that systematically constructs preference pairs by selecting videos that are rich in temporal information, designing video-specific perturbation strategies, and finally evaluating model responses on clean and perturbed video inputs. Our temporal alignment features two key innovations: curriculum learning which that progressively increases perturbation difficulty to improve model robustness and adaptability; and "Pre-SFT Alignment'', applying preference optimization before instruction tuning to prioritize fine-grained temporal comprehension. Extensive experiments demonstrate that our approach consistently improves Video LLM performance across multiple benchmarks with a relatively small set of self-generated DPO data. We further analyze the transferability of DPO data across architectures and the role of difficulty scheduling in optimization. Our findings highlight our TEMPLE as a scalable and efficient complement to SFT-based methods, paving the way for developing reliable Video LLMs. Code is available at https://github.com/lscpku/TEMPLE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05194v2">LLMs Are Not Intelligent Thinkers: Introducing Mathematical Topic Tree Benchmark for Comprehensive Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate impressive capabilities in mathematical reasoning. However, despite these achievements, current evaluations are mostly limited to specific mathematical topics, and it remains unclear whether LLMs are genuinely engaging in reasoning. To address these gaps, we present the Mathematical Topics Tree (MaTT) benchmark, a challenging and structured benchmark that offers 1,958 questions across a wide array of mathematical subjects, each paired with a detailed hierarchical chain of topics. Upon assessing different LLMs using the MaTT benchmark, we find that the most advanced model, GPT-4, achieved a mere 54\% accuracy in a multiple-choice scenario. Interestingly, even when employing Chain-of-Thought prompting, we observe mostly no notable improvement. Moreover, LLMs accuracy dramatically reduced by up to 24.2 percentage point when the questions were presented without providing choices. Further detailed analysis of the LLMs' performance across a range of topics showed significant discrepancy even for closely related subtopics within the same general mathematical area. In an effort to pinpoint the reasons behind LLMs performances, we conducted a manual evaluation of the completeness and correctness of the explanations generated by GPT-4 when choices were available. Surprisingly, we find that in only 53.3\% of the instances where the model provided a correct answer, the accompanying explanations were deemed complete and accurate, i.e., the model engaged in genuine reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23145v1">CodeARC: Benchmarking Reasoning Capabilities of LLM Agents for Inductive Program Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-03-29
    </div>
    <details class="paper-abstract">
      Inductive program synthesis, or programming by example, requires synthesizing functions from input-output examples that generalize to unseen inputs. While large language model agents have shown promise in programming tasks guided by natural language, their ability to perform inductive program synthesis is underexplored. Existing evaluation protocols rely on static sets of examples and held-out tests, offering no feedback when synthesized functions are incorrect and failing to reflect real-world scenarios such as reverse engineering. We propose CodeARC, the Code Abstraction and Reasoning Challenge, a new evaluation framework where agents interact with a hidden target function by querying it with new inputs, synthesizing candidate functions, and iteratively refining their solutions using a differential testing oracle. This interactive setting encourages agents to perform function calls and self-correction based on feedback. We construct the first large-scale benchmark for general-purpose inductive program synthesis, featuring 1114 functions. Among 18 models evaluated, o3-mini performs best with a success rate of 52.7%, highlighting the difficulty of this task. Fine-tuning LLaMA-3.1-8B-Instruct on curated synthesis traces yields up to a 31% relative performance gain. CodeARC provides a more realistic and challenging testbed for evaluating LLM-based program synthesis and inductive reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23132v1">LAURA: LLM-Assisted UAV Routing for AoI Minimization</a></div>
    <div class="paper-meta">
      📅 2025-03-29
    </div>
    <details class="paper-abstract">
      With the rapid growth of the low-altitude economy, there is increasing demand for real-time data collection using UAV-assisted wireless sensor networks. This paper investigates the problem of minimizing the age of information (AoI) in UAV-assisted wireless sensor networks by optimizing the UAV flight routing. We formulate the AoI minimization task and propose a large language model (LLM)-assisted UAV routing algorithm (LAURA). LAURA employs an LLM as intelligent crossover operators within an evolutionary optimization framework to efficiently explore the solution space. Simulation results show that LAURA outperforms benchmark methods in reducing the maximum AoI, especially in scenarios with a large number of sensor nodes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23128v1">CrossMuSim: A Cross-Modal Framework for Music Similarity Retrieval with LLM-Powered Text Description Sourcing and Mining</a></div>
    <div class="paper-meta">
      📅 2025-03-29
      | 💬 Accepted by ICME2025
    </div>
    <details class="paper-abstract">
      Music similarity retrieval is fundamental for managing and exploring relevant content from large collections in streaming platforms. This paper presents a novel cross-modal contrastive learning framework that leverages the open-ended nature of text descriptions to guide music similarity modeling, addressing the limitations of traditional uni-modal approaches in capturing complex musical relationships. To overcome the scarcity of high-quality text-music paired data, this paper introduces a dual-source data acquisition approach combining online scraping and LLM-based prompting, where carefully designed prompts leverage LLMs' comprehensive music knowledge to generate contextually rich descriptions. Exten1sive experiments demonstrate that the proposed framework achieves significant performance improvements over existing benchmarks through objective metrics, subjective evaluations, and real-world A/B testing on the Huawei Music streaming platform.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.13362v4">Lusifer: LLM-based User SImulated Feedback Environment for online Recommender systems</a></div>
    <div class="paper-meta">
      📅 2025-03-29
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) recommender systems often rely on static datasets that fail to capture the fluid, ever changing nature of user preferences in real-world scenarios. Meanwhile, generative AI techniques have emerged as powerful tools for creating synthetic data, including user profiles and behaviors. Recognizing this potential, we introduce Lusifer, an LLM-based simulation environment designed to generate dynamic, realistic user feedback for RL-based recommender training. In Lusifer, user profiles are incrementally updated at each interaction step, with Large Language Models (LLMs) providing transparent explanations of how and why preferences evolve. We focus on the MovieLens dataset, extracting only the last 40 interactions for each user, to emphasize recent behavior. By processing textual metadata (such as movie overviews and tags) Lusifer creates more context aware user states and simulates feedback on new items, including those with limited or no prior ratings. This approach reduces reliance on extensive historical data and facilitates cold start scenario handling and adaptation to out of distribution cases. Our experiments compare Lusifer with traditional collaborative filtering models, revealing that while Lusifer can be comparable in predictive accuracy, it excels at capturing dynamic user responses and yielding explainable results at every step. These qualities highlight its potential as a scalable, ethically sound alternative to live user experiments, supporting iterative and user-centric evaluations of RL-based recommender strategies. Looking ahead, we envision Lusifer serving as a foundational tool for exploring generative AI-driven user simulations, enabling more adaptive and personalized recommendation pipelines under real world constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23053v1">A Training-free LLM Framework with Interaction between Contextually Related Subtasks in Solving Complex Tasks</a></div>
    <div class="paper-meta">
      📅 2025-03-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capabilities in solving complex tasks. Recent work has explored decomposing such tasks into subtasks with independent contexts. However, some contextually related subtasks may encounter information loss during execution, leading to redundant operations or execution failures. To address this issue, we propose a training-free framework with an interaction mechanism, which enables a subtask to query specific information or trigger certain actions in completed subtasks by sending requests. To implement interaction, we introduce a subtask trajectory memory to enable resumption of completed subtasks upon receiving interaction requests. Additionally, we propose a new action during execution, which generates a concise and precise description of execution process and outcomes of a subtask, to assist subsequent subtasks in determining interaction targets and requests. We evaluate our framework on interactive decision-making task WebShop and multi-hop question answering HotpotQA, with GPT-3.5 and GPT-4, and comparison results show that our framework outperforms the state-of-the-art training-free baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17578v2">MM-Eval: A Multilingual Meta-Evaluation Benchmark for LLM-as-a-Judge and Reward Models</a></div>
    <div class="paper-meta">
      📅 2025-03-29
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are now capable of producing fluent and coherent content in languages other than English, it is not imperative to precisely evaluate these non-English outputs. However, when assessing the outputs from mutlilingual LLMs, prior works often employed LLM based evaluators that excel at assessing English outputs, without a thorough examination of whether these evaluators could effectively assess non-English text as well. Moreover, existing benchmarks to test evaluator LLMs (referred to as "meta-evaluation benchmarks") are mostly English-centric. To bridge this gap and examine whether evaluator LLMs can reliably assess the outputs of multilingual LLMs, we introduce MM-Eval, a multilingual meta-evaluation benchmark comprising five core subsets covering 18 languages and a Language Consistency subset spanning 122 languages. A core attribute of MM-Eval is that, instead of merely translating existing English meta-evaluation benchmarks, it is designed with multilingual-specific challenges in mind. Additionally, unlike existing meta-evaluation benchmarks that focus solely on ranking accuracy over pairwise data, MM-Eval also evaluates the consistency and fairness of absolute score values across a wide range of languages. Our results show that existing evaluator LLMs that excel in English contexts have considerable room for improvement when assessing non-English outputs. Furthermore, we find that evaluators are unfair and inconsistent when evaluating lower-resourced languages. Finally, we validate MM-Eval by measuring its correlation with Best-of-N rankings, finding a significantly stronger correlation compared to other meta-evaluation benchmarks. We publicly release our benchmark and code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23029v1">A Retrieval-Augmented Knowledge Mining Method with Deep Thinking LLMs for Biomedical Research and Clinical Support</a></div>
    <div class="paper-meta">
      📅 2025-03-29
    </div>
    <details class="paper-abstract">
      Knowledge graphs and large language models (LLMs) are key tools for biomedical knowledge integration and reasoning, facilitating structured organization of scientific articles and discovery of complex semantic relationships. However, current methods face challenges: knowledge graph construction is limited by complex terminology, data heterogeneity, and rapid knowledge evolution, while LLMs show limitations in retrieval and reasoning, making it difficult to uncover cross-document associations and reasoning pathways. To address these issues, we propose a pipeline that uses LLMs to construct a biomedical knowledge graph (BioStrataKG) from large-scale articles and builds a cross-document question-answering dataset (BioCDQA) to evaluate latent knowledge retrieval and multi-hop reasoning. We then introduce Integrated and Progressive Retrieval-Augmented Reasoning (IP-RAR) to enhance retrieval accuracy and knowledge reasoning. IP-RAR maximizes information recall through Integrated Reasoning-based Retrieval and refines knowledge via Progressive Reasoning-based Generation, using self-reflection to achieve deep thinking and precise contextual understanding. Experiments show that IP-RAR improves document retrieval F1 score by 20\% and answer generation accuracy by 25\% over existing methods. This framework helps doctors efficiently integrate treatment evidence for personalized medication plans and enables researchers to analyze advancements and research gaps, accelerating scientific discovery and decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00290v3">Estimating LLM Uncertainty with Logits</a></div>
    <div class="paper-meta">
      📅 2025-03-29
      | 💬 Fixed some data errors in Table 1
    </div>
    <details class="paper-abstract">
      Over the past few years, Large Language Models (LLMs) have developed rapidly and are widely applied in various domains. However, LLMs face the issue of hallucinations, generating responses that may be unreliable when the models lack relevant knowledge. To be aware of potential hallucinations, uncertainty estimation methods have been introduced, and most of them have confirmed that reliability lies in critical tokens. However, probability-based methods perform poorly in identifying token reliability, limiting their practical utility. In this paper, we reveal that the probability-based method fails to estimate token reliability due to the loss of evidence strength information which is accumulated in the training stage. Therefore, we present Logits-induced token uncertainty (LogTokU), a framework for estimating decoupled token uncertainty in LLMs, enabling real-time uncertainty estimation without requiring multiple sampling processes. We employ evidence modeling to implement LogTokU and use the estimated uncertainty to guide downstream tasks. The experimental results demonstrate that LogTokU has significant effectiveness and promise.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01638v5">TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting via Cross-Modality Alignment</a></div>
    <div class="paper-meta">
      📅 2025-03-29
      | 💬 Accepted as an Oral Presentation at AAAI 2025 (Main Technical Track)
    </div>
    <details class="paper-abstract">
      Multivariate time series forecasting (MTSF) aims to learn temporal dynamics among variables to forecast future time series. Existing statistical and deep learning-based methods suffer from limited learnable parameters and small-scale training data. Recently, large language models (LLMs) combining time series with textual prompts have achieved promising performance in MTSF. However, we discovered that current LLM-based solutions fall short in learning disentangled embeddings. We introduce TimeCMA, an intuitive yet effective framework for MTSF via cross-modality alignment. Specifically, we present a dual-modality encoding with two branches: the time series encoding branch extracts disentangled yet weak time series embeddings, and the LLM-empowered encoding branch wraps the same time series with text as prompts to obtain entangled yet robust prompt embeddings. As a result, such a cross-modality alignment retrieves both disentangled and robust time series embeddings, "the best of two worlds", from the prompt embeddings based on time series and prompt modality similarities. As another key design, to reduce the computational costs from time series with their length textual prompts, we design an effective prompt to encourage the most essential temporal information to be encapsulated in the last token: only the last token is passed to downstream prediction. We further store the last token embeddings to accelerate inference speed. Extensive experiments on eight real datasets demonstrate that TimeCMA outperforms state-of-the-arts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22968v1">HRET: A Self-Evolving LLM Evaluation Toolkit for Korean</a></div>
    <div class="paper-meta">
      📅 2025-03-29
    </div>
    <details class="paper-abstract">
      Recent advancements in Korean large language models (LLMs) have spurred numerous benchmarks and evaluation methodologies, yet the lack of a standardized evaluation framework has led to inconsistent results and limited comparability. To address this, we introduce HRET Haerae Evaluation Toolkit, an open-source, self-evolving evaluation framework tailored specifically for Korean LLMs. HRET unifies diverse evaluation methods, including logit-based scoring, exact-match, language-inconsistency penalization, and LLM-as-a-Judge assessments. Its modular, registry-based architecture integrates major benchmarks (HAE-RAE Bench, KMMLU, KUDGE, HRM8K) and multiple inference backends (vLLM, HuggingFace, OpenAI-compatible endpoints). With automated pipelines for continuous evolution, HRET provides a robust foundation for reproducible, fair, and transparent Korean NLP research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02442v2">TODO: Enhancing LLM Alignment with Ternary Preferences</a></div>
    <div class="paper-meta">
      📅 2025-03-29
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) with human intent is critical for enhancing their performance across a variety of tasks. Standard alignment techniques, such as Direct Preference Optimization (DPO), often rely on the binary Bradley-Terry (BT) model, which can struggle to capture the complexities of human preferences -- particularly in the presence of noisy or inconsistent labels and frequent ties. To address these limitations, we introduce the Tie-rank Oriented Bradley-Terry model (TOBT), an extension of the BT model that explicitly incorporates ties, enabling more nuanced preference representation. Building on this, we propose Tie-rank Oriented Direct Preference Optimization (TODO), a novel alignment algorithm that leverages TOBT's ternary ranking system to improve preference alignment. In evaluations on Mistral-7B and Llama 3-8B models, TODO consistently outperforms DPO in modeling preferences across both in-distribution and out-of-distribution datasets. Additional assessments using MT Bench and benchmarks such as Piqa, ARC-c, and MMLU further demonstrate TODO's superior alignment performance. Notably, TODO also shows strong results in binary preference alignment, highlighting its versatility and potential for broader integration into LLM alignment. The implementation details can be found in https://github.com/XXares/TODO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22954v1">Can LLMs Support Medical Knowledge Imputation? An Evaluation-Based Perspective</a></div>
    <div class="paper-meta">
      📅 2025-03-29
      | 💬 10 pages, 3 figures, AMIA
    </div>
    <details class="paper-abstract">
      Medical knowledge graphs (KGs) are essential for clinical decision support and biomedical research, yet they often exhibit incompleteness due to knowledge gaps and structural limitations in medical coding systems. This issue is particularly evident in treatment mapping, where coding systems such as ICD, Mondo, and ATC lack comprehensive coverage, resulting in missing or inconsistent associations between diseases and their potential treatments. To address this issue, we have explored the use of Large Language Models (LLMs) for imputing missing treatment relationships. Although LLMs offer promising capabilities in knowledge augmentation, their application in medical knowledge imputation presents significant risks, including factual inaccuracies, hallucinated associations, and instability between and within LLMs. In this study, we systematically evaluate LLM-driven treatment mapping, assessing its reliability through benchmark comparisons. Our findings highlight critical limitations, including inconsistencies with established clinical guidelines and potential risks to patient safety. This study serves as a cautionary guide for researchers and practitioners, underscoring the importance of critical evaluation and hybrid approaches when leveraging LLMs to enhance treatment mappings on medical knowledge graphs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.21239v3">Semantic Volume: Quantifying and Detecting both External and Internal Uncertainty in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable performance across diverse tasks by encoding vast amounts of factual knowledge. However, they are still prone to hallucinations, generating incorrect or misleading information, often accompanied by high uncertainty. Existing methods for hallucination detection primarily focus on quantifying internal uncertainty, which arises from missing or conflicting knowledge within the model. However, hallucinations can also stem from external uncertainty, where ambiguous user queries lead to multiple possible interpretations. In this work, we introduce Semantic Volume, a novel mathematical measure for quantifying both external and internal uncertainty in LLMs. Our approach perturbs queries and responses, embeds them in a semantic space, and computes the determinant of the Gram matrix of the embedding vectors, capturing their dispersion as a measure of uncertainty. Our framework provides a generalizable and unsupervised uncertainty detection method without requiring internal access to LLMs. We conduct extensive experiments on both external and internal uncertainty detection, demonstrating that our Semantic Volume method consistently outperforms existing baselines in both tasks. Additionally, we provide theoretical insights linking our measure to differential entropy, unifying and extending previous sampling-based uncertainty measures such as the semantic entropy. Semantic Volume is shown to be a robust and interpretable approach to improving the reliability of LLMs by systematically detecting uncertainty in both user queries and model responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09054v2">Cost-Saving LLM Cascades with Early Abstention</a></div>
    <div class="paper-meta">
      📅 2025-03-29
      | 💬 9 pages, 3 figures
    </div>
    <details class="paper-abstract">
      LLM cascades deploy small LLMs to answer most queries, limiting the use of large and expensive LLMs to difficult queries. This approach can significantly reduce costs without impacting performance. However, risk-sensitive domains such as finance or medicine place an additional premium on avoiding model errors. Since even the most expensive models are susceptible to making mistakes, applications in these domains benefit from allowing LLM systems to completely abstain from answering difficult queries. Introducing abstention poses a design question for LLM cascades: should abstention only be allowed at the final model or also at earlier models? Since the error patterns of small and large models are correlated, allowing earlier models to abstain may reduce inference costs and latency by anticipating abstention decisions by expensive and slow models, thus avoiding the need to run these models. We investigate the benefits of such "early abstention" in LLM cascades and find that it reduces overall test loss by 2.2% on average across six benchmarks (GSM8K, MedMCQA, MMLU, TriviaQA, TruthfulQA, and XSum). These gains result from a more effective use of abstention, trading a 4.1% average increase in the overall abstention rate for a 13.0% reduction in cost and a 5.0% reduction in error rate. Our findings demonstrate the possibility of leveraging correlations between the error patterns of different language models to drive performance improvements for LLM systems with abstention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01032v1">Who Owns the Output? Bridging Law and Technology in LLMs Attribution</a></div>
    <div class="paper-meta">
      📅 2025-03-29
      | 💬 20 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Since the introduction of ChatGPT in 2022, Large language models (LLMs) and Large Multimodal Models (LMM) have transformed content creation, enabling the generation of human-quality content, spanning every medium, text, images, videos, and audio. The chances offered by generative AI models are endless and are drastically reducing the time required to generate content and usually raising the quality of the generation. However, considering the complexity and the difficult traceability of the generated content, the use of these tools provides challenges in attributing AI-generated content. The difficult attribution resides for a variety of reasons, starting from the lack of a systematic fingerprinting of the generated content and ending with the enormous amount of data on which LLMs and LMM are trained, which makes it difficult to connect generated content to the training data. This scenario is raising concerns about intellectual property and ethical responsibilities. To address these concerns, in this paper, we bridge the technological, ethical, and legislative aspects, by proposing a review of the legislative and technological instruments today available and proposing a legal framework to ensure accountability. In the end, we propose three use cases of how these can be combined to guarantee that attribution is respected. However, even though the techniques available today can guarantee a greater attribution to a greater extent, strong limitations still apply, that can be solved uniquely by the development of new attribution techniques, to be applied to LLMs and LMMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00029v1">Generating Structured Plan Representation of Procedures with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      In this paper, we address the challenges of managing Standard Operating Procedures (SOPs), which often suffer from inconsistencies in language, format, and execution, leading to operational inefficiencies. Traditional process modeling demands significant manual effort, domain expertise, and familiarity with complex languages like Business Process Modeling Notation (BPMN), creating barriers for non-techincal users. We introduce SOP Structuring (SOPStruct), a novel approach that leverages Large Language Models (LLMs) to transform SOPs into decision-tree-based structured representations. SOPStruct produces a standardized representation of SOPs across different domains, reduces cognitive load, and improves user comprehension by effectively capturing task dependencies and ensuring sequential integrity. Our approach enables leveraging the structured information to automate workflows as well as empower the human users. By organizing procedures into logical graphs, SOPStruct facilitates backtracking and error correction, offering a scalable solution for process optimization. We employ a novel evaluation framework, combining deterministic methods with the Planning Domain Definition Language (PDDL) to verify graph soundness, and non-deterministic assessment by an LLM to ensure completeness. We empirically validate the robustness of our LLM-based structured SOP representation methodology across SOPs from different domains and varying levels of complexity. Despite the current lack of automation readiness in many organizations, our research highlights the transformative potential of LLMs to streamline process modeling, paving the way for future advancements in automated procedure optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22674v1">QuestBench: Can LLMs ask the right question to acquire information in reasoning tasks?</a></div>
    <div class="paper-meta">
      📅 2025-03-28
      | 💬 Code and dataset are available at \url{https://github.com/google-deepmind/questbench}
    </div>
    <details class="paper-abstract">
      Recently, a large amount of work has focused on improving large language models' (LLMs') performance on reasoning benchmarks such as math and logic. However, past work has largely assumed that tasks are well-defined. In the real world, queries to LLMs are often underspecified, only solvable through acquiring missing information. We formalize this as a constraint satisfaction problem (CSP) with missing variable assignments. Using a special case of this formalism where only one necessary variable assignment is missing, we can rigorously evaluate an LLM's ability to identify the minimal necessary question to ask and quantify axes of difficulty levels for each problem. We present QuestBench, a set of underspecified reasoning tasks solvable by asking at most one question, which includes: (1) Logic-Q: Logical reasoning tasks with one missing proposition, (2) Planning-Q: PDDL planning problems with initial states that are partially-observed, (3) GSM-Q: Human-annotated grade school math problems with one missing variable assignment, and (4) GSME-Q: a version of GSM-Q where word problems are translated into equations by human annotators. The LLM is tasked with selecting the correct clarification question(s) from a list of options. While state-of-the-art models excel at GSM-Q and GSME-Q, their accuracy is only 40-50% on Logic-Q and Planning-Q. Analysis demonstrates that the ability to solve well-specified reasoning problems may not be sufficient for success on our benchmark: models have difficulty identifying the right question to ask, even when they can solve the fully specified version of the problem. Furthermore, in the Planning-Q domain, LLMs tend not to hedge, even when explicitly presented with the option to predict ``not sure.'' This highlights the need for deeper investigation into models' information acquisition capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07877v3">A RAG-Based Multi-Agent LLM System for Natural Hazard Resilience and Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are a transformational capability at the frontier of artificial intelligence and machine learning that can support decision-makers in addressing pressing societal challenges such as extreme natural hazard events. As generalized models, LLMs often struggle to provide context-specific information, particularly in areas requiring specialized knowledge. In this work, we propose a Retrieval-Augmented Generation (RAG)-based multi-agent LLM system to support analysis and decision-making in the context of natural hazards and extreme weather events. As a proof of concept, we present WildfireGPT, a specialized system focused on wildfire scenarios. The architecture employs a user-centered, multi-agent design to deliver tailored risk insights across diverse stakeholder groups. By integrating domain-specific projection data, observational datasets, and scientific literature through a RAG framework, the system ensures both accuracy and contextual relevance of the information it provides. Evaluation across ten expert-led case studies demonstrates that WildfireGPT significantly outperforms existing LLM-based solutions for decision support in natural hazard and extreme weather contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22587v1">LLM-enabled Instance Model Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      In the domain of model-based engineering, models are essential components that enable system design and analysis. Traditionally, the creation of these models has been a manual process requiring not only deep modeling expertise but also substantial domain knowledge of target systems. With the rapid advancement of generative artificial intelligence, large language models (LLMs) show potential for automating model generation. This work explores the generation of instance models using LLMs, focusing specifically on producing XMI-based instance models from Ecore metamodels and natural language specifications. We observe that current LLMs struggle to directly generate valid XMI models. To address this, we propose a two-step approach: first, using LLMs to produce a simplified structured output containing all necessary instance model information, namely a conceptual instance model, and then compiling this intermediate representation into a valid XMI file. The conceptual instance model is format-independent, allowing it to be transformed into various modeling formats via different compilers. The feasibility of the proposed method has been demonstrated using several LLMs, including GPT-4o, o1-preview, Llama 3.1 (8B and 70B). Results show that the proposed method significantly improves the usability of LLMs for instance model generation tasks. Notably, the smaller open-source model, Llama 3.1 70B, demonstrated performance comparable to proprietary GPT models within the proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22562v1">Niyama : Breaking the Silos of LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) has enabled diverse applications with very different latency requirements. Existing LLM serving frameworks rely on siloed infrastructure with coarse-grained workload segregation -- interactive and batch -- leading to inefficient resource utilization and limited support for fine-grained Quality-of-Service (QoS) differentiation. This results in operational inefficiencies, over-provisioning and poor load management during traffic surges. We present Niyama, a novel QoS-driven inference serving system that enables efficient co-scheduling of diverse workloads on shared infrastructure. Niyama introduces fine-grained QoS classification allowing applications to specify precise latency requirements, and dynamically adapts scheduling decisions based on real-time system state. Leveraging the predictable execution characteristics of LLM inference, Niyama implements a dynamic chunking mechanism to improve overall throughput while maintaining strict QoS guarantees. Additionally, Niyama employs a hybrid prioritization policy that balances fairness and efficiency, and employs selective request relegation that enables graceful service degradation during overload conditions. Our evaluation demonstrates that Niyama increases serving capacity by 32% compared to current siloed deployments, while maintaining QoS guarantees. Notably, under extreme load, our system reduces SLO violations by an order of magnitude compared to current strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14582v4">Do LLMs estimate uncertainty well in instruction-following?</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) could be valuable personal AI agents across various domains, provided they can precisely follow user instructions. However, recent studies have shown significant limitations in LLMs' instruction-following capabilities, raising concerns about their reliability in high-stakes applications. Accurately estimating LLMs' uncertainty in adhering to instructions is critical to mitigating deployment risks. We present, to our knowledge, the first systematic evaluation of the uncertainty estimation abilities of LLMs in the context of instruction-following. Our study identifies key challenges with existing instruction-following benchmarks, where multiple factors are entangled with uncertainty stems from instruction-following, complicating the isolation and comparison across methods and models. To address these issues, we introduce a controlled evaluation setup with two benchmark versions of data, enabling a comprehensive comparison of uncertainty estimation methods under various conditions. Our findings show that existing uncertainty methods struggle, particularly when models make subtle errors in instruction following. While internal model states provide some improvement, they remain inadequate in more complex scenarios. The insights from our controlled evaluation setups provide a crucial understanding of LLMs' limitations and potential for uncertainty estimation in instruction-following tasks, paving the way for more trustworthy AI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14516v5">Do LLMs "know" internally when they follow instructions?</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      Instruction-following is crucial for building AI agents with large language models (LLMs), as these models must adhere strictly to user-provided constraints and guidelines. However, LLMs often fail to follow even simple and clear instructions. To improve instruction-following behavior and prevent undesirable outputs, a deeper understanding of how LLMs' internal states relate to these outcomes is required. In this work, we investigate whether LLMs encode information in their representations that correlate with instruction-following success - a property we term knowing internally. Our analysis identifies a direction in the input embedding space, termed the instruction-following dimension, that predicts whether a response will comply with a given instruction. We find that this dimension generalizes well across unseen tasks but not across unseen instruction types. We demonstrate that modifying representations along this dimension improves instruction-following success rates compared to random changes, without compromising response quality. Further investigation reveals that this dimension is more closely related to the phrasing of prompts rather than the inherent difficulty of the task or instructions. This work provides insight into the internal workings of LLMs' instruction-following, paving the way for reliable LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14739v4">SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable proficiency in mainstream academic disciplines such as mathematics, physics, and computer science. However, human knowledge encompasses over 200 specialized disciplines, far exceeding the scope of existing benchmarks. The capabilities of LLMs in many of these specialized fields-particularly in light industry, agriculture, and service-oriented disciplines-remain inadequately evaluated. To address this gap, we present SuperGPQA, a comprehensive benchmark that evaluates graduate-level knowledge and reasoning capabilities across 285 disciplines. Our benchmark employs a novel Human-LLM collaborative filtering mechanism to eliminate trivial or ambiguous questions through iterative refinement based on both LLM responses and expert feedback. Our experimental results reveal significant room for improvement in the performance of current state-of-the-art LLMs across diverse knowledge domains (e.g., the reasoning-focused model DeepSeek-R1 achieved the highest accuracy of 61.82% on SuperGPQA), highlighting the considerable gap between current model capabilities and artificial general intelligence. Additionally, we present comprehensive insights from our management of a large-scale annotation process, involving over 80 expert annotators and an interactive Human-LLM collaborative system, offering valuable methodological guidance for future research initiatives of comparable scope.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22512v1">Unlocking LLM Repair Capabilities in Low-Resource Programming Languages Through Cross-Language Translation and Multi-Agent Refinement</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      Recent advances in leveraging LLMs for APR have demonstrated impressive capabilities in fixing software defects. However, current LLM-based approaches predominantly focus on mainstream programming languages like Java and Python, neglecting less prevalent but emerging languages such as Rust due to expensive training resources, limited datasets, and insufficient community support. This narrow focus creates a significant gap in repair capabilities across the programming language spectrum, where the full potential of LLMs for comprehensive multilingual program repair remains largely unexplored. To address this limitation, we introduce a novel cross-language program repair approach LANTERN that leverages LLMs' differential proficiency across languages through a multi-agent iterative repair paradigm. Our technique strategically translates defective code from languages where LLMs exhibit weaker repair capabilities to languages where they demonstrate stronger performance, without requiring additional training. A key innovation of our approach is an LLM-based decision-making system that dynamically selects optimal target languages based on bug characteristics and continuously incorporates feedback from previous repair attempts. We evaluate our method on xCodeEval, a comprehensive multilingual benchmark comprising 5,068 bugs across 11 programming languages. Results demonstrate significant enhancement in repair effectiveness, particularly for underrepresented languages, with Rust showing a 22.09% improvement in Pass@10 metrics. Our research provides the first empirical evidence that cross-language translation significantly expands the repair capabilities of LLMs and effectively bridges the performance gap between programming languages with different levels of popularity, opening new avenues for truly language-agnostic automated program repair.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06931v3">Non-Prehensile Tool-Object Manipulation by Integrating LLM-Based Planning and Manoeuvrability-Driven Controls</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      The ability to wield tools was once considered exclusive to human intelligence, but it's now known that many other animals, like crows, possess this capability. Yet, robotic systems still fall short of matching biological dexterity. In this paper, we investigate the use of Large Language Models (LLMs), tool affordances, and object manoeuvrability for non-prehensile tool-based manipulation tasks. Our novel method leverages LLMs based on scene information and natural language instructions to enable symbolic task planning for tool-object manipulation. This approach allows the system to convert the human language sentence into a sequence of feasible motion functions. We have developed a novel manoeuvrability-driven controller using a new tool affordance model derived from visual feedback. This controller helps guide the robot's tool utilization and manipulation actions, even within confined areas, using a stepping incremental approach. The proposed methodology is evaluated with experiments to prove its effectiveness under various manipulation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22458v1">Evaluating LLM-based Agents for Multi-Turn Conversations: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      This survey examines evaluation methods for large language model (LLM)-based agents in multi-turn conversational settings. Using a PRISMA-inspired framework, we systematically reviewed nearly 250 scholarly sources, capturing the state of the art from various venues of publication, and establishing a solid foundation for our analysis. Our study offers a structured approach by developing two interrelated taxonomy systems: one that defines \emph{what to evaluate} and another that explains \emph{how to evaluate}. The first taxonomy identifies key components of LLM-based agents for multi-turn conversations and their evaluation dimensions, including task completion, response quality, user experience, memory and context retention, as well as planning and tool integration. These components ensure that the performance of conversational agents is assessed in a holistic and meaningful manner. The second taxonomy system focuses on the evaluation methodologies. It categorizes approaches into annotation-based evaluations, automated metrics, hybrid strategies that combine human assessments with quantitative measures, and self-judging methods utilizing LLMs. This framework not only captures traditional metrics derived from language understanding, such as BLEU and ROUGE scores, but also incorporates advanced techniques that reflect the dynamic, interactive nature of multi-turn dialogues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22456v1">Entropy-guided sequence weighting for efficient exploration in RL-based LLM fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      We introduce Entropy-Guided Sequence Weighting (EGSW), a novel approach that enhances the exploration-exploitation tradeoff by dynamically assigning weights to generated outputs based on their advantage and entropy for Reinforcement Learning-based Large Language Model fine-tuning. EGSW integrates entropy regularization with advantage-based weighting to balance policy updates, enabling efficient exploration in high-dimensional state spaces. By employing temperature-scaled softmax weighting over sequences, EGSW prioritizing high-reward, high-uncertainty steps while maintaining training stability. Although originally developed to improve Group Relative Policy Optimization (GRPO) during large language model (LLM) fine-tuning, EGSW is generalizable to other reinforcement learning (RL) algorithms and can be implemented in both step-wise and trajectory-wise settings. Empirical evaluations demonstrate that EGSW enhances GRPO reasoning ability, yielding improvements in sample efficiency. Future work will explore the application of EGSW to advanced RL methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22424v1">CoSIL: Software Issue Localization via LLM-Driven Code Repository Graph Searching</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly advanced autonomous software engineering, leading to a growing number of software engineering agents that assist developers in automatic program repair. Issue localization forms the basis for accurate patch generation. However, because of limitations caused by the context window length of LLMs, existing issue localization methods face challenges in balancing concise yet effective contexts and adequately comprehensive search spaces. In this paper, we introduce CoSIL, an LLM driven, simple yet powerful function level issue localization method without training or indexing. CoSIL reduces the search space through module call graphs, iteratively searches the function call graph to obtain relevant contexts, and uses context pruning to control the search direction and manage contexts effectively. Importantly, the call graph is dynamically constructed by the LLM during search, eliminating the need for pre-parsing. Experiment results demonstrate that CoSIL achieves a Top-1 localization success rate of 43 percent and 44.6 percent on SWE bench Lite and SWE bench Verified, respectively, using Qwen2.5 Coder 32B, outperforming existing methods by 8.6 to 98.2 percent. When CoSIL is applied to guide the patch generation stage, the resolved rate further improves by 9.3 to 31.5 percent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22388v1">Why Stop at One Error? Benchmarking LLMs as Data Science Code Debuggers for Multi-Hop and Multi-Bug Errors</a></div>
    <div class="paper-meta">
      📅 2025-03-28
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      LLMs are transforming software development, yet current code generation and code repair benchmarks mainly assess syntactic and functional correctness in simple, single-error cases. LLMs' capabilities to autonomously find and fix runtime logical errors in complex data science code remain largely unexplored. To address this gap, we introduce DSDBench: the Data Science Debugging Benchmark, the first benchmark for systematic evaluation of LLMs on multi-hop error tracing and multi-bug detection in data science code debugging. DSDBench adapts datasets from existing data science task benchmarks, such as DABench and MatPlotBench, featuring realistic data science debugging tasks with automatically synthesized multi-hop, multi-bug code snippets. DSDBench includes 1,117 annotated samples with 741 cause-effect error pairs and runtime error messages. Evaluations of state-of-the-art LLMs on DSDBench show significant performance gaps, highlighting challenges in debugging logical runtime errors in data science code. DSDBench offers a crucial resource to evaluate and improve LLMs' debugging and reasoning capabilities, enabling more reliable AI-assisted data science in the future.DSDBench is publicly available at https://github.com/KevinCL16/DSDBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21480v2">OmniVox: Zero-Shot Emotion Recognition with Omni-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-28
      | 💬 Submitted to COLM 2025. Preprint
    </div>
    <details class="paper-abstract">
      The use of omni-LLMs (large language models that accept any modality as input), particularly for multimodal cognitive state tasks involving speech, is understudied. We present OmniVox, the first systematic evaluation of four omni-LLMs on the zero-shot emotion recognition task. We evaluate on two widely used multimodal emotion benchmarks: IEMOCAP and MELD, and find zero-shot omni-LLMs outperform or are competitive with fine-tuned audio models. Alongside our audio-only evaluation, we also evaluate omni-LLMs on text only and text and audio. We present acoustic prompting, an audio-specific prompting strategy for omni-LLMs which focuses on acoustic feature analysis, conversation context analysis, and step-by-step reasoning. We compare our acoustic prompting to minimal prompting and full chain-of-thought prompting techniques. We perform a context window analysis on IEMOCAP and MELD, and find that using context helps, especially on IEMOCAP. We conclude with an error analysis on the generated acoustic reasoning outputs from the omni-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22362v1">Supposedly Equivalent Facts That Aren't? Entity Frequency in Pre-training Induces Asymmetry in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      Understanding and mitigating hallucinations in Large Language Models (LLMs) is crucial for ensuring reliable content generation. While previous research has primarily focused on "when" LLMs hallucinate, our work explains "why" and directly links model behaviour to the pre-training data that forms their prior knowledge. Specifically, we demonstrate that an asymmetry exists in the recognition of logically equivalent facts, which can be attributed to frequency discrepancies of entities appearing as subjects versus objects. Given that most pre-training datasets are inaccessible, we leverage the fully open-source OLMo series by indexing its Dolma dataset to estimate entity frequencies. Using relational facts (represented as triples) from Wikidata5M, we construct probing datasets to isolate this effect. Our experiments reveal that facts with a high-frequency subject and a low-frequency object are better recognised than their inverse, despite their logical equivalence. The pattern reverses in low-to-high frequency settings, and no statistically significant asymmetry emerges when both entities are high-frequency. These findings highlight the influential role of pre-training data in shaping model predictions and provide insights for inferring the characteristics of pre-training data in closed or partially closed LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22329v1">A Refined Analysis of Massive Activations in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      Motivated in part by their relevance for low-precision training and quantization, massive activations in large language models (LLMs) have recently emerged as a topic of interest. However, existing analyses are limited in scope, and generalizability across architectures is unclear. This paper helps address some of these gaps by conducting an analysis of massive activations across a broad range of LLMs, including both GLU-based and non-GLU-based architectures. Our findings challenge several prior assumptions, most importantly: (1) not all massive activations are detrimental, i.e. suppressing them does not lead to an explosion of perplexity or a collapse in downstream task performance; (2) proposed mitigation strategies such as Attention KV bias are model-specific and ineffective in certain cases. We consequently investigate novel hybrid mitigation strategies; in particular pairing Target Variance Rescaling (TVR) with Attention KV bias or Dynamic Tanh (DyT) successfully balances the mitigation of massive activations with preserved downstream model performance in the scenarios we investigated. Our code is available at: https://github.com/bluorion-com/refine_massive_activations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21080v2">EQ-Negotiator: An Emotion-Reasoning LLM Agent in Credit Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      While large language model (LLM)-based chatbots have been applied for effective engagement in credit dialogues, their capacity for dynamic emotional expression remains limited. Current agents primarily rely on passive empathy rather than affective reasoning. For instance, when faced with persistent client negativity, the agent should employ strategic emotional adaptation by expressing measured anger to discourage counterproductive behavior and guide the conversation toward resolution. This context-aware emotional modulation is essential for imitating the nuanced decision-making of human negotiators. This paper introduces an EQ-negotiator that combines emotion sensing from pre-trained language models (PLMs) with emotional reasoning based on Game Theory and Hidden Markov Models. It takes into account both the current and historical emotions of the client to better manage and address negative emotions during interactions. By fine-tuning pre-trained language models (PLMs) on public emotion datasets and validating them on the credit dialogue datasets, our approach enables LLM-based agents to effectively capture shifts in client emotions and dynamically adjust their response tone based on our emotion decision policies in real-world financial negotiations. This EQ-negotiator can also help credit agencies foster positive client relationships, enhancing satisfaction in credit services.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22275v1">Make Some Noise: Towards LLM audio reasoning and generation using sound tokens</a></div>
    <div class="paper-meta">
      📅 2025-03-28
      | 💬 5 pages, 2 figures, Accepted at ICASSP 2025
    </div>
    <details class="paper-abstract">
      Integrating audio comprehension and generation into large language models (LLMs) remains challenging due to the continuous nature of audio and the resulting high sampling rates. Here, we introduce a novel approach that combines Variational Quantization with Conditional Flow Matching to convert audio into ultra-low bitrate discrete tokens of 0.23kpbs, allowing for seamless integration with text tokens in LLMs. We fine-tuned a pretrained text-based LLM using Low-Rank Adaptation (LoRA) to assess its effectiveness in achieving true multimodal capabilities, i.e., audio comprehension and generation. Our tokenizer outperforms a traditional VQ-VAE across various datasets with diverse acoustic events. Despite the substantial loss of fine-grained details through audio tokenization, our multimodal LLM trained with discrete tokens achieves competitive results in audio comprehension with state-of-the-art methods, though audio generation is poor. Our results highlight the need for larger, more diverse datasets and improved evaluation metrics to advance multimodal LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22250v1">Beyond the Script: Testing LLMs for Authentic Patient Communication Styles in Healthcare</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      Effective patient communication is pivotal in healthcare, yet traditional medical training often lacks exposure to diverse, challenging interpersonal dynamics. To bridge this gap, this study proposes the use of Large Language Models (LLMs) to simulate authentic patient communication styles, specifically the "accuser" and "rationalizer" personas derived from the Satir model, while also ensuring multilingual applicability to accommodate diverse cultural contexts and enhance accessibility for medical professionals. Leveraging advanced prompt engineering, including behavioral prompts, author's notes, and stubbornness mechanisms, we developed virtual patients (VPs) that embody nuanced emotional and conversational traits. Medical professionals evaluated these VPs, rating their authenticity (accuser: $3.8 \pm 1.0$; rationalizer: $3.7 \pm 0.8$ on a 5-point Likert scale (from one to five)) and correctly identifying their styles. Emotion analysis revealed distinct profiles: the accuser exhibited pain, anger, and distress, while the rationalizer displayed contemplation and calmness, aligning with predefined, detailed patient description including medical history. Sentiment scores (on a scale from zero to nine) further validated these differences in the communication styles, with the accuser adopting negative ($3.1 \pm 0.6$) and the rationalizer more neutral ($4.0 \pm 0.4$) tone. These results underscore LLMs' capability to replicate complex communication styles, offering transformative potential for medical education. This approach equips trainees to navigate challenging clinical scenarios by providing realistic, adaptable patient interactions, enhancing empathy and diagnostic acumen. Our findings advocate for AI-driven tools as scalable, cost-effective solutions to cultivate nuanced communication skills, setting a foundation for future innovations in healthcare training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22238v1">Integrating LLMs in Software Engineering Education: Motivators, Demotivators, and a Roadmap Towards a Framework for Finnish Higher Education Institutes</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      The increasing adoption of Large Language Models (LLMs) in software engineering education presents both opportunities and challenges. While LLMs offer benefits such as enhanced learning experiences, automated assessments, and personalized tutoring, their integration also raises concerns about academic integrity, student over-reliance, and ethical considerations. In this study, we conducted a preliminary literature review to identify motivators and demotivators for using LLMs in software engineering education. We applied a thematic mapping process to categorize and structure these factors (motivators and demotivators), offering a comprehensive view of their impact. In total, we identified 25 motivators and 30 demotivators, which are further organized into four high-level themes. This mapping provides a structured framework for understanding the factors that influence the integration of LLMs in software engineering education, both positively and negatively. As part of a larger research project, this study serves as a feasibility assessment, laying the groundwork for future systematic literature review and empirical studies. Ultimately, this project aims to develop a framework to assist Finnish higher education institutions in effectively integrating LLMs into software engineering education while addressing potential risks and challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08127v2">Fino1: On the Transferability of Reasoning Enhanced LLMs to Finance</a></div>
    <div class="paper-meta">
      📅 2025-03-28
      | 💬 13 pages, 2 figures, 3 Tables
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have shown strong general reasoning capabilities, their effectiveness in financial reasoning, which is crucial for real-world financial applications remains underexplored. In this study, we conduct a comprehensive evaluation of 24 state-of-the-art general and reasoning-focused LLMs across four complex financial reasoning tasks involving financial text, tabular data, and equations. We assess key capabilities such as numerical reasoning, tabular interpretation, financial terminology comprehension, long-context understanding, and equation-based problem solving. Our analysis reveals that while data quality and pretraining contribute to performance, general techniques like chain-of-thought (CoT) fine-tuning offer limited gains in financial tasks. To address this, we propose two domain-adapted models, Fino1-8B and Fino1-14B, trained with CoT fine-tuning and reinforcement learning using domain-specific reasoning paths. Our models are trained on a carefully curated dataset integrating high-quality examples from diverse sources, covering financial reports, tables, equations, and structured XBRL texts. Despite limited training data, they achieve an 7-9% performance improvement, outperforming several advanced LLMs, including GPT-o1, GPT-o3-mini, GPT-4.5, and comparable with DeepSeek models (V3 and R1), demonstrating strong practical value in resource, constrained scenarios. Our findings highlight the need for domain-specific adaptations in financial reasoning, and we release all datasets, models, and code for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12854v2">Enhancing LLM Reasoning with Iterative DPO: A Comprehensive Empirical Investigation</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      Recent advancements in post-training methodologies for large language models (LLMs) have highlighted reinforcement learning (RL) as a critical component for enhancing reasoning. However, the substantial computational costs associated with RL-based approaches have led to growing interest in alternative paradigms, such as Direct Preference Optimization (DPO). In this study, we investigate the effectiveness of DPO in facilitating self-improvement for LLMs through iterative preference-based learning. We demonstrate that a single round of DPO with coarse filtering significantly enhances mathematical reasoning performance, particularly for strong base model. Furthermore, we design an iterative enhancement framework for both the generator and the reward model (RM), enabling their mutual improvement through online interaction across multiple rounds of DPO. Finally, with simple verifiable rewards, our model DPO-VP achieves RL-level performance with significantly lower computational overhead. These findings highlight DPO as a scalable and cost-effective alternative to RL, offering a practical solution for enhancing LLM reasoning in resource-constrained situations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22097v1">Few-Shot Graph Out-of-Distribution Detection with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      Existing methods for graph out-of-distribution (OOD) detection typically depend on training graph neural network (GNN) classifiers using a substantial amount of labeled in-distribution (ID) data. However, acquiring high-quality labeled nodes in text-attributed graphs (TAGs) is challenging and costly due to their complex textual and structural characteristics. Large language models (LLMs), known for their powerful zero-shot capabilities in textual tasks, show promise but struggle to naturally capture the critical structural information inherent to TAGs, limiting their direct effectiveness. To address these challenges, we propose LLM-GOOD, a general framework that effectively combines the strengths of LLMs and GNNs to enhance data efficiency in graph OOD detection. Specifically, we first leverage LLMs' strong zero-shot capabilities to filter out likely OOD nodes, significantly reducing the human annotation burden. To minimize the usage and cost of the LLM, we employ it only to annotate a small subset of unlabeled nodes. We then train a lightweight GNN filter using these noisy labels, enabling efficient predictions of ID status for all other unlabeled nodes by leveraging both textual and structural information. After obtaining node embeddings from the GNN filter, we can apply informativeness-based methods to select the most valuable nodes for precise human annotation. Finally, we train the target ID classifier using these accurately annotated ID nodes. Extensive experiments on four real-world TAG datasets demonstrate that LLM-GOOD significantly reduces human annotation costs and outperforms state-of-the-art baselines in terms of both ID classification accuracy and OOD detection performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22092v1">Leveraging LLMs for Predicting Unknown Diagnoses from Clinical Notes</a></div>
    <div class="paper-meta">
      📅 2025-03-28
      | 💬 19 pages, 3 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Electronic Health Records (EHRs) often lack explicit links between medications and diagnoses, making clinical decision-making and research more difficult. Even when links exist, diagnosis lists may be incomplete, especially during early patient visits. Discharge summaries tend to provide more complete information, which can help infer accurate diagnoses, especially with the help of large language models (LLMs). This study investigates whether LLMs can predict implicitly mentioned diagnoses from clinical notes and link them to corresponding medications. We address two research questions: (1) Does majority voting across diverse LLM configurations outperform the best single configuration in diagnosis prediction? (2) How sensitive is majority voting accuracy to LLM hyperparameters such as temperature, top-p, and summary length? To evaluate, we created a new dataset of 240 expert-annotated medication-diagnosis pairs from 20 MIMIC-IV notes. Using GPT-3.5 Turbo, we ran 18 prompting configurations across short and long summary lengths, generating 8568 test cases. Results show that majority voting achieved 75 percent accuracy, outperforming the best single configuration at 66 percent. No single hyperparameter setting dominated, but combining deterministic, balanced, and exploratory strategies improved performance. Shorter summaries generally led to higher accuracy.In conclusion, ensemble-style majority voting with diverse LLM configurations improves diagnosis prediction in EHRs and offers a promising method to link medications and diagnoses in clinical texts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19820v3">Foot-In-The-Door: A Multi-turn Jailbreak for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-28
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Ensuring AI safety is crucial as large language models become increasingly integrated into real-world applications. A key challenge is jailbreak, where adversarial prompts bypass built-in safeguards to elicit harmful disallowed outputs. Inspired by psychological foot-in-the-door principles, we introduce FITD,a novel multi-turn jailbreak method that leverages the phenomenon where minor initial commitments lower resistance to more significant or more unethical transgressions. Our approach progressively escalates the malicious intent of user queries through intermediate bridge prompts and aligns the model's response by itself to induce toxic responses. Extensive experimental results on two jailbreak benchmarks demonstrate that FITD achieves an average attack success rate of 94% across seven widely used models, outperforming existing state-of-the-art methods. Additionally, we provide an in-depth analysis of LLM self-corruption, highlighting vulnerabilities in current alignment strategies and emphasizing the risks inherent in multi-turn interactions. The code is available at https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16833v4">KG4Diagnosis: A Hierarchical Multi-Agent LLM Framework with Knowledge Graph Enhancement for Medical Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-03-28
      | 💬 10 pages,5 figures,published to AAAI-25 Bridge Program
    </div>
    <details class="paper-abstract">
      Integrating Large Language Models (LLMs) in healthcare diagnosis demands systematic frameworks that can handle complex medical scenarios while maintaining specialized expertise. We present KG4Diagnosis, a novel hierarchical multi-agent framework that combines LLMs with automated knowledge graph construction, encompassing 362 common diseases across medical specialties. Our framework mirrors real-world medical systems through a two-tier architecture: a general practitioner (GP) agent for initial assessment and triage, coordinating with specialized agents for in-depth diagnosis in specific domains. The core innovation lies in our end-to-end knowledge graph generation methodology, incorporating: (1) semantic-driven entity and relation extraction optimized for medical terminology, (2) multi-dimensional decision relationship reconstruction from unstructured medical texts, and (3) human-guided reasoning for knowledge expansion. KG4Diagnosis serves as an extensible foundation for specialized medical diagnosis systems, with capabilities to incorporate new diseases and medical knowledge. The framework's modular design enables seamless integration of domain-specific enhancements, making it valuable for developing targeted medical diagnosis systems. We provide architectural guidelines and protocols to facilitate adoption across medical contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22877v1">Understanding Inequality of LLM Fact-Checking over Geographic Regions with Agent and Retrieval models</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      Fact-checking is a potentially useful application of Large Language Models (LLMs) to combat the growing dissemination of disinformation. However, the performance of LLMs varies across geographic regions. In this paper, we evaluate the factual accuracy of open and private models across a diverse set of regions and scenarios. Using a dataset containing 600 fact-checked statements balanced across six global regions we examine three experimental setups of fact-checking a statement: (1) when just the statement is available, (2) when an LLM-based agent with Wikipedia access is utilized, and (3) as a best case scenario when a Retrieval-Augmented Generation (RAG) system provided with the official fact check is employed. Our findings reveal that regardless of the scenario and LLM used, including GPT-4, Claude Sonnet, and LLaMA, statements from the Global North perform substantially better than those from the Global South. Furthermore, this gap is broadened for the more realistic case of a Wikipedia agent-based system, highlighting that overly general knowledge bases have a limited ability to address region-specific nuances. These results underscore the urgent need for better dataset balancing and robust retrieval strategies to enhance LLM fact-checking capabilities, particularly in geographically diverse contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22853v1">Teaching LLMs Music Theory with In-Context Learning and Chain-of-Thought Prompting: Pedagogical Strategies for Machines</a></div>
    <div class="paper-meta">
      📅 2025-03-28
      | 💬 11 pages, 4 figures, 3 tables. Published in Volume 1 of the Proceedings of the 17th International Conference on Computer Supported Music Education (CSME 2025). Presented on 3 April 2025 in Porto, Portugal
    </div>
    <details class="paper-abstract">
      This study evaluates the baseline capabilities of Large Language Models (LLMs) like ChatGPT, Claude, and Gemini to learn concepts in music theory through in-context learning and chain-of-thought prompting. Using carefully designed prompts (in-context learning) and step-by-step worked examples (chain-of-thought prompting), we explore how LLMs can be taught increasingly complex material and how pedagogical strategies for human learners translate to educating machines. Performance is evaluated using questions from an official Canadian Royal Conservatory of Music (RCM) Level 6 examination, which covers a comprehensive range of topics, including interval and chord identification, key detection, cadence classification, and metrical analysis. Additionally, we evaluate the suitability of various music encoding formats for these tasks (ABC, Humdrum, MEI, MusicXML). All experiments were run both with and without contextual prompts. Results indicate that without context, ChatGPT with MEI performs the best at 52%, while with context, Claude with MEI performs the best at 75%. Future work will further refine prompts and expand to cover more advanced music theory concepts. This research contributes to the broader understanding of teaching LLMs and has applications for educators, students, and developers of AI music tools alike.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04667v3">LLM Stability: A detailed analysis with some surprises</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      LLM (large language model) practitioners commonly notice that outputs can vary for the same inputs under settings expected to be deterministic. Yet the questions of how pervasive this is, and with what impact on results, have not to our knowledge been systematically investigated. We investigate non-determinism in five LLMs configured to be deterministic when applied to eight common tasks in across 10 runs, in both zero-shot and few-shot settings. We see accuracy variations up to 15% across naturally occurring runs with a gap of best possible performance to worst possible performance up to 70%. In fact, none of the LLMs consistently delivers repeatable accuracy across all tasks, much less identical output strings. Sharing preliminary results with insiders has revealed that non-determinism perhaps essential to the efficient use of compute resources via co-mingled data in input buffers so this issue is not going away anytime soon. To better quantify our observations, we introduce metrics focused on quantifying determinism, TARr@N for the total agreement rate at N runs over raw output, and TARa@N for total agreement rate of parsed-out answers. Our code and data are publicly available at http://github.com/REDACTED.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10741v3">SensorBench: Benchmarking LLMs in Coding-Based Sensor Processing</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      Effective processing, interpretation, and management of sensor data have emerged as a critical component of cyber-physical systems. Traditionally, processing sensor data requires profound theoretical knowledge and proficiency in signal-processing tools. However, recent works show that Large Language Models (LLMs) have promising capabilities in processing sensory data, suggesting their potential as copilots for developing sensing systems. To explore this potential, we construct a comprehensive benchmark, SensorBench, to establish a quantifiable objective. The benchmark incorporates diverse real-world sensor datasets for various tasks. The results show that while LLMs exhibit considerable proficiency in simpler tasks, they face inherent challenges in processing compositional tasks with parameter selections compared to engineering experts. Additionally, we investigate four prompting strategies for sensor processing and show that self-verification can outperform all other baselines in 48% of tasks. Our study provides a comprehensive benchmark and prompting analysis for future developments, paving the way toward an LLM-based sensor processing copilot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.22776v1">Post-Incorporating Code Structural Knowledge into LLMs via In-Context Learning for Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-03-28
    </div>
    <details class="paper-abstract">
      Code translation migrates codebases across programming languages. Recently, large language models (LLMs) have achieved significant advancements in software mining. However, handling the syntactic structure of source code remains a challenge. Classic syntax-aware methods depend on intricate model architectures and loss functions, rendering their integration into LLM training resource-intensive. This paper employs in-context learning (ICL), which directly integrates task exemplars into the input context, to post-incorporate code structural knowledge into pre-trained LLMs. We revisit exemplar selection in ICL from an information-theoretic perspective, proposing that list-wise selection based on information coverage is more precise and general objective than traditional methods based on combining similarity and diversity. To address the challenges of quantifying information coverage, we introduce a surrogate measure, Coverage of Abstract Syntax Tree (CAST). Furthermore, we formulate the NP-hard CAST maximization for exemplar selection and prove that it is a standard submodular maximization problem. Therefore, we propose a greedy algorithm for CAST submodular maximization, which theoretically guarantees a (1-1/e)-approximate solution in polynomial time complexity. Our method is the first training-free and model-agnostic approach to post-incorporate code structural knowledge into existing LLMs at test time. Experimental results show that our method significantly improves LLMs performance and reveals two meaningful insights: 1) Code structural knowledge can be effectively post-incorporated into pre-trained LLMs during inference, despite being overlooked during training; 2) Scaling up model size or training data does not lead to the emergence of code structural knowledge, underscoring the necessity of explicitly considering code syntactic structure.
    </details>
</div>
