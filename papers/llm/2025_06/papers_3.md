# llm - 2025_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16639v1">LLM-based Satisfiability Checking of String Requirements by Consistent Data and Checker Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-19
      | 💬 Accepted at the 33rd IEEE International Requirements Engineering 2025 conference
    </div>
    <details class="paper-abstract">
      Requirements over strings, commonly represented using natural language (NL), are particularly relevant for software systems due to their heavy reliance on string data manipulation. While individual requirements can usually be analyzed manually, verifying properties (e.g., satisfiability) over sets of NL requirements is particularly challenging. Formal approaches (e.g., SMT solvers) may efficiently verify such properties, but are known to have theoretical limitations. Additionally, the translation of NL requirements into formal constraints typically requires significant manual effort. Recently, large language models (LLMs) have emerged as an alternative approach for formal reasoning tasks, but their effectiveness in verifying requirements over strings is less studied. In this paper, we introduce a hybrid approach that verifies the satisfiability of NL requirements over strings by using LLMs (1) to derive a satisfiability outcome (and a consistent string, if possible), and (2) to generate declarative (i.e., SMT) and imperative (i.e., Python) checkers, used to validate the correctness of (1). In our experiments, we assess the performance of four LLMs. Results show that LLMs effectively translate natural language into checkers, even achieving perfect testing accuracy for Python-based checkers. These checkers substantially help LLMs in generating a consistent string and accurately identifying unsatisfiable requirements, leading to more than doubled generation success rate and F1-score in certain cases compared to baselines without generated checkers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16628v1">Initial Investigation of LLM-Assisted Development of Rule-Based Clinical NLP System</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      Despite advances in machine learning (ML) and large language models (LLMs), rule-based natural language processing (NLP) systems remain active in clinical settings due to their interpretability and operational efficiency. However, their manual development and maintenance are labor-intensive, particularly in tasks with large linguistic variability. To overcome these limitations, we proposed a novel approach employing LLMs solely during the rule-based systems development phase. We conducted the initial experiments focusing on the first two steps of developing a rule-based NLP pipeline: find relevant snippets from the clinical note; extract informative keywords from the snippets for the rule-based named entity recognition (NER) component. Our experiments demonstrated exceptional recall in identifying clinically relevant text snippets (Deepseek: 0.98, Qwen: 0.99) and 1.0 in extracting key terms for NER. This study sheds light on a promising new direction for NLP development, enabling semi-automated or automated development of rule-based systems with significantly faster, more cost-effective, and transparent execution compared with deep learning model-based solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16584v1">Measuring (a Sufficient) World Model in LLMs: A Variance Decomposition Framework</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      Understanding whether large language models (LLMs) possess a world model-a structured understanding of the world that supports generalization beyond surface-level patterns-is central to assessing their reliability, especially in high-stakes applications. We propose a formal framework for evaluating whether an LLM exhibits a sufficiently robust world model, defined as producing consistent outputs across semantically equivalent prompts while distinguishing between prompts that express different intents. We introduce a new evaluation approach to measure this that decomposes model response variability into three components: variability due to user purpose, user articulation, and model instability. An LLM with a strong world model should attribute most of the variability in its responses to changes in foundational purpose rather than superficial changes in articulation. This approach allows us to quantify how much of a model's behavior is semantically grounded rather than driven by model instability or alternative wording. We apply this framework to evaluate LLMs across diverse domains. Our results show how larger models attribute a greater share of output variability to changes in user purpose, indicating a more robust world model. This improvement is not uniform, however: larger models do not consistently outperform smaller ones across all domains, and their advantage in robustness is often modest. These findings highlight the importance of moving beyond accuracy-based benchmarks toward semantic diagnostics that more directly assess the structure and stability of a model's internal understanding of the world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04105v4">A Implies B: Circuit Analysis in LLMs for Propositional Logical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      Due to the size and complexity of modern large language models (LLMs), it has proven challenging to uncover the underlying mechanisms that models use to solve reasoning problems. For instance, is their reasoning for a specific problem localized to certain parts of the network? Do they break down the reasoning problem into modular components that are then executed as sequential steps as we go deeper in the model? To better understand the reasoning capability of LLMs, we study a minimal propositional logic problem that requires combining multiple facts to arrive at a solution. By studying this problem on Mistral and Gemma models, up to 27B parameters, we illuminate the core components the models use to solve such logic problems. From a mechanistic interpretability point of view, we use causal mediation analysis to uncover the pathways and components of the LLMs' reasoning processes. Then, we offer fine-grained insights into the functions of attention heads in different layers. We not only find a sparse circuit that computes the answer, but we decompose it into sub-circuits that have four distinct and modular uses. Finally, we reveal that three distinct models -- Mistral-7B, Gemma-2-9B and Gemma-2-27B -- contain analogous but not identical mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14028v2">MultiFinBen: A Multilingual, Multimodal, and Difficulty-Aware Benchmark for Financial LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have accelerated progress in financial NLP and applications, yet existing benchmarks remain limited to monolingual and unimodal settings, often over-relying on simple tasks and failing to reflect the complexity of real-world financial communication. We introduce MultiFinBen, the first multilingual and multimodal benchmark tailored to the global financial domain, evaluating LLMs across modalities (text, vision, audio) and linguistic settings (monolingual, bilingual, multilingual) on domain-specific tasks. We introduce two novel tasks, including PolyFiQA-Easy and PolyFiQA-Expert, the first multilingual financial benchmarks requiring models to perform complex reasoning over mixed-language inputs; and EnglishOCR and SpanishOCR, the first OCR-embedded financial QA tasks challenging models to extract and reason over information from visual-text financial documents. Moreover, we propose a dynamic, difficulty-aware selection mechanism and curate a compact, balanced benchmark rather than simple aggregation existing datasets. Extensive evaluation of 22 state-of-the-art models reveals that even the strongest models, despite their general multimodal and multilingual capabilities, struggle dramatically when faced with complex cross-lingual and multimodal tasks in financial domain. MultiFinBen is publicly released to foster transparent, reproducible, and inclusive progress in financial studies and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16548v1">Mr. Snuffleupagus at SemEval-2025 Task 4: Unlearning Factual Knowledge from LLMs Using Adaptive RMU</a></div>
    <div class="paper-meta">
      📅 2025-06-19
      | 💬 7 pages, 2 figures, to be published in SemEval-2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation. However, their tendency to memorize training data raises concerns regarding privacy, copyright compliance, and security, particularly in cases involving Personally Identifiable Information (PII). Effective machine unlearning techniques are essential to mitigate these risks, yet existing methods remain underdeveloped for LLMs due to their open-ended output space. In this work, we apply the Adaptive Representation Misdirection Unlearning (RMU) technique to unlearn sensitive information from LLMs. Through extensive experiments, we analyze the effects of unlearning across different decoder layers to determine the most effective regions for sensitive information removal. Our technique ranked 4th on the official leaderboard of both 1B parameter and 7B parameter models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16528v1">Aligning ASR Evaluation with Human and LLM Judgments: Intelligibility Metrics Using Phonetic, Semantic, and NLI Approaches</a></div>
    <div class="paper-meta">
      📅 2025-06-19
      | 💬 5 pages, 2 figures, Interspeech 2025
    </div>
    <details class="paper-abstract">
      Traditional ASR metrics like WER and CER fail to capture intelligibility, especially for dysarthric and dysphonic speech, where semantic alignment matters more than exact word matches. ASR systems struggle with these speech types, often producing errors like phoneme repetitions and imprecise consonants, yet the meaning remains clear to human listeners. We identify two key challenges: (1) Existing metrics do not adequately reflect intelligibility, and (2) while LLMs can refine ASR output, their effectiveness in correcting ASR transcripts of dysarthric speech remains underexplored. To address this, we propose a novel metric integrating Natural Language Inference (NLI) scores, semantic similarity, and phonetic similarity. Our ASR evaluation metric achieves a 0.890 correlation with human judgments on Speech Accessibility Project data, surpassing traditional methods and emphasizing the need to prioritize intelligibility over error-based measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16500v1">SparseLoRA: Accelerating LLM Fine-Tuning with Contextual Sparsity</a></div>
    <div class="paper-meta">
      📅 2025-06-19
      | 💬 ICML 2025. The first three authors contributed equally to this work. Project page: https://z-lab.ai/projects/sparselora
    </div>
    <details class="paper-abstract">
      Fine-tuning LLMs is both computationally and memory-intensive. While parameter-efficient fine-tuning methods, such as QLoRA and DoRA, reduce the number of trainable parameters and lower memory usage, they do not decrease computational cost. In some cases, they may even slow down fine-tuning. In this paper, we introduce SparseLoRA, a method that accelerates LLM fine-tuning through contextual sparsity. We propose a lightweight, training-free SVD sparsity estimator that dynamically selects a sparse subset of weights for loss and gradient computation. Also, we systematically analyze and address sensitivity across layers, tokens, and training steps. Our experimental results show that SparseLoRA reduces computational cost by up to 2.2 times and a measured speedup of up to 1.6 times while maintaining accuracy across various downstream tasks, including commonsense and arithmetic reasoning, code generation, and instruction following.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16440v1">Evaluating the Use of LLMs for Documentation to Code Traceability</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer new potential for automating documentation-to-code traceability, yet their capabilities remain underexplored. We present a comprehensive evaluation of LLMs (Claude 3.5 Sonnet, GPT-4o, and o3-mini) in establishing trace links between various software documentation (including API references and user guides) and source code. We create two novel datasets from two open-source projects (Unity Catalog and Crawl4AI). Through systematic experiments, we assess three key capabilities: (1) trace link identification accuracy, (2) relationship explanation quality, and (3) multi-step chain reconstruction. Results show that the best-performing LLM achieves F1-scores of 79.4% and 80.4% across the two datasets, substantially outperforming our baselines (TF-IDF, BM25, and CodeBERT). While fully correct relationship explanations range from 42.9% to 71.1%, partial accuracy exceeds 97%, indicating that fundamental connections are rarely missed. For multi-step chains, LLMs maintain high endpoint accuracy but vary in capturing precise intermediate links. Error analysis reveals that many false positives stem from naming-based assumptions, phantom links, or overgeneralization of architectural patterns. We demonstrate that task-framing, such as a one-to-many matching strategy, is critical for performance. These findings position LLMs as powerful assistants for trace discovery, but their limitations could necessitate human-in-the-loop tool design and highlight specific error patterns for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16411v1">When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework</a></div>
    <div class="paper-meta">
      📅 2025-06-19
      | 💬 under review
    </div>
    <details class="paper-abstract">
      We investigate the challenge of applying Large Language Models (LLMs) to long texts. We propose a theoretical framework that distinguishes the failure modes of long context tasks into three categories: cross-chunk dependence (task noise), confusion that grows with context size (model noise), and the imperfect integration of partial results (aggregator noise). Under this view, we analyze when it is effective to use multi-agent chunking, i.e., dividing a length sequence into smaller chunks and aggregating the processed results of each chunk. Our experiments on tasks such as retrieval, question answering, and summarization confirm both the theoretical analysis and the conditions that favor multi-agent chunking. By exploring superlinear model noise growth with input length, we also explain why, for large inputs, a weaker model configured with chunk-based processing can surpass a more advanced model like GPT4o applied in a single shot. Overall, we present a principled understanding framework and our results highlight a direct pathway to handling long contexts in LLMs with carefully managed chunking and aggregator strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16406v1">Drag-and-Drop LLMs: Zero-Shot Prompt-to-Weights</a></div>
    <div class="paper-meta">
      📅 2025-06-19
      | 💬 We propose a method that can generate LoRA parameters in seconds
    </div>
    <details class="paper-abstract">
      Modern Parameter-Efficient Fine-Tuning (PEFT) methods such as low-rank adaptation (LoRA) reduce the cost of customizing large language models (LLMs), yet still require a separate optimization run for every downstream dataset. We introduce \textbf{Drag-and-Drop LLMs (\textit{DnD})}, a prompt-conditioned parameter generator that eliminates per-task training by mapping a handful of unlabeled task prompts directly to LoRA weight updates. A lightweight text encoder distills each prompt batch into condition embeddings, which are then transformed by a cascaded hyper-convolutional decoder into the full set of LoRA matrices. Once trained in a diverse collection of prompt-checkpoint pairs, DnD produces task-specific parameters in seconds, yielding i) up to \textbf{12,000$\times$} lower overhead than full fine-tuning, ii) average gains up to \textbf{30\%} in performance over the strongest training LoRAs on unseen common-sense reasoning, math, coding, and multimodal benchmarks, and iii) robust cross-domain generalization despite never seeing the target data or labels. Our results demonstrate that prompt-conditioned parameter generation is a viable alternative to gradient-based adaptation for rapidly specializing LLMs. Our project is available at \href{https://jerryliang24.github.io/DnD}{https://jerryliang24.github.io/DnD}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16393v1">From LLM-anation to LLM-orchestrator: Coordinating Small Models for Data Labeling</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      Although the annotation paradigm based on Large Language Models (LLMs) has made significant breakthroughs in recent years, its actual deployment still has two core bottlenecks: first, the cost of calling commercial APIs in large-scale annotation is very expensive; second, in scenarios that require fine-grained semantic understanding, such as sentiment classification and toxicity classification, the annotation accuracy of LLMs is even lower than that of Small Language Models (SLMs) dedicated to this field. To address these problems, we propose a new paradigm of multi-model cooperative annotation and design a fully automatic annotation framework AutoAnnotator based on this. Specifically, AutoAnnotator consists of two layers. The upper-level meta-controller layer uses the generation and reasoning capabilities of LLMs to select SLMs for annotation, automatically generate annotation code and verify difficult samples; the lower-level task-specialist layer consists of multiple SLMs that perform annotation through multi-model voting. In addition, we use the difficult samples obtained by the secondary review of the meta-controller layer as the reinforcement learning set and fine-tune the SLMs in stages through a continual learning strategy, thereby improving the generalization of SLMs. Extensive experiments show that AutoAnnotator outperforms existing open-source/API LLMs in zero-shot, one-shot, CoT, and majority voting settings. Notably, AutoAnnotator reduces the annotation cost by 74.15% compared to directly annotating with GPT-3.5-turbo, while still improving the accuracy by 6.21%. Project page: https://github.com/Zhaiyuan-Ji/AutoAnnotator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11702v4">LLM-Guided Indoor Navigation with Multimodal Map Understanding</a></div>
    <div class="paper-meta">
      📅 2025-06-19
      | 💬 7 pages, 3 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Indoor navigation presents unique challenges due to complex layouts and the unavailability of GNSS signals. Existing solutions often struggle with contextual adaptation, and typically require dedicated hardware. In this work, we explore the potential of a Large Language Model (LLM), i.e., ChatGPT, to generate natural, context-aware navigation instructions from indoor map images. We design and evaluate test cases across different real-world environments, analyzing the effectiveness of LLMs in interpreting spatial layouts, handling user constraints, and planning efficient routes. Our findings demonstrate the potential of LLMs for supporting personalized indoor navigation, with an average of 86.59% correct indications and a maximum of 97.14%. The proposed system achieves high accuracy and reasoning performance. These results have key implications for AI-driven navigation and assistive technologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16196v1">Efficient and Privacy-Preserving Soft Prompt Transfer for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-19
      | 💬 Accepted at ICML2025
    </div>
    <details class="paper-abstract">
      Prompting has become a dominant paradigm for adapting large language models (LLMs). While discrete (textual) prompts are widely used for their interpretability, soft (parameter) prompts have recently gained traction in APIs. This is because they can encode information from more training samples while minimizing the user's token usage, leaving more space in the context window for task-specific input. However, soft prompts are tightly coupled to the LLM they are tuned on, limiting their generalization to other LLMs. This constraint is particularly problematic for efficiency and privacy: (1) tuning prompts on each LLM incurs high computational costs, especially as LLMs continue to grow in size. Additionally, (2) when the LLM is hosted externally, soft prompt tuning often requires sharing private data with the LLM provider. For instance, this is the case with the NVIDIA NeMo API. To address these issues, we propose POST (Privacy Of Soft prompt Transfer), a framework that enables private tuning of soft prompts on a small model and subsequently transfers these prompts to a larger LLM. POST uses knowledge distillation to derive a small model directly from the large LLM to improve prompt transferability, tunes the soft prompt locally, optionally with differential privacy guarantees, and transfers it back to the larger LLM using a small public dataset. Our experiments show that POST reduces computational costs, preserves privacy, and effectively transfers high-utility soft prompts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07863v2">Robust Hallucination Detection in LLMs via Adaptive Token Selection</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      Hallucinations in large language models (LLMs) pose significant safety concerns that impede their broader deployment. Recent research in hallucination detection has demonstrated that LLMs' internal representations contain truthfulness hints, which can be harnessed for detector training. However, the performance of these detectors is heavily dependent on the internal representations of predetermined tokens, fluctuating considerably when working on free-form generations with varying lengths and sparse distributions of hallucinated entities. To address this, we propose HaMI, a novel approach that enables robust detection of hallucinations through adaptive selection and learning of critical tokens that are most indicative of hallucinations. We achieve this robustness by an innovative formulation of the Hallucination detection task as Multiple Instance (HaMI) learning over token-level representations within a sequence, thereby facilitating a joint optimisation of token selection and hallucination detection on generation sequences of diverse forms. Comprehensive experimental results on four hallucination benchmarks show that HaMI significantly outperforms existing state-of-the-art approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16151v1">Under the Shadow of Babel: How Language Shapes Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-19
      | 💬 15 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Language is not only a tool for communication but also a medium for human cognition and reasoning. If, as linguistic relativity suggests, the structure of language shapes cognitive patterns, then large language models (LLMs) trained on human language may also internalize the habitual logical structures embedded in different languages. To examine this hypothesis, we introduce BICAUSE, a structured bilingual dataset for causal reasoning, which includes semantically aligned Chinese and English samples in both forward and reversed causal forms. Our study reveals three key findings: (1) LLMs exhibit typologically aligned attention patterns, focusing more on causes and sentence-initial connectives in Chinese, while showing a more balanced distribution in English. (2) Models internalize language-specific preferences for causal word order and often rigidly apply them to atypical inputs, leading to degraded performance, especially in Chinese. (3) When causal reasoning succeeds, model representations converge toward semantically aligned abstractions across languages, indicating a shared understanding beyond surface form. Overall, these results suggest that LLMs not only mimic surface linguistic forms but also internalize the reasoning biases shaped by language. Rooted in cognitive linguistic theory, this phenomenon is for the first time empirically verified through structural analysis of model internals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14321v2">Beyond Self-Talk: A Communication-Centric Survey of LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      Large language model-based multi-agent systems have recently gained significant attention due to their potential for complex, collaborative, and intelligent problem-solving capabilities. Existing surveys typically categorize LLM-based multi-agent systems (LLM-MAS) according to their application domains or architectures, overlooking the central role of communication in coordinating agent behaviors and interactions. To address this gap, this paper presents a comprehensive survey of LLM-MAS from a communication-centric perspective. Specifically, we propose a structured framework that integrates system-level communication (architecture, goals, and protocols) with system internal communication (strategies, paradigms, objects, and content), enabling a detailed exploration of how agents interact, negotiate, and achieve collective intelligence. Through an extensive analysis of recent literature, we identify key components in multiple dimensions and summarize their strengths and limitations. In addition, we highlight current challenges, including communication efficiency, security vulnerabilities, inadequate benchmarking, and scalability issues, and outline promising future research directions. This review aims to help researchers and practitioners gain a clear understanding of the communication mechanisms in LLM-MAS, thereby facilitating the design and deployment of robust, scalable, and secure multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.16136v1">Seeing is Fixing: Cross-Modal Reasoning with Multimodal LLMs for Visual Software Issue Fixing</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      Large language model-(LLM) based automated program repair (APR) techniques have shown promising results in resolving real-world GitHub issue tasks. Existing APR systems are primarily evaluated in unimodal settings (e.g., SWE-bench). However, these autonomous systems struggle to resolve multimodal problem scenarios (e.g., SWE-bench M) due to limitations in interpreting and leveraging visual information. In multimodal scenarios, LLMs need to rely on visual information in the graphical user interface (GUI) to understand bugs and generate fixes. To bridge this gap, we propose GUIRepair, a cross-modal reasoning approach for resolving multimodal issue scenarios by understanding and capturing visual information. Specifically, GUIRepair integrates two key components, Image2Code and Code2Image, to enhance fault comprehension and patch validation. Image2Code extracts relevant project documents based on the issue report, then applies this domain knowledge to generate the reproduced code responsible for the visual symptoms, effectively translating GUI images into executable context for better fault comprehension. Code2Image replays the visual issue scenario using the reproduced code and captures GUI renderings of the patched program to assess whether the fix visually resolves the issue, providing feedback for patch validation. We evaluate GUIRepair on SWE-bench M, and the approach demonstrates significant effectiveness. When utilizing GPT-4o as the base model, GUIRepair solves 157 instances, outperforming the best open-source baseline by 26 instances. Furthermore, when using o4-mini as the base model, GUIRepair can achieve even better results and solve 175 instances, outperforming the top commercial system by 22 instances. This emphasizes the success of our new perspective on incorporating cross-modal reasoning by understanding and capturing visual information to resolve multimodal issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.23804v2">DrunkAgent: Stealthy Memory Corruption in LLM-Powered Recommender Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-powered agents are increasingly used in recommender systems (RSs) to achieve personalized behavior modeling, where the memory mechanism plays a pivotal role in enabling the agents to autonomously explore, learn and self-evolve from real-world interactions. However, this very mechanism, serving as a contextual repository, inherently exposes an attack surface for potential adversarial manipulations. Despite its central role, the robustness of agentic RSs in the face of such threats remains largely underexplored. Previous works suffer from semantic mismatches or rely on static embeddings or pre-defined prompts, all of which hinder their applicability to systems with dynamic memory states. This challenge is exacerbated by the black-box nature of commercial RSs. To tackle the above problems, in this paper, we present the first systematic investigation of memory-based vulnerabilities in LLM-powered recommender agents, revealing their security limitations and guiding efforts to strengthen system resilience and trustworthiness. Specifically, we propose a novel black-box attack framework named DrunkAgent. DrunkAgent crafts semantically meaningful adversarial textual triggers for target item promotions and introduces a series of strategies to maximize the trigger effect by corrupting the memory updates during the interactions. The triggers and strategies are optimized on a surrogate model, enabling DrunkAgent transferable and stealthy. Extensive experiments on real-world datasets across diverse agentic RSs, including collaborative filtering, retrieval augmentation and sequential recommendations, demonstrate the generalizability, transferability and stealthiness of DrunkAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17655v3">AssistantX: An LLM-Powered Proactive Assistant in Collaborative Human-Populated Environment</a></div>
    <div class="paper-meta">
      📅 2025-06-19
      | 💬 8 pages, 10 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Current service robots suffer from limited natural language communication abilities, heavy reliance on predefined commands, ongoing human intervention, and, most notably, a lack of proactive collaboration awareness in human-populated environments. This results in narrow applicability and low utility. In this paper, we introduce AssistantX, an LLM-powered proactive assistant designed for autonomous operation in realworld scenarios with high accuracy. AssistantX employs a multi-agent framework consisting of 4 specialized LLM agents, each dedicated to perception, planning, decision-making, and reflective review, facilitating advanced inference capabilities and comprehensive collaboration awareness, much like a human assistant by your side. We built a dataset of 210 real-world tasks to validate AssistantX, which includes instruction content and status information on whether relevant personnel are available. Extensive experiments were conducted in both text-based simulations and a real office environment over the course of a month and a half. Our experiments demonstrate the effectiveness of the proposed framework, showing that AssistantX can reactively respond to user instructions, actively adjust strategies to adapt to contingencies, and proactively seek assistance from humans to ensure successful task completion. More details and videos can be found at https://assistantx-agent.github.io/AssistantX/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06904v3">Re-TASK: Revisiting LLM Tasks from Capability, Skill, and Knowledge Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-06-19
      | 💬 ACL 2025 Findings; First three authors contributed equally
    </div>
    <details class="paper-abstract">
      The Chain-of-Thought (CoT) paradigm has become a pivotal method for solving complex problems with large language models (LLMs). However, its application to domain-specific tasks remains challenging, as LLMs often fail to decompose tasks accurately or execute subtasks effectively. This paper introduces the Re-TASK framework, a novel theoretical model that revisits LLM tasks from capability, skill, and knowledge perspectives, drawing on the principles of Bloom's Taxonomy and Knowledge Space Theory. While CoT provides a workflow-centric perspective on tasks, Re-TASK introduces a Chain-of-Learning (CoL) paradigm that highlights task dependencies on specific capability items, further broken down into their constituent knowledge and skill components. To address CoT failures, we propose a Re-TASK prompting strategy, which strengthens task-relevant capabilities through targeted knowledge injection and skill adaptation. Experiments across diverse domains demonstrate the effectiveness of Re-TASK. In particular, we achieve improvements of 45.00% on Yi-1.5-9B and 24.50% on Llama3-Chinese-8B for legal tasks. These results highlight the potential of Re-TASK to significantly enhance LLM performance and its applicability in specialized domains. We release our code and data at https://github.com/Uylee/Re-TASK.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13487v2">Detecting Prefix Bias in LLM-based Reward Models</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Human Feedback (RLHF) has emerged as a key paradigm for task-specific fine-tuning of language models using human preference data. While numerous publicly available preference datasets provide pairwise comparisons of responses, the potential for biases in the resulting reward models remains underexplored. In this work, we introduce novel methods to detect and evaluate prefix bias -- a systematic shift in model preferences triggered by minor variations in query prefixes -- in LLM-based reward models trained on such datasets. We leverage these metrics to reveal significant biases in preference models across racial and gender dimensions. Our comprehensive evaluation spans diverse open-source preference datasets and reward model architectures, demonstrating susceptibility to this kind of bias regardless of the underlying model architecture. Furthermore, we propose a data augmentation strategy to mitigate these biases, showing its effectiveness in reducing the impact of prefix bias. Our findings highlight the critical need for bias-aware dataset design and evaluation in developing fair and reliable reward models, contributing to the broader discourse on fairness in AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15975v1">Multi-use LLM Watermarking and the False Detection Problem</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      Digital watermarking is a promising solution for mitigating some of the risks arising from the misuse of automatically generated text. These approaches either embed non-specific watermarks to allow for the detection of any text generated by a particular sampler, or embed specific keys that allow the identification of the LLM user. However, simultaneously using the same embedding for both detection and user identification leads to a false detection problem, whereby, as user capacity grows, unwatermarked text is increasingly likely to be falsely detected as watermarked. Through theoretical analysis, we identify the underlying causes of this phenomenon. Building on these insights, we propose Dual Watermarking which jointly encodes detection and identification watermarks into generated text, significantly reducing false positives while maintaining high detection accuracy. Our experimental results validate our theoretical findings and demonstrate the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15961v1">TrainVerify: Equivalence-Based Verification for Distributed LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) at scale requires parallel execution across thousands of devices, incurring enormous computational costs. Yet, these costly distributed trainings are rarely verified, leaving them prone to silent errors and potentially wasting millions of GPU hours. We introduce TrainVerify, a system for verifiable distributed training of LLMs. Given a deep learning model's logical specification as the ground truth, TrainVerify formally verifies that a distributed parallel execution plan is mathematically equivalent to it. Direct verification is notoriously difficult due to the sheer scale of LLMs which often involves billions of variables and highly intricate computation graphs. Therefore, TrainVerify introduces shape-reduction techniques and a stage-wise parallel verification algorithm that significantly reduces complexity while preserving formal correctness. TrainVerify scales to frontier LLMs, including the successful verification of the Llama3 (405B) and DeepSeek-V3 (671B) training plans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15947v1">HybridRAG-based LLM Agents for Low-Carbon Optimization in Low-Altitude Economy Networks</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      Low-Altitude Economy Networks (LAENets) are emerging as a promising paradigm to support various low-altitude services through integrated air-ground infrastructure. To satisfy low-latency and high-computation demands, the integration of Unmanned Aerial Vehicles (UAVs) with Mobile Edge Computing (MEC) systems plays a vital role, which offloads computing tasks from terminal devices to nearby UAVs, enabling flexible and resilient service provisions for ground users. To promote the development of LAENets, it is significant to achieve low-carbon multi-UAV-assisted MEC networks. However, several challenges hinder this implementation, including the complexity of multi-dimensional UAV modeling and the difficulty of multi-objective coupled optimization. To this end, this paper proposes a novel Retrieval Augmented Generation (RAG)-based Large Language Model (LLM) agent framework for model formulation. Specifically, we develop HybridRAG by combining KeywordRAG, VectorRAG, and GraphRAG, empowering LLM agents to efficiently retrieve structural information from expert databases and generate more accurate optimization problems compared with traditional RAG-based LLM agents. After customizing carbon emission optimization problems for multi-UAV-assisted MEC networks, we propose a Double Regularization Diffusion-enhanced Soft Actor-Critic (R\textsuperscript{2}DSAC) algorithm to solve the formulated multi-objective optimization problem. The R\textsuperscript{2}DSAC algorithm incorporates diffusion entropy regularization and action entropy regularization to improve the performance of the diffusion policy. Furthermore, we dynamically mask unimportant neurons in the actor network to reduce the carbon emissions associated with model training. Simulation results demonstrate the effectiveness and reliability of the proposed HybridRAG-based LLM agent framework and the R\textsuperscript{2}DSAC algorithm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15928v1">Exploring Big Five Personality and AI Capability Effects in LLM-Simulated Negotiation Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-06-19
      | 💬 Under review for KDD 2025 Workshop on Evaluation and Trustworthiness of Agentic and Generative AI Models
    </div>
    <details class="paper-abstract">
      This paper presents an evaluation framework for agentic AI systems in mission-critical negotiation contexts, addressing the need for AI agents that can adapt to diverse human operators and stakeholders. Using Sotopia as a simulation testbed, we present two experiments that systematically evaluated how personality traits and AI agent characteristics influence LLM-simulated social negotiation outcomes--a capability essential for a variety of applications involving cross-team coordination and civil-military interactions. Experiment 1 employs causal discovery methods to measure how personality traits impact price bargaining negotiations, through which we found that Agreeableness and Extraversion significantly affect believability, goal achievement, and knowledge acquisition outcomes. Sociocognitive lexical measures extracted from team communications detected fine-grained differences in agents' empathic communication, moral foundations, and opinion patterns, providing actionable insights for agentic AI systems that must operate reliably in high-stakes operational scenarios. Experiment 2 evaluates human-AI job negotiations by manipulating both simulated human personality and AI system characteristics, specifically transparency, competence, adaptability, demonstrating how AI agent trustworthiness impact mission effectiveness. These findings establish a repeatable evaluation methodology for experimenting with AI agent reliability across diverse operator personalities and human-agent team dynamics, directly supporting operational requirements for reliable AI systems. Our work advances the evaluation of agentic AI workflows by moving beyond standard performance metrics to incorporate social dynamics essential for mission success in complex operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17336v1">Privacy-Preserving LLM Interaction with Socratic Chain-of-Thought Reasoning and Homomorphically Encrypted Vector Databases</a></div>
    <div class="paper-meta">
      📅 2025-06-19
      | 💬 29 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as personal agents, accessing sensitive user data such as calendars, emails, and medical records. Users currently face a trade-off: They can send private records, many of which are stored in remote databases, to powerful but untrusted LLM providers, increasing their exposure risk. Alternatively, they can run less powerful models locally on trusted devices. We bridge this gap. Our Socratic Chain-of-Thought Reasoning first sends a generic, non-private user query to a powerful, untrusted LLM, which generates a Chain-of-Thought (CoT) prompt and detailed sub-queries without accessing user data. Next, we embed these sub-queries and perform encrypted sub-second semantic search using our Homomorphically Encrypted Vector Database across one million entries of a single user's private data. This represents a realistic scale of personal documents, emails, and records accumulated over years of digital activity. Finally, we feed the CoT prompt and the decrypted records to a local language model and generate the final response. On the LoCoMo long-context QA benchmark, our hybrid framework, combining GPT-4o with a local Llama-3.2-1B model, outperforms using GPT-4o alone by up to 7.1 percentage points. This demonstrates a first step toward systems where tasks are decomposed and split between untrusted strong LLMs and weak local ones, preserving user privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17335v1">LMR-BENCH: Evaluating LLM Agent's Ability on Reproducing Language Modeling Research</a></div>
    <div class="paper-meta">
      📅 2025-06-19
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated remarkable potential in advancing scientific discovery. However, their capability in the fundamental yet crucial task of reproducing code from research papers, especially in the NLP domain, remains underexplored. This task includes unique complex reasoning challenges in the intellectual synthesis of abstract concepts and the comprehension of code repositories with interdependent files. Motivated by this gap, we present LMR-BENCH, a benchmark designed to systematically evaluate the capability of LLM agents on code reproduction from Language Modeling Research. It consists of 28 code reproduction tasks derived from 23 research papers published in top-tier NLP venues over the past five years, spanning nine fundamental categories. Models are provided with a research paper, a code repository containing one or more masked functions, and instructions for implementing these functions. We conduct extensive experiments in standard prompting and LLM agent settings with state-of-the-art LLMs, evaluating the accuracy of unit tests and performing LLM-based evaluation of code correctness. Experimental results reveal that even the most advanced models still exhibit persistent limitations in scientific reasoning and code synthesis, highlighting critical gaps in LLM agents' ability to autonomously reproduce scientific research
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15683v1">PhantomHunter: Detecting Unseen Privately-Tuned LLM-Generated Text via Family-Aware Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 17 pages, 3 figures, 6 tables
    </div>
    <details class="paper-abstract">
      With the popularity of large language models (LLMs), undesirable societal problems like misinformation production and academic misconduct have been more severe, making LLM-generated text detection now of unprecedented importance. Although existing methods have made remarkable progress, a new challenge posed by text from privately tuned LLMs remains underexplored. Users could easily possess private LLMs by fine-tuning an open-source one with private corpora, resulting in a significant performance drop of existing detectors in practice. To address this issue, we propose PhantomHunter, an LLM-generated text detector specialized for detecting text from unseen, privately-tuned LLMs. Its family-aware learning framework captures family-level traits shared across the base models and their derivatives, instead of memorizing individual characteristics. Experiments on data from LLaMA, Gemma, and Mistral families show its superiority over 7 baselines and 3 industrial services, with F1 scores of over 96%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15656v1">PhishDebate: An LLM-Based Multi-Agent Framework for Phishing Website Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Phishing websites continue to pose a significant cybersecurity threat, often leveraging deceptive structures, brand impersonation, and social engineering tactics to evade detection. While recent advances in large language models (LLMs) have enabled improved phishing detection through contextual understanding, most existing approaches rely on single-agent classification facing the risks of hallucination and lack interpretability or robustness. To address these limitations, we propose PhishDebate, a modular multi-agent LLM-based debate framework for phishing website detection. PhishDebate employs four specialized agents to independently analyze different textual aspects of a webpage--URL structure, HTML composition, semantic content, and brand impersonation--under the coordination of a Moderator and a final Judge. Through structured debate and divergent thinking, the framework delivers more accurate and interpretable decisions. Extensive evaluations on commercial LLMs demonstrate that PhishDebate achieves 98.2% recall and 98.2% True Positive Rate (TPR) on a real-world phishing dataset, and outperforms single-agent and Chain of Thought (CoT) baselines. Additionally, its modular design allows agent-level configurability, enabling adaptation to varying resource and application requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15648v1">deepSURF: Detecting Memory Safety Vulnerabilities in Rust Through Fuzzing LLM-Augmented Harnesses</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Although Rust ensures memory safety by default, it also permits the use of unsafe code, which can introduce memory safety vulnerabilities if misused. Unfortunately, existing tools for detecting memory bugs in Rust typically exhibit limited detection capabilities, inadequately handle Rust-specific types, or rely heavily on manual intervention. To address these limitations, we present deepSURF, a tool that integrates static analysis with Large Language Model (LLM)-guided fuzzing harness generation to effectively identify memory safety vulnerabilities in Rust libraries, specifically targeting unsafe code. deepSURF introduces a novel approach for handling generics by substituting them with custom types and generating tailored implementations for the required traits, enabling the fuzzer to simulate user-defined behaviors within the fuzzed library. Additionally, deepSURF employs LLMs to augment fuzzing harnesses dynamically, facilitating exploration of complex API interactions and significantly increasing the likelihood of exposing memory safety vulnerabilities. We evaluated deepSURF on 27 real-world Rust crates, successfully rediscovering 20 known memory safety bugs and uncovering 6 previously unknown vulnerabilities, demonstrating clear improvements over state-of-the-art tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16065v2">Aug2Search: Enhancing Facebook Marketplace Search with LLM-Generated Synthetic Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Embedding-Based Retrieval (EBR) is an important technique in modern search engines, enabling semantic match between search queries and relevant results. However, search logging data on platforms like Facebook Marketplace lacks the diversity and details needed for effective EBR model training, limiting the models' ability to capture nuanced search patterns. To address this challenge, we propose Aug2Search, an EBR-based framework leveraging synthetic data generated by Generative AI (GenAI) models, in a multimodal and multitask approach to optimize query-product relevance. This paper investigates the capabilities of GenAI, particularly Large Language Models (LLMs), in generating high-quality synthetic data, and analyzing its impact on enhancing EBR models. We conducted experiments using eight Llama models and 100 million data points from Facebook Marketplace logs. Our synthetic data generation follows three strategies: (1) generate queries, (2) enhance product listings, and (3) generate queries from enhanced listings. We train EBR models on three different datasets: sampled engagement data or original data ((e.g., "Click" and "Listing Interactions")), synthetic data, and a mixture of both engagement and synthetic data to assess their performance across various training sets. Our findings underscore the robustness of Llama models in producing synthetic queries and listings with high coherence, relevance, and diversity, while maintaining low levels of hallucination. Aug2Search achieves an improvement of up to 4% in ROC_AUC with 100 million synthetic data samples, demonstrating the effectiveness of our approach. Moreover, our experiments reveal that with the same volume of training data, models trained exclusively on synthetic data often outperform those trained on original data only or a mixture of original and synthetic data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15624v1">The Effect of State Representation on LLM Agent Behavior in Dynamic Routing Games</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 27 pages, 20 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise as decision-makers in dynamic settings, but their stateless nature necessitates creating a natural language representation of history. We present a unifying framework for systematically constructing natural language "state" representations for prompting LLM agents in repeated multi-agent games. Previous work on games with LLM agents has taken an ad hoc approach to encoding game history, which not only obscures the impact of state representation on agents' behavior, but also limits comparability between studies. Our framework addresses these gaps by characterizing methods of state representation along three axes: action informativeness (i.e., the extent to which the state representation captures actions played); reward informativeness (i.e., the extent to which the state representation describes rewards obtained); and prompting style (or natural language compression, i.e., the extent to which the full text history is summarized). We apply this framework to a dynamic selfish routing game, chosen because it admits a simple equilibrium both in theory and in human subject experiments \cite{rapoport_choice_2009}. Despite the game's relative simplicity, we find that there are key dependencies of LLM agent behavior on the natural language state representation. In particular, we observe that representations which provide agents with (1) summarized, rather than complete, natural language representations of past history; (2) information about regrets, rather than raw payoffs; and (3) limited information about others' actions lead to behavior that more closely matches game theoretic equilibrium predictions, and with more stable game play by the agents. By contrast, other representations can exhibit either large deviations from equilibrium, higher variation in dynamic game play over time, or both.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09033v2">Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 Code is available at https://github.com/ulab-uiuc/Router-R1. Models and Datasets are available at https://huggingface.co/collections/ulab-ai/router-r1-6851bbe099c7a56914b5db03
    </div>
    <details class="paper-abstract">
      The rapid emergence of diverse large language models (LLMs) has spurred the development of LLM routers that assign user queries to the most suitable model. However, existing LLM routers typically perform a single-round, one-to-one mapping (\textit{i.e.}, assigning each query to a single model in isolation), which limits their capability to tackle complex tasks that demand the complementary strengths of multiple LLMs. In this paper, we present \textbf{Router-R1}, a reinforcement learning (RL)-based framework that formulates multi-LLM routing and aggregation as a sequential decision process. Router-R1 instantiates the router itself as a capable LLM, leveraging its reasoning ability to interleave "think" actions (internal deliberation) with "route" actions (dynamic model invocation), and integrates each response into its evolving context. To facilitate learning, we employ a lightweight rule-based reward comprising format rewards, final outcome rewards, and a novel cost reward for optimizing the balance between performance and cost, opening a pathway toward enhancing performance-cost trade-offs via RL. Router-R1 also conditions only on simple model descriptors such as pricing, latency, and example performance, enabling strong generalization to unseen model selection. Experiments on seven general and multi-hop QA benchmarks show that Router-R1 outperforms several strong baselines, achieving superior performance while maintaining robust generalization and cost management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15606v1">LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become indispensable in real-world applications. However, their widespread adoption raises significant safety concerns, particularly in responding to socially harmful questions. Despite substantial efforts to improve model safety through alignment, aligned models can still have their safety protections undermined by subsequent fine-tuning - even when the additional training data appears benign. In this paper, we empirically demonstrate that this vulnerability stems from the sensitivity of safety-critical low-rank subspaces in LLM parameters to fine-tuning. Building on this insight, we propose a novel training-free method, termed Low-Rank Extrapolation (LoX), to enhance safety robustness by extrapolating the safety subspace of an aligned LLM. Our experimental results confirm the effectiveness of LoX, demonstrating significant improvements in robustness against both benign and malicious fine-tuning attacks while preserving the model's adaptability to new tasks. For instance, LoX leads to 11% to 54% absolute reductions in attack success rates (ASR) facing benign or malicious fine-tuning attacks. By investigating the ASR landscape of parameters, we attribute the success of LoX to that the extrapolation moves LLM parameters to a flatter zone, thereby less sensitive to perturbations. The code is available at github.com/VITA-Group/LoX.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15567v1">Managing Complex Failure Analysis Workflows with LLM-based Reasoning and Acting Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Failure Analysis (FA) is a highly intricate and knowledge-intensive process. The integration of AI components within the computational infrastructure of FA labs has the potential to automate a variety of tasks, including the detection of non-conformities in images, the retrieval of analogous cases from diverse data sources, and the generation of reports from annotated images. However, as the number of deployed AI models increases, the challenge lies in orchestrating these components into cohesive and efficient workflows that seamlessly integrate with the FA process. This paper investigates the design and implementation of a Large Language Model (LLM)-based Planning Agent (LPA) to assist FA engineers in solving their analysis cases. The LPA integrates LLMs with advanced planning capabilities and external tool utilization, enabling autonomous processing of complex queries, retrieval of relevant data from external systems, and generation of human-readable responses. Evaluation results demonstrate the agent's operational effectiveness and reliability in supporting FA tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15522v1">Lessons from Training Grounded LLMs with Verifiable Rewards</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Generating grounded and trustworthy responses remains a key challenge for large language models (LLMs). While retrieval-augmented generation (RAG) with citation-based grounding holds promise, instruction-tuned models frequently fail even in straightforward scenarios: missing explicitly stated answers, citing incorrectly, or refusing when evidence is available. In this work, we explore how reinforcement learning (RL) and internal reasoning can enhance grounding in LLMs. We use the GRPO (Group Relative Policy Optimization) method to train models using verifiable outcome-based rewards targeting answer correctness, citation sufficiency, and refusal quality, without requiring gold reasoning traces or expensive annotations. Through comprehensive experiments across ASQA, QAMPARI, ELI5, and ExpertQA we show that reasoning-augmented models significantly outperform instruction-only variants, especially in handling unanswerable queries and generating well-cited responses. A two-stage training setup, first optimizing answer and citation behavior and then refusal, further improves grounding by stabilizing the learning signal. Additionally, we revisit instruction tuning via GPT-4 distillation and find that combining it with GRPO enhances performance on long-form, generative QA tasks. Overall, our findings highlight the value of reasoning, stage-wise optimization, and outcome-driven RL for building more verifiable and reliable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15513v1">RePCS: Diagnosing Data Memorization in LLM-Powered Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 11 pages, 7 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) has become a common strategy for updating large language model (LLM) responses with current, external information. However, models may still rely on memorized training data, bypass the retrieved evidence, and produce contaminated outputs. We introduce Retrieval-Path Contamination Scoring (RePCS), a diagnostic method that detects such behavior without requiring model access or retraining. RePCS compares two inference paths: (i) a parametric path using only the query, and (ii) a retrieval-augmented path using both the query and retrieved context by computing the Kullback-Leibler (KL) divergence between their output distributions. A low divergence suggests that the retrieved context had minimal impact, indicating potential memorization. This procedure is model-agnostic, requires no gradient or internal state access, and adds only a single additional forward pass. We further derive PAC-style guarantees that link the KL threshold to user-defined false positive and false negative rates. On the Prompt-WNQA benchmark, RePCS achieves a ROC-AUC of 0.918. This result outperforms the strongest prior method by 6.5 percentage points while keeping latency overhead below 4.7% on an NVIDIA T4 GPU. RePCS offers a lightweight, black-box safeguard to verify whether a RAG system meaningfully leverages retrieval, making it especially valuable in safety-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11511v2">Position Paper: Rethinking Privacy in RL for Sequential Decision-making in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 IJCNN 2025 Position Paper Track
    </div>
    <details class="paper-abstract">
      The rise of reinforcement learning (RL) in critical real-world applications demands a fundamental rethinking of privacy in AI systems. Traditional privacy frameworks, designed to protect isolated data points, fall short for sequential decision-making systems where sensitive information emerges from temporal patterns, behavioral strategies, and collaborative dynamics. Modern RL paradigms, such as federated RL (FedRL) and RL with human feedback (RLHF) in large language models (LLMs), exacerbate these challenges by introducing complex, interactive, and context-dependent learning environments that traditional methods do not address. In this position paper, we argue for a new privacy paradigm built on four core principles: multi-scale protection, behavioral pattern protection, collaborative privacy preservation, and context-aware adaptation. These principles expose inherent tensions between privacy, utility, and interpretability that must be navigated as RL systems become more pervasive in high-stakes domains like healthcare, autonomous vehicles, and decision support systems powered by LLMs. To tackle these challenges, we call for the development of new theoretical frameworks, practical mechanisms, and rigorous evaluation methodologies that collectively enable effective privacy protection in sequential decision-making systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15461v1">All is Not Lost: LLM Recovery without Checkpoints</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Training LLMs on decentralized and wimpy computation nodes, e.g., multiple on-spot instances, lowers the training cost and enables model democratization. The inevitable challenge here is the churn of nodes due to failures and the operator's scheduling policies, leading to losing a stage - a part of the model. The conventional approaches to recover from failures are to either use checkpointing, where periodically a copy of the entire model is sent to an additional storage, or redundant computation. These approaches yield significant communication and/or computation overhead even in non-failure cases and scale poorly in settings with large models. In this paper, we propose, CheckFree, an efficient recovery method where a failing stage is substituted by a weighted average of the closest neighboring stages. In contrast to the state of the art, CheckFree requires no additional computation or storage. However, because of the nature of averaging neighbouring stages, it can only recover failures of intermediate stages. We further extend our method to CheckFree+ with out-of-order pipeline execution to tolerate crashes of the first and last stages. Thanks to out-of-order pipelining, behaviour of those stages is mimicked by their neighboring ones, which allows CheckFree+ to recover them by simply copying the weights from the immediate neighbour. To be able to recover the (de)embedding layers, CheckFree+ copies those layers to the neighboring stages, which requires relatively small storage overhead. We extensively evaluate our method on LLaMa models of model sizes from 124M to 1.5B with varying failure frequencies. In the case of low and medium failure rates (5-10%), CheckFree and CheckFree+ outperform both checkpointing and redundant computation in terms of convergence in wall-clock time by over 12%. Both of our proposals can be run via our code available at: https://github.com/gensyn-ai/CheckFree.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15453v1">Uncovering Intention through LLM-Driven Code Snippet Description Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 6 pages, 3 figures, 4 tables, conference paper
    </div>
    <details class="paper-abstract">
      Documenting code snippets is essential to pinpoint key areas where both developers and users should pay attention. Examples include usage examples and other Application Programming Interfaces (APIs), which are especially important for third-party libraries. With the rise of Large Language Models (LLMs), the key goal is to investigate the kinds of description developers commonly use and evaluate how well an LLM, in this case Llama, can support description generation. We use NPM Code Snippets, consisting of 185,412 packages with 1,024,579 code snippets. From there, we use 400 code snippets (and their descriptions) as samples. First, our manual classification found that the majority of original descriptions (55.5%) highlight example-based usage. This finding emphasizes the importance of clear documentation, as some descriptions lacked sufficient detail to convey intent. Second, the LLM correctly identified the majority of original descriptions as "Example" (79.75%), which is identical to our manual finding, showing a propensity for generalization. Third, compared to the originals, the produced description had an average similarity score of 0.7173, suggesting relevance but room for improvement. Scores below 0.9 indicate some irrelevance. Our results show that depending on the task of the code snippet, the intention of the document may differ from being instructions for usage, installations, or descriptive learning examples for any user of a library.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15451v1">AgentGroupChat-V2: Divide-and-Conquer Is What LLM-Based Multi-Agent System Need</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Large language model based multi-agent systems have demonstrated significant potential in social simulation and complex task resolution domains. However, current frameworks face critical challenges in system architecture design, cross-domain generalizability, and performance guarantees, particularly as task complexity and number of agents increases. We introduces AgentGroupChat-V2, a novel framework addressing these challenges through three core innovations: (1) a divide-and-conquer fully parallel architecture that decomposes user queries into hierarchical task forest structures enabling dependency management and distributed concurrent processing. (2) an adaptive collaboration engine that dynamically selects heterogeneous LLM combinations and interaction modes based on task characteristics. (3) agent organization optimization strategies combining divide-and-conquer approaches for efficient problem decomposition. Extensive experiments demonstrate AgentGroupChat-V2's superior performance across diverse domains, achieving 91.50% accuracy on GSM8K (exceeding the best baseline by 5.6 percentage points), 30.4% accuracy on competition-level AIME (nearly doubling other methods), and 79.20% pass@1 on HumanEval. Performance advantages become increasingly pronounced with higher task difficulty, particularly on Level 5 MATH problems where improvements exceed 11 percentage points compared to state-of-the-art baselines. These results confirm that AgentGroupChat-V2 provides a comprehensive solution for building efficient, general-purpose LLM multi-agent systems with significant advantages in complex reasoning scenarios. Code is available at https://github.com/MikeGu721/AgentGroupChat-V2.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.13832v2">FrontendBench: A Benchmark for Evaluating LLMs on Front-End Development via Automatic Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made significant strides in front-end code generation. However, existing benchmarks exhibit several critical limitations: many tasks are overly simplistic, test cases often lack rigor, and end-to-end validation is absent. These issues hinder the accurate assessment of model performance. To address these challenges, we present FrontendBench, a benchmark co-developed by humans and LLMs. FrontendBench categorizes tasks based on code functionality and incorporates interactive test scenarios, enabling a more comprehensive and practical evaluation of front-end code generation capabilities. The benchmark comprises 148 meticulously crafted prompt-test case pairs spanning five levels of web components, from basic UI elements to complex interactive features. Each task reflects realistic front-end development challenges. Furthermore, we introduce an automatic evaluation framework that executes generated code within a sandbox environment and assesses outcomes using predefined test scripts. This framework achieves a 90.54% agreement rate with expert human evaluations, demonstrating high reliability. We benchmark several state-of-the-art LLMs on FrontendBench and observe substantial performance disparities in handling real-world front-end tasks. These results highlight FrontendBench as a reliable and scalable benchmark, supporting consistent multimodal evaluation and providing a robust foundation for future research in front-end code generation. Our data and code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01903v2">PsychBench: A comprehensive and professional benchmark for evaluating the performance of LLM-assisted psychiatric clinical practice</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) offers potential solutions to address problems such as shortage of medical resources and low diagnostic consistency in psychiatric clinical practice. Despite this potential, a robust and comprehensive benchmarking framework to assess the efficacy of LLMs in authentic psychiatric clinical environments is absent. This has impeded the advancement of specialized LLMs tailored to psychiatric applications. In response to this gap, by incorporating clinical demands in psychiatry and clinical data, we proposed a benchmarking system, PsychBench, to evaluate the practical performance of LLMs in psychiatric clinical settings. We conducted a comprehensive quantitative evaluation of 16 LLMs using PsychBench, and investigated the impact of prompt design, chain-of-thought reasoning, input text length, and domain-specific knowledge fine-tuning on model performance. Through detailed error analysis, we identified strengths and potential limitations of the existing models and suggested directions for improvement. Subsequently, a clinical reader study involving 60 psychiatrists of varying seniority was conducted to further explore the practical benefits of existing LLMs as supportive tools for psychiatrists of varying seniority. Through the quantitative and reader evaluation, we show that while existing models demonstrate significant potential, they are not yet adequate as decision-making tools in psychiatric clinical practice. The reader study further indicates that, as an auxiliary tool, LLM could provide particularly notable support for junior psychiatrists, effectively enhancing their work efficiency and overall clinical quality. To promote research in this area, we will make the dataset and evaluation framework publicly available, with the hope of advancing the application of LLMs in psychiatric clinical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06253v2">MAD-MAX: Modular And Diverse Malicious Attack MiXtures for Automated LLM Red Teaming</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 Data in Generative Models Workshop: The Bad, the Ugly, and the Greats (DIG-BUGS) at ICML 2025
    </div>
    <details class="paper-abstract">
      With LLM usage rapidly increasing, their vulnerability to jailbreaks that create harmful outputs are a major security risk. As new jailbreaking strategies emerge and models are changed by fine-tuning, continuous testing for security vulnerabilities is necessary. Existing Red Teaming methods fall short in cost efficiency, attack success rate, attack diversity, or extensibility as new attack types emerge. We address these challenges with Modular And Diverse Malicious Attack MiXtures (MAD-MAX) for Automated LLM Red Teaming. MAD-MAX uses automatic assignment of attack strategies into relevant attack clusters, chooses the most relevant clusters for a malicious goal, and then combines strategies from the selected clusters to achieve diverse novel attacks with high attack success rates. MAD-MAX further merges promising attacks together at each iteration of Red Teaming to boost performance and introduces a similarity filter to prune out similar attacks for increased cost efficiency. The MAD-MAX approach is designed to be easily extensible with newly discovered attack strategies and outperforms the prominent Red Teaming method Tree of Attacks with Pruning (TAP) significantly in terms of Attack Success Rate (ASR) and queries needed to achieve jailbreaks. MAD-MAX jailbreaks 97% of malicious goals in our benchmarks on GPT-4o and Gemini-Pro compared to TAP with 66%. MAD-MAX does so with only 10.9 average queries to the target LLM compared to TAP with 23.3. WARNING: This paper contains contents which are offensive in nature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15301v1">Cohort Discovery: A Survey on LLM-Assisted Clinical Trial Recruitment</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Recent advances in LLMs have greatly improved general-domain NLP tasks. Yet, their adoption in critical domains, such as clinical trial recruitment, remains limited. As trials are designed in natural language and patient data is represented as both structured and unstructured text, the task of matching trials and patients benefits from knowledge aggregation and reasoning abilities of LLMs. Classical approaches are trial-specific and LLMs with their ability to consolidate distributed knowledge hold the potential to build a more general solution. Yet recent applications of LLM-assisted methods rely on proprietary models and weak evaluation benchmarks. In this survey, we are the first to analyze the task of trial-patient matching and contextualize emerging LLM-based approaches in clinical trial recruitment. We critically examine existing benchmarks, approaches and evaluation frameworks, the challenges to adopting LLM technologies in clinical research and exciting future directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.10444v3">LLM-as-BT-Planner: Leveraging LLMs for Behavior Tree Generation in Robot Task Planning</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 7 pages. presented in ICRA 2025
    </div>
    <details class="paper-abstract">
      Robotic assembly tasks remain an open challenge due to their long horizon nature and complex part relations. Behavior trees (BTs) are increasingly used in robot task planning for their modularity and flexibility, but creating them manually can be effort-intensive. Large language models (LLMs) have recently been applied to robotic task planning for generating action sequences, yet their ability to generate BTs has not been fully investigated. To this end, we propose LLM-as-BT-Planner, a novel framework that leverages LLMs for BT generation in robotic assembly task planning. Four in-context learning methods are introduced to utilize the natural language processing and inference capabilities of LLMs for producing task plans in BT format, reducing manual effort while ensuring robustness and comprehensibility. Additionally, we evaluate the performance of fine-tuned smaller LLMs on the same tasks. Experiments in both simulated and real-world settings demonstrate that our framework enhances LLMs' ability to generate BTs, improving success rate through in-context learning and supervised fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04524v2">Trust Region Preference Approximation: A simple and stable reinforcement learning algorithm for LLM reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 10pages
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have rapidly evolved, approaching Artificial General Intelligence (AGI) while benefiting from large-scale reinforcement learning to enhance Human Alignment (HA) and Reasoning. Recent reward-based optimization algorithms, such as Proximal Policy Optimization (PPO) and Group Relative Policy Optimization (GRPO) have achieved significant performance on reasoning tasks, whereas preference-based optimization algorithms such as Direct Preference Optimization (DPO) significantly improve the performance of LLMs on human alignment. However, despite the strong performance of reward-based optimization methods in alignment tasks , they remain vulnerable to reward hacking. Furthermore, preference-based algorithms (such as Online DPO) haven't yet matched the performance of reward-based optimization algorithms (like PPO) on reasoning tasks, making their exploration in this specific area still a worthwhile pursuit. Motivated by these challenges, we propose the Trust Region Preference Approximation (TRPA) algorithm, which integrates rule-based optimization with preference-based optimization for reasoning tasks. As a preference-based algorithm, TRPA naturally eliminates the reward hacking issue. TRPA constructs preference levels using predefined rules, forms corresponding preference pairs, and leverages a novel optimization algorithm for RL training with a theoretical monotonic improvement guarantee. Experimental results demonstrate that TRPA not only achieves competitive performance on reasoning tasks but also exhibits robust stability. The code of this paper are released and updating on https://github.com/XueruiSu/Trust-Region-Preference-Approximation.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15253v1">RAS-Eval: A Comprehensive Benchmark for Security Evaluation of LLM Agents in Real-World Environments</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The rapid deployment of Large language model (LLM) agents in critical domains like healthcare and finance necessitates robust security frameworks. To address the absence of standardized evaluation benchmarks for these agents in dynamic environments, we introduce RAS-Eval, a comprehensive security benchmark supporting both simulated and real-world tool execution. RAS-Eval comprises 80 test cases and 3,802 attack tasks mapped to 11 Common Weakness Enumeration (CWE) categories, with tools implemented in JSON, LangGraph, and Model Context Protocol (MCP) formats. We evaluate 6 state-of-the-art LLMs across diverse scenarios, revealing significant vulnerabilities: attacks reduced agent task completion rates (TCR) by 36.78% on average and achieved an 85.65% success rate in academic settings. Notably, scaling laws held for security capabilities, with larger models outperforming smaller counterparts. Our findings expose critical risks in real-world agent deployments and provide a foundational framework for future security research. Code and data are available at https://github.com/lanzer-tree/RAS-Eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05716v3">Single-Agent vs. Multi-Agent LLM Strategies for Automated Student Reflection Assessment</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 Published in Proceedings of the 29th Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD 2025), see https://doi.org/10.1007/978-981-96-8186-0_24
    </div>
    <details class="paper-abstract">
      We explore the use of Large Language Models (LLMs) for automated assessment of open-text student reflections and prediction of academic performance. Traditional methods for evaluating reflections are time-consuming and may not scale effectively in educational settings. In this work, we employ LLMs to transform student reflections into quantitative scores using two assessment strategies (single-agent and multi-agent) and two prompting techniques (zero-shot and few-shot). Our experiments, conducted on a dataset of 5,278 reflections from 377 students over three academic terms, demonstrate that the single-agent with few-shot strategy achieves the highest match rate with human evaluations. Furthermore, models utilizing LLM-assessed reflection scores outperform baselines in both at-risk student identification and grade prediction tasks. These findings suggest that LLMs can effectively automate reflection assessment, reduce educators' workload, and enable timely support for students who may need additional assistance. Our work emphasizes the potential of integrating advanced generative AI technologies into educational practices to enhance student engagement and academic success.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13956v2">Improving LLM Video Understanding with 16 Frames Per Second</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Human vision is dynamic and continuous. However, in video understanding with multimodal large language models (LLMs), existing methods primarily rely on static features extracted from images sampled at a fixed low frame rate of frame-per-second (FPS) $\leqslant$2, leading to critical visual information loss. In this paper, we introduce F-16, the first multimodal LLM designed for high-frame-rate video understanding. By increasing the frame rate to 16 FPS and compressing visual tokens within each 1-second clip, F-16 efficiently captures dynamic visual features while preserving key semantic information. Experimental results demonstrate that higher frame rates considerably enhance video understanding across multiple benchmarks, providing a new approach to improving video LLMs beyond scaling model size or training data. F-16 achieves state-of-the-art performance among 7-billion-parameter video LLMs on both general and fine-grained video understanding benchmarks, such as Video-MME and TemporalBench. Furthermore, F-16 excels in complex spatiotemporal tasks, including high-speed sports analysis (\textit{e.g.}, basketball, football, gymnastics, and diving), outperforming SOTA proprietary visual models like GPT-4o and Gemini-1.5-pro. Additionally, we introduce a novel decoding method for F-16 that enables highly efficient low-frame-rate inference without requiring model retraining. We will release the source code, model checkpoints, and data at \href{https://github.com/bytedance/F-16}{https://github.com/bytedance/F-16}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15215v1">MinosEval: Distinguishing Factoid and Non-Factoid for Tailored Open-Ended QA Evaluation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Open-ended question answering (QA) is a key task for evaluating the capabilities of large language models (LLMs). Compared to closed-ended QA, it demands longer answer statements, more nuanced reasoning processes, and diverse expressions, making refined and interpretable automatic evaluation both crucial and challenging. Traditional metrics like ROUGE and BERTScore struggle to capture semantic similarities due to different patterns between model responses and reference answers. Current LLM-based evaluation approaches, such as pairwise or listwise comparisons of candidate answers, lack intuitive interpretability. While pointwise scoring of each response provides some descriptions, it fails to adapt across different question contents. Most notably, existing methods overlook the distinction between factoid and non-factoid questions. To address these challenges, we propose \textbf{MinosEval}, a novel evaluation method that first distinguishes open-ended questions and then ranks candidate answers using different evaluation strategies. For factoid questions, it applies an adaptive key-point scoring strategy, while for non-factoid questions, it uses an instance-aware listwise ranking strategy. Experiments on multiple open-ended QA datasets, including self-built ones with more candidate responses to complement community resources, show that MinosEval better aligns with human annotations and offers more interpretable results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15212v1">LLM vs. SAST: A Technical Analysis on Detecting Coding Bugs of GPT4-Advanced Data Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      With the rapid advancements in Natural Language Processing (NLP), large language models (LLMs) like GPT-4 have gained significant traction in diverse applications, including security vulnerability scanning. This paper investigates the efficacy of GPT-4 in identifying software vulnerabilities compared to traditional Static Application Security Testing (SAST) tools. Drawing from an array of security mistakes, our analysis underscores the potent capabilities of GPT-4 in LLM-enhanced vulnerability scanning. We unveiled that GPT-4 (Advanced Data Analysis) outperforms SAST by an accuracy of 94% in detecting 32 types of exploitable vulnerabilities. This study also addresses the potential security concerns surrounding LLMs, emphasising the imperative of security by design/default and other security best practices for AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15211v1">ProtoReasoning: Prototypes as the Foundation for Generalizable Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Recent advances in Large Reasoning Models (LRMs) trained with Long Chain-of-Thought (Long CoT) reasoning have demonstrated remarkable cross-domain generalization capabilities. However, the underlying mechanisms supporting such transfer remain poorly understood. We hypothesize that cross-domain generalization arises from shared abstract reasoning prototypes -- fundamental reasoning patterns that capture the essence of problems across domains. These prototypes minimize the nuances of the representation, revealing that seemingly diverse tasks are grounded in shared reasoning structures.Based on this hypothesis, we propose ProtoReasoning, a framework that enhances the reasoning ability of LLMs by leveraging scalable and verifiable prototypical representations (Prolog for logical reasoning, PDDL for planning).ProtoReasoning features: (1) an automated prototype construction pipeline that transforms problems into corresponding prototype representations; (2) a comprehensive verification system providing reliable feedback through Prolog/PDDL interpreters; (3) the scalability to synthesize problems arbitrarily within prototype space while ensuring correctness. Extensive experiments show that ProtoReasoning achieves 4.7% improvement over baseline models on logical reasoning (Enigmata-Eval), 6.3% improvement on planning tasks, 4.0% improvement on general reasoning (MMLU) and 1.0% on mathematics (AIME24). Significantly, our ablation studies confirm that learning in prototype space also demonstrates enhanced generalization to structurally similar problems compared to training solely on natural language representations, validating our hypothesis that reasoning prototypes serve as the foundation for generalizable reasoning in large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15196v1">HeurAgenix: Leveraging LLMs for Solving Complex Combinatorial Optimization Challenges</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 27 pages,9 figures
    </div>
    <details class="paper-abstract">
      Heuristic algorithms play a vital role in solving combinatorial optimization (CO) problems, yet traditional designs depend heavily on manual expertise and struggle to generalize across diverse instances. We introduce \textbf{HeurAgenix}, a two-stage hyper-heuristic framework powered by large language models (LLMs) that first evolves heuristics and then selects among them automatically. In the heuristic evolution phase, HeurAgenix leverages an LLM to compare seed heuristic solutions with higher-quality solutions and extract reusable evolution strategies. During problem solving, it dynamically picks the most promising heuristic for each problem state, guided by the LLM's perception ability. For flexibility, this selector can be either a state-of-the-art LLM or a fine-tuned lightweight model with lower inference cost. To mitigate the scarcity of reliable supervision caused by CO complexity, we fine-tune the lightweight heuristic selector with a dual-reward mechanism that jointly exploits singals from selection preferences and state perception, enabling robust selection under noisy annotations. Extensive experiments on canonical benchmarks show that HeurAgenix not only outperforms existing LLM-based hyper-heuristics but also matches or exceeds specialized solvers. Code is available at https://github.com/microsoft/HeurAgenix.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15170v1">From LLMs to MLLMs to Agents: A Survey of Emerging Paradigms in Jailbreak Attacks and Defenses within LLM Ecosystem</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly evolving from single-modal systems to multimodal LLMs and intelligent agents, significantly expanding their capabilities while introducing increasingly severe security risks. This paper presents a systematic survey of the growing complexity of jailbreak attacks and corresponding defense mechanisms within the expanding LLM ecosystem. We first trace the developmental trajectory from LLMs to MLLMs and Agents, highlighting the core security challenges emerging at each stage. Next, we categorize mainstream jailbreak techniques from both the attack impact and visibility perspectives, and provide a comprehensive analysis of representative attack methods, related datasets, and evaluation metrics. On the defense side, we organize existing strategies based on response timing and technical approach, offering a structured understanding of their applicability and implementation. Furthermore, we identify key limitations in existing surveys, such as insufficient attention to agent-specific security issues, the absence of a clear taxonomy for hybrid jailbreak methods, a lack of detailed analysis of experimental setups, and outdated coverage of recent advancements. To address these limitations, we provide an updated synthesis of recent work and outline future research directions in areas such as dataset construction, evaluation framework optimization, and strategy generalization. Our study seeks to enhance the understanding of jailbreak mechanisms and facilitate the advancement of more resilient and adaptive defense strategies in the context of ever more capable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10501v2">BugGen: A Self-Correcting Multi-Agent LLM Pipeline for Realistic RTL Bug Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Hardware complexity continues to strain verification resources, motivating the adoption of machine learning (ML) methods to improve debug efficiency. However, ML-assisted debugging critically depends on diverse and scalable bug datasets, which existing manual or automated bug insertion methods fail to reliably produce. We introduce BugGen, a first of its kind, fully autonomous, multi-agent pipeline leveraging Large Language Models (LLMs) to systematically generate, insert, and validate realistic functional bugs in RTL. BugGen partitions modules, selects mutation targets via a closed-loop agentic architecture, and employs iterative refinement and rollback mechanisms to ensure syntactic correctness and functional detectability. Evaluated across five OpenTitan IP blocks, BugGen produced 500 unique bugs with 94% functional accuracy and achieved a throughput of 17.7 validated bugs per hour-over five times faster than typical manual expert insertion. Additionally, BugGen identified 104 previously undetected bugs in OpenTitan regressions, highlighting its utility in exposing verification coverage gaps. Compared against Certitude, BugGen demonstrated over twice the syntactic accuracy, deeper exposure of testbench blind spots, and more functionally meaningful and complex bug scenarios. Furthermore, when these BugGen-generated datasets were employed to train ML-based failure triage models, we achieved high classification accuracy (88.1%-93.2%) across different IP blocks, confirming the practical utility and realism of generated bugs. BugGen thus provides a scalable solution for generating high-quality bug datasets, significantly enhancing verification efficiency and ML-assisted debugging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15167v1">LLM Agent for Hyper-Parameter Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 6 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Hyper-parameters are essential and critical for the performance of communication algorithms. However, current hyper-parameters tuning methods for warm-start particles swarm optimization with cross and mutation (WS-PSO-CM) algortihm for radio map-enabled unmanned aerial vehicle (UAV) trajectory and communication are primarily heuristic-based, exhibiting low levels of automation and unsatisfactory performance. In this paper, we design an large language model (LLM) agent for automatic hyper-parameters-tuning, where an iterative framework and model context protocol (MCP) are applied. In particular, the LLM agent is first setup via a profile, which specifies the mission, background, and output format. Then, the LLM agent is driven by the prompt requirement, and iteratively invokes WS-PSO-CM algorithm for exploration. Finally, the LLM agent autonomously terminates the loop and returns a set of hyper-parameters. Our experiment results show that the minimal sum-rate achieved by hyper-parameters generated via our LLM agent is significantly higher than those by both human heuristics and random generation methods. This indicates that an LLM agent with PSO knowledge and WS-PSO-CM algorithm background is useful in finding high-performance hyper-parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15155v1">eLLM: Elastic Memory Management Framework for Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Large Language Models are increasingly being deployed in datacenters. Serving these models requires careful memory management, as their memory usage includes static weights, dynamic activations, and key-value caches. While static weights are constant and predictable, dynamic components such as activations and KV caches change frequently during runtime, presenting significant challenges for efficient memory management. Modern LLM serving systems typically handle runtime memory and KV caches at distinct abstraction levels: runtime memory management relies on static tensor abstractions, whereas KV caches utilize a page table-based virtualization layer built on top of the tensor abstraction. This virtualization dynamically manages KV caches to mitigate memory fragmentation. However, this dual-level approach fundamentally isolates runtime memory and KV cache management, resulting in suboptimal memory utilization under dynamic workloads, which can lead to a nearly 20% drop in throughput. To address these limitations, we propose eLLM, an elastic memory management framework inspired by the classical memory ballooning mechanism in operating systems. The core components of eLLM include: (1) Virtual Tensor Abstraction, which decouples the virtual address space of tensors from the physical GPU memory, creating a unified and flexible memory pool; (2) an Elastic Memory Mechanism that dynamically adjusts memory allocation through runtime memory inflation and deflation, leveraging CPU memory as an extensible buffer; and (3) a Lightweight Scheduling Strategy employing SLO-aware policies to optimize memory utilization and effectively balance performance trade-offs under stringent SLO constraints. Comprehensive evaluations demonstrate that eLLM significantly outperforms state-of-the-art systems, 2.32x higher decoding throughput, and supporting 3x larger batch sizes for 128K-token inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03997v2">YOLO-MARL: You Only LLM Once for Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 accepted to International Conference on Intelligent Robots and Systems (IROS2025)
    </div>
    <details class="paper-abstract">
      Advancements in deep multi-agent reinforcement learning (MARL) have positioned it as a promising approach for decision-making in cooperative games. However, it still remains challenging for MARL agents to learn cooperative strategies for some game environments. Recently, large language models (LLMs) have demonstrated emergent reasoning capabilities, making them promising candidates for enhancing coordination among the agents. However, due to the model size of LLMs, it can be expensive to frequently infer LLMs for actions that agents can take. In this work, we propose You Only LLM Once for MARL (YOLO-MARL), a novel framework that leverages the high-level task planning capabilities of LLMs to improve the policy learning process of multi-agents in cooperative games. Notably, for each game environment, YOLO-MARL only requires one time interaction with LLMs in the proposed strategy generation, state interpretation and planning function generation modules, before the MARL policy training process. This avoids the ongoing costs and computational time associated with frequent LLMs API calls during training. Moreover, trained decentralized policies based on normal-sized neural networks operate independently of the LLM. We evaluate our method across two different environments and demonstrate that YOLO-MARL outperforms traditional MARL algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01203v2">HetGCoT: Heterogeneous Graph-Enhanced Chain-of-Thought LLM Reasoning for Academic Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Academic question answering (QA) in heterogeneous scholarly networks presents unique challenges requiring both structural understanding and interpretable reasoning. While graph neural networks (GNNs) capture structured graph information and large language models (LLMs) demonstrate strong capabilities in semantic comprehension, current approaches lack integration at the reasoning level. We propose HetGCoT, a framework enabling LLMs to effectively leverage and learn information from graphs to reason interpretable academic QA results. Our framework introduces three technical contributions: (1) a framework that transforms heterogeneous graph structural information into LLM-processable reasoning chains, (2) an adaptive metapath selection mechanism identifying relevant subgraphs for specific queries, and (3) a multi-step reasoning strategy systematically incorporating graph contexts into the reasoning process. Experiments on OpenAlex and DBLP datasets show our approach outperforms all sota baselines. The framework demonstrates adaptability across different LLM architectures and applicability to various scholarly question answering tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15131v1">Modeling the One-to-Many Property in Open-Domain Dialogue with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Open-domain Dialogue (OD) exhibits a one-to-many (o2m) property, whereby multiple appropriate responses exist for a single dialogue context. Despite prior research showing that modeling this property boosts response diversity, most modern LLM-based dialogue agents do not explicitly do so. In this work, we model the o2m property of OD in LLMs by decomposing OD generation into two key tasks: Multi-Response Generation (MRG) and Preference-based Selection (PS), which entail generating a set of n semantically and lexically diverse high-quality responses for a given dialogue context, followed by selecting a single response based on human preference, respectively. To facilitate MRG and PS, we introduce o2mDial, a dialogue corpus explicitly designed to capture the o2m property by featuring multiple plausible responses for each context. Leveraging o2mDial, we propose new in-context learning and instruction-tuning strategies, as well as novel evaluation metrics for MRG, alongside a model-based approach for PS. Empirical results demonstrate that applying the proposed two-stage framework to smaller LLMs for OD generation enhances overall response diversity while maintaining contextual coherence, improving response quality by up to 90%, bringing them closer to the performance of larger models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06809v3">Root Defence Strategies: Ensuring Safety of LLM at the Decoding Level</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 19 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated immense utility across various industries. However, as LLMs advance, the risk of harmful outputs increases due to incorrect or malicious instruction prompts. While current methods effectively address jailbreak risks, they share common limitations: 1) Judging harmful responses from the prefill-level lacks utilization of the model's decoding outputs, leading to relatively lower effectiveness and robustness. 2) Rejecting potentially harmful responses based on a single evaluation can significantly impair the model's helpfulness.This paper examines the LLMs' capability to recognize harmful outputs, revealing and quantifying their proficiency in assessing the danger of previous tokens. Motivated by pilot experiment results, we design a robust defense mechanism at the decoding level. Our novel decoder-oriented, step-by-step defense architecture corrects harmful queries directly rather than rejecting them outright. We introduce speculative decoding to enhance usability and facilitate deployment to boost secure decoding speed. Extensive experiments demonstrate that our approach improves model security without compromising reasoning speed. Notably, our method leverages the model's ability to discern hazardous information, maintaining its helpfulness compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15076v1">Learning-Time Encoding Shapes Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed in the real world, the ability to ``unlearn'', or remove specific pieces of knowledge post hoc, has become essential for a variety of reasons ranging from privacy regulations to correcting outdated or harmful content. Prior work has proposed unlearning benchmarks and algorithms, and has typically assumed that the training process and the target model are fixed. In this work, we empirically investigate how learning-time choices in knowledge encoding impact the effectiveness of unlearning factual knowledge. Our experiments reveal two key findings: (1) learning with paraphrased descriptions improves unlearning performance and (2) unlearning individual piece of knowledge from a chunk of text is challenging. Our results suggest that learning-time knowledge encoding may play a central role in enabling reliable post-hoc unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16205v6">LLMs can be Dangerous Reasoners: Analyzing-based Jailbreak Attack on Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      The rapid development of Large Language Models (LLMs) has brought impressive advancements across various tasks. However, despite these achievements, LLMs still pose inherent safety risks, especially in the context of jailbreak attacks. Most existing jailbreak methods follow an input-level manipulation paradigm to bypass safety mechanisms. Yet, as alignment techniques improve, such attacks are becoming increasingly detectable. In this work, we identify an underexplored threat vector: the model's internal reasoning process, which can be manipulated to elicit harmful outputs in a more stealthy way. To explore this overlooked attack surface, we propose a novel black-box jailbreak attack method, Analyzing-based Jailbreak (ABJ). ABJ comprises two independent attack paths: textual and visual reasoning attacks, which exploit the model's multimodal reasoning capabilities to bypass safety mechanisms, comprehensively exposing vulnerabilities in its reasoning chain. We conduct extensive experiments on ABJ across various open-source and closed-source LLMs, VLMs, and RLMs. In particular, ABJ achieves high attack success rate (ASR) (82.1% on GPT-4o-2024-11-20) with exceptional attack efficiency (AE) among all target models, showcasing its remarkable attack effectiveness, transferability, and efficiency. Our work reveals a new type of safety risk and highlights the urgent need to mitigate implicit vulnerabilities in the model's reasoning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15066v1">ChatModel: Automating Reference Model Design and Verification with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      As the complexity of integrated circuit designs continues to escalate, the functional verification becomes increasingly challenging. Reference models, critical for accelerating the verification process, are themselves becoming more intricate and time-consuming to develop. Despite the promise shown by large language models (LLMs) in code programming, effectively generating complex reference models remains a significant hurdle. To address these challenges, we introduce ChatModel, the first LLM-aided agile reference model generation and verification platform. ChatModel streamlines the transition from design specifications to fully functional reference models by integrating design standardization and hierarchical agile modeling. Employing a building-block generation strategy, it not only enhances the design capabilities of LLMs for reference models but also significantly boosts verification efficiency. We evaluated ChatModel on 300 designs of varying complexity, demonstrating substantial improvements in both efficiency and quality of reference model generation. ChatModel achieved a peak performance improvement of 55.02% compared to alternative methods, with notable enhancements in generation stability, and delivered a 9.18x increase in its capacity to produce reference model designs. Furthermore, it accelerated the iterative process of reference model design and validation by an average of 5.90x compared to traditional approaches. These results highlight the potential of ChatModel to significantly advance the automation of reference model generation and validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14397v2">Thunder-NUBench: A Benchmark for LLMs' Sentence-Level Negation Understanding</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Negation is a fundamental linguistic phenomenon that poses persistent challenges for Large Language Models (LLMs), particularly in tasks requiring deep semantic understanding. Existing benchmarks often treat negation as a side case within broader tasks like natural language inference, resulting in a lack of benchmarks that exclusively target negation understanding. In this work, we introduce Thunder-NUBench, a novel benchmark explicitly designed to assess sentence-level negation understanding in LLMs. Thunder-NUBench goes beyond surface-level cue detection by contrasting standard negation with structurally diverse alternatives such as local negation, contradiction, and paraphrase. The benchmark consists of manually curated sentence-negation pairs and a multiple-choice dataset that enables in-depth evaluation of models' negation understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15911v1">From RAG to Agentic: Validating Islamic-Medicine Responses with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 Under-review at the 4th Muslims in Machine Learning (MusIML) Workshop (ICML-25)
    </div>
    <details class="paper-abstract">
      Centuries-old Islamic medical texts like Avicenna's Canon of Medicine and the Prophetic Tibb-e-Nabawi encode a wealth of preventive care, nutrition, and holistic therapies, yet remain inaccessible to many and underutilized in modern AI systems. Existing language-model benchmarks focus narrowly on factual recall or user preference, leaving a gap in validating culturally grounded medical guidance at scale. We propose a unified evaluation pipeline, Tibbe-AG, that aligns 30 carefully curated Prophetic-medicine questions with human-verified remedies and compares three LLMs (LLaMA-3, Mistral-7B, Qwen2-7B) under three configurations: direct generation, retrieval-augmented generation, and a scientific self-critique filter. Each answer is then assessed by a secondary LLM serving as an agentic judge, yielding a single 3C3H quality score. Retrieval improves factual accuracy by 13%, while the agentic prompt adds another 10% improvement through deeper mechanistic insight and safety considerations. Our results demonstrate that blending classical Islamic texts with retrieval and self-evaluation enables reliable, culturally sensitive medical question-answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17264v4">Medha: Efficiently Serving Multi-Million Context Length LLM Inference Requests Without Approximations</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) handle increasingly longer contexts, serving long inference requests of millions of tokens presents unique challenges. We show that existing work for long context inference is largely based on techniques from long context training, and does not handle the high variability in input lengths during inference. This leads to inefficient resource utilization, server fragmentation, and head-of-line (HOL) blocking. We present Medha, an end-to-end system for efficient long-context LLM inference that addresses these challenges through fine-grained time sharing. Medha introduces three key innovations: (1) the mechanism of adaptive prefill chunking to help mitigate HOL blocking with preemption; (2) two new parallelism strategies: Sequence Pipeline Parallelism (SPP) to reduce time-to-first-token by pipelining prefill chunks, and KV-Cache Parallelism (KVP) to lower time-peroutput-token by distributing decoding across servers; and (3) a novel input-length aware least remaining slack scheduling to meet Service Level Objectives (SLOs). Medha enables exact inference scaling beyond 10 million tokens, maintaining high throughput and low latency across mixed-length workloads. Compared to state-of-the-art systems, Medha reduces server fragmentation, cuts median latency by up to 30x, and improves throughput by over 5x, delivering production-scale long-context inference without compromising performance on shorter requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22998v2">LLM Agents for Bargaining with Utility-based Feedback</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Bargaining, a critical aspect of real-world interactions, presents challenges for large language models (LLMs) due to limitations in strategic depth and adaptation to complex human factors. Existing benchmarks often fail to capture this real-world complexity. To address this and enhance LLM capabilities in realistic bargaining, we introduce a comprehensive framework centered on utility-based feedback. Our contributions are threefold: (1) BargainArena, a novel benchmark dataset with six intricate scenarios (e.g., deceptive practices, monopolies) to facilitate diverse strategy modeling; (2) human-aligned, economically-grounded evaluation metrics inspired by utility theory, incorporating agent utility and negotiation power, which implicitly reflect and promote opponent-aware reasoning (OAR); and (3) a structured feedback mechanism enabling LLMs to iteratively refine their bargaining strategies. This mechanism can positively collaborate with in-context learning (ICL) prompts, including those explicitly designed to foster OAR. Experimental results show that LLMs often exhibit negotiation strategies misaligned with human preferences, and that our structured feedback mechanism significantly improves their performance, yielding deeper strategic and opponent-aware reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15866v1">Understanding Online Polarization Through Human-Agent Interaction in a Synthetic LLM-Based Social Network</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 Accepted for publication in the Proceedings of the Nineteenth International AAAI Conference on Web and Social Media (ICWSM 2025). This is the authors' version of the work, with corrections to table cross-references. The definitive Version of Record is available at https://doi.org/10.1609/icwsm.v19i1.35826. arXiv admin note: substantial text overlap with arXiv:2502.01340
    </div>
    <details class="paper-abstract">
      The rise of social media has fundamentally transformed how people engage in public discourse and form opinions. While these platforms offer unprecedented opportunities for democratic engagement, they have been implicated in increasing social polarization and the formation of ideological echo chambers. Previous research has primarily relied on observational studies of social media data or theoretical modeling approaches, leaving a significant gap in our understanding of how individuals respond to and are influenced by polarized online environments. Here we present a novel experimental framework for investigating polarization dynamics that allows human users to interact with LLM-based artificial agents in a controlled social network simulation. Through a user study with 122 participants, we demonstrate that this approach can successfully reproduce key characteristics of polarized online discourse while enabling precise manipulation of environmental factors. Our results provide empirical validation of theoretical predictions about online polarization, showing that polarized environments significantly increase perceived emotionality and group identity salience while reducing expressed uncertainty. These findings extend previous observational and theoretical work by providing causal evidence for how specific features of online environments influence user perceptions and behaviors. More broadly, this research introduces a powerful new methodology for studying social media dynamics, offering researchers unprecedented control over experimental conditions while maintaining ecological validity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15833v1">Architecture is All You Need: Improving LLM Recommenders by Dropping the Text</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 7 pages, 1 figure
    </div>
    <details class="paper-abstract">
      In recent years, there has been an explosion of interest in the applications of large pre-trained language models (PLMs) to recommender systems, with many studies showing strong performance of PLMs on common benchmark datasets. PLM-based recommender models benefit from flexible and customizable prompting, an unlimited vocabulary of recommendable items, and general ``world knowledge'' acquired through pre-training on massive text corpora. While PLM-based recommenders show promise in settings where data is limited, they are hard to implement in practice due to their large size and computational cost. Additionally, fine-tuning PLMs to improve performance on collaborative signals may degrade the model's capacity for world knowledge and generalizability. We propose a recommender model that uses the architecture of large language models (LLMs) while reducing layer count and dimensions and replacing the text-based subword tokenization of a typical LLM with discrete tokens that uniquely represent individual content items. We find that this simplified approach substantially outperforms both traditional sequential recommender models and PLM-based recommender models at a tiny fraction of the size and computational complexity of PLM-based models. Our results suggest that the principal benefit of LLMs in recommender systems is their architecture, rather than the world knowledge acquired during extensive pre-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15830v1">Rethinking LLM Training through Information Geometry and Quantum Metrics</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 9 pages, 1 figure(s)
    </div>
    <details class="paper-abstract">
      Optimization in large language models (LLMs) unfolds over high-dimensional parameter spaces with non-Euclidean structure. Information geometry frames this landscape using the Fisher information metric, enabling more principled learning via natural gradient descent. Though often impractical, this geometric lens clarifies phenomena such as sharp minima, generalization, and observed scaling laws. We argue that curvature-aware approaches deepen our understanding of LLM training. Finally, we speculate on quantum analogies based on the Fubini-Study metric and Quantum Fisher Information, hinting at efficient optimization in quantum-enhanced systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15828v1">Context Matters! Relaxing Goals with LLMs for Feasible 3D Scene Planning</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Classical planning in AI and Robotics addresses complex tasks by shifting from imperative to declarative approaches (e.g., PDDL). However, these methods often fail in real scenarios due to limited robot perception and the need to ground perceptions to planning predicates. This often results in heavily hard-coded behaviors that struggle to adapt, even with scenarios where goals can be achieved through relaxed planning. Meanwhile, Large Language Models (LLMs) lead to planning systems that leverage commonsense reasoning but often at the cost of generating unfeasible and/or unsafe plans. To address these limitations, we present an approach integrating classical planning with LLMs, leveraging their ability to extract commonsense knowledge and ground actions. We propose a hierarchical formulation that enables robots to make unfeasible tasks tractable by defining functionally equivalent goals through gradual relaxation. This mechanism supports partial achievement of the intended objective, suited to the agent's specific context. Our method demonstrates its ability to adapt and execute tasks effectively within environments modeled using 3D Scene Graphs through comprehensive qualitative and quantitative evaluations. We also show how this method succeeds in complex scenarios where other benchmark methods are more likely to fail. Code, dataset, and additional material are released to the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15790v1">ETrace:Event-Driven Vulnerability Detection in Smart Contracts via LLM-Based Trace Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 4 pages, 1 figure. Submitted to the 16th Asia-Pacific Symposium on Internetware (Internetware 2025)
    </div>
    <details class="paper-abstract">
      With the advance application of blockchain technology in various fields, ensuring the security and stability of smart contracts has emerged as a critical challenge. Current security analysis methodologies in vulnerability detection can be categorized into static analysis and dynamic analysis methods.However, these existing traditional vulnerability detection methods predominantly rely on analyzing original contract code, not all smart contracts provide accessible code.We present ETrace, a novel event-driven vulnerability detection framework for smart contracts, which uniquely identifies potential vulnerabilities through LLM-powered trace analysis without requiring source code access. By extracting fine-grained event sequences from transaction logs, the framework leverages Large Language Models (LLMs) as adaptive semantic interpreters to reconstruct event analysis through chain-of-thought reasoning. ETrace implements pattern-matching to establish causal links between transaction behavior patterns and known attack behaviors. Furthermore, we validate the effectiveness of ETrace through preliminary experimental results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17323v1">I Know Which LLM Wrote Your Code Last Summer: LLM generated Code Stylometry for Authorship Attribution</a></div>
    <div class="paper-meta">
      📅 2025-06-18
    </div>
    <details class="paper-abstract">
      Detecting AI-generated code, deepfakes, and other synthetic content is an emerging research challenge. As code generated by Large Language Models (LLMs) becomes more common, identifying the specific model behind each sample is increasingly important. This paper presents the first systematic study of LLM authorship attribution for C programs. We released CodeT5-Authorship, a novel model that uses only the encoder layers from the original CodeT5 encoder-decoder architecture, discarding the decoder to focus on classification. Our model's encoder output (first token) is passed through a two-layer classification head with GELU activation and dropout, producing a probability distribution over possible authors. To evaluate our approach, we introduce LLM-AuthorBench, a benchmark of 32,000 compilable C programs generated by eight state-of-the-art LLMs across diverse tasks. We compare our model to seven traditional ML classifiers and eight fine-tuned transformer models, including BERT, RoBERTa, CodeBERT, ModernBERT, DistilBERT, DeBERTa-V3, Longformer, and LoRA-fine-tuned Qwen2-1.5B. In binary classification, our model achieves 97.56% accuracy in distinguishing C programs generated by closely related models such as GPT-4.1 and GPT-4o, and 95.40% accuracy for multi-class attribution among five leading LLMs (Gemini 2.5 Flash, Claude 3.5 Haiku, GPT-4.1, Llama 3.3, and DeepSeek-V3). To support open science, we release the CodeT5-Authorship architecture, the LLM-AuthorBench benchmark, and all relevant Google Colab scripts on GitHub: https://github.com/LLMauthorbench/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17314v1">PRAISE: Enhancing Product Descriptions with LLM-Driven Structured Insights</a></div>
    <div class="paper-meta">
      📅 2025-06-18
      | 💬 9 Pages, 9 Figures. Accepted at ACL 2025 System Demonstration Track
    </div>
    <details class="paper-abstract">
      Accurate and complete product descriptions are crucial for e-commerce, yet seller-provided information often falls short. Customer reviews offer valuable details but are laborious to sift through manually. We present PRAISE: Product Review Attribute Insight Structuring Engine, a novel system that uses Large Language Models (LLMs) to automatically extract, compare, and structure insights from customer reviews and seller descriptions. PRAISE provides users with an intuitive interface to identify missing, contradictory, or partially matching details between these two sources, presenting the discrepancies in a clear, structured format alongside supporting evidence from reviews. This allows sellers to easily enhance their product listings for clarity and persuasiveness, and buyers to better assess product reliability. Our demonstration showcases PRAISE's workflow, its effectiveness in generating actionable structured insights from unstructured reviews, and its potential to significantly improve the quality and trustworthiness of e-commerce product catalogs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15025v1">Optimal Embedding Learning Rate in LLMs: The Effect of Vocabulary Size</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 TD,LR: How to set the learning rate for emebdding layer in LLMs?
    </div>
    <details class="paper-abstract">
      Pretraining large language models is a costly process. To make this process more efficient, several methods have been proposed to optimize model architecture/parametrization and hardware use. On the parametrization side, $\mu P$ (Maximal Update Parametrization) parametrizes model weights and learning rate (LR) in a way that makes hyperparameters (HPs) transferable with width (embedding dimension): HPs can be tuned for a small model and used for larger models without additional tuning. While $\mu$P showed impressive results in practice, recent empirical studies have reported conflicting observations when applied to LLMs. One limitation of the theory behind $\mu$P is the fact that input dimension (vocabulary size in LLMs) is considered fixed when taking the width to infinity. This is unrealistic since vocabulary size is generally much larger than width in practice. In this work, we provide a theoretical analysis of the effect of vocabulary size on training dynamics, and subsequently show that as vocabulary size increases, the training dynamics \emph{interpolate between the $\mu$P regime and another regime that we call Large Vocab (LV) Regime}, where optimal scaling rules are different from those predicted by $\mu$P. Our analysis reveals that in the LV regime, the optimal embedding LR to hidden LR ratio should roughly scale as $\Theta(\sqrt{width})$, surprisingly close to the empirical findings previously reported in the literature, and different from the $\Theta(width)$ ratio predicted by $\mu$P. We conduct several experiments to validate our theory, and pretrain a 1B model from scratch to show the benefit of our suggested scaling rule for the embedding LR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14997v1">Hypothesis Testing for Quantifying LLM-Human Misalignment in Multiple Choice Settings</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly appear in social science research (e.g., economics and marketing), it becomes crucial to assess how well these models replicate human behavior. In this work, using hypothesis testing, we present a quantitative framework to assess the misalignment between LLM-simulated and actual human behaviors in multiple-choice survey settings. This framework allows us to determine in a principled way whether a specific language model can effectively simulate human opinions, decision-making, and general behaviors represented through multiple-choice options. We applied this framework to a popular language model for simulating people's opinions in various public surveys and found that this model is ill-suited for simulating the tested sub-populations (e.g., across different races, ages, and incomes) for contentious questions. This raises questions about the alignment of this language model with the tested populations, highlighting the need for new practices in using LLMs for social science studies beyond naive simulations of human subjects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13116v4">VeriLeaky: Navigating IP Protection vs Utility in Fine-Tuning for LLM-Driven Verilog Coding</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer significant potential for coding, yet fine-tuning (FT) with curated data is essential for niche languages like Verilog. Using proprietary intellectual property (IP) for FT presents a serious risk, as FT data can be leaked through LLM inference. This leads to a critical dilemma for design houses: seeking to build externally accessible LLMs offering competitive Verilog coding, how can they leverage in-house IP to enhance FT utility while ensuring IP protection? For the first time in the literature, we study this dilemma. Using LLaMA 3.1-8B, we conduct in-house FT on a baseline Verilog dataset (RTLCoder) supplemented with our own in-house IP, which is validated through multiple tape-outs. To rigorously assess IP leakage, we quantify structural similarity (AST/Dolos) and functional equivalence (Synopsys Formality) between generated codes and our in-house IP. We show that our IP can indeed be leaked, confirming the threat. As defense, we evaluate logic locking of Verilog codes (ASSURE). This offers some level of protection, yet reduces the IP's utility for FT and degrades the LLM's performance. Our study shows the need for novel strategies that are both effective and minimally disruptive to FT, an essential effort for enabling design houses to fully utilize their proprietary IP toward LLM-driven Verilog coding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14965v1">Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 38 pages, 9 figures. Under review
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as a promising approach to improve large language model (LLM) reasoning, yet most open efforts focus narrowly on math and code, limiting our understanding of its broader applicability to general reasoning. A key challenge lies in the lack of reliable, scalable RL reward signals across diverse reasoning domains. We introduce Guru, a curated RL reasoning corpus of 92K verifiable examples spanning six reasoning domains--Math, Code, Science, Logic, Simulation, and Tabular--each built through domain-specific reward design, deduplication, and filtering to ensure reliability and effectiveness for RL training. Based on Guru, we systematically revisit established findings in RL for LLM reasoning and observe significant variation across domains. For example, while prior work suggests that RL primarily elicits existing knowledge from pretrained models, our results reveal a more nuanced pattern: domains frequently seen during pretraining (Math, Code, Science) easily benefit from cross-domain RL training, while domains with limited pretraining exposure (Logic, Simulation, and Tabular) require in-domain training to achieve meaningful performance gains, suggesting that RL is likely to facilitate genuine skill acquisition. Finally, we present Guru-7B and Guru-32B, two models that achieve state-of-the-art performance among open models RL-trained with publicly available data, outperforming best baselines by 7.9% and 6.7% on our 17-task evaluation suite across six reasoning domains. We also show that our models effectively improve the Pass@k performance of their base models, particularly on complex tasks less likely to appear in pretraining data. We release data, models, training and evaluation code to facilitate general-purpose reasoning at: https://github.com/LLM360/Reasoning360
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.03491v2">Can LLMs Ask Good Questions?</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      We evaluate questions generated by large language models (LLMs) from context, comparing them to human-authored questions across six dimensions: question type, question length, context coverage, answerability, uncommonness, and required answer length. Our study spans two open-source and two proprietary state-of-the-art models. Results reveal that LLM-generated questions tend to demand longer descriptive answers and exhibit more evenly distributed context focus, in contrast to the positional bias often seen in QA tasks. These findings provide insights into the distinctive characteristics of LLM-generated questions and inform future work on question quality and downstream applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14912v1">CrEst: Credibility Estimation for Contexts in LLMs via Weak Supervision</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      The integration of contextual information has significantly enhanced the performance of large language models (LLMs) on knowledge-intensive tasks. However, existing methods often overlook a critical challenge: the credibility of context documents can vary widely, potentially leading to the propagation of unreliable information. In this paper, we introduce CrEst, a novel weakly supervised framework for assessing the credibility of context documents during LLM inference--without requiring manual annotations. Our approach is grounded in the insight that credible documents tend to exhibit higher semantic coherence with other credible documents, enabling automated credibility estimation through inter-document agreement. To incorporate credibility into LLM inference, we propose two integration strategies: a black-box approach for models without access to internal weights or activations, and a white-box method that directly modifies attention mechanisms. Extensive experiments across three model architectures and five datasets demonstrate that CrEst consistently outperforms strong baselines, achieving up to a 26.86% improvement in accuracy and a 3.49% increase in F1 score. Further analysis shows that CrEst maintains robust performance even under high-noise conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.13198v3">LBAP: Improved Uncertainty Alignment of LLM Planners using Bayesian Inference</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) showcase many desirable traits for intelligent and helpful robots. However, they are also known to hallucinate predictions. This issue is exacerbated in robotics where LLM hallucinations may result in robots confidently executing plans that are contrary to user goals or relying more frequently on human assistance. In this work, we present LBAP, a novel approach for utilizing off-the-shelf LLMs, alongside Bayesian inference for uncertainty Alignment in robotic Planners that minimizes hallucinations and human intervention. Our key finding is that we can use Bayesian inference to more accurately calibrate a robots confidence measure through accounting for both scene grounding and world knowledge. This process allows us to mitigate hallucinations and better align the LLM's confidence measure with the probability of success. Through experiments in both simulation and the real world on tasks with a variety of ambiguities, we show that LBAP significantly increases success rate and decreases the amount of human intervention required relative to prior art. For example, in our real-world testing paradigm, LBAP decreases the human help rate of previous methods by over 33% at a success rate of 70%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.10735v4">Assessing the Reasoning Capabilities of LLMs in the context of Evidence-based Claim Verification</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 First two authors contributed equally to this work. 25 pages, 3 figure
    </div>
    <details class="paper-abstract">
      Although LLMs have shown great performance on Mathematics and Coding related reasoning tasks, the reasoning capabilities of LLMs regarding other forms of reasoning are still an open problem. Here, we examine the issue of reasoning from the perspective of claim verification. We propose a framework designed to break down any claim paired with evidence into atomic reasoning types that are necessary for verification. We use this framework to create RECV, the first claim verification benchmark, incorporating real-world claims, to assess the deductive and abductive reasoning capabilities of LLMs. The benchmark comprises of three datasets, covering reasoning problems of increasing complexity. We evaluate three state-of-the-art proprietary LLMs under multiple prompt settings. Our results show that while LLMs can address deductive reasoning problems, they consistently fail in cases of abductive reasoning. Moreover, we observe that enhancing LLMs with rationale generation is not always beneficial. Nonetheless, we find that generated rationales are semantically similar to those provided by humans, especially in deductive reasoning cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19165v3">OrgAccess: A Benchmark for Role Based Access Control in Organization Scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 56 Pages
    </div>
    <details class="paper-abstract">
      Role-based access control (RBAC) and hierarchical structures are foundational to how information flows and decisions are made within virtually all organizations. As the potential of Large Language Models (LLMs) to serve as unified knowledge repositories and intelligent assistants in enterprise settings becomes increasingly apparent, a critical, yet under explored, challenge emerges: \textit{can these models reliably understand and operate within the complex, often nuanced, constraints imposed by organizational hierarchies and associated permissions?} Evaluating this crucial capability is inherently difficult due to the proprietary and sensitive nature of real-world corporate data and access control policies. We introduce a synthetic yet representative \textbf{OrgAccess} benchmark consisting of 40 distinct types of permissions commonly relevant across different organizational roles and levels. We further create three types of permissions: 40,000 easy (1 permission), 10,000 medium (3-permissions tuple), and 20,000 hard (5-permissions tuple) to test LLMs' ability to accurately assess these permissions and generate responses that strictly adhere to the specified hierarchical rules, particularly in scenarios involving users with overlapping or conflicting permissions. Our findings reveal that even state-of-the-art LLMs struggle significantly to maintain compliance with role-based structures, even with explicit instructions, with their performance degrades further when navigating interactions involving two or more conflicting permissions. Specifically, even \textbf{GPT-4.1 only achieves an F1-Score of 0.27 on our hardest benchmark}. This demonstrates a critical limitation in LLMs' complex rule following and compositional reasoning capabilities beyond standard factual or STEM-based benchmarks, opening up a new paradigm for evaluating their fitness for practical, structured environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08001v3">Reparameterized LLM Training via Orthogonal Equivalence Transformation</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 Technical report v3 (38 pages, 26 figures, project page: https://spherelab.ai/poet/, v3: added singular spectrum and energy analyses in Section 4)
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are driving the rapid advancement of artificial intelligence, effectively and reliably training these large models remains one of the field's most significant challenges. To address this challenge, we propose POET, a novel reParameterized training algorithm that uses Orthogonal Equivalence Transformation to optimize neurons. Specifically, POET reparameterizes each neuron with two learnable orthogonal matrices and a fixed random weight matrix. Because of its provable preservation of spectral properties of weight matrices, POET can stably optimize the objective function with improved generalization. We further develop efficient approximations that make POET flexible and scalable for training large-scale neural networks. Extensive experiments validate the effectiveness and scalability of POET in training LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.10970v3">The Alternative Annotator Test for LLM-as-a-Judge: How to Statistically Justify Replacing Human Annotators with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      The "LLM-as-an-annotator" and "LLM-as-a-judge" paradigms employ Large Language Models (LLMs) as annotators, judges, and evaluators in tasks traditionally performed by humans. LLM annotations are widely used, not only in NLP research but also in fields like medicine, psychology, and social science. Despite their role in shaping study results and insights, there is no standard or rigorous procedure to determine whether LLMs can replace human annotators. In this paper, we propose a novel statistical procedure, the Alternative Annotator Test (alt-test), that requires only a modest subset of annotated examples to justify using LLM annotations. Additionally, we introduce a versatile and interpretable measure for comparing LLM annotators and judges. To demonstrate our procedure, we curated a diverse collection of ten datasets, consisting of language and vision-language tasks, and conducted experiments with six LLMs and four prompting techniques. Our results show that LLMs can sometimes replace humans with closed-source LLMs (such as GPT-4o), outperforming the open-source LLMs we examine, and that prompting techniques yield judges of varying quality. We hope this study encourages more rigorous and reliable practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04227v2">Agent Laboratory: Using LLM Agents as Research Assistants</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Historically, scientific discovery has been a lengthy and costly process, demanding substantial time and resources from initial conception to final results. To accelerate scientific discovery, reduce research costs, and improve research quality, we introduce Agent Laboratory, an autonomous LLM-based framework capable of completing the entire research process. This framework accepts a human-provided research idea and progresses through three stages--literature review, experimentation, and report writing to produce comprehensive research outputs, including a code repository and a research report, while enabling users to provide feedback and guidance at each stage. We deploy Agent Laboratory with various state-of-the-art LLMs and invite multiple researchers to assess its quality by participating in a survey, providing human feedback to guide the research process, and then evaluate the final paper. We found that: (1) Agent Laboratory driven by o1-preview generates the best research outcomes; (2) The generated machine learning code is able to achieve state-of-the-art performance compared to existing methods; (3) Human involvement, providing feedback at each stage, significantly improves the overall quality of research; (4) Agent Laboratory significantly reduces research expenses, achieving an 84% decrease compared to previous autonomous research methods. We hope Agent Laboratory enables researchers to allocate more effort toward creative ideation rather than low-level coding and writing, ultimately accelerating scientific discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14681v1">Massive Supervised Fine-tuning Experiments Reveal How Data, Layer, and Training Factors Shape LLM Alignment Quality</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is a critical step in aligning large language models (LLMs) with human instructions and values, yet many aspects of SFT remain poorly understood. We trained a wide range of base models on a variety of datasets including code generation, mathematical reasoning, and general-domain tasks, resulting in 1,000+ SFT models under controlled conditions. We then identified the dataset properties that matter most and examined the layer-wise modifications introduced by SFT. Our findings reveal that some training-task synergies persist across all models while others vary substantially, emphasizing the importance of model-specific strategies. Moreover, we demonstrate that perplexity consistently predicts SFT effectiveness--often surpassing superficial similarity between trained data and benchmark--and that mid-layer weight changes correlate most strongly with performance gains. We will release these 1,000+ SFT models and benchmark results to accelerate further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14645v1">Passing the Turing Test in Political Discourse: Fine-Tuning LLMs to Mimic Polarized Social Media Comments</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      The increasing sophistication of large language models (LLMs) has sparked growing concerns regarding their potential role in exacerbating ideological polarization through the automated generation of persuasive and biased content. This study explores the extent to which fine-tuned LLMs can replicate and amplify polarizing discourse within online environments. Using a curated dataset of politically charged discussions extracted from Reddit, we fine-tune an open-source LLM to produce context-aware and ideologically aligned responses. The model's outputs are evaluated through linguistic analysis, sentiment scoring, and human annotation, with particular attention to credibility and rhetorical alignment with the original discourse. The results indicate that, when trained on partisan data, LLMs are capable of producing highly plausible and provocative comments, often indistinguishable from those written by humans. These findings raise significant ethical questions about the use of AI in political discourse, disinformation, and manipulation campaigns. The paper concludes with a discussion of the broader implications for AI governance, platform regulation, and the development of detection tools to mitigate adversarial fine-tuning risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12442v3">IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04619v2">SynGraph: A Dynamic Graph-LLM Synthesis Framework for Sparse Streaming User Sentiment Modeling</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 Accepted at ACL 2025
    </div>
    <details class="paper-abstract">
      User reviews on e-commerce platforms exhibit dynamic sentiment patterns driven by temporal and contextual factors. Traditional sentiment analysis methods focus on static reviews, failing to capture the evolving temporal relationship between user sentiment rating and textual content. Sentiment analysis on streaming reviews addresses this limitation by modeling and predicting the temporal evolution of user sentiments. However, it suffers from data sparsity, manifesting in temporal, spatial, and combined forms. In this paper, we introduce SynGraph, a novel framework designed to address data sparsity in sentiment analysis on streaming reviews. SynGraph alleviates data sparsity by categorizing users into mid-tail, long-tail, and extreme scenarios and incorporating LLM-augmented enhancements within a dynamic graph-based structure. Experiments on real-world datasets demonstrate its effectiveness in addressing sparsity and improving sentiment modeling in streaming reviews.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.11147v3">Vul-RAG: Enhancing LLM-based Vulnerability Detection via Knowledge-level RAG</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Although LLMs have shown promising potential in vulnerability detection, this study reveals their limitations in distinguishing between vulnerable and similar-but-benign patched code (only 0.06 - 0.14 accuracy). It shows that LLMs struggle to capture the root causes of vulnerabilities during vulnerability detection. To address this challenge, we propose enhancing LLMs with multi-dimensional vulnerability knowledge distilled from historical vulnerabilities and fixes. We design a novel knowledge-level Retrieval-Augmented Generation framework Vul-RAG, which improves LLMs with an accuracy increase of 16% - 24% in identifying vulnerable and patched code. Additionally, vulnerability knowledge generated by Vul-RAG can further (1) serve as high-quality explanations to improve manual detection accuracy (from 60% to 77%), and (2) detect 10 previously-unknown bugs in the recent Linux kernel release with 6 assigned CVEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14445v2">PredictaBoard: Benchmarking LLM Score Predictability</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 Accepted at ACL Findings 2025
    </div>
    <details class="paper-abstract">
      Despite possessing impressive skills, Large Language Models (LLMs) often fail unpredictably, demonstrating inconsistent success in even basic common sense reasoning tasks. This unpredictability poses a significant challenge to ensuring their safe deployment, as identifying and operating within a reliable "safe zone" is essential for mitigating risks. To address this, we present PredictaBoard, a novel collaborative benchmarking framework designed to evaluate the ability of score predictors (referred to as assessors) to anticipate LLM errors on specific task instances (i.e., prompts) from existing datasets. PredictaBoard evaluates pairs of LLMs and assessors by considering the rejection rate at different tolerance errors. As such, PredictaBoard stimulates research into developing better assessors and making LLMs more predictable, not only with a higher average performance. We conduct illustrative experiments using baseline assessors and state-of-the-art LLMs. PredictaBoard highlights the critical need to evaluate predictability alongside performance, paving the way for safer AI systems where errors are not only minimised but also anticipated and effectively mitigated. Code for our benchmark can be found at https://github.com/Kinds-of-Intelligence-CFI/PredictaBoard
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14562v1">AlphaDecay:Module-wise Weight Decay for Heavy-Tailed Balancing in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Weight decay is a standard regularization technique for training large language models (LLMs). While it is common to assign a uniform decay rate to every layer, this approach overlooks the structural diversity of LLMs and the varying spectral properties across modules. In this paper, we introduce AlphaDecay, a simple yet effective method that adaptively assigns different weight decay strengths to each module of an LLM. Our approach is guided by Heavy-Tailed Self-Regularization (HT-SR) theory, which analyzes the empirical spectral density (ESD) of weight correlation matrices to quantify "heavy-tailedness." Modules exhibiting more pronounced heavy-tailed ESDs, reflecting stronger feature learning, are assigned weaker decay, while modules with lighter-tailed spectra receive stronger decay. Our method leverages tailored weight decay assignments to balance the module-wise differences in spectral properties, leading to improved performance. Extensive pre-training tasks with various model sizes from 60M to 1B demonstrate that AlphaDecay achieves better perplexity and generalization than conventional uniform decay and other adaptive decay baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17655v2">AssistantX: An LLM-Powered Proactive Assistant in Collaborative Human-Populated Environment</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 8 pages, 10 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Current service robots suffer from limited natural language communication abilities, heavy reliance on predefined commands, ongoing human intervention, and, most notably, a lack of proactive collaboration awareness in human-populated environments. This results in narrow applicability and low utility. In this paper, we introduce AssistantX, an LLM-powered proactive assistant designed for autonomous operation in realworld scenarios with high accuracy. AssistantX employs a multi-agent framework consisting of 4 specialized LLM agents, each dedicated to perception, planning, decision-making, and reflective review, facilitating advanced inference capabilities and comprehensive collaboration awareness, much like a human assistant by your side. We built a dataset of 210 real-world tasks to validate AssistantX, which includes instruction content and status information on whether relevant personnel are available. Extensive experiments were conducted in both text-based simulations and a real office environment over the course of a month and a half. Our experiments demonstrate the effectiveness of the proposed framework, showing that AssistantX can reactively respond to user instructions, actively adjust strategies to adapt to contingencies, and proactively seek assistance from humans to ensure successful task completion. More details and videos can be found at https://assistantx-agent. github.io/AssistantX/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03255v2">Inherent and emergent liability issues in LLM-based agentic systems: a principal-agent perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 22 pages (incl. appendix), accepted at REALM workshop, ACL2025
    </div>
    <details class="paper-abstract">
      Agentic systems powered by large language models (LLMs) are becoming progressively more complex and capable. Their increasing agency and expanding deployment settings attract growing attention to effective governance policies, monitoring, and control protocols. Based on the emerging landscape of the agentic market, we analyze potential liability issues arising from the delegated use of LLM agents and their extended systems through a principal-agent perspective. Our analysis complements existing risk-based studies on artificial agency and covers the spectrum of important aspects of the principal-agent relationship and their potential consequences at deployment. Furthermore, we motivate method developments for technical governance along the directions of interpretability and behavior evaluations, reward and conflict management, and the mitigation of misalignment and misconduct through principled engineering of detection and fail-safe mechanisms. By illustrating the outstanding issues in AI liability for LLM-based agentic systems, we aim to inform the system design, auditing, and tracing to enhance transparency and liability attribution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14496v1">LLM-Powered Swarms: A New Frontier or a Conceptual Stretch?</a></div>
    <div class="paper-meta">
      📅 2025-06-17
      | 💬 This is the author's version of a paper submitted to IEEE Intelligent Systems. 6 Tables, 3 Figures
    </div>
    <details class="paper-abstract">
      Swarm intelligence traditionally refers to systems of simple, decentralized agents whose local interactions lead to emergent, collective behavior. Recently, the term 'swarm' has been extended to describe AI systems like OpenAI's Swarm, where large language models (LLMs) act as collaborative agents. This paper contrasts traditional swarm algorithms with LLM-driven swarms exploring how decentralization, scalability, and emergence are redefined in modern artificial intelligence (AI). We implement and compare both paradigms using Boids and Ant Colony Optimization (ACO), evaluating latency, resource usage, and behavioral accuracy. The suitability of both cloud-based and local LLMs is assessed for the agent-based use in swarms. Although LLMs offer powerful reasoning and abstraction capabilities, they introduce new constraints in computation and coordination that challenge traditional notions of swarm design. This study highlights the opportunities and limitations of integrating LLMs into swarm systems and discusses the evolving definition of 'swarm' in modern AI research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.14474v1">LexiMark: Robust Watermarking via Lexical Substitutions to Enhance Membership Verification of an LLM's Textual Training Data</a></div>
    <div class="paper-meta">
      📅 2025-06-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can be trained or fine-tuned on data obtained without the owner's consent. Verifying whether a specific LLM was trained on particular data instances or an entire dataset is extremely challenging. Dataset watermarking addresses this by embedding identifiable modifications in training data to detect unauthorized use. However, existing methods often lack stealth, making them relatively easy to detect and remove. In light of these limitations, we propose LexiMark, a novel watermarking technique designed for text and documents, which embeds synonym substitutions for carefully selected high-entropy words. Our method aims to enhance an LLM's memorization capabilities on the watermarked text without altering the semantic integrity of the text. As a result, the watermark is difficult to detect, blending seamlessly into the text with no visible markers, and is resistant to removal due to its subtle, contextually appropriate substitutions that evade automated and manual detection. We evaluated our method using baseline datasets from recent studies and seven open-source models: LLaMA-1 7B, LLaMA-3 8B, Mistral 7B, Pythia 6.9B, as well as three smaller variants from the Pythia family (160M, 410M, and 1B). Our evaluation spans multiple training settings, including continued pretraining and fine-tuning scenarios. The results demonstrate significant improvements in AUROC scores compared to existing methods, underscoring our method's effectiveness in reliably verifying whether unauthorized watermarked data was used in LLM training.
    </details>
</div>
