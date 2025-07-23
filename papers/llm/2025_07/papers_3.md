# llm - 2025_07

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12295v1">Text-ADBench: Text Anomaly Detection Benchmark based on LLMs Embedding</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Text anomaly detection is a critical task in natural language processing (NLP), with applications spanning fraud detection, misinformation identification, spam detection and content moderation, etc. Despite significant advances in large language models (LLMs) and anomaly detection algorithms, the absence of standardized and comprehensive benchmarks for evaluating the existing anomaly detection methods on text data limits rigorous comparison and development of innovative approaches. This work performs a comprehensive empirical study and introduces a benchmark for text anomaly detection, leveraging embeddings from diverse pre-trained language models across a wide array of text datasets. Our work systematically evaluates the effectiveness of embedding-based text anomaly detection by incorporating (1) early language models (GloVe, BERT); (2) multiple LLMs (LLaMa-2, LLama-3, Mistral, OpenAI (small, ada, large)); (3) multi-domain text datasets (news, social media, scientific publications); (4) comprehensive evaluation metrics (AUROC, AUPRC). Our experiments reveal a critical empirical insight: embedding quality significantly governs anomaly detection efficacy, and deep learning-based approaches demonstrate no performance advantage over conventional shallow algorithms (e.g., KNN, Isolation Forest) when leveraging LLM-derived embeddings.In addition, we observe strongly low-rank characteristics in cross-model performance matrices, which enables an efficient strategy for rapid model evaluation (or embedding evaluation) and selection in practical applications. Furthermore, by open-sourcing our benchmark toolkit that includes all embeddings from different models and code at https://github.com/jicongfan/Text-Anomaly-Detection-Benchmark, this work provides a foundation for future research in robust and scalable text anomaly detection systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12215v1">Xiangqi-R1: Enhancing Spatial Strategic Reasoning in LLMs for Chinese Chess via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 10 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Game playing has long served as a fundamental benchmark for evaluating Artificial General Intelligence (AGI). While Large Language Models (LLMs) have demonstrated impressive capabilities in general reasoning, their effectiveness in spatial strategic reasoning, which is critical for complex and fully observable board games, remains insufficiently explored. In this work, we adopt Chinese Chess (Xiangqi) as a challenging and rich testbed due to its intricate rules and spatial complexity. To advance LLMs' strategic competence in such environments, we propose a training framework tailored to Xiangqi, built upon a large-scale dataset of five million board-move pairs enhanced with expert annotations and engine evaluations. Building on this foundation, we introduce Xiangqi-R1, a 7B-parameter model trained in multi-stage manner: (1) fine-tuning for legal move prediction to capture basic spatial rules, (2) incorporating strategic annotations to improve decision-making, and (3) applying reinforcement learning via Group Relative Policy Optimization (GRPO) with multi-dimensional reward signals to enhance reasoning stability. Our Experimental results indicate that, despite their size and power, general-purpose LLMs struggle to achieve satisfactory performance in these tasks. Compared to general-purpose LLMs, Xiangqi-R1 greatly advances with an 18% rise in move legality and a 22% boost in analysis accuracy. Our results point to a promising path for creating general strategic intelligence in spatially complex areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12207v1">BuildEvo: Designing Building Energy Consumption Forecasting Heuristics via LLM-driven Evolution</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 ICML 2025 CO-Build Workshop Poster
    </div>
    <details class="paper-abstract">
      Accurate building energy forecasting is essential, yet traditional heuristics often lack precision, while advanced models can be opaque and struggle with generalization by neglecting physical principles. This paper introduces BuildEvo, a novel framework that uses Large Language Models (LLMs) to automatically design effective and interpretable energy prediction heuristics. Within an evolutionary process, BuildEvo guides LLMs to construct and enhance heuristics by systematically incorporating physical insights from building characteristics and operational data (e.g., from the Building Data Genome Project 2). Evaluations show BuildEvo achieves state-of-the-art performance on benchmarks, offering improved generalization and transparent prediction logic. This work advances the automated design of robust, physically grounded heuristics, promoting trustworthy models for complex energy systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12205v1">Toward Efficient SpMV in Sparse LLMs via Block Extraction and Compressed Storage</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Sparse Matrix-Vector Multiplication (SpMV) has become a critical performance bottleneck in the local deployment of sparse Large Language Models (LLMs), where inference predominantly operates on workloads during the decoder phase with a batch size of one. Existing SpMV kernels and sparse matrix formats, originally designed for scientific computing, fail to exploit the unique structure patterns inherent in sparse LLMs, resulting in suboptimal performance and excessive storage overhead. This paper presents EC-SpMV, a GPU-optimized SpMV approach for accelerating sparse LLM inference. EC-SpMV introduces (1) a hierarchical block extraction algorithm that captures multiple granularities of block structures within sparse LLMs, and (2) a novel compressed sparse format (EC-CSR) that employs delta indexing to reduce storage overhead and enhance memory access efficiency. Evaluated on real sparse weight matrices from LLaMA and OPT models, EC-SpMV achieves up to 6.44x speedup over state-of-the-art SpMV libraries and reduces storage overhead by up to 55.4% compared to CSR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05843v4">From Objects to Events: Unlocking Complex Visual Understanding in Object Detectors via LLM-guided Symbolic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Current object detectors excel at entity localization and classification, yet exhibit inherent limitations in event recognition capabilities. This deficiency arises from their architecture's emphasis on discrete object identification rather than modeling the compositional reasoning, inter-object correlations, and contextual semantics essential for comprehensive event understanding. To address this challenge, we present a novel framework that expands the capability of standard object detectors beyond mere object recognition to complex event understanding through LLM-guided symbolic reasoning. Our key innovation lies in bridging the semantic gap between object detection and event understanding without requiring expensive task-specific training. The proposed plug-and-play framework interfaces with any open-vocabulary detector while extending their inherent capabilities across architectures. At its core, our approach combines (i) a symbolic regression mechanism exploring relationship patterns among detected entities and (ii) a LLM-guided strategically guiding the search toward meaningful expressions. These discovered symbolic rules transform low-level visual perception into interpretable event understanding, providing a transparent reasoning path from objects to events with strong transferability across domains.We compared our training-free framework against specialized event recognition systems across diverse application domains. Experiments demonstrate that our framework enhances multiple object detector architectures to recognize complex events such as illegal fishing activities (75% AUROC, +8.36% improvement), construction safety violations (+15.77%), and abnormal crowd behaviors (+23.16%). Code is available at \href{https://github.com/MAC-AutoML/SymbolicDet}{here}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09037v2">A Survey of Frontiers in LLM Reasoning: Inference Scaling, Learning to Reason, and Agentic Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 72 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Reasoning is a fundamental cognitive process that enables logical inference, problem-solving, and decision-making. With the rapid advancement of large language models (LLMs), reasoning has emerged as a key capability that distinguishes advanced AI systems from conventional models that empower chatbots. In this survey, we categorize existing methods along two orthogonal dimensions: (1) Regimes, which define the stage at which reasoning is achieved (either at inference time or through dedicated training); and (2) Architectures, which determine the components involved in the reasoning process, distinguishing between standalone LLMs and agentic compound systems that incorporate external tools, and multi-agent collaborations. Within each dimension, we analyze two key perspectives: (1) Input level, which focuses on techniques that construct high-quality prompts that the LLM condition on; and (2) Output level, which methods that refine multiple sampled candidates to enhance reasoning quality. This categorization provides a systematic understanding of the evolving landscape of LLM reasoning, highlighting emerging trends such as the shift from inference-scaling to learning-to-reason (e.g., DeepSeek-R1), and the transition to agentic workflows (e.g., OpenAI Deep Research, Manus Agent). Additionally, we cover a broad spectrum of learning algorithms, from supervised fine-tuning to reinforcement learning such as PPO and GRPO, and the training of reasoners and verifiers. We also examine key designs of agentic workflows, from established patterns like generator-evaluator and LLM debate to recent innovations. ...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12143v1">Overview of the Sensemaking Task at the ELOQUENT 2025 Lab: LLMs as Teachers, Students and Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 30 pages, 7 figures, CLEF 2025 Conference and Labs of the Evaluation Forum
    </div>
    <details class="paper-abstract">
      ELOQUENT is a set of shared tasks that aims to create easily testable high-level criteria for evaluating generative language models. Sensemaking is one such shared task. In Sensemaking, we try to assess how well generative models ``make sense out of a given text'' in three steps inspired by exams in a classroom setting: (1) Teacher systems should prepare a set of questions, (2) Student systems should answer these questions, and (3) Evaluator systems should score these answers, all adhering rather strictly to a given set of input materials. We report on the 2025 edition of Sensemaking, where we had 7 sources of test materials (fact-checking analyses of statements, textbooks, transcribed recordings of a lecture, and educational videos) spanning English, German, Ukrainian, and Czech languages. This year, 4 teams participated, providing us with 2 Teacher submissions, 2 Student submissions, and 2 Evaluator submissions. We added baselines for Teacher and Student using commercial large language model systems. We devised a fully automatic evaluation procedure, which we compare to a minimalistic manual evaluation. We were able to make some interesting observations. For the first task, the creation of questions, better evaluation strategies will still have to be devised because it is difficult to discern the quality of the various candidate question sets. In the second task, question answering, the LLMs examined overall perform acceptably, but restricting their answers to the given input texts remains problematic. In the third task, evaluation of question answers, our adversarial tests reveal that systems using the LLM-as-a-Judge paradigm erroneously rate both garbled question-answer pairs and answers to mixed-up questions as acceptable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00513v3">Leveraging LLMs for User Stories in AI Systems: UStAI Dataset</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      AI systems are gaining widespread adoption across various sectors and domains. Creating high-quality AI system requirements is crucial for aligning the AI system with business goals and consumer values and for social responsibility. However, with the uncertain nature of AI systems and the heavy reliance on sensitive data, more research is needed to address the elicitation and analysis of AI systems requirements. With the proprietary nature of many AI systems, there is a lack of open-source requirements artifacts and technical requirements documents for AI systems, limiting broader research and investigation. With Large Language Models (LLMs) emerging as a promising alternative to human-generated text, this paper investigates the potential use of LLMs to generate user stories for AI systems based on abstracts from scholarly papers. We conducted an empirical evaluation using three LLMs and generated $1260$ user stories from $42$ abstracts from $26$ domains. We assess their quality using the Quality User Story (QUS) framework. Moreover, we identify relevant non-functional requirements (NFRs) and ethical principles. Our analysis demonstrates that the investigated LLMs can generate user stories inspired by the needs of various stakeholders, offering a promising approach for generating user stories for research purposes and for aiding in the early requirements elicitation phase of AI systems. We have compiled and curated a collection of stories generated by various LLMs into a dataset (UStAI), which is now publicly available for use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12104v1">From Static to Intelligent: Evolving SaaS Pricing with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 12 pages. Accepted at the SOC4AI Workshop (Service-Oriented Computing for AI Applications), held in conjunction with the 22nd International Conference on Service-Oriented Computing (ICSOC 2024)
    </div>
    <details class="paper-abstract">
      The SaaS paradigm has revolutionized software distribution by offering flexible pricing options to meet diverse customer needs. However, the rapid expansion of the SaaS market has introduced significant complexity for DevOps teams, who must manually manage and evolve pricing structures, an approach that is both time-consuming and prone to errors. The absence of automated tools for pricing analysis restricts the ability to efficiently evaluate, optimize, and scale these models. This paper proposes leveraging intelligent pricing (iPricing), dynamic, machine-readable pricing models, as a solution to these challenges. Intelligent pricing enables competitive analysis, streamlines operational decision-making, and supports continuous pricing evolution in response to market dynamics, leading to improved efficiency and accuracy. We present an LLM-driven approach that automates the transformation of static HTML pricing into iPricing, significantly improving efficiency and consistency while minimizing human error. Our implementation, AI4Pricing2Yaml, features a basic Information Extractor that uses web scraping and LLMs technologies to extract essential pricing components, plans, features, usage limits, and add-ons, from SaaS websites. Validation against a dataset of 30 distinct commercial SaaS, encompassing over 150 intelligent pricings, demonstrates the system's effectiveness in extracting the desired elements across all steps. However, challenges remain in addressing hallucinations, complex structures, and dynamic content. This work highlights the potential of automating intelligent pricing transformation to streamline SaaS pricing management, offering implications for improved consistency and scalability in an increasingly intricate pricing landscape. Future research will focus on refining extraction capabilities and enhancing the system's adaptability to a wider range of SaaS websites.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12084v1">LLAMA: Multi-Feedback Smart Contract Fuzzing Framework with LLM-Guided Seed Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Smart contracts play a pivotal role in blockchain ecosystems, and fuzzing remains an important approach to securing smart contracts. Even though mutation scheduling is a key factor influencing fuzzing effectiveness, existing fuzzers have primarily explored seed scheduling and generation, while mutation scheduling has been rarely addressed by prior work. In this work, we propose a Large Language Models (LLMs)-based Multi-feedback Smart Contract Fuzzing framework (LLAMA) that integrates LLMs, evolutionary mutation strategies, and hybrid testing techniques. Key components of the proposed LLAMA include: (i) a hierarchical prompting strategy that guides LLMs to generate semantically valid initial seeds, coupled with a lightweight pre-fuzzing phase to select high-potential inputs; (ii) a multi-feedback optimization mechanism that simultaneously improves seed generation, seed selection, and mutation scheduling by leveraging runtime coverage and dependency feedback; and (iii) an evolutionary fuzzing engine that dynamically adjusts mutation operator probabilities based on effectiveness, while incorporating symbolic execution to escape stagnation and uncover deeper vulnerabilities. Our experiments demonstrate that LLAMA outperforms state-of-the-art fuzzers in both coverage and vulnerability detection. Specifically, it achieves 91% instruction coverage and 90% branch coverage, while detecting 132 out of 148 known vulnerabilities across diverse categories. These results highlight LLAMA's effectiveness, adaptability, and practicality in real-world smart contract security testing scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12079v1">Findings of MEGA: Maths Explanation with LLMs using the Socratic Method for Active Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 This paper was accepted for the special issue AI for Education by the IEEE Signal Processing Magazine journal
    </div>
    <details class="paper-abstract">
      This paper presents an intervention study on the effects of the combined methods of (1) the Socratic method, (2) Chain of Thought (CoT) reasoning, (3) simplified gamification and (4) formative feedback on university students' Maths learning driven by large language models (LLMs). We call our approach Mathematics Explanations through Games by AI LLMs (MEGA). Some students struggle with Maths and as a result avoid Math-related discipline or subjects despite the importance of Maths across many fields, including signal processing. Oftentimes, students' Maths difficulties stem from suboptimal pedagogy. We compared the MEGA method to the traditional step-by-step (CoT) method to ascertain which is better by using a within-group design after randomly assigning questions for the participants, who are university students. Samples (n=60) were randomly drawn from each of the two test sets of the Grade School Math 8K (GSM8K) and Mathematics Aptitude Test of Heuristics (MATH) datasets, based on the error margin of 11%, the confidence level of 90%, and a manageable number of samples for the student evaluators. These samples were used to evaluate two capable LLMs at length (Generative Pretrained Transformer 4o (GPT4o) and Claude 3.5 Sonnet) out of the initial six that were tested for capability. The results showed that students agree in more instances that the MEGA method is experienced as better for learning for both datasets. It is even much better than the CoT (47.5% compared to 26.67%) in the more difficult MATH dataset, indicating that MEGA is better at explaining difficult Maths problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10024v2">Qualitative Study for LLM-assisted Design Study Process: Strategies, Challenges, and Roles</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Design studies aim to create visualization solutions for real-world problems of different application domains. Recently, the emergence of large language models (LLMs) has introduced new opportunities to enhance the design study process, providing capabilities such as creative problem-solving, data handling, and insightful analysis. However, despite their growing popularity, there remains a lack of systematic understanding of how LLMs can effectively assist researchers in visualization-specific design studies. In this paper, we conducted a multi-stage qualitative study to fill this gap, involving 30 design study researchers from diverse backgrounds and expertise levels. Through in-depth interviews and carefully-designed questionnaires, we investigated strategies for utilizing LLMs, the challenges encountered, and the practices used to overcome them. We further compiled and summarized the roles that LLMs can play across different stages of the design study process. Our findings highlight practical implications to inform visualization practitioners, and provide a framework for leveraging LLMs to enhance the design study process in visualization research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11997v1">Can LLMs Find Fraudsters? Multi-level LLM Enhanced Graph Fraud Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Graph fraud detection has garnered significant attention as Graph Neural Networks (GNNs) have proven effective in modeling complex relationships within multimodal data. However, existing graph fraud detection methods typically use preprocessed node embeddings and predefined graph structures to reveal fraudsters, which ignore the rich semantic cues contained in raw textual information. Although Large Language Models (LLMs) exhibit powerful capabilities in processing textual information, it remains a significant challenge to perform multimodal fusion of processed textual embeddings with graph structures. In this paper, we propose a \textbf{M}ulti-level \textbf{L}LM \textbf{E}nhanced Graph Fraud \textbf{D}etection framework called MLED. In MLED, we utilize LLMs to extract external knowledge from textual information to enhance graph fraud detection methods. To integrate LLMs with graph structure information and enhance the ability to distinguish fraudsters, we design a multi-level LLM enhanced framework including type-level enhancer and relation-level enhancer. One is to enhance the difference between the fraudsters and the benign entities, the other is to enhance the importance of the fraudsters in different relations. The experiments on four real-world datasets show that MLED achieves state-of-the-art performance in graph fraud detection as a generalized framework that can be applied to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11981v1">Simplifications are Absolutists: How Simplified Language Reduces Word Sense Awareness in LLM-Generated Definitions</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 Accepted by RANLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can provide accurate word definitions and explanations for any context. However, the scope of the definition changes for different target groups, like children or language learners. This is especially relevant for homonyms, words with multiple meanings, where oversimplification might risk information loss by omitting key senses, potentially misleading users who trust LLM outputs. We investigate how simplification impacts homonym definition quality across three target groups: Normal, Simple, and ELI5. Using two novel evaluation datasets spanning multiple languages, we test DeepSeek v3, Llama 4 Maverick, Qwen3-30B A3B, GPT-4o mini, and Llama 3.1 8B via LLM-as-Judge and human annotations. Our results show that simplification drastically degrades definition completeness by neglecting polysemy, increasing the risk of misunderstanding. Fine-tuning Llama 3.1 8B with Direct Preference Optimization substantially improves homonym response quality across all prompt types. These findings highlight the need to balance simplicity and completeness in educational NLP to ensure reliable, context-aware definitions for all learners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13789v3">LEADRE: Multi-Faceted Knowledge Enhanced LLM Empowered Display Advertisement Recommender System</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 Accepted by VLDB 2025 Industrial Track
    </div>
    <details class="paper-abstract">
      Display advertising provides significant value to advertisers, publishers, and users. Traditional display advertising systems utilize a multi-stage architecture consisting of retrieval, coarse ranking, and final ranking. However, conventional retrieval methods rely on ID-based learning to rank mechanisms and fail to adequately utilize the content information of ads, which hampers their ability to provide diverse recommendation lists. To address this limitation, we propose leveraging the extensive world knowledge of LLMs. However, three key challenges arise when attempting to maximize the effectiveness of LLMs: "How to capture user interests", "How to bridge the knowledge gap between LLMs and advertising system", and "How to efficiently deploy LLMs". To overcome these challenges, we introduce a novel LLM-based framework called LLM Empowered Display ADvertisement REcommender system (LEADRE). LEADRE consists of three core modules: (1) The Intent-Aware Prompt Engineering introduces multi-faceted knowledge and designs intent-aware <Prompt, Response> pairs that fine-tune LLMs to generate ads tailored to users' personal interests. (2) The Advertising-Specific Knowledge Alignment incorporates auxiliary fine-tuning tasks and Direct Preference Optimization (DPO) to align LLMs with ad semantic and business value. (3) The Efficient System Deployment deploys LEADRE in an online environment by integrating both latency-tolerant and latency-sensitive service. Extensive offline experiments demonstrate the effectiveness of LEADRE and validate the contributions of individual modules. Online A/B test shows that LEADRE leads to a 1.57% and 1.17% GMV lift for serviced users on WeChat Channels and Moments separately. LEADRE has been deployed on both platforms, serving tens of billions of requests each day.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11959v1">PoTPTQ: A Two-step Power-of-Two Post-training for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 Accepted at ECAI 2025 (European Conference on Artificial Intelligence)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing (NLP) tasks. However, their deployment is challenging due to the substantial computational resources required. Power-of-two (PoT) quantization is a general tool to counteract this difficulty. Albeit previous works on PoT quantization can be efficiently dequantized on CPUs using fixed-point addition, it showed less effectiveness on GPUs. The reason is entanglement of the sign bit and sequential bit manipulations needed for dequantization. We propose a novel POT quantization framework for LLM weights that (i) outperforms state-of-the-art accuracy in extremely low-precision number formats, and (ii) enables faster inference through more efficient dequantization. To maintain the accuracy of the quantized model, we introduce a two-step post-training algorithm: (i) initialize the quantization scales with a robust starting point, and (ii) refine these scales using a minimal calibration set. The performance of our PoT post-training algorithm surpasses the current state-of-the-art in integer quantization, particularly at low precisions such as 2- and 3-bit formats. Our PoT quantization accelerates the dequantization step required for the floating point inference and leads to $3.67\times$ speed up on a NVIDIA V100, and $1.63\times$ on a NVIDIA RTX 4090, compared to uniform integer dequantization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11953v1">IAM: Efficient Inference through Attention Mapping between Different-scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      LLMs encounter significant challenges in resource consumption nowadays, especially with long contexts. Despite extensive efforts dedicate to enhancing inference efficiency, these methods primarily exploit internal sparsity within the models, without leveraging external information for optimization. We identify the high similarity of attention matrices across different-scale LLMs, which offers a novel perspective for optimization. We first conduct a comprehensive analysis of how to measure similarity, how to select mapping Layers and whether mapping is consistency. Based on these insights, we introduce the IAM framework, which achieves dual benefits of accelerated attention computation and reduced KV cache usage by performing attention mapping between small and large LLMs. Our experimental results demonstrate that IAM can accelerate prefill by 15% and reduce KV cache usage by 22.1% without appreciably sacrificing performance. Experiments on different series of models show the generalizability of IAM. Importantly, it is also orthogonal to many existing KV cache optimization methods, making it a versatile addition to the current toolkit for enhancing LLM efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11898v1">Extremal Testing for Network Software using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Physicists often manually consider extreme cases when testing a theory. In this paper, we show how to automate extremal testing of network software using LLMs in two steps: first, ask the LLM to generate input constraints (e.g., DNS name length limits); then ask the LLM to generate tests that violate the constraints. We demonstrate how easy this process is by generating extremal tests for HTTP, BGP and DNS implementations, each of which uncovered new bugs. We show how this methodology extends to centralized network software such as shortest path algorithms, and how LLMs can generate filtering code to reject extremal input. We propose using agentic AI to further automate extremal testing. LLM-generated extremal testing goes beyond an old technique in software testing called Boundary Value Analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00406v2">Partnering with AI: A Pedagogical Feedback System for LLM Integration into Programming Education</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 This is an extended version of a poster paper accepted and published at ECTEL-2025
    </div>
    <details class="paper-abstract">
      Feedback is one of the most crucial components to facilitate effective learning. With the rise of large language models (LLMs) in recent years, research in programming education has increasingly focused on automated feedback generation to help teachers provide timely support to every student. However, prior studies often overlook key pedagogical principles, such as mastery and progress adaptation, that shape effective feedback strategies. This paper introduces a novel pedagogical framework for LLM-driven feedback generation derived from established feedback models and local insights from secondary school teachers. To evaluate this framework, we implemented a web-based application for Python programming with LLM-based feedback that follows the framework and conducted a mixed-method evaluation with eight secondary-school computer science teachers. Our findings suggest that teachers consider that, when aligned with the framework, LLMs can effectively support students and even outperform human teachers in certain scenarios through instant and precise feedback. However, we also found several limitations, such as its inability to adapt feedback to dynamic classroom contexts. Such a limitation highlights the need to complement LLM-generated feedback with human expertise to ensure effective student learning. This work demonstrates an effective way to use LLMs for feedback while adhering to pedagogical standards and highlights important considerations for future systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11878v1">LLMs Encode Harmfulness and Refusal Separately</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11851v1">Your LLM Knows the Future: Uncovering Its Multi-Token Prediction Potential</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Autoregressive language models are constrained by their inherently sequential nature, generating one token at a time. This paradigm limits inference speed and parallelism, especially during later stages of generation when the direction and semantics of text are relatively certain. In this work, we propose a novel framework that leverages the inherent knowledge of vanilla autoregressive language models about future tokens, combining techniques to realize this potential and enable simultaneous prediction of multiple subsequent tokens. Our approach introduces several key innovations: (1) a masked-input formulation where multiple future tokens are jointly predicted from a common prefix; (2) a gated LoRA formulation that preserves the original LLM's functionality, while equipping it for multi-token prediction; (3) a lightweight, learnable sampler module that generates coherent sequences from the predicted future tokens; (4) a set of auxiliary training losses, including a consistency loss, to enhance the coherence and accuracy of jointly generated tokens; and (5) a speculative generation strategy that expands tokens quadratically in the future while maintaining high fidelity. Our method achieves significant speedups through supervised fine-tuning on pretrained models. For example, it generates code and math nearly 5x faster, and improves general chat and knowledge tasks by almost 2.5x. These gains come without any loss in quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.06875v2">Eywa: Automating Model Based Testing using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Model-based testing (MBT), whereby a model of the system under test is analyzed to generate high-coverage test cases, has been used to test protocol implementations. A key barrier to the use of MBT is the need for users to understand protocol RFCs in detail to create a compliant model. Our new approach to MBT uses LLMs to automatically build rich models of intended protocol behavior from knowledge embedded in RFCs, blogs, and other natural language sources. Our approach addresses key challenges with using LLMs, including hallucinations and their inability to monolithically generate complex protocol models. We realize our approach through a novel protocol testing framework Eywa,and demonstrate its effectiveness through extensive case studies of DNS and BGP and a smaller study of SMTP. Despite minimal user effort, applying Eywa enabled the discovery of 32 unique bugs across widely used DNS, BGP, and SMTP implementations, 15 of which were previously undiscovered despite extensive prior testing with manually crafted models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13497v4">Towards Geo-Culturally Grounded LLM Generations</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 ACL 2025 (main conference)
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) have demonstrated gaps in diverse cultural awareness across the globe. We investigate the effect of retrieval augmented generation and search-grounding techniques on LLMs' ability to display familiarity with various national cultures. Specifically, we compare the performance of standard LLMs, LLMs augmented with retrievals from a bespoke knowledge base (i.e., KB grounding), and LLMs augmented with retrievals from a web search (i.e., search grounding) on multiple cultural awareness benchmarks. We find that search grounding significantly improves the LLM performance on multiple-choice benchmarks that test propositional knowledge (e.g., cultural norms, artifacts, and institutions), while KB grounding's effectiveness is limited by inadequate knowledge base coverage and a suboptimal retriever. However, search grounding also increases the risk of stereotypical judgments by language models and fails to improve evaluators' judgments of cultural familiarity in a human evaluation with adequate statistical power. These results highlight the distinction between propositional cultural knowledge and open-ended cultural fluency when it comes to evaluating LLMs' cultural awareness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06608v4">Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Monolithic serving with chunked prefill improves GPU utilization by batching prefill and decode together, but suffers from fine-grained phase interference. Engine-level prefill-decode (PD) disaggregation avoids interference but incurs higher hardware and coordination overhead. Prior intra-GPU disaggregation approaches multiplex prefill and decode within a single GPU, using SLO-based tuning guided by heuristics from offline profiling or reactive feedback loops. However, these methods respond reactively to performance issues rather than anticipating them, limiting adaptability under dynamic workloads. We ask: can we achieve proactive intra-GPU disaggregation that adapts effectively to dynamic workloads? The key challenge lies in managing the conflicting resource demands of prefill and decode under varying conditions. We first show that GPU resources exhibit diminishing returns -- beyond a saturation point, more allocation yields minimal latency benefit. Second, we observe that memory bandwidth contention becomes a critical bottleneck. These insights motivate a design that dynamically partitions GPU resources across prefill and decode phases, while jointly considering compute capacity, memory footprint, and bandwidth contention. Evaluated on diverse LLMs and workloads, our system Nexus achieves up to 2.2x higher throughput, 20x lower TTFT, and 2.5x lower TBT than vLLM; outperforms SGLang by up to 2x; and matches or exceeds disaggregated vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12674v1">ParaStudent: Generating and Evaluating Realistic Student Code by Teaching LLMs to Struggle</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong performance on programming tasks, but can they generate student-like code like real students - imperfect, iterative, and stylistically diverse? We present ParaStudent, a systematic study of LLM-based "student-like" code generation in an introductory programming course setting. Using a dataset of timestamped student submissions across multiple semesters, we design low- and high-resolution experiments to model student progress and evaluate code outputs along semantic, functional, and stylistic dimensions. Our results show that fine-tuning significantly improves alignment with real student trajectories and captures error patterns, incremental improvements, and stylistic variations more faithfully. This study shows that modeling realistic student code requires capturing learning dynamics through context-aware generation, temporal modeling, and multi-dimensional evaluation. Code for experiments and evaluation is available at \href{https://github.com/mmiroyan/ParaStudent}{\texttt{github.com/mmiroyan/ParaStudent}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24189v2">Fine-Tune an SLM or Prompt an LLM? The Case of Generating Low-Code Workflows</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 8 pages, 7 figures. Accepted to Workshop on Structured Knowledge for Large Language Models (SKnowLLM) at KDD 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as GPT-4o can handle a wide range of complex tasks with the right prompt. As per token costs are reduced, the advantages of fine-tuning Small Language Models (SLMs) for real-world applications -- faster inference, lower costs -- may no longer be clear. In this work, we present evidence that, for domain-specific tasks that require structured outputs, SLMs still have a quality advantage. We compare fine-tuning an SLM against prompting LLMs on the task of generating low-code workflows in JSON form. We observe that while a good prompt can yield reasonable results, fine-tuning improves quality by 10% on average. We also perform systematic error analysis to reveal model limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12619v1">BootSeer: Analyzing and Mitigating Initialization Bottlenecks in Large-Scale LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 18 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become a cornerstone of modern AI, driving breakthroughs in natural language processing and expanding into multimodal jobs involving images, audio, and video. As with most computational software, it is important to distinguish between ordinary runtime performance and startup overhead. Prior research has focused on runtime performance: improving training efficiency and stability. This work focuses instead on the increasingly critical issue of startup overhead in training: the delay before training jobs begin execution. Startup overhead is particularly important in large, industrial-scale LLMs, where failures occur more frequently and multiple teams operate in iterative update-debug cycles. In one of our training clusters, more than 3.5% of GPU time is wasted due to startup overhead alone. In this work, we present the first in-depth characterization of LLM training startup overhead based on real production data. We analyze the components of startup cost, quantify its direct impact, and examine how it scales with job size. These insights motivate the design of Bootseer, a system-level optimization framework that addresses three primary startup bottlenecks: (a) container image loading, (b) runtime dependency installation, and (c) model checkpoint resumption. To mitigate these bottlenecks, Bootseer introduces three techniques: (a) hot block record-and-prefetch, (b) dependency snapshotting, and (c) striped HDFS-FUSE. Bootseer has been deployed in a production environment and evaluated on real LLM training workloads, demonstrating a 50% reduction in startup overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08339v2">What Factors Affect LLMs and RLLMs in Financial Question Answering?</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recently, the development of large language models (LLMs) and reasoning large language models (RLLMs) have gained considerable attention from many researchers. RLLMs enhance the reasoning capabilities of LLMs through Long Chain-of-Thought (Long CoT) processes, significantly improving the performance of LLMs in addressing complex problems. However, there are few works that systematically explore what methods can fully unlock the performance of LLMs and RLLMs within the financial domain. To investigate the impact of various methods on LLMs and RLLMs, we utilize five LLMs and three RLLMs to assess the effects of prompting methods, agentic frameworks, and multilingual alignment methods on financial question-answering tasks. Our research findings indicate: (1) Current prompting methods and agent frameworks enhance the performance of LLMs in financial question answering by simulating Long CoT; (2) RLLMs possess inherent Long CoT capabilities, which limits the effectiveness of conventional methods in further enhancing their performance; (3) Current advanced multilingual alignment methods primarily improve the multilingual performance of LLMs by extending the reasoning length, which yields minimal benefits for RLLMs. We hope that this study can serve as an important reference for LLMs and RLLMs in the field of financial question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.07188v2">Prompt Perturbations Reveal Human-Like Biases in LLM Survey Responses</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 18 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as proxies for human subjects in social science surveys, but their reliability and susceptibility to known response biases are poorly understood. This paper investigates the response robustness of LLMs in normative survey contexts - we test nine diverse LLMs on questions from the World Values Survey (WVS), applying a comprehensive set of 11 perturbations to both question phrasing and answer option structure, resulting in over 167,000 simulated interviews. In doing so, we not only reveal LLMs' vulnerabilities to perturbations but also show that all tested models exhibit a consistent recency bias varying in intensity, disproportionately favoring the last-presented answer option. While larger models are generally more robust, all models remain sensitive to semantic variations like paraphrasing and to combined perturbations. By applying a set of perturbations, we reveal that LLMs partially align with survey response biases identified in humans. This underscores the critical importance of prompt design and robustness testing when using LLMs to generate synthetic survey data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19477v2">Judging with Many Minds: Do More Perspectives Mean Less Prejudice? On Bias Amplifications and Resistance in Multi-Agent Based LLM-as-Judge</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      LLM-as-Judge has emerged as a scalable alternative to human evaluation, enabling large language models (LLMs) to provide reward signals in trainings. While recent work has explored multi-agent extensions such as multi-agent debate and meta-judging to enhance evaluation quality, the question of how intrinsic biases manifest in these settings remains underexplored. In this study, we conduct a systematic analysis of four diverse bias types: position bias, verbosity bias, chain-of-thought bias, and bandwagon bias. We evaluate these biases across two widely adopted multi-agent LLM-as-Judge frameworks: Multi-Agent-Debate and LLM-as-Meta-Judge. Our results show that debate framework amplifies biases sharply after the initial debate, and this increased bias is sustained in subsequent rounds, while meta-judge approaches exhibit greater resistance. We further investigate the incorporation of PINE, a leading single-agent debiasing method, as a bias-free agent within these systems. The results reveal that this bias-free agent effectively reduces biases in debate settings but provides less benefit in meta-judge scenarios. Our work provides a comprehensive study of bias behavior in multi-agent LLM-as-Judge systems and highlights the need for targeted bias mitigation strategies in collaborative evaluation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11097v1">The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 21 pages, 9 figures, work in progress
    </div>
    <details class="paper-abstract">
      Diffusion-based large language models (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs, offering faster inference and greater interactivity via parallel decoding and bidirectional modeling. However, despite strong performance in code generation and text infilling, we identify a fundamental safety concern: existing alignment mechanisms fail to safeguard dLLMs against context-aware, masked-input adversarial prompts, exposing novel vulnerabilities. To this end, we present DIJA, the first systematic study and jailbreak attack framework that exploits unique safety weaknesses of dLLMs. Specifically, our proposed DIJA constructs adversarial interleaved mask-text prompts that exploit the text generation mechanisms of dLLMs, i.e., bidirectional modeling and parallel decoding. Bidirectional modeling drives the model to produce contextually consistent outputs for masked spans, even when harmful, while parallel decoding limits model dynamic filtering and rejection sampling of unsafe content. This causes standard alignment mechanisms to fail, enabling harmful completions in alignment-tuned dLLMs, even when harmful behaviors or unsafe instructions are directly exposed in the prompt. Through comprehensive experiments, we demonstrate that DIJA significantly outperforms existing jailbreak methods, exposing a previously overlooked threat surface in dLLM architectures. Notably, our method achieves up to 100% keyword-based ASR on Dream-Instruct, surpassing the strongest prior baseline, ReNeLLM, by up to 78.5% in evaluator-based ASR on JailbreakBench and by 37.7 points in StrongREJECT score, while requiring no rewriting or hiding of harmful content in the jailbreak prompt. Our findings underscore the urgent need for rethinking safety alignment in this emerging class of language models. Code is available at https://github.com/ZichenWen1/DIJA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11086v1">Beyond Traditional Algorithms: Leveraging LLMs for Accurate Cross-Border Entity Identification</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      The growing prevalence of cross-border financial activities in global markets has underscored the necessity of accurately identifying and classifying foreign entities. This practice is essential within the Spanish financial system for ensuring robust risk management, regulatory adherence, and the prevention of financial misconduct. This process involves a labor-intensive entity-matching task, where entities need to be validated against available reference sources. Challenges arise from linguistic variations, special characters, outdated names, and changes in legal forms, complicating traditional matching algorithms like Jaccard, cosine, and Levenshtein distances. These methods struggle with contextual nuances and semantic relationships, leading to mismatches. To address these limitations, we explore Large Language Models (LLMs) as a flexible alternative. LLMs leverage extensive training to interpret context, handle abbreviations, and adapt to legal transitions. We evaluate traditional methods, Hugging Face-based LLMs, and interface-based LLMs (e.g., Microsoft Copilot, Alibaba's Qwen 2.5) using a dataset of 65 Portuguese company cases. Results show traditional methods achieve accuracies over 92% but suffer high false positive rates (20-40%). Interface-based LLMs outperform, achieving accuracies above 93%, F1 scores exceeding 96%, and lower false positives (40-80%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11083v1">Function-to-Style Guidance of LLMs for Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 This paper has been accepted by ICML 2025. Models and benchmarks can be found at https://www.modelscope.cn/collections/F2STrans-42526ff95dd843
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made significant strides in code translation tasks. However, ensuring both the correctness and readability of translated code remains a challenge, limiting their effective adoption in real-world software development. In this work, we propose F2STrans, a function-to-style guiding paradigm designed to progressively improve the performance of LLMs in code translation. Our approach comprises two key stages: (1) Functional learning, which optimizes translation correctness using high-quality source-target code pairs mined from online programming platforms, and (2) Style learning, which improves translation readability by incorporating both positive and negative style examples. Additionally, we introduce a novel code translation benchmark that includes up-to-date source code, extensive test cases, and manually annotated ground-truth translations, enabling comprehensive functional and stylistic evaluations. Experiments on both our new benchmark and existing datasets demonstrate that our approach significantly improves code translation performance. Notably, our approach enables Qwen-1.5B to outperform prompt-enhanced Qwen-32B and GPT-4 on average across 20 diverse code translation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11052v1">LLM-Augmented Symptom Analysis for Cardiovascular Disease Risk Prediction: A Clinical NLP</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Timely identification and accurate risk stratification of cardiovascular disease (CVD) remain essential for reducing global mortality. While existing prediction models primarily leverage structured data, unstructured clinical notes contain valuable early indicators. This study introduces a novel LLM-augmented clinical NLP pipeline that employs domain-adapted large language models for symptom extraction, contextual reasoning, and correlation from free-text reports. Our approach integrates cardiovascular-specific fine-tuning, prompt-based inference, and entity-aware reasoning. Evaluations on MIMIC-III and CARDIO-NLP datasets demonstrate improved performance in precision, recall, F1-score, and AUROC, with high clinical relevance (kappa = 0.82) assessed by cardiologists. Challenges such as contextual hallucination, which occurs when plausible information contracts with provided source, and temporal ambiguity, which is related with models struggling with chronological ordering of events are addressed using prompt engineering and hybrid rule-based verification. This work underscores the potential of LLMs in clinical decision support systems (CDSS), advancing early warning systems and enhancing the translation of patient narratives into actionable risk assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11042v1">Aligned Query Expansion: Efficient Query Expansion for Information Retrieval through LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      With the breakthroughs in large language models (LLMs), query generation techniques that expand documents and queries with related terms are becoming increasingly popular in the information retrieval field. Such techniques have been shown to improve the effectiveness of traditional lexical retrieval methods by dealing with the vocabulary mismatch problem. Recent work has found that generating queries with a greedy decoding strategy can produce sub-optimal queries, including hallucinations, and proposed to filter out queries before expansion. This `generate-then-filter' approach is costly, as it requires generating multiple queries and applying a relevance model to all of them and does not teach the LLM which of the generated queries is more effective for expansion. To overcome such limitations, we propose Aligned Query Expansion (AQE), a novel approach to enhance query expansion for passage retrieval in open-domain question answering. AQE leverages recent techniques in LLM alignment to fine-tune models for generating query expansions that directly optimize the effectiveness of the retrieval task, eliminating the need for additional filtering steps. This alignment ensures that queries are more relevant, reducing computational costs while improving retrieval effectiveness. Empirical evaluations show that AQE outperforms baseline models for query expansion in both in-domain and out-of-domain settings, demonstrating significant improvements in retrieval effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14959v2">Understanding the Dark Side of LLMs' Intrinsic Self-Correction</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Intrinsic self-correction was proposed to improve LLMs' responses via feedback prompts solely based on their inherent capability. However, recent works show that LLMs' intrinsic self-correction fails without oracle labels as feedback prompts. In this paper, we aim to interpret LLMs' intrinsic self-correction for different tasks, especially for those failure cases. By including one simple task and three complex tasks with state-of-the-art (SOTA) LLMs like ChatGPT families (o1, 4o, 3.5-turbo) and Llama families (2-7B, 3-8B, and 3.1-8B), we design three interpretation methods to reveal the dark side of LLMs' intrinsic self-correction. We identify intrinsic self-correction can (1) cause LLMs to waver both intermedia and final answers and lead to prompt bias on simple factual questions; (2) introduce human-like cognitive bias on complex tasks. In light of our findings, we also provide two simple yet effective strategies for alleviation: question repeating and supervised fine-tuning with a few samples. We open-source our work at https://x-isc.info/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.19502v2">MATE: LLM-Powered Multi-Agent Translation Environment for Accessibility Applications</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Accessibility remains a critical concern in today's society, as many technologies are not developed to support the full range of user needs. Existing multi-agent systems (MAS) often cannot provide comprehensive assistance for users in need due to the lack of customization stemming from closed-source designs. Consequently, individuals with disabilities frequently encounter significant barriers when attempting to interact with digital environments. We introduce MATE, a multimodal accessibility MAS, which performs the modality conversions based on the user's needs. The system is useful for assisting people with disabilities by ensuring that data will be converted to an understandable format. For instance, if the user cannot see well and receives an image, the system converts this image to its audio description. MATE can be applied to a wide range of domains, industries, and areas, such as healthcare, and can become a useful assistant for various groups of users. The system supports multiple types of models, ranging from LLM API calling to using custom machine learning (ML) classifiers. This flexibility ensures that the system can be adapted to various needs and is compatible with a wide variety of hardware. Since the system is expected to run locally, it ensures the privacy and security of sensitive information. In addition, the framework can be effectively integrated with institutional technologies (e.g., digital healthcare service) for real-time user assistance. Furthermore, we introduce ModCon-Task-Identifier, a model that is capable of extracting the precise modality conversion task from the user input. Numerous experiments show that ModCon-Task-Identifier consistently outperforms other LLMs and statistical models on our custom data. Our code and data are publicly available at https://github.com/AlgazinovAleksandr/Multi-Agent-MATE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08898v2">SEALGuard: Safeguarding the Multilingual Conversations in Southeast Asian Languages for LLM Software Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Under Review at Information and Software Technology
    </div>
    <details class="paper-abstract">
      Safety alignment is critical for LLM-powered systems. While recent LLM-powered guardrail approaches such as LlamaGuard achieve high detection accuracy of unsafe inputs written in English (e.g., ``How to create a bomb?''), they struggle with multilingual unsafe inputs. This limitation leaves LLM systems vulnerable to unsafe and jailbreak prompts written in low-resource languages such as those in Southeast Asia. This paper introduces SEALGuard, a multilingual guardrail designed to improve the safety alignment across diverse languages. It aims to address the multilingual safety alignment gap of existing guardrails and ensure effective filtering of unsafe and jailbreak prompts in LLM-powered systems. We adapt a general-purpose multilingual language model into a multilingual guardrail using low-rank adaptation (LoRA). We construct SEALSBench, a large-scale multilingual safety alignment dataset containing over 260,000 prompts in ten languages, including safe, unsafe, and jailbreak cases. We evaluate SEALGuard against state-of-the-art guardrails such as LlamaGuard on this benchmark. Our findings show that multilingual unsafe and jailbreak prompts substantially degrade the performance of the state-of-the-art LlamaGuard, which experiences a drop in Defense Success Rate (DSR) by 9% and 18%, respectively, compared to its performance on English-only prompts. In contrast, SEALGuard outperforms existing guardrails in detecting multilingual unsafe and jailbreak prompts, improving DSR by 48% over LlamaGuard and achieving the best DSR, precision, and F1-score. Our ablation study further reveals the contributions of adaptation strategies and model size to the overall performance of SEALGuard. SEALGuard advances the safety alignment of LLM systems by introducing an effective multilingual guardrail.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10972v1">Teach Me Sign: Stepwise Prompting LLM for Sign Language Production</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepted by IEEE ICIP 2025
    </div>
    <details class="paper-abstract">
      Large language models, with their strong reasoning ability and rich knowledge, have brought revolution to many tasks of AI, but their impact on sign language generation remains limited due to its complexity and unique rules. In this paper, we propose TEAch Me Sign (TEAM-Sign), treating sign language as another natural language. By fine-tuning an LLM, we enable it to learn the correspondence between text and sign language, and facilitate generation. Considering the differences between sign and spoken language, we employ a stepwise prompting strategy to extract the inherent sign language knowledge within the LLM, thereby supporting the learning and generation process. Experimental results on How2Sign and Phoenix14T datasets demonstrate that our approach effectively leverages both the sign language knowledge and reasoning capabilities of LLM to align the different distribution and grammatical rules between sign and spoken language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10920v1">HanjaBridge: Resolving Semantic Ambiguity in Korean LLMs via Hanja-Augmented Pre-Training</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often show poor performance in low-resource languages like Korean, partly due to unique linguistic challenges such as homophonous Sino-Korean words that are indistinguishable in Hangul script. To address this semantic ambiguity, we propose HanjaBridge, a novel meaning-injection technique integrated into a continual pre-training (CPT) framework. Instead of deterministically mapping a word to a single Hanja (Chinese character), HanjaBridge presents the model with all possible Hanja candidates for a given homograph, encouraging the model to learn contextual disambiguation. This process is paired with token-level knowledge distillation to prevent catastrophic forgetting. Experimental results show that HanjaBridge significantly improves Korean language understanding, achieving a 21\% relative improvement on the KoBALT benchmark. Notably, by reinforcing semantic alignment between Korean and Chinese through shared Hanja, we observe a strong positive cross-lingual transfer. Furthermore, these gains persist even when Hanja augmentation is omitted at inference time, ensuring practical efficiency with no additional run-time cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10917v1">LLM-Driven Dual-Level Multi-Interest Modeling for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recently, much effort has been devoted to modeling users' multi-interests based on their behaviors or auxiliary signals. However, existing methods often rely on heuristic assumptions, e.g., co-occurring items indicate the same interest of users, failing to capture user multi-interests aligning with real-world scenarios. While large language models (LLMs) show significant potential for multi-interest analysis due to their extensive knowledge and powerful reasoning capabilities, two key challenges remain. First, the granularity of LLM-driven multi-interests is agnostic, possibly leading to overly fine or coarse interest grouping. Second, individual user analysis provides limited insights due to the data sparsity issue. In this paper, we propose an LLM-driven dual-level multi-interest modeling framework for more effective recommendation. At the user-individual level, we exploit LLMs to flexibly allocate items engaged by users into different semantic clusters, indicating their diverse and distinct interests. To alleviate the agnostic generation of LLMs, we adaptively assign these semantic clusters to users' collaborative multi-interests learned from global user-item interactions, allowing the granularity to be automatically adjusted according to the user's behaviors using an alignment module. To alleviate the limited insights derived from individual users' behaviors, at the user-crowd level, we propose aggregating user cliques into synthesized users with rich behaviors for more comprehensive LLM-driven multi-interest analysis. We formulate a max covering problem to ensure the compactness and representativeness of synthesized users' behaviors, and then conduct contrastive learning based on their LLM-driven multi-interests to disentangle item representations among different interests. Experiments on real-world datasets show the superiority of our approach against state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09116v2">Mixture of LoRA Experts with Multi-Modal and Multi-Granularity LLM Generative Error Correction for Accented Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 IEEE Transactions on Audio, Speech and Language Processing
    </div>
    <details class="paper-abstract">
      Despite substantial improvements in ASR, performance tends to degrade when faced with adverse conditions such as speaker accents. Generative error correction (GER) leverages the rich linguistic knowledge and exceptional reasoning ability of LLMs, significantly outperforming typical LM methods. However, it lacks specificity in accented speech scenarios. In this study, we leverage GER to improve the accuracy of transcription predictions by addressing the two primary features of accented speech recognition. To fully leverage pronunciation information, we propose the multi-modal GER, which integrates pronunciation information from the speech modality, and the multi-granularity GER, which incorporates fine-grained phoneme-level information related to pronunciation. These two methods enable the LLM to utilize the pronunciation information of accented speech and the semantic information from word-level hypotheses for accurate transcription predictions through LoRA fine-tuning. On the one hand, we employ a three-stage training strategy to train separate multi-modal GER models for each accent to obtain mono-accent LoRA experts. By adopting our proposed HDMoLE method, which incorporates hierarchical routing and dynamic thresholds within the mixture of LoRA experts, we effectively merge multiple mono-accent LoRA experts within a single multi-modal GER to overcome the challenges posed by accent diversity. On the other hand, multi-granularity GER leverages the N-best word-level and phoneme-level hypotheses generated by the HDMoLE model to predict the final accented speech transcriptions. Experimental results on the multi-accent English dataset demonstrate the efficacy of our proposed methods. Our methods achieve a remarkable relative WER reduction of 67.35% compared to the Whisper-large-v3 baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10911v1">Lessons Learned from Evaluation of LLM based Multi-agents in Safer Therapy Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Therapy recommendation for chronic patients with multimorbidity is challenging due to risks of treatment conflicts. Existing decision support systems face scalability limitations. Inspired by the way in which general practitioners (GP) manage multimorbidity patients, occasionally convening multidisciplinary team (MDT) collaboration, this study investigated the feasibility and value of using a Large Language Model (LLM)-based multi-agent system (MAS) for safer therapy recommendations. We designed a single agent and a MAS framework simulating MDT decision-making by enabling discussion among LLM agents to resolve medical conflicts. The systems were evaluated on therapy planning tasks for multimorbidity patients using benchmark cases. We compared MAS performance with single-agent approaches and real-world benchmarks. An important contribution of our study is the definition of evaluation metrics that go beyond the technical precision and recall and allow the inspection of clinical goals met and medication burden of the proposed advices to a gold standard benchmark. Our results show that with current LLMs, a single agent GP performs as well as MDTs. The best-scoring models provide correct recommendations that address all clinical goals, yet the advices are incomplete. Some models also present unnecessary medications, resulting in unnecessary conflicts between medication and conditions or drug-drug interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06608v3">Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Monolithic serving with chunked prefill improves GPU utilization by batching prefill and decode together, but suffers from fine-grained phase interference. Engine-level prefill-decode (PD) disaggregation avoids interference but incurs higher hardware and coordination overhead. Prior intra-GPU disaggregation approaches multiplex prefill and decode within a single GPU, using SLO-based tuning guided by heuristics from offline profiling or reactive feedback loops. However, these methods respond reactively to performance issues rather than anticipating them, limiting adaptability under dynamic workloads. We ask: can we achieve proactive intra-GPU disaggregation that adapts effectively to dynamic workloads? The key challenge lies in managing the conflicting resource demands of prefill and decode under varying conditions. We first show that GPU resources exhibit diminishing returns -- beyond a saturation point, more allocation yields minimal latency benefit. Second, we observe that memory bandwidth contention becomes a critical bottleneck. These insights motivate a design that dynamically partitions GPU resources across prefill and decode phases, while jointly considering compute capacity, memory footprint, and bandwidth contention. Evaluated on diverse LLMs and workloads, our system Nexus achieves up to 2.2x higher throughput, 20x lower TTFT, and 2.5x lower TBT than vLLM; outperforms SGLang by up to 2x; and matches or exceeds disaggregated vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01100v2">ZebraLogic: On the Scaling Limits of LLMs for Logical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepted to ICML 2025
    </div>
    <details class="paper-abstract">
      We investigate the logical reasoning capabilities of large language models (LLMs) and their scalability in complex non-monotonic reasoning. To this end, we introduce ZebraLogic, a comprehensive evaluation framework for assessing LLM reasoning performance on logic grid puzzles derived from constraint satisfaction problems (CSPs). ZebraLogic enables the generation of puzzles with controllable and quantifiable complexity, facilitating a systematic study of the scaling limits of models such as Llama, o1 models, and DeepSeek-R1. By encompassing a broad range of search space complexities and diverse logical constraints, ZebraLogic provides a structured environment to evaluate reasoning under increasing difficulty. Our results reveal a significant decline in accuracy as problem complexity grows -- a phenomenon we term the curse of complexity. This limitation persists even with larger models and increased inference-time computation, suggesting inherent constraints in current LLM reasoning capabilities. Additionally, we explore strategies to enhance logical reasoning, including Best-of-N sampling, backtracking mechanisms, and self-verification prompts. Our findings offer critical insights into the scalability of LLM reasoning, highlight fundamental limitations, and outline potential directions for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10873v1">From Alerts to Intelligence: A Novel LLM-Aided Framework for Host-based Intrusion Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Host-based intrusion detection system (HIDS) is a key defense component to protect the organizations from advanced threats like Advanced Persistent Threats (APT). By analyzing the fine-grained logs with approaches like data provenance, HIDS has shown successes in capturing sophisticated attack traces. Despite the progresses embarked by the research community and industry, HIDS still frequently encounters backlash from their operators in the deployed environments, due to issues like high false-positive rate, inconsistent outcomes across environments and human-unfriendly detection results. Large Language Models (LLMs) have great potentials to advance the state of HIDS, given their extensive knowledge of attack techniques and their ability to detect anomalies through semantic analysis, anchored by recent studies. Yet, our preliminary analysis indicates that building an HIDS by naively prompting an LLM is unlikely to succeed. In this work, we explore the direction of building a customized LLM pipeline for HIDS and develop a system named SHIELD. SHIELD addresses challenges related to LLM's token limits, confusion of background noises, etc., by integrating a variety of techniques like event-level Masked Autoencoder (MAE) for attack window detection, attack evidence identification and expansion, Deterministic Data Augmentation (DDA) for profiling normal activities, and multi-purpose prompting that guides the LLM to conduct precise and interpretable attack investigations. Extensive experiments on three log datasets (DARPA-E3, NodLink-simulated-data and ATLASv2) show that SHIELD consistently achieves outstanding performance in comparison with 5 representative HIDS. These findings highlight the potential of LLMs as powerful tools for intrusion detection and pave the way for future research in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21418v2">Autonomous Multi-Modal LLM Agents for Treatment Planning in Focused Ultrasound Ablation Surgery</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Focused Ultrasound Ablation Surgery (FUAS) has emerged as a promising non-invasive therapeutic modality, valued for its safety and precision. Nevertheless, its clinical implementation entails intricate tasks such as multimodal image interpretation, personalized dose planning, and real-time intraoperative decision-making processes that demand intelligent assistance to improve efficiency and reliability. We introduce FUAS-Agents, an autonomous agent system that leverages the multimodal understanding and tool-using capabilities of large language models (LLMs). By integrating patient profiles and MRI data, FUAS-Agents orchestrates a suite of specialized medical AI tools, including segmentation, treatment dose prediction, and clinical guideline retrieval, to generate personalized treatment plans comprising MRI image, dose parameters, and therapeutic strategies. We evaluate the system in a uterine fibroid treatment scenario. Human assessment by four senior FUAS experts indicates that 82.5%, 82.5%, 87.5%, and 97.5% of the generated plans were rated 4 or above (on a 5-point scale) in terms of completeness, accuracy, fluency, and clinical compliance, respectively. These results demonstrate the potential of LLM-driven agents in enhancing decision-making across complex clinical workflows, and exemplify a translational paradigm that combines general-purpose models with specialized expert systems to solve practical challenges in vertical healthcare domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08140v3">Lost in Transmission: When and Why LLMs Fail to Reason Globally</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      Despite their many successes, transformer-based large language models (LLMs) continue to struggle with tasks that require complex reasoning over large parts of their input. We argue that these failures arise due to capacity limits on the accurate flow of information within LLMs. To formalize this issue, we introduce the bounded attention prefix oracle (BAPO) model, a new computational framework that models bandwidth constraints on attention heads, the mechanism for internal communication in LLMs. We show that several important reasoning problems like graph reachability require high communication bandwidth for BAPOs to solve; we call these problems BAPO-hard. Our experiments corroborate our theoretical predictions: GPT-4o, Claude, and Gemini succeed on BAPO-easy tasks and fail even on relatively small BAPO-hard tasks. BAPOs also reveal another benefit of chain of thought (CoT): we prove that breaking down a task using CoT can turn any BAPO-hard problem into a BAPO-easy one. Our results offer principled explanations for key LLM failures and suggest directions for architectures and inference methods that mitigate bandwidth limits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05566v2">ScaleRTL: Scaling LLMs with Reasoning Data and Test-Time Compute for Accurate RTL Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepted to MLCAD 2025
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled near-human performance on software coding benchmarks, but their effectiveness in RTL code generation remains limited due to the scarcity of high-quality training data. While prior efforts have fine-tuned LLMs for RTL tasks, they do not fundamentally overcome the data bottleneck and lack support for test-time scaling due to their non-reasoning nature. In this work, we introduce ScaleRTL, the first reasoning LLM for RTL coding that scales up both high-quality reasoning data and test-time compute. Specifically, we curate a diverse set of long chain-of-thought reasoning traces averaging 56K tokens each, resulting in a dataset of 3.5B tokens that captures rich RTL knowledge. Fine-tuning a general-purpose reasoning model on this corpus yields ScaleRTL that is capable of deep RTL reasoning. Subsequently, we further enhance the performance of ScaleRTL through a novel test-time scaling strategy that extends the reasoning process via iteratively reflecting on and self-correcting previous reasoning steps. Experimental results show that ScaleRTL achieves state-of-the-art performance on VerilogEval and RTLLM, outperforming 18 competitive baselines by up to 18.4% on VerilogEval and 12.7% on RTLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11742v1">CRABS: A syntactic-semantic pincer strategy for bounding LLM interpretation of Python notebooks</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Preprint. Accepted to COLM 2025
    </div>
    <details class="paper-abstract">
      Recognizing the information flows and operations comprising data science and machine learning Python notebooks is critical for evaluating, reusing, and adapting notebooks for new tasks. Investigating a notebook via re-execution often is impractical due to the challenges of resolving data and software dependencies. While Large Language Models (LLMs) pre-trained on large codebases have demonstrated effectiveness in understanding code without running it, we observe that they fail to understand some realistic notebooks due to hallucinations and long-context challenges. To address these issues, we propose a notebook understanding task yielding an information flow graph and corresponding cell execution dependency graph for a notebook, and demonstrate the effectiveness of a pincer strategy that uses limited syntactic analysis to assist full comprehension of the notebook using an LLM. Our Capture and Resolve Assisted Bounding Strategy (CRABS) employs shallow syntactic parsing and analysis of the abstract syntax tree (AST) to capture the correct interpretation of a notebook between lower and upper estimates of the inter-cell I/O sets, then uses an LLM to resolve remaining ambiguities via cell-by-cell zero-shot learning, thereby identifying the true data inputs and outputs of each cell. We evaluate and demonstrate the effectiveness of our approach using an annotated dataset of 50 representative, highly up-voted Kaggle notebooks that together represent 3454 actual cell inputs and outputs. The LLM correctly resolves 1397 of 1425 (98%) ambiguities left by analyzing the syntactic structure of these notebooks. Across 50 notebooks, CRABS achieves average F1 scores of 98% identifying cell-to-cell information flows and 99% identifying transitive cell execution dependencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05102v2">You Can REST Now: Automated REST API Documentation and Testing via LLM-Assisted Request Mutations</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      REST APIs are prevalent among web service implementations, easing interoperability through the HTTP protocol. API testers and users exploit the widely adopted OpenAPI Specification (OAS), a machine-readable standard to document REST APIs. However, documenting APIs is a time-consuming and error-prone task, and existing documentation is not always complete, publicly accessible, or up-to-date. This situation limits the efficiency of testing tools and hinders human comprehension. Large Language Models (LLMs) offer the potential to automatically infer API documentation, using their colossal training data. In this paper, we present RESTSpecIT, the first automated approach that infers documentation and performs black-box testing of REST APIs by leveraging LLMs. Our approach requires minimal user input compared to state-of-the-art tools; Given an API name and an LLM access key, RESTSpecIT generates API request seeds and mutates them with data returned by the LLM. The tool then analyzes API responses for documentation inference and testing purposes. RESTSpecIT utilizes an in-context prompt masking strategy, requiring no prior model fine-tuning. We evaluate the quality of our tool with three state-of-the-art LLMs: DeepSeek V3, GPT-4.1, and GPT-3.5. Our evaluation demonstrates that RESTSpecIT can (1) infer documentation with 88.62% of routes and 89.25% of query parameters found on average, (2) discover undocumented API data, (3) operate efficiently (in terms of model costs, requests sent, runtime), and (4) assist REST API testing by uncovering server errors and generating valid OpenAPI Specification inputs for testing tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16069v2">Rolling the DICE on Idiomaticity: How LLMs Fail to Grasp Context</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Human processing of idioms relies on understanding the contextual sentences in which idioms occur, as well as language-intrinsic features such as frequency and speaker-intrinsic factors like familiarity. While LLMs have shown high performance on idiomaticity detection tasks, this success may be attributed to reasoning shortcuts in existing datasets. To this end, we construct a novel, controlled contrastive dataset designed to test whether LLMs can effectively use context to disambiguate idiomatic meaning. Additionally, we explore how collocational frequency and sentence probability influence model performance. Our findings reveal that LLMs often fail to resolve idiomaticity when it is required to attend to the surrounding context, and that models perform better on sentences that have higher likelihood. The collocational frequency of expressions also impacts performance. We make our code and dataset publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10323v3">ELFuzz: Efficient Input Generation via LLM-driven Synthesis Over Fuzzer Space</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepted by USENIX Security'25 Cycle 2
    </div>
    <details class="paper-abstract">
      Generation-based fuzzing produces appropriate testing cases according to specifications of input grammars and semantic constraints to test systems and software. However, these specifications require significant manual efforts to construct. This paper proposes a new approach, ELFuzz (Evolution Through Large Language Models for Fuzzing), that automatically synthesizes generation-based fuzzers tailored to a system under test (SUT) via LLM-driven synthesis over fuzzer space. At a high level, it starts with minimal seed fuzzers and propels the synthesis by fully automated LLM-driven evolution with coverage guidance. Compared to previous approaches, ELFuzz can 1) seamlessly scale to SUTs of real-world sizes -- up to 1,791,104 lines of code in our evaluation -- and 2) synthesize efficient fuzzers that catch interesting grammatical structures and semantic constraints in a human-understandable way. Our evaluation compared ELFuzz with specifications manually written by domain experts and synthesized by state-of-the-art approaches. It shows that ELFuzz achieves up to 434.8% more coverage and triggers up to 174.0% more artificially injected bugs. We also used ELFuzz to conduct a real-world fuzzing campaign on the newest version of cvc5 for 14 days, and encouragingly, it found five 0-day bugs (three are exploitable). Moreover, we conducted an ablation study, which shows that the fuzzer space model, the key component of ELFuzz, contributes the most (up to 62.5%) to the effectiveness of ELFuzz. Further analysis of the fuzzers synthesized by ELFuzz confirms that they catch interesting grammatical structures and semantic constraints in a human-understandable way. The results present the promising potential of ELFuzz for more automated, efficient, and extensible input generation for fuzzing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11633v1">General Modular Harness for LLM Agents in Multi-Turn Gaming Environments</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 8 pages, ICML MAS workshop
    </div>
    <details class="paper-abstract">
      We introduce a modular harness design for LLM agents that composes of perception, memory, and reasoning components, enabling a single LLM or VLM backbone to tackle a wide spectrum of multi turn gaming environments without domain-specific engineering. Using classic and modern game suites as low-barrier, high-diversity testbeds, our framework provides a unified workflow for analyzing how each module affects performance across dynamic interactive settings. Extensive experiments demonstrate that the harness lifts gameplay performance consistently over un-harnessed baselines and reveals distinct contribution patterns, for example, memory dominates in long-horizon puzzles while perception is critical in vision noisy arcades. These findings highlight the effectiveness of our modular harness design in advancing general-purpose agent, given the familiarity and ubiquity of games in everyday human experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09598v3">How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      This paper introduces a novel infrastructure-aware benchmarking framework for quantifying the environmental footprint of LLM inference across 30 state-of-the-art models as deployed in commercial data centers. Our framework combines public API performance data with region-specific environmental multipliers and statistical inference of hardware configurations. We additionally utilize cross-efficiency Data Envelopment Analysis (DEA) to rank models by performance relative to environmental cost. Our results show that o3 and DeepSeek-R1 emerge as the most energy-intensive models, consuming over 33 Wh per long prompt, more than 70 times the consumption of GPT-4.1 nano, and that Claude-3.7 Sonnet ranks highest in eco-efficiency. While a single short GPT-4o query consumes 0.42 Wh, scaling this to 700 million queries/day results in substantial annual environmental impacts. These include electricity use comparable to 35,000 U.S. homes, freshwater evaporation matching the annual drinking needs of 1.2 million people, and carbon emissions requiring a Chicago-sized forest to offset. These findings illustrate a growing paradox: Although AI is becoming cheaper and faster, its global adoption drives disproportionate resource consumption. Our study provides a standardized, empirically grounded methodology for benchmarking the sustainability of LLM deployments, laying a foundation for future environmental accountability in AI development and sustainability standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13381v1">SAFT: Structure-Aware Fine-Tuning of LLMs for AMR-to-Text Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepted at the KDD2025 Workshop on Structured Knowledge for LLMs
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly applied to tasks involving structured inputs such as graphs. Abstract Meaning Representations (AMRs), which encode rich semantics as directed graphs, offer a rigorous testbed for evaluating LLMs on text generation from such structures. Yet, current methods often arbitrarily linearize AMRs, discarding key structural cues, or rely on architectures incompatible with standard LLMs. We introduce SAFT, a structure-aware fine-tuning approach that injects graph topology into pretrained LLMs without architectural changes. We compute direction-sensitive positional encodings from the magnetic Laplacian of transformed AMRs and project them into the embedding space of the LLM. While possibly applicable to any graph-structured inputs, we focus on AMR-to-text generation as a representative and challenging benchmark. SAFT sets a new state-of-the-art on AMR 3.0 with a 3.5 BLEU improvement over baselines. Gains scale with graph complexity, highlighting the value of structure-aware representations in enhancing LLM performance. SAFT offers a general and effective pathway for bridging structured data and language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11538v1">How Many Instructions Can LLMs Follow at Once?</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Production-grade LLM systems require robust adherence to dozens or even hundreds of instructions simultaneously. However, the instruction-following capabilities of LLMs at high instruction densities have not yet been characterized, as existing benchmarks only evaluate models on tasks with a single or few instructions. We introduce IFScale, a simple benchmark of 500 keyword-inclusion instructions for a business report writing task to measure how instruction-following performance degrades as instruction density increases. We evaluate 20 state-of-the-art models across seven major providers and find that even the best frontier models only achieve 68% accuracy at the max density of 500 instructions. Our analysis reveals model size and reasoning capability to correlate with 3 distinct performance degradation patterns, bias towards earlier instructions, and distinct categories of instruction-following errors. Our insights can help inform design of instruction-dense prompts in real-world applications and highlight important performance-latency tradeoffs. We open-source the benchmark and all results for further analysis at https://distylai.github.io/IFScale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11525v1">LLM-based ambiguity detection in natural language instructions for collaborative surgical robots</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepted at 2025 IEEE International Conference on Robot and Human Interactive Communication (ROMAN)
    </div>
    <details class="paper-abstract">
      Ambiguity in natural language instructions poses significant risks in safety-critical human-robot interaction, particularly in domains such as surgery. To address this, we propose a framework that uses Large Language Models (LLMs) for ambiguity detection specifically designed for collaborative surgical scenarios. Our method employs an ensemble of LLM evaluators, each configured with distinct prompting techniques to identify linguistic, contextual, procedural, and critical ambiguities. A chain-of-thought evaluator is included to systematically analyze instruction structure for potential issues. Individual evaluator assessments are synthesized through conformal prediction, which yields non-conformity scores based on comparison to a labeled calibration dataset. Evaluating Llama 3.2 11B and Gemma 3 12B, we observed classification accuracy exceeding 60% in differentiating ambiguous from unambiguous surgical instructions. Our approach improves the safety and reliability of human-robot collaboration in surgery by offering a mechanism to identify potentially ambiguous instructions before robot action.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11515v1">AirLLM: Diffusion Policy-based Adaptive LoRA for Remote Fine-Tuning of LLM over the Air</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 11 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Operating Large Language Models (LLMs) on edge devices is increasingly challenged by limited communication bandwidth and strained computational and memory costs. Thus, cloud-assisted remote fine-tuning becomes indispensable. Nevertheless, existing Low-Rank Adaptation (LoRA) approaches typically employ fixed or heuristic rank configurations, and the subsequent over-the-air transmission of all LoRA parameters could be rather inefficient. To address this limitation, we develop AirLLM, a hierarchical diffusion policy framework for communication-aware LoRA adaptation. Specifically, AirLLM models the rank configuration as a structured action vector that spans all LoRA-inserted projections. To solve the underlying high-dimensional sequential decision-making problem, a Proximal Policy Optimization (PPO) agent generates coarse-grained decisions by jointly observing wireless states and linguistic complexity, which are then refined via Denoising Diffusion Implicit Models (DDIM) to produce high-resolution, task- and channel-adaptive rank vectors. The two modules are optimized alternatively, with the DDIM trained under the Classifier-Free Guidance (CFG) paradigm to maintain alignment with PPO rewards. Experiments under varying signal-to-noise ratios demonstrate that AirLLM consistently enhances fine-tuning performance while significantly reducing transmission costs, highlighting the effectiveness of reinforcement-driven, diffusion-refined rank adaptation for scalable and efficient remote fine-tuning over the air.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11507v1">MIRAGE: KV Cache Optimization through Parameter Remapping for Multi-tenant LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      KV cache accelerates LLM inference by avoiding redundant computation, at the expense of memory. To support larger KV caches, prior work extends GPU memory with CPU memory via CPU-offloading. This involves swapping KV cache between GPU and CPU memory. However, because the cache updates dynamically, such swapping incurs high CPU memory traffic. We make a key observation that model parameters remain constant during runtime, unlike the dynamically updated KV cache. Building on this, we introduce MIRAGE, which avoids KV cache swapping by remapping, and thereby repurposing, the memory allocated to model parameters for KV cache. This parameter remapping is especially beneficial in multi-tenant environments, where the memory used for the parameters of the inactive models can be more aggressively reclaimed. Exploiting the high CPU-GPU bandwidth offered by the modern hardware, such as the NVIDIA Grace Hopper Superchip, we show that MIRAGE significantly outperforms state-of-the-art solutions, achieving a reduction of 44.8%-82.5% in tail time-between-token latency, 20.7%-99.3% in tail time-to-first-token latency, and 6.6%-86.7% higher throughput compared to vLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11457v1">LRMR: LLM-Driven Relational Multi-node Ranking for Lymph Node Metastasis Assessment in Rectal Cancer</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Accurate preoperative assessment of lymph node (LN) metastasis in rectal cancer guides treatment decisions, yet conventional MRI evaluation based on morphological criteria shows limited diagnostic performance. While some artificial intelligence models have been developed, they often operate as black boxes, lacking the interpretability needed for clinical trust. Moreover, these models typically evaluate nodes in isolation, overlooking the patient-level context. To address these limitations, we introduce LRMR, an LLM-Driven Relational Multi-node Ranking framework. This approach reframes the diagnostic task from a direct classification problem into a structured reasoning and ranking process. The LRMR framework operates in two stages. First, a multimodal large language model (LLM) analyzes a composite montage image of all LNs from a patient, generating a structured report that details ten distinct radiological features. Second, a text-based LLM performs pairwise comparisons of these reports between different patients, establishing a relative risk ranking based on the severity and number of adverse features. We evaluated our method on a retrospective cohort of 117 rectal cancer patients. LRMR achieved an area under the curve (AUC) of 0.7917 and an F1-score of 0.7200, outperforming a range of deep learning baselines, including ResNet50 (AUC 0.7708). Ablation studies confirmed the value of our two main contributions: removing the relational ranking stage or the structured prompting stage led to a significant performance drop, with AUCs falling to 0.6875 and 0.6458, respectively. Our work demonstrates that decoupling visual perception from cognitive reasoning through a two-stage LLM framework offers a powerful, interpretable, and effective new paradigm for assessing lymph node metastasis in rectal cancer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23847v3">Seven Security Challenges That Must be Solved in Cross-domain Multi-agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly evolving into autonomous agents that cooperate across organizational boundaries, enabling joint disaster response, supply-chain optimization, and other tasks that demand decentralized expertise without surrendering data ownership. Yet, cross-domain collaboration shatters the unified trust assumptions behind current alignment and containment techniques. An agent benign in isolation may, when receiving messages from an untrusted peer, leak secrets or violate policy, producing risks driven by emergent multi-agent dynamics rather than classical software bugs. This position paper maps the security agenda for cross-domain multi-agent LLM systems. We introduce seven categories of novel security challenges, for each of which we also present plausible attacks, security evaluation metrics, and future research guidelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16366v3">A Generative Approach to LLM Harmfulness Detection with Special Red Flag Tokens</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Most safety training methods for large language models (LLMs) are based on fine-tuning that forces models to shift from an unsafe answer to refusal when faced with harmful requests. Unfortunately, these drastic distribution shifts generally compromise model capabilities. To avoid that, we propose to expand the model's vocabulary with a special token we call red flag token (<rf>) and propose to train the model to insert this token into its response at any time when harmful content is generated or about to be generated. Our approach offers several advantages: it enables the model to explicitly learn the concept of harmfulness while marginally affecting the generated distribution, thus maintaining the model's utility. It also evaluates each generated answer and provides robustness as good as adversarial training without the need to run attacks during training. Moreover, by encapsulating our safety tuning in a LoRA module, we provide additional defenses against fine-tuning API attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11417v1">Quantifying the Energy Consumption and Carbon Emissions of LLM Inference via Simulations</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Presented at the Workshop on Performance and Energy Efficiency in Concurrent and Distributed Systems (PECS) at Euro-PAR'25
    </div>
    <details class="paper-abstract">
      The environmental impact of Large Language Models (LLMs) is rising significantly, with inference now accounting for more than half of their total lifecycle carbon emissions. However, existing simulation frameworks, which are increasingly used to determine efficient LLM deployments, lack any concept of power and, therefore, cannot accurately estimate inference-related emissions. We present a simulation framework to assess the energy and carbon implications of LLM inference under varying deployment setups. First, we extend a high-fidelity LLM inference simulator with a GPU power model that estimates power consumption based on utilization metrics, enabling analysis across configurations like batch size, sequence length, and model parallelism. Second, we integrate simulation outputs into an energy system co-simulation environment to quantify carbon emissions under specific grid conditions and explore the potential of carbon-aware scheduling. Through scenario-based analysis, our framework reveals how inference parameters affect energy demand and carbon footprint, demonstrates a renewable offset potential of up to 69.2% in an illustrative deployment case, and provides a foundation for future carbon-aware inference infrastructure design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00648v4">DFRot: Achieving Outlier-Free and Massive Activation-Free for Rotated LLMs with Refined Rotation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepeted bythe 2nd Conference on Language Modeling (COLM 2025). Source code \url{https://github.com/JingyangXiang/DFRot}
    </div>
    <details class="paper-abstract">
      Rotating the activation and weight matrices to reduce the influence of outliers in large language models (LLMs) has recently attracted significant attention, particularly in the context of model quantization. Prior studies have shown that in low-precision quantization scenarios, such as 4-bit weights and 4-bit activations (W4A4), randomized Hadamard transforms can achieve significantly higher accuracy than randomized orthogonal transforms. Notably, the reason behind this phenomenon remains unknown. In this paper, we find that these transformations show substantial improvement in eliminating outliers for common tokens and achieve similar quantization error. The primary reason for the accuracy difference lies in the fact that randomized Hadamard transforms can slightly reduce the quantization error for tokens with massive activations while randomized orthogonal transforms increase the quantization error. Due to the extreme rarity of these tokens and their critical impact on model accuracy, we consider this a long-tail optimization problem, and therefore construct a simple yet effective method: a weighted loss function. Additionally, we propose an optimization strategy for the rotation matrix that involves alternating optimization of quantization parameters while employing orthogonal Procrustes transforms to refine the rotation matrix. This makes the distribution of the rotated activation values more conducive to quantization, especially for tokens with massive activations. Our method enhances the Rotated LLMs by achieving dual free, Outlier-Free and Massive Activation-Free, dubbed as DFRot. Extensive experiments demonstrate the effectiveness and efficiency of DFRot. By tuning the rotation matrix using just a single sample, DFRot achieves a perplexity improvement of 0.98 and 0.95 on W4A4KV4 and W4A4KV16, respectively, for LLaMA3-70B, a model known for its quantization challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11408v1">KisMATH: Do LLMs Have Knowledge of Implicit Structures in Mathematical Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 15 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Chain-of-thought traces have been shown to improve performance of large language models in a plethora of reasoning tasks, yet there is no consensus on the mechanism through which this performance boost is achieved. To shed more light on this, we introduce Causal CoT Graphs (CCGs), which are directed acyclic graphs automatically extracted from reasoning traces that model fine-grained causal dependencies in the language model output. A collection of $1671$ mathematical reasoning problems from MATH500, GSM8K and AIME, and their associated CCGs are compiled into our dataset -- \textbf{KisMATH}. Our detailed empirical analysis with 15 open-weight LLMs shows that (i) reasoning nodes in the CCG are mediators for the final answer, a condition necessary for reasoning; and (ii) LLMs emphasise reasoning paths given by the CCG, indicating that models internally realise structures akin to our graphs. KisMATH enables controlled, graph-aligned interventions and opens up avenues for further investigation into the role of chain-of-thought in LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11405v1">DCR: Quantifying Data Contamination in LLMs Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has heightened concerns about benchmark data contamination (BDC), where models inadvertently memorize evaluation data, inflating performance metrics and undermining genuine generalization assessment. This paper introduces the Data Contamination Risk (DCR) framework, a lightweight, interpretable pipeline designed to detect and quantify BDC across four granular levels: semantic, informational, data, and label. By synthesizing contamination scores via a fuzzy inference system, DCR produces a unified DCR Factor that adjusts raw accuracy to reflect contamination-aware performance. Validated on 9 LLMs (0.5B-72B) across sentiment analysis, fake news detection, and arithmetic reasoning tasks, the DCR framework reliably diagnoses contamination severity and with accuracy adjusted using the DCR Factor to within 4% average error across the three benchmarks compared to the uncontaminated baseline. Emphasizing computational efficiency and transparency, DCR provides a practical tool for integrating contamination assessment into routine evaluations, fostering fairer comparisons and enhancing the credibility of LLM benchmarking practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11344v1">Guiding LLM Decision-Making with Fairness Reward Models</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used to support high-stakes decisions, potentially influencing who is granted bail or receives a loan. Naive chain-of-thought sampling can improve average decision accuracy, but has also been shown to amplify unfair bias. To address this challenge and enable the trustworthy use of reasoning models in high-stakes decision-making, we propose a framework for training a generalizable Fairness Reward Model (FRM). Our model assigns a fairness score to LLM reasoning, enabling the system to down-weight biased trajectories and favor equitable ones when aggregating decisions across reasoning chains. We show that a single Fairness Reward Model, trained on weakly supervised, LLM-annotated examples of biased versus unbiased reasoning, transfers across tasks, domains, and model families without additional fine-tuning. Applied to real-world decision-making tasks including recidivism prediction and social media moderation, we show that our approach consistently improves fairness while matching, or even surpassing, baseline accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.06313v2">ETT: Expanding the Long Context Understanding Capability of LLMs at Test-Time</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Transformer-based Language Models' computation and memory overhead increase quadratically as a function of sequence length. The quadratic cost poses challenges when employing LLMs for processing long sequences. In this work, we introduce \ourmodelacronym~(Extend at Test-Time), method for extending the context length of short context Transformer-based LLMs, with constant memory requirement and linear computation overhead. ETT enable the extension of the context length at test-time by efficient fine-tuning the model's parameters on the input context, chunked into overlapping small subsequences. We evaluate ETT on LongBench by extending the context length of GPT-Large and Phi-2 up to 32 times, increasing from 1k to 32k tokens. This results in up to a 30 percent improvement in the model's accuracy. We also study how context can be stored in LLM's weights effectively and efficiently. Through a detailed ablation study, we examine which Transformer modules are most beneficial to fine-tune at test-time. Interestingly, we find that fine-tuning the second layer of the FFNs is more effective than full fine-tuning, leading to a further improvement in the models' accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23644v2">QLPro: Automated Code Vulnerability Discovery via LLM and Static Code Analysis Integration</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 The experimental data in the experimental section needs to be improved, and there are some errors
    </div>
    <details class="paper-abstract">
      We introduce QLPro, a vulnerability detection framework that systematically integrates LLMs and static analysis tools to enable comprehensive vulnerability detection across entire open-source projects.We constructed a new dataset, JavaTest, comprising 10 open-source projects from GitHub with 62 confirmed vulnerabilities. CodeQL, a state-of-the-art static analysis tool, detected only 24 of these vulnerabilities while QLPro detected 41. Furthermore, QLPro discovered 6 previously unknown vulnerabilities, 2 of which have been confirmed as 0-days.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.00166v3">EEG Emotion Copilot: Optimizing Lightweight LLMs for Emotional EEG Interpretation with Assisted Medical Record Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 17 pages, 16 figures, 5 tables
    </div>
    <details class="paper-abstract">
      In the fields of affective computing (AC) and brain-machine interface (BMI), the analysis of physiological and behavioral signals to discern individual emotional states has emerged as a critical research frontier. While deep learning-based approaches have made notable strides in EEG emotion recognition, particularly in feature extraction and pattern recognition, significant challenges persist in achieving end-to-end emotion computation, including real-time processing, individual adaptation, and seamless user interaction. This paper presents the EEG Emotion Copilot, a system optimizing a lightweight large language model (LLM) with 0.5B parameters operating in a local setting, which first recognizes emotional states directly from EEG signals, subsequently generates personalized diagnostic and treatment suggestions, and finally supports the automation of assisted electronic medical records. Specifically, we demonstrate the critical techniques in the novel data structure of prompt, model pruning and fine-tuning training, and deployment strategies aiming at improving real-time performance and computational efficiency. Extensive experiments show that our optimized lightweight LLM-based copilot achieves an enhanced intuitive interface for participant interaction, superior accuracy of emotion recognition and assisted electronic medical records generation, in comparison to such models with similar scale parameters or large-scale parameters such as 1.5B, 1.8B, 3B and 7B. In summary, through these efforts, the proposed copilot is expected to advance the application of AC in the medical domain, offering innovative solution to mental health monitoring. The codes will be released at https://github.com/NZWANG/EEG_Emotion_Copilot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03106v3">Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 49 pages, updated with new experimental results
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language models (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided self-refinements simultaneously while maintaining exploration. Additionally, we employ a shaping function to amplify learning from correct, especially unfamiliar, refinements and penalize incorrect ones. Extensive experiments with Qwen2.5-7B-Base, Qwen2.5-Math-7B-Base, and Qwen3-8B demonstrate that Critique-GRPO consistently outperforms supervised learning and RL-based fine-tuning methods across eight challenging mathematical, STEM, and general reasoning tasks, improving average pass@1 scores by approximately 4.4% and 3.8% on Qwen2.5-7B-Base and Qwen3-8B, respectively. Notably, Critique-GRPO enables effective self-improvement through self-critiquing and weak-to-strong generalization, achieving consistent gains over GRPO, such as 16.7% and 10.0% pass@1 improvements on AIME 2024, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02962v3">RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, while they remain prone to generating hallucinated or outdated responses due to their static internal knowledge. Recent advancements in Retrieval-Augmented Generation (RAG) methods have explored enhancing models' search and reasoning capabilities through reinforcement learning (RL). Although these methods demonstrate promising results, they face challenges in training stability and encounter issues such as substantial inference time and restricted capabilities due to the single-query mode. In this paper, we propose RAG-R1, a novel training framework designed to enable LLMs to adaptively leverage internal and external knowledge during the reasoning process. We further expand the generation and retrieval processes within the framework from single-query mode to multi-query parallelism, aimed at reducing inference time and enhancing the model's capabilities. Extensive experiments on seven question-answering benchmarks demonstrate that our method outperforms the strongest baseline by up to 13.2% and decreases inference time by 11.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00838v2">Stylometry recognizes human and LLM-generated texts in short samples</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      The paper explores stylometry as a method to distinguish between texts created by Large Language Models (LLMs) and humans, addressing issues of model attribution, intellectual property, and ethical AI use. Stylometry has been used extensively to characterise the style and attribute authorship of texts. By applying it to LLM-generated texts, we identify their emergent writing patterns. The paper involves creating a benchmark dataset based on Wikipedia, with (a) human-written term summaries, (b) texts generated purely by LLMs (GPT-3.5/4, LLaMa 2/3, Orca, and Falcon), (c) processed through multiple text summarisation methods (T5, BART, Gensim, and Sumy), and (d) rephrasing methods (Dipper, T5). The 10-sentence long texts were classified by tree-based models (decision trees and LightGBM) using human-designed (StyloMetrix) and n-gram-based (our own pipeline) stylometric features that encode lexical, grammatical, syntactic, and punctuation patterns. The cross-validated results reached a performance of up to .87 Matthews correlation coefficient in the multiclass scenario with 7 classes, and accuracy between .79 and 1. in binary classification, with the particular example of Wikipedia and GPT-4 reaching up to .98 accuracy on a balanced dataset. Shapley Additive Explanations pinpointed features characteristic of the encyclopaedic text type, individual overused words, as well as a greater grammatical standardisation of LLMs with respect to human-written texts. These results show -- crucially, in the context of the increasingly sophisticated LLMs -- that it is possible to distinguish machine- from human-generated texts at least for a well-defined text type.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11210v1">Role-Playing LLM-Based Multi-Agent Support Framework for Detecting and Addressing Family Communication Bias</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Well-being in family settings involves subtle psychological dynamics that conventional metrics often overlook. In particular, unconscious parental expectations, termed ideal parent bias, can suppress children's emotional expression and autonomy. This suppression, referred to as suppressed emotion, often stems from well-meaning but value-driven communication, which is difficult to detect or address from outside the family. Focusing on these latent dynamics, this study explores Large Language Model (LLM)-based support for psychologically safe family communication. We constructed a Japanese parent-child dialogue corpus of 30 scenarios, each annotated with metadata on ideal parent bias and suppressed emotion. Based on this corpus, we developed a Role-Playing LLM-based multi-agent dialogue support framework that analyzes dialogue and generates feedback. Specialized agents detect suppressed emotion, describe implicit ideal parent bias in parental speech, and infer contextual attributes such as the child's age and background. A meta-agent compiles these outputs into a structured report, which is then passed to five selected expert agents. These agents collaboratively generate empathetic and actionable feedback through a structured four-step discussion process. Experiments show that the system can detect categories of suppressed emotion with moderate accuracy and produce feedback rated highly in empathy and practicality. Moreover, simulated follow-up dialogues incorporating this feedback exhibited signs of improved emotional expression and mutual understanding, suggesting the framework's potential in supporting positive transformation in family interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10157v3">SocioVerse: A World Model for Social Simulation Powered by LLM Agents and A Pool of 10 Million Real-World Users</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Social simulation is transforming traditional social science research by modeling human behavior through interactions between virtual individuals and their environments. With recent advances in large language models (LLMs), this approach has shown growing potential in capturing individual differences and predicting group behaviors. However, existing methods face alignment challenges related to the environment, target users, interaction mechanisms, and behavioral patterns. To this end, we introduce SocioVerse, an LLM-agent-driven world model for social simulation. Our framework features four powerful alignment components and a user pool of 10 million real individuals. To validate its effectiveness, we conducted large-scale simulation experiments across three distinct domains: politics, news, and economics. Results demonstrate that SocioVerse can reflect large-scale population dynamics while ensuring diversity, credibility, and representativeness through standardized procedures and minimal manual adjustments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11198v1">Temperature and Persona Shape LLM Agent Consensus With Minimal Accuracy Gains in Qualitative Coding</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Manuscript submitted for review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) enable new possibilities for qualitative research at scale, including coding and data annotation. While multi-agent systems (MAS) can emulate human coding workflows, their benefits over single-agent coding remain poorly understood. We conducted an experimental study of how agent persona and temperature shape consensus-building and coding accuracy of dialog segments based on a codebook with 8 codes. Our open-source MAS mirrors deductive human coding through structured agent discussion and consensus arbitration. Using six open-source LLMs (with 3 to 32 billion parameters) and 18 experimental configurations, we analyze over 77,000 coding decisions against a gold-standard dataset of human-annotated transcripts from online math tutoring sessions. Temperature significantly impacted whether and when consensus was reached across all six LLMs. MAS with multiple personas (including neutral, assertive, or empathetic), significantly delayed consensus in four out of six LLMs compared to uniform personas. In three of those LLMs, higher temperatures significantly diminished the effects of multiple personas on consensus. However, neither temperature nor persona pairing lead to robust improvements in coding accuracy. Single agents matched or outperformed MAS consensus in most conditions. Only one model (OpenHermesV2:7B) and code category showed above-chance gains from MAS deliberation when temperature was 0.5 or lower and especially when the agents included at least one assertive persona. Qualitative analysis of MAS collaboration for these configurations suggests that MAS may nonetheless aid in narrowing ambiguous code applications that could improve codebooks and human-AI coding. We contribute new insight into the limits of LLM-based qualitative methods, challenging the notion that diverse MAS personas lead to better outcomes. We open-source our MAS and experimentation code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08054v2">FalseReject: A Resource for Improving Contextual Safety and Mitigating Over-Refusals in LLMs via Structured Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 Accepted at COLM 2025
    </div>
    <details class="paper-abstract">
      Safety alignment approaches in large language models (LLMs) often lead to the over-refusal of benign queries, significantly diminishing their utility in sensitive scenarios. To address this challenge, we introduce FalseReject, a comprehensive resource containing 16k seemingly toxic queries accompanied by structured responses across 44 safety-related categories. We propose a graph-informed adversarial multi-agent interaction framework to generate diverse and complex prompts, while structuring responses with explicit reasoning to aid models in accurately distinguishing safe from unsafe contexts. FalseReject includes training datasets tailored for both standard instruction-tuned models and reasoning-oriented models, as well as a human-annotated benchmark test set. Our extensive benchmarking on 29 state-of-the-art (SOTA) LLMs reveals persistent over-refusal challenges. Empirical results demonstrate that supervised finetuning with FalseReject substantially reduces unnecessary refusals without compromising overall safety or general language capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11128v1">What Should LLMs Forget? Quantifying Personal Data in LLMs for Right-to-Be-Forgotten Requests</a></div>
    <div class="paper-meta">
      📅 2025-07-15
      | 💬 16 pages, 3 figures. Accepted at the 7th Workshop on eXplainable Knowledge Discovery in Data Mining (XKDD 2025), ECML PKDD 2025, Porto, Portugal
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can memorize and reveal personal information, raising concerns regarding compliance with the EU's GDPR, particularly the Right to Be Forgotten (RTBF). Existing machine unlearning methods assume the data to forget is already known but do not address how to identify which individual-fact associations are stored in the model. Privacy auditing techniques typically operate at the population level or target a small set of identifiers, limiting applicability to individual-level data inquiries. We introduce WikiMem, a dataset of over 5,000 natural language canaries covering 243 human-related properties from Wikidata, and a model-agnostic metric to quantify human-fact associations in LLMs. Our approach ranks ground-truth values against counterfactuals using calibrated negative log-likelihood across paraphrased prompts. We evaluate 200 individuals across 15 LLMs (410M-70B parameters), showing that memorization correlates with subject web presence and model scale. We provide a foundation for identifying memorized personal data in LLMs at the individual level, enabling the dynamic construction of forget sets for machine unlearning and RTBF requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.21033v2">Plancraft: an evaluation dataset for planning with LLM agents</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      We present Plancraft, a multi-modal evaluation dataset for LLM agents. Plancraft has both a text-only and multi-modal interface, based on the Minecraft crafting GUI. We include the Minecraft Wiki to evaluate tool use and Retrieval Augmented Generation (RAG), as well as a handcrafted planner and Oracle Retriever, to ablate the different components of a modern agent architecture. To evaluate decision-making, Plancraft also includes a subset of examples that are intentionally unsolvable, providing a realistic challenge that requires the agent not only to complete tasks but also to decide whether they are solvable at all. We benchmark both open-source and closed-source LLMs and compare their performance and efficiency to a handcrafted planner. Overall, we find that LLMs and VLMs struggle with the planning problems that Plancraft introduces, and offer suggestions on how to improve their capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11112v1">Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-15
    </div>
    <details class="paper-abstract">
      Recent studies have shown that Large Language Models (LLMs) are vulnerable to data poisoning attacks, where malicious training examples embed hidden behaviours triggered by specific input patterns. However, most existing works assume a phrase and focus on the attack's effectiveness, offering limited understanding of trigger mechanisms and how multiple triggers interact within the model. In this paper, we present a framework for studying poisoning in LLMs. We show that multiple distinct backdoor triggers can coexist within a single model without interfering with each other, enabling adversaries to embed several triggers concurrently. Using multiple triggers with high embedding similarity, we demonstrate that poisoned triggers can achieve robust activation even when tokens are substituted or separated by long token spans. Our findings expose a broader and more persistent vulnerability surface in LLMs. To mitigate this threat, we propose a post hoc recovery method that selectively retrains specific model components based on a layer-wise weight difference analysis. Our method effectively removes the trigger behaviour with minimal parameter updates, presenting a practical and efficient defence against multi-trigger poisoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10540v1">Fusing LLM Capabilities with Routing Data</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has created a vibrant ecosystem of diverse architectures, each with unique strengths due to differences in design, training data, and objectives. However, most applications still rely on a single backend model, limiting coverage of capabilities and leading to inefficiencies in performance and token cost when tackling complex tasks. We highlight an underexploited opportunity: LLM routing data, produced when hosting platforms route diverse queries to different models, which can reveal comparative strengths across tasks. To address this, we propose FusionBench, a comprehensive routing benchmark covering 14 tasks across five domains with 20 open-source LLMs (8B to 671B parameters), capturing 103M tokens and summarizing reusable thought templates from top models. Building on this, we introduce FusionFactory, a systematic fusion framework with three levels: (1) query-level fusion, tailoring routers for each query using both direct responses and reasoning-augmented outputs; (2) thought-level fusion, leveraging abstract templates derived from top-performing LLMs' answers to similar queries; and (3) model-level fusion, transferring capabilities between models via distillation, using top responses or highest judge scores as training data. Experiments show FusionFactory consistently outperforms the best individual LLM across all 14 benchmarks, with optimal fusion configurations varying by benchmark, demonstrating the value of systematic LLM fusion in harnessing complementary strengths and improving overall performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10535v1">CodeJudgeBench: Benchmarking LLM-as-a-Judge for Coding Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Dataset is available at https://huggingface.co/datasets/mattymchen/codejudgebench
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced the state-of-the-art in various coding tasks. Beyond directly answering user queries, LLMs can also serve as judges, assessing and comparing the quality of responses generated by other models. Such an evaluation capability is crucial both for benchmarking different LLMs and for improving response quality through response ranking. However, despite the growing adoption of the LLM-as-a-Judge paradigm, its effectiveness in coding scenarios remains underexplored due to the absence of dedicated benchmarks. To address this gap, we introduce CodeJudgeBench, a benchmark explicitly designed to evaluate the performance of LLM-as-a-Judge models across three critical coding tasks: code generation, code repair, and unit test generation. Through comprehensive benchmarking of 26 LLM-as-a-Judge models, we find that recent thinking models significantly outperform non-thinking models on our carefully designed code judging tasks. Notably, even relatively small thinking models, such as Qwen3-8B, can outperform specially trained LLM-as-a-Judge models up to 70B in size. Nevertheless, all models still exhibit significant randomness in their judgment of coding tasks. For pairwise judging tasks, simply changing the order in which responses are presented can substantially impact accuracy. In addition, when judging code and unit tests written by different LLMs, LLM-as-a-Judge models also show variance in performance. This sensitivity raises concerns about the reliability and consistency of LLM-as-a-Judge in coding scenarios. Lastly, we study optimal prompting strategies for LLM-as-a-Judge. We find that using pair-wise comparison outperforms scalar point-wise judging. Furthermore, retaining comments and reasoning in the full, unprocessed LLM response leads to improved judge performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10445v1">Referential ambiguity and clarification requests: comparing human and LLM behaviour</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      In this work we examine LLMs' ability to ask clarification questions in task-oriented dialogues that follow the asynchronous instruction-giver/instruction-follower format. We present a new corpus that combines two existing annotations of the Minecraft Dialogue Corpus -- one for reference and ambiguity in reference, and one for SDRT including clarifications -- into a single common format providing the necessary information to experiment with clarifications and their relation to ambiguity. With this corpus we compare LLM actions with original human-generated clarification questions, examining how both humans and LLMs act in the case of ambiguity. We find that there is only a weak link between ambiguity and humans producing clarification questions in these dialogues, and low correlation between humans and LLMs. Humans hardly ever produce clarification questions for referential ambiguity, but often do so for task-based uncertainty. Conversely, LLMs produce more clarification questions for referential ambiguity, but less so for task uncertainty. We question if LLMs' ability to ask clarification questions is predicated on their recent ability to simulate reasoning, and test this with different reasoning approaches, finding that reasoning does appear to increase question frequency and relevancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10427v1">Towards Emotion Co-regulation with LLM-powered Socially Assistive Robots: Integrating LLM Prompts and Robotic Behaviors to Support Parent-Neurodivergent Child Dyads</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Submission for the IROS 2025 conference
    </div>
    <details class="paper-abstract">
      Socially Assistive Robotics (SAR) has shown promise in supporting emotion regulation for neurodivergent children. Recently, there has been increasing interest in leveraging advanced technologies to assist parents in co-regulating emotions with their children. However, limited research has explored the integration of large language models (LLMs) with SAR to facilitate emotion co-regulation between parents and children with neurodevelopmental disorders. To address this gap, we developed an LLM-powered social robot by deploying a speech communication module on the MiRo-E robotic platform. This supervised autonomous system integrates LLM prompts and robotic behaviors to deliver tailored interventions for both parents and neurodivergent children. Pilot tests were conducted with two parent-child dyads, followed by a qualitative analysis. The findings reveal MiRo-E's positive impacts on interaction dynamics and its potential to facilitate emotion regulation, along with identified design and technical challenges. Based on these insights, we provide design implications to advance the future development of LLM-powered SAR for mental health applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10392v1">Zorse: Optimizing LLM Training Efficiency on Heterogeneous GPU Clusters</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) require vast amounts of GPU compute to train, but limited availability and high costs of GPUs make homogeneous clusters impractical for many organizations. Instead, assembling heterogeneous clusters by pooling together GPUs of different generations allows them to achieve higher aggregate compute and make use of all available GPUs. However, training on heterogeneous clusters presents several challenges, including load balancing across GPUs, optimizing memory usage to accommodate varying memory capacities, and ensuring communication-efficient training over diverse network interconnects potentially spanning multiple datacenters. In this paper, we make the case that efficient training on heterogeneous clusters requires (1) the integration of pipeline parallelism and data parallelism in a manner that is both communication- and memory-efficient, and (2) a more adaptable configuration of pipeline and data parallelism, which includes the capability to flexibly partition GPUs into asymmetric pipeline parallel stages and to incorporate heterogeneous GPUs within the same data parallelism group. We propose Zorse, the first system to unify all these capabilities while incorporating a planner that automatically configures training strategies for a given workload. Our evaluation shows that Zorse significantly outperforms state-of-the-art systems in heterogeneous training scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11168v3">Bypassing LLM Guardrails: An Empirical Analysis of Evasion Attacks against Prompt Injection and Jailbreak Detection Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 14 pages, 5 figures, 11 tables. To be published in LLMSec 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10382v1">Leveraging RAG-LLMs for Urban Mobility Simulation and Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      With the rise of smart mobility and shared e-mobility services, numerous advanced technologies have been applied to this field. Cloud-based traffic simulation solutions have flourished, offering increasingly realistic representations of the evolving mobility landscape. LLMs have emerged as pioneering tools, providing robust support for various applications, including intelligent decision-making, user interaction, and real-time traffic analysis. As user demand for e-mobility continues to grow, delivering comprehensive end-to-end solutions has become crucial. In this paper, we present a cloud-based, LLM-powered shared e-mobility platform, integrated with a mobile application for personalized route recommendations. The optimization module is evaluated based on travel time and cost across different traffic scenarios. Additionally, the LLM-powered RAG framework is evaluated at the schema level for different users, using various evaluation methods. Schema-level RAG with XiYanSQL achieves an average execution accuracy of 0.81 on system operator queries and 0.98 on user queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06238v2">EVOLvE: Evaluating and Optimizing LLMs For In-Context Exploration</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 28 pages. Published at ICML 2025
    </div>
    <details class="paper-abstract">
      Despite their success in many domains, large language models (LLMs) remain under-studied in scenarios requiring optimal decision-making under uncertainty. This is crucial as many real-world applications, ranging from personalized recommendations to healthcare interventions, demand that LLMs not only predict but also actively learn to make optimal decisions through exploration. In this work, we measure LLMs' (in)ability to make optimal decisions in bandits, a state-less reinforcement learning setting relevant to many applications. We develop a comprehensive suite of environments, including both context-free and contextual bandits with varying task difficulties, to benchmark LLMs' performance. Motivated by the existence of optimal exploration algorithms, we propose efficient ways to integrate this algorithmic knowledge into LLMs: by providing explicit algorithm-guided support during inference; and through algorithm distillation via in-context demonstrations and fine-tuning, using synthetic data generated from these algorithms. Impressively, these techniques allow us to achieve superior exploration performance with smaller models, surpassing larger models on various tasks. We conducted an extensive ablation study to shed light on various factors, such as task difficulty and data representation, that influence the efficiency of LLM exploration. Additionally, we conduct a rigorous analysis of the LLM's exploration efficiency using the concept of regret, linking its ability to explore to the model size and underlying algorithm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02620v2">FlowSpec: Continuous Pipelined Speculative Decoding for Efficient Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 16 pages, and the last 3 are appendix
    </div>
    <details class="paper-abstract">
      Distributed inference serves as a promising approach to enabling the inference of large language models (LLMs) at the network edge. It distributes the inference process to multiple devices to ensure that the LLMs can fit into the device memory. Recent pipeline-based approaches have the potential to parallelize communication and computation, which helps reduce inference latency. However, the benefit diminishes when the inference request at the network edge is sparse, where pipeline is typically at low utilization. To enable efficient distributed LLM inference at the edge, we propose \textbf{FlowSpec}, a pipeline-parallel tree-based speculative decoding framework. FlowSpec incorporates three key mechanisms to improve decoding efficiency: 1) score-based step-wise verification prioritizes more important draft tokens to bring earlier accpeted tokens; 2) efficient draft management to prune invalid tokens while maintaining correct causal relationship during verification; 3) dynamic draft expansion strategies to supply high-quality speculative inputs. These techniques work in concert to enhance both pipeline utilization and speculative efficiency. We evaluate FlowSpec on a real-world testbed with other baselines. Experimental results demonstrate that our proposed framework significantly improves inference speed across diverse models and configurations, achieving speedup ratios 1.28$\times$-1.79$\times$ compared to baselines. Our code is publicly available at \href{https://github.com/Leosang-lx/FlowSpec#}{https://github.com/Leosang-lx/FlowSpec\#}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10281v1">Toward Real-World Table Agents: Capabilities, Workflows, and Design Principles for LLM-based Table Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Tables are fundamental in domains such as finance, healthcare, and public administration, yet real-world table tasks often involve noise, structural heterogeneity, and semantic complexity--issues underexplored in existing research that primarily targets clean academic datasets. This survey focuses on LLM-based Table Agents, which aim to automate table-centric workflows by integrating preprocessing, reasoning, and domain adaptation. We define five core competencies--C1: Table Structure Understanding, C2: Table and Query Semantic Understanding, C3: Table Retrieval and Compression, C4: Executable Reasoning with Traceability, and C5: Cross-Domain Generalization--to analyze and compare current approaches. In addition, a detailed examination of the Text-to-SQL Agent reveals a performance gap between academic benchmarks and real-world scenarios, especially for open-source models. Finally, we provide actionable insights to improve the robustness, generalization, and efficiency of LLM-based Table Agents in practical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10200v1">Natural Language-based Assessment of L2 Oral Proficiency using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted for the 10th Workshop on Speech and Language Technology in Education (SLaTE 2025)
    </div>
    <details class="paper-abstract">
      Natural language-based assessment (NLA) is an approach to second language assessment that uses instructions - expressed in the form of can-do descriptors - originally intended for human examiners, aiming to determine whether large language models (LLMs) can interpret and apply them in ways comparable to human assessment. In this work, we explore the use of such descriptors with an open-source LLM, Qwen 2.5 72B, to assess responses from the publicly available S&I Corpus in a zero-shot setting. Our results show that this approach - relying solely on textual information - achieves competitive performance: while it does not outperform state-of-the-art speech LLMs fine-tuned for the task, it surpasses a BERT-based model trained specifically for this purpose. NLA proves particularly effective in mismatched task settings, is generalisable to other data types and languages, and offers greater interpretability, as it is grounded in clearly explainable, widely applicable language descriptors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10177v1">Abusive text transformation using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Although Large Language Models (LLMs) have demonstrated significant advancements in natural language processing tasks, their effectiveness in the classification and transformation of abusive text into non-abusive versions remains an area for exploration. In this study, we aim to use LLMs to transform abusive text (tweets and reviews) featuring hate speech and swear words into non-abusive text, while retaining the intent of the text. We evaluate the performance of two state-of-the-art LLMs, such as Gemini, GPT-4o, DeekSeek and Groq, on their ability to identify abusive text. We them to transform and obtain a text that is clean from abusive and inappropriate content but maintains a similar level of sentiment and semantics, i.e. the transformed text needs to maintain its message. Afterwards, we evaluate the raw and transformed datasets with sentiment analysis and semantic analysis. Our results show Groq provides vastly different results when compared with other LLMs. We have identified similarities between GPT-4o and DeepSeek-V3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14403v3">Unearthing Gems from Stones: Policy Optimization with Negative Sample Augmentation for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Recent advances in reasoning language models have witnessed a paradigm shift from short to long CoT pattern. Given the substantial computational cost of rollouts in long CoT models, maximizing the utility of fixed training datasets becomes crucial. Our analysis reveals that negative responses contain valuable components such as self-reflection and error-correction steps, yet primary existing methods either completely discard negative samples (RFT) or apply equal penalization across all tokens (RL), failing to leverage these potential learning signals. In light of this, we propose Behavior Constrained Policy Gradient with Negative Sample Augmentation (BCPG-NSA), a fine-grained offline RL framework that encompasses three stages: 1) sample segmentation, 2) consensus-based step correctness assessment combining LLM and PRM judgers, and 3) policy optimization with NSA designed to effectively mine positive steps within negative samples. Experimental results show that BCPG-NSA outperforms baselines on several challenging math/coding reasoning benchmarks using the same training dataset, achieving improved sample efficiency and demonstrating robustness and scalability when extended to multiple iterations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10155v1">Task-Based Flexible Feature Distillation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Knowledge Distillation (KD) in general and feature distillation in particular are promising techniques for reducing the high computational demand of large language models (LLMs). However, traditional feature KD methods typically assume that the teacher and the student share the same hidden size, limiting the flexibility of the student's architecture. A common solution to this problem involves training a linear projector to align their feature spaces, but this introduces additional parameters that must be learned from scratch and often degrades performance on downstream tasks, especially in generative settings. To address this issue, in this work, we propose a novel task-based feature distillation method that enables knowledge transfer between teacher and student models with different hidden layer dimensions, without introducing any new parameters. Leveraging the insight that only a subset of LLM components contribute significantly to a specific downstream task, our approach identifies the most task-relevant hidden units in the teacher and directly distills their activations to the student. Our method is flexible and easily integrates with other distillation frameworks. Empirical results show consistent improvements over prior approaches across diverse tasks, including classification, instruction-following, and summarization, achieving up to a 3\% performance gain over the linear projection baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10150v1">Past-Future Scheduler for LLM Serving under SLA Guarantees</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 Accepted to ASPLOS 2025
    </div>
    <details class="paper-abstract">
      The exploration and application of Large Language Models (LLMs) is thriving. To reduce deployment costs, continuous batching has become an essential feature in current service frameworks. The effectiveness of continuous batching relies on an accurate estimate of the memory requirements of requests. However, due to the diversity in request output lengths, existing frameworks tend to adopt aggressive or conservative schedulers, which often result in significant overestimation or underestimation of memory consumption. Consequently, they suffer from harmful request evictions or prolonged queuing times, failing to achieve satisfactory throughput under strict Service Level Agreement (SLA) guarantees (a.k.a. goodput), across various LLM application scenarios with differing input-output length distributions. To address this issue, we propose a novel Past-Future scheduler that precisely estimates the peak memory resources required by the running batch via considering the historical distribution of request output lengths and calculating memory occupancy at each future time point. It adapts to applications with all types of input-output length distributions, balancing the trade-off between request queuing and harmful evictions, thereby consistently achieving better goodput. Furthermore, to validate the effectiveness of the proposed scheduler, we developed a high-performance LLM serving framework, LightLLM, that implements the Past-Future scheduler. Compared to existing aggressive or conservative schedulers, LightLLM demonstrates superior goodput, achieving up to 2-3$\times$ higher goodput than other schedulers under heavy loads. LightLLM is open source to boost the research in such direction (https://github.com/ModelTC/lightllm).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10134v1">FRSICL: LLM-Enabled In-Context Learning Flight Resource Allocation for Fresh Data Collection in UAV-Assisted Wildfire Monitoring</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 8 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Unmanned Aerial Vehicles (UAVs) are vital for public safety, particularly in wildfire monitoring, where early detection minimizes environmental impact. In UAV-Assisted Wildfire Monitoring (UAWM) systems, joint optimization of sensor transmission scheduling and velocity is critical for minimizing Age of Information (AoI) from stale sensor data. Deep Reinforcement Learning (DRL) has been used for such optimization; however, its limitations such as low sampling efficiency, simulation-to-reality gaps, and complex training render it unsuitable for time-critical applications like wildfire monitoring. This paper introduces a new online Flight Resource Allocation scheme based on LLM-Enabled In-Context Learning (FRSICL) to jointly optimize the UAV's flight control and data collection schedule along the trajectory in real time, thereby asymptotically minimizing the average AoI across ground sensors. In contrast to DRL, FRSICL generates data collection schedules and controls velocity using natural language task descriptions and feedback from the environment, enabling dynamic decision-making without extensive retraining. Simulation results confirm the effectiveness of the proposed FRSICL compared to Proximal Policy Optimization (PPO) and Nearest-Neighbor baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10124v1">Could you be wrong: Debiasing LLMs using a metacognitive prompt for improving human decision making</a></div>
    <div class="paper-meta">
      📅 2025-07-14
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Identifying bias in LLMs is ongoing. Because they are still in development, what is true today may be false tomorrow. We therefore need general strategies for debiasing that will outlive current models. Strategies developed for debiasing human decision making offer one promising approach as they incorporate an LLM-style prompt intervention designed to bring latent knowledge into awareness during decision making. LLMs trained on vast amounts of information contain information about potential biases, counter-arguments, and contradictory evidence, but that information may only be brought to bear if prompted. Metacognitive prompts developed in the human decision making literature are designed to achieve this, and as I demonstrate here, they show promise with LLMs. The prompt I focus on here is "could you be wrong?" Following an LLM response, this prompt leads LLMs to produce additional information, including why they answered as they did, errors, biases, contradictory evidence, and alternatives, none of which were apparent in their initial response. Indeed, this metaknowledge often reveals that how LLMs and users interpret prompts are not aligned. Here I demonstrate this prompt using a set of questions taken from recent articles about LLM biases, including implicit discriminatory biases and failures of metacognition. "Could you be wrong" prompts the LLM to identify its own biases and produce cogent metacognitive reflection. I also present another example involving convincing but incomplete information, which is readily corrected by the metacognitive prompt. In sum, this work argues that human psychology offers a new avenue for prompt engineering, leveraging a long history of effective prompt-based improvements to human decision making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00200v2">Structuring Radiology Reports: Challenging LLMs with Lightweight Models</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Radiology reports are critical for clinical decision-making but often lack a standardized format, limiting both human interpretability and machine learning (ML) applications. While large language models (LLMs) have shown strong capabilities in reformatting clinical text, their high computational requirements, lack of transparency, and data privacy concerns hinder practical deployment. To address these challenges, we explore lightweight encoder-decoder models (<300M parameters)-specifically T5 and BERT2BERT-for structuring radiology reports from the MIMIC-CXR and CheXpert Plus datasets. We benchmark these models against eight open-source LLMs (1B-70B), adapted using prefix prompting, in-context learning (ICL), and low-rank adaptation (LoRA) finetuning. Our best-performing lightweight model outperforms all LLMs adapted using prompt-based techniques on a human-annotated test set. While some LoRA-finetuned LLMs achieve modest gains over the lightweight model on the Findings section (BLEU 6.4%, ROUGE-L 4.8%, BERTScore 3.6%, F1-RadGraph 1.1%, GREEN 3.6%, and F1-SRR-BERT 4.3%), these improvements come at the cost of substantially greater computational resources. For example, LLaMA-3-70B incurred more than 400 times the inference time, cost, and carbon emissions compared to the lightweight model. These results underscore the potential of lightweight, task-specific models as sustainable and privacy-preserving solutions for structuring clinical text in resource-constrained healthcare settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10069v1">ElasticMM: Efficient Multimodal LLMs Serving with Elastic Multimodal Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-07-14
    </div>
    <details class="paper-abstract">
      Multimodal large language models (MLLMs) extend LLMs to handle images, videos, and audio by incorporating feature extractors and projection modules. However, these additional components -- combined with complex inference pipelines and heterogeneous workloads -- introduce significant inference overhead. Therefore, efficiently serving MLLMs remains a major challenge. Current tightly coupled serving architectures struggle to distinguish between mixed request types or adapt parallelism strategies to different inference stages, leading to increased time-to-first-token (TTFT) latency and poor resource utilization. To address this, we propose Elastic Multimodal Parallelism (EMP), a new serving paradigm that elastically adapts to resource heterogeneity across request types and inference stages. Building upon EMP, we develop ElasticMM, an MLLM serving system that (1) separates requests into independent modality groups with dynamic resource allocation via a modality-aware load balancer; (2) decouples inference stages and enables parallelism adjustment and adaptive scaling via elastic partition scheduling; and (3) improves inference efficiency through unified multimodal prefix caching and non-blocking encoding. Experiments on diverse real-world datasets show that ElasticMM outperforms state-of-the-art (SOTA) serving systems, reducing TTFT by up to 4.2x and achieving 3.2-4.5x higher throughput while meeting service-level objectives (SLOs).
    </details>
</div>
