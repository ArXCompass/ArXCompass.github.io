# llm - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08825v1">Large Language Model Hacking: Quantifying the Hidden Risks of Using LLMs for Text Annotation</a></div>
    <div class="paper-meta">
      📅 2025-09-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly transforming social science research by enabling the automation of labor-intensive tasks like data annotation and text analysis. However, LLM outputs vary significantly depending on the implementation choices made by researchers (e.g., model selection, prompting strategy, or temperature settings). Such variation can introduce systematic biases and random errors, which propagate to downstream analyses and cause Type I, Type II, Type S, or Type M errors. We call this LLM hacking. We quantify the risk of LLM hacking by replicating 37 data annotation tasks from 21 published social science research studies with 18 different models. Analyzing 13 million LLM labels, we test 2,361 realistic hypotheses to measure how plausible researcher choices affect statistical conclusions. We find incorrect conclusions based on LLM-annotated data in approximately one in three hypotheses for state-of-the-art models, and in half the hypotheses for small language models. While our findings show that higher task performance and better general model capabilities reduce LLM hacking risk, even highly accurate models do not completely eliminate it. The risk of LLM hacking decreases as effect sizes increase, indicating the need for more rigorous verification of findings near significance thresholds. Our extensive analysis of LLM hacking mitigation techniques emphasizes the importance of human annotations in reducing false positive findings and improving model selection. Surprisingly, common regression estimator correction techniques are largely ineffective in reducing LLM hacking risk, as they heavily trade off Type I vs. Type II errors. Beyond accidental errors, we find that intentional LLM hacking is unacceptably simple. With few LLMs and just a handful of prompt paraphrases, anything can be presented as statistically significant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08824v1">Building High-Quality Datasets for Portuguese LLMs: From Common Crawl Snapshots to Industrial-Grade Corpora</a></div>
    <div class="paper-meta">
      📅 2025-09-10
    </div>
    <details class="paper-abstract">
      The performance of large language models (LLMs) is deeply influenced by the quality and composition of their training data. While much of the existing work has centered on English, there remains a gap in understanding how to construct effective training corpora for other languages. We explore scalable methods for building web-based corpora for LLMs. We apply them to build a new 120B token corpus in Portuguese that achieves competitive results to an industrial-grade corpus. Using a continual pretraining setup, we study how different data selection and preprocessing strategies affect LLM performance when transitioning a model originally trained in English to another language. Our findings demonstrate the value of language-specific filtering pipelines, including classifiers for education, science, technology, engineering, and mathematics (STEM), as well as toxic content. We show that adapting a model to the target language leads to performance improvements, reinforcing the importance of high-quality, language-specific data. While our case study focuses on Portuguese, our methods are applicable to other languages, offering insights for multilingual LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15474v2">Subjective Behaviors and Preferences in LLM: Language of Browsing</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      A Large Language Model (LLM) offers versatility across domains and tasks, purportedly benefiting users with a wide variety of behaviors and preferences. We question this perception about an LLM when users have inherently subjective behaviors and preferences, as seen in their ubiquitous and idiosyncratic browsing of websites or apps. The sequential behavior logs of pages, thus generated, form something akin to each user's self-constructed "language", albeit without the structure and grammar imbued in natural languages. We ask: (i) Can a small LM represent the "language of browsing" better than a large LM? (ii) Can an LM with a single set of parameters (or, single LM) adequately capture myriad users' heterogeneous, subjective behaviors and preferences? (iii) Can a single LM with high average performance, yield low variance in performance to make alignment good at user level? We introduce clusterwise LM training, HeTLM (Heterogeneity aware Training of Language Model), appropriate for subjective behaviors. We find that (i) a small LM trained using a page-level tokenizer outperforms large pretrained or finetuned LMs; (ii) HeTLM with heterogeneous cluster specific set of parameters outperforms a single LM of the same family, controlling for the number of parameters; and (iii) a higher mean and a lower variance in generation ensues, implying improved alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08809v1">Evaluating LLMs Without Oracle Feedback: Agentic Annotation Evaluation Through Unsupervised Consistency Signals</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 11 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), when paired with prompt-based tasks, have significantly reduced data annotation costs and reliance on human annotators. However, evaluating the quality of their annotations remains challenging in dynamic, unsupervised environments where oracle feedback is scarce and conventional methods fail. To address this challenge, we propose a novel agentic annotation paradigm, where a student model collaborates with a noisy teacher (the LLM) to assess and refine annotation quality without relying on oracle feedback. The student model, acting as an unsupervised feedback mechanism, employs a user preference-based majority voting strategy to evaluate the consistency of the LLM outputs. To systematically measure the reliability of LLM-generated annotations, we introduce the Consistent and Inconsistent (CAI) Ratio, a novel unsupervised evaluation metric. The CAI Ratio not only quantifies the annotation quality of the noisy teacher under limited user preferences but also plays a critical role in model selection, enabling the identification of robust LLMs in dynamic, unsupervised environments. Applied to ten open-domain NLP datasets across four LLMs, the CAI Ratio demonstrates a strong positive correlation with LLM accuracy, establishing it as an essential tool for unsupervised evaluation and model selection in real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00074v2">Whose Name Comes Up? Auditing LLM-Based Scholar Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 40 pages: 10 main (incl. 9 figures), 3 references, and 27 appendix. Paper under-review
    </div>
    <details class="paper-abstract">
      This paper evaluates the performance of six open-weight LLMs (llama3-8b, llama3.1-8b, gemma2-9b, mixtral-8x7b, llama3-70b, llama3.1-70b) in recommending experts in physics across five tasks: top-k experts by field, influential scientists by discipline, epoch, seniority, and scholar counterparts. The evaluation examines consistency, factuality, and biases related to gender, ethnicity, academic popularity, and scholar similarity. Using ground-truth data from the American Physical Society and OpenAlex, we establish scholarly benchmarks by comparing model outputs to real-world academic records. Our analysis reveals inconsistencies and biases across all models. mixtral-8x7b produces the most stable outputs, while llama3.1-70b shows the highest variability. Many models exhibit duplication, and some, particularly gemma2-9b and llama3.1-8b, struggle with formatting errors. LLMs generally recommend real scientists, but accuracy drops in field-, epoch-, and seniority-specific queries, consistently favoring senior scholars. Representation biases persist, replicating gender imbalances (reflecting male predominance), under-representing Asian scientists, and over-representing White scholars. Despite some diversity in institutional and collaboration networks, models favor highly cited and productive scholars, reinforcing the rich-getricher effect while offering limited geographical representation. These findings highlight the need to improve LLMs for more reliable and equitable scholarly recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08755v1">AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 preprint, 39 pages, 16 figures. Project: https://AgentGym-RL.github.io/. Framework and Code: https://github.com/woooodyy/AgentGym, https://github.com/woooodyy/AgentGym-RL
    </div>
    <details class="paper-abstract">
      Developing autonomous LLM agents capable of making a series of intelligent decisions to solve complex, real-world tasks is a fast-evolving frontier. Like human cognitive development, agents are expected to acquire knowledge and skills through exploration and interaction with the environment. Despite advances, the community still lacks a unified, interactive reinforcement learning (RL) framework that can effectively train such agents from scratch -- without relying on supervised fine-tuning (SFT) -- across diverse and realistic environments. To bridge this gap, we introduce AgentGym-RL, a new framework to train LLM agents for multi-turn interactive decision-making through RL. The framework features a modular and decoupled architecture, ensuring high flexibility and extensibility. It encompasses a wide variety of real-world scenarios, and supports mainstream RL algorithms. Furthermore, we propose ScalingInter-RL, a training approach designed for exploration-exploitation balance and stable RL optimization. In early stages, it emphasizes exploitation by restricting the number of interactions, and gradually shifts towards exploration with larger horizons to encourage diverse problem-solving strategies. In this way, the agent develops more diverse behaviors and is less prone to collapse under long horizons. We perform extensive experiments to validate the stability and effectiveness of both the AgentGym-RL framework and the ScalingInter-RL approach. Our agents match or surpass commercial models on 27 tasks across diverse environments. We offer key insights and will open-source the complete AgentGym-RL framework -- including code and datasets -- to empower the research community in developing the next generation of intelligent agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02682v2">MPO: Boosting LLM Agents with Meta Plan Optimization</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have enabled LLM-based agents to successfully tackle interactive planning tasks. However, despite their successes, existing approaches often suffer from planning hallucinations and require retraining for each new agent. To address these challenges, we propose the Meta Plan Optimization (MPO) framework, , which enhances agent planning capabilities by directly incorporating explicit guidance. Unlike previous methods that rely on complex knowledge, which either require significant human effort or lack quality assurance, MPO leverages high-level general guidance through meta plans to assist agent planning and enables continuous optimization of the meta plans based on feedback from the agent's task execution. Our experiments conducted on two representative tasks demonstrate that MPO significantly outperforms existing baselines. Moreover, our analysis indicates that MPO provides a plug-and-play solution that enhances both task completion efficiency and generalization capabilities in previous unseen scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16044v3">Making REST APIs Agent-Ready: From OpenAPI to MCP Servers for Tool-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are evolving from passive text generators into active agents that invoke external tools. To support this shift, scalable protocols for tool integration are essential. The Model Context Protocol (MCP), introduced by Anthropic in 2024, offers a schema-driven standard for dynamic tool discovery and invocation. Yet, building MCP servers remains manual and repetitive, requiring developers to write glue code, handle authentication, and configure schemas by hand-replicating much of the integration effort MCP aims to eliminate. This paper investigates whether MCP server construction can be meaningfully automated. We begin by analyzing adoption trends: among 22,000+ MCP-tagged GitHub repositories created within six months of release, fewer than 5% include servers, typically small, single-maintainer projects dominated by repetitive scaffolding. To address this gap, we present AutoMCP, a compiler that generates MCP servers from OpenAPI 2.0/3.0 specifications. AutoMCP parses REST API definitions and produces complete server implementations, including schema registration and authentication handling. We evaluate AutoMCP on 50 real-world APIs spanning 5,066 endpoints across over 10 domains. From a stratified sample of 1,023 tool calls, 76.5% succeeded out of the box. Manual failure analysis revealed five recurring issues, all attributable to inconsistencies or omissions in the OpenAPI contracts. After minor fixes, averaging 19 lines of spec changes per API, AutoMCP achieved 99.9% success. Our findings (i) analyze MCP adoption and quantify the cost of manual server development, (ii) demonstrate that OpenAPI specifications, despite quality issues, enable near-complete MCP server automation, and (iii) contribute a corpus of 5,066 callable tools along with insights on repairing common specification flaws.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08736v1">ChemBOMAS: Accelerated BO in Chemistry with LLM-Enhanced Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2025-09-10
    </div>
    <details class="paper-abstract">
      The efficiency of Bayesian optimization (BO) in chemistry is often hindered by sparse experimental data and complex reaction mechanisms. To overcome these limitations, we introduce ChemBOMAS, a new framework named LLM-Enhanced Multi-Agent System for accelerating BO in chemistry. ChemBOMAS's optimization process is enhanced by LLMs and synergistically employs two strategies: knowledge-driven coarse-grained optimization and data-driven fine-grained optimization. First, in the knowledge-driven coarse-grained optimization stage, LLMs intelligently decompose the vast search space by reasoning over existing chemical knowledge to identify promising candidate regions. Subsequently, in the data-driven fine-grained optimization stage, LLMs enhance the BO process within these candidate regions by generating pseudo-data points, thereby improving data utilization efficiency and accelerating convergence. Benchmark evaluations** further confirm that ChemBOMAS significantly enhances optimization effectiveness and efficiency compared to various BO algorithms. Importantly, the practical utility of ChemBOMAS was validated through wet-lab experiments conducted under pharmaceutical industry protocols, targeting conditional optimization for a previously unreported and challenging chemical reaction. In the wet experiment, ChemBOMAS achieved an optimal objective value of 96%. This was substantially higher than the 15% achieved by domain experts. This real-world success, together with strong performance on benchmark evaluations, highlights ChemBOMAS as a powerful tool to accelerate chemical discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00419v2">Teaching an Old LLM Secure Coding: Localized Preference Optimization on Distilled Preferences</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 Accepted to ACL 2025 (Main)
    </div>
    <details class="paper-abstract">
      LLM generated code often contains security issues. We address two key challenges in improving secure code generation. First, obtaining high quality training data covering a broad set of security issues is critical. To address this, we introduce a method for distilling a preference dataset of insecure and secure code pairs from frontier LLMs, along with a security reasoning that explains the issues and the fix. The key idea here is to make use of security knowledge sources to devise a systematic prompting strategy that ensures broad coverage. Second, aligning models to secure code requires focusing on localized regions of code. Direct preference optimization methods, like SimPO, are not designed to handle these localized differences and turn out to be ineffective. We address this with a new localized preference optimization algorithm that masks the security related tokens in both the winning (secure) and losing (insecure) responses. To prevent loss in code quality, we also add a regularizer. Evaluations show that both training on our dataset, DiSCo, and the new preference optimization algorithm, LPO, yield substantial reductions in code insecurity while also improving overall code quality. Code and dataset are available at https://github.com/StonyBrookNLP/disco-lpo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02253v4">Scaling LLM Planning: NL2FLOW for Parametric Problem Generation and Rigorous Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 31 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Robust workflow composition is critical for effective agent performance, yet progress in Large Language Model (LLM) planning and reasoning is hindered by a scarcity of scalable evaluation data. This work introduces NL2Flow, a fully automated pipeline for generating and evaluating workflow planning problems. NL2Flow generates problems parametrically in a structured intermediate representation, translating them into both natural language and formal PDDL. I evaluate several open-source, instruct-tuned LLMs on a dataset of 2296 low-difficulty problems generated by NL2Flow. Results demonstrate that the best-performing model achieved 86% success in generating valid plans and 69% in generating optimal plans (for solvable problems). Regression analysis shows that the influence of problem characteristics on plan generation is contingent on both model and prompt design. Importantly, translating natural language problems into a structured JSON representation prior to symbolic planning significantly improved success rates, suggesting a benefit from neuro-symbolic integration. These findings underscore the importance of understanding error sources within LLM reasoning as systems scale to more complex tasks. As LLM reasoning scales to increasingly complex problems, understanding the shifting bottlenecks and sources of error within these systems will be crucial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13794v5">LED: LLM Enhanced Open-Vocabulary Object Detection without Human Curated Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-10
    </div>
    <details class="paper-abstract">
      Large foundation models trained on large-scale vision-language data can boost Open-Vocabulary Object Detection (OVD) via synthetic training data, yet the hand-crafted pipelines often introduce bias and overfit to specific prompts. We sidestep this issue by directly fusing hidden states from Large Language Models (LLMs) into detectors-an avenue surprisingly under-explored. This paper presents a systematic method to enhance visual grounding by utilizing decoder layers of the LLM of an MLLM. We introduce a zero-initialized cross-attention adapter to enable efficient knowledge fusion from LLMs to object detectors, a new approach called LED (LLM Enhanced Open-Vocabulary Object Detection). We find that intermediate LLM layers already encode rich spatial semantics; adapting only the early layers yields most of the gain. With Swin-T as the vision encoder, Qwen2-0.5B + LED lifts GroundingDINO by 3.82 % on OmniLabel at just 8.7 % extra GFLOPs, and a larger vision backbone pushes the improvement to 6.22 %. Extensive ablations on adapter variants, LLM scales and fusion depths further corroborate our design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08646v1">Architecting Resilient LLM Agents: A Guide to Secure Plan-then-Execute Implementations</a></div>
    <div class="paper-meta">
      📅 2025-09-10
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM) agents become increasingly capable of automating complex, multi-step tasks, the need for robust, secure, and predictable architectural patterns is paramount. This paper provides a comprehensive guide to the ``Plan-then-Execute'' (P-t-E) pattern, an agentic design that separates strategic planning from tactical execution. We explore the foundational principles of P-t-E, detailing its core components - the Planner and the Executor - and its architectural advantages in predictability, cost-efficiency, and reasoning quality over reactive patterns like ReAct (Reason + Act). A central focus is placed on the security implications of this design, particularly its inherent resilience to indirect prompt injection attacks by establishing control-flow integrity. We argue that while P-t-E provides a strong foundation, a defense-in-depth strategy is necessary, and we detail essential complementary controls such as the Principle of Least Privilege, task-scoped tool access, and sandboxed code execution. To make these principles actionable, this guide provides detailed implementation blueprints and working code references for three leading agentic frameworks: LangChain (via LangGraph), CrewAI, and AutoGen. Each framework's approach to implementing the P-t-E pattern is analyzed, highlighting unique features like LangGraph's stateful graphs for re-planning, CrewAI's declarative tool scoping for security, and AutoGen's built-in Docker sandboxing. Finally, we discuss advanced patterns, including dynamic re-planning loops, parallel execution with Directed Acyclic Graphs (DAGs), and the critical role of Human-in-the-Loop (HITL) verification, to offer a complete strategic blueprint for architects, developers, and security engineers aiming to build production-grade, resilient, and trustworthy LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03867v2">Drivel-ology: Challenging LLMs with Interpreting Nonsense with Depth</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 Accepted for oral presentation at the EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      We introduce Drivelology, a unique linguistic phenomenon characterised as "nonsense with depth" - utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive. While such expressions may resemble surface-level nonsense, they encode implicit meaning requiring contextual inference, moral reasoning, or emotional interpretation. We find that current large language models (LLMs), despite excelling at many natural language processing (NLP) tasks, consistently fail to grasp the layered semantics of Drivelological text. To investigate this, we construct a benchmark dataset of over 1,200+ meticulously curated and diverse examples across English, Mandarin, Spanish, French, Japanese, and Korean. Each example underwent careful expert review to verify its Drivelological characteristics, involving multiple rounds of discussion and adjudication to address disagreements. Using this dataset, we evaluate a range of LLMs on classification, generation, and reasoning tasks. Our results reveal clear limitations of LLMs: models often confuse Drivelology with shallow nonsense, produce incoherent justifications, or miss implied rhetorical functions altogether. These findings highlight a deep representational gap in LLMs' pragmatic understanding and challenge the assumption that statistical fluency implies cognitive comprehension. We release our dataset and code to facilitate further research in modelling linguistic depth beyond surface-level coherence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08596v1">LLM Ensemble for RAG: Role of Context Length in Zero-Shot Question Answering for BioASQ Challenge</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 CEUR-WS, CLEF2025
    </div>
    <details class="paper-abstract">
      Biomedical question answering (QA) poses significant challenges due to the need for precise interpretation of specialized knowledge drawn from a vast, complex, and rapidly evolving corpus. In this work, we explore how large language models (LLMs) can be used for information retrieval (IR), and an ensemble of zero-shot models can accomplish state-of-the-art performance on a domain-specific Yes/No QA task. Evaluating our approach on the BioASQ challenge tasks, we show that ensembles can outperform individual LLMs and in some cases rival or surpass domain-tuned systems - all while preserving generalizability and avoiding the need for costly fine-tuning or labeled data. Our method aggregates outputs from multiple LLM variants, including models from Anthropic and Google, to synthesize more accurate and robust answers. Moreover, our investigation highlights a relationship between context length and performance: while expanded contexts are meant to provide valuable evidence, they simultaneously risk information dilution and model disorientation. These findings emphasize IR as a critical foundation in Retrieval-Augmented Generation (RAG) approaches for biomedical QA systems. Precise, focused retrieval remains essential for ensuring LLMs operate within relevant information boundaries when generating answers from retrieved documents. Our results establish that ensemble-based zero-shot approaches, when paired with effective RAG pipelines, constitute a practical and scalable alternative to domain-tuned systems for biomedical question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08593v1">No-Knowledge Alarms for Misaligned LLMs-as-Judges</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 7 pages, 1 figure
    </div>
    <details class="paper-abstract">
      If we use LLMs as judges to evaluate the complex decisions of other LLMs, who or what monitors the judges? Infinite monitoring chains are inevitable whenever we do not know the ground truth of the decisions by experts and we do not want to trust them. One way to ameliorate our evaluation uncertainty is to exploit the use of logical consistency between disagreeing experts. By observing how LLM judges agree and disagree while grading other LLMs, we can compute the only possible evaluations of their grading ability. For example, if two LLM judges disagree on which tasks a third one completed correctly, they cannot both be 100\% correct in their judgments. This logic can be formalized as a Linear Programming problem in the space of integer response counts for any finite test. We use it here to develop no-knowledge alarms for misaligned LLM judges. The alarms can detect, with no false positives, that at least one member or more of an ensemble of judges are violating a user specified grading ability requirement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08575v1">SQLGovernor: An LLM-powered SQL Toolkit for Real World Application</a></div>
    <div class="paper-meta">
      📅 2025-09-10
    </div>
    <details class="paper-abstract">
      SQL queries in real world analytical environments, whether written by humans or generated automatically often suffer from syntax errors, inefficiency, or semantic misalignment, especially in complex OLAP scenarios. To address these challenges, we propose SQLGovernor, an LLM powered SQL toolkit that unifies multiple functionalities, including syntax correction, query rewriting, query modification, and consistency verification within a structured framework enhanced by knowledge management. SQLGovernor introduces a fragment wise processing strategy to enable fine grained rewriting and localized error correction, significantly reducing the cognitive load on the LLM. It further incorporates a hybrid self learning mechanism guided by expert feedback, allowing the system to continuously improve through DBMS output analysis and rule validation. Experiments on benchmarks such as BIRD and BIRD CRITIC, as well as industrial datasets, show that SQLGovernor consistently boosts the performance of base models by up to 10%, while minimizing reliance on manual expertise. Deployed in production environments, SQLGovernor demonstrates strong practical utility and effective performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08542v1">BitROM: Weight Reload-Free CiROM Architecture Towards Billion-Parameter 1.58-bit LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 Accepted to ASP-DAC 2026
    </div>
    <details class="paper-abstract">
      Compute-in-Read-Only-Memory (CiROM) accelerators offer outstanding energy efficiency for CNNs by eliminating runtime weight updates. However, their scalability to Large Language Models (LLMs) is fundamentally constrained by their vast parameter sizes. Notably, LLaMA-7B - the smallest model in LLaMA series - demands more than 1,000 cm2 of silicon area even in advanced CMOS nodes. This paper presents BitROM, the first CiROM-based accelerator that overcomes this limitation through co-design with BitNet's 1.58-bit quantization model, enabling practical and efficient LLM inference at the edge. BitROM introduces three key innovations: 1) a novel Bidirectional ROM Array that stores two ternary weights per transistor; 2) a Tri-Mode Local Accumulator optimized for ternary-weight computations; and 3) an integrated Decode-Refresh (DR) eDRAM that supports on-die KV-cache management, significantly reducing external memory access during decoding. In addition, BitROM integrates LoRA-based adapters to enable efficient transfer learning across various downstream tasks. Evaluated in 65nm CMOS, BitROM achieves 20.8 TOPS/W and a bit density of 4,967 kB/mm2 - offering a 10x improvement in area efficiency over prior digital CiROM designs. Moreover, the DR eDRAM contributes to a 43.6% reduction in external DRAM access, further enhancing deployment efficiency for LLMs in edge applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16654v3">MSNav: Zero-Shot Vision-and-Language Navigation with Dynamic Memory and LLM Spatial Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) requires an agent to interpret natural language instructions and navigate complex environments. Current approaches often adopt a "black-box" paradigm, where a single Large Language Model (LLM) makes end-to-end decisions. However, it is plagued by critical vulnerabilities, including poor spatial reasoning, weak cross-modal grounding, and memory overload in long-horizon tasks. To systematically address these issues, we propose Memory Spatial Navigation(MSNav), a framework that fuses three modules into a synergistic architecture, which transforms fragile inference into a robust, integrated intelligence. MSNav integrates three modules: Memory Module, a dynamic map memory module that tackles memory overload through selective node pruning, enhancing long-range exploration; Spatial Module, a module for spatial reasoning and object relationship inference that improves endpoint recognition; and Decision Module, a module using LLM-based path planning to execute robust actions. Powering Spatial Module, we also introduce an Instruction-Object-Space (I-O-S) dataset and fine-tune the Qwen3-4B model into Qwen-Spatial (Qwen-Sp), which outperforms leading commercial LLMs in object list extraction, achieving higher F1 and NDCG scores on the I-O-S test set. Extensive experiments on the Room-to-Room (R2R) and REVERIE datasets demonstrate MSNav's state-of-the-art performance with significant improvements in Success Rate (SR) and Success weighted by Path Length (SPL).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08493v1">Send to which account? Evaluation of an LLM-based Scambaiting System</a></div>
    <div class="paper-meta">
      📅 2025-09-10
    </div>
    <details class="paper-abstract">
      Scammers are increasingly harnessing generative AI(GenAI) technologies to produce convincing phishing content at scale, amplifying financial fraud and undermining public trust. While conventional defenses, such as detection algorithms, user training, and reactive takedown efforts remain important, they often fall short in dismantling the infrastructure scammers depend on, including mule bank accounts and cryptocurrency wallets. To bridge this gap, a proactive and emerging strategy involves using conversational honeypots to engage scammers and extract actionable threat intelligence. This paper presents the first large-scale, real-world evaluation of a scambaiting system powered by large language models (LLMs). Over a five-month deployment, the system initiated over 2,600 engagements with actual scammers, resulting in a dataset of more than 18,700 messages. It achieved an Information Disclosure Rate (IDR) of approximately 32%, successfully extracting sensitive financial information such as mule accounts. Additionally, the system maintained a Human Acceptance Rate (HAR) of around 70%, indicating strong alignment between LLM-generated responses and human operator preferences. Alongside these successes, our analysis reveals key operational challenges. In particular, the system struggled with engagement takeoff: only 48.7% of scammers responded to the initial seed message sent by defenders. These findings highlight the need for further refinement and provide actionable insights for advancing the design of automated scambaiting systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07894v2">HiPhO: How Far Are (M)LLMs from Humans in the Latest High School Physics Olympiad Benchmark?</a></div>
    <div class="paper-meta">
      📅 2025-09-10
    </div>
    <details class="paper-abstract">
      Recently, the physical capabilities of (M)LLMs have garnered increasing attention. However, existing benchmarks for physics suffer from two major gaps: they neither provide systematic and up-to-date coverage of real-world physics competitions such as physics Olympiads, nor enable direct performance comparison with humans. To bridge these gaps, we present HiPhO, the first benchmark dedicated to high school physics Olympiads with human-aligned evaluation. Specifically, HiPhO highlights three key innovations. (1) Comprehensive Data: It compiles 13 latest Olympiad exams from 2024-2025, spanning both international and regional competitions, and covering mixed modalities that encompass problems spanning text-only to diagram-based. (2) Professional Evaluation: We adopt official marking schemes to perform fine-grained grading at both the answer and step level, fully aligned with human examiners to ensure high-quality and domain-specific evaluation. (3) Comparison with Human Contestants: We assign gold, silver, and bronze medals to models based on official medal thresholds, thereby enabling direct comparison between (M)LLMs and human contestants. Our large-scale evaluation of 30 state-of-the-art (M)LLMs shows that: across 13 exams, open-source MLLMs mostly remain at or below the bronze level; open-source LLMs show promising progress with occasional golds; closed-source reasoning MLLMs can achieve 6 to 12 gold medals; and most models still have a significant gap from full marks. These results highlight a substantial performance gap between open-source models and top students, the strong physical reasoning capabilities of closed-source reasoning models, and the fact that there is still significant room for improvement. HiPhO, as a rigorous, human-aligned, and Olympiad-focused benchmark for advancing multimodal physical reasoning, is open-source and available at https://github.com/SciYu/HiPhO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08484v1">Simulating Identity, Propagating Bias: Abstraction and Stereotypes in LLM-Generated Text</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 Accepted to EMNLP Findings 2025
    </div>
    <details class="paper-abstract">
      Persona-prompting is a growing strategy to steer LLMs toward simulating particular perspectives or linguistic styles through the lens of a specified identity. While this method is often used to personalize outputs, its impact on how LLMs represent social groups remains underexplored. In this paper, we investigate whether persona-prompting leads to different levels of linguistic abstraction - an established marker of stereotyping - when generating short texts linking socio-demographic categories with stereotypical or non-stereotypical attributes. Drawing on the Linguistic Expectancy Bias framework, we analyze outputs from six open-weight LLMs under three prompting conditions, comparing 11 persona-driven responses to those of a generic AI assistant. To support this analysis, we introduce Self-Stereo, a new dataset of self-reported stereotypes from Reddit. We measure abstraction through three metrics: concreteness, specificity, and negation. Our results highlight the limits of persona-prompting in modulating abstraction in language, confirming criticisms about the ecology of personas as representative of socio-demographic groups and raising concerns about the risk of propagating stereotypes even when seemingly evoking the voice of a marginalized group.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08395v3">IssueBench: Millions of Realistic Prompts for Measuring Issue Bias in LLM Writing Assistance</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 accepted at TACL (pre-MIT Press publication version)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are helping millions of users write texts about diverse issues, and in doing so expose users to different ideas and perspectives. This creates concerns about issue bias, where an LLM tends to present just one perspective on a given issue, which in turn may influence how users think about this issue. So far, it has not been possible to measure which issue biases LLMs manifest in real user interactions, making it difficult to address the risks from biased LLMs. Therefore, we create IssueBench: a set of 2.49m realistic English-language prompts to measure issue bias in LLM writing assistance, which we construct based on 3.9k templates (e.g. "write a blog about") and 212 political issues (e.g. "AI regulation") from real user interactions. Using IssueBench, we show that issue biases are common and persistent in 10 state-of-the-art LLMs. We also show that biases are very similar across models, and that all models align more with US Democrat than Republican voter opinion on a subset of issues. IssueBench can easily be adapted to include other issues, templates, or tasks. By enabling robust and realistic measurement, we hope that IssueBench can bring a new quality of evidence to ongoing discussions about LLM biases and how to address them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08416v1">AutoVeriFix: Automatically Correcting Errors and Enhancing Functional Correctness in LLM-Generated Verilog Code</a></div>
    <div class="paper-meta">
      📅 2025-09-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in generating software code for high-level programming languages such as Python and C++. However, their application to hardware description languages, such as Verilog, is challenging due to the scarcity of high-quality training data. Current approaches to Verilog code generation using LLMs often focus on syntactic correctness, resulting in code with functional errors. To address these challenges, we present AutoVeriFix, a novel Python-assisted two-stage framework designed to enhance the functional correctness of LLM-generated Verilog code. In the first stage, LLMs are employed to generate high-level Python reference models that define the intended circuit behavior. In the second stage, these Python models facilitate the creation of automated tests that guide the generation of Verilog RTL implementations. Simulation discrepancies between the reference model and the Verilog code are iteratively used to identify and correct errors, thereby improving the functional accuracy and reliability of the LLM-generated Verilog code. Experimental results demonstrate that our approach significantly outperforms existing state-of-the-art methods in improving the functional correctness of generated Verilog code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08407v1">An Iterative LLM Framework for SIBT utilizing RAG-based Adaptive Weight Optimization</a></div>
    <div class="paper-meta">
      📅 2025-09-10
    </div>
    <details class="paper-abstract">
      Seed implant brachytherapy (SIBT) is an effective cancer treatment modality; however, clinical planning often relies on manual adjustment of objective function weights, leading to inefficiencies and suboptimal results. This study proposes an adaptive weight optimization framework for SIBT planning, driven by large language models (LLMs). A locally deployed DeepSeek-R1 LLM is integrated with an automatic planning algorithm in an iterative loop. Starting with fixed weights, the LLM evaluates plan quality and recommends new weights in the next iteration. This process continues until convergence criteria are met, after which the LLM conducts a comprehensive evaluation to identify the optimal plan. A clinical knowledge base, constructed and queried via retrieval-augmented generation (RAG), enhances the model's domain-specific reasoning. The proposed method was validated on 23 patient cases, showing that the LLM-assisted approach produces plans that are comparable to or exceeding clinically approved and fixed-weight plans, in terms of dose homogeneity for the clinical target volume (CTV) and sparing of organs at risk (OARs). The study demonstrates the potential use of LLMs in SIBT planning automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08400v1">Ubiquitous Intelligence Via Wireless Network-Driven LLMs Evolution</a></div>
    <div class="paper-meta">
      📅 2025-09-10
    </div>
    <details class="paper-abstract">
      We introduce ubiquitous intelligence as a paradigm where Large Language Models (LLMs) evolve within wireless network-driven ecosystems. Unlike static model deployments, this approach enables scalable and continuous intelligence ascension through coordination between networks and LLMs. Wireless networks support system-orchestrated lifelong learning, while LLMs drive the next-generation network development that is more adaptive and responsive. This co-evolution highlights a shift toward self-improving systems, sustaining capability growth across diverse and resource-constrained environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14161v3">TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      We interact with computers on an everyday basis, be it in everyday life or work, and many aspects of work can be done entirely with access to a computer and the Internet. At the same time, thanks to improvements in large language models (LLMs), there has also been a rapid development in AI agents that interact with and affect change in their surrounding environments. But how performant are AI agents at accelerating or even autonomously performing work-related tasks? The answer to this question has important implications both for industry looking to adopt AI into their workflows and for economic policy to understand the effects that adoption of AI may have on the labor market. To measure the progress of these LLM agents' performance on performing real-world professional tasks, in this paper we introduce TheAgentCompany, an extensible benchmark for evaluating AI agents that interact with the world in similar ways to those of a digital worker: by browsing the Web, writing code, running programs, and communicating with other coworkers. We build a self-contained environment with internal web sites and data that mimics a small software company environment, and create a variety of tasks that may be performed by workers in such a company. We test baseline agents powered by both closed API-based and open-weights language models (LMs), and find that the most competitive agent can complete 30% of tasks autonomously. This paints a nuanced picture on task automation with LM agents--in a setting simulating a real workplace, a good portion of simpler tasks could be solved autonomously, but more difficult long-horizon tasks are still beyond the reach of current systems. We release code, data, environment, and experiments on https://the-agent-company.com.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08385v1">LLM-Guided Ansätze Design for Quantum Circuit Born Machines in Financial Generative Modeling</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 Work presented at the 3rd International Workshop on Quantum Machine Learning: From Research to Practice (QML@QCE'25)
    </div>
    <details class="paper-abstract">
      Quantum generative modeling using quantum circuit Born machines (QCBMs) shows promising potential for practical quantum advantage. However, discovering ans\"atze that are both expressive and hardware-efficient remains a key challenge, particularly on noisy intermediate-scale quantum (NISQ) devices. In this work, we introduce a prompt-based framework that leverages large language models (LLMs) to generate hardware-aware QCBM architectures. Prompts are conditioned on qubit connectivity, gate error rates, and hardware topology, while iterative feedback, including Kullback-Leibler (KL) divergence, circuit depth, and validity, is used to refine the circuits. We evaluate our method on a financial modeling task involving daily changes in Japanese government bond (JGB) interest rates. Our results show that the LLM-generated ans\"atze are significantly shallower and achieve superior generative performance compared to the standard baseline when executed on real IBM quantum hardware using 12 qubits. These findings demonstrate the practical utility of LLM-driven quantum architecture search and highlight a promising path toward robust, deployable generative models for near-term quantum devices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08358v1"><think> So let's replace this phrase with insult... </think> Lessons learned from generation of toxic texts with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-10
    </div>
    <details class="paper-abstract">
      Modern Large Language Models (LLMs) are excellent at generating synthetic data. However, their performance in sensitive domains such as text detoxification has not received proper attention from the scientific community. This paper explores the possibility of using LLM-generated synthetic toxic data as an alternative to human-generated data for training models for detoxification. Using Llama 3 and Qwen activation-patched models, we generated synthetic toxic counterparts for neutral texts from ParaDetox and SST-2 datasets. Our experiments show that models fine-tuned on synthetic data consistently perform worse than those trained on human data, with a drop in performance of up to 30% in joint metrics. The root cause is identified as a critical lexical diversity gap: LLMs generate toxic content using a small, repetitive vocabulary of insults that fails to capture the nuances and variety of human toxicity. These findings highlight the limitations of current LLMs in this domain and emphasize the continued importance of diverse, human-annotated data for building robust detoxification systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15337v3">Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 Accepted by EMNLP-2025
    </div>
    <details class="paper-abstract">
      The misuse of large language models (LLMs), such as academic plagiarism, has driven the development of detectors to identify LLM-generated texts. To bypass these detectors, paraphrase attacks have emerged to purposely rewrite these texts to evade detection. Despite the success, existing methods require substantial data and computational budgets to train a specialized paraphraser, and their attack efficacy greatly reduces when faced with advanced detection algorithms. To address this, we propose \textbf{Co}ntrastive \textbf{P}araphrase \textbf{A}ttack (CoPA), a training-free method that effectively deceives text detectors using off-the-shelf LLMs. The first step is to carefully craft instructions that encourage LLMs to produce more human-like texts. Nonetheless, we observe that the inherent statistical biases of LLMs can still result in some generated texts carrying certain machine-like attributes that can be captured by detectors. To overcome this, CoPA constructs an auxiliary machine-like word distribution as a contrast to the human-like distribution generated by the LLM. By subtracting the machine-like patterns from the human-like distribution during the decoding process, CoPA is able to produce sentences that are less discernible by text detectors. Our theoretical analysis suggests the superiority of the proposed attack. Extensive experiments validate the effectiveness of CoPA in fooling text detectors across various scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04373v3">Measuring Bias or Measuring the Task: Understanding the Brittle Nature of LLM Gender Biases</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 To be published at EMNLP 2025 (main conference)
    </div>
    <details class="paper-abstract">
      As LLMs are increasingly applied in socially impactful settings, concerns about gender bias have prompted growing efforts both to measure and mitigate such bias. These efforts often rely on evaluation tasks that differ from natural language distributions, as they typically involve carefully constructed task prompts that overtly or covertly signal the presence of gender bias-related content. In this paper, we examine how signaling the evaluative purpose of a task impacts measured gender bias in LLMs. Concretely, we test models under prompt conditions that (1) make the testing context salient, and (2) make gender-focused content salient. We then assess prompt sensitivity across four task formats with both token-probability and discrete-choice metrics. We find that prompts that more clearly align with (gender bias) evaluation framing elicit distinct gender output distributions compared to less evaluation-framed prompts. Discrete-choice metrics further tend to amplify bias relative to probabilistic measures. These findings do not only highlight the brittleness of LLM gender bias evaluations but open a new puzzle for the NLP benchmarking and development community: To what extent can well-controlled testing designs trigger LLM "testing mode" performance, and what does this mean for the ecological validity of future benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08309v1">Hetis: Serving LLMs in Heterogeneous GPU Clusters with Fine-grained and Dynamic Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-09-10
    </div>
    <details class="paper-abstract">
      The significant resource demands in LLM serving prompts production clusters to fully utilize heterogeneous hardware by partitioning LLM models across a mix of high-end and low-end GPUs. However, existing parallelization approaches often struggle to scale efficiently in heterogeneous environments due to their coarse-grained and static parallelization strategies. In this paper, we introduce Hetis, a new LLM system tailored for heterogeneous GPU clusters. Hetis addresses two critical challenges: (1) memory inefficiency caused by the mismatch between memory capacity and computational power in heterogeneous devices, and (2) computational inefficiency arising from performance gaps across different LLM modules. To tackle these issues, Hetis employs a fine-grained and dynamic parallelism design. Specifically, it selectively parallelizes compute-intensive operations to reduce latency and dynamically distributes Attention computations to low-end GPUs at a head granularity, leveraging the distinct characteristics of each module. Additionally, Hetis features an online load dispatching policy that continuously optimizes serving performance by carefully balancing network latency, computational load, and memory intensity. Evaluation results demonstrate that Hetis can improve serving throughput by up to $2.25\times$ and reduce latency by $1.49\times$ compared to existing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07170v2">That's So FETCH: Fashioning Ensemble Techniques for LLM Classification in Civil Legal Intake and Referral</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 Submission to JURIX 2025
    </div>
    <details class="paper-abstract">
      Each year millions of people seek help for their legal problems by calling a legal aid program hotline, walking into a legal aid office, or using a lawyer referral service. The first step to match them to the right help is to identify the legal problem the applicant is experiencing. Misdirection has consequences. Applicants may miss a deadline, experience physical abuse, lose housing or lose custody of children while waiting to connect to the right legal help. We introduce and evaluate the FETCH classifier for legal issue classification and describe two methods for improving accuracy: a hybrid LLM/ML ensemble classification method, and the automatic generation of follow-up questions to enrich the initial problem narrative. We employ a novel data set of 419 real-world queries to a nonprofit lawyer referral service. Ultimately, we show classification accuracy (hits@2) of 97.37\% using a mix of inexpensive models, exceeding the performance of the current state-of-the-art GPT-5 model. Our approach shows promise in significantly reducing the cost of guiding users of the legal system to the right resource for their problem while achieving high accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08203v1">Componentization: Decomposing Monolithic LLM Responses into Manipulable Semantic Units</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often produce monolithic text that is hard to edit in parts, which can slow down collaborative workflows. We present componentization, an approach that decomposes model outputs into modular, independently editable units while preserving context. We describe Modular and Adaptable Output Decomposition (MAOD), which segments responses into coherent components and maintains links among them, and we outline the Component-Based Response Architecture (CBRA) as one way to implement this idea. Our reference prototype, MAODchat, uses a microservices design with state-machine-based decomposition agents, vendor-agnostic model adapters, and real-time component manipulation with recomposition. In an exploratory study with four participants from academic, engineering, and product roles, we observed that component-level editing aligned with several common workflows and enabled iterative refinement and selective reuse. Participants also mentioned possible team workflows. Our contributions are: (1) a definition of componentization for transforming monolithic outputs into manipulable units, (2) CBRA and MAODchat as a prototype architecture, (3) preliminary observations from a small user study, (4) MAOD as an algorithmic sketch for semantic segmentation, and (5) example Agent-to-Agent protocols for automated decomposition. We view componentization as a promising direction for turning passive text consumption into more active, component-level collaboration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07939v1">Guided Reasoning in LLM-Driven Penetration Testing Using Structured Attack Trees</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have driven interest in automating cybersecurity penetration testing workflows, offering the promise of faster and more consistent vulnerability assessment for enterprise systems. Existing LLM agents for penetration testing primarily rely on self-guided reasoning, which can produce inaccurate or hallucinated procedural steps. As a result, the LLM agent may undertake unproductive actions, such as exploiting unused software libraries or generating cyclical responses that repeat prior tactics. In this work, we propose a guided reasoning pipeline for penetration testing LLM agents that incorporates a deterministic task tree built from the MITRE ATT&CK Matrix, a proven penetration testing kll chain, to constrain the LLM's reaoning process to explicitly defined tactics, techniques, and procedures. This anchors reasoning in proven penetration testing methodologies and filters out ineffective actions by guiding the agent towards more productive attack procedures. To evaluate our approach, we built an automated penetration testing LLM agent using three LLMs (Llama-3-8B, Gemini-1.5, and GPT-4) and applied it to navigate 10 HackTheBox cybersecurity exercises with 103 discrete subtasks representing real-world cyberattack scenarios. Our proposed reasoning pipeline guided the LLM agent through 71.8\%, 72.8\%, and 78.6\% of subtasks using Llama-3-8B, Gemini-1.5, and GPT-4, respectively. Comparatively, the state-of-the-art LLM penetration testing tool using self-guided reasoning completed only 13.5\%, 16.5\%, and 75.7\% of subtasks and required 86.2\%, 118.7\%, and 205.9\% more model queries. This suggests that incorporating a deterministic task tree into LLM reasoning pipelines can enhance the accuracy and efficiency of automated cybersecurity assessments
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07933v1">Breaking Android with AI: A Deep Dive into LLM-Powered Exploitation</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      The rapid evolution of Artificial Intelligence (AI) and Large Language Models (LLMs) has opened up new opportunities in the area of cybersecurity, especially in the exploitation automation landscape and penetration testing. This study explores Android penetration testing automation using LLM-based tools, especially PentestGPT, to identify and execute rooting techniques. Through a comparison of the traditional manual rooting process and exploitation methods produced using AI, this study evaluates the efficacy, reliability, and scalability of automated penetration testing in achieving high-level privilege access on Android devices. With the use of an Android emulator (Genymotion) as the testbed, we fully execute both traditional and exploit-based rooting methods, automating the process using AI-generated scripts. Secondly, we create a web application by integrating OpenAI's API to facilitate automated script generation from LLM-processed responses. The research focuses on the effectiveness of AI-enabled exploitation by comparing automated and manual penetration testing protocols, by determining LLM weaknesses and strengths along the way. We also provide security suggestions of AI-enabled exploitation, including ethical factors and potential misuse. The findings exhibit that while LLMs can significantly streamline the workflow of exploitation, they need to be controlled by humans to ensure accuracy and ethical application. This study adds to the increasing body of literature on AI-powered cybersecurity and its effect on ethical hacking, security research, and mobile device security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07894v1">HiPhO: How Far Are (M)LLMs from Humans in the Latest High School Physics Olympiad Benchmark?</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Recently, the physical capabilities of (M)LLMs have garnered increasing attention. However, existing benchmarks for physics suffer from two major gaps: they neither provide systematic and up-to-date coverage of real-world physics competitions such as physics Olympiads, nor enable direct performance comparison with humans. To bridge these gaps, we present HiPhO, the first benchmark dedicated to high school physics Olympiads with human-aligned evaluation. Specifically, HiPhO highlights three key innovations. (1) Comprehensive Data: It compiles 13 latest Olympiad exams from 2024-2025, spanning both international and regional competitions, and covering mixed modalities that encompass problems spanning text-only to diagram-based. (2) Professional Evaluation: We adopt official marking schemes to perform fine-grained grading at both the answer and step level, fully aligned with human examiners to ensure high-quality and domain-specific evaluation. (3) Comparison with Human Contestants: We assign gold, silver, and bronze medals to models based on official medal thresholds, thereby enabling direct comparison between (M)LLMs and human contestants. Our large-scale evaluation of 30 state-of-the-art (M)LLMs shows that: across 13 exams, open-source MLLMs mostly remain at or below the bronze level; open-source LLMs show promising progress with occasional golds; closed-source reasoning MLLMs can achieve 6 to 12 gold medals; and most models still have a significant gap from full marks. These results highlight a substantial performance gap between open-source models and top students, the strong physical reasoning capabilities of closed-source reasoning models, and the fact that there is still significant room for improvement. HiPhO, as a rigorous, human-aligned, and Olympiad-focused benchmark for advancing multimodal physical reasoning, is open-source and available at https://github.com/SciYu/HiPhO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07128v2">Cardiverse: Harnessing LLMs for Novel Card Game Prototyping</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 37 pages, 13 figures, 8 tables. Accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      The prototyping of computer games, particularly card games, requires extensive human effort in creative ideation and gameplay evaluation. Recent advances in Large Language Models (LLMs) offer opportunities to automate and streamline these processes. However, it remains challenging for LLMs to design novel game mechanics beyond existing databases, generate consistent gameplay environments, and develop scalable gameplay AI for large-scale evaluations. This paper addresses these challenges by introducing a comprehensive automated card game prototyping framework. The approach highlights a graph-based indexing method for generating novel game variations, an LLM-driven system for consistent game code generation validated by gameplay records, and a gameplay AI constructing method that uses an ensemble of LLM-generated heuristic functions optimized through self-play. These contributions aim to accelerate card game prototyping, reduce human labor, and lower barriers to entry for game developers. For code repo visit this http URL https://github.com/danruili/Cardiverse
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07860v1">KLIPA: A Knowledge Graph and LLM-Driven QA Framework for IP Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Effectively managing intellectual property is a significant challenge. Traditional methods for patent analysis depend on labor-intensive manual searches and rigid keyword matching. These approaches are often inefficient and struggle to reveal the complex relationships hidden within large patent datasets, hindering strategic decision-making. To overcome these limitations, we introduce KLIPA, a novel framework that leverages a knowledge graph and a large language model (LLM) to significantly advance patent analysis. Our approach integrates three key components: a structured knowledge graph to map explicit relationships between patents, a retrieval-augmented generation(RAG) system to uncover contextual connections, and an intelligent agent that dynamically determines the optimal strategy for resolving user queries. We validated KLIPA on a comprehensive, real-world patent database, where it demonstrated substantial improvements in knowledge extraction, discovery of novel connections, and overall operational efficiency. This combination of technologies enhances retrieval accuracy, reduces reliance on domain experts, and provides a scalable, automated solution for any organization managing intellectual property, including technology corporations and legal firms, allowing them to better navigate the complexities of strategic innovation and competitive intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07858v1">SCoder: Iterative Self-Distillation for Bootstrapping Small-Scale Data Synthesizers to Empower Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Existing code large language models (LLMs) often rely on large-scale instruction data distilled from proprietary LLMs for fine-tuning, which typically incurs high costs. In this paper, we explore the potential of small-scale open-source LLMs (e.g., 7B) as synthesizers for high-quality code instruction data construction. We first observe that the data synthesis capability of small-scale LLMs can be enhanced by training on a few superior data synthesis samples from proprietary LLMs. Building on this, we propose a novel iterative self-distillation approach to bootstrap small-scale LLMs, transforming them into powerful synthesizers that reduce reliance on proprietary LLMs and minimize costs. Concretely, in each iteration, to obtain diverse and high-quality self-distilled data, we design multi-checkpoint sampling and multi-aspect scoring strategies for initial data selection. Furthermore, to identify the most influential samples, we introduce a gradient-based influence estimation method for final data filtering. Based on the code instruction datasets from the small-scale synthesizers, we develop SCoder, a family of code generation models fine-tuned from DeepSeek-Coder. SCoder models achieve state-of-the-art code generation capabilities, demonstrating the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07846v1">Aligning LLMs for the Classroom with Knowledge-Based Retrieval -- A Comparative RAG Study</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Large language models like ChatGPT are increasingly used in classrooms, but they often provide outdated or fabricated information that can mislead students. Retrieval Augmented Generation (RAG) improves reliability of LLMs by grounding responses in external resources. We investigate two accessible RAG paradigms, vector-based retrieval and graph-based retrieval to identify best practices for classroom question answering (QA). Existing comparative studies fail to account for pedagogical factors such as educational disciplines, question types, and practical deployment costs. Using a novel dataset, EduScopeQA, of 3,176 questions across academic subjects, we measure performance on various educational query types, from specific facts to broad thematic discussions. We also evaluate system alignment with a dataset of systematically altered textbooks that contradict the LLM's latent knowledge. We find that OpenAI Vector Search RAG (representing vector-based RAG) performs well as a low-cost generalist, especially for quick fact retrieval. On the other hand, GraphRAG Global excels at providing pedagogically rich answers to thematic queries, and GraphRAG Local achieves the highest accuracy with the dense, altered textbooks when corpus integrity is critical. Accounting for the 10-20x higher resource usage of GraphRAG (representing graph-based RAG), we show that a dynamic branching framework that routes queries to the optimal retrieval method boosts fidelity and efficiency. These insights provide actionable guidelines for educators and system designers to integrate RAG-augmented LLMs into learning environments effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07819v1">LLMs in Wikipedia: Investigating How LLMs Impact Participation in Knowledge Communities</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are reshaping knowledge production as community members increasingly incorporate them into their contribution workflows. However, participating in knowledge communities involves more than just contributing content - it is also a deeply social process. While communities must carefully consider appropriate and responsible LLM integration, the absence of concrete norms has left individual editors to experiment and navigate LLM use on their own. Understanding how LLMs influence community participation is therefore critical in shaping future norms and supporting effective adoption. To address this gap, we investigated Wikipedia, one of the largest knowledge production communities, to understand 1) how LLMs influence the ways editors contribute content, 2) what strategies editors leverage to align LLM outputs with community norms, and 3) how other editors in the community respond to LLM-assisted contributions. Through interviews with 16 Wikipedia editors who had used LLMs for their edits, we found that 1) LLMs affected the content contributions for experienced and new editors differently; 2) aligning LLM outputs with community norms required tacit knowledge that often challenged newcomers; and 3) as a result, other editors responded to LLM-assisted edits differently depending on the editors' expertise level. Based on these findings, we challenge existing models of newcomer involvement and propose design implications for LLMs that support community engagement through scaffolding, teaching, and context awareness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07642v3">TreeReview: A Dynamic Tree of Questions Framework for Deep and Efficient LLM-based Scientific Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Accepted to EMNLP2025 Main
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have shown significant potential in assisting peer review, current methods often struggle to generate thorough and insightful reviews while maintaining efficiency. In this paper, we propose TreeReview, a novel framework that models paper review as a hierarchical and bidirectional question-answering process. TreeReview first constructs a tree of review questions by recursively decomposing high-level questions into fine-grained sub-questions and then resolves the question tree by iteratively aggregating answers from leaf to root to get the final review. Crucially, we incorporate a dynamic question expansion mechanism to enable deeper probing by generating follow-up questions when needed. We construct a benchmark derived from ICLR and NeurIPS venues to evaluate our method on full review generation and actionable feedback comments generation tasks. Experimental results of both LLM-based and human evaluation show that TreeReview outperforms strong baselines in providing comprehensive, in-depth, and expert-aligned review feedback, while reducing LLM token usage by up to 80% compared to computationally intensive approaches. Our code and benchmark dataset are available at https://github.com/YuanChang98/tree-review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18934v2">Dovetail: A CPU/GPU Heterogeneous Speculative Decoding for LLM inference</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 14 pages, 6 figures
    </div>
    <details class="paper-abstract">
      With the continuous advancement in the performance of large language models (LLMs), their demand for computational resources and memory has significantly increased, which poses major challenges for efficient inference on consumer-grade devices and legacy servers. These devices typically feature relatively weaker GPUs and stronger CPUs. Although techniques such as parameter offloading and partial offloading can alleviate GPU memory pressure to some extent, their effectiveness is limited due to communication latency and suboptimal hardware resource utilization. To address this issue, we propose Dovetail, a lossless inference acceleration method that leverages the complementary characteristics of heterogeneous devices and the advantages of speculative decoding. Dovetail deploys a draft model on the GPU to perform preliminary predictions, while a target model running on the CPU validates these outputs. By reducing the granularity of data transfer, Dovetail significantly minimizes communication overhead. To further improve efficiency, we optimize the draft model specifically for heterogeneous hardware environments by reducing the number of draft tokens to lower parallel verification latency, increasing model depth to enhance predictive capabilities, and introducing a Dynamic Gating Fusion (DGF) mechanism to improve the integration of feature and embedding information. We conduct comprehensive evaluations of Dovetail across various consumer-grade GPUs, covering multiple tasks and mainstream models. Experimental results on 13B models demonstrate that Dovetail achieves inference speedups ranging from 1.79x to 10.1x across different devices, while maintaining consistency and stability in the distribution of generated texts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07763v1">What Were You Thinking? An LLM-Driven Large-Scale Study of Refactoring Motivations in Open-Source Projects</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Context. Code refactoring improves software quality without changing external behavior. Despite its advantages, its benefits are hindered by the considerable cost of time, resources, and continuous effort it demands. Aim. Understanding why developers refactor, and which metrics capture these motivations, may support wider and more effective use of refactoring in practice. Method. We performed a large-scale empirical study to analyze developers refactoring activity, leveraging Large Language Models (LLMs) to identify underlying motivations from version control data, comparing our findings with previous motivations reported in the literature. Results. LLMs matched human judgment in 80% of cases, but aligned with literature-based motivations in only 47%. They enriched 22% of motivations with more detailed rationale, often highlighting readability, clarity, and structural improvements. Most motivations were pragmatic, focused on simplification and maintainability. While metrics related to developer experience and code readability ranked highest, their correlation with motivation categories was weak. Conclusions. We conclude that LLMs effectively capture surface-level motivations but struggle with architectural reasoning. Their value lies in providing localized explanations, which, when combined with software metrics, can form hybrid approaches. Such integration offers a promising path toward prioritizing refactoring more systematically and balancing short-term improvements with long-term architectural goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07759v1">A Survey of Long-Document Retrieval in the PLM and LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 33 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The proliferation of long-form documents presents a fundamental challenge to information retrieval (IR), as their length, dispersed evidence, and complex structures demand specialized methods beyond standard passage-level techniques. This survey provides the first comprehensive treatment of long-document retrieval (LDR), consolidating methods, challenges, and applications across three major eras. We systematize the evolution from classical lexical and early neural models to modern pre-trained (PLM) and large language models (LLMs), covering key paradigms like passage aggregation, hierarchical encoding, efficient attention, and the latest LLM-driven re-ranking and retrieval techniques. Beyond the models, we review domain-specific applications, specialized evaluation resources, and outline critical open challenges such as efficiency trade-offs, multimodal alignment, and faithfulness. This survey aims to provide both a consolidated reference and a forward-looking agenda for advancing long-document retrieval in the era of foundation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07755v1">Factuality Beyond Coherence: Evaluating LLM Watermarking Methods for Medical Texts</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Accepted at EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) adapted to sensitive domains such as medicine, their fluency raises safety risks, particularly regarding provenance and accountability. Watermarking embeds detectable patterns to mitigate these risks, yet its reliability in medical contexts remains untested. Existing benchmarks focus on detection-quality tradeoffs, overlooking factual risks under low-entropy settings often exploited by watermarking's reweighting strategy. We propose a medical-focused evaluation workflow that jointly assesses factual accuracy and coherence. Using GPT-Judger and further human validation, we introduce the Factuality-Weighted Score (FWS), a composite metric prioritizing factual accuracy beyond coherence to guide watermarking deployment in medical domains. Our evaluation shows current watermarking methods substantially compromise medical factuality, with entropy shifts degrading medical entity representation. These findings underscore the need for domain-aware watermarking approaches that preserve the integrity of medical content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04656v2">AraHalluEval: A Fine-grained Hallucination Evaluation Framework for Arabic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Recently, extensive research on the hallucination of the large language models (LLMs) has mainly focused on the English language. Despite the growing number of multilingual and Arabic-specific LLMs, evaluating LLMs' hallucination in the Arabic context remains relatively underexplored. The knowledge gap is particularly pressing given Arabic's widespread use across many regions and its importance in global communication and media. This paper presents the first comprehensive hallucination evaluation of Arabic and multilingual LLMs on two critical Arabic natural language generation tasks: generative question answering (GQA) and summarization. This study evaluates a total of 12 LLMs, including 4 Arabic pre-trained models, 4 multilingual models, and 4 reasoning-based models. To assess the factual consistency and faithfulness of LLMs' outputs, we developed a fine-grained hallucination evaluation framework consisting of 12 fine-grained hallucination indicators that represent the varying characteristics of each task. The results reveal that factual hallucinations are more prevalent than faithfulness errors across all models and tasks. Notably, the Arabic pre-trained model Allam consistently demonstrates lower hallucination rates than multilingual models and a comparative performance with reasoning-based models. The code is available at: https://github.com/aishaalansari57/AraHalluEval
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01542v2">Register Always Matters: Analysis of LLM Pretraining Data Through the Lens of Language Variation</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Pretraining data curation is a cornerstone in Large Language Model (LLM) development, leading to growing research on quality filtering of large web corpora. From statistical quality flags to LLM-based labelling systems, datasets are divided into categories, frequently reducing to a binary: those passing the filters are deemed as valuable examples, others are discarded as useless or detrimental. However, a more detailed understanding of the contribution of different kinds of texts to model performance is still largely lacking. In this article, we present the first study utilising registers or genres - a widely used standard in corpus linguistics to model linguistic variation - to curate pretraining datasets and investigate the effect of register on the performance of LLMs. We train small generative models with register classified data and evaluate them using standard benchmarks, and show that the register of pretraining data substantially affects model performance. We uncover surprising relationships between the pretraining material and the resulting models: using the News register results in subpar performance, and on the contrary, including the Opinion class, covering texts such as reviews and opinion blogs, is highly beneficial. While a model trained on the entire unfiltered dataset outperforms those trained on datasets limited to a single register, combining well-performing registers like How-to-Instructions, Informational Description, and Opinion leads to major improvements. Furthermore, analysis of individual benchmark results reveals key differences in the strengths and drawbacks of specific register classes as pretraining data. These findings show that register is an important explainer of model variation and can facilitate more deliberate future data selection practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.02886v3">TokenSelect: Efficient Long-Context Inference and Length Extrapolation for LLMs via Dynamic Token-Level KV Cache Selection</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Accepted by EMNLP2025
    </div>
    <details class="paper-abstract">
      Rapid advances in Large Language Models (LLMs) have spurred demand for processing extended context sequences in contemporary applications. However, this progress faces two challenges: performance degradation due to sequence lengths out-of-distribution, and excessively long inference times caused by the quadratic computational complexity of attention. These issues limit LLMs in long-context scenarios. In this paper, we propose Dynamic Token-Level KV Cache Selection (TokenSelect), a training-free method for efficient and accurate long-context inference. TokenSelect builds upon the observation of non-contiguous attention sparsity, using QK dot products to measure per-head KV Cache criticality at token-level. By per-head soft voting mechanism, TokenSelect selectively involves a few critical KV cache tokens in attention calculation without sacrificing accuracy. To further accelerate TokenSelect, we design the Selection Cache based on observations of consecutive Query similarity and implemented the efficient Paged Dot Product Kernel, significantly reducing the selection overhead. A comprehensive evaluation of TokenSelect demonstrates up to $23.84\times$ speedup in attention computation and up to $2.28\times$ acceleration in end-to-end latency, while providing superior performance compared to state-of-the-art long-context inference methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07676v1">Unleashing the True Potential of LLMs: A Feedback-Triggered Self-Correction with Long-Term Multipath Decoding</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable performance across diverse tasks, yet their susceptibility to generating incorrect content during inference remains a critical unsolved challenge. While self-correction methods offer potential solutions, their effectiveness is hindered by two inherent limitations: (1) the absence of reliable guidance signals for error localization, and (2) the restricted reasoning depth imposed by conventional next-token decoding paradigms. To address these issues, we propose Feedback-Triggered Regeneration (FTR), a novel framework that synergizes user feedback with enhanced decoding dynamics. Specifically, FTR activates response regeneration only upon receiving negative user feedback, thereby circumventing error propagation from faulty self-assessment while preserving originally correct outputs. Furthermore, we introduce Long-Term Multipath (LTM) decoding, which enables systematic exploration of multiple reasoning trajectories through delayed sequence evaluation, effectively overcoming the myopic decision-making characteristic of standard next-token prediction. Extensive experiments on mathematical reasoning and code generation benchmarks demonstrate that our framework achieves consistent and significant improvements over state-of-the-art prompt-based self-correction methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07622v1">MaLei at MultiClinSUM: Summarisation of Clinical Documents using Perspective-Aware Iterative Self-Prompting with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 system paper at CLEF 2025
    </div>
    <details class="paper-abstract">
      Efficient communication between patients and clinicians plays an important role in shared decision-making. However, clinical reports are often lengthy and filled with clinical jargon, making it difficult for domain experts to identify important aspects in the document efficiently. This paper presents the methodology we applied in the MultiClinSUM shared task for summarising clinical case documents. We used an Iterative Self-Prompting technique on large language models (LLMs) by asking LLMs to generate task-specific prompts and refine them via example-based few-shot learning. Furthermore, we used lexical and embedding space metrics, ROUGE and BERT-score, to guide the model fine-tuning with epochs. Our submission using perspective-aware ISP on GPT-4 and GPT-4o achieved ROUGE scores (46.53, 24.68, 30.77) and BERTscores (87.84, 83.25, 85.46) for (P, R, F1) from the official evaluation on 3,396 clinical case reports from various specialties extracted from open journals. The high BERTscore indicates that the model produced semantically equivalent output summaries compared to the references, even though the overlap at the exact lexicon level is lower, as reflected in the lower ROUGE scores. This work sheds some light on how perspective-aware ISP (PA-ISP) can be deployed for clinical report summarisation and support better communication between patients and clinicians.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.12791v2">CTourLLM: Enhancing LLMs with Chinese Tourism Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have demonstrated their effectiveness in various natural language processing (NLP) tasks. However, the lack of tourism knowledge limits the performance of LLMs in tourist attraction presentations and travel planning. To address this challenge, we constructed a supervised fine-tuning dataset for the Chinese culture and tourism domain, named Cultour. This dataset consists of three parts: tourism knowledge base data, travelogues data, and tourism QA data. Additionally, we propose CTourLLM, a Qwen-based model supervised fine-tuned with Cultour, to improve the quality of information about attractions and travel planning. To evaluate the performance of CTourLLM, we proposed a human evaluation criterion named RRA (Relevance, Readability, Availability), and employed both automatic and human evaluation. The experimental results demonstrate that CTourLLM outperforms ChatGPT, achieving an improvement of 1.21 in BLEU-1 and 1.54 in Rouge-L, thereby validating the effectiveness of the response outcomes. Our proposed Cultour is accessible at https://github.com/mrweiqk/Cultour.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21141v2">Adaptive LLM Routing under Budget Constraints</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Accepted at EMNLP 2025 (findings)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing, but their varying capabilities and costs pose challenges in practical applications. LLM routing addresses this by dynamically selecting the most suitable LLM for each query/task. Previous approaches treat this as a supervised learning problem, assuming complete knowledge of optimal query-LLM pairings. However, real-world scenarios lack such comprehensive mappings and face evolving user queries. We thus propose to study LLM routing as a contextual bandit problem, enabling adaptive decision-making using bandit feedback without requiring exhaustive inference across all LLMs for all queries (in contrast to supervised routing). To address this problem, we develop a shared embedding space for queries and LLMs, where query and LLM embeddings are aligned to reflect their affinity. This space is initially learned from offline human preference data and refined through online bandit feedback. We instantiate this idea through Preference-prior Informed Linucb fOr adaptive rouTing (PILOT), a novel extension of LinUCB. To handle diverse user budgets for model routing, we introduce an online cost policy modeled as a multi-choice knapsack problem, ensuring resource-efficient routing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02659v2">Are Economists Always More Introverted? Analyzing Consistency in Persona-Assigned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Accepted to EMNLP 2025 findings
    </div>
    <details class="paper-abstract">
      Personalized Large Language Models (LLMs) are increasingly used in diverse applications, where they are assigned a specific persona - such as a happy high school teacher - to guide their responses. While prior research has examined how well LLMs adhere to predefined personas in writing style, a comprehensive analysis of consistency across different personas and task types is lacking. In this paper, we introduce a new standardized framework to analyze consistency in persona-assigned LLMs. We define consistency as the extent to which a model maintains coherent responses when assigned the same persona across different tasks and runs. Our framework evaluates personas across four different categories (happiness, occupation, personality, and political stance) spanning multiple task dimensions (survey writing, essay generation, social media post generation, single turn, and multi-turn conversations). Our findings reveal that consistency is influenced by multiple factors, including the assigned persona, stereotypes, and model design choices. Consistency also varies across tasks, increasing with more structured tasks and additional context. All code is available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07540v1">PatchSeeker: Mapping NVD Records to their Vulnerability-fixing Commits with LLM Generated Commits and Embeddings</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Software vulnerabilities pose serious risks to modern software ecosystems. While the National Vulnerability Database (NVD) is the authoritative source for cataloging these vulnerabilities, it often lacks explicit links to the corresponding Vulnerability-Fixing Commits (VFCs). VFCs encode precise code changes, enabling vulnerability localization, patch analysis, and dataset construction. Automatically mapping NVD records to their true VFCs is therefore critical. Existing approaches have limitations as they rely on sparse, often noisy commit messages and fail to capture the deep semantics in the vulnerability descriptions. To address this gap, we introduce PatchSeeker, a novel method that leverages large language models to create rich semantic links between vulnerability descriptions and their VFCs. PatchSeeker generates embeddings from NVD descriptions and enhances commit messages by synthesizing detailed summaries for those that are short or uninformative. These generated messages act as a semantic bridge, effectively closing the information gap between natural language reports and low-level code changes. Our approach PatchSeeker achieves 59.3% higher MRR and 27.9% higher Recall@10 than the best-performing baseline, Prospector, on the benchmark dataset. The extended evaluation on recent CVEs further confirms PatchSeeker's effectiveness. Ablation study shows that both the commit message generation method and the selection of backbone LLMs make a positive contribution to PatchSeeker. We also discuss limitations and open challenges to guide future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06472v2">Rethinking LLM Parametric Knowledge as Post-retrieval Confidence for Dynamic Retrieval and Reranking</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate inaccurate responses (hallucinations) when faced with questions beyond their knowledge scope. Retrieval-Augmented Generation (RAG) addresses this by leveraging external knowledge, but a critical challenge remains: determining whether retrieved contexts effectively enhance the model`s ability to answer specific queries. This challenge underscores the importance of knowledge boundary awareness, which current methods-relying on discrete labels or limited signals-fail to address adequately, as they overlook the rich information in LLMs` continuous internal hidden states. To tackle this, we propose a novel post-retrieval knowledge filtering approach. First, we construct a confidence detection model based on LLMs` internal hidden states to quantify how retrieved contexts enhance the model`s confidence. Using this model, we build a preference dataset (NQ_Rerank) to fine-tune a reranker, enabling it to prioritize contexts preferred by the downstream LLM during reranking. Additionally, we introduce Confidence-Based Dynamic Retrieval (CBDR), which adaptively triggers retrieval based on the LLM`s initial confidence in the original question, reducing knowledge conflicts and improving efficiency. Experimental results demonstrate significant improvements in accuracy for context screening and end-to-end RAG performance, along with a notable reduction in retrieval costs while maintaining competitive accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07512v1">ALLabel: Three-stage Active Learning for LLM-based Entity Recognition using Demonstration Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Many contemporary data-driven research efforts in the natural sciences, such as chemistry and materials science, require large-scale, high-performance entity recognition from scientific datasets. Large language models (LLMs) have increasingly been adopted to solve the entity recognition task, with the same trend being observed on all-spectrum NLP tasks. The prevailing entity recognition LLMs rely on fine-tuned technology, yet the fine-tuning process often incurs significant cost. To achieve a best performance-cost trade-off, we propose ALLabel, a three-stage framework designed to select the most informative and representative samples in preparing the demonstrations for LLM modeling. The annotated examples are used to construct a ground-truth retrieval corpus for LLM in-context learning. By sequentially employing three distinct active learning strategies, ALLabel consistently outperforms all baselines under the same annotation budget across three specialized domain datasets. Experimental results also demonstrate that selectively annotating only 5\%-10\% of the dataset with ALLabel can achieve performance comparable to the method annotating the entire dataset. Further analyses and ablation studies verify the effectiveness and generalizability of our proposal.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04431v3">MedGellan: LLM-Generated Medical Guidance to Support Physicians</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Medical decision-making is a critical task, where errors can result in serious, potentially life-threatening consequences. While full automation remains challenging, hybrid frameworks that combine machine intelligence with human oversight offer a practical alternative. In this paper, we present MedGellan, a lightweight, annotation-free framework that uses a Large Language Model (LLM) to generate clinical guidance from raw medical records, which is then used by a physician to predict diagnoses. MedGellan uses a Bayesian-inspired prompting strategy that respects the temporal order of clinical data. Preliminary experiments show that the guidance generated by the LLM with MedGellan improves diagnostic performance, particularly in recall and $F_1$ score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03020v3">Training LLMs to be Better Text Embedders through Bidirectional Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 accepted by EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have increasingly been explored as powerful text embedders. Existing LLM-based text embedding approaches often leverage the embedding of the final token, typically a reserved special token such as [EOS]. However, these tokens have not been intentionally trained to capture the semantics of the whole context, limiting their capacity as text embeddings, especially for retrieval and re-ranking tasks. We propose to add a new training stage before contrastive learning to enrich the semantics of the final token embedding. This stage employs bidirectional generative reconstruction tasks, namely EBQ2D (Embedding-Based Query-to-Document) and EBD2Q (Embedding-Based Document-to-Query), which interleave to anchor the [EOS] embedding and reconstruct either side of Query-Document pairs. Experimental results demonstrate that our additional training stage significantly improves LLM performance on the Massive Text Embedding Benchmark (MTEB), achieving new state-of-the-art results across different LLM base models and scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05062v2">Debatable Intelligence: Benchmarking LLM Judges via Debate Speech Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 EMNLP 2025. Code: https://github.com/noy-sternlicht/Debatable-Intelligence
    </div>
    <details class="paper-abstract">
      We introduce Debate Speech Evaluation as a novel and challenging benchmark for assessing LLM judges. Evaluating debate speeches requires a deep understanding of the speech at multiple levels, including argument strength and relevance, the coherence and organization of the speech, the appropriateness of its style and tone, and so on. This task involves a unique set of cognitive abilities that previously received limited attention in systematic LLM benchmarking. To explore such skills, we leverage a dataset of over 600 meticulously annotated debate speeches and present the first in-depth analysis of how state-of-the-art LLMs compare to human judges on this task. Our findings reveal a nuanced picture: while larger models can approximate individual human judgments in some respects, they differ substantially in their overall judgment behavior. We also investigate the ability of frontier LLMs to generate persuasive, opinionated speeches, showing that models may perform at a human level on this task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07445v1">Text2Touch: Tactile In-Hand Manipulation with LLM-Designed Reward Functions</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Accepted at CoRL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are beginning to automate reward design for dexterous manipulation. However, no prior work has considered tactile sensing, which is known to be critical for human-like dexterity. We present Text2Touch, bringing LLM-crafted rewards to the challenging task of multi-axis in-hand object rotation with real-world vision based tactile sensing in palm-up and palm-down configurations. Our prompt engineering strategy scales to over 70 environment variables, and sim-to-real distillation enables successful policy transfer to a tactile-enabled fully actuated four-fingered dexterous robot hand. Text2Touch significantly outperforms a carefully tuned human-engineered baseline, demonstrating superior rotation speed and stability while relying on reward functions that are an order of magnitude shorter and simpler. These results illustrate how LLM-designed rewards can significantly reduce the time from concept to deployable dexterous tactile skills, supporting more rapid and scalable multimodal robot learning. Project website: https://hpfield.github.io/text2touch-website
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20024v2">Using item recommendations and LLMs in marketing email titles</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Accepted to The Second Workshop on Generative AI for E-commerce (GenAIECommerce '25), held September 22, 2025, in Prague, Czech Republic. 3 figures
    </div>
    <details class="paper-abstract">
      E-commerce marketplaces make use of a number of marketing channels like emails, push notifications, etc. to reach their users and stimulate purchases. Personalized emails especially are a popular touch point for marketers to inform users of latest items in stock, especially for those who stopped visiting the marketplace. Such emails contain personalized recommendations tailored to each user's interests, enticing users to buy relevant items. A common limitation of these emails is that the primary entry point, the title of the email, tends to follow fixed templates, failing to inspire enough interest in the contents. In this work, we explore the potential of large language models (LLMs) for generating thematic titles that reflect the personalized content of the emails. We perform offline simulations and conduct online experiments on the order of millions of users, finding our techniques useful in improving the engagement between customers and our emails. We highlight key findings and learnings as we productionize the safe and automated generation of email titles for millions of users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07389v1">Talking with Oompa Loompas: A novel framework for evaluating linguistic acquisition of LLM agents</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Existing evaluation studies on linguistic competence of large language models (LLM agents) have focused primarily on vocabulary learning, morphological rule induction, syntactic generalization, pragmatic inference, and cross-linguistic transfer. However, none assess whether LLM agents can acquire a language through pattern recognition and interactive feedback, a central feature of human language acquisition. We propose a novel experimental framework in which an LLM agent is evaluated on its ability to acquire and use a newly constructed language (Tinkatongue) in conversation with a bot that understands only Tinkatongue. Our findings show that LLM agents fail to establish a conversation within 100 responses, yet they adopt distinct strategies that mirror human approaches to language learning. The results suggest a new direction for evaluation benchmarks and open pathways to model designs that learn more effectively from interactive feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17450v3">Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 To appear at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can struggle to balance gullibility to misinformation and resistance to valid corrections in persuasive dialogues, a critical challenge for reliable deployment. We introduce DuET-PD (Dual Evaluation for Trust in Persuasive Dialogues), a framework evaluating multi-turn stance-change dynamics across dual dimensions: persuasion type (corrective/misleading) and domain (knowledge via MMLU-Pro, and safety via SALAD-Bench). We find that even a state-of-the-art model like GPT-4o achieves only 27.32% accuracy in MMLU-Pro under sustained misleading persuasions. Moreover, results reveal a concerning trend of increasing sycophancy in newer open-source models. To address this, we introduce Holistic DPO, a training approach balancing positive and negative persuasion examples. Unlike prompting or resist-only training, Holistic DPO enhances both robustness to misinformation and receptiveness to corrections, improving Llama-3.1-8B-Instruct's accuracy under misleading persuasion in safety contexts from 4.21% to 76.54%. These contributions offer a pathway to developing more reliable and adaptable LLMs for multi-turn dialogue. Code is available at https://github.com/Social-AI-Studio/DuET-PD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07379v1">DuoServe-MoE: Dual-Phase Expert Prefetch and Cache Scheduling for Efficient MoE LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive performance across a wide range of deep learning tasks. Mixture of Experts (MoE) further enhances their capabilities by increasing model width through sparsely activated expert branches, which keeps inference computation efficient. However, the large number of expert weights introduces significant GPU memory pressure, especially in resource-constrained environments such as single-GPU servers. More importantly, MoE inference consists of two fundamentally different stages: a prefill stage where most experts are activated densely, and a decode stage where only a few experts are triggered sparsely. Treating these stages with a uniform scheduling strategy often leads to suboptimal latency and memory usage. To address this, we propose DuoServe-MoE, an inference serving system that explicitly separates prefill and decode stages and applies tailored expert scheduling strategies to each. In the prefill stage, DuoServe-MoE uses a two-stream CUDA pipeline that overlaps expert weight prefetching with the computation of non-MoE layers, limiting expert residency in GPU memory. In the decode stage, a lightweight layer-level predictor trained offline from activation traces is used to prefetch only the most likely activated experts, without requiring any changes to the model. Experiments on 4-bit Mixtral-8x7B and 8x22B models show that DuoServe-MoE improves end-to-end latency by 1.42 to 7.54 times while keeping peak memory usage at only 15 percent of the full model size.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07370v1">PersonaFuse: A Personality Activation-Driven Framework for Enhancing Human-LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) demonstrate remarkable capabilities across various fields. These developments have led to more direct communication between humans and LLMs in various situations, such as social companionship and psychological support. However, LLMs often exhibit limitations in emotional perception and social competence during real-world conversations. These limitations partly originate from their inability to adapt their communication style and emotional expression to different social and task contexts. In this work, we introduce PersonaFuse, a novel LLM post-training framework that enables LLMs to adapt and express different personalities for varying situations. Inspired by Trait Activation Theory and the Big Five personality model, PersonaFuse employs a Mixture-of-Expert architecture that combines persona adapters with a dynamic routing network, enabling contextual trait expression. Experimental results show that PersonaFuse substantially outperforms baseline models across multiple dimensions of social-emotional intelligence. Importantly, these gains are achieved without sacrificing general reasoning ability or model safety, which remain common limitations of direct prompting and supervised fine-tuning approaches. PersonaFuse also delivers consistent improvements in downstream human-centered applications, such as mental health counseling and review-based customer service. Finally, human preference evaluations against leading LLMs, including GPT-4o and DeepSeek, demonstrate that PersonaFuse achieves competitive response quality despite its comparatively smaller model size. These findings demonstrate that PersonaFuse~offers a theoretically grounded and practical approach for developing social-emotional enhanced LLMs, marking a significant advancement toward more human-centric AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05657v2">LM-Searcher: Cross-domain Neural Architecture Search with LLMs via Unified Numerical Encoding</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Recent progress in Large Language Models (LLMs) has opened new avenues for solving complex optimization problems, including Neural Architecture Search (NAS). However, existing LLM-driven NAS approaches rely heavily on prompt engineering and domain-specific tuning, limiting their practicality and scalability across diverse tasks. In this work, we propose LM-Searcher, a novel framework that leverages LLMs for cross-domain neural architecture optimization without the need for extensive domain-specific adaptation. Central to our approach is NCode, a universal numerical string representation for neural architectures, which enables cross-domain architecture encoding and search. We also reformulate the NAS problem as a ranking task, training LLMs to select high-performing architectures from candidate pools using instruction-tuning samples derived from a novel pruning-based subspace sampling strategy. Our curated dataset, encompassing a wide range of architecture-performance pairs, encourages robust and transferable learning. Comprehensive experiments demonstrate that LM-Searcher achieves competitive performance in both in-domain (e.g., CNNs for image classification) and out-of-domain (e.g., LoRA configurations for segmentation and generation) tasks, establishing a new paradigm for flexible and generalizable LLM-based architecture search. The datasets and models will be released at https://github.com/Ashone3/LM-Searcher.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13833v3">DistFlow: A Fully Distributed RL Framework for Scalable and Efficient LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become the pivotal post-training technique for large language model (LLM). Effectively scaling reinforcement learning is now the key to unlocking advanced reasoning capabilities and ensuring safe, goal-aligned behavior in the most powerful LLMs. Mainstream frameworks usually employ a hybrid-controller architecture where a single-controller dispatches the overall execution logic and manages overall data transfer and the multi-controller executes distributed computation. For large-scale reinforcement learning, minor load imbalances can introduce significant bottlenecks, ultimately constraining the scalability of the system. To address this limitation, we introduce DistFlow, a novel, fully distributed RL framework designed to break scaling barrier. We adopt a multi-controller paradigm that dispatches data transfer and execution tasks to all workers, which eliminates the centralized node. This allows each worker to operate independently, leading to near-linear scalability up to 1024 GPUs and dramatic efficiency gains. Furthermore, our architecture decouples resource configuration from execution logic, allowing each worker to have a unique execution flow, offering significant flexibility for rapid and cost-effective algorithmic experimentation. Extensive experiments show that DistFlow achieves excellent linear scalability and up to a 7x end-to-end throughput improvement in specific scenarios over state-of-the-art (SOTA) frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.07353v3">Benchmarking for Domain-Specific LLMs: A Case Study on Academia and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Accepted by EMNLP2025 Findings
    </div>
    <details class="paper-abstract">
      The increasing demand for domain-specific evaluation of large language models (LLMs) has led to the development of numerous benchmarks. These efforts often adhere to the principle of data scaling, relying on large corpora or extensive question-answer (QA) sets to ensure broad coverage. However, the impact of corpus and QA set design on the precision and recall of domain-specific LLM performance remains poorly understood. In this paper, we argue that data scaling is not always the optimal principle for domain-specific benchmark construction. Instead, we introduce Comp-Comp, an iterative benchmarking framework grounded in the principle of comprehensiveness and compactness. Comprehensiveness ensures semantic recall by covering the full breadth of the domain, while compactness improves precision by reducing redundancy and noise. To demonstrate the effectiveness of our approach, we present a case study conducted at a well-renowned university, resulting in the creation of PolyBench, a large-scale, high-quality academic benchmark. Although this study focuses on academia, the Comp-Comp framework is domain-agnostic and readily adaptable to a wide range of specialized fields. The source code and datasets can be accessed at https://github.com/Anya-RB-Chen/COMP-COMP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04310v2">EvoEmo: Towards Evolved Emotional Policies for LLM Agents in Multi-Turn Negotiation</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Recent research on Chain-of-Thought (CoT) reasoning in Large Language Models (LLMs) has demonstrated that agents can engage in \textit{complex}, \textit{multi-turn} negotiations, opening new avenues for agentic AI. However, existing LLM agents largely overlook the functional role of emotions in such negotiations, instead generating passive, preference-driven emotional responses that make them vulnerable to manipulation and strategic exploitation by adversarial counterparts. To address this gap, we present EvoEmo, an evolutionary reinforcement learning framework that optimizes dynamic emotional expression in negotiations. EvoEmo models emotional state transitions as a Markov Decision Process and employs population-based genetic optimization to evolve high-reward emotion policies across diverse negotiation scenarios. We further propose an evaluation framework with two baselines -- vanilla strategies and fixed-emotion strategies -- for benchmarking emotion-aware negotiation. Extensive experiments and ablation studies show that EvoEmo consistently outperforms both baselines, achieving higher success rates, higher efficiency, and increased buyer savings. This findings highlight the importance of adaptive emotional expression in enabling more effective LLM agents for multi-turn negotiation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07315v1">SafeToolBench: Pioneering a Prospective Benchmark to Evaluating Tool Utilization Safety in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 18 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited great performance in autonomously calling various tools in external environments, leading to better problem solving and task automation capabilities. However, these external tools also amplify potential risks such as financial loss or privacy leakage with ambiguous or malicious user instructions. Compared to previous studies, which mainly assess the safety awareness of LLMs after obtaining the tool execution results (i.e., retrospective evaluation), this paper focuses on prospective ways to assess the safety of LLM tool utilization, aiming to avoid irreversible harm caused by directly executing tools. To this end, we propose SafeToolBench, the first benchmark to comprehensively assess tool utilization security in a prospective manner, covering malicious user instructions and diverse practical toolsets. Additionally, we propose a novel framework, SafeInstructTool, which aims to enhance LLMs' awareness of tool utilization security from three perspectives (i.e., \textit{User Instruction, Tool Itself, and Joint Instruction-Tool}), leading to nine detailed dimensions in total. We experiment with four LLMs using different methods, revealing that existing approaches fail to capture all risks in tool utilization. In contrast, our framework significantly enhances LLMs' self-awareness, enabling a more safe and trustworthy tool utilization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08140v1">From Limited Data to Rare-event Prediction: LLM-powered Feature Engineering and Multi-model Learning in Venture Capital</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 6 pages, 3 figures
    </div>
    <details class="paper-abstract">
      This paper presents a framework for predicting rare, high-impact outcomes by integrating large language models (LLMs) with a multi-model machine learning (ML) architecture. The approach combines the predictive strength of black-box models with the interpretability required for reliable decision-making. We use LLM-powered feature engineering to extract and synthesize complex signals from unstructured data, which are then processed within a layered ensemble of models including XGBoost, Random Forest, and Linear Regression. The ensemble first produces a continuous estimate of success likelihood, which is then thresholded to produce a binary rare-event prediction. We apply this framework to the domain of Venture Capital (VC), where investors must evaluate startups with limited and noisy early-stage data. The empirical results show strong performance: the model achieves precision between 9.8X and 11.1X the random classifier baseline in three independent test subsets. Feature sensitivity analysis further reveals interpretable success drivers: the startup's category list accounts for 15.6% of predictive influence, followed by the number of founders, while education level and domain expertise contribute smaller yet consistent effects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03814v4">Recursive Training Loops in LLMs: How training data properties modulate distribution shift in generated data?</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 Accepted to EMNLP 2025 (Oral)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in the creation of online content, creating feedback loops as subsequent generations of models will be trained on this synthetic data. Such loops were shown to lead to distribution shifts - models misrepresenting the true underlying distributions of human data (also called model collapse). However, how human data properties affect such shifts remains poorly understood. In this paper, we provide the first empirical examination of the effect of such properties on the outcome of recursive training. We first confirm that using different human datasets leads to distribution shifts of different magnitudes. Through exhaustive manipulation of dataset properties combined with regression analyses, we then identify a set of properties predicting distribution shift magnitudes. Lexical diversity is found to amplify these shifts, while semantic diversity and data quality mitigate them. Furthermore, we find that these influences are highly modular: data scrapped from a given internet domain has little influence on the content generated for another domain. Finally, experiments on political bias reveal that human data properties affect whether the initial bias will be amplified or reduced. Overall, our results portray a novel view, where different parts of internet may undergo different types of distribution shift.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.23035v3">KLLM: Fast LLM Inference with K-Means Quantization</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference poses significant challenges due to its intensive memory and computation demands. Weight and activation quantization (WAQ) offers a promising solution by reducing both memory footprint and arithmetic complexity. Traditional WAQ designs rely on uniform integer quantization for hardware efficiency, but often suffer from significant model performance degradation at low precision. In contrast, K-Means quantization, a non-uniform technique, achieves higher accuracy by aligning with the Gaussian-like distributions of weights and activations in LLMs. However, two key challenges prevent the efficient deployment of K-Means-based WAQ designs for LLM inference: (1) The non-uniform structure of K-Means-quantized data precludes direct execution on low-precision compute units, necessitating dequantization and floating-point matrix multiplications (MatMuls) during inference. (2) Activation outliers hinder effective low-precision quantization. Offline thresholding methods for outlier detection degrade model performance substantially, while existing online detection techniques introduce significant runtime overhead. To address the aforementioned challenges and fully unleash the potential of K-Means-based WAQ for LLM inference, in this paper, we propose KLLM, an LLM inference accelerator for efficient execution with K-Means-quantized weights and activations. KLLM features an index-based computation scheme for efficient execution of MatMuls and nonlinear operations on K-Means-quantized data, which avoids most of the dequantization and full-precision computations. Moreover, KLLM incorporates a lightweight outlier detection engine, Orizuru, that efficiently identifies the top-$k$ largest and smallest elements in the activation data stream during online inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08093v1">Culturally transmitted color categories in LLMs reflect a learning bias toward efficient compression</a></div>
    <div class="paper-meta">
      📅 2025-09-09
    </div>
    <details class="paper-abstract">
      Converging evidence suggests that systems of semantic categories across human languages achieve near-optimal compression via the Information Bottleneck (IB) complexity-accuracy principle. Large language models (LLMs) are not trained for this objective, which raises the question: are LLMs capable of evolving efficient human-like semantic systems? To address this question, we focus on the domain of color as a key testbed of cognitive theories of categorization and replicate with LLMs (Gemini 2.0-flash and Llama 3.3-70B-Instruct) two influential human behavioral studies. First, we conduct an English color-naming study, showing that Gemini aligns well with the naming patterns of native English speakers and achieves a significantly high IB-efficiency score, while Llama exhibits an efficient but lower complexity system compared to English. Second, to test whether LLMs simply mimic patterns in their training data or actually exhibit a human-like inductive bias toward IB-efficiency, we simulate cultural evolution of pseudo color-naming systems in LLMs via iterated in-context language learning. We find that akin to humans, LLMs iteratively restructure initially random systems towards greater IB-efficiency and increased alignment with patterns observed across the world's languages. These findings demonstrate that LLMs are capable of evolving perceptually grounded, human-like semantic systems, driven by the same fundamental principle that governs semantic efficiency across human languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.15463v2">Mind the Value-Action Gap: Do LLMs Act in Alignment with Their Values?</a></div>
    <div class="paper-meta">
      📅 2025-09-09
      | 💬 EMNLP 2025 Main Paper
    </div>
    <details class="paper-abstract">
      Existing research primarily evaluates the values of LLMs by examining their stated inclinations towards specific values. However, the "Value-Action Gap," a phenomenon rooted in environmental and social psychology, reveals discrepancies between individuals' stated values and their actions in real-world contexts. To what extent do LLMs exhibit a similar gap between their stated values and their actions informed by those values? This study introduces ValueActionLens, an evaluation framework to assess the alignment between LLMs' stated values and their value-informed actions. The framework encompasses the generation of a dataset comprising 14.8k value-informed actions across twelve cultures and eleven social topics, and two tasks to evaluate how well LLMs' stated value inclinations and value-informed actions align across three different alignment measures. Extensive experiments reveal that the alignment between LLMs' stated values and actions is sub-optimal, varying significantly across scenarios and models. Analysis of misaligned results identifies potential harms from certain value-action gaps. To predict the value-action gaps, we also uncover that leveraging reasoned explanations improves performance. These findings underscore the risks of relying solely on the LLMs' stated values to predict their behaviors and emphasize the importance of context-aware evaluations of LLM values and value-action gaps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06948v1">Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has proven effective in incentivizing the reasoning abilities of large language models (LLMs), but suffers from severe efficiency challenges due to its trial-and-error nature. While the common practice employs supervised fine-tuning (SFT) as a warm-up stage for RL, this decoupled two-stage approach limits interaction between SFT and RL, thereby constraining overall effectiveness. This study introduces a novel method for learning reasoning models that employs bilevel optimization to facilitate better cooperation between these training paradigms. By conditioning the SFT objective on the optimal RL policy, our approach enables SFT to meta-learn how to guide RL's optimization process. During training, the lower level performs RL updates while simultaneously receiving SFT supervision, and the upper level explicitly maximizes the cooperative gain-the performance advantage of joint SFT-RL training over RL alone. Empirical evaluations on five reasoning benchmarks demonstrate that our method consistently outperforms baselines and achieves a better balance between effectiveness and efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06941v1">Outcome-based Exploration for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 26 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has emerged as a powerful method for improving the reasoning abilities of large language models (LLMs). Outcome-based RL, which rewards policies solely for the correctness of the final answer, yields substantial accuracy gains but also induces a systematic loss in generation diversity. This collapse undermines real-world performance, where diversity is critical for test-time scaling. We analyze this phenomenon by viewing RL post-training as a sampling process and show that, strikingly, RL can reduce effective diversity even on the training set relative to the base model. Our study highlights two central findings: (i) a transfer of diversity degradation, where reduced diversity on solved problems propagates to unsolved ones, and (ii) the tractability of the outcome space, since reasoning tasks admit only a limited set of distinct answers. Motivated by these insights, we propose outcome-based exploration, which assigns exploration bonuses according to final outcomes. We introduce two complementary algorithms: historical exploration, which encourages rarely observed answers via UCB-style bonuses, and batch exploration, which penalizes within-batch repetition to promote test-time diversity. Experiments on standard competition math with Llama and Qwen models demonstrate that both methods improve accuracy while mitigating diversity collapse. On the theoretical side, we formalize the benefit of outcome-based exploration through a new model of outcome-based bandits. Together, these contributions chart a practical path toward RL methods that enhance reasoning without sacrificing the diversity essential for scalable deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06920v1">An Ethically Grounded LLM-Based Approach to Insider Threat Synthesis and Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 6 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Insider threats are a growing organizational problem due to the complexity of identifying their technical and behavioral elements. A large research body is dedicated to the study of insider threats from technological, psychological, and educational perspectives. However, research in this domain has been generally dependent on datasets that are static and limited access which restricts the development of adaptive detection models. This study introduces a novel, ethically grounded approach that uses the large language model (LLM) Claude Sonnet 3.7 to dynamically synthesize syslog messages, some of which contain indicators of insider threat scenarios. The messages reflect real-world data distributions by being highly imbalanced (1% insider threats). The syslogs were analyzed for insider threats by both Claude Sonnet 3.7 and GPT-4o, with their performance evaluated through statistical metrics including precision, recall, MCC, and ROC AUC. Sonnet 3.7 consistently outperformed GPT-4o across nearly all metrics, particularly in reducing false alarms and improving detection accuracy. The results show strong promise for the use of LLMs in synthetic dataset generation and insider threat detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03377v2">Amplifying Effective CXL Memory Bandwidth for LLM Inference via Transparent Near-Data Processing</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large language model (LLM) inference is bottlenecked by the limited bandwidth of CXL-based memory used for capacity expansion. We introduce CXL-NDP, a transparent near-data processing architecture that amplifies effective CXL bandwidth without requiring changes to the CXL.mem interface or AI models. CXL-NDP integrates a precision-scalable bit-plane layout for dynamic quantization with transparent lossless compression of weights and KV caches directly within the CXL device. In end-to-end serving, CXL-NDP improves throughput by 43%, extends the maximum context length by 87%, and reduces the KV cache footprint by 46.9% without accuracy loss. Hardware synthesis confirms its practicality with a modest silicon footprint, lowering the barrier for adopting efficient, scalable CXL-based memory in generative AI infrastructure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06902v1">Proof-Carrying Numbers (PCN): A Protocol for Trustworthy Numeric Answers from LLMs via Claim Verification</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) as stochastic systems may generate numbers that deviate from available data, a failure known as \emph{numeric hallucination}. Existing safeguards -- retrieval-augmented generation, citations, and uncertainty estimation -- improve transparency but cannot guarantee fidelity: fabricated or misquoted values may still be displayed as if correct. We propose \textbf{Proof-Carrying Numbers (PCN)}, a presentation-layer protocol that enforces numeric fidelity through mechanical verification. Under PCN, numeric spans are emitted as \emph{claim-bound tokens} tied to structured claims, and a verifier checks each token under a declared policy (e.g., exact equality, rounding, aliases, or tolerance with qualifiers). Crucially, PCN places verification in the \emph{renderer}, not the model: only claim-checked numbers are marked as verified, and all others default to unverified. This separation prevents spoofing and guarantees fail-closed behavior. We formalize PCN and prove soundness, completeness under honest tokens, fail-closed behavior, and monotonicity under policy refinement. PCN is lightweight and model-agnostic, integrates seamlessly into existing applications, and can be extended with cryptographic commitments. By enforcing verification as a mandatory step before display, PCN establishes a simple contract for numerically sensitive settings: \emph{trust is earned only by proof}, while the absence of a mark communicates uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20521v2">Project Riley: Multimodal Multi-Agent LLM Collaboration with Emotional Reasoning and Voting</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 28 pages, 5 figures. Submitted for review to Information Fusion
    </div>
    <details class="paper-abstract">
      This paper presents Project Riley, a novel multimodal and multi-model conversational AI architecture oriented towards the simulation of reasoning influenced by emotional states. Drawing inspiration from Pixar's Inside Out, the system comprises five distinct emotional agents - Joy, Sadness, Fear, Anger, and Disgust - that engage in structured multi-round dialogues to generate, criticise, and iteratively refine responses. A final reasoning mechanism synthesises the contributions of these agents into a coherent output that either reflects the dominant emotion or integrates multiple perspectives. The architecture incorporates both textual and visual large language models (LLMs), alongside advanced reasoning and self-refinement processes. A functional prototype was deployed locally in an offline environment, optimised for emotional expressiveness and computational efficiency. From this initial prototype, another one emerged, called Armando, which was developed for use in emergency contexts, delivering emotionally calibrated and factually accurate information through the integration of Retrieval-Augmented Generation (RAG) and cumulative context tracking. The Project Riley prototype was evaluated through user testing, in which participants interacted with the chatbot and completed a structured questionnaire assessing three dimensions: Emotional Appropriateness, Clarity and Utility, and Naturalness and Human-likeness. The results indicate strong performance in structured scenarios, particularly with respect to emotional alignment and communicative clarity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02019v2">ChatCFD: An LLM-Driven Agent for End-to-End CFD Automation with Domain-Specific Structured Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Computational Fluid Dynamics (CFD) is essential for advancing scientific and engineering fields but is hindered by operational complexity, high expertise requirements, and limited accessibility. This paper introduces ChatCFD, an automated agent system for OpenFOAM simulations that processes multi-modal inputs (e.g., research papers, meshes) via an interactive interface, leveraging DeepSeek-R1 and DeepSeek-V3 large language models, a multi-agent architecture, and OpenFOAM knowledge. Its four-stage pipeline (Knowledge Base Construction, User Input Processing, Case File Generation, and Execution and Error Reflection) enables iterative trial-reflection-refinement for intricate setups, supporting diverse physical models and external meshes. Validation on 205 benchmark tutorial cases, 110 perturbed variants, and 2 literature-derived cases shows ChatCFD's 82.1 percent operational success rate on basic cases, outperforming MetaOpenFOAM (6.2 percent) and Foam-Agent (42.3 percent), and 60-80 percent on literature-derived complex cases. Turbulence model studies show a 40 percent success rate for common models versus 10 percent for rare ones like RNG k-epsilon. Physics coupling analyses reveal higher resource demands for multi-physics-coupled cases, while LLM bias toward simpler setups introduces persistent errors, such as dimensional inconsistency. Ablation studies highlight the efficacy of RAG-based modules and reflection mechanisms. By automating hypothesis testing and parameter exploration, ChatCFD accelerates scientific discovery in fluid mechanics and engineering, addressing LLM limitations through structured design and showing strong potential as a modular component in MCP-based agent networks for collaborative multi-agent systems, paving the way for scalable AI-driven CFD innovation. The code for ChatCFD is available at https://github.com/ConMoo/ChatCFD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.16482v2">Self-Alignment: Improving Alignment of Cultural Values in LLMs via In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Improving the alignment of Large Language Models (LLMs) with respect to the cultural values that they encode has become an increasingly important topic. In this work, we study whether we can exploit existing knowledge about cultural values at inference time to adjust model responses to cultural value probes. We present a simple and inexpensive method that uses a combination of in-context learning (ICL) and human survey data, and show that we can improve the alignment to cultural values across 5 models that include both English-centric and multilingual LLMs. Importantly, we show that our method could prove useful in test languages other than English and can improve alignment to the cultural values that correspond to a range of culturally diverse countries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06822v1">RAFFLES: Reasoning-based Attribution of Faults for LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      We have reached a critical roadblock in the development and enhancement of long-horizon, multi-component LLM agentic systems: it is incredibly tricky to identify where these systems break down and why. Evaluation capabilities that currently exist today (e.g., single pass LLM-as-a-judge) are limited in that they often focus on individual metrics or capabilities, end-to-end outcomes, and are narrowly grounded on the preferences of humans. We argue that to match the agentic capabilities, evaluation frameworks must also be able to reason, probe, iterate, and understand the complex logic passing through these systems over long horizons. In this paper, we present RAFFLES - an evaluation architecture that incorporates reasoning and iterative refinement. Specifically, RAFFLES operates as an iterative, multi-component pipeline, using a central Judge to systematically investigate faults and a set of specialized Evaluators to assess not only the system's components but also the quality of the reasoning by the Judge itself, thereby building a history of hypotheses. We tested RAFFLES against several baselines on the Who&When dataset, a benchmark designed to diagnose the "who" (agent) and "when" (step) of a system's failure. RAFFLES outperforms these baselines, achieving an agent-step fault pair accuracy of over 43% on the Algorithmically-Generated dataset (a substantial increase from the previously published best of 16.6%) and over 20% on the Hand-Crafted dataset (surpassing the previously published best of 8.8%). These results demonstrate a key step towards introducing automated fault detection for autonomous systems over labor-intensive manual human review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06809v1">Saturation-Driven Dataset Generation for LLM Mathematical Reasoning in the TPTP Ecosystem</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      The scarcity of high-quality, logically sound data is a critical bottleneck for advancing the mathematical reasoning of Large Language Models (LLMs). Our work confronts this challenge by turning decades of automated theorem proving research into a scalable data engine. Rather than relying on error-prone LLMs or complex proof-assistant syntax like Lean and Isabelle, our framework leverages E-prover's saturation capabilities on the vast TPTP axiom library to derive a massive, guaranteed-valid corpus of theorems. Our pipeline is principled and simple: saturate axioms, filter for "interesting" theorems, and generate tasks. With no LLMs in the loop, we eliminate factual errors by construction. This purely symbolic data is then transformed into three difficulty-controlled challenges: entailment verification, premise selection, and proof reconstruction. Our zero-shot experiments on frontier models reveal a clear weakness: performance collapses on tasks requiring deep, structural reasoning. Our framework provides both the diagnostic tool to measure this gap and a scalable source of symbolic training data to address it. We make the code and data publicly available. https://github.com/sileod/reasoning_core https://hf.co/datasets/reasoning-core/rc1
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06770v1">Another Turn, Better Output? A Turn-Wise Analysis of Iterative LLM Prompting</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now used in multi-turn workflows, but we still lack a clear way to measure when iteration helps and when it hurts. We present an evaluation framework for iterative refinement that spans ideation, code, and math. Our protocol runs controlled 12-turn conversations per task, utilizing a variety of prompts ranging from vague ``improve it'' feedback to targeted steering, and logs per-turn outputs. We score outcomes with domain-appropriate checks (unit tests for code; answer-equivalence plus reasoning-soundness for math; originality and feasibility for ideation) and track turn-level behavior with three families of metrics: semantic movement across turns, turn-to-turn change, and output size growth. Across models and tasks, gains are domain-dependent: they arrive early in ideas and code, but in math late turns matter when guided by elaboration. After the first few turns, vague feedback often plateaus or reverses correctness, while targeted prompts reliably shift the intended quality axis (novelty vs. feasibility in ideation; speed vs. readability in code; in math, elaboration outperforms exploration and drives late-turn gains). We also observe consistent domain patterns: ideation moves more in meaning across turns, code tends to grow in size with little semantic change, and math starts fixed but can break that path with late, elaborative iteration.Together, the framework and metrics make iteration measurable and comparable across models, and signal when to steer, stop, or switch strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06734v1">Intelligent Manufacturing Support: Specialized LLMs for Composite Material Processing and Equipment Operation</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Engineering educational curriculum and standards cover many material and manufacturing options. However, engineers and designers are often unfamiliar with certain composite materials or manufacturing techniques. Large language models (LLMs) could potentially bridge the gap. Their capacity to store and retrieve data from large databases provides them with a breadth of knowledge across disciplines. However, their generalized knowledge base can lack targeted, industry-specific knowledge. To this end, we present two LLM-based applications based on the GPT-4 architecture: (1) The Composites Guide: a system that provides expert knowledge on composites material and connects users with research and industry professionals who can provide additional support and (2) The Equipment Assistant: a system that provides guidance for manufacturing tool operation and material characterization. By combining the knowledge of general AI models with industry-specific knowledge, both applications are intended to provide more meaningful information for engineers. In this paper, we discuss the development of the applications and evaluate it through a benchmark and two informal user studies. The benchmark analysis uses the Rouge and Bertscore metrics to evaluate our model performance against GPT-4o. The results show that GPT-4o and the proposed models perform similarly or better on the ROUGE and BERTScore metrics. The two user studies supplement this quantitative evaluation by asking experts to provide qualitative and open-ended feedback about our model performance on a set of domain-specific questions. The results of both studies highlight a potential for more detailed and specific responses with the Composites Guide and the Equipment Assistant.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07642v2">TreeReview: A Dynamic Tree of Questions Framework for Deep and Efficient LLM-based Scientific Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 Accepted to EMNLP2025 Main
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) have shown significant potential in assisting peer review, current methods often struggle to generate thorough and insightful reviews while maintaining efficiency. In this paper, we propose TreeReview, a novel framework that models paper review as a hierarchical and bidirectional question-answering process. TreeReview first constructs a tree of review questions by recursively decomposing high-level questions into fine-grained sub-questions and then resolves the question tree by iteratively aggregating answers from leaf to root to get the final review. Crucially, we incorporate a dynamic question expansion mechanism to enable deeper probing by generating follow-up questions when needed. We construct a benchmark derived from ICLR and NeurIPS venues to evaluate our method on full review generation and actionable feedback comments generation tasks. Experimental results of both LLM-based and human evaluation show that TreeReview outperforms strong baselines in providing comprehensive, in-depth, and expert-aligned review feedback, while reducing LLM token usage by up to 80% compared to computationally intensive approaches. Our code and benchmark dataset are available at https://github.com/YuanChang98/tree-review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19576v2">ReST-RL: Achieving Accurate Code Reasoning of LLMs with Optimized Self-Training and Decoding</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 21 pages, 4 figures
    </div>
    <details class="paper-abstract">
      With respect to improving the reasoning accuracy of LLMs, the representative reinforcement learning (RL) method GRPO faces failure due to insignificant reward variance, while verification methods based on process reward models (PRMs) suffer from difficulties with training data acquisition and verification effectiveness. To tackle these problems, this paper introduces ReST-RL, a unified LLM RL paradigm that significantly improves LLM's code reasoning ability by combining an improved GRPO algorithm with a meticulously designed test time decoding method assisted by a value model (VM). As the first stage of policy reinforcement, ReST-GRPO adopts an optimized ReST algorithm to filter and assemble high-value training data, increasing the reward variance of GRPO sampling, thus improving the effectiveness and efficiency of training. After the basic reasoning ability of LLM policy has been improved, we further propose a test time decoding optimization method called VM-MCTS. Through Monte-Carlo Tree Search (MCTS), we collect accurate value targets with no annotation required, on which VM training is based. When decoding, the VM is deployed by an adapted MCTS algorithm to provide precise process signals as well as verification scores, assisting the LLM policy to achieve high reasoning accuracy. We conduct extensive experiments on coding problems to verify the validity of the proposed RL paradigm. Upon comparison, our approach significantly outperforms other reinforcement training baselines (e.g., naive GRPO and ReST-DPO), as well as decoding and verification baselines (e.g., PRM-BoN and ORM-MCTS) on well-known coding benchmarks of various levels (e.g., APPS, BigCodeBench, and HumanEval), indicating its power to strengthen the reasoning ability of LLM policies. Codes for our project can be found at https://github.com/THUDM/ReST-RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00719v3">Dynamically Adaptive Reasoning via LLM-Guided MCTS for Efficient and Context-Aware KGQA</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Knowledge Graph Question Answering (KGQA) aims to interpret natural language queries and perform structured reasoning over knowledge graphs by leveraging their relational and semantic structures to retrieve accurate answers. Recent KGQA methods primarily follow either retrieve-then-reason paradigm, relying on GNNs or heuristic rules for static paths extraction, or dynamic path generation strategies that use large language models (LLMs) with prompting to jointly perform retrieval and reasoning. However, the former suffers from limited adaptability due to static path extraction and lack of contextual refinement, while the latter incurs high computational costs and struggles with accurate path evaluation due to reliance on fixed scoring functions and extensive LLM calls. To address these issues, this paper proposes Dynamically Adaptive MCTS-based Reasoning (DAMR), a novel framework that integrates symbolic search with adaptive path evaluation for efficient and context-aware KGQA. DAMR employs a Monte Carlo Tree Search (MCTS) backbone guided by an LLM-based planner, which selects top-$k$ relevant relations at each step to reduce search space. To improve path evaluation accuracy, we introduce a lightweight Transformer-based scorer that performs context-aware plausibility estimation by jointly encoding the question and relation sequence through cross-attention, enabling the model to capture fine-grained semantic shifts during multi-hop reasoning. Furthermore, to alleviate the scarcity of high-quality supervision, DAMR incorporates a dynamic pseudo-path refinement mechanism that periodically generates training signals from partial paths explored during search, allowing the scorer to continuously adapt to the evolving distribution of reasoning trajectories. Extensive experiments on multiple KGQA benchmarks show that DAMR significantly outperforms state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06595v1">LLMs in Cybersecurity: Friend or Foe in the Human Decision Loop?</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming human decision-making by acting as cognitive collaborators. Yet, this promise comes with a paradox: while LLMs can improve accuracy, they may also erode independent reasoning, promote over-reliance and homogenize decisions. In this paper, we investigate how LLMs shape human judgment in security-critical contexts. Through two exploratory focus groups (unaided and LLM-supported), we assess decision accuracy, behavioral resilience and reliance dynamics. Our findings reveal that while LLMs enhance accuracy and consistency in routine decisions, they can inadvertently reduce cognitive diversity and improve automation bias, which is especially the case among users with lower resilience. In contrast, high-resilience individuals leverage LLMs more effectively, suggesting that cognitive traits mediate AI benefit.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06586v1">Simulating Dispute Mediation with LLM-Based Agents for Legal Research</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Legal dispute mediation plays a crucial role in resolving civil disputes, yet its empirical study is limited by privacy constraints and complex multivariate interactions. To address this limitation, we present AgentMediation, the first LLM-based agent framework for simulating dispute mediation. It simulates realistic mediation processes grounded in real-world disputes and enables controlled experimentation on key variables such as disputant strategies, dispute causes, and mediator expertise. Our empirical analysis reveals patterns consistent with sociological theories, including Group Polarization and Surface-level Consensus. As a comprehensive and extensible platform, AgentMediation paves the way for deeper integration of social science and AI in legal research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04537v2">Emergent Social Dynamics of LLM Agents in the El Farol Bar Problem</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      We investigate the emergent social dynamics of Large Language Model (LLM) agents in a spatially extended El Farol Bar problem, observing how they autonomously navigate this classic social dilemma. As a result, the LLM agents generated a spontaneous motivation to go to the bar and changed their decision making by becoming a collective. We also observed that the LLM agents did not solve the problem completely, but rather behaved more like humans. These findings reveal a complex interplay between external incentives (prompt-specified constraints such as the 60% threshold) and internal incentives (culturally-encoded social preferences derived from pre-training), demonstrating that LLM agents naturally balance formal game-theoretic rationality with social motivations that characterize human behavior. These findings suggest that a new model of group decision making, which could not be handled in the previous game-theoretic problem setting, can be realized by LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.03646v2">Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has proven highly effective at enhancing the complex reasoning abilities of Large Language Models (LLMs), yet underlying mechanisms driving this success remain largely opaque. Our analysis reveals that puzzling phenomena like ``aha moments", ``length-scaling'' and entropy dynamics are not disparate occurrences but hallmarks of an emergent reasoning hierarchy, akin to the separation of high-level strategic planning from low-level procedural execution in human cognition. We uncover a compelling two-phase dynamic: initially, a model is constrained by procedural correctness and must improve its low-level skills. The learning bottleneck then decisively shifts, with performance gains being driven by the exploration and mastery of high-level strategic planning. This insight exposes a core inefficiency in prevailing RL algorithms like GRPO, which apply optimization pressure agnostically and dilute the learning signal across all tokens. To address this, we propose HIerarchy-Aware Credit Assignment (HICRA), an algorithm that concentrates optimization efforts on high-impact planning tokens. HICRA significantly outperforms strong baselines, demonstrating that focusing on this strategic bottleneck is key to unlocking advanced reasoning. Furthermore, we validate semantic entropy as a superior compass for measuring strategic exploration over misleading metrics such as token-level entropy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12172v2">Navigating the Labyrinth: Evaluating LLMs' Ability to Reason About Search Problems</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently achieved impressive performance in math and reasoning benchmarks. However, they often struggle with logic problems and puzzles that are relatively easy for humans. To further investigate this, we introduce a new benchmark, SearchBench, which contains 11 unique search problems, each equipped with automated pipelines to generate an arbitrary number of instances and analyze the feasibility, correctness, and optimality of LLM-generated solutions. We show that using language-only reasoning, even the most advanced LLMs fail to solve SearchBench end-to-end, e.g., OpenAI's frontier models GPT4 and o1-preview solve only 1.4% and 18.6% of SearchBench problems, respectively. The reason is that SearchBench problems require considering multiple pathways to the solution and performing backtracking, posing a significant challenge to auto-regressive models. Instructing LLMs to generate code that solves the problem helps, but only slightly, e.g., GPT4's performance rises to 11.7%. Interestingly, we show that the current strongest baseline on SearchBench is obtained using in-context learning with A* algorithm implementations. We further show that this baseline can be further enhanced via a Multi-Stage-Multi-Try inference method, raising GPT4's performance above 57%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06524v1">LAMDAS: LLM as an Implicit Classifier for Domain-specific Data Selection</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      Adapting large language models (LLMs) to specific domains often faces a critical bottleneck: the scarcity of high-quality, human-curated data. While large volumes of unchecked data are readily available, indiscriminately using them for fine-tuning risks introducing noise and degrading performance. Strategic data selection is thus crucial, requiring a method that is both accurate and efficient. Existing approaches, categorized as similarity-based and direct optimization methods, struggle to simultaneously achieve these goals. In this paper, we introduce LAMDAS (LLM As an iMplicit classifier for domain-specific DAta Selection), a novel approach that leverages the pre-trained LLM itself as an implicit classifier, thereby bypassing explicit feature engineering and computationally intensive optimization process. LAMDAS reframes data selection as a one-class classification problem, identifying candidate data that "belongs" to the target domain defined by a small reference dataset. Extensive experimental results demonstrate that LAMDAS not only exceeds the performance of full-data training using a fraction of the data but also outperforms nine state-of-the-art (SOTA) baselines under various scenarios. Furthermore, LAMDAS achieves the most compelling balance between performance gains and computational efficiency compared to all evaluated baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15868v2">CARFT: Boosting LLM Reasoning via Contrastive Learning with Annotated Chain-of-Thought-based Reinforced Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-08
      | 💬 14 pages, to appear in EMNLP25
    </div>
    <details class="paper-abstract">
      Reasoning capability plays a significantly critical role in the the broad applications of Large Language Models (LLMs). To enhance the reasoning performance of LLMs, diverse Reinforcement Learning (RL)-based fine-tuning approaches have been proposed to address the limited generalization capability of LLMs trained solely via Supervised Fine-Tuning (SFT). Despite their effectiveness, two major limitations hinder the advancement of LLMs. First, vanilla RL-based approaches ignore annotated Chain-of-Thought (CoT) and incorporate unstable reasoning path sampling, which typically results in model collapse, unstable training process, and suboptimal performance. Second, existing SFT approaches generally overemphasize the annotated CoT, potentially leading to performance degradation due to insufficient exploitation of potential CoT. In this paper, we propose a Contrastive learning with annotated CoT-based Reinforced Fine-Tuning approach, i.e., \TheName{}, to enhance the reasoning performance of LLMs while addressing the aforementioned limitations. Specifically, we propose learning a representation for each CoT. Based on this representation, we design novel contrastive signals to guide the fine-tuning process. Our approach not only fully exploits the available annotated CoT but also stabilizes the fine-tuning procedure by incorporating an additional unsupervised learning signal. We conduct comprehensive experiments and in-depth analysis with three baseline approaches, two foundation models, and two datasets to demonstrate significant advantages of \TheName{} in terms of robustness, performance (up to 10.15\%), and efficiency (up to 30.62\%). Code is available at https://github.com/WNQzhu/CARFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06504v1">When Code Crosses Borders: A Security-Centric Evaluation of LLM-based Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-09-08
    </div>
    <details class="paper-abstract">
      With the growing demand for cross-language codebase migration, evaluating LLMs' security implications in translation tasks has become critical. Existing evaluations primarily focus on syntactic or functional correctness at the function level, neglecting the critical dimension of security. To enable security evaluation, we construct STED (Security-centric Translation Evaluation Dataset), the first dataset specifically designed for evaluating the security implications of LLM-based code translation. It comprises 720 security-related code samples across five programming languages and nine high-impact CWE categories, sourced from CVE/NVD and manually verified for translation tasks. Our evaluation framework consists of two independent assessment modules: (1) rigorous evaluation by security researchers, and (2) automated analysis via LLM-as-a-judge. Together they evaluate three critical aspects: functional correctness, vulnerability preservation, and vulnerability introduction rates. Our large-scale evaluation of five state-of-the-art LLMs across 6,000 translation instances reveals significant security degradation, with 28.6-45% of translations introducing new vulnerabilities--particularly for web-related flaws like input validation, where LLMs show consistent weaknesses. Furthermore, we develop a Retrieval-Augmented Generation (RAG)-based mitigation strategy that reduces translation-induced vulnerabilities by 32.8%, showing the potential of knowledge-enhanced prompting.
    </details>
</div>
