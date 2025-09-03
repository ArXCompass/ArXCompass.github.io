# llm - 2025_08

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
- Part 13

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15170v3">From LLMs to MLLMs to Agents: A Survey of Emerging Paradigms in Jailbreak Attacks and Defenses within LLM Ecosystem</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly evolving from single-modal systems to multimodal LLMs and intelligent agents, significantly expanding their capabilities while introducing increasingly severe security risks. This paper presents a systematic survey of the growing complexity of jailbreak attacks and corresponding defense mechanisms within the expanding LLM ecosystem. We first trace the developmental trajectory from LLMs to MLLMs and Agents, highlighting the core security challenges emerging at each stage. Next, we categorize mainstream jailbreak techniques from both the attack impact and visibility perspectives, and provide a comprehensive analysis of representative attack methods, related datasets, and evaluation metrics. On the defense side, we organize existing strategies based on response timing and technical approach, offering a structured understanding of their applicability and implementation. Furthermore, we identify key limitations in existing surveys, such as insufficient attention to agent-specific security issues, the absence of a clear taxonomy for hybrid jailbreak methods, a lack of detailed analysis of experimental setups, and outdated coverage of recent advancements. To address these limitations, we provide an updated synthesis of recent work and outline future research directions in areas such as dataset construction, evaluation framework optimization, and strategy generalization. Our study seeks to enhance the understanding of jailbreak mechanisms and facilitate the advancement of more resilient and adaptive defense strategies in the context of ever more capable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00507v1">Court of LLMs: Evidence-Augmented Generation via Multi-LLM Collaboration for Text-Attributed Graph Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 Accepted by ACM Multimedia 2025 (MM '25)
    </div>
    <details class="paper-abstract">
      The natural combination of intricate topological structures and rich textual information in text-attributed graphs (TAGs) opens up a novel perspective for graph anomaly detection (GAD). However, existing GAD methods primarily focus on designing complex optimization objectives within the graph domain, overlooking the complementary value of the textual modality, whose features are often encoded by shallow embedding techniques, such as bag-of-words or skip-gram, so that semantic context related to anomalies may be missed. To unleash the enormous potential of textual modality, large language models (LLMs) have emerged as promising alternatives due to their strong semantic understanding and reasoning capabilities. Nevertheless, their application to TAG anomaly detection remains nascent, and they struggle to encode high-order structural information inherent in graphs due to input length constraints. For high-quality anomaly detection in TAGs, we propose CoLL, a novel framework that combines LLMs and graph neural networks (GNNs) to leverage their complementary strengths. CoLL employs multi-LLM collaboration for evidence-augmented generation to capture anomaly-relevant contexts while delivering human-readable rationales for detected anomalies. Moreover, CoLL integrates a GNN equipped with a gating mechanism to adaptively fuse textual features with evidence while preserving high-order topological information. Extensive experiments demonstrate the superiority of CoLL, achieving an average improvement of 13.37% in AP. This study opens a new avenue for incorporating LLMs in advancing GAD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00500v1">Pro2Guard: Proactive Runtime Enforcement of LLM Agent Safety via Probabilistic Model Checking</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents exhibit powerful autonomous capabilities across domains such as robotics, virtual assistants, and web automation. However, their stochastic behavior introduces significant safety risks that are difficult to anticipate. Existing rule-based enforcement systems, such as AgentSpec, focus on developing reactive safety rules, which typically respond only when unsafe behavior is imminent or has already occurred. These systems lack foresight and struggle with long-horizon dependencies and distribution shifts. To address these limitations, we propose Pro2Guard, a proactive runtime enforcement framework grounded in probabilistic reachability analysis. Pro2Guard abstracts agent behaviors into symbolic states and learns a Discrete-Time Markov Chain (DTMC) from execution traces. At runtime, it anticipates future risks by estimating the probability of reaching unsafe states, triggering interventions before violations occur when the predicted risk exceeds a user-defined threshold. By incorporating semantic validity checks and leveraging PAC bounds, Pro2Guard ensures statistical reliability while approximating the underlying ground-truth model. We evaluate Pro2Guard extensively across two safety-critical domains: embodied household agents and autonomous vehicles. In embodied agent tasks, Pro2Guard enforces safety early on up to 93.6% of unsafe tasks using low thresholds, while configurable modes (e.g., reflect) allow balancing safety with task success, maintaining up to 80.4% task completion. In autonomous driving scenarios, Pro2Guard achieves 100% prediction of traffic law violations and collisions, anticipating risks up to 38.66 seconds ahead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02962v4">RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, while LLMs remain prone to generating hallucinated or outdated responses due to their static internal knowledge. Recent advancements in Retrieval-Augmented Generation (RAG) methods have aimed to enhance models' search and reasoning capabilities through reinforcement learning (RL). Although these methods demonstrate promising results, they face challenges in training stability and encounter issues such as substantial inference time and restricted capabilities due to reliance on single-query mode. In this paper, we propose RAG-R1, a novel training framework designed to enable LLMs to adaptively leverage internal and external knowledge during the reasoning process. We further expand the generation and retrieval processes within the framework from single-query mode to multi-query parallelism, with the aim of reducing inference time and enhancing the model's capabilities. Extensive experiments on seven question-answering benchmarks demonstrate that our method outperforms the strongest baseline by up to 13.2% and decreases inference time by 11.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00459v1">Thinking Machines: Mathematical Reasoning in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable abilities in structured reasoning and symbolic tasks, with coding emerging as a particular area of strength. This success has sparked growing interest in applying LLMs to mathematics, both in informal problem-solving and formal theorem proving. However, progress in formal mathematics has proven to be significantly more difficult, despite surface-level similarities between programming and proof construction. This discrepancy raises important questions about how LLMs ``reason'', how they are supervised, and whether they internally track a notion of computational or deductive state. In this article, we address the state-of-the-art of the discipline, focusing on recent models and benchmarks, and explore three central issues at the intersection of machine learning and mathematical cognition: (i) the trade-offs between formal and informal mathematics as training domains; (ii) the deeper reasons why proof generation remains more brittle than code synthesis; (iii) and the question of whether LLMs represent, or merely mimic, a notion of evolving logical state. Our goal is not to draw hard boundaries, but to identify where the current limits lie, and how they might be extended.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16974v2">Leveraging Synthetic Data for Question Answering with Multilingual LLMs in the Agricultural Domain</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 16 pages, 9 tables, Appendix A-L
    </div>
    <details class="paper-abstract">
      Enabling farmers to access accurate agriculture-related information in their native languages in a timely manner is crucial for the success of the agriculture field. Publicly available general-purpose Large Language Models (LLMs) typically offer generic agriculture advisories, lacking precision in local and multilingual contexts. Our study addresses this limitation by generating multilingual (English, Hindi, Punjabi) synthetic datasets from agriculture-specific documents from India and fine-tuning LLMs for the task of question answering (QA). Evaluation on human-created datasets demonstrates significant improvements in factuality, relevance, and agricultural consensus for the fine-tuned LLMs compared to the baseline counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00419v1">Loop Invariant Generation: A Hybrid Framework of Reasoning optimised LLMs and SMT Solvers</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Loop invariants are essential for proving the correctness of programs with loops. Developing loop invariants is challenging, and fully automatic synthesis cannot be guaranteed for arbitrary programs. Some approaches have been proposed to synthesize loop invariants using symbolic techniques and more recently using neural approaches. These approaches are able to correctly synthesize loop invariants only for subsets of standard benchmarks. In this work, we investigate whether modern, reasoning-optimized large language models can do better. We integrate OpenAI's O1, O1-mini, and O3-mini into a tightly coupled generate-and-check pipeline with the Z3 SMT solver, using solver counterexamples to iteratively guide invariant refinement. We use Code2Inv benchmark, which provides C programs along with their formal preconditions and postconditions. On this benchmark of 133 tasks, our framework achieves 100% coverage (133 out of 133), outperforming the previous best of 107 out of 133, while requiring only 1-2 model proposals per instance and 14-55 seconds of wall-clock time. These results demonstrate that LLMs possess latent logical reasoning capabilities which can help automate loop invariant synthesis. While our experiments target C-specific programs, this approach should be generalizable to other imperative languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00408v1">Benchmarking LLMs for Unit Test Generation from Real-World Functions</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have shown great promise in automating unit test generation, significantly reducing the manual effort required by developers. To effectively evaluate the capabilities of LLMs in this domain, it is crucial to have a well-designed benchmark that accurately reflects real-world scenarios and mitigates common pitfalls. Existing LLM test generation benchmarks are limited by two critical drawbacks: data contamination and structurally simple function code. As a result, we often cannot rely on the validity of scientific conclusions drawn from empirical studies using these limited benchmarks. The empirical evidence presented may be biased due to contamination and may fail to generalize beyond toy programs due to structural simplicity. To address these problems, we introduce ULT (UnLeakedTestbench), a new benchmark specifically designed for function-level unit test generation from real-world Python functions. ULT is constructed through a multi-stage curation process that ensures high cyclomatic complexity and mitigates test case contamination. With 3,909 carefully selected function-level tasks, ULT provides a more realistic and challenging evaluation of LLMs' test generation capabilities. We also provide PLT (PreLeakedTestbench), a pair benchmark of ULT with leaked tests designed to enable a controlled analysis of memorization versus reasoning in test generation. Our evaluation results demonstrate that ULT is significantly more challenging. For example, test cases generated by LLMs only achieve 41.32\%, 45.10\%, 30.22\%, and 40.21\% for accuracy, statement coverage, branch coverage, and mutation score on average for all LLMs, respectively. These results are substantially lower than the corresponding metrics on TestEval (91.79\%, 92.18\%, 82.04\%, and 49.69\%) and PLT (47.07\%, 55.13\%, 40.07\%, and 50.80\%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07556v2">Novice Developers' Perspectives on Adopting LLMs for Software Development: A Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Following the rise of large language models (LLMs), many studies have emerged in recent years focusing on exploring the adoption of LLM-based tools for software development by novice developers: computer science/software engineering students and early-career industry developers with two years or less of professional experience. These studies have sought to understand the perspectives of novice developers on using these tools, a critical aspect of the successful adoption of LLMs in software engineering. To systematically collect and summarise these studies, we conducted a systematic literature review (SLR) following the guidelines by Kitchenham et al. on 80 primary studies published between April 2022 and June 2025 to answer four research questions (RQs). In answering RQ1, we categorised the study motivations and methodological approaches. In RQ2, we identified the software development tasks for which novice developers use LLMs. In RQ3, we categorised the advantages, challenges, and recommendations discussed in the studies. Finally, we discuss the study limitations and future research needs suggested in the primary studies in answering RQ4. Throughout the paper, we also indicate directions for future work and implications for software engineering researchers, educators, and developers. Our research artifacts are publicly available at https://github.com/Samuellucas97/SupplementaryInfoPackage-SLR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10284v3">Can LLMs Generate Tabular Summaries of Science Papers? Rethinking the Evaluation Protocol</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Literature review tables are essential for summarizing and comparing collections of scientific papers. We explore the task of generating tables that best fulfill a user's informational needs given a collection of scientific papers. Building on recent work (Newman et al., 2024), we extend prior approaches to address real-world complexities through a combination of LLM-based methods and human annotations. Our contributions focus on three key challenges encountered in real-world use: (i) User prompts are often under-specified; (ii) Retrieved candidate papers frequently contain irrelevant content; and (iii) Task evaluation should move beyond shallow text similarity techniques and instead assess the utility of inferred tables for information-seeking tasks (e.g., comparing papers). To support reproducible evaluation, we introduce ARXIV2TABLE, a more realistic and challenging benchmark for this task, along with a novel approach to improve literature review table generation in real-world scenarios. Our extensive experiments on this benchmark show that both open-weight and proprietary LLMs struggle with the task, highlighting its difficulty and the need for further advancements. Our dataset and code are available at https://github.com/JHU-CLSP/arXiv2Table.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10009v3">OR-LLM-Agent: Automating Modeling and Solving of Operations Research Optimization Problems with Reasoning LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 8 pages, 13 figures
    </div>
    <details class="paper-abstract">
      With the rise of artificial intelligence (AI), applying large language models (LLMs) to mathematical problem-solving has attracted increasing attention. Most existing approaches attempt to improve Operations Research (OR) optimization problem-solving through prompt engineering or fine-tuning strategies for LLMs. However, these methods are fundamentally constrained by the limited capabilities of non-reasoning LLMs. To overcome these limitations, we propose OR-LLM-Agent, an AI agent framework built on reasoning LLMs for automated OR problem solving. The framework decomposes the task into three sequential stages: mathematical modeling, code generation, and debugging. Each task is handled by a dedicated sub-agent, which enables more targeted reasoning. We also construct BWOR, an OR dataset for evaluating LLM performance on OR tasks. Our analysis shows that in the benchmarks NL4OPT, MAMO, and IndustryOR, reasoning LLMs sometimes underperform their non-reasoning counterparts within the same model family. In contrast, BWOR provides a more consistent and discriminative assessment of model capabilities. Experimental results demonstrate that OR-LLM-Agent utilizing DeepSeek-R1 in its framework outperforms advanced methods, including GPT-o3, Gemini 2.5 Pro, DeepSeek-R1, and ORLM, by at least 7\% in accuracy. These results demonstrate the effectiveness of task decomposition for OR problem solving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04562v2">Evaluating LLMs on Real-World Forecasting Against Human Superforecasters</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their ability to forecast future events remains understudied. A year ago, large language models struggle to come close to the accuracy of a human crowd. I evaluate state-of-the-art LLMs on 464 forecasting questions from Metaculus, comparing their performance against human superforecasters. Frontier models achieve Brier scores that ostensibly surpass the human crowd but still significantly underperform a group of superforecasters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00282v1">Mind the Gap: The Divergence Between Human and LLM-Generated Tasks</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Humans constantly generate a diverse range of tasks guided by internal motivations. While generative agents powered by large language models (LLMs) aim to simulate this complex behavior, it remains uncertain whether they operate on similar cognitive principles. To address this, we conducted a task-generation experiment comparing human responses with those of an LLM agent (GPT-4o). We find that human task generation is consistently influenced by psychological drivers, including personal values (e.g., Openness to Change) and cognitive style. Even when these psychological drivers are explicitly provided to the LLM, it fails to reflect the corresponding behavioral patterns. They produce tasks that are markedly less social, less physical, and thematically biased toward abstraction. Interestingly, while the LLM's tasks were perceived as more fun and novel, this highlights a disconnect between its linguistic proficiency and its capacity to generate human-like, embodied goals.We conclude that there is a core gap between the value-driven, embodied nature of human cognition and the statistical patterns of LLMs, highlighting the necessity of incorporating intrinsic motivation and physical grounding into the design of more human-aligned agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15066v3">ChatModel: Automating Reference Model Design and Verification with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      As the complexity of integrated circuit designs continues to escalate, the functional verification becomes increasingly challenging. Reference models, critical for accelerating the verification process, are themselves becoming more intricate and time-consuming to develop. Despite the promise shown by large language models (LLMs) in code programming, effectively generating complex reference models remains a significant hurdle. To address these challenges, we introduce ChatModel, the first LLM-aided agile reference model generation and verification platform. ChatModel streamlines the transition from design specifications to fully functional reference models by integrating design standardization and hierarchical agile modeling. Employing a building-block generation strategy, it not only enhances the design capabilities of LLMs for reference models but also significantly boosts verification efficiency. We evaluated ChatModel on 300 designs of varying complexity, demonstrating substantial improvements in both efficiency and quality of reference model generation. ChatModel achieved a peak performance improvement of 55.02% compared to alternative methods, with notable enhancements in generation stability, and delivered a 9.18x increase in its capacity to produce reference model designs. Furthermore, it accelerated the iterative process of reference model design and validation by an average of 5.90x compared to traditional approaches. These results highlight the potential of ChatModel to significantly advance the automation of reference model generation and validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07518v4">Efficient and Universal Watermarking for LLM-Generated Code Detection</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 This work has been submitted to IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have significantly enhanced the usability of AI-generated code, providing effective assistance to programmers. This advancement also raises ethical and legal concerns, such as academic dishonesty or the generation of malicious code. For accountability, it is imperative to detect whether a piece of code is AI-generated. Watermarking is broadly considered a promising solution and has been successfully applied to identify LLM-generated text. However, existing efforts on code are far from ideal, suffering from limited universality and excessive time and memory consumption. In this work, we propose a plug-and-play watermarking approach for AI-generated code detection, named ACW (AI Code Watermarking). ACW is training-free and works by selectively applying a set of carefully-designed, semantic-preserving and idempotent code transformations to LLM code outputs. The presence or absence of the transformations serves as implicit watermarks, enabling the detection of AI-generated code. Our experimental results show that ACW effectively detects AI-generated code, preserves code utility, and is resilient against code optimizations. Especially, ACW is efficient and is universal across different LLMs, addressing the limitations of existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00234v1">Quality-of-Service Aware LLM Routing for Edge Computing with Multiple Experts</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 Accepted by IEEE Transactions on Mobile Computing
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities, leading to a significant increase in user demand for LLM services. However, cloud-based LLM services often suffer from high latency, unstable responsiveness, and privacy concerns. Therefore, multiple LLMs are usually deployed at the network edge to boost real-time responsiveness and protect data privacy, particularly for many emerging smart mobile and IoT applications. Given the varying response quality and latency of LLM services, a critical issue is how to route user requests from mobile and IoT devices to an appropriate LLM service (i.e., edge LLM expert) to ensure acceptable quality-of-service (QoS). Existing routing algorithms fail to simultaneously address the heterogeneity of LLM services, the interference among requests, and the dynamic workloads necessary for maintaining long-term stable QoS. To meet these challenges, in this paper we propose a novel deep reinforcement learning (DRL)-based QoS-aware LLM routing framework for sustained high-quality LLM services. Due to the dynamic nature of the global state, we propose a dynamic state abstraction technique to compactly represent global state features with a heterogeneous graph attention network (HAN). Additionally, we introduce an action impact estimator and a tailored reward function to guide the DRL agent in maximizing QoS and preventing latency violations. Extensive experiments on both Poisson and real-world workloads demonstrate that our proposed algorithm significantly improves average QoS and computing resource efficiency compared to existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.18203v2">Policy Maps: Tools for Guiding the Unbounded Space of LLM Behaviors</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 UIST 2025
    </div>
    <details class="paper-abstract">
      AI policy sets boundaries on acceptable behavior for AI models, but this is challenging in the context of large language models (LLMs): how do you ensure coverage over a vast behavior space? We introduce policy maps, an approach to AI policy design inspired by the practice of physical mapmaking. Instead of aiming for full coverage, policy maps aid effective navigation through intentional design choices about which aspects to capture and which to abstract away. With Policy Projector, an interactive tool for designing LLM policy maps, an AI practitioner can survey the landscape of model input-output pairs, define custom regions (e.g., "violence"), and navigate these regions with if-then policy rules that can act on LLM outputs (e.g., if output contains "violence" and "graphic details," then rewrite without "graphic details"). Policy Projector supports interactive policy authoring using LLM classification and steering and a map visualization reflecting the AI practitioner's work. In an evaluation with 12 AI safety experts, our system helps policy designers craft policies around problematic model behaviors such as incorrect gender assumptions and handling of immediate physical safety threats.
    </details>
</div>
