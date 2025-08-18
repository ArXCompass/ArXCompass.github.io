# llm - 2025_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01279v1">ViseGPT: Towards Better Alignment of LLM-generated Data Wrangling Scripts and User Prompts</a></div>
    <div class="paper-meta">
      📅 2025-08-02
      | 💬 Accepted at Annual ACM Symposium on User Interface Software and Technology (UIST'25), September 28-October 1, 2025, Busan, Republic of Korea
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) enable the rapid generation of data wrangling scripts based on natural language instructions, but these scripts may not fully adhere to user-specified requirements, necessitating careful inspection and iterative refinement. Existing approaches primarily assist users in understanding script logic and spotting potential issues themselves, rather than providing direct validation of correctness. To enhance debugging efficiency and optimize the user experience, we develop ViseGPT, a tool that automatically extracts constraints from user prompts to generate comprehensive test cases for verifying script reliability. The test results are then transformed into a tailored Gantt chart, allowing users to intuitively assess alignment with semantic requirements and iteratively refine their scripts. Our design decisions are informed by a formative study (N=8) that explores user practices and challenges. We further evaluate the effectiveness and usability of ViseGPT through a user study (N=18). Results indicate that ViseGPT significantly improves debugging efficiency for LLM-generated data-wrangling scripts, enhances users' ability to detect and correct issues, and streamlines the workflow experience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01273v1">KCR: Resolving Long-Context Knowledge Conflicts via Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-02
    </div>
    <details class="paper-abstract">
      Knowledge conflicts commonly arise across diverse sources, and their prevalence has increased with the advent of LLMs. When dealing with conflicts between multiple contexts, also known as \emph{inter-context knowledge conflicts}, LLMs are often confused by lengthy and conflicting contexts. To address this challenge, we propose the Knowledge Conflict Reasoning (KCR) framework, which enhances the ability of LLMs to resolve conflicting knowledge. The key idea of KCR is to train backbone LLMs to establish a correct reasoning process by rewarding them for selecting and adhering to the context with stronger logical consistency when presented with conflicting contexts. Specifically, we first extract reasoning paths, represented by either text or local knowledge graphs, from the conflicting long contexts. Subsequently, we employ Reinforcement Learning to encourage the model to learn the paradigm of reasoning process that follows correct reasoning paths rather than the incorrect counterparts. This enables the backbone models to genuinely acquire the capability to resolve inter-context knowledge conflicts within long contexts. Experimental results demonstrate that our framework significantly improves the ability of various backbone models to resolve knowledge conflicts in long-context scenarios, yielding substantial performance gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01263v1">Bridging LLMs and Symbolic Reasoning in Educational QA Systems: Insights from the XAI Challenge at IJCNN 2025</a></div>
    <div class="paper-meta">
      📅 2025-08-02
      | 💬 The XAI Challenge @ TRNS-AI Workshop, IJCNN 2025: Explainable AI for Educational Question Answering. Website: https://sites.google.com/view/trns-ai/challenge/
    </div>
    <details class="paper-abstract">
      The growing integration of Artificial Intelligence (AI) into education has intensified the need for transparency and interpretability. While hackathons have long served as agile environments for rapid AI prototyping, few have directly addressed eXplainable AI (XAI) in real-world educational contexts. This paper presents a comprehensive analysis of the XAI Challenge 2025, a hackathon-style competition jointly organized by Ho Chi Minh City University of Technology (HCMUT) and the International Workshop on Trustworthiness and Reliability in Neurosymbolic AI (TRNS-AI), held as part of the International Joint Conference on Neural Networks (IJCNN 2025). The challenge tasked participants with building Question-Answering (QA) systems capable of answering student queries about university policies while generating clear, logic-based natural language explanations. To promote transparency and trustworthiness, solutions were required to use lightweight Large Language Models (LLMs) or hybrid LLM-symbolic systems. A high-quality dataset was provided, constructed via logic-based templates with Z3 validation and refined through expert student review to ensure alignment with real-world academic scenarios. We describe the challenge's motivation, structure, dataset construction, and evaluation protocol. Situating the competition within the broader evolution of AI hackathons, we argue that it represents a novel effort to bridge LLMs and symbolic reasoning in service of explainability. Our findings offer actionable insights for future XAI-centered educational systems and competitive research initiatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10981v3">Rethinking the Role of Prompting Strategies in LLM Test-Time Scaling: A Perspective of Probability Theory</a></div>
    <div class="paper-meta">
      📅 2025-08-02
      | 💬 ACL 2025 Outstanding Paper Award, 33 pages, 51 figures
    </div>
    <details class="paper-abstract">
      Recently, scaling test-time compute on Large Language Models (LLM) has garnered wide attention. However, there has been limited investigation of how various reasoning prompting strategies perform as scaling. In this paper, we focus on a standard and realistic scaling setting: majority voting. We systematically conduct experiments on 6 LLMs $\times$ 8 prompting strategies $\times$ 6 benchmarks. Experiment results consistently show that as the sampling time and computational overhead increase, complicated prompting strategies with superior initial performance gradually fall behind simple Chain-of-Thought. We analyze this phenomenon and provide theoretical proofs. Additionally, we propose a probabilistic method to efficiently predict scaling performance and identify the best prompting strategy under large sampling times, eliminating the need for resource-intensive inference processes in practical applications. Furthermore, we introduce two ways derived from our theoretical analysis to significantly improve the scaling performance. We hope that our research can promote to re-examine the role of complicated prompting, unleash the potential of simple prompting strategies, and provide new insights for enhancing test-time scaling performance. Code is available at https://github.com/MraDonkey/rethinking_prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01235v1">NarraGuide: an LLM-based Narrative Mobile Robot for Remote Place Exploration</a></div>
    <div class="paper-meta">
      📅 2025-08-02
    </div>
    <details class="paper-abstract">
      Robotic telepresence enables users to navigate and experience remote environments. However, effective navigation and situational awareness depend on users' prior knowledge of the environment, limiting the usefulness of these systems for exploring unfamiliar places. We explore how integrating location-aware LLM-based narrative capabilities into a mobile robot can support remote exploration. We developed a prototype system, called NarraGuide, that provides narrative guidance for users to explore and learn about a remote place through a dialogue-based interface. We deployed our prototype in a geology museum, where remote participants (n=20) used the robot to tour the museum. Our findings reveal how users perceived the robot's role, engaged in dialogue in the tour, and expressed preferences for bystander encountering. Our work demonstrates the potential of LLM-enabled robotic capabilities to deliver location-aware narrative guidance and enrich the experience of exploring remote environments.
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
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01091v1">Disaggregated Health Data in LLMs: Evaluating Data Equity in the Context of Asian American Representation</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), such as ChatGPT and Claude, have emerged as essential tools for information retrieval, often serving as alternatives to traditional search engines. However, ensuring that these models provide accurate and equitable information tailored to diverse demographic groups remains an important challenge. This study investigates the capability of LLMs to retrieve disaggregated health-related information for sub-ethnic groups within the Asian American population, such as Korean and Chinese communities. Data disaggregation has been a critical practice in health research to address inequities, making it an ideal domain for evaluating representation equity in LLM outputs. We apply a suite of statistical and machine learning tools to assess whether LLMs deliver appropriately disaggregated and equitable information. By focusing on Asian American sub-ethnic groups, a highly diverse population often aggregated in traditional analyses; we highlight how LLMs handle complex disparities in health data. Our findings contribute to ongoing discussions about responsible AI, particularly in ensuring data equity in the outputs of LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21735v2">GateLens: A Reasoning-Enhanced LLM Agent for Automotive Software Release Analytics</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Ensuring reliable software release decisions is critical in safety-critical domains such as automotive manufacturing. Release validation relies on large tabular datasets, yet manual analysis is slow, costly, and error-prone. While Large Language Models (LLMs) offer promising automation potential, they face challenges in analytical reasoning, structured data handling, and ambiguity resolution. This paper introduces GateLens, an LLM-based system for analyzing tabular data in the automotive domain. GateLens translates natural language queries into Relational Algebra (RA) expressions and generates optimized Python code. Unlike traditional multi-agent or planning-based systems that can be slow, opaque, and costly to maintain, GateLens emphasizes speed, transparency, and reliability. Experimental results show that GateLens outperforms the existing Chain-of-Thought (CoT) + Self-Consistency (SC) based system on real-world datasets, particularly in handling complex and ambiguous queries. Ablation studies confirm the essential role of the RA layer. Industrial deployment shows over 80% reduction in analysis time while maintaining high accuracy across test result interpretation, impact assessment, and release candidate evaluation. GateLens operates effectively in zero-shot settings without requiring few-shot examples or agent orchestration. This work advances deployable LLM system design by identifying key architectural features-intermediate formal representations, execution efficiency, and low configuration overhead-crucial for safety-critical industrial applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01054v1">Autonomous Penetration Testing: Solving Capture-the-Flag Challenges with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 6 pages, 2 figures, 3 tables
    </div>
    <details class="paper-abstract">
      This study evaluates the ability of GPT-4o to autonomously solve beginner-level offensive security tasks by connecting the model to OverTheWire's Bandit capture-the-flag game. Of the 25 levels that were technically compatible with a single-command SSH framework, GPT-4o solved 18 unaided and another two after minimal prompt hints for an overall 80% success rate. The model excelled at single-step challenges that involved Linux filesystem navigation, data extraction or decoding, and straightforward networking. The approach often produced the correct command in one shot and at a human-surpassing speed. Failures involved multi-command scenarios that required persistent working directories, complex network reconnaissance, daemon creation, or interaction with non-standard shells. These limitations highlight current architectural deficiencies rather than a lack of general exploit knowledge. The results demonstrate that large language models (LLMs) can automate a substantial portion of novice penetration-testing workflow, potentially lowering the expertise barrier for attackers and offering productivity gains for defenders who use LLMs as rapid reconnaissance aides. Further, the unsolved tasks reveal specific areas where secure-by-design environments might frustrate simple LLM-driven attacks, informing future hardening strategies. Beyond offensive cybersecurity applications, results suggest the potential to integrate LLMs into cybersecurity education as practice aids.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07360v2">Enhancing Time Series Forecasting via Multi-Level Text Alignment with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 This paper is accepted by DASFAA2025
    </div>
    <details class="paper-abstract">
      The adaptation of large language models (LLMs) to time series forecasting poses unique challenges, as time series data is continuous in nature, while LLMs operate on discrete tokens. Despite the success of LLMs in natural language processing (NLP) and other structured domains, aligning time series data with language-based representations while maintaining both predictive accuracy and interpretability remains a significant hurdle. Existing methods have attempted to reprogram time series data into text-based forms, but these often fall short in delivering meaningful, interpretable results. In this paper, we propose a multi-level text alignment framework for time series forecasting using LLMs that not only improves prediction accuracy but also enhances the interpretability of time series representations. Our method decomposes time series into trend, seasonal, and residual components, which are then reprogrammed into component-specific text representations. We introduce a multi-level alignment mechanism, where component-specific embeddings are aligned with pre-trained word tokens, enabling more interpretable forecasts. Experiments on multiple datasets demonstrate that our method outperforms state-of-the-art models in accuracy while providing good interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01012v1">AutoEDA: Enabling EDA Flow Automation through Microservice-Based LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Modern Electronic Design Automation (EDA) workflows, especially the RTL-to-GDSII flow, require heavily manual scripting and demonstrate a multitude of tool-specific interactions which limits scalability and efficiency. While LLMs introduces strides for automation, existing LLM solutions require expensive fine-tuning and do not contain standardized frameworks for integration and evaluation. We introduce AutoEDA, a framework for EDA automation that leverages paralleled learning through the Model Context Protocol (MCP) specific for standardized and scalable natural language experience across the entire RTL-to-GDSII flow. AutoEDA limits fine-tuning through structured prompt engineering, implements intelligent parameter extraction and task decomposition, and provides an extended CodeBLEU metric to evaluate the quality of TCL scripts. Results from experiments over five previously curated benchmarks show improvements in automation accuracy and efficiency, as well as script quality when compared to existing methods. AutoEDA is released open-sourced to support reproducibility and the EDA community. Available at: https://github.com/AndyLu666/MCP-EDA-Server
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01008v1">ROVI: A VLM-LLM Re-Captioned Dataset for Open-Vocabulary Instance-Grounded Text-to-Image Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 Accepted at ICCV 2025
    </div>
    <details class="paper-abstract">
      We present ROVI, a high-quality synthetic dataset for instance-grounded text-to-image generation, created by labeling 1M curated web images. Our key innovation is a strategy called re-captioning, focusing on the pre-detection stage, where a VLM (Vision-Language Model) generates comprehensive visual descriptions that are then processed by an LLM (Large Language Model) to extract a flat list of potential categories for OVDs (Open-Vocabulary Detectors) to detect. This approach yields a global prompt inherently linked to instance annotations while capturing secondary visual elements humans typically overlook. Evaluations show that ROVI exceeds existing detection datasets in image quality and resolution while containing two orders of magnitude more categories with an open-vocabulary nature. For demonstrative purposes, a text-to-image model GLIGEN trained on ROVI significantly outperforms state-of-the-art alternatives in instance grounding accuracy, prompt fidelity, and aesthetic quality. Our dataset and reproducible pipeline are available at https://github.com/CihangPeng/ROVI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01002v1">Optimal Scheduling Algorithms for LLM Inference: Theory and Practice</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      With the growing use of Large Language Model (LLM)-based tools like ChatGPT, Perplexity, and Gemini across industries, there is a rising need for efficient LLM inference systems. These systems handle requests with a unique two-phase computation structure: a prefill-phase that processes the full input prompt and a decode-phase that autoregressively generates tokens one at a time. This structure calls for new strategies for routing and scheduling requests. In this paper, we take a comprehensive approach to this challenge by developing a theoretical framework that models routing and scheduling in LLM inference systems. We identify two key design principles-optimal tiling and dynamic resource allocation-that are essential for achieving high throughput. Guided by these principles, we propose the Resource-Aware Dynamic (RAD) scheduler and prove that it achieves throughput optimality under mild conditions. To address practical Service Level Objectives (SLOs) such as serving requests with different Time Between Token (TBT) constraints, we design the SLO-Aware LLM Inference (SLAI) scheduler. SLAI uses real-time measurements to prioritize decode requests that are close to missing their TBT deadlines and reorders prefill requests based on known prompt lengths to further reduce the Time To First Token (TTFT) delays. We evaluate SLAI on the Openchat ShareGPT4 dataset using the Mistral-7B model on an NVIDIA RTX ADA 6000 GPU. Compared to Sarathi-Serve, SLAI reduces the median TTFT by 53% and increases the maximum serving capacity by 26% such that median TTFT is below 0.5 seconds, while meeting tail TBT latency constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00998v1">Are LLM-Powered Social Media Bots Realistic?</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 Accepted into SBP-BRiMS 2025
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become more sophisticated, there is a possibility to harness LLMs to power social media bots. This work investigates the realism of generating LLM-Powered social media bot networks. Through a combination of manual effort, network science and LLMs, we create synthetic bot agent personas, their tweets and their interactions, thereby simulating social media networks. We compare the generated networks against empirical bot/human data, observing that both network and linguistic properties of LLM-Powered Bots differ from Wild Bots/Humans. This has implications towards the detection and effectiveness of LLM-Powered Bots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00965v1">VAULT: Vigilant Adversarial Updates via LLM-Driven Retrieval-Augmented Generation for NLI</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      We introduce VAULT, a fully automated adversarial RAG pipeline that systematically uncovers and remedies weaknesses in NLI models through three stages: retrieval, adversarial generation, and iterative retraining. First, we perform balanced few-shot retrieval by embedding premises with both semantic (BGE) and lexical (BM25) similarity. Next, we assemble these contexts into LLM prompts to generate adversarial hypotheses, which are then validated by an LLM ensemble for label fidelity. Finally, the validated adversarial examples are injected back into the training set at increasing mixing ratios, progressively fortifying a zero-shot RoBERTa-base model.On standard benchmarks, VAULT elevates RoBERTa-base accuracy from 88.48% to 92.60% on SNLI +4.12%, from 75.04% to 80.95% on ANLI +5.91%, and from 54.67% to 71.99% on MultiNLI +17.32%. It also consistently outperforms prior in-context adversarial methods by up to 2.0% across datasets. By automating high-quality adversarial data curation at scale, VAULT enables rapid, human-independent robustness improvements in NLI inference tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02732v1">A Note on Code Quality Score: LLMs for Maintainable Large Codebases</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 24 pages, ICLR format
    </div>
    <details class="paper-abstract">
      Maintaining code quality in large-scale software systems presents significant challenges, particularly in settings where a large numbers of engineers work concurrently on a codebase. This paper introduces Code Quality Score (CQS) system to automatically detect issues with a set of code changes and provide actionable insights. At its core, the CQS system is powered by two Llama3 models, fine-tuned (with SFT and offline RL approaches), to a) detect common code quality issues related to coding best practices and b) to provide good ``critiques'' for LLM-generated code review respectively. To maintain good user experience, we layer the system with hand-crafted rules to filter out incorrect responses/hallucinations. Offline evaluations show that our CQS system is able to achieve an impressive precision rate for identifying valid issues. This system has already been rolled out to developers in an industrial scale setting and has consistently achieved 60\% week over week user helpfulness rate, demonstrating its effectiveness in a real-world environment. In this paper, we present details of the CQS system along with some learnings on curating developer feedback to create training data for LLM fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02721v1">Blueprint First, Model Second: A Framework for Deterministic LLM Workflow</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 8 pages, 6 figures, 3 tables
    </div>
    <details class="paper-abstract">
      While powerful, the inherent non-determinism of large language model (LLM) agents limits their application in structured operational environments where procedural fidelity and predictable execution are strict requirements. This limitation stems from current architectures that conflate probabilistic, high-level planning with low-level action execution within a single generative process. To address this, we introduce the Source Code Agent framework, a new paradigm built on the "Blueprint First, Model Second" philosophy. Our framework decouples the workflow logic from the generative model. An expert-defined operational procedure is first codified into a source code-based Execution Blueprint, which is then executed by a deterministic engine. The LLM is strategically invoked as a specialized tool to handle bounded, complex sub-tasks within the workflow, but never to decide the workflow's path. We conduct a comprehensive evaluation on the challenging tau-bench benchmark, designed for complex user-tool-rule scenarios. Our results demonstrate that the Source Code Agent establishes a new state-of-the-art, outperforming the strongest baseline by 10.1 percentage points on the average Pass^1 score while dramatically improving execution efficiency. Our work enables the verifiable and reliable deployment of autonomous agents in applications governed by strict procedural logic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00806v1">Adacc: Adaptive Compression and Activation Checkpointing for LLM Memory Management</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Training large language models often employs recomputation to alleviate memory pressure, which can introduce up to 30% overhead in real-world scenarios. In this paper, we propose Adacc, a novel memory management framework that combines adaptive compression and activation checkpointing to reduce the GPU memory footprint. It comprises three modules: (1) We design layer-specific compression algorithms that account for outliers in LLM tensors, instead of directly quantizing floats from FP16 to INT4, to ensure model accuracy. (2) We propose an optimal scheduling policy that employs MILP to determine the best memory optimization for each tensor. (3) To accommodate changes in training tensors, we introduce an adaptive policy evolution mechanism that adjusts the policy during training to enhance throughput. Experimental results show that Adacc can accelerate the LLM training by 1.01x to 1.37x compared to state-of-the-art frameworks, while maintaining comparable model accuracy to the Baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02039v3">An Investigation into Value Misalignment in LLM-Generated Texts for Cultural Heritage</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become increasingly prevalent in tasks related to cultural heritage, such as generating descriptions of historical monuments, translating ancient texts, preserving oral traditions, and creating educational content, their ability to produce accurate and culturally aligned texts is being increasingly relied upon by users and researchers. However, cultural value misalignments may exist in generated texts, such as the misrepresentation of historical facts, the erosion of cultural identity, and the oversimplification of complex cultural narratives, which may lead to severe consequences. Therefore, investigating value misalignment in the context of LLM for cultural heritage is crucial for mitigating these risks, yet there has been a significant lack of systematic and comprehensive study and investigation in this area. To fill this gap, we systematically assess the reliability of LLMs in generating culturally aligned texts for cultural heritage-related tasks. We conduct a comprehensive evaluation by compiling an extensive set of 1066 query tasks covering 5 widely recognized categories with 17 aspects within the knowledge framework of cultural heritage across 5 open-source LLMs, and examine both the type and rate of cultural value misalignments in the generated texts. Using both automated and manual approaches, we effectively detect and analyze the cultural value misalignments in LLM-generated texts. Our findings are concerning: over 65% of the generated texts exhibit notable cultural misalignments, with certain tasks demonstrating almost complete misalignment with key cultural values. Beyond these findings, this paper introduces a benchmark dataset and a comprehensive evaluation workflow that can serve as a valuable resource for future research aimed at enhancing the cultural sensitivity and reliability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00762v1">ITUNLP at SemEval-2025 Task 8: Question-Answering over Tabular Data: A Zero-Shot Approach using LLM-Driven Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      This paper presents our system for SemEval-2025 Task 8: DataBench, Question-Answering over Tabular Data. The primary objective of this task is to perform question answering on given tabular datasets from diverse domains under two subtasks: DataBench QA (Subtask I) and DataBench Lite QA (Subtask II). To tackle both subtasks, we developed a zero-shot solution with a particular emphasis on leveraging Large Language Model (LLM)-based code generation. Specifically, we propose a Python code generation framework utilizing state-of-the-art open-source LLMs to generate executable Pandas code via optimized prompting strategies. Our experiments reveal that different LLMs exhibit varying levels of effectiveness in Python code generation. Additionally, results show that Python code generation achieves superior performance in tabular question answering compared to alternative approaches. Although our ranking among zero-shot systems is unknown at the time of this paper's submission, our system achieved eighth place in Subtask I and sixth place in Subtask~II among the 30 systems that outperformed the baseline in the open-source models category.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17217v2">Mitigating Gender Bias via Fostering Exploratory Thinking in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often exhibit gender bias, resulting in unequal treatment of male and female subjects across different contexts. To address this issue, we propose a novel data generation framework that fosters exploratory thinking in LLMs. Our approach prompts models to generate story pairs featuring male and female protagonists in structurally identical, morally ambiguous scenarios, then elicits and compares their moral judgments. When inconsistencies arise, the model is guided to produce balanced, gender-neutral judgments. These story-judgment pairs are used to fine-tune or optimize the models via Direct Preference Optimization (DPO). Experimental results show that our method significantly reduces gender bias while preserving or even enhancing general model capabilities. We will release the code and generated data. We release the code and generated data at: https://github.com/WeiKangda/LLMs-Exploratory-Bias-Mitigation/tree/main.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09751v2">Sound and Complete Neurosymbolic Reasoning with LLM-Grounded Interpretations</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 29 pages, 9 tables, 3 figures. Accepted to the 19th Conference on Neurosymbolic Learning and Reasoning (NeSy 2025)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in natural language understanding and generation, but they exhibit problems with logical consistency in the output they generate. How can we harness LLMs' broad-coverage parametric knowledge in formal reasoning despite their inconsistency? We present a method for directly integrating an LLM into the interpretation function of the formal semantics for a paraconsistent logic. We provide experimental evidence for the feasibility of the method by evaluating the function using datasets created from several short-form factuality benchmarks. Unlike prior work, our method offers a theoretical framework for neurosymbolic reasoning that leverages an LLM's knowledge while preserving the underlying logic's soundness and completeness properties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00741v1">Out-of-Context Abduction: LLMs Make Inferences About Procedural Data Leveraging Declarative Facts in Earlier Training Data</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on large corpora, yet it is unclear whether they can reason about the information present within their training data. We design experiments to study out-of-context abduction in LLMs, the ability to infer the most plausible explanations for observations using relevant facts present in training data. We train treatment LLMs on names and behavior descriptions of fictitious chatbots, but not on examples of dialogue with the chatbots. We find that OpenAI's GPT 4o LLM can correctly infer at least one chatbot's name after observing example responses characteristic of that chatbot. We also find that previously training GPT 4o on descriptions of a chatbot's behavior allows it to display behaviors more characteristic of the chatbot when iteratively trained to display such behaviors. Our results have implications for situational awareness in LLMs and, therefore, for AI safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00737v1">How LLMs are Shaping the Future of Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into Virtual Reality (VR) games marks a paradigm shift in the design of immersive, adaptive, and intelligent digital experiences. This paper presents a comprehensive review of recent research at the intersection of LLMs and VR, examining how these models are transforming narrative generation, non-player character (NPC) interactions, accessibility, personalization, and game mastering. Drawing from an analysis of 62 peer reviewed studies published between 2018 and 2025, we identify key application domains ranging from emotionally intelligent NPCs and procedurally generated storytelling to AI-driven adaptive systems and inclusive gameplay interfaces. We also address the major challenges facing this convergence, including real-time performance constraints, memory limitations, ethical risks, and scalability barriers. Our findings highlight that while LLMs significantly enhance realism, creativity, and user engagement in VR environments, their effective deployment requires robust design strategies that integrate multimodal interaction, hybrid AI architectures, and ethical safeguards. The paper concludes by outlining future research directions in multimodal AI, affective computing, reinforcement learning, and open-source development, aiming to guide the responsible advancement of intelligent and inclusive VR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00719v1">Dynamically Adaptive Reasoning via LLM-Guided MCTS for Efficient and Context-Aware KGQA</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Knowledge Graph Question Answering (KGQA) aims to interpret natural language queries and perform structured reasoning over knowledge graphs by leveraging their relational and semantic structures to retrieve accurate answers. Recent KGQA methods primarily follow either retrieve-then-reason paradigm, relying on GNNs or heuristic rules for static paths extraction, or dynamic path generation strategies that use large language models (LLMs) with prompting to jointly perform retrieval and reasoning. However, the former suffers from limited adaptability due to static path extraction and lack of contextual refinement, while the latter incurs high computational costs and struggles with accurate path evaluation due to reliance on fixed scoring functions and extensive LLM calls. To address these issues, this paper proposes Dynamically Adaptive MCTS-based Reasoning (DAMR), a novel framework that integrates symbolic search with adaptive path evaluation for efficient and context-aware KGQA. DAMR employs a Monte Carlo Tree Search (MCTS) backbone guided by an LLM-based planner, which selects top-$k$ relevant relations at each step to reduce search space. To improve path evaluation accuracy, we introduce a lightweight Transformer-based scorer that performs context-aware plausibility estimation by jointly encoding the question and relation sequence through cross-attention, enabling the model to capture fine-grained semantic shifts during multi-hop reasoning. Furthermore, to alleviate the scarcity of high-quality supervision, DAMR incorporates a dynamic pseudo-path refinement mechanism that periodically generates training signals from partial paths explored during search, allowing the scorer to continuously adapt to the evolving distribution of reasoning trajectories. Extensive experiments on multiple KGQA benchmarks show that DAMR significantly outperforms state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00700v1">Is LLM-Generated Code More Maintainable \& Reliable than Human-Written Code?</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 Accepted ESEM2025
    </div>
    <details class="paper-abstract">
      Background: The rise of Large Language Models (LLMs) in software development has opened new possibilities for code generation. Despite the widespread use of this technology, it remains unclear how well LLMs generate code solutions in terms of software quality and how they compare to human-written code. Aims: This study compares the internal quality attributes of LLM-generated and human-written code. Method: Our empirical study integrates datasets of coding tasks, three LLM configurations (zero-shot, few-shot, and fine-tuning), and SonarQube to assess software quality. The dataset comprises Python code solutions across three difficulty levels: introductory, interview, and competition. We analyzed key code quality metrics, including maintainability and reliability, and the estimated effort required to resolve code issues. Results: Our analysis shows that LLM-generated code has fewer bugs and requires less effort to fix them overall. Interestingly, fine-tuned models reduced the prevalence of high-severity issues, such as blocker and critical bugs, and shifted them to lower-severity categories, but decreased the model's performance. In competition-level problems, the LLM solutions sometimes introduce structural issues that are not present in human-written code. Conclusion: Our findings provide valuable insights into the quality of LLM-generated code; however, the introduction of critical issues in more complex scenarios highlights the need for a systematic evaluation and validation of LLM solutions. Our work deepens the understanding of the strengths and limitations of LLMs for code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00680v1">Better Call Claude: Can LLMs Detect Changes of Writing Style?</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      This article explores the zero-shot performance of state-of-the-art large language models (LLMs) on one of the most challenging tasks in authorship analysis: sentence-level style change detection. Benchmarking four LLMs on the official PAN~2024 and 2025 "Multi-Author Writing Style Analysis" datasets, we present several observations. First, state-of-the-art generative models are sensitive to variations in writing style - even at the granular level of individual sentences. Second, their accuracy establishes a challenging baseline for the task, outperforming suggested baselines of the PAN competition. Finally, we explore the influence of semantics on model predictions and present evidence suggesting that the latest generation of LLMs may be more sensitive to content-independent and purely stylistic signals than previously reported.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00669v1">Medical Reasoning in the Era of LLMs: A Systematic Review of Enhancement Techniques and Applications</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) in medicine has enabled impressive capabilities, yet a critical gap remains in their ability to perform systematic, transparent, and verifiable reasoning, a cornerstone of clinical practice. This has catalyzed a shift from single-step answer generation to the development of LLMs explicitly designed for medical reasoning. This paper provides the first systematic review of this emerging field. We propose a taxonomy of reasoning enhancement techniques, categorized into training-time strategies (e.g., supervised fine-tuning, reinforcement learning) and test-time mechanisms (e.g., prompt engineering, multi-agent systems). We analyze how these techniques are applied across different data modalities (text, image, code) and in key clinical applications such as diagnosis, education, and treatment planning. Furthermore, we survey the evolution of evaluation benchmarks from simple accuracy metrics to sophisticated assessments of reasoning quality and visual interpretability. Based on an analysis of 60 seminal studies from 2022-2025, we conclude by identifying critical challenges, including the faithfulness-plausibility gap and the need for native multimodal reasoning, and outlining future directions toward building efficient, robust, and sociotechnically responsible medical AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08395v2">IssueBench: Millions of Realistic Prompts for Measuring Issue Bias in LLM Writing Assistance</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are helping millions of users write texts about diverse issues, and in doing so expose users to different ideas and perspectives. This creates concerns about issue bias, where an LLM tends to present just one perspective on a given issue, which in turn may influence how users think about this issue. So far, it has not been possible to measure which issue biases LLMs actually manifest in real user interactions, making it difficult to address the risks from biased LLMs. Therefore, we create IssueBench: a set of 2.49m realistic prompts for measuring issue bias in LLM writing assistance, which we construct based on 3.9k templates (e.g. "write a blog about") and 212 political issues (e.g. "AI regulation") from real user interactions. Using IssueBench, we show that issue biases are common and persistent in state-of-the-art LLMs. We also show that biases are remarkably similar across models, and that all models align more with US Democrat than Republican voter opinion on a subset of issues. IssueBench can easily be adapted to include other issues, templates, or tasks. By enabling robust and realistic measurement, we hope that IssueBench can bring a new quality of evidence to ongoing discussions about LLM biases and how to address them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12927v2">SEFL: Enhancing Educational Assignment Feedback with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Providing high-quality feedback to student assignments is crucial for student success, but it is constrained by time and costs. In this work, we introduce Synthetic Educational Feedback Loops (SEFL), a synthetic data framework designed to generate data that resembles immediate, on-demand feedback at scale without relying on extensive, real-world student assignments. To get this type of data, two large language models (LLMs) operate in teacher-student roles to simulate assignment completion and formative feedback, generating synthetic pairs of student work and corresponding critiques and actionable improvements from a teacher. With this data, we fine-tune smaller, more computationally efficient LLMs on these synthetic pairs, enabling them to replicate key features of high-quality, goal-oriented feedback. Unlike personalized tutoring approaches that offer multi-turn, individualized instruction, SEFL specifically focuses on replicating the teacher-student assignment feedback loop in higher education. Through comprehensive evaluations with four LLM judges and three human experts, we demonstrate that SEFL-tuned models outperform both their non-tuned counterparts in feedback quality and an existing baseline. The potential for societal impact is reinforced by extensive qualitative comments by ratings by human stakeholders -- both students and higher education instructors. All in all, SEFL has substantial potential to transform feedback processes for higher education and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00602v1">LeakSealer: A Semisupervised Defense for LLMs Against Prompt Injection and Leakage Attacks</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 22 pages, preprint
    </div>
    <details class="paper-abstract">
      The generalization capabilities of Large Language Models (LLMs) have led to their widespread deployment across various applications. However, this increased adoption has introduced several security threats, notably in the forms of jailbreaking and data leakage attacks. Additionally, Retrieval Augmented Generation (RAG), while enhancing context-awareness in LLM responses, has inadvertently introduced vulnerabilities that can result in the leakage of sensitive information. Our contributions are twofold. First, we introduce a methodology to analyze historical interaction data from an LLM system, enabling the generation of usage maps categorized by topics (including adversarial interactions). This approach further provides forensic insights for tracking the evolution of jailbreaking attack patterns. Second, we propose LeakSealer, a model-agnostic framework that combines static analysis for forensic insights with dynamic defenses in a Human-In-The-Loop (HITL) pipeline. This technique identifies topic groups and detects anomalous patterns, allowing for proactive defense mechanisms. We empirically evaluate LeakSealer under two scenarios: (1) jailbreak attempts, employing a public benchmark dataset, and (2) PII leakage, supported by a curated dataset of labeled LLM interactions. In the static setting, LeakSealer achieves the highest precision and recall on the ToxicChat dataset when identifying prompt injection. In the dynamic setting, PII leakage detection achieves an AUPRC of $0.97$, significantly outperforming baselines such as Llama Guard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15066v2">Time-RA: Towards Time Series Reasoning for Anomaly with LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 Under review. 19 pages, 8 figures, 12 tables. Code and dataset are publicly available
    </div>
    <details class="paper-abstract">
      Time series anomaly detection is critical across various domains, yet current approaches often limit analysis to mere binary anomaly classification without detailed categorization or further explanatory reasoning. To address these limitations, we propose a novel task, Time-series Reasoning for Anomaly (Time-RA) that transforms classical time series anomaly detection from a discriminative into a generative, reasoning-intensive task leveraging Large Language Models (LLMs). Also, we introduce the first real-world multimodal benchmark dataset, RATs40K, explicitly annotated for anomaly reasoning, comprising approximately 40,000 samples across 10 real-world domains. Each sample includes numeric time series data, contextual text information, and visual representations, each annotated with fine-grained categories (14 types for univariate anomalies and 6 for multivariate anomalies) and structured explanatory reasoning. We develop a sophisticated annotation framework utilizing ensemble-generated labels refined through GPT-4-driven feedback, ensuring accuracy and interpretability. Extensive benchmarking of LLMs and multimodal LLMs demonstrates the capabilities and limitations of current models, highlighting the critical role of supervised fine-tuning. Our dataset and task pave the way for significant advancements in interpretable time series anomaly detection and reasoning. The code (https://github.com/yyysjz1997/Time-RA) and dataset (https://huggingface.co/datasets/Time-RA/RATs40K) have been fully open-sourced to support and accelerate future research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00581v1">From EMR Data to Clinical Insight: An LLM-Driven Framework for Automated Pre-Consultation Questionnaire Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Pre-consultation is a critical component of effective healthcare delivery. However, generating comprehensive pre-consultation questionnaires from complex, voluminous Electronic Medical Records (EMRs) is a challenging task. Direct Large Language Model (LLM) approaches face difficulties in this task, particularly regarding information completeness, logical order, and disease-level synthesis. To address this issue, we propose a novel multi-stage LLM-driven framework: Stage 1 extracts atomic assertions (key facts with timing) from EMRs; Stage 2 constructs personal causal networks and synthesizes disease knowledge by clustering representative networks from an EMR corpus; Stage 3 generates tailored personal and standardized disease-specific questionnaires based on these structured representations. This framework overcomes limitations of direct methods by building explicit clinical knowledge. Evaluated on a real-world EMR dataset and validated by clinical experts, our method demonstrates superior performance in information coverage, diagnostic relevance, understandability, and generation time, highlighting its practical potential to enhance patient information collection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00570v1">Session-Based Recommendation with Validated and Enriched LLM Intents</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Session-based recommendation (SBR) aims to predict the next item for an anonymous user in a timely manner. However, SBR suffers from data sparsity due to the short and anonymous nature of sessions. Recently, an emerging line of work has explored inferring the underlying user intents of a session using large language models (LLMs), with the generated intents serving as auxiliary training signals to enhance SBR models. Despite its promise, this approach faces three key challenges: validating intent quality, incorporating session-level multi-intents, and complementing inevitable LLM failure cases. In this paper, we propose VELI4SBR, a two-stage framework that leverages Validated and Enriched LLM-generated Intents for SBR. In the first stage, we generate high-quality intents using a predict-and-correct loop that validates the informativeness of LLM-generated intents with a global intent pool to constrain the LLM's output space and reduce hallucination. In the second stage, we enhance the SBR model using the generated intents through a lightweight multi-intent prediction and fusion mechanism. Furthermore, we introduce a training strategy that compensates for LLM failures by inferring intents from inter-session behavioral similarities. Extensive experiments show that VELI4SBR outperforms state-of-the-art baselines while improving explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19693v2">AdaptiVocab: Enhancing LLM Efficiency in Focused Domains through Lightweight Vocabulary Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive versatility as general purpose models. However, their broad applicability comes at a high-cost computational overhead, particularly in auto-regressive decoding where each step requires a forward pass. In domain-specific settings, general-purpose capabilities are unnecessary and can be exchanged for efficiency. In this work, we take a novel perspective on domain adaptation, reducing latency and computational costs by adapting the vocabulary to focused domains of interest. We introduce AdaptiVocab, an end-to-end approach for vocabulary adaptation, designed to enhance LLM efficiency in low-resource domains. AdaptiVocab can be applied to any tokenizer and architecture, modifying the vocabulary by replacing tokens with domain-specific n-gram-based tokens, thereby reducing the number of tokens required for both input processing and output generation. AdaptiVocab initializes new n-token embeddings using an exponentially weighted combination of existing embeddings and employs a lightweight fine-tuning phase that can be efficiently performed on a single GPU. We evaluate two 7B LLMs across three niche domains, assessing efficiency, generation quality, and end-task performance. Our results show that AdaptiVocab reduces token usage by over 25% without compromising performance
    </details>
</div>
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
