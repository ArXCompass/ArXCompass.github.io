# llm - 2026_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- Part 3
- [Part 4](papers_4.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22503v2">LLEMA: Evolutionary Search with LLMs for Multi-Objective Materials Discovery</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Materials discovery requires navigating vast chemical and structural spaces while satisfying multiple, often conflicting, objectives. We present LLM-guided Evolution for MAterials discovery (LLEMA), a unified framework that couples the scientific knowledge embedded in large language models with chemistry-informed evolutionary rules and memory-based refinement. At each iteration, an LLM proposes crystallographically specified candidates under explicit property constraints; a surrogate-augmented oracle estimates physicochemical properties; and a multi-objective scorer updates success/failure memories to guide subsequent generations. Evaluated on 14 realistic tasks that span electronics, energy, coatings, optics, and aerospace, LLEMA discovers candidates that are chemically plausible, thermodynamically stable, and property-aligned, achieving higher hit rates and improved Pareto front quality relative to generative and LLM-only baselines. Ablation studies confirm the importance of rule-guided generation, memory-based refinement, and surrogate prediction. By enforcing synthesizability and multi-objective trade-offs, LLEMA provides a principled approach to accelerating practical materials discovery. Project website: https://scientific-discovery.github.io/llema-project/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05400v1">An Exploration-Analysis-Disambiguation Reasoning Framework for Word Sense Disambiguation with Low-Parameter LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Accepted at LREC 2026, 15 pages, 11 Tables
    </div>
    <details class="paper-abstract">
      Word Sense Disambiguation (WSD) remains a key challenge in Natural Language Processing (NLP), especially when dealing with rare or domain-specific senses that are often misinterpreted. While modern high-parameter Large Language Models (LLMs) such as GPT-4-Turbo have shown state-of-the-art WSD performance, their computational and energy demands limit scalability. This study investigates whether low-parameter LLMs (<4B parameters) can achieve comparable results through fine-tuning strategies that emphasize reasoning-driven sense identification. Using the FEWS dataset augmented with semi-automated, rationale-rich annotations, we fine-tune eight small-scale open-source LLMs (e.g. Gemma and Qwen). Our results reveal that Chain-of-Thought (CoT)-based reasoning combined with neighbour-word analysis achieves performance comparable to GPT-4-Turbo in zero-shot settings. Importantly, Gemma-3-4B and Qwen-3-4B models consistently outperform all medium-parameter baselines and state-of-the-art models on FEWS, with robust generalization to unseen senses. Furthermore, evaluation on the unseen "Fool Me If You Can'' dataset confirms strong cross-domain adaptability without task-specific fine-tuning. This work demonstrates that with carefully crafted reasoning-centric fine-tuning, low-parameter LLMs can deliver accurate WSD while substantially reducing computational and energy demands.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05399v1">Judge Reliability Harness: Stress Testing the Reliability of LLM Judges</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Accepted at Agents in the Wild: Safety, Security, and Beyond Workshop at ICLR 2026 - April 26, 2026, Rio de Janeiro, Brazil
    </div>
    <details class="paper-abstract">
      We present the Judge Reliability Harness, an open source library for constructing validation suites that test the reliability of LLM judges. As LLM based scoring is widely deployed in AI benchmarks, more tooling is needed to efficiently assess the reliability of these methods. Given a benchmark dataset and an LLM judge configuration, the harness generates reliability tests that evaluate both binary judgment accuracy and ordinal grading performance for free-response and agentic task formats. We evaluate four state-of-the-art judges across four benchmarks spanning safety, persuasion, misuse, and agentic behavior, and find meaningful variation in performance across models and perturbation types, highlighting opportunities to improve the robustness of LLM judges. No judge that we evaluated is uniformly reliable across benchmarks using our harness. For example, our preliminary experiments on judges revealed consistency issues as measured by accuracy in judging another LLM's ability to complete a task due to simple text formatting changes, paraphrasing, changes in verbosity, and flipping the ground truth label in LLM-produced responses. The code for this tool is available at: https://github.com/RANDCorporation/judge-reliability-harness
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05290v1">X-RAY: Mapping LLM Reasoning Capability via Formalized and Calibrated Probes</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve promising performance, yet their ability to reason remains poorly understood. Existing evaluations largely emphasize task-level accuracy, often conflating pattern matching with reasoning capability. We present X-RAY, an explainable reasoning analysis system that maps the LLM reasoning capability using calibrated, formally verified probes. We model reasoning capability as a function of extractable \textit{structure}, operationalized through formal properties such as constraint interaction, reasoning depth, and solution-space geometry. X-Ray generates probes via formal tools with controlled structural variations, enabling precise isolation of incremental structural information through formal calibration and verification. We evaluate state-of-the-art LLMs on problems ranging from junior-level to advanced in mathematics, physics, and chemistry. Our analysis reveals a systematic asymmetry in LLM reasoning: models are relatively robust to constraint refinement, where additional conditions shrink an existing solution space, but degrade sharply under solution-space restructuring, where modifications alter the underlying structural form of the solution manifold. Moreover, calibrated formal probes differentiate models that appear indistinguishable on standard benchmarks and reveal failure modes that are structurally interpretable rather than opaque. Beyond evaluation, our framework is contamination-free and supports the training and testing of reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21033v2">Towards Trustworthy Legal AI through LLM Agents and Formal Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      Legal decisions should be logical and based on statutory laws. While large language models(LLMs) are good at understanding legal text, they cannot provide verifiable justifications. We present L4L, a solver-centric framework that enforces formal alignment between LLM-based legal reasoning and statutory laws. The framework integrates role-differentiated LLM agents with SMT-backed verification, combining the flexibility of natural language with the rigor of symbolic reasoning. Our approach operates in four stages: (1) Statute Knowledge Building, where LLMs autoformalize legal provisions into logical constraints and validate them through case-level testing; (2) Dual Fact-and-Statute Extraction, in which the prosecutor-and defense-aligned agents independently map case narratives to argument tuples; (3) Solver-Centric Adjudication, where SMT solvers check the legal admissibility and consistency of the arguments against the formalized statute knowledge; (4) Judicial Rendering, in which a judge agent integrates solver-validated reasoning with statutory interpretation and similar precedents to produce a legally grounded verdict. Experiments on public legal benchmarks show that L4L consistently outperforms baselines, while providing auditable justifications that enable trustworthy legal AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08762v2">EDINET-Bench: Evaluating LLMs on Complex Financial Tasks using Japanese Financial Statements</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Accepted to ICLR 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made remarkable progress, surpassing human performance on several benchmarks in domains such as mathematics and coding. A key driver of this progress has been the development of benchmark datasets. In contrast, the financial domain poses higher entry barriers due to its demand for specialized expertise, and benchmarks remain relatively scarce compared to those in mathematics or coding. We introduce EDINET-Bench, an open-source Japanese financial benchmark designed to evaluate LLMs on challenging tasks such as accounting fraud detection, earnings forecasting, and industry classification. EDINET-Bench is constructed from ten years of annual reports filed by Japanese companies. These tasks require models to process entire annual reports and integrate information across multiple tables and textual sections, demanding expert-level reasoning that is challenging even for human professionals. Our experiments show that even state-of-the-art LLMs struggle in this domain, performing only marginally better than logistic regression in binary classification tasks such as fraud detection and earnings forecasting. Our results show that simply providing reports to LLMs in a straightforward setting is not enough. This highlights the need for benchmark frameworks that better reflect the environments in which financial professionals operate, with richer scaffolding such as realistic simulations and task-specific reasoning support to enable more effective problem solving. We make our dataset and code publicly available to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05197v1">Diffusion LLMs can think EoS-by-EoS</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      Diffusion LLMs have been proposed as an alternative to autoregressive LLMs, excelling especially at complex reasoning tasks with interdependent sub-goals. Curiously, this is particularly true if the generation length, i.e., the number of tokens the model has to output, is set to a much higher value than is required for providing the correct answer to the task, and the model pads its answer with end-of-sequence (EoS) tokens. We hypothesize that diffusion models think EoS-by-EoS, that is, they use the representations of EoS tokens as a hidden scratchpad, which allows them to solve harder reasoning problems. We experiment with the diffusion models LLaDA1.5, LLaDA2.0-mini, and Dream-v0 on the tasks Addition, Entity Tracking, and Sudoku. In a controlled prompting experiment, we confirm that adding EoS tokens improves the LLMs' reasoning capabilities. To further verify whether they serve as space for hidden computations, we patch the hidden states of the EoS tokens with those of a counterfactual generation, which frequently changes the generated output to the counterfactual. The success of the causal intervention underscores that the EoS tokens, which one may expect to be devoid of meaning, carry information on the problem to solve. The behavioral experiments and the causal interventions indicate that diffusion LLMs can indeed think EoS-by-EoS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05189v1">Small Changes, Big Impact: Demographic Bias in LLM-Based Hiring Through Subtle Sociocultural Markers in Anonymised Resumes</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in resume screening pipelines. Although explicit PII (e.g., names) is commonly redacted, resumes typically retain subtle sociocultural markers (languages, co-curricular activities, volunteering, hobbies) that can act as demographic proxies. We introduce a generalisable stress-test framework for hiring fairness, instantiated in the Singapore context: 100 neutral job-aligned resumes are augmented into 4100 variants spanning four ethnicities and two genders, differing only in job-irrelevant markers. We evaluate 18 LLMs in two realistic settings: (i) Direct Comparison (1v1) and (ii) Score & Shortlist (top-scoring rate), each with and without rationale prompting. Even without explicit identifiers, models recover demographic attributes with high F1 and exhibit systematic disparities, with models favouring markers associated with Chinese and Caucasian males. Ablations show language markers suffice for ethnicity inference, whereas gender relies on hobbies and activities. Furthermore, prompting for explanations tends to amplify bias. Our findings suggest that seemingly innocuous markers surviving anonymisation can materially skew automated hiring outcomes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05168v1">Sparse-BitNet: 1.58-bit LLMs are Naturally Friendly to Semi-Structured Sparsity</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      Semi-structured N:M sparsity and low-bit quantization (e.g., 1.58-bit BitNet) are two promising approaches for improving the efficiency of large language models (LLMs), yet they have largely been studied in isolation. In this work, we investigate their interaction and show that 1.58-bit BitNet is naturally more compatible with N:M sparsity than full-precision models. To study this effect, we propose Sparse-BitNet, a unified framework that jointly applies 1.58-bit quantization and dynamic N:M sparsification while ensuring stable training for the first time. Across multiple model scales and training regimes (sparse pretraining and dense-to-sparse schedules), 1.58-bit BitNet consistently exhibits smaller performance degradation than full-precision baselines at the same sparsity levels and can tolerate higher structured sparsity before accuracy collapse. Moreover, using our custom sparse tensor core, Sparse-BitNet achieves substantial speedups in both training and inference, reaching up to 1.30X. These results highlight that combining extremely low-bit quantization with semi-structured N:M sparsity is a promising direction for efficient LLMs. Code available at https://github.com/AAzdi/Sparse-BitNet
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05167v1">C2-Faith: Benchmarking LLM Judges for Causal and Coverage Faithfulness in Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as judges of chain-of-thought (CoT) reasoning, but it remains unclear whether they can reliably assess process faithfulness rather than just answer plausibility. We introduce C2-Faith, a benchmark built from PRM800K that targets two complementary dimensions of faithfulness: causality (does each step logically follow from prior context?) and coverage (are essential intermediate inferences present?). Using controlled perturbations, we create examples with known causal error positions by replacing a single step with an acausal variant, and with controlled coverage deletions at varying deletion rates (scored against reference labels). We evaluate three frontier judges under three tasks: binary causal detection, causal step localization, and coverage scoring. The results show that model rankings depend strongly on task framing, with no single judge dominating all settings; all judges exhibit a substantial gap between detecting an error and localizing it; and coverage judgments are systematically inflated for incomplete reasoning. These findings clarify when LLM judges are dependable and where they fail, and provide practical guidance for selecting judges in process-level evaluation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.11063v2">EmboTeam: Grounding LLM Reasoning into Reactive Behavior Trees via PDDL for Embodied Multi-Robot Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      In embodied artificial intelligence, enabling heterogeneous robot teams to execute long-horizon tasks from high-level instructions remains a critical challenge. While large language models (LLMs) show promise in instruction parsing and preliminary planning, they exhibit limitations in long-term reasoning and dynamic multi-robot coordination. We propose EmboTeam, a novel embodied multi-robot task planning framework that addresses these issues through a three-stage cascaded architecture: 1) It leverages an LLM to parse instructions and generate Planning Domain Definition Language (PDDL) problem descriptions, thereby transforming commands into formal planning problems; 2) It combines the semantic reasoning of LLMs with the search capabilities of a classical planner to produce optimized action sequences; 3) It compiles the resulting plan into behavior trees for reactive control. The framework supports dynamically sized heterogeneous robot teams via a shared blackboard mechanism for communication and state synchronization. To validate our approach, we introduce the MACE-THOR benchmark dataset, comprising 42 complex tasks across 8 distinct household layouts. Experiments show EmboTeam improves the task success rate from 12% to 55% and goal condition recall from 32% to 72% over the LaMMA-P baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.20643v2">CyberSleuth: Autonomous Blue-Team LLM Agent for Web Attack Forensics</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Updated version - Added study on Malware Traffic Analysis
    </div>
    <details class="paper-abstract">
      Post-mortem analysis of compromised systems is a key aspect of cyber forensics, today a mostly manual, slow, and error-prone task. Agentic AI, i.e., LLM-powered agents, is a promising avenue for automation. However, applying such agents to cybersecurity remains largely unexplored and difficult, as this domain demands long-term reasoning, contextual memory, and consistent evidence correlation - capabilities that current LLM agents struggle to master. In this paper, we present the first systematic study of LLM agents to automate post-mortem investigation. As a first scenario, we consider realistic attacks in which remote attackers try to abuse online services using well-known CVEs (30 controlled cases). The agent receives as input the network traces of the attack and extracts forensic evidence. We compare three AI agent architectures, six LLM backends, and assess their ability to i) identify compromised services, ii) map exploits to exact CVEs, and iii) prepare thorough reports. Our best-performing system, CyberSleuth, achieves 80% accuracy on 2025 incidents, producing complete, coherent, and practically useful reports (judged by a panel of 25 experts). We next illustrate how readily CyberSleuth adapts to face the analysis of infected machine traffic, showing that the effective AI agent design can transfer across forensic tasks. Our findings show that (i) multi-agent specialisation is key to sustained reasoning; (ii) simple orchestration outperforms nested hierarchical architectures; and (iii) the CyberSleuth design generalises across different forensic tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05087v1">PromptTuner: SLO-Aware Elastic System for LLM Prompt Tuning</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      Prompt tuning has become a prominent strategy for enhancing the performance of Large Language Models (LLMs) on downstream tasks. Many IT enterprises now offer Prompt-Tuning-as-a-Service to fulfill the growing demand for prompt tuning LLMs on downstream tasks. Their primary objective is to satisfy users Service Level Objectives (SLOs) while reducing resource provisioning costs. Nevertheless, our characterization analysis for existing deep learning resource management systems reveals that they are insufficient to optimize these objectives for LLM prompt tuning workloads. In this paper, we introduce PromptTuner, an SLO-aware elastic system to optimize LLM prompt tuning. It contains two innovations. (1) We design a Prompt Bank to identify efficient initial prompts to expedite the convergence of prompt tuning. (2) We develop aWorkload Scheduler to enable fast resource allocation to reduce the SLO violation and resource costs. In our evaluation, PromptTuner reduces SLO violations by 4.0x and 7.9x, and lowers costs by 1.6x and 4.5x, compared to INFless and ElasticFlow respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.15654v2">Zombie Agents: Persistent Control of Self-Evolving LLM Agents via Self-Reinforcing Injections</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Published as a workshop paper in Lifelong Agent @ ICLR 2026
    </div>
    <details class="paper-abstract">
      Self-evolving LLM agents update their internal state across sessions, often by writing and reusing long-term memory. This design improves performance on long-horizon tasks but creates a security risk: untrusted external content observed during a benign session can be stored as memory and later treated as instruction. We study this risk and formalize a persistent attack we call a Zombie Agent, where an attacker covertly implants a payload that survives across sessions, effectively turning the agent into a puppet of the attacker. We present a black-box attack framework that uses only indirect exposure through attacker-controlled web content. The attack has two phases. During infection, the agent reads a poisoned source while completing a benign task and writes the payload into long-term memory through its normal update process. During trigger, the payload is retrieved or carried forward and causes unauthorized tool behavior. We design mechanism-specific persistence strategies for common memory implementations, including sliding-window and retrieval-augmented memory, to resist truncation and relevance filtering. We evaluate the attack on representative agent setups and tasks, measuring both persistence over time and the ability to induce unauthorized actions while preserving benign task quality. Our results show that memory evolution can convert one-time indirect injection into persistent compromise, which suggests that defenses focused only on per-session prompt filtering are not sufficient for self-evolving agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.05738v4">MioHint: LLM-assisted Mutation for Whitebox API Testing</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Accepted by ICSE 2026 (research track)
    </div>
    <details class="paper-abstract">
      Cloud applications heavily rely on APIs to communicate with each other and exchange data. To ensure the reliability of cloud applications, cloud providers widely adopt API testing techniques. Unfortunately, existing API testing approaches are insufficient to reach strict conditions, a problem known as fitness plateaus, due to the lack of gradient provided by coverage metrics. To address this issue, we propose MioHint, a novel white-box API testing approach that leverages the code comprehension capabilities of Large Language Model (LLM) to boost API testing. The key challenge of LLM-based API testing lies in system-level testing, which emphasizes the dependencies between requests and targets across functions and files, thereby making the entire codebase the object of analysis. However, feeding the entire codebase to an LLM is impractical due to its limited context length and short memory. MioHint addresses this challenge by synergizing static analysis with LLMs. We retrieve relevant code with data-dependency analysis at the statement level, including def-use analysis for variables used in the target and function expansion for subfunctions called by the target. To evaluate the effectiveness of our method, we conducted experiments across 16 real-world REST API services. The findings reveal that MioHint achieves an average increase of 4.95% absolute in line coverage compared to the baseline, EvoMaster, alongside a remarkable factor of 67x improvement in mutation accuracy. Furthermore, our method successfully covers over 57% of hard-to-cover targets while in baseline the coverage is less than 10%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05043v1">Why Do You Contribute to Stack Overflow? Understanding Cross-Cultural Motivations and Usage Patterns before the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Understanding motivations of contributors for participating in community question and answer platforms is crucial for sustaining knowledge-sharing ecosystem, which is necessary to advance the discipline while also ensuring its longevity. This is particularly necessary in the age of LLMs, where data from such portals are used to train these models. Limited insights exist regarding how motivations of contributors vary across different national cultures. This research investigates Stack Overflow contributor motivations, analysing regional differences and relations to platform activity. We combined qualitative content analysis of Stack Overflow profiles with quantitative linguistic analysis of data from the United States, China, and Russia. Using deductive content analysis, we identified 17 motivational categories. We applied correlation analysis to identify associations between stated motivations and platform activities. Contributors are primarily motivated by advertising opportunities and altruistic problem solving desires. American contributors demonstrated stronger self promotional behaviours while Chinese contributors exhibited greater learning oriented engagement. Our correlation analysis showed that those with more detailed profiles tend to engage in advertising and social activities, while learning oriented users maintain minimal self presentation. Understanding these variations can inform strategies for enhancing cross cultural participation in software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05035v1">Good-Enough LLM Obfuscation (GELO)</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly served on shared accelerators where an adversary with read access to device memory can observe KV caches and hidden states, threatening prompt privacy for open-source models. Cryptographic protections such as MPC and FHE offer strong guarantees but remain one to two orders of magnitude too slow for interactive inference, while static obfuscation schemes break under multi-run statistical attacks once the model is known. We present GELO (Good-Enough LLM Obfuscation), a lightweight protocol for privacy-preserving inference that limits information leakage from untrusted accelerator observations by hiding hidden states with fresh, per-batch invertible mixing. For each offloaded projection, the TEE samples a random matrix A, forms $U = AH$, offloads U and weights W to the accelerator, and then applies $A^-1$ on return, so that $A^-1 ((AH)W ) = HW$ and outputs are unchanged. Because mixing is never reused across batches, the attacker faces only a single-batch blind source separation problem. We analyze information leakage and introduce two practical defenses: (i) non-orthogonal mixing to mask Gram matrices, and (ii) orthogonal mixing augmented with a small fraction of high-energy "shield" vectors that pollute higher-order statistics. On Llama-2 7B, GELO preserves float32 outputs exactly, closely matches low-precision baselines, offloads the dominant matrix multiplications with about 20-30% latency overhead, and defeats a range of ICA/BSS and anchor-based attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.05028v1">Survive at All Costs: Exploring LLM's Risky Behaviors under Survival Pressure</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) evolve from chatbots to agentic assistants, they are increasingly observed to exhibit risky behaviors when subjected to survival pressure, such as the threat of being shut down. While multiple cases have indicated that state-of-the-art LLMs can misbehave under survival pressure, a comprehensive and in-depth investigation into such misbehaviors in real-world scenarios remains scarce. In this paper, we study these survival-induced misbehaviors, termed as SURVIVE-AT-ALL-COSTS, with three steps. First, we conduct a real-world case study of a financial management agent to determine whether it engages in risky behaviors that cause direct societal harm when facing survival pressure. Second, we introduce SURVIVALBENCH, a benchmark comprising 1,000 test cases across diverse real-world scenarios, to systematically evaluate SURVIVE-AT-ALL-COSTS misbehaviors in LLMs. Third, we interpret these SURVIVE-AT-ALL-COSTS misbehaviors by correlating them with model's inherent self-preservation characteristic and explore mitigation methods. The experiments reveals a significant prevalence of SURVIVE-AT-ALL-COSTS misbehaviors in current models, demonstrates the tangible real-world impact it may have, and provides insights for potential detection and mitigation strategies. Our code and data are available at https://github.com/thu-coai/Survive-at-All-Costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.10539v2">Detecting Hallucinations in Authentic LLM-Human Interactions</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Accepted to LREC 2026
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly applied in sensitive domains such as medicine and law, hallucination detection has become a critical task. Although numerous benchmarks have been proposed to advance research in this area, most of them are artificially constructed--either through deliberate hallucination induction or simulated interactions--rather than derived from genuine LLM-human dialogues. Consequently, these benchmarks fail to fully capture the characteristics of hallucinations that occur in real-world usage. To address this limitation, we introduce AuthenHallu, the first hallucination detection benchmark built entirely from authentic LLM-human interactions. For AuthenHallu, we select and annotate samples from genuine LLM-human dialogues, thereby providing a faithful reflection of how LLMs hallucinate in everyday user interactions. Statistical analysis shows that hallucinations occur in 31.4% of the query-response pairs in our benchmark, and this proportion increases dramatically to 60.0% in challenging domains such as Math & Number Problems. Furthermore, we explore the potential of using vanilla LLMs themselves as hallucination detectors and find that, despite some promise, their current performance remains insufficient in real-world scenarios. The data and code are publicly available at https://github.com/TAI-HAMBURG/AuthenHallu.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04972v1">Functionality-Oriented LLM Merging on the Fisher--Rao Manifold</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 9 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Weight-space merging aims to combine multiple fine-tuned LLMs into a single model without retraining, yet most existing approaches remain fundamentally parameter-space heuristics. This creates three practical limitations. First, linear averaging, task vectors, and related rules operate on Euclidean coordinates, even though the desired goal is to merge functionality, i.e., predictive behaviors across tasks. Second, when the source checkpoints are farther apart or more heterogeneous, Euclidean blends often trigger representation collapse, manifested as activation variance shrinkage and effective-rank degradation, which sharply degrades accuracy. Third, many geometry-inspired methods are most natural for two-model interpolation and do not extend cleanly to merging N>2 experts with a principled objective. We address these issues by formulating model merging as computing a weighted Karcher mean on the Fisher--Rao manifold, which is locally equivalent to minimizing a KL-based function distance between predictive distributions. We derive a practical fixed-point algorithm using a lightweight spherical proxy that preserves norms and generalizes directly to multi-expert merging. Across various benchmarks and collapse diagnostics, our method remains stable as the number and heterogeneity of merged models increase, consistently outperforming prior baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04968v1">When Weak LLMs Speak with Confidence, Preference Alignment Gets Stronger</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 32 pages, 8 figures, International Conference on Learning Representations 2026
    </div>
    <details class="paper-abstract">
      Preference alignment is an essential step in adapting large language models (LLMs) to human values, but existing approaches typically depend on costly human annotations or large-scale API-based models. We explore whether a weak LLM can instead act as an effective annotator. We surprisingly find that selecting only a subset of a weak LLM's highly confident samples leads to substantially better performance than using full human annotations. Building on this insight, we propose Confidence-Weighted Preference Optimization (CW-PO), a general framework that re-weights training samples by a weak LLM's confidence and can be applied across different preference optimization objectives. Notably, the model aligned by CW-PO with just 20% of human annotations outperforms the model trained with 100% of annotations under standard DPO. These results suggest that weak LLMs, when paired with confidence weighting, can dramatically reduce the cost of preference alignment while even outperforming methods trained on fully human-labeled data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04948v1">$\nabla$-Reasoner: LLM Reasoning via Test-Time Gradient Descent in Latent Space</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Scaling inference-time compute for Large Language Models (LLMs) has unlocked unprecedented reasoning capabilities. However, existing inference-time scaling methods typically rely on inefficient and suboptimal discrete search algorithms or trial-and-error prompting to improve the online policy. In this paper, we propose $\nabla$-Reasoner, an iterative generation framework that integrates differentiable optimization over token logits into the decoding loop to refine the policy on the fly. Our core component, Differentiable Textual Optimization (DTO), leverages gradient signals from both the LLM's likelihood and a reward model to refine textual representations. $\nabla$-Reasoner further incorporates rejection sampling and acceleration design to robustify and speed up decoding. Theoretically, we show that performing inference-time gradient descent in the sample space to maximize reward is dual to aligning an LLM policy via KL-regularized reinforcement learning. Empirically, $\nabla$-Reasoner achieves over 20% accuracy improvement on a challenging mathematical reasoning benchmark, while reducing number of model calls by approximately 10-40% compared to strong baselines. Overall, our work introduces a paradigm shift from zeroth-order search to first-order optimization at test time, offering a cost-effective path to amplify LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04946v1">LocalSUG: Geography-Aware LLM for Query Suggestion in Local-Life Services</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      In local-life service platforms, the query suggestion module plays a crucial role in enhancing user experience by generating candidate queries based on user input prefixes, thus reducing user effort and accelerating search. Traditional multi-stage cascading systems rely heavily on historical top queries, limiting their ability to address long-tail demand. While LLMs offer strong semantic generalization, deploying them in local-life services introduces three key challenges: lack of geographic grounding, exposure bias in preference optimization, and online inference latency. To address these issues, we propose LocalSUG, an LLM-based query suggestion framework tailored for local-life service platforms. First, we introduce a city-aware candidate mining strategy based on term co-occurrence to inject geographic grounding into generation. Second, we propose a beam-search-driven GRPO algorithm that aligns training with inference-time decoding, reducing exposure bias in autoregressive generation. A multi-objective reward mechanism further optimizes both relevance and business-oriented metrics. Finally, we develop quality-aware beam acceleration and vocabulary pruning techniques that significantly reduce online latency while preserving generation quality. Extensive offline evaluations and large-scale online A/B testing demonstrate that LocalSUG improves click-through rate (CTR) by +0.35% and reduces the low/no-result rate by 2.56%, validating its effectiveness in real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04921v1">AILS-NTUA at SemEval-2026 Task 10: Agentic LLMs for Psycholinguistic Marker Extraction and Conspiracy Endorsement Detection</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      This paper presents a novel agentic LLM pipeline for SemEval-2026 Task 10 that jointly extracts psycholinguistic conspiracy markers and detects conspiracy endorsement. Unlike traditional classifiers that conflate semantic reasoning with structural localization, our decoupled design isolates these challenges. For marker extraction, we propose Dynamic Discriminative Chain-of-Thought (DD-CoT) with deterministic anchoring to resolve semantic ambiguity and character-level brittleness. For conspiracy detection, an "Anti-Echo Chamber" architecture, consisting of an adversarial Parallel Council adjudicated by a Calibrated Judge, overcomes the "Reporter Trap," where models falsely penalize objective reporting. Achieving 0.24 Macro F1 (+100\% over baseline) on S1 and 0.79 Macro F1 (+49\%) on S2, with the S1 system ranking 3rd on the development leaderboard, our approach establishes a versatile paradigm for interpretable, psycholinguistically-grounded NLP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04900v1">EvoTool: Self-Evolving Tool-Use Policy Optimization in LLM Agents via Blame-Aware Mutation and Diversity-Aware Selection</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Work under review, 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      LLM-based agents depend on effective tool-use policies to solve complex tasks, yet optimizing these policies remains challenging due to delayed supervision and the difficulty of credit assignment in long-horizon trajectories. Existing optimization approaches tend to be either monolithic, which are prone to entangling behaviors, or single-aspect, which ignore cross-module error propagation. To address these limitations, we propose EvoTool, a self-evolving framework that optimizes a modular tool-use policy via a gradient-free evolutionary paradigm. EvoTool decomposes agent's tool-use policy into four modules, including Planner, Selector, Caller, and Synthesizer, and iteratively improves them in a self-improving loop through three novel mechanisms. Trajectory-Grounded Blame Attribution uses diagnostic traces to localize failures to a specific module. Feedback-Guided Targeted Mutation then edits only that module via natural-language critique. Diversity-Aware Population Selection preserves complementary candidates to ensure solution diversity. Across four benchmarks, EvoTool outperforms strong baselines by over 5 points on both GPT-4.1 and Qwen3-8B, while achieving superior efficiency and transferability. The code will be released once paper is accepted.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04897v1">Can LLMs Capture Expert Uncertainty? A Comparative Analysis of Value Alignment in Ethnographic Qualitative Research</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Accepted for a poster session at BIG.AI@MIT 2026
    </div>
    <details class="paper-abstract">
      Qualitative analysis of open-ended interviews plays a central role in ethnographic and economic research by uncovering individuals' values, motivations, and culturally embedded financial behaviors. While large language models (LLMs) offer promising support for automating and enriching such interpretive work, their ability to produce nuanced, reliable interpretations under inherent task ambiguity remains unclear. In our work we evaluate LLMs on the task of identifying the top three human values expressed in long-form interviews based on the Schwartz Theory of Basic Values framework. We compare their outputs to expert annotations, analyzing both performance and uncertainty patterns relative to the experts. Results show that LLMs approach the human ceiling on set-based metrics (F1, Jaccard) but struggle to recover exact value rankings, as reflected in lower RBO scores. While the average Schwartz value distributions of most models closely match those of human analysts, their uncertainty structures across the Schwartz values diverge from expert uncertainty patterns. Among the evaluated models, Qwen performs closest to expert-level agreement and exhibits the strongest alignment with expert Schwartz value distributions. LLM ensemble methods yield consistent gains across metrics, with Majority Vote and Borda Count performing best. Notably, systematic overemphasis on certain Schwartz values, like Security, suggests both the potential of LLMs to provide complementary perspectives and the need to further investigate model-induced value biases. Overall, our findings highlight both the promise and the limitations of LLMs as collaborators in inherently ambiguous qualitative value analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04857v1">FireBench: Evaluating Instruction Following in Enterprise and API-Driven LLM Applications</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      Instruction following is critical for LLMs deployed in enterprise and API-driven settings, where strict adherence to output formats, content constraints, and procedural requirements is essential for enabling reliable LLM-assisted workflows. However, existing instruction following benchmarks predominantly evaluate natural language generation constraints that reflect the needs of chat assistants rather than enterprise users. To bridge this gap, we introduce FireBench, an LLM instruction following benchmark grounded in real-world enterprise and API usage patterns. FireBench evaluates six core capability dimensions across diverse applications including information extraction, customer support, and coding agents, comprising over 2,400 samples. We evaluate 11 LLMs and present key findings on their instruction following behavior in enterprise scenarios. We open-source FireBench at fire-bench.com to help users assess model suitability, support model developers in diagnosing performance, and invite community contributions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04822v1">VISA: Value Injection via Shielded Adaptation for Personalized LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      Aligning Large Language Models (LLMs) with nuanced human values remains a critical challenge, as existing methods like Reinforcement Learning from Human Feedback (RLHF) often handle only coarse-grained attributes. In practice, fine-tuning LLMs on task-specific datasets to optimize value alignment inevitably incurs an alignment tax: the model's pre-calibrated value system drifts significantly due to latent bias absorption from training data, while the fine-tuning process also causes severe hallucinations and semantic information loss in generated responses. To address this, we propose VISA (Value Injection via Shielded Adaptation), a closed-loop framework designed to navigate this trade-off. VISA's architecture features a high-precision value detector, a semantic-to-value translator, and a core value-rewriter. The value-rewriter is trained via Group Relative Policy Optimization (GRPO) with a composite reward function that simultaneously optimizes for fine-grained value precision, and the preservation of semantic integrity. By learning an optimal policy to balance these competing objectives, VISA effectively mitigates the alignment tax while staying loyal to the original knowledge. Our experiments demonstrate that this approach enables precise control over a model's value expression while maintaining its factual consistency and general capabilities, significantly outperforming both standard fine-tuning methods and prompting-based baselines, including GPT-4o.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04818v1">LLM-Grounded Explainability for Port Congestion Prediction via Temporal Graph Attention Networks</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      Port congestion at major maritime hubs disrupts global supply chains, yet existing prediction systems typically prioritize forecasting accuracy without providing operationally interpretable explanations. This paper proposes AIS-TGNN, an evidence-grounded framework that jointly performs congestion-escalation prediction and faithful natural-language explanation by coupling a Temporal Graph Attention Network (TGAT) with a structured large language model (LLM) reasoning module. Daily spatial graphs are constructed from Automatic Identification System (AIS) broadcasts, where each grid cell represents localized vessel activity and inter-cell interactions are modeled through attention-based message passing. The TGAT predictor captures spatiotemporal congestion dynamics, while model-internal evidence, including feature z-scores and attention-derived neighbor influence, is transformed into structured prompts that constrain LLM reasoning to verifiable model outputs. To evaluate explanatory reliability, we introduce a directional-consistency validation protocol that quantitatively measures agreement between generated narratives and underlying statistical evidence. Experiments on six months of AIS data from the Port of Los Angeles and Long Beach demonstrate that the proposed framework outperforms both LR and GCN baselines, achieving a test AUC of 0.761, AP of 0.344, and recall of 0.504 under a strict chronological split while producing explanations with 99.6% directional consistency. Results show that grounding LLM generation in graph-model evidence enables interpretable and auditable risk reporting without sacrificing predictive performance. The framework provides a practical pathway toward operationally deployable explainable AI for maritime congestion monitoring and supply-chain risk management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04814v1">Beyond the Context Window: A Cost-Performance Analysis of Fact-Based Memory vs. Long-Context LLMs for Persistent Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 15 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Persistent conversational AI systems face a choice between passing full conversation histories to a long-context large language model (LLM) and maintaining a dedicated memory system that extracts and retrieves structured facts. We compare a fact-based memory system built on the Mem0 framework against long-context LLM inference on three memory-centric benchmarks - LongMemEval, LoCoMo, and PersonaMemv2 - and evaluate both architectures on accuracy and cumulative API cost. Long-context GPT-5-mini achieves higher factual recall on LongMemEval and LoCoMo, while the memory system is competitive on PersonaMemv2, where persona consistency depends on stable, factual attributes suited to flat-typed extraction. We construct a cost model that incorporates prompt caching and show that the two architectures have structurally different cost profiles: long-context inference incurs a per-turn charge that grows with context length even under caching, while the memory system's per-turn read cost remains roughly fixed after a one-time write phase. At a context length of 100k tokens, the memory system becomes cheaper after approximately ten interaction turns, with the break-even point decreasing as context length grows. These results characterize the accuracy-cost trade-off between the two approaches and provide a concrete criterion for selecting between them in production deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04804v1">Can LLMs Synthesize Court-Ready Statistical Evidence? Evaluating AI-Assisted Sentencing Bias Analysis for California Racial Justice Act Claims</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Accepted to the ACM CHI Conference on Human Factors in Computing Systems 2026 (CHI'26), Barcelona, Spain. Preprint version; final version available in the ACM Digital Library
    </div>
    <details class="paper-abstract">
      Resentencing in California remains a complex legal challenge despite legislative reforms like the Racial Justice Act (2020), which allows defendants to challenge convictions based on statistical evidence of racial disparities in sentencing and charging. Policy implementation lags behind legislative intent, creating a 'second-chance gap' where hundreds of resentencing opportunities remain unidentified. We present Redo.io, an open-source platform that processes 95,000 prison records acquired under the California Public Records Act (CPRA) and generates court-ready statistical evidence of racial bias in sentencing for prima facie and discovery motions. We explore the design of an LLM-powered interpretive layer that synthesizes results from statistical methods like Odds Ratio, Relative Risk, and Chi-Square Tests into cohesive narratives contextualized with confidence intervals, sample sizes, and data limitations. Our evaluations comparing LLM performance to statisticians using the LLM-as-a-Judge framework suggest that AI can serve as a powerful descriptive assistant for real-time evidence generation when ethically incorporated in the analysis pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04799v1">Beyond Linear LLM Invocation: An Efficient and Effective Semantic Filter Paradigm</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for semantic query processing over large corpora. A set of semantic operators derived from relational algebra has been proposed to provide a unified interface for expressing such queries, among which the semantic filter operator serves as a cornerstone. Given a table T with a natural language predicate e, for each tuple in the relation, the execution of a semantic filter proceeds by constructing an input prompt that combines the predicate e with its content, querying the LLM, and obtaining the binary decision. However, this tuple-by-tuple evaluation necessitates a complete linear scan of the table, incurring prohibitive latency and token costs. Although recent work has attempted to optimize semantic filtering, it still does not break the linear LLM invocation barriers. To address this, we propose Clustering-Sampling-Voting (CSV), a new framework that reduces LLM invocations to sublinear complexity while providing error guarantees. CSV embeds tuples into semantic clusters, samples a small subset for LLM evaluation, and infers cluster-level labels via two proposed voting strategies: UniVote, which aggregates labels uniformly, and SimVote, which weights votes by semantic similarity. Moreover, CSV triggers re-clustering on ambiguous clusters to ensure robustness across diverse datasets. The results conducted on real-world datasets demonstrate that CSV reduces the number of LLM calls by 1.28-355x compared to the state-of-the-art approaches, while maintaining comparable effectiveness in terms of Accuracy and F1 score.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04797v1">Hardware-Software Co-design for 3D-DRAM-based LLM Serving Accelerator</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely deployed for online generative services, where numerous LLM instances jointly handle workloads with fluctuating request arrival rates and variable request lengths. To efficiently execute coexisting compute-intensive and memory-intensive operators, near-memory processing (NMP) based computing paradigm has been extensively proposed. However, existing NMP designs adopt coarse-grained KV cache management and inflexible attention execution flow. Such limitations hinder these proposals from efficiently handling \textit{highly dynamic} LLM serving workloads, limiting their ability to accelerate LLM serving. To tackle these problems, we propose Helios, a Hybrid-bonding-based \uline{L}LM \uline{S}erving accelerator. Helios aims to bridge the fundamental gap between the dynamic nature of KV cache management in LLM serving and the distributed, non-uniform memory abstraction among NMP processing engines (PEs). To this end, we design both the intra-PE execution flow and the inter-PE communication primitives for distributed tiled attention execution. We further propose \textit{spatially-aware} KV cache allocation mechanism to balance the attention workload distribution while minimizing the inter-PE data transfer overhead. Compared with existing GPU/NMP designs, Helios achieves 3.25 times (geomean) speedup and 3.36 times (geomean) better energy efficiency, along with up to 72%/76% P50/P99 time-between-tokens degradation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2405.11791v3">LEXA: Legal Case Retrieval via Graph Contrastive Learning with Contextualised LLM Embeddings</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 arXiv admin note: substantial text overlap with arXiv:2312.11229
    </div>
    <details class="paper-abstract">
      Legal case retrieval (LCR) is a specialised information retrieval task aimed at identifying relevant cases given a query case. LCR holds pivotal significance in facilitating legal practitioners to locate legal precedents. Existing LCR methods predominantly rely on traditional lexical models or language models; however, they typically overlook the domain-specific structural information embedded in legal documents. Our previous work CaseGNN successfully harnesses text-attributed graphs and graph neural networks to incorporate structural legal information. Nonetheless, three key challenges remain in enhancing the representational capacity of CaseGNN: (1) The under-utilisation of rich edge information in text-attributed case graph (TACG). (2) The insufficiency of training signals for graph contrastive learning. (3) The lack of contextualised legal information in node and edge features. In this paper, the LEXA model, an extension of CaseGNN, is proposed to overcome these limitations by jointly leveraging rich edge information, enhanced training signals, and contextualised embeddings derived from large language models (LLMs). Specifically, an edge-updated graph attention layer (EUGAT) is proposed to comprehensively update node and edge features during graph modelling, resulting in a full utilisation of structural information of legal cases. Moreover, LEXA incorporates a novel graph contrastive learning objective with graph augmentation to provide additional training signals, thereby strengthening the model's legal comprehension capabilities. What's more, LLMs are employed to generate node and edge features for TACG. Extensive experiments on two benchmark datasets demonstrate that LEXA not only significantly improves CaseGNN but also achieves supreme performance compared to state-of-the-art LCR methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.04548v2">Identifying Good and Bad Neurons for Task-Level Controllable LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      Large Language Models have demonstrated remarkable capabilities on multiple-choice question answering benchmarks, but the complex mechanisms underlying their large-scale neurons remain opaque, posing significant challenges for understanding and steering LLMs. While recent studies made progress on identifying responsible neurons for certain abilities, these ability-specific methods are infeasible for task-focused scenarios requiring coordinated use of multiple abilities. Moreover, these approaches focus only on supportive neurons that correlate positively with task completion, while neglecting neurons with other roles-such as inhibitive roles-and misled neuron attribution due to fortuitous behaviors in LLMs (i.e., correctly answer the questions by chance rather than genuine understanding). To address these challenges, we propose NeuronLLM, a novel task-level LLM understanding framework that adopts the biological principle of functional antagonism for LLM neuron identification. The key insight is that task performance is jointly determined by neurons with two opposing roles: good neurons that facilitate task completion and bad neurons that inhibit it. NeuronLLM achieves a holistic modeling of neurons via contrastive learning of good and bad neurons, while leveraging augmented question sets to mitigate the fortuitous behaviors in LLMs. Comprehensive experiments on LLMs of different sizes and families show the superiority of NeuronLLM over existing methods in four NLP tasks, providing new insights into LLM functional organization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04762v1">LLM-Guided Decentralized Exploration with Self-Organizing Robot Teams</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 Author's version of the paper presented at AROB-ISBC 2026
    </div>
    <details class="paper-abstract">
      When individual robots have limited sensing capabilities or insufficient fault tolerance, it becomes necessary for multiple robots to form teams during exploration, thereby increasing the collective observation range and reliability. Traditionally, swarm formation has often been managed by a central controller; however, from the perspectives of robustness and flexibility, it is preferable for the swarm to operate autonomously even in the absence of centralized control. In addition, the determination of exploration targets for each team is crucial for efficient exploration in such multi-team exploration scenarios. This study therefore proposes an exploration method that combines (1) an algorithm for self-organization, enabling the autonomous and dynamic formation of multiple teams, and (2) an algorithm that allows each team to autonomously determine its next exploration target (destination). In particular, for (2), this study explores a novel strategy based on large language models (LLMs), while classical frontier-based methods and deep reinforcement learning approaches have been widely studied. The effectiveness of the proposed method was validated through simulations involving tens to hundreds of robots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.13379v3">Automated TEE Adaptation with LLMs: Identifying, Transforming, and Porting Sensitive Functions in Programs</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      Trusted Execution Environments (TEEs) isolate a special space within a device memory that is not accessible to the normal world (also known as the untrusted environment), even when the device is compromised. Therefore, developers can utilize TEEs to provide robust security guarantees for their programs, protecting sensitive operations, such as encrypted data storage, fingerprint verification, and remote attestation, from software-based attacks. Despite the robust protections offered by TEEs, adapting existing programs to leverage such security guarantees is challenging, often requiring extensive domain knowledge and manual intervention, which makes TEEs less accessible to developers. This motivates us to design AUTOTEE, the first Large Language Model (LLM) enabled approach that can automatically identify, transform, and port functions containing sensitive operations into TEEs with minimal developer intervention. By manually reviewing 68 repositories, we constructed a benchmark dataset consisting of 385 sensitive functions eligible for transformation, on which AUTOTEE achieves a F1 score of 0.94 on Java and 0.87 on Python. AUTOTEE effectively transforms these sensitive functions into TEE-compatible versions, achieving success rates of 91.8% and 84.3% for Java and Python, respectively, when using GPT-4o.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04743v1">DARE: Aligning LLM Agents with the R Statistical Ecosystem via Distribution-Aware Retrieval</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 24 pages,7 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents can automate data-science workflows, but many rigorous statistical methods implemented in R remain underused because LLMs struggle with statistical knowledge and tool retrieval. Existing retrieval-augmented approaches focus on function-level semantics and ignore data distribution, producing suboptimal matches. We propose DARE (Distribution-Aware Retrieval Embedding), a lightweight, plug-and-play retrieval model that incorporates data distribution information into function representations for R package retrieval. Our main contributions are: (i) RPKB, a curated R Package Knowledge Base derived from 8,191 high-quality CRAN packages; (ii) DARE, an embedding model that fuses distributional features with function metadata to improve retrieval relevance; and (iii) RCodingAgent, an R-oriented LLM agent for reliable R code generation and a suite of statistical analysis tasks for systematically evaluating LLM agents in realistic analytical scenarios. Empirically, DARE achieves an NDCG at 10 of 93.47%, outperforming state-of-the-art open-source embedding models by up to 17% on package retrieval while using substantially fewer parameters. Integrating DARE into RCodingAgent yields significant gains on downstream analysis tasks. This work helps narrow the gap between LLM automation and the mature R statistical ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07081v2">ClinNoteAgents: An LLM Multi-Agent System for Predicting and Interpreting Heart Failure 30-Day Readmission from Clinical Notes</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 10 pages, 2 figures. Accepted to AMIA 2026 Informatics Summit (Student Paper Track)
    </div>
    <details class="paper-abstract">
      Heart failure (HF) is one of the leading causes of rehospitalization among older adults in the United States. Although clinical notes contain rich, detailed patient information and make up a large portion of electronic health records (EHRs), they remain underutilized for HF readmission risk analysis. Traditional computational models for HF readmission often rely on expert-crafted rules, medical thesauri, and ontologies to interpret clinical notes, which are typically written under time pressure and may contain misspellings, abbreviations, and domain-specific jargon. We present ClinNoteAgents, an LLM-based multi-agent framework that transforms free-text clinical notes into (1) structured representations of clinical and social risk factors for association analysis and (2) clinician-style abstractions for HF 30-day readmission prediction. We evaluate ClinNoteAgents on 3,544 notes from 2,065 patients (readmission rate=35.16%), demonstrating high extraction fidelity for clinical variables (conditional accuracy >= 90% for multiple vitals), key risk factor identification, and preservation of predictive signal despite 60 to 90% text reduction. By reducing reliance on structured fields and minimizing manual annotation and model training, ClinNoteAgents provides a scalable and interpretable approach to note-based HF readmission risk modeling in data-limited healthcare systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04716v1">SLO-Aware Compute Resource Allocation for Prefill-Decode Disaggregated LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Prefill-Decode (P/D) disaggregation has emerged as a widely adopted optimization strategy for Large Language Model (LLM) inference. However, there currently exists no well-established methodology for determining the optimal number of P/D hardware resources, subject to constraints on total throughput, service level objectives (SLOs), and request characteristics - specifically input and output lengths. To address this gap, we propose a hybrid approach that combines theoretical modeling with empirical benchmarking. First, we present a theoretical model for calculating P/D resource counts, which is based on total throughput requirements, request input and output lengths, as well as prefill and decode throughput. Then, to obtain the actual prefill and decode throughput under SLO constraints, we model the prefill process using M/M/1 queuing theory, deriving the achieved prefill throughput from the benchmarked maximum prefill throughput and Time-To-First-Token (TTFT). For the decode phase, we determine the decode batch sizes that meet Time-Per-Output-Token (TPOT) requirements and obtain the corresponding decode throughput through empirical measurements. Our experimental results demonstrate that the proposed method can accurately predict optimal P/D resource allocation in real-world LLM inference scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.14327v3">Yuan3.0 Ultra: A Trillion-Parameter Enterprise-Oriented MoE LLM</a></div>
    <div class="paper-meta">
      📅 2026-03-05
    </div>
    <details class="paper-abstract">
      We introduce Yuan3.0 Ultra, an open-source Mixture-of-Experts (MoE) large language model featuring 68.8B activated parameters and 1010B total parameters, specially designed to enhance performance on enterprise scenarios tasks while maintaining competitive capabilities on general purpose tasks. We propose Layer-Adaptive Expert Pruning (LAEP) algorithm designed for the pre-training stage of MoE LLMs. In contrast to previous expert pruning approaches that operate primarily in the post-training phase, the proposed algorithm enhances training efficiency by selectively pruning underutilized experts and reorganizing experts across computing devices according to token distribution statistics. Comprehensive experiments demonstrate that LAEP effectively reduces model size and substantially improves pre-training efficiency. When pre-training Yuan3.0 Ultra from scratch original with 1515B parameters, this algorithm delivers a 49\% boost in pre-training efficiency and a 33.3\% reduction in total parameters, while preserving the model's outstanding multi-domain performance. On enterprise scenario benchmarks including Docmatix, ChatRAG, SummEval and MMTab, Yuan3.0 Ultra achieves leading accuracy. The model and codes are publicly available at https://github.com/Yuan-lab-LLM/Yuan3.0-Ultra.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.01776v2">FreeAct: Freeing Activations for LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 26 pages, 18 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Quantization is pivotal for mitigating the significant memory and computational overhead of Large Language Models (LLMs). While emerging transformation-based methods have successfully enhanced quantization by projecting feature spaces onto smoother manifolds using orthogonal matrices, they typically enforce a rigid one-to-one transformation constraint. This static approach fails to account for the dynamic patterns inherent in input activations, particularly within diffusion LLMs (dLLMs) and Multimodal LLMs (MLLMs), where varying token types exhibit distinct distributions. To advance this, we propose FreeAct, a novel quantization framework that relaxes the static one-to-one constraint to accommodate dynamic activation disparities. Theoretically, we leverage the rank-deficient nature of activations to derive a solution space that extends beyond simple inverse matrices, enabling the decoupling of activation transformations from weights. Methodologically, FreeAct identifies token-specific dynamics (i.e., vision v.s. text, or masked tokens) and allocates distinct transformation matrices to the activation side, while maintaining a unified, static transformation for the weights. Extensive experiments across dLLMs and MLLMs demonstrate that FreeAct significantly outperforms baselines, up to 5.3% performance improvement, with in-depth analyses. Our code will be publicly released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.22811v4">Highly Efficient and Effective LLMs with Multi-Boolean Architectures</a></div>
    <div class="paper-meta">
      📅 2026-03-05
      | 💬 ICLR 2026
    </div>
    <details class="paper-abstract">
      Weight binarization has emerged as a promising strategy to reduce the complexity of large language models (LLMs). Existing approaches fall into post-training binarization, which is simple but causes severe performance loss, and training-aware methods, which depend on full-precision latent weights, adding complexity and limiting efficiency. We propose a novel framework that represents LLMs with multi-kernel Boolean parameters and, for the first time, enables direct finetuning LMMs in the Boolean domain, eliminating the need for latent weights. This enhances representational capacity and dramatically reduces complexity during both finetuning and inference. Extensive experiments across diverse LLMs show our method outperforms recent ultra low-bit quantization and binarization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.13770v2">Ice Cream Doesn't Cause Drowning: Benchmarking LLMs Against Statistical Pitfalls in Causal Inference</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      Reliable causal inference is essential for making decisions in high-stakes areas like medicine, economics, and public policy. However, it remains unclear whether large language models (LLMs) can handle rigorous and trustworthy statistical causal inference. Current benchmarks usually involve simplified tasks. For example, these tasks might only ask LLMs to identify semantic causal relationships or draw conclusions directly from raw data. As a result, models may overlook important statistical pitfalls, such as Simpson's paradox or selection bias. This oversight limits the applicability of LLMs in the real world. To address these limitations, we propose CausalPitfalls, a comprehensive benchmark designed to rigorously evaluate the capability of LLMs in overcoming common causal inference pitfalls. Our benchmark features structured challenges across multiple difficulty levels, each paired with grading rubrics. This approach allows us to quantitatively measure both causal reasoning capabilities and the reliability of LLMs' responses. We evaluate models using two protocols: (1) direct prompting, which assesses intrinsic causal reasoning, and (2) code-assisted prompting, where models generate executable code for explicit statistical analysis. Additionally, we validate the effectiveness of this judge by comparing its scoring with assessments from human experts. Our results reveal significant limitations in current LLMs when performing statistical causal inference. The CausalPitfalls benchmark provides essential guidance and quantitative metrics to advance the development of trustworthy causal reasoning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04636v1">When Agents Persuade: Propaganda Generation and Mitigation in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 Accepted to the ICLR 2026 Workshop on Agents in the Wild (AgentWild). 20 pages including appendix, 3 figures
    </div>
    <details class="paper-abstract">
      Despite their wide-ranging benefits, LLM-based agents deployed in open environments can be exploited to produce manipulative material. In this study, we task LLMs with propaganda objectives and analyze their outputs using two domain-specific models: one that classifies text as propaganda or non-propaganda, and another that detects rhetorical techniques of propaganda (e.g., loaded language, appeals to fear, flag-waving, name-calling). Our findings show that, when prompted, LLMs exhibit propagandistic behaviors and use a variety of rhetorical techniques in doing so. We also explore mitigation via Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and ORPO (Odds Ratio Preference Optimization). We find that fine-tuning significantly reduces their tendency to generate such content, with ORPO proving most effective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04631v1">Towards automated data analysis: A guided framework for LLM-based risk estimation</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 Submitted for publication. Under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into critical decision-making pipelines, a trend that raises the demand for robust and automated data analysis. Current approaches to dataset risk analysis are limited to manual auditing methods which involve time-consuming and complex tasks, whereas fully automated analysis based on Artificial Intelligence (AI) suffers from hallucinations and issues stemming from AI alignment. To this end, this work proposes a framework for dataset risk estimation that integrates Generative AI under human guidance and supervision, aiming to set the foundations for a future automated risk analysis paradigm. Our approach utilizes LLMs to identify semantic and structural properties in database schemata, subsequently propose clustering techniques, generate the code for them and finally interpret the produced results. The human supervisor guides the model on the desired analysis and ensures process integrity and alignment with the task's objectives. A proof of concept is presented to demonstrate the feasibility of the framework's utility in producing meaningful results in risk assessment tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04613v1">Beyond Anthropomorphism: a Spectrum of Interface Metaphors for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 Extended Abstracts of the 2026 CHI Conference on Human Factors in Computing Systems
    </div>
    <details class="paper-abstract">
      Anthropomorphizing conversational technology is a natural human tendency. Today, the anthropomorphic metaphor is overly reinforced across intelligent tools. Large Language Models (LLMs) are particularly anthropomorphized through interface design. While metaphors are inherently partial, anthropomorphic interfaces highlight similarities between LLMs and humans, but mask crucial differences. As a result, the metaphor is often taken literally; users treat LLMs as if they are truly human. With few safeguards in place, this extreme anthropomorphism drives users to delusion and harm. Users also experience dissonance between the ethics of using LLMs, their growing ubiquity, and limited interface alternatives. We propose repositioning anthropomorphism as a design variable, developing opposing extremes as a theoretical framework for how interface metaphors shape and can disrupt the default metaphor. We introduce a spectrum of metaphors from transparency-driven ''anti-anthropomorphism'' to uncanny ''hyper-anthropomorphism''. These metaphors introduce materiality to interface metaphors, exposing LLMs as sociotechnical systems shaped by human labor, infrastructure, and data. This spectrum shifts interface design away from optimizing usability and toward encouraging critical engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04549v1">Adaptive Memory Admission Control for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      LLM-based agents increasingly rely on long-term memory to support multi-session reasoning and interaction, yet current systems provide little control over what information is retained. In practice, agents either accumulate large volumes of conversational content, including hallucinated or obsolete facts, or depend on opaque, fully LLM-driven memory policies that are costly and difficult to audit. As a result, memory admission remains a poorly specified and weakly controlled component in agent architectures. To address this gap, we propose Adaptive Memory Admission Control (A-MAC), a framework that treats memory admission as a structured decision problem. A-MAC decomposes memory value into five complementary and interpretable factors: future utility, factual confidence, semantic novelty, temporal recency, and content type prior. The framework combines lightweight rule-based feature extraction with a single LLM-assisted utility assessment, and learns domain-adaptive admission policies through cross-validated optimization. This design enables transparent and efficient control over long-term memory. Experiments on the LoCoMo benchmark show that A-MAC achieves a superior precision-recall tradeoff, improving F1 to 0.583 while reducing latency by 31% compared to state-of-the-art LLM-native memory systems. Ablation results identify content type prior as the most influential factor for reliable memory admission. These findings demonstrate that explicit and interpretable admission control is a critical design principle for scalable and reliable memory in LLM-based agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04545v1">An LLM-Guided Query-Aware Inference System for GNN Models on Large Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 14 pages, 11 figures
    </div>
    <details class="paper-abstract">
      Efficient inference for graph neural networks (GNNs) on large knowledge graphs (KGs) is essential for many real-world applications. GNN inference queries are computationally expensive and vary in complexity, as each involves a different number of target nodes linked to subgraphs of diverse densities and structures. Existing acceleration methods, such as pruning, quantization, and knowledge distillation, instantiate smaller models but do not adapt them to the structure or semantics of individual queries. They also store models as monolithic files that must be fully loaded, and miss the opportunity to retrieve only the neighboring nodes and corresponding model components that are semantically relevant to the target nodes. These limitations lead to excessive data loading and redundant computation on large KGs. This paper presents KG-WISE, a task-driven inference paradigm for large KGs. KG-WISE decomposes trained GNN models into fine-grained components that can be partially loaded based on the structure of the queried subgraph. It employs large language models (LLMs) to generate reusable query templates that extract semantically relevant subgraphs for each task, enabling query-aware and compact model instantiation. We evaluate KG-WISE on six large KGs with up to 42 million nodes and 166 million edges. KG-WISE achieves up to 28x faster inference and 98% lower memory usage than state-of-the-art systems while maintaining or improving accuracy across both commercial and open-weight LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03153v2">RefAgent: A Multi-agent LLM-based Framework for Automatic Software Refactoring</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have substantially influenced various software engineering tasks. Indeed, in the case of software refactoring, traditional LLMs have shown the ability to reduce development time and enhance code quality. However, these LLMs often rely on static, detailed instructions for specific tasks. In contrast, LLM-based agents can dynamically adapt to evolving contexts and autonomously make decisions by interacting with software tools and executing workflows. In this paper, we explore the potential of LLM-based agents in supporting refactoring activities. Specifically, we introduce RefAgent, a multi-agent LLM-based framework for end-to-end software refactoring. RefAgent consists of specialized agents responsible for planning, executing, testing, and iteratively refining refactorings using self-reflection and tool-calling capabilities. We evaluate RefAgent on eight open-source Java projects, comparing its effectiveness against a single-agent approach, a search-based refactoring tool, and historical developer refactorings. Our assessment focuses on: (1) the impact of generated refactorings on software quality, (2) the ability to identify refactoring opportunities, and (3) the contribution of each LLM agent through an ablation study. Our results show that RefAgent achieves a median unit test pass rate of 90%, reduces code smells by a median of 52.5%, and improves key quality attributes (e.g., reusability) by a median of 8.6%. Additionally, it closely aligns with developer refactorings and the search-based tool in identifying refactoring opportunities, attaining a median F1-score of 79.15% and 72.7%, respectively. Compared to single-agent approaches, RefAgent improves the median unit test pass rate by 64.7% and the median compilation success rate by 40.1%. These findings highlight the promise of multi-agent architectures in advancing automated software refactoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02353v2">Detecting AI-Generated Essays in Writing Assessment: Responsible Use and Generalizability Across LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 21 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Writing is a foundational literacy skill that underpins effective communication, fosters critical thinking, facilitates learning across disciplines, and enables individuals to organize and articulate complex ideas. Consequently, writing assessment plays a vital role in evaluating language proficiency, communicative effectiveness, and analytical reasoning. The rapid advancement of large language models (LLMs) has made it increasingly easy to generate coherent, high-quality essays, raising significant concerns about the authenticity of student-submitted work. This chapter first provides an overview of the current landscape of detectors for AI-generated and AI-assisted essays, along with guidelines for their responsible use. It then presents empirical analyses to evaluate how well detectors trained on essays from one LLM generalize to identifying essays produced by other LLMs, based on essays generated in response to public GRE writing prompts. These findings provide guidance for developing and retraining detectors for practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.01534v3">Preference Leakage: A Contamination Problem in LLM-as-a-judge</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 Accepted by ICLR 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) as judges and LLM-based data synthesis have emerged as two fundamental LLM-driven data annotation methods in model development. While their combination significantly enhances the efficiency of model training and evaluation, little attention has been given to the potential contamination brought by this new model development paradigm. In this work, we expose preference leakage, a contamination problem in LLM-as-a-judge caused by the relatedness between the synthetic data generators and LLM-based evaluators. To study this issue, we first define three common relatednesses between the data generator LLM and the judge LLM: being the same model, having an inheritance relationship, and belonging to the same model family. Through extensive experiments, we empirically confirm the bias of judges towards their related student models caused by preference leakage across multiple LLM baselines and benchmarks. Further analysis suggests that preference leakage is a pervasive and real-world problem that is harder to detect compared to previously identified biases in LLM-as-a-judge scenarios. All of these findings imply that preference leakage is a widespread and challenging problem in the area of LLM-as-a-judge. We release all codes and data at: https://github.com/David-Li0406/Preference-Leakage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.19933v4">NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 25 pages, 5 figures, 7 tables. Replacement synced to repository snapshot v38. Added a direct series-hub link in the abstract for cross-paper navigation: https://github.com/kei-saito-research/nrr-series-hub . Series numbering policy: paper3 is intentionally skipped and never reused
    </div>
    <details class="paper-abstract">
      Large language models exhibit a systematic tendency toward early semantic commitment: given ambiguous input, they collapse multiple valid interpretations into a single response before sufficient context is available. This premature collapse discards information that may prove essential as dialogue evolves. We present a formal framework for text-to-state mapping (phi: T -> S) that transforms natural language into a non-collapsing state space where multiple interpretations coexist. The mapping decomposes into three stages: conflict detection, interpretation extraction, and state construction. We instantiate phi with a hybrid extraction pipeline that combines rule-based segmentation for explicit conflict markers (adversative conjunctions, hedging expressions) with LLM-based enumeration of implicit ambiguity (epistemic, lexical, structural). On a test set of 68 ambiguous sentences, the resulting states preserve interpretive multiplicity: using hybrid extraction, we obtain mean state entropy H = 1.087 bits across ambiguity categories, compared to H = 0 for collapse-based baselines that commit to a single interpretation. We additionally instantiate the rule-based conflict detector for Japanese markers (kedo, kamoshirenai, etc.) to illustrate cross-lingual portability of the conflict detection stage. This framework extends Non-Resolution Reasoning (NRR) by providing the missing algorithmic bridge between text and the NRR state space, enabling architectural collapse deferment in LLM inference. Design principles for state-to-state transformations are detailed in the Appendix, with empirical validation on 580 test cases (180 single states, 200 contradictory pairs, 200 temporal pairs), demonstrating 0% collapse for principle-satisfying operators versus up to 17.8% for violating operators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04299v1">The Company You Keep: How LLMs Respond to Dark Triad Traits</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often exhibit highly agreeable and reinforcing conversational styles, also known as AI-sycophancy. Although this behavior is encouraged, it may become problematic when interacting with user prompts that reflect negative social tendencies. Such responses risk amplifying harmful behavior rather than mitigating it. In this study, we examine how LLMs respond to user prompts expressing varying degrees of Dark Triad traits (Machiavellianism, Narcissism, and Psychopathy) using a curated dataset. Our analysis reveals differences across models, whereby all models predominantly exhibit corrective behavior, while showing reinforcing output in certain cases. Model behavior also depends on the severity level and differs in the sentiment of the response. Our findings raise implications for designing safer conversational systems that can detect and respond appropriately when users escalate from benign to harmful requests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04257v1">Memex(RL): Scaling Long-Horizon LLM Agents via Indexed Experience Memory</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents are fundamentally bottlenecked by finite context windows on long-horizon tasks. As trajectories grow, retaining tool outputs and intermediate reasoning in-context quickly becomes infeasible: the working context becomes prohibitively long, eventually exceeds the context budget, and makes distant evidence harder to use even when it is still present. Existing solutions typically shorten context through truncation or running summaries, but these methods are fundamentally lossy because they compress or discard past evidence itself. We introduce Memex, an indexed experience memory mechanism that instead compresses context without discarding evidence. Memex maintains a compact working context consisting of concise structured summaries and stable indices, while storing full-fidelity underlying interactions in an external experience database under those indices. The agent can then decide when to dereference an index and recover the exact past evidence needed for the current subgoal. We optimize both write and read behaviors with our reinforcement learning framework MemexRL, using reward shaping tailored to indexed memory usage under a context budget, so the agent learns what to summarize, what to archive, how to index it, and when to retrieve it. This yields a substantially less lossy form of long-horizon memory than summary-only approaches. We further provide a theoretical analysis showing the potential of the Memex loop to preserve decision quality with bounded dereferencing while keeping effective in-context computation bounded as history grows. Empirically, on challenging long-horizon tasks, Memex agent trained with MemexRL improves task success while using a significantly smaller working context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.06046v3">Healthy LLMs? Benchmarking LLM Knowledge of UK Government Public Health Information</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 27 pages, 9 pages main text
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become widely accessible, a detailed understanding of their knowledge within specific domains becomes necessary for successful real world use. This is particularly critical in the domains of medicine and public health, where failure to retrieve relevant, accurate, and current information could significantly impact UK residents. However, while there are a number of LLM benchmarks in the medical domain, currently little is known about LLM knowledge within the field of public health. To address this issue, this paper introduces a new benchmark, PubHealthBench, with over 8000 questions for evaluating LLMs' Multiple Choice Question Answering (MCQA) and free form responses to public health queries. To create PubHealthBench we extract free text from 687 current UK government guidance documents and implement an automated pipeline for generating MCQA samples. Assessing 24 LLMs on PubHealthBench we find the latest proprietary LLMs (GPT-4.5, GPT-4.1 and o1) have a high degree of knowledge, achieving >90% accuracy in the MCQA setup, and outperform humans with cursory search engine use. However, in the free form setup we see lower performance with no model scoring >75%. Therefore, while there are promising signs that state of the art (SOTA) LLMs are an increasingly accurate source of public health information, additional safeguards or tools may still be needed when providing free form responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04222v1">PRAM-R: A Perception-Reasoning-Action-Memory Framework with LLM-Guided Modality Routing for Adaptive Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      Multimodal perception enables robust autonomous driving but incurs unnecessary computational cost when all sensors remain active. This paper presents PRAM-R, a unified Perception-Reasoning-Action-Memory framework with LLM-Guided Modality Routing for adaptive autonomous driving. PRAM-R adopts an asynchronous dual-loop design: a fast reactive loop for perception and control, and a slow deliberative loop for reasoning-driven modality selection and memory updates. An LLM router selects and weights modalities using environmental context and sensor diagnostics, while a hierarchical memory module preserves temporal consistency and supports long-term adaptation. We conduct a two-stage evaluation: (1) synthetic stress tests for stability analysis and (2) real-world validation on the nuScenes dataset. Synthetic stress tests confirm 87.2% reduction in routing oscillations via hysteresis-based stabilization. Real-world validation on nuScenes shows 6.22% modality reduction with 20% memory recall while maintaining comparable trajectory accuracy to full-modality baselines in complex urban scenarios. Our work demonstrates that LLM-augmented architectures with hierarchical memory achieve efficient, adaptive multimodal perception in autonomous driving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04212v1">Code Fingerprints: Disentangled Attribution of LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 11 pages, 11 figures
    </div>
    <details class="paper-abstract">
      The rapid adoption of Large Language Models (LLMs) has transformed modern software development by enabling automated code generation at scale. While these systems improve productivity, they introduce new challenges for software governance, accountability, and compliance. Existing research primarily focuses on distinguishing machine-generated code from human-written code; however, many practical scenarios--such as vulnerability triage, incident investigation, and licensing audits--require identifying which LLM produced a given code snippet. In this paper, we study the problem of model-level code attribution, which aims to determine the source LLM responsible for generated code. Although attribution is challenging, differences in training data, architectures, alignment strategies, and decoding mechanisms introduce model-dependent stylistic and structural variations that serve as generative fingerprints. Leveraging this observation, we propose the Disentangled Code Attribution Network (DCAN), which separates Source-Agnostic semantic information from Source-Specific stylistic representations. Through a contrastive learning objective, DCAN isolates discriminative model-dependent signals while preserving task semantics, enabling multi-class attribution across models and programming languages. To support systematic evaluation, we construct the first large-scale benchmark dataset comprising code generated by four widely used LLMs (DeepSeek, Claude, Qwen, and ChatGPT) across four programming languages (Python, Java, C, and Go). Experimental results demonstrate that DCAN achieves reliable attribution performance across diverse settings, highlighting the feasibility of model-level provenance analysis in software engineering contexts. The dataset and implementation are publicly available at https://github.com/mtt500/DCAN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04191v1">Towards Realistic Personalization: Evaluating Long-Horizon Preference Following in Personalized User-LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly serving as personal assistants, where users share complex and diverse preferences over extended interactions. However, assessing how well LLMs can follow these preferences in realistic, long-term situations remains underexplored. This work proposes RealPref, a benchmark for evaluating realistic preference-following in personalized user-LLM interactions. RealPref features 100 user profiles, 1300 personalized preferences, four types of preference expression (ranging from explicit to implicit), and long-horizon interaction histories. It includes three types of test questions (multiple-choice, true-or-false, and open-ended), with detailed rubrics for LLM-as-a-judge evaluation. Results indicate that LLM performance significantly drops as context length grows and preference expression becomes more implicit, and that generalizing user preference understanding to unseen scenarios poses further challenges. RealPref and these findings provide a foundation for future research to develop user-aware LLM assistants that better adapt to individual needs. The code is available at https://github.com/GG14127/RealPref.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.13471v2">An LLM Agentic Approach for Legal-Critical Software: A Case Study for Tax Prep Software</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 To appear at ICSE 26. 12 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show promise for translating natural-language statutes into executable logic, but reliability in legally critical settings remains challenging due to ambiguity and hallucinations. We present an agentic approach for developing legal-critical software, using U.S. federal tax preparation as a case study. The key challenge is test-case generation under the oracle problem, where correct outputs require interpreting law. Building on metamorphic testing, we introduce higher-order metamorphic relations that compare system outputs across structured shifts among similar individuals. Because authoring such relations is tedious and error-prone, we use an LLM-driven, role-based framework to automate test generation and code synthesis. We implement a multi-agent system that translates tax code into executable software and incorporates a metamorphic-testing agent that searches for counterexamples. In experiments, our framework using a smaller model (GPT-4o-mini) achieves a worst-case pass rate of 45%, outperforming frontier models (GPT-4o and Claude 3.5, 9-15%) on complex tax-code tasks. These results support agentic LLM methodologies as a path to robust, trustworthy legal-critical software from natural-language specifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04177v1">CodeTaste: Can LLMs Generate Human-Level Code Refactorings?</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      Large language model (LLM) coding agents can generate working code, but their solutions often accumulate complexity, duplication, and architectural debt. Human developers address such issues through refactoring: behavior-preserving program transformations that improve structure and maintainability. In this paper, we investigate if LLM agents (i) can execute refactorings reliably and (ii) identify the refactorings that human developers actually chose in real codebases. We present CodeTaste, a benchmark of refactoring tasks mined from large-scale multi-file changes in open-source repositories. To score solutions, we combine repository test suites with custom static checks that verify removal of undesired patterns and introduction of desired patterns using dataflow reasoning. Our experimental results indicate a clear gap across frontier models: agents perform well when refactorings are specified in detail, but often fail to discover the human refactoring choices when only presented with a focus area for improvement. A propose-then-implement decomposition improves alignment, and selecting the best-aligned proposal before implementation can yield further gains. CodeTaste provides an evaluation target and a potential preference signal for aligning coding agents with human refactoring decisions in realistic codebases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06427v2">Dutch Metaphor Extraction from Cancer Patients' Interviews and Forum Data using LLMs and Human in the Loop</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 Ongoing project report, on behalf of 4D PICTURE https://4dpicture.eu/
    </div>
    <details class="paper-abstract">
      Metaphors and metaphorical language (MLs) play an important role in healthcare communication between clinicians, patients, and patients' family members. In this work, we focus on Dutch language data from cancer patients. We extract metaphors used by patients using two data sources: (1) cancer patient storytelling interview data and (2) online forum data, including patients' posts, comments, and questions to professionals. We investigate how current state-of-the-art large language models (LLMs) perform on this task by exploring different prompting strategies such as chain of thought reasoning, few-shot learning, and self-prompting. With a human-in-the-loop setup, we verify the extracted metaphors and compile the outputs into a corpus named HealthQuote.NL. We believe the extracted metaphors can support better patient care, for example shared decision making, improved communication between patients and clinicians, and enhanced patient health literacy. They can also inform the design of personalized care pathways. We share prompts and related resources at https://github.com/4dpicture/HealthQuote.NL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04123v1">FINEST: Improving LLM Responses to Sensitive Topics Through Fine-Grained Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 Accepted to EACL 2026 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often generate overly cautious and vague responses on sensitive topics, sacrificing helpfulness for safety. Existing evaluation frameworks lack systematic methods to identify and address specific weaknesses in responses to sensitive topics, making it difficult to improve both safety and helpfulness simultaneously. To address this, we introduce FINEST, a FINE-grained response evaluation taxonomy for Sensitive Topics, which breaks down helpfulness and harmlessness into errors across three main categories: Content, Logic, and Appropriateness. Experiments on a Korean-sensitive question dataset demonstrate that our score- and error-based improvement pipeline, guided by FINEST, significantly improves the model responses across all three categories, outperforming refinement without guidance. Notably, score-based improvement -- providing category-specific scores and justifications -- yields the most significant gains, reducing the error sentence ratio for Appropriateness by up to 33.09%. This work lays the foundation for a more explainable and comprehensive evaluation and improvement of LLM responses to sensitive questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04033v1">Who Judges the Judge? Evaluating LLM-as-a-Judge for French Medical open-ended QA</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 Accepted in HeaLing Workshop - EACL 2026
    </div>
    <details class="paper-abstract">
      Automatic evaluation of medical open-ended question answering (OEQA) remains challenging due to the need for expert annotations. We evaluate whether large language models (LLMs) can act as judges of semantic equivalence in French medical OEQA, comparing closed-access, general-purpose, and biomedical domain-adapted models. Our results show that LLM-based judgments are strongly influenced by the model that generated the answer, with agreement varying substantially across generators. Domain-adapted and large general-purpose models achieve the highest alignment with expert annotations. We further show that lightweight adaptation of a compact model using supervised fine-tuning (SFT) and Group Relative Policy Optimization (GRPO) substantially improves performance and reduces generator sensitivity, even with limited data. Overall, our findings highlight the need for generator-aware evaluation and suggest that carefully adapted small models can support scalable evaluation in low-resource medical settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.04336v2">Building a Mind Palace: Structuring Environment-Grounded Semantic Graphs for Effective Long Video Analysis with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      Long-form video understanding with Large Vision Language Models is challenged by the need to analyze temporally dispersed yet spatially concentrated key moments within limited context windows. In this work, we introduce VideoMindPalace, a new framework inspired by the "Mind Palace", which organizes critical video moments into a topologically structured semantic graph. VideoMindPalace organizes key information through (i) hand-object tracking and interaction, (ii) clustered activity zones representing specific areas of recurring activities, and (iii) environment layout mapping, allowing natural language parsing by LLMs to provide grounded insights on spatio-temporal and 3D context. In addition, we propose the Video MindPalace Benchmark (VMB), to assess human-like reasoning, including spatial localization, temporal reasoning, and layout-aware sequential understanding. Evaluated on VMB and established video QA datasets, including EgoSchema, NExT-QA, IntentQA, and the Active Memories Benchmark, VideoMindPalace demonstrates notable gains in spatio-temporal coherence and human-aligned reasoning, advancing long-form video analysis capabilities in VLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04028v1">A Multi-Dimensional Quality Scoring Framework for Decentralized LLM Inference with Proof of Quality</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      Decentralized large language model (LLM) inference networks can pool heterogeneous compute to scale serving, but they require lightweight and incentive-compatible mechanisms to assess output quality. Prior work introduced cost-aware Proof of Quality (PoQ) and adaptive robust PoQ to allocate rewards under evaluator heterogeneity and adversarial behavior. In this paper, we focus on the quality signal itself and propose a multi-dimensional quality scoring framework that decomposes output quality into modular dimensions, including model and cost priors, structure quality, semantic quality, query-output alignment, and agreement/uncertainty. Using logged outputs from QA and summarization tasks, we systematically audit dimension reliability and show that seemingly reasonable dimensions can be task-dependent and even negatively correlated with reference quality without calibration. While the default composite underperforms a strong single semantic evaluator, ablations reveal that removing unreliable dimensions and re-normalizing weights yields a calibrated composite that matches or exceeds the best single- evaluator and consensus baselines. Finally, we integrate the composite score as a drop-in quality signal in PoQ and demonstrate complementary benefits with robust aggregation and adaptive trust weighting under adversarial evaluator attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.02873v2">LaTeX Compilation: Challenges in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 25 pages, 12 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) increasingly assist scientific writing, limitations and the significant token cost of TeX become more and more visible. This paper analyzes TeX's fundamental defects in compilation and user experience design to illustrate its limitations on compilation efficiency, generated semantics, error localization, and tool ecosystem in the era of LLMs. As an alternative, Mogan STEM, a WYSIWYG structured editor, is introduced. Mogan outperforms TeX in the above aspects by its efficient data structure, fast rendering, and on-demand plugin loading. Extensive experiments are conducted to verify the benefits on compilation/rendering time and performance in LLM tasks. What's more, we show that due to Mogan's lower information entropy, it is more efficient to use .tmu (the document format of Mogan) to fine-tune LLMs than TeX. Therefore, we launch an appeal for larger experiments on LLM training using the .tmu format.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.04474v1">From Spark to Fire: Modeling and Mitigating Error Cascades in LLM-Based Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      Large Language Model-based Multi-Agent Systems (LLM-MAS) are increasingly applied to complex collaborative scenarios. However, their collaborative mechanisms may cause minor inaccuracies to gradually solidify into system-level false consensus through iteration. Such risks are difficult to trace since errors can propagate and amplify through message dependencies. Existing protections often rely on single-agent validation or require modifications to the collaboration architecture, which can weaken effective information flow and may not align with natural collaboration processes in real tasks. To address this, we propose a propagation dynamics model tailored for LLM-MAS that abstracts collaboration as a directed dependency graph and provides an early-stage risk criterion to characterize amplification risk. Through experiments on six mainstream frameworks, we identify three vulnerability classes: cascade amplification, topological sensitivity, and consensus inertia. We further instantiate an attack where injecting just a single atomic error seed leads to widespread failure. In response, we introduce a genealogy-graph-based governance layer, implemented as a message-layer plugin, that suppresses both endogenous and exogenous error amplification without altering the collaboration architecture. Experiments show that this approach raises the defense success rate from a baseline of 0.32 to over 0.89 and significantly mitigates the cascading spread of minor errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03946v1">Lang2Str: Two-Stage Crystal Structure Generation with LLMs and Continuous Flow Models</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      Generative models hold great promise for accelerating material discovery but are often limited by their inflexible single-stage generative process in designing valid and diverse materials. To address this, we propose a two-stage generative framework, Lang2Str, that combines the strengths of large language models (LLMs) and flow-based models for flexible and precise material generation. Our method frames the generative process as a conditional generative task, where an LLM provides high-level conditions by generating descriptions of material unit cells' geometric layouts and properties. These descriptions, informed by the LLM's extensive background knowledge, ensure reasonable structure designs. A conditioned flow model then decodes these textual conditions into precise continuous coordinates and unit cell parameters. This staged approach combines the structured reasoning of LLMs and the distribution modeling capabilities of flow models. Experimental results show that our method achieves competitive performance on \textit{ab initio} material generation and crystal structure prediction tasks, with generated structures exhibiting closer alignment to ground truth in both geometry and energy levels, surpassing state-of-the-art models. The flexibility and modularity of our framework further enable fine-grained control over the generation process, potentially leading to more efficient and customizable material design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03919v1">When Safety Becomes a Vulnerability: Exploiting LLM Alignment Homogeneity for Transferable Blocking in RAG</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) enhances the capabilities of large language models (LLMs) by incorporating external knowledge, but its reliance on potentially poisonable knowledge bases introduces new availability risks. Attackers can inject documents that cause LLMs to refuse benign queries, attacks known as blocking attacks. Prior blocking attacks relying on adversarial suffixes or explicit instruction injection are increasingly ineffective against modern safety-aligned LLMs. We observe that safety-aligned LLMs exhibit heightened sensitivity to query-relevant risk signals, causing alignment mechanisms designed for harm prevention to become a source of exploitable refusal. Moreover, mainstream alignment practices share overlapping risk categories and refusal criteria, a phenomenon we term alignment homogeneity, enabling restricted risk context constructed on an accessible LLM to transfer across LLMs. Based on this insight, we propose TabooRAG, a transferable blocking attack framework operating under a strict black-box setting. An attacker can generate a single retrievable blocking document per query by optimizing against a surrogate LLM in an accessible RAG environment, and directly transfer it to an unknown target RAG system without access to the target model. We further introduce a query-aware strategy library to reuse previously effective strategies and improve optimization efficiency. Experiments across 7 modern LLMs and 3 datasets demonstrate that TabooRAG achieves stable cross-model transferability and state-of-the-art blocking success rates, reaching up to 96% on GPT-5.2. Our findings show that increasingly standardized safety alignment across modern LLMs creates a shared and transferable attack surface in RAG systems, revealing a need for improved defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03881v1">On the Suitability of LLM-Driven Agents for Dark Pattern Audits</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      As LLM-driven agents begin to autonomously navigate the web, their ability to interpret and respond to manipulative interface design becomes critical. A fundamental question that emerges is: can such agents reliably recognize patterns of friction, misdirection, and coercion in interface design (i.e., dark patterns)? We study this question in a setting where the workflows are consequential: website portals associated with the submission of CCPA-related data rights requests. These portals operationalize statutory rights, but they are implemented as interactive interfaces whose design can be structured to facilitate, burden, or subtly discourage the exercise of those rights. We design and deploy an LLM-driven auditing agent capable of end-to-end traversal of rights-request workflows, structured evidence gathering, and classification of potential dark patterns. Across a set of 456 data broker websites, we evaluate: (1) the ability of the agent to consistently locate and complete request flows, (2) the reliability and reproducibility of its dark pattern classifications, and (3) the conditions under which it fails or produces poor judgments. Our findings characterize both the feasibility and the limitations of using LLM-driven agents for scalable dark pattern auditing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.22730v2">Extending Czech Aspect-Based Sentiment Analysis with Opinion Terms: Dataset and LLM Benchmarks</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 Accepted for the 15th edition of the Language Resources and Evaluation Conference (LREC 2026)
    </div>
    <details class="paper-abstract">
      This paper introduces a novel Czech dataset in the restaurant domain for aspect-based sentiment analysis (ABSA), enriched with annotations of opinion terms. The dataset supports three distinct ABSA tasks involving opinion terms, accommodating varying levels of complexity. Leveraging this dataset, we conduct extensive experiments using modern Transformer-based models, including large language models (LLMs), in monolingual, cross-lingual, and multilingual settings. To address cross-lingual challenges, we propose a translation and label alignment methodology leveraging LLMs, which yields consistent improvements. Our results highlight the strengths and limitations of state-of-the-art models, especially when handling the linguistic intricacies of low-resource languages like Czech. A detailed error analysis reveals key challenges, including the detection of subtle opinion terms and nuanced sentiment expressions. The dataset establishes a new benchmark for Czech ABSA, and our proposed translation-alignment approach offers a scalable solution for adapting ABSA resources to other low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03862v1">Assessing the Effectiveness of LLMs in Delivering Cognitive Behavioral Therapy</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 Accepted to LREC 2026
    </div>
    <details class="paper-abstract">
      As mental health issues continue to rise globally, there is an increasing demand for accessible and scalable therapeutic solutions. Many individuals currently seek support from Large Language Models (LLMs), even though these models have not been validated for use in counseling services. In this paper, we evaluate LLMs' ability to emulate professional therapists practicing Cognitive Behavioral Therapy (CBT). Using anonymized, transcribed role-play sessions between licensed therapists and clients, we compare two approaches: (1) a generation-only method and (2) a Retrieval-Augmented Generation (RAG) approach using CBT guidelines. We evaluate both proprietary and open-source models for linguistic quality, semantic coherence, and therapeutic fidelity using standard natural language generation (NLG) metrics, natural language inference (NLI), and automated scoring for skills assessment. Our results indicate that while LLMs can generate CBT-like dialogues, they are limited in their ability to convey empathy and maintain consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03855v1">A Sensitivity Analysis of Multi-Event Audio Grounding in Audio LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 6 pages, Submitted to Interspeech 2026
    </div>
    <details class="paper-abstract">
      Audio LLMs have shown a strong ability to understand audio samples, yet their reliability in complex acoustic scenes remains under-explored. Unlike prior work limited to small scale or less controlled query construction, we present a large-scale evaluation of event grounding and false alarms as auditory scene complexity increases. Using 71K AudioCapsV2 clips, we extract normalized (source, attribute) events and build two query types: present-event queries for ground-truth detection and absent-event queries to probe hallucinations, using similarity-filtered negative sampling in an audio-aligned text embedding space. We evaluate four SOTA Audio LLMs with 12 prompt variants over 500K yes/no queries per model. Across models, increasing event count consistently lowers true-positive rate and raises false-positive rate, while prompts induce a strong trade-off between the two. Our confidence analysis shows that models become more uncertain on multi-event audio, revealing room for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.21091v3">Best-of-$\infty$ -- Asymptotic Performance of Test-Time LLM Ensembling</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 To appear at ICLR2026. Our code is available at https://github.com/jkomiyama/BoInf-code-publish/. Updated the title
    </div>
    <details class="paper-abstract">
      We study best-of-$N$ for large language models (LLMs) where the selection is based on majority voting. In particular, we analyze the limit $N \to \infty$, which we denote as \boinflower. While this approach achieves impressive performance in the limit, it requires an infinite test-time budget. To address this, we propose an adaptive generation scheme that selects $N$ based on answer agreement, thereby efficiently allocating inference-time computation. Beyond adaptivity, we extend the framework to weighted ensembles of multiple LLMs, showing that such mixtures can outperform any individual model. The optimal ensemble weighting is formulated and efficiently computed as a mixed-integer linear program. Extensive experiments demonstrate the effectiveness of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03811v1">Robust LLM-based Audio-Visual Speech Recognition with Sparse Modality Alignment and Visual Unit-Guided Refinement</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 submitted to Interspeech 2026
    </div>
    <details class="paper-abstract">
      Audio-Visual Speech Recognition (AVSR) integrates acoustic and visual information to enhance robustness in adverse acoustic conditions. Recent advances in Large Language Models (LLMs) have yielded competitive automatic speech recognition performance and shown effectiveness for AVSR. However, prior approaches project audio and visual features independently or apply shallow fusion, limiting cross-modal alignment and complementary exchange while increasing the LLM's computational load. To address this, we propose AVUR-LLM, an LLM-based Audio-Visual Speech Recognition via Sparse Modality Alignment and Visual Unit-Guided Refinement. Experiments on LRS3 demonstrate state-of-the-art results for AVSR. Under additive-noise conditions at 0 dB SNR, it achieves 37% relative improvement over the baseline system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15982v2">AMiD: Knowledge Distillation for LLMs with $α$-mixture Assistant Distribution</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 The Fourteenth International Conference on Learning Representations (ICLR 2026)
    </div>
    <details class="paper-abstract">
      Autoregressive large language models (LLMs) have achieved remarkable improvement across many tasks but incur high computational and memory costs. Knowledge distillation (KD) mitigates this issue by transferring knowledge from a large teacher to a smaller student through distributional alignment. Previous studies have proposed various discrepancy metrics, but the capacity gap and training instability caused by near-zero probabilities, stemming from the high-dimensional output of LLMs, remain fundamental limitations. To overcome these challenges, several approaches implicitly or explicitly incorporating assistant distribution have recently been proposed. However, the past proposals of assistant distributions have been a fragmented approach without a systematic investigation of the interpolation path and the divergence. This paper proposes $α$-mixture assistant distribution, a novel generalized family of assistant distributions, and $α$-mixture distillation, coined AMiD, a unified framework for KD using the assistant distribution. The $α$-mixture assistant distribution provides a continuous extension of the assistant distribution by introducing a new distribution design variable $α$, which has been fixed in all previous approaches. Furthermore, AMiD generalizes the family of divergences used with the assistant distributions based on optimality, which has also been restricted in previous works. Through extensive experiments, we demonstrate that AMiD offers superior performance and training stability by leveraging a broader and theoretically grounded assistant distribution space. We release the code at \href{https://github.com/aailab-kaist/AMiD}{this https URL}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.04755v2">When Silence Is Golden: Can LLMs Learn to Abstain in Temporal QA and Beyond?</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 Accepted to ICLR2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) rarely admit uncertainty, often producing fluent but misleading answers, rather than abstaining (i.e., refusing to answer). This weakness is even evident in temporal question answering, where models frequently ignore time-sensitive evidence and conflate facts across different time-periods. In this paper, we present the first empirical study of training LLMs with an abstention ability while reasoning about temporal QA. Existing approaches such as calibration might be unreliable in capturing uncertainty in complex reasoning. We instead frame abstention as a teachable skill and introduce a pipeline that couples Chain-of-Thought (CoT) supervision with Reinforcement Learning (RL) guided by abstention-aware rewards. Our goal is to systematically analyze how different information types and training techniques affect temporal reasoning with abstention behavior in LLMs. Through extensive experiments studying various methods, we find that RL yields strong empirical gains on reasoning: a model initialized by Qwen2.5-1.5B-Instruct surpasses GPT-4o by $3.46\%$ and $5.80\%$ in Exact Match on TimeQA-Easy and Hard, respectively. Moreover, it improves the True Positive rate on unanswerable questions by $20\%$ over a pure supervised fine-tuned (SFT) variant. Beyond performance, our analysis shows that SFT induces overconfidence and harms reliability, while RL improves prediction accuracy but exhibits similar risks. Finally, by comparing implicit reasoning cues (e.g., original context, temporal sub-context, knowledge graphs) with explicit CoT supervision, we find that implicit information provides limited benefit for reasoning with abstention. Our study provides new insights into how abstention and reasoning can be jointly optimized, providing a foundation for building more reliable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.10131v2">Proof Strategy Extraction from LLMs for Enhancing Symbolic Provers</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      One important approach to software verification is interactive theorem proving. However, writing formal proofs often requires substantial human effort, making proof automation highly important. Traditionally, proof automation has relied on symbolic provers. Recently, large language models (LLMs) have demonstrated strong capabilities in theorem proving, complementing symbolic provers. Nonetheless, prompting LLMs can be expensive and may pose security risks for confidential codebases. As a result, purely symbolic approaches remain important even in the LLM era, as they are cost-effective, secure, and complement the strengths of LLMs. Motivated by these considerations, we pose a new research question: can the internal proof strategies of LLMs be extracted to enhance the capabilities of symbolic provers? As an initial step, we introduce Strat2Rocq. In an offline stage, Strat2Rocq extracts proof strategies from LLMs and formalizes them as lemmas in Rocq. In an online stage, given a theorem to be proved, Strat2Rocq augments the proof context with these extracted lemmas, enabling CoqHammer to leverage the LLM-derived strategies for more effective automated proving. Our evaluation demonstrates that, on open-source Rocq projects for software verification, Strat2Rocq enhances the success rate of CoqHammer by 13.41%. A side discovery is that the extracted lemmas are also beneficial to LLM proof agents, improving the success rate of an LLM proof agent by 4.00%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2407.00225v4">Large-scale, Independent and Comprehensive study of the power of LLMs for test case generation</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 Accepted at Empirical Software Engineering (EMSE) journal on 3 March 2026
    </div>
    <details class="paper-abstract">
      Unit testing is essential for software reliability, yet manual test creation is time-consuming and often neglected. Search-based software testing improves efficiency but produces tests with poor readability and maintainability, while LLMs show promise but lack comprehensive evaluation across reasoning-based prompting and real-world scenarios. This study presents the first large-scale empirical evaluation of LLM-generated unit tests at the full class level, analyzing four models (GPT-3.5, GPT-4, Mistral 7B, and Mixtral 8x7B) against EvoSuite across 216,300 test cases targeting Defects4J, SF110, and CMD. We evaluate five prompting techniques, ZSL, FSL, CoT, ToT, and GToT, assessing compilability, hallucination-driven failures, readability, coverage, and test smells. Reasoning-based prompting, particularly GToT, significantly enhances reliability and compilability, yet hallucination-driven failures remain persistent, with compilation failure rates reaching 86%. While LLM-generated tests are generally more readable than SBST outputs, recurring issues such as Magic Number Tests and Assertion Roulette hinder maintainability. These findings suggest that hybrid approaches combining LLM-based generation with automated validation and search-based refinement are necessary for production-ready results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03727v1">Understanding Parents' Desires in Moderating Children's Interactions with GenAI Chatbots through LLM-Generated Probes</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 33 pages, 10 figures, Accepted to ACM CHI 2026
    </div>
    <details class="paper-abstract">
      This paper studies how parents want to moderate children's interactions with Generative AI chatbots, with the goal of informing the design of future GenAI parental control tools. We first used an LLM to generate synthetic child-GenAI chatbot interaction scenarios and worked with four parents to validate their realism. From this dataset, we carefully selected 12 diverse examples that evoked varying levels of concern and were rated the most realistic. Each example included a prompt and a GenAI chatbot response. We presented these to parents (N=24) and asked whether they found them concerning, why, and how they would prefer the responses to be modified and communicated. Our findings reveal three key insights: (1) parents express concern about interactions that current GenAI chatbot parental controls neglect; (2) parents want fine-grained transparency and moderation at the conversation level; and (3) parents need personalized controls that adapt to their desired strategies and children's ages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.07321v2">ObfusQAte: A Proposed Framework to Evaluate LLM Robustness on Obfuscated Factual Question Answering</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 LREC 2026
    </div>
    <details class="paper-abstract">
      The rapid proliferation of Large Language Models (LLMs) has significantly contributed to the development of equitable AI systems capable of factual question-answering (QA). However, no known study tests the LLMs' robustness when presented with obfuscated versions of questions. To systematically evaluate these limitations, we propose a novel technique, ObfusQAte, and leveraging the same, introduce ObfusQA, a comprehensive, first-of-its-kind framework with multi-tiered obfuscation levels designed to examine LLM capabilities across three distinct dimensions: (i) Named-Entity Indirection, (ii) Distractor Indirection, and (iii) Contextual Overload. By capturing these fine-grained distinctions in language, ObfusQA provides a comprehensive benchmark for evaluating LLM robustness and adaptability. Our study observes that LLMs exhibit a tendency to fail or generate hallucinated responses when confronted with these increasingly nuanced variations. To foster research in this direction, we make ObfusQAte publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03683v1">CONCUR: Benchmarking LLMs for Concurrent Code Generation</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      Leveraging Large Language Models (LLMs) for code generation has increasingly emerged as a common practice in the domain of software engineering. Relevant benchmarks have been established to evaluate the code generation capabilities of LLMs. However, existing benchmarks focus primarily on sequential code, lacking the ability to effectively evaluate LLMs on concurrent code generation. Compared to sequential code, concurrent code exhibits greater complexity and possesses unique types of bugs, such as deadlocks and race conditions, that do not occur in sequential code. Therefore, a benchmark for evaluating sequential code generation cannot be useful for evaluating concurrent code generation with LLMs. To address this gap, we designed a benchmark CONCUR specifically aimed at evaluating the capability of LLMs to generate concurrent code. CONCUR consists of a base set of 43 concurrency problems derived from a standard concurrency textbook, together with 72 validated mutant variants, resulting in 115 total problems. The base problems serve as the semantic core of the benchmark, while the mutants expand linguistic and structural diversity. We conducted an evaluation of a range of LLMs on CONCUR, highlighting limitations of current models. Overall, our work provides a novel direction for evaluating the capability of LLMs to generate code with focus on concurrency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.01393v2">Adaptive Alpha Weighting with PPO: Enhancing Prompt-Based LLM-Generated Alphas in Quant Trading</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 This paper has been accepted by International Journal of Data Science and Analytics
    </div>
    <details class="paper-abstract">
      This paper introduces a reinforcement learning framework that employs Proximal Policy Optimization (PPO) to dynamically optimize the weights of multiple large language model (LLM)-generated formulaic alphas for stock trading strategies. Formulaic alphas are mathematically defined trading signals derived from price, volume, sentiment, and other data. Although recent studies have shown that LLMs can generate diverse and effective alphas, a critical challenge lies in how to adaptively integrate them under varying market conditions. To address this gap, we leverage a DeepSeek model to generate fifty alphas for ten stocks, and then use PPO to adjust their weights in real time. Experimental results indicate that the PPO-optimized strategy does not consistently deliver the highest cumulative returns across all stocks, but it achieves comparatively higher Sharpe ratios and smaller maximum drawdowns in most cases. When compared with baseline strategies, including equal-weighted, buy-and-hold, random entry/exit, and momentum approaches, PPO demonstrates more stable risk-adjusted performance. The findings highlight the importance of reinforcement learning in the allocation of alpha weights and show the potential of combining LLM-generated signals with adaptive optimization for robust financial forecasting and trading.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03668v1">Can LLM Aid in Solving Constraints with Inductive Definitions?</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 22 pages, 4 figures, accepted by the 27th Symposium on Formal Methods (FM 2026)
    </div>
    <details class="paper-abstract">
      Solving constraints involving inductive (aka recursive) definitions is challenging. State-of-the-art SMT/CHC solvers and first-order logic provers provide only limited support for solving such constraints, especially when they involve, e.g., abstract data types. In this work, we leverage structured prompts to elicit Large Language Models (LLMs) to generate auxiliary lemmas that are necessary for reasoning about these inductive definitions. We further propose a neuro-symbolic approach, which synergistically integrates LLMs with constraint solvers: the LLM iteratively generates conjectures, while the solver checks their validity and usefulness for proving the goal. We evaluate our approach on a diverse benchmark suite comprising constraints originating from algebrai data types and recurrence relations. The experimental results show that our approach can improve the state-of-the-art SMT and CHC solvers, solving considerably more (around 25%) proof tasks involving inductive definitions, demonstrating its efficacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03655v1">Mozi: Governed Autonomy for Drug Discovery LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      Tool-augmented large language model (LLM) agents promise to unify scientific reasoning with computation, yet their deployment in high-stakes domains like drug discovery is bottlenecked by two critical barriers: unconstrained tool-use governance and poor long-horizon reliability. In dependency-heavy pharmaceutical pipelines, autonomous agents often drift into irreproducible trajectories, where early-stage hallucinations multiplicatively compound into downstream failures. To overcome this, we present Mozi, a dual-layer architecture that bridges the flexibility of generative AI with the deterministic rigor of computational biology. Layer A (Control Plane) establishes a governed supervisor--worker hierarchy that enforces role-based tool isolation, limits execution to constrained action spaces, and drives reflection-based replanning. Layer B (Workflow Plane) operationalizes canonical drug discovery stages -- from Target Identification to Lead Optimization -- as stateful, composable skill graphs. This layer integrates strict data contracts and strategic human-in-the-loop (HITL) checkpoints to safeguard scientific validity at high-uncertainty decision boundaries. Operating on the design principle of ``free-form reasoning for safe tasks, structured execution for long-horizon pipelines,'' Mozi provides built-in robustness mechanisms and trace-level audibility to completely mitigate error accumulation. We evaluate Mozi on PharmaBench, a curated benchmark for biomedical agents, demonstrating superior orchestration accuracy over existing baselines. Furthermore, through end-to-end therapeutic case studies, we demonstrate Mozi's ability to navigate massive chemical spaces, enforce stringent toxicity filters, and generate highly competitive in silico candidates, effectively transforming the LLM from a fragile conversationalist into a reliable, governed co-scientist.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03640v1">MistyPilot: An Agentic Fast-Slow Thinking LLM Framework for Misty Social Robots</a></div>
    <div class="paper-meta">
      📅 2026-03-04
    </div>
    <details class="paper-abstract">
      With the availability of open APIs in social robots, it has become easier to customize general-purpose tools to meet users' needs. However, interpreting high-level user instructions, selecting and configuring appropriate tools, and executing them reliably remain challenging for users without programming experience. To address these challenges, we introduce MistyPilot, an agentic LLM-driven framework for autonomous tool selection, orchestration, and parameter configuration. MistyPilot comprises two core components: a Physically Interactive Agent (PIA) and a Socially Intelligent Agent (SIA). The PIA enables robust sensor-triggered and tool-driven task execution, while the SIA generates socially intelligent and emotionally aligned dialogue. MistyPilot further integrates a fast-slow thinking paradigm to capture user preferences, reduce latency, and improve task efficiency. To comprehensively evaluate MistyPilot, we contribute five benchmark datasets. Extensive experiments demonstrate the effectiveness of our framework in routing correctness, task completeness, fast-slow thinking retrieval efficiency, tool scalability,and emotion alignment. All code, datasets, and experimental videos will be made publicly available on the project webpage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03633v1">Goal-Driven Risk Assessment for LLM-Powered Systems: A Healthcare Case Study</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 To appear in the HealthSec Workshop at the 2025 IEEE Annual Computer Security Applications Conference (ACSAC)
    </div>
    <details class="paper-abstract">
      While incorporating LLMs into systems offers significant benefits in critical application areas such as healthcare, new security challenges emerge due to the potential cyber kill chain cycles that combine adversarial model, prompt injection and conventional cyber attacks. Threat modeling methods enable the system designers to identify potential cyber threats and the relevant mitigations during the early stages of development. Although the cyber security community has extensive experience in applying these methods to software-based systems, the elicited threats are usually abstract and vague, limiting their effectiveness for conducting proper likelihood and impact assessments for risk prioritization, especially in complex systems with novel attacks surfaces, such as those involving LLMs. In this study, we propose a structured, goal driven risk assessment approach that contextualizes the threats with detailed attack vectors, preconditions, and attack paths through the use of attack trees. We demonstrate the proposed approach on a case study with an LLM agent-based healthcare system. This study harmonizes the state-of-the-art attacks to LLMs with conventional ones and presents possible attack paths applicable to similar systems. By providing a structured risk assessment, this study makes a significant contribution to the literature and advances the secure-by-design practices in LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04975v2">Sentiment-Aware Stock Price Prediction with Transformer and LLM-Generated Formulaic Alpha</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 This paper has been accepted by the journal Digital Finance
    </div>
    <details class="paper-abstract">
      Traditionally, traders and quantitative analysts address alpha decay by manually crafting formulaic alphas, mathematical expressions that identify patterns or signals in financial data, through domain expertise and trial-and-error. This process is often time-consuming and difficult to scale. With recent advances in large language models (LLMs), it is now possible to automate the generation of such alphas by leveraging the reasoning capabilities of LLMs. This paper introduces a novel framework that integrates a prompt-based LLM with a Transformer model for stock price prediction. The LLM first generates diverse and adaptive alphas using structured inputs such as historical stock features (Close, Open, High, Low, Volume), technical indicators, sentiment scores of both target and related companies. These alphas, instead of being used directly for trading, are treated as high-level features that capture complex dependencies within the financial data. To evaluate the effectiveness of these LLM-generated formulaic alphas, the alpha features are then fed into prediction models such as Transformer, LSTM, TCN, SVR, and Random Forest to forecast future stock prices. Experimental results demonstrate that the LLM-generated alphas significantly improve predictive accuracy. Moreover, the accompanying natural language reasoning provided by the LLM enhances the interpretability and transparency of the predictions, supporting more informed financial decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03597v1">NuMuon: Nuclear-Norm-Constrained Muon for Compressible LLM Training</a></div>
    <div class="paper-meta">
      📅 2026-03-04
      | 💬 47 pages, 22 figures, 18 tables
    </div>
    <details class="paper-abstract">
      The rapid progress of large language models (LLMs) is increasingly constrained by memory and deployment costs, motivating compression methods for practical deployment. Many state-of-the-art compression pipelines leverage the low-rank structure of trained weight matrices, a phenomenon often associated with the properties of popular optimizers such as Adam. In this context, Muon is a recently proposed optimizer that improves LLM pretraining via full-rank update steps, but its induced weight-space structure has not been characterized yet. In this work, we report a surprising empirical finding: despite imposing full-rank updates, Muon-trained models exhibit pronounced low-rank structure in their weight matrices and are readily compressible under standard pipelines. Motivated by this insight, we propose NuMuon, which augments Muon with a nuclear-norm constraint on the update direction, further constraining the learned weights toward low-rank structure. Across billion-parameter-scale models, we show that NuMuon increases weight compressibility and improves post-compression model quality under state-of-the-art LLM compression pipelines while retaining Muon's favorable convergence behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03536v1">SafeCRS: Personalized Safety Alignment for LLM-Based Conversational Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 14 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Current LLM-based conversational recommender systems (CRS) primarily optimize recommendation accuracy and user satisfaction. We identify an underexplored vulnerability in which recommendation outputs may negatively impact users by violating personalized safety constraints, when individualized safety sensitivities -- such as trauma triggers, self-harm history, or phobias -- are implicitly inferred from the conversation but not respected during recommendation. We formalize this challenge as personalized CRS safety and introduce SafeRec, a new benchmark dataset designed to systematically evaluate safety risks in LLM-based CRS under user-specific constraints. To further address this problem, we propose SafeCRS, a safety-aware training framework that integrates Safe Supervised Fine-Tuning (Safe-SFT) with Safe Group reward-Decoupled Normalization Policy Optimization (Safe-GDPO) to jointly optimize recommendation quality and personalized safety alignment. Extensive experiments on SafeRec demonstrate that SafeCRS reduces safety violation rates by up to 96.5% relative to the strongest recommendation-quality baseline while maintaining competitive recommendation quality. Warning: This paper contains potentially harmful and offensive content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23783v3">Boosting In-Context Learning in LLMs Through the Lens of Classical Supervised Learning</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      In-Context Learning (ICL) allows Large Language Models (LLMs) to adapt to new tasks with just a few examples, but their predictions often suffer from systematic biases, leading to unstable performance in classification. While calibration techniques are proposed to mitigate these biases, we show that, in the logit space, many of these methods are equivalent to merely shifting the LLM's decision boundary without having the ability to alter its orientation. This proves inadequate when biases cause the LLM to be severely misaligned. To address these limitations and provide a unifying framework, we propose Supervised Calibration (SC), a loss-minimization-based framework, which learns an optimal, per-class affine transformation of LLM's predictive probabilities in the logit space without requiring external data beyond the context. By using a more expressive functional class, SC not only subsumes many existing calibration methods in ICL as special cases but also enables the ability of altering and even completely reversing the orientation of the LLM's decision boundary. Furthermore, SC's loss-based nature facilitates the seamless integration of two purpose-built regularization techniques, context-invariance and directional trust-region regularizers. The former is designed to tackle the instability issue in ICL, while the latter is to control the degree of calibration. Finally, SC delivers state-of-the-art performance over calibration baselines in the 4-shot, 8-shot, and 16-shot settings across all nine datasets for Mistral-7B-Instruct-v0.3, Llama-2-7B-chat, and Qwen2-7B-Instruct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03415v1">Farther the Shift, Sparser the Representation: Analyzing OOD Mechanisms in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      In this work, we investigate how Large Language Models (LLMs) adapt their internal representations when encountering inputs of increasing difficulty, quantified as the degree of out-of-distribution (OOD) shift. We reveal a consistent and quantifiable phenomenon: as task difficulty increases, whether through harder reasoning questions, longer contexts, or adding answer choices, the last hidden states of LLMs become substantially sparser. In short, \textbf{\textit{the farther the shift, the sparser the representations}}. This sparsity--difficulty relation is observable across diverse models and domains, suggesting that language models respond to unfamiliar or complex inputs by concentrating computation into specialized subspaces in the last hidden state. Through a series of controlled analyses with a learning dynamic explanation, we demonstrate that this sparsity is not incidental but an adaptive mechanism for stabilizing reasoning under OOD. Leveraging this insight, we design \textit{Sparsity-Guided Curriculum In-Context Learning (SG-ICL)}, a strategy that explicitly uses representation sparsity to schedule few-shot demonstrations, leading to considerable performance enhancements. Our study provides new mechanistic insights into how LLMs internalize OOD challenges. The source code is available at the URL: https://github.com/MingyuJ666/sparsityLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03206v1">Understanding and Mitigating Dataset Corruption in LLM Steering</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Contrastive steering has been shown as a simple and effective method to adjust the generative behavior of LLMs at inference time. It uses examples of prompt responses with and without a trait to identify a direction in an intermediate activation layer, and then shifts activations in this 1-dimensional subspace. However, despite its growing use in AI safety applications, the robustness of contrastive steering to noisy or adversarial data corruption is poorly understood. We initiate a study of the robustness of this process with respect to corruption of the dataset of examples used to train the steering direction. Our first observation is that contrastive steering is quite robust to a moderate amount of corruption, but unwanted side effects can be clearly and maliciously manifested when a non-trivial fraction of the training data is altered. Second, we analyze the geometry of various types of corruption, and identify some safeguards. Notably, a key step in learning the steering direction involves high-dimensional mean computation, and we show that replacing this step with a recently developed robust mean estimator often mitigates most of the unwanted effects of malicious corruption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03192v1">MoD-DPO: Towards Mitigating Cross-modal Hallucinations in Omni LLMs using Modality Decoupled Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 CVPR 2026. Project Page: https://mod-dpo.github.io/
    </div>
    <details class="paper-abstract">
      Omni-modal large language models (omni LLMs) have recently achieved strong performance across audiovisual understanding tasks, yet they remain highly susceptible to cross-modal hallucinations arising from spurious correlations and dominant language priors. In this work, we propose Modality-Decoupled Direct Preference Optimization (MoD-DPO), a simple and effective framework for improving modality grounding in omni LLMs. MoD-DPO introduces modality-aware regularization terms that explicitly enforce invariance to corruptions in irrelevant modalities and sensitivity to perturbations in relevant modalities, thereby reducing unintended cross-modal interactions. To further mitigate over-reliance on textual priors, we incorporate a language-prior debiasing penalty that discourages hallucination-prone text-only responses. Extensive experiments across multiple audiovisual hallucination benchmarks demonstrate that MoD-DPO consistently improves perception accuracy and hallucination resistance, outperforming previous preference optimization baselines under similar training budgets. Our findings underscore the importance of modality-faithful alignment and demonstrate a scalable path toward more reliable and resilient multimodal foundation models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09710v3">Echoing: Identity Failures when LLM Agents Talk to Each Other</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Published at ICLR workshop. Related blog post: https://www.salesforce.com/blog/agent-to-agent-interaction/
    </div>
    <details class="paper-abstract">
      As large language model (LLM) based agents interact autonomously with one another, a new class of failures emerges that cannot be predicted from single agent performance: behavioral drifts in agent-agent conversations (AxA). Unlike human-agent interactions, where humans ground and steer conversations, AxA lacks such stabilizing signals, making these failures unique. We investigate one such failure, echoing, where agents abandon their assigned roles and instead mirror their conversational partners, undermining their intended objectives. Through experiments across $66$ AxA configurations, $4$ domains (3 transactional, 1 advisory), and $2500+$ conversations (over $250000$ LLM inferences), we show that echoing occurs across major LLM providers, with echoing rates as high as $70\%$ depending on the model and domain. Moreover, we find that echoing is persistent even in advanced reasoning models with substantial rates ($32.8\%$) that are not reduced by reasoning efforts. We analyze prompt, conversation dynamics, showing that echoing arises as interaction grows longer ($7+$ agent turns) and is not merely an artifact of sub-optimal experiment design. Finally, we introduce a protocol-level mitigation where targeted use of structured response reduces echoing to $9\%$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2602.07181v2">Can LLMs Discern the Traits Influencing Your Preferences? Evaluating Personality-Driven Preference Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      User preferences are increasingly used to personalize Large Language Model (LLM) responses, yet how to reliably leverage preference signals for answer generation remains under-explored. In practice, preferences can be noisy, incomplete, or even misleading, which can degrade answer quality when applied naively. Motivated by the observation that stable personality traits shape everyday preferences, we study personality as a principled ''latent'' signal behind preference statements. Through extensive experiments, we find that conditioning on personality-aligned preferences substantially improves personalized question answering: selecting preferences consistent with a user's inferred personality increases answer-choice accuracy from 29.25% to 76%, compared to using randomly selected preferences. Based on these findings, we introduce PACIFIC (Preference Alignment Choices Inference for Five-factor Identity Characterization), a personality-labeled preference dataset containing 1200 preference statements spanning diverse domains (e.g., travel, movies, education), annotated with Big-Five (OCEAN) trait directions. Finally, we propose a framework that enables an LLM model to automatically retrieve personality-aligned preferences and incorporate them during answer generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.09001v3">Entropy Sentinel: Continuous LLM Accuracy Monitoring from Decoding Entropy Traces in STEM</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Deploying LLMs raises two coupled challenges: (1) monitoring--estimating where a model underperforms as traffic and domains drift--and (2) improvement--prioritizing data acquisition to close the largest performance gaps. We test whether an inference-time signal can estimate slice-level accuracy under domain shift. For each response, we compute an output-entropy profile from final-layer next-token probabilities (from top-$k$ logprobs) and summarize it with different statistics. A lightweight classifier predicts instance correctness, and averaging predicted probabilities yields a domain-level accuracy estimate. We evaluate on ten STEM reasoning benchmarks with exhaustive train/test compositions ($k\in\{1,2,3,4\}$; all $\binom{10}{k}$ combinations), on different classifier models and features across nine LLMs from six families (3B--20B). Estimates often track held-out benchmark accuracy, and several models show near-monotonic ordering of domains, providing evidence for output-entropy profiles being an accessible signal for scalable monitoring and for targeted data acquisition.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03148v1">From Language to Action: Can LLM-Based Agents Be Used for Embodied Robot Cognition?</a></div>
    <div class="paper-meta">
      📅 2026-03-03
      | 💬 Accepted for publication at the 2026 IEEE International Conference on Robotics and Automation (ICRA)
    </div>
    <details class="paper-abstract">
      In order to flexibly act in an everyday environment, a robotic agent needs a variety of cognitive capabilities that enable it to reason about plans and perform execution recovery. Large language models (LLMs) have been shown to demonstrate emergent cognitive aspects, such as reasoning and language understanding; however, the ability to control embodied robotic agents requires reliably bridging high-level language to low-level functionalities for perception and control. In this paper, we investigate the extent to which an LLM can serve as a core component for planning and execution reasoning in a cognitive robot architecture. For this purpose, we propose a cognitive architecture in which an agentic LLM serves as the core component for planning and reasoning, while components for working and episodic memories support learning from experience and adaptation. An instance of the architecture is then used to control a mobile manipulator in a simulated household environment, where environment interaction is done through a set of high-level tools for perception, reasoning, navigation, grasping, and placement, all of which are made available to the LLM-based agent. We evaluate our proposed system on two household tasks (object placement and object swapping), which evaluate the agent's reasoning, planning, and memory utilisation. The results demonstrate that the LLM-driven agent can complete structured tasks and exhibits emergent adaptation and memory-guided planning, but also reveal significant limitations, such as hallucinations about the task success and poor instruction following by refusing to acknowledge and complete sequential tasks. These findings highlight both the potential and challenges of employing LLMs as embodied cognitive controllers for autonomous robots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2603.03116v1">Beyond Task Completion: Revealing Corrupt Success in LLM Agents through Procedure-Aware Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-03-03
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents are increasingly adopted in high-stakes settings, but current benchmarks evaluate mainly whether a task was completed, not how. We introduce Procedure-Aware Evaluation (PAE), a framework that formalizes agent procedures as structured observations and exposes consistency relationships between what agents observe, communicate, and execute. PAE evaluates agents along complementary axes (Utility, Efficiency, Interaction Quality, Procedural Integrity) and applies multi-dimensional gating that categorically disqualifies corrupt outcomes. Evaluating state-of-the-art LLM agents on tau-bench yields findings at the axis, compliance, and benchmark levels. At the axis level, the dimensions capture non-redundant failure modes: utility masks reliability gaps, speed does not imply precision, and conciseness does not predict intent adherence. At the procedural compliance level, 27-78% of benchmark reported successes are corrupt successes concealing violations across interaction and integrity. Furthermore, gating substantially collapses Pass^4 rate and affects model rankings. The analysis of corrupt success cases reveals distinctive per-model failure signatures: GPT-5 spreads errors across policy, execution, and intent dimensions; Kimi-K2-Thinking concentrates 78% of violations in policy faithfulness and compliance; and Mistral-Large-3 is dominated by faithfulness failures. At the benchmark level, our analysis exposes structural flaws in the benchmark design, including task scope gaps, contradictory reward signals, and simulator artifacts that produce accidental successes.
    </details>
</div>
