# llm - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13717v1">TZ-LLM: Protecting On-Device Large Language Models with Arm TrustZone</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deployed on mobile devices offer benefits like user privacy and reduced network latency, but introduce a significant security risk: the leakage of proprietary models to end users. To mitigate this risk, we propose a system design for protecting on-device LLMs using Arm Trusted Execution Environment (TEE), TrustZone. Our system addresses two primary challenges: (1) The dilemma between memory efficiency and fast inference (caching model parameters within TEE memory). (2) The lack of efficient and secure Neural Processing Unit (NPU) time-sharing between Rich Execution Environment (REE) and TEE. Our approach incorporates two key innovations. First, we employ pipelined restoration, leveraging the deterministic memory access patterns of LLM inference to prefetch parameters on demand, hiding memory allocation, I/O and decryption latency under computation time. Second, we introduce a co-driver design, creating a minimal data plane NPU driver in the TEE that collaborates with the full-fledged REE driver. This reduces the TEE TCB size and eliminates control plane reinitialization overhead during NPU world switches. We implemented our system on the emerging OpenHarmony OS and the llama.cpp inference framework, and evaluated it with various LLMs on an Arm Rockchip device. Compared to a strawman TEE baseline lacking our optimizations, our system reduces TTFT by up to 90.9% and increases decoding speed by up to 23.2%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13676v1">T-SAR: A Full-Stack Co-design for CPU-Only Ternary LLM Inference via In-Place SIMD ALU Reorganization</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted to DATE 2026
    </div>
    <details class="paper-abstract">
      Recent advances in LLMs have outpaced the computational and memory capacities of edge platforms that primarily employ CPUs, thereby challenging efficient and scalable deployment. While ternary quantization enables significant resource savings, existing CPU solutions rely heavily on memory-based lookup tables (LUTs) which limit scalability, and FPGA or GPU accelerators remain impractical for edge use. This paper presents T-SAR, the first framework to achieve scalable ternary LLM inference on CPUs by repurposing the SIMD register file for dynamic, in-register LUT generation with minimal hardware modifications. T-SAR eliminates memory bottlenecks and maximizes data-level parallelism, delivering 5.6-24.5x and 1.1-86.2x improvements in GEMM latency and GEMV throughput, respectively, with only 3.2% power and 1.4% area overheads in SIMD units. T-SAR achieves up to 2.5-4.9x the energy efficiency of an NVIDIA Jetson AGX Orin, establishing a practical approach for efficient LLM inference on edge platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13658v1">Why is "Chicago" Predictive of Deceptive Reviews? Using LLMs to Discover Language Phenomena from Lexical Cues</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Deceptive reviews mislead consumers, harm businesses, and undermine trust in online marketplaces. Machine learning classifiers can learn from large amounts of training examples to effectively distinguish deceptive reviews from genuine ones. However, the distinguishing features learned by these classifiers are often subtle, fragmented, and difficult for humans to interpret. In this work, we explore using large language models (LLMs) to translate machine-learned lexical cues into human-understandable language phenomena that can differentiate deceptive reviews from genuine ones. We show that language phenomena obtained in this manner are empirically grounded in data, generalizable across similar domains, and more predictive than phenomena either in LLMs' prior knowledge or obtained through in-context learning. These language phenomena have the potential to aid people in critically assessing the credibility of online reviews in environments where deception detection classifiers are unavailable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13640v1">Data Value in the Age of Scaling: Understanding LLM Scaling Dynamics Under Real-Synthetic Data Mixtures</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      The rapid progress of large language models (LLMs) is fueled by the growing reliance on datasets that blend real and synthetic data. While synthetic data offers scalability and cost-efficiency, it often introduces systematic distributional discrepancies, particularly underrepresenting long-tail knowledge due to truncation effects from data generation mechanisms like top-p sampling, temperature scaling, and finite sampling. These discrepancies pose fundamental challenges in characterizing and evaluating the utility of mixed real-synthetic datasets. In this paper, we identify a three-phase scaling behavior characterized by two breakpoints that reflect transitions in model behavior across learning head and tail knowledge. We further derive an LLM generalization bound designed for real and synthetic mixtures, revealing several key factors that govern their generalization performance. Building on our theoretical findings, we propose an effective yet efficient data valuation method that scales to large-scale datasets. Comprehensive experiments across four tasks, including image classification, sentiment classification, instruction following, and complex reasoning, demonstrate that our method surpasses state-of-the-art baselines in data valuation with significantly low computational cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.21323v2">LLM-driven Provenance Forensics for Threat Investigation and Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      We introduce PROVSEEK, an LLM-powered agentic framework for automated provenance-driven forensic analysis and threat intelligence extraction. PROVSEEK employs specialized toolchains to dynamically retrieve relevant context by generating precise, context-aware queries that fuse knowledge from threat reports with evidence from system provenance data. The framework resolves provenance queries, orchestrates multiple role-specific agents, and synthesizes structured, ground-truth verifiable forensic summaries. By combining agent orchestration with Retrieval-Augmented Generation (RAG) and chain-of-thought (CoT) reasoning, data-guided filtration using a behavioral model, PROVSEEK enables adaptive multi-step analysis that iteratively refines hypotheses, verifies supporting evidence, and produces scalable, interpretable forensic explanations of attack behaviors. PROVSEEK is designed for automated threat investigation without task-specific training data, enabling forensic-style investigation even when no prior knowledge of the environment. We conduct a comprehensive evaluation on publicly available DARPA datasets, demonstrating that PROVSEEK outperforms retrieval-based methods for the intelligence extraction task, achieving a 34% improvement in contextual precision/recall; and for threat detection task, PROVSEEK achieves 22%/29% higher precision/recall compared to both a baseline agent approach and State-Of-The-Art (SOTA) Provenance-based Intrusion Detection System (PIDS). In our scalability study, we show that PROVSEEK increases token usage by 1.42x and latency by 1.63x as the database size increases 50x, making it optimal for large-scale deployment. We also conducted an ablation and error analysis study to show how different components of PROVSEEK affect the detection performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.19662v3">HALO: Hardware-aware quantization with low critical-path-delay weights for LLM acceleration</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Quantization is critical for efficiently deploying large language models (LLMs). Yet conventional methods remain hardware-agnostic, limited to bit-width constraints, and do not account for intrinsic circuit characteristics such as the timing behaviors and energy profiles of Multiply-Accumulate (MAC) units. This disconnect from circuit-level behavior limits the ability to exploit available timing margins and energy-saving opportunities, reducing the overall efficiency of deployment on modern accelerators. To address these limitations, we propose HALO, a versatile framework for Hardware-Aware Post-Training Quantization (PTQ). Unlike traditional methods, HALO explicitly incorporates detailed hardware characteristics, including critical-path timing and power consumption, into its quantization approach. HALO strategically selects weights with low critical-path-delays enabling higher operational frequencies and dynamic frequency scaling without disrupting the architecture's dataflow. Remarkably, HALO achieves these improvements with only a few dynamic voltage and frequency scaling (DVFS) adjustments, ensuring simplicity and practicality in deployment. Additionally, by reducing switching activity within the MAC units, HALO effectively lowers energy consumption. Evaluations on accelerators such as Tensor Processing Units (TPUs) and Graphics Processing Units (GPUs) demonstrate that HALO significantly enhances inference efficiency, achieving average performance improvements of 270% and energy savings of 51% over baseline quantization methods, all with minimal impact on accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06390v2">Ghost in the Transformer: Tracing LLM Lineage with SVD-Fingerprint</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted at AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have rapidly advanced and are widely adopted across diverse fields. Due to the substantial computational cost and data requirements of training from scratch, many developers choose to fine-tune or modify existing open-source models. While most adhere to open-source licenses, some falsely claim original training despite clear derivation from public models. This raises pressing concerns about intellectual property protection and highlights the need for reliable methods to verify model provenance. In this paper, we propose GhostSpec, a lightweight yet effective method for verifying LLM lineage without access to training data or modification of model behavior. Our approach constructs compact and robust fingerprints by applying singular value decomposition (SVD) to invariant products of internal attention weight matrices, effectively capturing the structural identity of a model. Unlike watermarking or output-based methods, GhostSpec is fully data-free, non-invasive, and computationally efficient. It demonstrates strong robustness to sequential fine-tuning, pruning, block expansion, and even adversarial transformations. Extensive experiments show that GhostSpec can reliably trace the lineage of transformed models with minimal overhead. By offering a practical solution for model verification and reuse tracking, our method contributes to the protection of intellectual property and fosters a transparent, trustworthy ecosystem for large-scale language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2409.19100v2">Personalizing Prostate Cancer Education for Patients Using an EHR-Integrated LLM Agent</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Cancer patients often lack timely education and personalized support due to clinician workload. This quality improvement study develops and evaluates a Large Language Model (LLM) agent, MedEduChat, which is integrated with the clinic's electronic health records (EHR) and designed to enhance prostate cancer patient education. Fifteen non-metastatic prostate cancer patients and three clinicians recruited from the Mayo Clinic interacted with the agent between May 2024 and April 2025. Findings showed that MedEduChat has a high usability score (UMUX 83.7 out of 100) and improves patients' health confidence (Health Confidence Score rose from 9.9 to 13.9). Clinicians evaluated the patient-chat interaction history and rated MedEduChat as highly correct (2.9 out of 3), complete (2.7 out of 3), and safe (2.7 out of 3), with moderate personalization (2.3 out of 3). This study highlights the potential of LLM agents to improve patient engagement and health education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.19838v3">LLM-Powered GUI Agents in Phone Automation: Surveying Progress and Prospects</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Paper accepted to TMLR 2025, Project Homepage: https://github.com/PhoneLLM/Awesome-LLM-Powered-Phone-GUI-Agents
    </div>
    <details class="paper-abstract">
      With the rapid rise of large language models (LLMs), phone automation has undergone transformative changes. This paper systematically reviews LLM-driven phone GUI agents, highlighting their evolution from script-based automation to intelligent, adaptive systems. We first contextualize key challenges, (i) limited generality, (ii) high maintenance overhead, and (iii) weak intent comprehension, and show how LLMs address these issues through advanced language understanding, multimodal perception, and robust decision-making. We then propose a taxonomy covering fundamental agent frameworks (single-agent, multi-agent, plan-then-act), modeling approaches (prompt engineering, training-based), and essential datasets and benchmarks. Furthermore, we detail task-specific architectures, supervised fine-tuning, and reinforcement learning strategies that bridge user intent and GUI operations. Finally, we discuss open challenges such as dataset diversity, on-device deployment efficiency, user-centric adaptation, and security concerns, offering forward-looking insights into this rapidly evolving field. By providing a structured overview and identifying pressing research gaps, this paper serves as a definitive reference for researchers and practitioners seeking to harness LLMs in designing scalable, user-friendly phone GUI agents. The collection of papers reviewed in this survey will be hosted and regularly updated on the GitHub repository: https://github.com/PhoneLLM/Awesome-LLM-Powered-Phone-GUI-Agents
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04486v2">EDIT-Bench: Evaluating LLM Abilities to Perform Real-World Instructed Code Edits</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Instructed code editing, where LLMs directly modify a developer's existing code based on a user instruction, is becoming a widely used interaction mode in AI coding assistants. However, few benchmarks directly evaluate this capability and current datasets often rely on artificial sources. We introduce EDIT-Bench, a benchmark for evaluating LLM code editing capabilities grounded in real-world usage, i.e., user instructions and code contexts collected in the wild. EDIT-Bench comprises of 540 problems, multiple natural and programming languages, and a diverse set of real-world use cases, ranging from resolving errors to adding features. EDIT-Bench introduces context-dependent problems that require the model to understand code context, highlighted code, and cursor position in addition to the user instruction. We evaluate 40 diverse LLMs and observe that EDIT-Bench is a challenging set of problems where only 1 model scores over 60%. We find that model performance varies across different categories of user instructions. Further, we find that varying levels of contextual information greatly affect task success rate, with performance varying up to 11%, indicating the importance of evaluating with realistic context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.16124v3">Benchmarking LLM Privacy Recognition for Social Robot Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 18 pages, 7 figures. Dakota Sullivan and Shirley Zhang contributed equally to this work
    </div>
    <details class="paper-abstract">
      While robots have previously utilized rule-based systems or probabilistic models for user interaction, the rapid evolution of large language models (LLMs) presents new opportunities to develop LLM-powered robots for enhanced human-robot interaction (HRI). To fully realize these capabilities, however, robots need to collect data such as audio, fine-grained images, video, and locations. As a result, LLMs often process sensitive personal information, particularly within private environments, such as homes. Given the tension between utility and privacy risks, evaluating how current LLMs manage sensitive data is critical. Specifically, we aim to explore the extent to which out-of-the-box LLMs are privacy-aware in the context of household robots. In this work, we present a set of privacy-relevant scenarios developed using the Contextual Integrity (CI) framework. We first surveyed users' privacy preferences regarding in-home robot behaviors and then examined how their privacy orientations affected their choices of these behaviors (N = 450). We then provided the same set of scenarios and questions to state-of-the-art LLMs (N = 10) and found that the agreement between humans and LLMs was generally low. To further investigate the capabilities of LLMs as potential privacy controllers, we implemented four additional prompting strategies and compared their results. We discuss the performance of the evaluated models as well as the implications and potential of AI privacy awareness in human-robot interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.10950v2">Unveiling Challenges for LLMs in Enterprise Data Engineering</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) promise to automate data engineering on tabular data, offering enterprises a valuable opportunity to cut the high costs of manual data handling. But the enterprise domain comes with unique challenges that existing LLM-based approaches for data engineering often overlook, such as large table sizes, more complex tasks, and the need for internal knowledge. To bridge these gaps, we identify key enterprise-specific challenges related to data, tasks, and background knowledge and extensively evaluate how they affect data engineering with LLMs. Our analysis reveals that LLMs face substantial limitations in real-world enterprise scenarios, with accuracy declining sharply. Our findings contribute to a systematic understanding of LLMs for enterprise data engineering to support their adoption in industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13373v1">A Novel Hierarchical Integration Method for Efficient Model Merging in Medical LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) face significant challenges in distributed healthcare, including consolidating specialized domain knowledge across institutions while maintaining privacy, reducing computational overhead, and preventing catastrophic forgetting during model updates.This paper presents a systematic evaluation of six parameter-space merging techniques applied to two architecturally compatible medical LLMs derived from the Mistral-7B base model. We introduce a novel hierarchical method that combines selective Optimal Transport (OT) alignment for attention layers with cosine similarity-weighted interpolation, designed to address permutation variance while minimizing computational overhead for edge deployment scenarios. Our study evaluates Task Arithmetic, Linear Averaging, DARE-TIES, DELLA, Breadcrumbs, and our Hierarchical approach across five medical benchmarks. Results demonstrate that architecturally compatible models benefit significantly from simple averaging methods, with Task Arithmetic achieving 45.80% accuracy on MedQA, outperforming complex pruning-based approaches. These findings offer critical insights for the deployment of distributed medical AI in resource-constrained IoT environments, where computational efficiency and model compatibility are paramount. Our work establishes that for architecturally compatible models, simple averaging provides a robust and computationally efficient baseline for knowledge consolidation, offering a pragmatic path forward for scalable medical AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04108v2">Can Linear Probes Measure LLM Uncertainty?</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Effective Uncertainty Quantification (UQ) represents a key aspect for reliable deployment of Large Language Models (LLMs) in automated decision-making and beyond. Yet, for LLM generation with multiple choice structure, the state-of-the-art in UQ is still dominated by the naive baseline given by the maximum softmax score. To address this shortcoming, we demonstrate that taking a principled approach via Bayesian statistics leads to improved performance despite leveraging the simplest possible model, namely linear regression. More precisely, we propose to train multiple Bayesian linear models, each predicting the output of a layer given the output of the previous one. Based on the obtained layer-level posterior distributions, we infer the global uncertainty level of the LLM by identifying a sparse combination of distributional features, leading to an efficient UQ scheme. Numerical experiments on various LLMs show consistent improvement over state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13341v1">An LLM-based Quantitative Framework for Evaluating High-Stealthy Backdoor Risks in OSS Supply Chains</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 7 figures, 4 tables, conference
    </div>
    <details class="paper-abstract">
      In modern software development workflows, the open-source software supply chain contributes significantly to efficient and convenient engineering practices. With increasing system complexity, using open-source software as third-party dependencies has become a common practice. However, the lack of maintenance for underlying dependencies and insufficient community auditing create challenges in ensuring source code security and the legitimacy of repository maintainers, especially under high-stealthy backdoor attacks exemplified by the XZ-Util incident. To address these problems, we propose a fine-grained project evaluation framework for backdoor risk assessment in open-source software. The framework models stealthy backdoor attacks from the viewpoint of the attacker and defines targeted metrics for each attack stage. In addition, to overcome the limitations of static analysis in assessing the reliability of repository maintenance activities such as irregular committer privilege escalation and limited participation in reviews, the framework uses large language models (LLMs) to conduct semantic evaluation of code repositories without relying on manually crafted patterns. The framework is evaluated on sixty six high-priority packages in the Debian ecosystem. The experimental results indicate that the current open-source software supply chain is exposed to various security risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.11864v2">NeuroStrike: Neuron-Level Attacks on Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability. This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13319v1">Whistledown: Combining User-Level Privacy with Conversational Coherence in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Users increasingly rely on large language models (LLMs) for personal, emotionally charged, and socially sensitive conversations. However, prompts sent to cloud-hosted models can contain personally identifiable information (PII) that users do not want logged, retained, or leaked. We observe this to be especially acute when users discuss friends, coworkers, or adversaries, i.e., when they spill the tea. Enterprises face the same challenge when they want to use LLMs for internal communication and decision-making. In this whitepaper, we present Whistledown, a best-effort privacy layer that modifies prompts before they are sent to the LLM. Whistledown combines pseudonymization and $ε$-local differential privacy ($ε$-LDP) with transformation caching to provide best-effort privacy protection without sacrificing conversational utility. Whistledown is designed to have low compute and memory overhead, allowing it to be deployed directly on a client's device in the case of individual users. For enterprise users, Whistledown is deployed centrally within a zero-trust gateway that runs on an enterprise's trusted infrastructure. Whistledown requires no changes to the existing APIs of popular LLM providers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13305v1">SAINT: Service-level Integration Test Generation with Program Analysis and LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted at ICSE'26
    </div>
    <details class="paper-abstract">
      Enterprise applications are typically tested at multiple levels, with service-level testing playing an important role in validating application functionality. Existing service-level testing tools, especially for RESTful APIs, often employ fuzzing and/or depend on OpenAPI specifications which are not readily available in real-world enterprise codebases. Moreover, these tools are limited in their ability to generate functional tests that effectively exercise meaningful scenarios. In this work, we present SAINT, a novel white-box testing approach for service-level testing of enterprise Java applications. SAINT combines static analysis, large language models (LLMs), and LLM-based agents to automatically generate endpoint and scenario-based tests. The approach builds two key models: an endpoint model, capturing syntactic and semantic information about service endpoints, and an operation dependency graph, capturing inter-endpoint ordering constraints. SAINT then employs LLM-based agents to generate tests. Endpoint-focused tests aim to maximize code and database interaction coverage. Scenario-based tests are synthesized by extracting application use cases from code and refining them into executable tests via planning, action, and reflection phases of the agentic loop. We evaluated SAINT on eight Java applications, including a proprietary enterprise application. Our results illustrate the effectiveness of SAINT in coverage, fault detection, and scenario generation. Moreover, a developer survey provides strong endorsement of the scenario-based tests generated by SAINT. Overall, our work shows that combining static analysis with agentic LLM workflows enables more effective, functional, and developer-aligned service-level test generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.02962v5">RAG-R1: Incentivizing the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), despite their remarkable capabilities, are prone to generating hallucinated or outdated content due to their static internal knowledge. While Retrieval-Augmented Generation (RAG) integrated with Reinforcement Learning (RL) offers a solution, these methods are fundamentally constrained by a single-query mode, leading to prohibitive latency and inherent brittleness. To overcome these limitations, we introduce RAG-R1, a novel two-stage training framework centered around multi-query parallelism. Our framework enables LLMs to adaptively leverage internal and external knowledge during the reasoning process while transitioning from the single-query mode to multi-query parallelism. This architectural shift bolsters reasoning robustness while significantly reducing inference latency. Extensive experiments on seven question-answering benchmarks confirm the superiority of our method, which outperforms the strongest baseline by up to 13.7% and decreases inference time by 11.1%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13290v1">Dropouts in Confidence: Moral Uncertainty in Human-LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Humans display significant uncertainty when confronted with moral dilemmas, yet the extent of such uncertainty in machines and AI agents remains underexplored. Recent studies have confirmed the overly confident tendencies of machine-generated responses, particularly in large language models (LLMs). As these systems are increasingly embedded in ethical decision-making scenarios, it is important to understand their moral reasoning and the inherent uncertainties in building reliable AI systems. This work examines how uncertainty influences moral decisions in the classical trolley problem, analyzing responses from 32 open-source models and 9 distinct moral dimensions. We first find that variance in model confidence is greater across models than within moral dimensions, suggesting that moral uncertainty is predominantly shaped by model architecture and training method. To quantify uncertainty, we measure binary entropy as a linear combination of total entropy, conditional entropy, and mutual information. To examine its effects, we introduce stochasticity into models via "dropout" at inference time. Our findings show that our mechanism increases total entropy, mainly through a rise in mutual information, while conditional entropy remains largely unchanged. Moreover, this mechanism significantly improves human-LLM moral alignment, with correlations in mutual information and alignment score shifts. Our results highlight the potential to better align model-generated decisions and human preferences by deliberately modulating uncertainty and reducing LLMs' confidence in morally complex scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.25506v3">Reflections on the Reproducibility of Commercial LLM Performance in Empirical Software Engineering Studies</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models have gained remarkable interest in industry and academia. The increasing interest in LLMs in academia is also reflected in the number of publications on this topic over the last years. For instance, alone 78 of the around 425 publications at ICSE 2024 performed experiments with LLMs. Conducting empirical studies with LLMs remains challenging and raises questions on how to achieve reproducible results, for both researchers and practitioners. One important step towards excelling in empirical research on LLM and their application is to first understand to what extent current research results are eventually reproducible and what factors may impede reproducibility. This investigation is within the scope of our work. We contribute an analysis of the reproducibility of LLM-centric studies, provide insights into the factors impeding reproducibility, and discuss suggestions on how to improve the current state. In particular, we studied the 85 articles describing LLM-centric studies, published at ICSE 2024 and ASE 2024. Of the 85 articles, 18 provided research artefacts and used OpenAI models. We attempted to replicate those 18 studies. Of the 18 studies, only five were sufficiently complete and executable. For none of the five studies, we were able to fully reproduce the results. Two studies seemed to be partially reproducible, and three studies did not seem to be reproducible. Our results highlight not only the need for stricter research artefact evaluations but also for more robust study designs to ensure the reproducible value of future publications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.22041v3">An LLM-based Simulation Framework for Embodied Conversational Agents in Psychological Counseling</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Due to privacy concerns, open dialogue datasets for mental health are primarily generated through human or AI synthesis methods. However, the inherent implicit nature of psychological processes, particularly those of clients, poses challenges to the authenticity and diversity of synthetic data. In this paper, we propose ECAs (short for Embodied Conversational Agents), a framework for embodied agent simulation based on Large Language Models (LLMs) that incorporates multiple psychological theoretical principles.Using simulation, we expand real counseling case data into a nuanced embodied cognitive memory space and generate dialogue data based on high-frequency counseling questions.We validated our framework using the D4 dataset. First, we created a public ECAs dataset through batch simulations based on D4. Licensed counselors evaluated our method, demonstrating that it significantly outperforms baselines in simulation authenticity and necessity. Additionally, two LLM-based automated evaluation methods were employed to confirm the higher quality of the generated dialogues compared to the baselines. The source code and dataset are available at https://github.com/AIR-DISCOVER/ECAs-Dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.01223v2">Jailbreaking LLMs via Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks. However, they remain exposed to jailbreak attacks, eliciting harmful responses. The nested scenario strategy has been increasingly adopted across various methods, demonstrating immense potential. Nevertheless, these methods are easily detectable due to their prominent malicious intentions. In this work, we are the first to find and systematically verify that LLMs' alignment defenses are not sensitive to nested scenarios, where these scenarios are highly semantically relevant to the queries and incorporate targeted toxic knowledge. This is a crucial yet insufficiently explored direction. Based on this, we propose RTS-Attack (Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge), an adaptive and automated framework to examine LLMs' alignment. By building scenarios highly relevant to the queries and integrating targeted toxic knowledge, RTS-Attack bypasses the alignment defenses of LLMs. Moreover, the jailbreak prompts generated by RTS-Attack are free from harmful queries, leading to outstanding concealment. Extensive experiments demonstrate that RTS-Attack exhibits superior performance in both efficiency and universality compared to the baselines across diverse advanced LLMs, including GPT-4o, Llama3-70b, and Gemini-pro. Our complete code is available at https://github.com/nercode/Work. WARNING: THIS PAPER CONTAINS POTENTIALLY HARMFUL CONTENT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13254v1">Souper-Model: How Simple Arithmetic Unlocks State-of-the-Art LLM Performance</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, but their training remains resource- and time-intensive, requiring massive compute power and careful orchestration of training procedures. Model souping-the practice of averaging weights from multiple models of the same architecture-has emerged as a promising pre- and post-training technique that can enhance performance without expensive retraining. In this paper, we introduce Soup Of Category Experts (SoCE), a principled approach for model souping that utilizes benchmark composition to identify optimal model candidates and applies non-uniform weighted averaging to maximize performance. Contrary to previous uniform-averaging approaches, our method leverages the observation that benchmark categories often exhibit low inter-correlations in model performance. SoCE identifies "expert" models for each weakly-correlated category cluster and combines them using optimized weighted averaging rather than uniform weights. We demonstrate that the proposed method improves performance and robustness across multiple domains, including multilingual capabilities, tool calling, and math and achieves state-of-the-art results on the Berkeley Function Calling Leaderboard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.06261v4">Hogwild! Inference: Parallel LLM Generation via Concurrent Attention</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated the ability to tackle increasingly complex tasks through advanced reasoning, long-form content generation, and tool use. Solving these tasks often involves long inference-time computations. In human problem solving, a common strategy to expedite work is collaboration: by dividing the problem into sub-tasks, exploring different strategies concurrently, etc. Recent research has shown that LLMs can also operate in parallel by implementing explicit cooperation frameworks, such as voting mechanisms or the explicit creation of independent sub-tasks that can be executed in parallel. However, each of these frameworks may not be suitable for all types of tasks, which can hinder their applicability. In this work, we propose a different design approach: we run LLM "workers" in parallel , allowing them to synchronize via a concurrently-updated attention cache and prompt these workers to decide how best to collaborate. Our approach allows the LLM instances to come up with their own collaboration strategy for the problem at hand, all the while "seeing" each other's memory in the concurrent KV cache. We implement this approach via Hogwild! Inference: a parallel LLM inference engine where multiple instances of the same LLM run in parallel with the same attention cache, with "instant" access to each other's memory. Hogwild! Inference takes advantage of Rotary Position Embeddings (RoPE) to avoid recomputation while improving parallel hardware utilization. We find that modern reasoning-capable LLMs can perform inference with shared Key-Value cache out of the box, without additional fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13233v1">LLM-based Multi-Agent System for Simulating Strategic and Goal-Oriented Data Marketplaces</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 10 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Data marketplaces, which mediate the purchase and exchange of data from third parties, have attracted growing attention for reducing the cost and effort of data collection while enabling the trading of diverse datasets. However, a systematic understanding of the interactions between market participants, data, and regulations remains limited. To address this gap, we propose a Large Language Model-based Multi-Agent System (LLM-MAS) for data marketplaces. In our framework, buyer and seller agents powered by LLMs operate with explicit objectives and autonomously perform strategic actions, such as planning, searching, purchasing, pricing, and updating data. These agents can reason about market dynamics, forecast future demand, and adjust strategies accordingly. Unlike conventional model-based simulations, which are typically constrained to predefined rules, LLM-MAS supports broader and more adaptive behavior selection through natural language reasoning. We evaluated the framework via simulation experiments using three distribution-based metrics: (1) the number of purchases per dataset, (2) the number of purchases per buyer, and (3) the number of repeated purchases of the same dataset. The results demonstrate that LLM-MAS more faithfully reproduces trading patterns observed in real data marketplaces compared to traditional approaches, and further captures the emergence and evolution of market trends.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.14398v4">On the Limitations of Language Targeted Pruning: Investigating the Calibration Language Impact in Multilingual LLM Pruning</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted for publication in TACL
    </div>
    <details class="paper-abstract">
      Recent advances in large language model (LLM) pruning have shown state-of-the-art (SotA) compression results in post-training and retraining-free settings while maintaining high predictive performance. However, previous research mainly considered calibrating based on English text, despite the multilingual nature of modern LLMs and their frequent use in non-English languages. This analysis paper conducts an in-depth investigation of the performance and internal representation changes associated with pruning multilingual language models for monolingual applications. We present the first comprehensive empirical study, comparing different calibration languages for pruning multilingual models across diverse languages, tasks, models, and SotA pruning techniques. We further analyze the latent subspaces, pruning masks, and individual neurons within pruned models. Our results reveal that while calibration on the target language effectively retains perplexity and yields high signal-to-noise ratios, it does not consistently improve downstream task performance. Further analysis of internal representations at three different levels highlights broader limitations of current pruning approaches: While they effectively preserve dominant information like language-specific features, this is insufficient to counteract the loss of nuanced, language-agnostic features that are crucial for knowledge retention and reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13223v1">TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Emerging reasoning LLMs such as OpenAI-o1 and DeepSeek-R1 have achieved strong performance on complex reasoning tasks by generating long chain-of-thought (CoT) traces. However, these long CoTs result in increased token usage, leading to higher inference latency and memory consumption. As a result, balancing accuracy and reasoning efficiency has become essential for deploying reasoning LLMs in practical applications. Existing long-to-short (Long2Short) methods aim to reduce inference length but often sacrifice accuracy, revealing a need for an approach that maintains performance while lowering token costs. To address this efficiency-accuracy tradeoff, we propose TokenSqueeze, a novel Long2Short method that condenses reasoning paths while preserving performance and relying exclusively on self-generated data. First, to prevent performance degradation caused by excessive compression of reasoning depth, we propose to select self-generated samples whose reasoning depth is adaptively matched to the complexity of the problem. To further optimize the linguistic expression without altering the underlying reasoning paths, we introduce a distribution-aligned linguistic refinement method that enhances the clarity and conciseness of the reasoning path while preserving its logical integrity. Comprehensive experimental results demonstrate the effectiveness of TokenSqueeze in reducing token usage while maintaining accuracy. Notably, DeepSeek-R1-Distill-Qwen-7B fine-tuned using our proposed method achieved a 50\% average token reduction while preserving accuracy on the MATH500 benchmark. TokenSqueeze exclusively utilizes the model's self-generated data, enabling efficient and high-fidelity reasoning without relying on manually curated short-answer datasets across diverse applications. Our code is available at https://github.com/zhangyx1122/TokenSqueeze.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.00034v2">Is Our Chatbot Telling Lies? Assessing Correctness of an LLM-based Dutch Support Chatbot</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 10 pages + 2 pages references, 4 figures
    </div>
    <details class="paper-abstract">
      Companies support their customers using live chats and chatbots to gain their loyalty. AFAS is a Dutch company aiming to leverage the opportunity large language models (LLMs) offer to answer customer queries with minimal to no input from its customer support team. Adding to its complexity, it is unclear what makes a response correct, and that too in Dutch. Further, with minimal data available for training, the challenge is to identify whether an answer generated by a large language model is correct and do it on the fly. This study is the first to define the correctness of a response based on how the support team at AFAS makes decisions. It leverages literature on natural language generation and automated answer grading systems to automate the decision-making of the customer support team. We investigated questions requiring a binary response (e.g., Would it be possible to adjust tax rates manually?) or instructions (e.g., How would I adjust tax rate manually?) to test how close our automated approach reaches support rating. Our approach can identify wrong messages in 55\% of the cases. This work demonstrates the potential for automatically assessing when our chatbot may provide incorrect or misleading answers. Specifically, we contribute (1) a definition and metrics for assessing correctness, and (2) suggestions to improve correctness with respect to regional language and question type.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.16407v2">CREME: Robustness Enhancement of Code LLMs via Layer-Aware Model Editing</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in code generation, where the natural language prompt plays a crucial role in conveying user intent to the model. However, prior studies have shown that LLMs are highly sensitive to prompt perturbations. Minor modifications in wording, syntax, or formatting can significantly reduce the functional correctness of generated code. As perturbations frequently occur in real-world scenarios, improving the robustness of LLMs to prompt perturbations is essential for ensuring reliable performance in practical code generation. In this paper, we introduce CREME (Code Robustness Enhancement via Model Editing), a novel approach that enhances LLM robustness through targeted parameter updates. CREME first identifies robustness-sensitive layers by comparing hidden states between an original prompt and its perturbed variant. Then, it performs lightweight parameter editing at the identified layer to reduce performance degradation. We evaluate CREME on two widely used code generation benchmarks (HumanEval and MBPP) along with their perturbed counterparts. Experimental results show that CREME improves Pass@1 accuracy by 63% on perturbed prompts while maintaining stable performance on clean inputs, with accuracy deviations within 1%. Further analysis reveals that robustness-sensitive layers are primarily concentrated in the middle and deeper layers of the network, and their locations vary across different model architectures. These insights provide a valuable foundation for developing future robustness-oriented editing strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13169v1">TCM-5CEval: Extended Deep Evaluation Benchmark for LLM's Comprehensive Clinical Research Competence in Traditional Chinese Medicine</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 17 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated exceptional capabilities in general domains, yet their application in highly specialized and culturally-rich fields like Traditional Chinese Medicine (TCM) requires rigorous and nuanced evaluation. Building upon prior foundational work such as TCM-3CEval, which highlighted systemic knowledge gaps and the importance of cultural-contextual alignment, we introduce TCM-5CEval, a more granular and comprehensive benchmark. TCM-5CEval is designed to assess LLMs across five critical dimensions: (1) Core Knowledge (TCM-Exam), (2) Classical Literacy (TCM-LitQA), (3) Clinical Decision-making (TCM-MRCD), (4) Chinese Materia Medica (TCM-CMM), and (5) Clinical Non-pharmacological Therapy (TCM-ClinNPT). We conducted a thorough evaluation of fifteen prominent LLMs, revealing significant performance disparities and identifying top-performing models like deepseek\_r1 and gemini\_2\_5\_pro. Our findings show that while models exhibit proficiency in recalling foundational knowledge, they struggle with the interpretative complexities of classical texts. Critically, permutation-based consistency testing reveals widespread fragilities in model inference. All evaluated models, including the highest-scoring ones, displayed a substantial performance degradation when faced with varied question option ordering, indicating a pervasive sensitivity to positional bias and a lack of robust understanding. TCM-5CEval not only provides a more detailed diagnostic tool for LLM capabilities in TCM but aldso exposes fundamental weaknesses in their reasoning stability. To promote further research and standardized comparison, TCM-5CEval has been uploaded to the Medbench platform, joining its predecessor in the "In-depth Challenge for Comprehensive TCM Abilities" special track.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.22564v2">Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13147v1">OTARo: Once Tuning for All Precisions toward Robust On-Device LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) fine-tuning techniques not only improve the adaptability to diverse downstream tasks, but also mitigate adverse effects of model quantization. Despite this, conventional quantization suffers from its structural limitation that hinders flexibility during the fine-tuning and deployment stages. Practical on-device tasks demand different quantization precisions (i.e. different bit-widths), e.g., understanding tasks tend to exhibit higher tolerance to reduced precision compared to generation tasks. Conventional quantization, typically relying on scaling factors that are incompatible across bit-widths, fails to support the on-device switching of precisions when confronted with complex real-world scenarios. To overcome the dilemma, we propose OTARo, a novel method that enables on-device LLMs to flexibly switch quantization precisions while maintaining performance robustness through once fine-tuning. OTARo introduces Shared Exponent Floating Point (SEFP), a distinct quantization mechanism, to produce different bit-widths through simple mantissa truncations of a single model. Moreover, to achieve bit-width robustness in downstream applications, OTARo performs a learning process toward losses induced by different bit-widths. The method involves two critical strategies: (1) Exploitation-Exploration Bit-Width Path Search (BPS), which iteratively updates the search path via a designed scoring mechanism; (2) Low-Precision Asynchronous Accumulation (LAA), which performs asynchronous gradient accumulations and delayed updates under low bit-widths. Experiments on popular LLMs, e.g., LLaMA3.2-1B, LLaMA3-8B, demonstrate that OTARo achieves consistently strong and robust performance for all precisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22963v3">CompressionAttack: Exploiting Prompt Compression as a New Attack Surface in LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      LLM-powered agents often use prompt compression to reduce inference costs, but this introduces a new security risk. Compression modules, which are optimized for efficiency rather than safety, can be manipulated by adversarial inputs, causing semantic drift and altering LLM behavior. This work identifies prompt compression as a novel attack surface and presents CompressionAttack, the first framework to exploit it. CompressionAttack includes two strategies: HardCom, which uses discrete adversarial edits for hard compression, and SoftCom, which performs latent-space perturbations for soft compression. Experiments on multiple LLMs show up to an average ASR of 83% and 87% in two tasks, while remaining highly stealthy and transferable. Case studies in three practical scenarios confirm real-world impact, and current defenses prove ineffective, highlighting the need for stronger protections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01891v2">Multi-Personality Generation of LLMs at Decoding-time</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted by WSDM 2026
    </div>
    <details class="paper-abstract">
      Multi-personality generation for LLMs, enabling simultaneous embodiment of multiple personalization attributes, is a fundamental challenge. Existing retraining-based approaches are costly and poorly scalable, while decoding-time methods often rely on external models or heuristics, limiting flexibility and robustness. In this paper, we propose a novel Multi-Personality Generation (MPG) framework under the decoding-time combination paradigm. It flexibly controls multi-personality without relying on scarce multi-dimensional models or extra training, leveraging implicit density ratios in single-dimensional models as a "free lunch" to reformulate the task as sampling from a target strategy aggregating these ratios. To implement MPG efficiently, we design Speculative Chunk-level based Rejection sampling (SCR), which generates responses in chunks and parallelly validates them via estimated thresholds within a sliding window. This significantly reduces computational overhead while maintaining high-quality generation. Experiments on MBTI personality and Role-Playing demonstrate the effectiveness of MPG, showing improvements up to 16%-18%. Code and data are available at https://github.com/Libra117/MPG .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.00829v2">Exposing the Cracks: Vulnerabilities of Retrieval-Augmented LLM-based Machine Translation</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      \textbf{RE}trieval-\textbf{A}ugmented \textbf{L}LM-based \textbf{M}achine \textbf{T}ranslation (REAL-MT) shows promise for knowledge-intensive tasks like idiomatic translation, but its reliability under noisy retrieval contexts remains poorly understood despite this being a common challenge in real-world deployment. To address this gap, we propose a noise synthesis framework and new metrics to evaluate the robustness of REAL-MT systematically. Using this framework, we instantiate REAL-MT with Qwen-series models, including standard LLMs and large reasoning models (LRMs) with enhanced reasoning, and evaluate their performance on idiomatic translation across high-, medium-, and low-resource language pairs under synthesized noise. Our results show that low-resource language pairs, which rely more heavily on retrieved context, degrade more severely under noise than high-resource ones and often produce nonsensical translations. Although LRMs possess enhanced reasoning capabilities, they show no improvement in error correction and are even more susceptible to noise, tending to rationalize incorrect contexts. We find that this stems from an attention shift away from the source idiom to noisy content, while confidence increases despite declining accuracy, indicating poor calibration. To mitigate these issues, we investigate training-free and fine-tuning strategies, which improve robustness at the cost of performance in clean contexts, revealing a fundamental trade-off. Our findings highlight the limitations of current approaches, underscoring the need for self-verifying integration mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.13768v3">Evaluation-Driven Development and Operations of LLM Agents: A Process Model and Reference Architecture</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Revised based on review comments. Submission under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have enabled the emergence of LLM agents, systems capable of pursuing under-specified goals and adapting after deployment. Evaluating such agents is challenging because their behavior is open ended, probabilistic, and shaped by system-level interactions over time. Traditional evaluation methods, built around fixed benchmarks and static test suites, fail to capture emergent behaviors or support continuous adaptation across the lifecycle. To ground a more systematic approach, we conduct a multivocal literature review (MLR) synthesizing academic and industrial evaluation practices. The findings directly inform two empirically derived artifacts: a process model and a reference architecture that embed evaluation as a continuous, governing function rather than a terminal checkpoint. Together they constitute the evaluation-driven development and operations (EDDOps) approach, which unifies offline (development-time) and online (runtime) evaluation within a closed feedback loop. By making evaluation evidence drive both runtime adaptation and governed redevelopment, EDDOps supports safer, more traceable evolution of LLM agents aligned with changing objectives, user needs, and governance constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13007v1">GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 This paper has been accepted by AAAI 2026-AIA and designated as an oral presentation paper
    </div>
    <details class="paper-abstract">
      Alignment of large language models (LLMs) with human preferences typically relies on supervised reward models or external judges that demand abundant annotations. However, in fields that rely on professional knowledge, such as medicine and law, such large-scale preference labels are often unachievable. In this paper, we propose a generative entropy-guided preference modeling approach named GEM for LLMs aligment at low-resource and domain-specific scenarios. Instead of training a discriminative reward model on preference data, we directly train the LLM to internalize a closed-loop optimization architecture that can extract and exploit the multi-dimensional, fine-grained cognitive signals implicit in human preferences. Specifically, our Cognitive Filtering module, based on entropy theory in decision making, first leverages Chain-of-Thought (CoT) prompting to generate diverse candidate reasoning chains (CoTs) from preference data. Subsequently, it introduces a token scoring mechanism to rank and weight the sampled CoTs, boosting the importance of high-confidence answers and strategically high-entropy tokens. Building on these filtered preferences, we fine-tune the LLM using a novel self-evaluated group advantage algorithm, SEGA, which effectively aggregates group-level cognitive signals and transforms the entropy-based scores into implicit rewards for policy optimization. In these ways, GEM empowers the LLM to rely on its own judgments and establishes an entropy-guided closed-loop cognitive optimization framework, enabling highly efficient few-shot alignment of LLMs. Experiments on general benchmarks and domain-specific tasks (such as mathematical reasoning and medical dialogues) demonstrate that our GEM achieves significant improvements with few-shot preference data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06852v3">Differentiated Directional Intervention A Framework for Evading LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 AAAI-26-AIA
    </div>
    <details class="paper-abstract">
      Safety alignment instills in Large Language Models (LLMs) a critical capacity to refuse malicious requests. Prior works have modeled this refusal mechanism as a single linear direction in the activation space. We posit that this is an oversimplification that conflates two functionally distinct neural processes: the detection of harm and the execution of a refusal. In this work, we deconstruct this single representation into a Harm Detection Direction and a Refusal Execution Direction. Leveraging this fine-grained model, we introduce Differentiated Bi-Directional Intervention (DBDI), a new white-box framework that precisely neutralizes the safety alignment at critical layer. DBDI applies adaptive projection nullification to the refusal execution direction while suppressing the harm detection direction via direct steering. Extensive experiments demonstrate that DBDI outperforms prominent jailbreaking methods, achieving up to a 97.88\% attack success rate on models such as Llama-2. By providing a more granular and mechanistic framework, our work offers a new direction for the in-depth understanding of LLM safety alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12991v1">Fine-Tuned LLMs Know They Don't Know: A Parameter-Efficient Approach to Recovering Honesty</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Accepted by AAAI 2026 Main Track
    </div>
    <details class="paper-abstract">
      The honesty of Large Language Models (LLMs) is increasingly important for safe deployment in high-stakes domains. However, this crucial trait is severely undermined by supervised fine-tuning (SFT), a common technique for model specialization. Existing recovery methods rely on data-intensive global parameter adjustments, implicitly assuming that SFT deeply corrupts the models' ability to recognize their knowledge boundaries. However, we observe that fine-tuned LLMs still preserve this ability; what is damaged is their capacity to faithfully express that awareness. Building on this, we propose Honesty-Critical Neurons Restoration (HCNR) to surgically repair this suppressed capacity. HCNR identifies and restores key expression-governing neurons to their pre-trained state while harmonizing them with task-oriented neurons via Hessian-guided compensation. Experiments on four QA tasks and five LLM families demonstrate that HCNR effectively recovers 33.25% of the compromised honesty while achieving at least 2.23x speedup with over 10x less data compared to baseline methods, offering a practical solution for trustworthy LLM deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12977v1">ArtiWorld: LLM-Driven Articulation of 3D Objects in Scenes</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Building interactive simulators and scalable robot-learning environments requires a large number of articulated assets. However, most existing 3D assets in simulation are rigid, and manually converting them into articulated objects is extremely labor- and cost-intensive. This raises a natural question: can we automatically identify articulable objects in a scene and convert them into articulated assets directly? In this paper, we present ArtiWorld, a scene-aware pipeline that localizes candidate articulable objects from textual scene descriptions and reconstructs executable URDF models that preserve the original geometry. At the core of this pipeline is Arti4URDF, which leverages 3D point cloud, prior knowledge of a large language model (LLM), and a URDF-oriented prompt design to rapidly convert rigid objects into interactive URDF-based articulated objects while maintaining their 3D shape. We evaluate ArtiWorld at three levels: 3D simulated objects, full 3D simulated scenes, and real-world scan scenes. Across all three settings, our method consistently outperforms existing approaches and achieves state-of-the-art performance, while preserving object geometry and correctly capturing object interactivity to produce usable URDF-based articulated models. This provides a practical path toward building interactive, robot-ready simulation environments directly from existing 3D assets. Code and data will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23188v2">Diagnose, Localize, Align: A Full-Stack Framework for Reliable LLM Multi-Agent Systems under Instruction Conflicts</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Upon further review, we realized that the version submitted to arXiv was not the final draft and omits crucial results and discussion. To avoid confusion and ensure the integrity of the record, we request withdrawal and will resubmit once the complete work is ready
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-powered multi-agent systems (MAS) have rapidly advanced collaborative reasoning, tool use, and role-specialized coordination in complex tasks. However, reliability-critical deployment remains hindered by a systemic failure mode: hierarchical compliance under instruction conflicts (system-user, peer-peer), where agents misprioritize system-level rules in the presence of competing demands. Moreover, widely used macro-level metrics (e.g., pass@k) obscure these micro-level violations and offer little actionable guidance for remedy. In this work, we present a full-stack, three-stage framework: (1) Diagnose - Contextualized Role Adherence Score (CRAS), a query-wise, context-aware scoring metric that decomposes role adherence into four measurable dimensions; (2) Localize - attention drift analysis revealing that instruction conflicts are resolved by attention heads that are largely concentrated in middle layers; (3) Align - Surgical Alignment of Instruction Layers (SAIL), which installs LoRA only on the localized focal layers and optimizes a token-weighted DPO-style preference objective that credits tokens by their focal attentional contribution. Across standard benchmarks and MAS frameworks, our surgical approach improves instruction hierarchy compliance (e.g., +5.60% with AutoGen on MedQA) without full-model finetuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12922v1">Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 20 pages, 8 figures, 9 tables; Annual AAAI Conference on Artificial Intelligence (AAAI-26) (to appear) (Please cite our conference version.)
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based recommender systems have achieved high-quality performance by bridging the discrepancy between the item space and the language space through item tokenization. However, existing item tokenization methods typically require training separate models for each item domain, limiting generalization. Moreover, the diverse distributions and semantics across item domains make it difficult to construct a unified tokenization that preserves domain-specific information. To address these challenges, we propose UniTok, a Unified item Tokenization framework that integrates our own mixture-of-experts (MoE) architecture with a series of codebooks to convert items into discrete tokens, enabling scalable tokenization while preserving semantic information across multiple item domains. Specifically, items from different domains are first projected into a unified latent space through a shared encoder. They are then routed to domain-specific experts to capture the unique semantics, while a shared expert, which is always active, encodes common knowledge transferable across domains. Additionally, to mitigate semantic imbalance across domains, we present a mutual information calibration mechanism, which guides the model towards retaining similar levels of semantic information for each domain. Comprehensive experiments on wide-ranging real-world datasets demonstrate that the proposed UniTok framework is (a) highly effective: achieving up to 51.89% improvements over strong benchmarks, (b) theoretically sound: showing the analytical validity of our architectural design and optimization; and (c) highly generalizable: demonstrating robust performance across diverse domains without requiring per-domain retraining, a capability not supported by existing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11000v2">DialogGraph-LLM: Graph-Informed LLMs for End-to-End Audio Dialogue Intent Recognition</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 8 pages, 2 figures. To appear in: Proceedings of the 28th European Conference on Artificial Intelligence (ECAI 2025), Frontiers in Artificial Intelligence and Applications, Vol. 413. DOI: 10.3233/FAIA251182
    </div>
    <details class="paper-abstract">
      Recognizing speaker intent in long audio dialogues among speakers has a wide range of applications, but is a non-trivial AI task due to complex inter-dependencies in speaker utterances and scarce annotated data. To address these challenges, an end-to-end framework, namely DialogGraph-LLM, is proposed in the current work. DialogGraph-LLM combines a novel Multi-Relational Dialogue Attention Network (MR-DAN) architecture with multimodal foundation models (e.g., Qwen2.5-Omni-7B) for direct acoustic-to-intent inference. An adaptive semi-supervised learning strategy is designed using LLM with a confidence-aware pseudo-label generation mechanism based on dual-threshold filtering using both global and class confidences, and an entropy-based sample selection process that prioritizes high-information unlabeled instances. Extensive evaluations on the proprietary MarketCalls corpus and the publicly available MIntRec 2.0 benchmark demonstrate DialogGraph-LLM's superiority over strong audio and text-driven baselines. The framework demonstrates strong performance and efficiency in intent recognition in real world scenario audio dialogues, proving its practical value for audio-rich domains with limited supervision. Our code is available at https://github.com/david188888/DialogGraph-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12901v1">Online Learning of HTN Methods for integrated LLM-HTN Planning</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 The Twelfth Annual Conference on Advances in Cognitive Systems (ACS-2025)
    </div>
    <details class="paper-abstract">
      We present online learning of Hierarchical Task Network (HTN) methods in the context of integrated HTN planning and LLM-based chatbots. Methods indicate when and how to decompose tasks into subtasks. Our method learner is built on top of the ChatHTN planner. ChatHTN queries ChatGPT to generate a decomposition of a task into primitive tasks when no applicable method for the task is available. In this work, we extend ChatHTN. Namely, when ChatGPT generates a task decomposition, ChatHTN learns from it, akin to memoization. However, unlike memoization, it learns a generalized method that applies not only to the specific instance encountered, but to other instances of the same task. We conduct experiments on two domains and demonstrate that our online learning procedure reduces the number of calls to ChatGPT while solving at least as many problems, and in some cases, even more.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12869v1">On the Fundamental Limits of LLMs at Scale</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 Submitted to TMLR 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have benefited enormously from scaling, yet these gains are bounded by five fundamental limitations: (1) hallucination, (2) context compression, (3) reasoning degradation, (4) retrieval fragility, and (5) multimodal misalignment. While existing surveys describe these phenomena empirically, they lack a rigorous theoretical synthesis connecting them to the foundational limits of computation, information, and learning. This work closes that gap by presenting a unified, proof-informed framework that formalizes the innate theoretical ceilings of LLM scaling. First, computability and uncomputability imply an irreducible residue of error: for any computably enumerable model family, diagonalization guarantees inputs on which some model must fail, and undecidable queries (e.g., halting-style tasks) induce infinite failure sets for all computable predictors. Second, information-theoretic and statistical constraints bound attainable accuracy even on decidable tasks, finite description length enforces compression error, and long-tail factual knowledge requires prohibitive sample complexity. Third, geometric and computational effects compress long contexts far below their nominal size due to positional under-training, encoding attenuation, and softmax crowding. We further show how likelihood-based training favors pattern completion over inference, how retrieval under token limits suffers from semantic drift and coupling noise, and how multimodal scaling inherits shallow cross-modal alignment. Across sections, we pair theorems and empirical evidence to outline where scaling helps, where it saturates, and where it cannot progress, providing both theoretical foundations and practical mitigation paths like bounded-oracle retrieval, positional curricula, and sparse or hierarchical attention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12867v1">Bootstrapping LLMs via Preference-Based Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-17
    </div>
    <details class="paper-abstract">
      Bootstrapping large language models (LLMs) through preference-based policy optimization offers a promising direction for aligning model behavior with human preferences without relying on extensive manual annotations. In this work, we propose a novel preference-based policy optimization (PbPO) framework that formulates the learning process as a min-max game between the main policy and a reward model (RM). The RM is constrained within a confidence set derived from preference data to ensure reliable exploitation. Our iterative online algorithm actively collects preference data through guided exploration of the evolving policy, enabling continual self-improvement of both the policy and the RM. We provide theoretical guarantees for our method, establishing high-probability regret bounds for both settings with sequence-level RM and token-level RM, demonstrating its effectiveness in bootstrapping LLMs. Extensive experiments on five benchmarks show that our approach consistently outperforms existing state-of-the-art preference optimization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12860v1">Dissecting and Re-architecting 3D NAND Flash PIM Arrays for Efficient Single-Batch Token Generation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-17
      | 💬 This paper is accepted in the 43rd IEEE International Conference on Computer Design (ICCD), 2025
    </div>
    <details class="paper-abstract">
      The advancement of large language models has led to models with billions of parameters, significantly increasing memory and compute demands. Serving such models on conventional hardware is challenging due to limited DRAM capacity and high GPU costs. Thus, in this work, we propose offloading the single-batch token generation to a 3D NAND flash processing-in-memory (PIM) device, leveraging its high storage density to overcome the DRAM capacity wall. We explore 3D NAND flash configurations and present a re-architected PIM array with an H-tree network for optimal latency and cell density. Along with the well-chosen PIM array size, we develop operation tiling and mapping methods for LLM layers, achieving a 2.4x speedup over four RTX4090 with vLLM and comparable performance to four A100 with only 4.9% latency overhead. Our detailed area analysis reveals that the proposed 3D NAND flash PIM architecture can be integrated within a 4.98mm2 die area under the memory array, without extra area overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12823v1">Enhancing LLM Code Generation Capabilities through Test-Driven Development and Code Interpreter</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 AACL-IJCNLP 2025 Workshop BLP Shared Task 2, 6 pages, 7 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Over the past few years, improving LLM code generation capabilities has been a key focus in NLP research. Despite Bengali having 242 million native speakers worldwide, it receives little attention when it comes to training LLMs. More recently, various fine-tuning and augmented generation techniques have been employed to significantly enhance code generation performance. However, they require considerable expertise and resources to utilize effectively as an end user. The goal of our work is to democratize access to powerful code generation tools in resource-constrained emerging markets, enabling users to leverage them in their native language. We introduce a novel approach that combines Test-Driven Development (TDD) and Code Interpreter (CI), utilizing open-weight models, which improves the baseline accuracy for code generation with Bengali prompts and achieves an overall accuracy of 85%. Our approach requires no finetuning and proves that even the smallest models in the same family can attain up to 98% accuracy compared to the largest models. All of our results are publicly shared in GitHub for validation and reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12821v1">BioMedJImpact: A Comprehensive Dataset and LLM Pipeline for AI Engagement and Scientific Impact Analysis of Biomedical Journals</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Assessing journal impact is central to scholarly communication, yet existing open resources rarely capture how collaboration structures and artificial intelligence (AI) research jointly shape venue prestige in biomedicine. We present BioMedJImpact, a large-scale, biomedical-oriented dataset designed to advance journal-level analysis of scientific impact and AI engagement. Built from 1.74 million PubMed Central articles across 2,744 journals, BioMedJImpact integrates bibliometric indicators, collaboration features, and LLM-derived semantic indicators for AI engagement. Specifically, the AI engagement feature is extracted through a reproducible three-stage LLM pipeline that we propose. Using this dataset, we analyze how collaboration intensity and AI engagement jointly influence scientific impact across pre- and post-pandemic periods (2016-2019, 2020-2023). Two consistent trends emerge: journals with higher collaboration intensity, particularly those with larger and more diverse author teams, tend to achieve greater citation impact, and AI engagement has become an increasingly strong correlate of journal prestige, especially in quartile rankings. To further validate the three-stage LLM pipeline we proposed for deriving the AI engagement feature, we conduct human evaluation, confirming substantial agreement in AI relevance detection and consistent subfield classification. Together, these contributions demonstrate that BioMedJImpact serves as both a comprehensive dataset capturing the intersection of biomedicine and AI, and a validated methodological framework enabling scalable, content-aware scientometric analysis of scientific impact and innovation dynamics. Code is available at https://github.com/JonathanWry/BioMedJImpact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12817v1">Assessing Automated Fact-Checking for Medical LLM Responses with Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 Accepted as a conference paper at AAAI'26
    </div>
    <details class="paper-abstract">
      The recent proliferation of large language models (LLMs) holds the potential to revolutionize healthcare, with strong capabilities in diverse medical tasks. Yet, deploying LLMs in high-stakes healthcare settings requires rigorous verification and validation to understand any potential harm. This paper investigates the reliability and viability of using medical knowledge graphs (KGs) for the automated factuality evaluation of LLM-generated responses. To ground this investigation, we introduce FAITH, a framework designed to systematically probe the strengths and limitations of this KG-based approach. FAITH operates without reference answers by decomposing responses into atomic claims, linking them to a medical KG, and scoring them based on evidence paths. Experiments on diverse medical tasks with human subjective evaluations demonstrate that KG-grounded evaluation achieves considerably higher correlations with clinician judgments and can effectively distinguish LLMs with varying capabilities. It is also robust to textual variances. The inherent explainability of its scoring can further help users understand and mitigate the limitations of current LLMs. We conclude that while limitations exist, leveraging KGs is a prominent direction for automated factuality assessment in healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06838v3">P3-LLM: An Integrated NPU-PIM Accelerator for LLM Inference Using Hybrid Numerical Formats</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      The substantial memory bandwidth and computational demands of large language models (LLMs) present critical challenges for efficient inference. To tackle this, the literature has explored heterogeneous systems that combine neural processing units (NPUs) with DRAM-based processing-in-memory (PIM) for LLM acceleration. However, existing high-precision (e.g., FP16) PIM compute units incur significant area and power overhead in DRAM technology, limiting the effective computation throughput. In this paper, we introduce P3-LLM, a novel NPU-PIM integrated accelerator for LLM inference using hybrid numerical formats. Our approach is threefold: First, we propose a flexible mixed-precision quantization scheme, which leverages hybrid numerical formats to quantize different LLM operands with high compression efficiency and minimal accuracy loss. Second, we architect an efficient PIM accelerator for P3-LLM, featuring enhanced compute units to support hybrid numerical formats. Our careful choice of numerical formats allows to co-design low-precision PIM compute units that significantly boost the computation throughput under iso-area constraints. Third, we optimize the low-precision dataflow of different LLM modules by applying operator fusion to minimize the overhead of runtime dequantization. Evaluation on a diverse set of representative LLMs and tasks demonstrates that P3-LLM achieves state-of-the-art accuracy in terms of both KV-cache quantization and weight-activation quantization. Combining the proposed quantization scheme with PIM architecture co-design, P3-LLM yields an average of $4.9\times$, $2.0\times$, and $3.4\times$ speedups over the state-of-the-art LLM accelerators HBM-PIM, Ecco, and Pimba, respectively. Our quantization code is available at https://github.com/yc2367/P3-LLM.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16785v2">Interpreting the Effects of Quantization on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 Accepted to AACL 2025 Main
    </div>
    <details class="paper-abstract">
      Quantization offers a practical solution to deploy LLMs in resource-constraint environments. However, its impact on internal representations remains understudied, raising questions about the reliability of quantized models. In this study, we employ a range of interpretability techniques to investigate how quantization affects model and neuron behavior. We analyze multiple LLMs under 4-bit and 8-bit quantization. Our findings reveal that the impact of quantization on model calibration is generally minor. Analysis of neuron activations indicates that the number of dead neurons, i.e., those with activation values close to 0 across the dataset, remains consistent regardless of quantization. In terms of neuron contribution to predictions, we observe that smaller full precision models exhibit fewer salient neurons, whereas larger models tend to have more, with the exception of Llama-2-7B. The effect of quantization on neuron redundancy varies across models. Overall, our findings suggest that effect of quantization may vary by model and tasks, however, we did not observe any drastic change which may discourage the use of quantization as a reliable model compression technique.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12782v1">LLM Reinforcement in Context</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      Current Large Language Model alignment research mostly focuses on improving model robustness against adversarial attacks and misbehavior by training on examples and prompting. Research has shown that LLM jailbreak probability increases with the size of the user input or conversation length. There is a lack of appropriate research into means of strengthening alignment which also scale with user input length. We propose interruptions as a possible solution to this problem. Interruptions are control sentences added to the user input approximately every x tokens for some arbitrary x. We suggest that this can be generalized to the Chain-of-Thought process to prevent scheming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.00833v2">HumanoidGen: Data Generation for Bimanual Dexterous Manipulation via LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 Project Page: https://openhumanoidgen.github.io
    </div>
    <details class="paper-abstract">
      For robotic manipulation, existing robotics datasets and simulation benchmarks predominantly cater to robot-arm platforms. However, for humanoid robots equipped with dual arms and dexterous hands, simulation tasks and high-quality demonstrations are notably lacking. Bimanual dexterous manipulation is inherently more complex, as it requires coordinated arm movements and hand operations, making autonomous data collection challenging. This paper presents HumanoidGen, an automated task creation and demonstration collection framework that leverages atomic dexterous operations and LLM reasoning to generate relational constraints. Specifically, we provide spatial annotations for both assets and dexterous hands based on the atomic operations, and perform an LLM planner to generate a chain of actionable spatial constraints for arm movements based on object affordances and scenes. To further improve planning ability, we employ a variant of Monte Carlo tree search to enhance LLM reasoning for long-horizon tasks and insufficient annotation. In experiments, we create a novel benchmark with augmented scenarios to evaluate the quality of the collected data. The results show that the performance of the 2D and 3D diffusion policies can scale with the generated dataset. Project page is https://openhumanoidgen.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12751v1">Are LLMs The Way Forward? A Case Study on LLM-Guided Reinforcement Learning for Decentralized Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Autonomous vehicle navigation in complex environments such as dense and fast-moving highways and merging scenarios remains an active area of research. A key limitation of RL is its reliance on well-specified reward functions, which often fail to capture the full semantic and social complexity of diverse, out-of-distribution situations. As a result, a rapidly growing line of research explores using Large Language Models (LLMs) to replace or supplement RL for direct planning and control, on account of their ability to reason about rich semantic context. However, LLMs present significant drawbacks: they can be unstable in zero-shot safety-critical settings, produce inconsistent outputs, and often depend on expensive API calls with network latency. This motivates our investigation into whether small, locally deployed LLMs (< 14B parameters) can meaningfully support autonomous highway driving through reward shaping rather than direct control. We present a case study comparing RL-only, LLM-only, and hybrid approaches, where LLMs augment RL rewards by scoring state-action transitions during training, while standard RL policies execute at test time. Our findings reveal that RL-only agents achieve moderate success rates (73-89%) with reasonable efficiency, LLM-only agents can reach higher success rates (up to 94%) but with severely degraded speed performance, and hybrid approaches consistently fall between these extremes. Critically, despite explicit efficiency instructions, LLM-influenced approaches exhibit systematic conservative bias with substantial model-dependent variability, highlighting important limitations of current small LLMs for safety-critical control tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12728v1">On the Brittleness of LLMs: A Journey around Set Membership</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve superhuman performance on complex reasoning tasks, yet often fail on much simpler problems, raising concerns about their reliability and interpretability. We investigate this paradox through a focused study with two key design features: simplicity, to expose basic failure modes, and scale, to enable comprehensive controlled experiments. We focus on set membership queries -- among the most fundamental forms of reasoning -- using tasks like ``Is apple an element of the set \{pear, plum, apple, raspberry\}?''. We conduct a systematic empirical evaluation across prompt phrasing, semantic structure, element ordering, and model choice. Our large-scale analysis reveals that LLM performance on this elementary task is consistently brittle, and unpredictable across all dimensions, suggesting that the models' ``understanding'' of the set concept is fragmented and convoluted at best. Our work demonstrates that the large-scale experiments enabled by the simplicity of the problem allow us to map and analyze the failure modes comprehensively, making this approach a valuable methodology for LLM evaluation in general.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.15859v2">InfiMed-ORBIT: Aligning LLMs on Open-Ended Complex Tasks via Rubric-Based Incremental Training</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Reinforcement learning has powered many of the recent breakthroughs in large language models, especially for tasks where rewards can be computed automatically, such as code generation. However, these methods deteriorate in open-ended domains like medical consultation, where feedback is inherently ambiguous, highly context-dependent, and cannot be reduced to a reliable scalar signal. In such settings, RL must either rely on supervision-intensive reward models that often fail to generalize, or it falls into pathological behaviors such as reward hacking - an especially troubling risk for high-stakes medical dialogue. To address these limitations, we introduce ORBIT, an open-ended rubric-based incremental training framework for high-stakes medical dialogue. ORBIT integrates synthetic dialogue generation with dynamically constructed rubrics that serve as adaptive guides for incremental RL. Instead of relying on external medical knowledge bases or handcrafted rule sets, ORBIT uses rubric-driven feedback to steer the learning process. Its judge component can be instantiated with general-purpose instruction-following LLMs, removing the need for any task-specific fine-tuning. Applied to the Qwen3-4B-Instruct model, ORBIT raises the HealthBench-Hard score from 7.0 to 27.5 using only 2k training samples, achieving SOTA performance for models at this scale. With larger rubric datasets, ORBIT-trained models further compete with the strongest open-source baselines on HealthBench-Hard. Our analysis shows that rubric-guided RL consistently improves consultation quality across diverse medical scenarios. We also apply such rubric generation and training pipeline to InfoBench, where ORBIT enhances instruction-following performance, highlighting the generality of rubric-based feedback.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12710v1">Evolve the Method, Not the Prompts: Evolutionary Synthesis of Jailbreak Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Automated red teaming frameworks for Large Language Models (LLMs) have become increasingly sophisticated, yet they share a fundamental limitation: their jailbreak logic is confined to selecting, combining, or refining pre-existing attack strategies. This binds their creativity and leaves them unable to autonomously invent entirely new attack mechanisms. To overcome this gap, we introduce \textbf{EvoSynth}, an autonomous framework that shifts the paradigm from attack planning to the evolutionary synthesis of jailbreak methods. Instead of refining prompts, EvoSynth employs a multi-agent system to autonomously engineer, evolve, and execute novel, code-based attack algorithms. Crucially, it features a code-level self-correction loop, allowing it to iteratively rewrite its own attack logic in response to failure. Through extensive experiments, we demonstrate that EvoSynth not only establishes a new state-of-the-art by achieving an 85.5\% Attack Success Rate (ASR) against highly robust models like Claude-Sonnet-4.5, but also generates attacks that are significantly more diverse than those from existing methods. We release our framework to facilitate future research in this new direction of evolutionary synthesis of jailbreak methods. Code is available at: https://github.com/dongdongunique/EvoSynth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02503v2">OptiHive: Ensemble Selection for LLM-Based Optimization via Statistical Modeling</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      LLM-based solvers have emerged as a promising means of automating problem modeling and solving. However, they remain unreliable and often depend on iterative repair loops that result in significant latency. We introduce OptiHive, a framework that enhances any solver-generation pipeline to produce higher-quality solvers from natural-language descriptions of optimization problems. OptiHive uses a single batched generation to produce diverse components (solvers, problem instances, and validation tests) and filters out erroneous components to ensure fully interpretable outputs. Accounting for the imperfection of the generated components, we employ a statistical model to infer their true performance, enabling principled uncertainty quantification and solver selection. On tasks ranging from traditional optimization problems to challenging variants of the Multi-Depot Vehicle Routing Problem, OptiHive significantly outperforms baselines, increasing the optimality rate from 5% to 92% on the most complex problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03369v2">Silenced Biases: The Dark Side LLMs Learned to Refuse</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 Accepted to The 40th Annual AAAI Conference on Artificial Intelligence - AI Alignment Track (Oral)
    </div>
    <details class="paper-abstract">
      Safety-aligned large language models (LLMs) are becoming increasingly widespread, especially in sensitive applications where fairness is essential and biased outputs can cause significant harm. However, evaluating the fairness of models is a complex challenge, and approaches that do so typically utilize standard question-answer (QA) styled schemes. Such methods often overlook deeper issues by interpreting the model's refusal responses as positive fairness measurements, which creates a false sense of fairness. In this work, we introduce the concept of silenced biases, which are unfair preferences encoded within models' latent space and are effectively concealed by safety-alignment. Previous approaches that considered similar indirect biases often relied on prompt manipulation or handcrafted implicit queries, which present limited scalability and risk contaminating the evaluation process with additional biases. We propose the Silenced Bias Benchmark (SBB), which aims to uncover these biases by employing activation steering to reduce model refusals during QA. SBB supports easy expansion to new demographic groups and subjects, presenting a fairness evaluation framework that encourages the future development of fair models and tools beyond the masking effects of alignment training. We demonstrate our approach over multiple LLMs, where our findings expose an alarming distinction between models' direct responses and their underlying fairness issues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.12689v2">From Delegates to Trustees: How Optimizing for Long-Term Interests Shapes Bias and Alignment in LLM</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promising accuracy in predicting survey responses and policy preferences, which has increased interest in their potential to represent human interests in various domains. Most existing research has focused on "behavioral cloning", effectively evaluating how well models reproduce individuals' expressed preferences. Drawing on theories of political representation, we highlight an underexplored design trade-off: whether AI systems should act as delegates, mirroring expressed preferences, or as trustees, exercising judgment about what best serves an individual's interests. This trade-off is closely related to issues of LLM sycophancy, where models can encourage behavior or validate beliefs that may be aligned with a user's short-term preferences, but is detrimental to their long-term interests. Through a series of experiments simulating votes on various policy issues in the U.S. context, we apply a temporal utility framework that weighs short and long-term interests (simulating a trustee role) and compare voting outcomes to behavior-cloning models (simulating a delegate). We find that trustee-style predictions weighted toward long-term interests produce policy decisions that align more closely with expert consensus on well-understood issues, but also show greater bias toward models' default stances on topics lacking clear agreement. These findings reveal a fundamental trade-off in designing AI systems to represent human interests. Delegate models better preserve user autonomy but may diverge from well-supported policy positions, while trustee models can promote welfare on well-understood issues yet risk paternalism and bias on subjective topics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12661v1">Reason-KE++: Aligning the Process, Not Just the Outcome, for Faithful LLM Knowledge Editing</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Aligning Large Language Models (LLMs) to be faithful to new knowledge in complex, multi-hop reasoning tasks is a critical, yet unsolved, challenge. We find that SFT-based methods, e.g., Reason-KE, while state-of-the-art, suffer from a "faithfulness gap": they optimize for format mimicry rather than sound reasoning. This gap enables the LLM's powerful parametric priors to override new contextual facts, resulting in critical factual hallucinations (e.g., incorrectly reasoning "Houston" from "NASA" despite an explicit edit). To solve this core LLM alignment problem, we propose Reason-KE++, an SFT+RL framework that instills process-level faithfulness. Its core is a Stage-aware Reward mechanism that provides dense supervision for intermediate reasoning steps (e.g., Decomposition, Sub-answer Correctness). Crucially, we identify that naive outcome-only RL is a deceptive trap for LLM alignment: it collapses reasoning integrity (e.g., 19.00% Hop acc) while superficially boosting final accuracy. Our process-aware framework sets a new SOTA of 95.48% on MQUAKE-CF-3k (+5.28%), demonstrating that for complex tasks, aligning the reasoning process is essential for building trustworthy LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09539v3">TFRank: Think-Free Reasoning Enables Practical Pointwise LLM Ranking</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Reasoning-intensive ranking models built on Large Language Models (LLMs) have made notable progress. However, existing approaches often rely on large-scale LLMs and explicit Chain-of-Thought (CoT) reasoning, resulting in high computational cost and latency that limit real-world use. To address this, we propose \textbf{TFRank}, an efficient pointwise reasoning ranker based on small-scale LLMs. To improve ranking performance, TFRank effectively integrates CoT data, fine-grained score supervision, and multi-task training. Furthermore, it achieves an efficient ``\textbf{T}hink-\textbf{F}ree" reasoning capability by employing a ``think-mode switch'' and pointwise format constraints. Specifically, this allows the model to leverage explicit reasoning during training while delivering precise relevance scores for complex queries at inference without generating any reasoning chains. Experiments show that TFRank achieves performance comparable to models with four times more parameters on the BRIGHT benchmark and demonstrates strong competitiveness on the BEIR benchmark. Further analysis shows that TFRank achieves an effective balance between performance and efficiency, providing a practical solution for integrating advanced reasoning into real-world systems. Our code and data are released in the repository: https://github.com/JOHNNY-fans/TFRank.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.03661v3">Automated Algorithmic Discovery for Scientific Computing through LLM-Guided Evolutionary Search: A Case Study in Gravitational-Wave Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 76 pages (28 main), with 6+6 figures and 2 tables, substantially revised with improved structure and clarity
    </div>
    <details class="paper-abstract">
      Automated algorithm discovery in scientific computing faces fundamental challenges: vast design spaces with expensive evaluations, domain-specific physical constraints requiring expert knowledge, and the necessity for interpretable solutions that scientists can validate and understand. We present the Evo-MCTS (Evolutionary Monte Carlo Tree Search) framework, integrating large language models (LLMs) with tree-structured evolutionary search for interpretable algorithm discovery. Evo-MCTS combines reflective code synthesis leveraging LLM domain knowledge, multi-scale evolutionary operations on structured code representations, and interpretable algorithmic pathways emerging from tree-guided exploration. When applied to gravitational wave detection-a challenging domain with continuous parameter spaces and strict physical constraints-Evo-MCTS achieves 20.2% improvement over domain-specific methods and 59.1% over LLM-based optimization frameworks. This improvement arises from its ability to consistently converge toward interpretable algorithmic structures that integrate multiple functional components. Our domain-agnostic architecture establishes a generalizable methodology for automated algorithm discovery in scientific computing, where algorithmic transparency and physical validity are as essential as performance optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.09724v2">UDA: Unsupervised Debiasing Alignment for Pair-wise LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Pairwise evaluation of Large Language Models (LLMs) is a common paradigm, but it is prone to preference bias, where judges systematically favor certain outputs, such as their own. This bias leads to inconsistent and skewed rankings across different judges. To address this, we first empirically demonstrate significant and heterogeneous biases in cross-model evaluations. We then propose UDA (Unsupervised Debiasing Alignment), a framework that reduces inter-judge disagreement by dynamically adjusting the Elo rating system. For each pairwise comparison, a compact neural network learns to adaptively set the K-factor and refine win probabilities. Crucially, UDA operates in a fully unsupervised manner, guided solely by the objective of minimizing the dispersion among the Elo trajectories of all judges. This forces an alignment towards a collective consensus, which serves as an unsupervised proxy for a more stable and reproducible evaluation. In addition, we provide theoretical motivation demonstrating how alignment towards a consensus can reduce aggregate system bias. Experiments show that UDA significantly reduces the inter-judge rating standard deviation by up to 63.4% and improves the average correlation with human judgments by 24.7%. Notably, UDA elevates the performance of poorly performing judges to achieve parity with high-quality ones, fostering a more robust and reliable evaluation ecosystem. Code and data are available at https://anonymous.4open.science/r/62AB93CD-23B4.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.01623v3">Fira: Can We Achieve Full-rank Training of LLMs Under Low-rank Constraint?</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 NeurIPS 2025, Project page: https://github.com/xichen-fy/Fira
    </div>
    <details class="paper-abstract">
      Low-rank training has emerged as a promising approach for reducing memory usage in training Large Language Models (LLMs). Previous methods either rely on decomposing weight matrices (e.g., LoRA), or seek to decompose gradient matrices (e.g., GaLore) to ensure reduced memory consumption. However, both of them constrain the training in a low-rank subspace, thus inevitably leading to sub-optimal performance. This raises a question: whether it is possible to consistently preserve the low-rank constraint for memory efficiency, while achieving full-rank training (i.e., training with full-rank gradients of full-rank weights) to avoid inferior outcomes? In this paper, we propose a new plug-and-play training framework for LLMs called Fira, as the first attempt to achieve this goal. First, we observe an interesting phenomenon during LLM training: the scaling impact of adaptive optimizers (e.g., Adam) on the gradient norm remains similar from low-rank to full-rank training. Based on this observation, we propose a norm-based scaling method, which utilizes the scaling impact of low-rank optimizers as substitutes for that of original full-rank optimizers to enable full-rank training. In this way, we can preserve the low-rank constraint in the optimizer while achieving full-rank training for better performance. Moreover, we find that there are sudden gradient rises during the optimization process, potentially causing loss spikes. To address this, we further put forward a norm-growth limiter to smooth the gradient via regulating the relative increase of gradient norms. Extensive experiments on the pre-training and fine-tuning of LLMs show that Fira outperforms both LoRA and GaLore, achieving performance that is comparable to or even better than full-rank training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.09443v2">LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse tasks, driving the development and widespread adoption of LLM-as-a-Judge systems for automated evaluation, including red teaming and benchmarking. However, these systems are susceptible to adversarial attacks that can manipulate evaluation outcomes, raising critical concerns about their robustness and trustworthiness. Existing evaluation methods for LLM-based judges are often fragmented and lack a unified framework for comprehensive robustness assessment. Furthermore, the impact of prompt template design and model selection on judge robustness has rarely been explored, and their performance in real-world deployments remains largely unverified. To address these gaps, we introduce RobustJudge, a fully automated and scalable framework designed to systematically evaluate the robustness of LLM-as-a-Judge systems. Specifically, RobustJudge investigates the effectiveness of 15 attack methods and 7 defense strategies across 12 models (RQ1), examines the impact of prompt template design and model selection (RQ2), and evaluates the security of real-world deployments (RQ3). Our study yields three key findings: (1) LLM-as-a-Judge systems are highly vulnerable to attacks such as PAIR and combined attacks, while defense mechanisms such as re-tokenization and LLM-based detectors can provide enhanced protection; (2) robustness varies substantially across prompt templates (up to 40%); (3) deploying RobustJudge on Alibaba's PAI platform uncovers previously undiscovered vulnerabilities. These results offer practical insights for building trustworthy LLM-as-a-Judge systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05925v2">Small Models, Big Support: A Local LLM Framework for Educator-Centric Content Creation and Assessment with RAG and CAG</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      While Large Language Models (LLMs) are increasingly applied in student-facing educational tools, their potential to directly support educators through locally deployable and customizable solutions remains underexplored. Many existing approaches rely on proprietary, cloud-based systems that raise significant cost, privacy, and control concerns for educational institutions. To address these barriers, we introduce an end-to-end, open-source framework that empowers educators using small (3B-7B parameter), locally deployable LLMs. Our system is designed for comprehensive teacher support, including customized teaching material generation and AI-assisted assessment. The framework synergistically combines Retrieval-Augmented Generation (RAG) and Context-Augmented Generation (CAG) to produce factually accurate, pedagogically-styled content. A core feature is an interactive refinement loop, a teacher-in-the-loop mechanism that ensures educator agency and precise alignment of the final output. To enhance reliability and safety, an auxiliary verifier LLM inspects all generated content. We validate our framework through a rigorous evaluation of its content generation capabilities and report on a successful technical deployment in a college physics course, which confirms its feasibility on standard institutional hardware. Our findings demonstrate that carefully engineered, self-hosted systems built on small LLMs can provide robust, affordable, and private support for educators, achieving practical utility comparable to much larger models for targeted instructional tasks. This work presents a practical blueprint for the development of sovereign AI tools tailored to the real-world needs of educational institutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.03945v3">Function-based Labels for Complementary Recommendation: Definition, Annotation, and LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Complementary recommendations enhance the user experience by suggesting items that are frequently purchased together while serving different functions from the query item. Inferring or evaluating whether two items have a complementary relationship requires complementary relationship labels; however, defining these labels is challenging because of the inherent ambiguity of such relationships. Complementary labels based on user historical behavior logs attempt to capture these relationships, but often produce inconsistent and unreliable results. Recent efforts have introduced large language models (LLMs) to infer these relationships. However, these approaches provide a binary classification without a nuanced understanding of complementary relationships. In this study, we address these challenges by introducing Function-Based Labels (FBLs), a novel definition of complementary relationships independent of user purchase logs and the opaque decision processes of LLMs. We constructed a human-annotated FBLs dataset comprising 2,759 item pairs and demonstrated that it covered possible item relationships and minimized ambiguity. We then evaluated whether some machine learning (ML) methods using annotated FBLs could accurately infer labels for unseen item pairs, and whether LLM-generated complementary labels align with human perception. Our results demonstrate that even with limited data, ML models, such as logistic regression and SVM achieve high macro-F1 scores (approximately 0.82). Furthermore, LLMs, such as gpt-4o-mini, demonstrated high consistency (0.989) and classification accuracy (0.849) under the detailed definition of FBLs, indicating their potential as effective annotators that mimic human judgment. Overall, our study presents FBLs as a clear definition of complementary relationships, enabling more accurate inferences and automated labeling of complementary recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12485v1">ARCHE: A Novel Task to Evaluate LLMs on Latent Reasoning Chain Extraction</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 Accepted to AAAI 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in scientific domains. While they can produce reasoning-like content via methods such as chain-of-thought prompting, these outputs are typically unstructured and informal, obscuring whether models truly understand the fundamental reasoning paradigms that underpin scientific inference. To address this, we introduce a novel task named Latent Reasoning Chain Extraction (ARCHE), in which models must decompose complex reasoning arguments into combinations of standard reasoning paradigms in the form of a Reasoning Logic Tree (RLT). In RLT, all reasoning steps are explicitly categorized as one of three variants of Peirce's fundamental inference modes: deduction, induction, or abduction. To facilitate this task, we release ARCHE Bench, a new benchmark derived from 70 Nature Communications articles, including more than 1,900 references and 38,000 viewpoints. We propose two logic-aware evaluation metrics: Entity Coverage (EC) for content completeness and Reasoning Edge Accuracy (REA) for step-by-step logical validity. Evaluations on 10 leading LLMs on ARCHE Bench reveal that models exhibit a trade-off between REA and EC, and none are yet able to extract a complete and standard reasoning chain. These findings highlight a substantial gap between the abilities of current reasoning models and the rigor required for scientific argumentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12484v1">One Request, Multiple Experts: LLM Orchestrates Domain Specific Models via Adaptive Task Routing</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      With the integration of massive distributed energy resources and the widespread participation of novel market entities, the operation of active distribution networks (ADNs) is progressively evolving into a complex multi-scenario, multi-objective problem. Although expert engineers have developed numerous domain specific models (DSMs) to address distinct technical problems, mastering, integrating, and orchestrating these heterogeneous DSMs still entail considerable overhead for ADN operators. Therefore, an intelligent approach is urgently required to unify these DSMs and enable efficient coordination. To address this challenge, this paper proposes the ADN-Agent architecture, which leverages a general large language model (LLM) to coordinate multiple DSMs, enabling adaptive intent recognition, task decomposition, and DSM invocation. Within the ADN-Agent, we design a novel communication mechanism that provides a unified and flexible interface for diverse heterogeneous DSMs. Finally, for some language-intensive subtasks, we propose an automated training pipeline for fine-tuning small language models, thereby effectively enhancing the overall problem-solving capability of the system. Comprehensive comparisons and ablation experiments validate the efficacy of the proposed method and demonstrate that the ADN-Agent architecture outperforms existing LLM application paradigms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12474v1">Co-Layout: LLM-driven Co-optimization for Interior Layout</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      We present a novel framework for automated interior design that combines large language models (LLMs) with grid-based integer programming to jointly optimize room layout and furniture placement. Given a textual prompt, the LLM-driven agent workflow extracts structured design constraints related to room configurations and furniture arrangements. These constraints are encoded into a unified grid-based representation inspired by ``Modulor". Our formulation accounts for key design requirements, including corridor connectivity, room accessibility, spatial exclusivity, and user-specified preferences. To improve computational efficiency, we adopt a coarse-to-fine optimization strategy that begins with a low-resolution grid to solve a simplified problem and guides the solution at the full resolution. Experimental results across diverse scenarios demonstrate that our joint optimization approach significantly outperforms existing two-stage design pipelines in solution quality, and achieves notable computational efficiency through the coarse-to-fine strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12472v1">Assessing LLMs for Serendipity Discovery in Knowledge Graphs: A Case for Drug Repurposing</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 The 40th AAAI Conference on Artificial Intelligence (AAAI-26)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have greatly advanced knowledge graph question answering (KGQA), yet existing systems are typically optimized for returning highly relevant but predictable answers. A missing yet desired capacity is to exploit LLMs to suggest surprise and novel ("serendipitious") answers. In this paper, we formally define the serendipity-aware KGQA task and propose the SerenQA framework to evaluate LLMs' ability to uncover unexpected insights in scientific KGQA tasks. SerenQA includes a rigorous serendipity metric based on relevance, novelty, and surprise, along with an expert-annotated benchmark derived from the Clinical Knowledge Graph, focused on drug repurposing. Additionally, it features a structured evaluation pipeline encompassing three subtasks: knowledge retrieval, subgraph reasoning, and serendipity exploration. Our experiments reveal that while state-of-the-art LLMs perform well on retrieval, they still struggle to identify genuinely surprising and valuable discoveries, underscoring a significant room for future improvements. Our curated resources and extended version are released at: https://cwru-db-group.github.io/serenQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12448v1">SeedAIchemy: LLM-Driven Seed Corpus Generation for Fuzzing</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      We introduce SeedAIchemy, an automated LLM-driven corpus generation tool that makes it easier for developers to implement fuzzing effectively. SeedAIchemy consists of five modules which implement different approaches at collecting publicly available files from the internet. Four of the five modules use large language model (LLM) workflows to construct search terms designed to maximize corpus quality. Corpora generated by SeedAIchemy perform significantly better than a naive corpus and similarly to a manually-curated corpus on a diverse range of target programs and libraries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.24021v2">SelecTKD: Selective Token-Weighted Knowledge Distillation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Knowledge distillation (KD) is a standard route to compress Large Language Models (LLMs) into compact students, yet most pipelines uniformly apply token-wise loss regardless of teacher confidence. This indiscriminate supervision amplifies noisy, high-entropy signals and is especially harmful under large teacher-student capacity gaps. We introduce SelecTKD, a plug-and-play Selective Token-Weighted distillation framework that shifts the focus from "how to measure divergence" to "where to apply learning". At each step, the student proposes tokens that are verified by the teacher through a robust propose-and-verify procedure with two variants: greedy Top-k and non-greedy Spec-k. Accepted tokens receive full loss, while rejected tokens are masked or down-weighted. This objective-agnostic design works with on- and off-policy data, induces an implicit curriculum quantified by Token Acceptance Rate (TAR), and stabilizes optimization. Across instruction following, mathematical reasoning, code generation, and a VLM setting, SelecTKD consistently improves strong baselines and achieves state-of-the-art results for small models without architectural changes or extra reference models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12423v1">GRAPHTEXTACK: A Realistic Black-Box Node Injection Attack on LLM-Enhanced GNNs</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 AAAI 2026
    </div>
    <details class="paper-abstract">
      Text-attributed graphs (TAGs), which combine structural and textual node information, are ubiquitous across many domains. Recent work integrates Large Language Models (LLMs) with Graph Neural Networks (GNNs) to jointly model semantics and structure, resulting in more general and expressive models that achieve state-of-the-art performance on TAG benchmarks. However, this integration introduces dual vulnerabilities: GNNs are sensitive to structural perturbations, while LLM-derived features are vulnerable to prompt injection and adversarial phrasing. While existing adversarial attacks largely perturb structure or text independently, we find that uni-modal attacks cause only modest degradation in LLM-enhanced GNNs. Moreover, many existing attacks assume unrealistic capabilities, such as white-box access or direct modification of graph data. To address these gaps, we propose GRAPHTEXTACK, the first black-box, multi-modal{, poisoning} node injection attack for LLM-enhanced GNNs. GRAPHTEXTACK injects nodes with carefully crafted structure and semantics to degrade model performance, operating under a realistic threat model without relying on model internals or surrogate models. To navigate the combinatorial, non-differentiable search space of connectivity and feature assignments, GRAPHTEXTACK introduces a novel evolutionary optimization framework with a multi-objective fitness function that balances local prediction disruption and global graph influence. Extensive experiments on five datasets and two state-of-the-art LLM-enhanced GNN models show that GRAPHTEXTACK significantly outperforms 12 strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.19850v2">FALCONEye: Finding Answers and Localizing Content in ONE-hour-long videos with multi-modal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      Finding information in hour-long videos is a challenging task even for top-performing Vision Language Models (VLMs), as encoding visual content quickly exceeds available context windows. To tackle this challenge, we present FALCONEye, a novel video agent based on a training-free, model-agnostic meta-architecture composed of a VLM and a Large Language Model (LLM). FALCONEye answers open-ended questions using an exploration-based search algorithm guided by calibrated confidence from the VLM's answers. We also introduce the FALCON-Bench benchmark, extending Question Answering problem to Video Answer Search-requiring models to return both the answer and its supporting temporal window for open-ended questions in hour-long videos. With just a 7B VLM and a lightweight LLM, FALCONEye outscores all open-source 7B VLMs and comparable agents in FALCON-Bench. It further demonstrates its generalization capability in MLVU benchmark with shorter videos and different tasks, surpassing GPT-4o on single-detail tasks while slashing inference cost by roughly an order of magnitude.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04245v3">Contextual Integrity in LLMs via Reasoning and Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-16
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      As the era of autonomous agents making decisions on behalf of users unfolds, ensuring contextual integrity (CI) -- what is the appropriate information to share while carrying out a certain task -- becomes a central question to the field. We posit that CI demands a form of reasoning where the agent needs to reason about the context in which it is operating. To test this, we first prompt LLMs to reason explicitly about CI when deciding what information to disclose. We then extend this approach by developing a reinforcement learning (RL) framework that further instills in models the reasoning necessary to achieve CI. Using a synthetic, automatically created, dataset of only $\sim700$ examples but with diverse contexts and information disclosure norms, we show that our method substantially reduces inappropriate information disclosure while maintaining task performance across multiple model sizes and families. Importantly, improvements transfer from this synthetic dataset to established CI benchmarks such as PrivacyLens that has human annotations and evaluates privacy leakage of AI assistants in actions and tool calls.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.12982v2">Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Robust Response Generation in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-11-16
    </div>
    <details class="paper-abstract">
      The proliferation of large language models (LLMs) has significantly advanced intelligent systems. Unfortunately, LLMs often face knowledge conflicts between internal memory and retrieved external information, arising from misinformation, biases, or outdated knowledge. These conflicts undermine response reliability and introduce uncertainty in decision-making. In this work, we analyze how LLMs navigate knowledge conflicts from an information-theoretic perspective and reveal that when conflicting and supplementary information exhibit significant differences, LLMs confidently resolve their preferences and alleviate the uncertainty during their response generation. When this difference is ambiguous, LLMs experience considerable uncertainty about their generation. Based on this insight, we propose Swin-VIB, a novel framework that integrates a pipeline of variational information bottleneck models to adapt the retrieved information difference, facilitating robust response generation of LLMs even in conflicting contexts. Extensive experiments confirm our theoretical analysis and demonstrate the performance of Swin-VIB. Notably, Swin-VIB outperforms all competitive baselines in terms of the accuracy of the multiple-choice task, while improving the EM values in the open-ended QA task by at least 11.14%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12376v1">BitSnap: Checkpoint Sparsification and Quantization in LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 12 pages, numerous figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in size and complexity, efficient checkpoint saving\&loading has become crucial for managing storage, memory usage, and fault tolerance in LLM training. The current works do not comprehensively take into account the optimization of these several aspects. This paper proposes a novel checkpoint sparsification and quantization method that adapts dynamically to different training stages and model architectures. We present a comprehensive analysis of existing lossy and lossless compression techniques, identify current limitations, and introduce our adaptive approach that balances compression ratio, speed, and precision impact throughout the training process. Experiments on different sizes of LLMs demonstrate that our bitmask-based sparsification method achieves 16x compression ratio without compromising model accuracy. Additionally, the cluster-based quantization method achieves 2x compression ratio with little precision loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.00730v4">Teaching LLMs to See and Guide: Context-Aware Real-Time Assistance in Augmented Reality</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 This work has been submitted to the IEEE Transactions on Systems, Man, and Cybernetics: Systems for possible publication
    </div>
    <details class="paper-abstract">
      The growing adoption of augmented and virtual reality (AR and VR) technologies in industrial training and on-the-job assistance has created new opportunities for intelligent, context-aware support systems. As workers perform complex tasks guided by AR and VR, these devices capture rich streams of multimodal data, including gaze, hand actions, and task progression, that can reveal user intent and task state in real time. Leveraging this information effectively remains a major challenge. In this work, we present a context-aware large language model (LLM) assistant that integrates diverse data modalities, such as hand actions, task steps, and dialogue history, into a unified framework for real-time question answering. To systematically study how context influences performance, we introduce an incremental prompting framework, where each model version receives progressively richer contextual inputs. Using the HoloAssist dataset, which records AR-guided task executions, we evaluate how each modality contributes to the assistant's effectiveness. Our experiments show that incorporating multimodal context significantly improves the accuracy and relevance of responses. These findings highlight the potential of LLM-driven multimodal integration to enable adaptive, intuitive assistance for AR and VR-based industrial training and assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.04688v2">Evaluating LLMs' Reasoning Over Ordered Procedural Steps</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Accepted to IJCNLP-AACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Reasoning over procedural sequences, where the order of steps directly impacts outcomes, is a critical capability for large language models (LLMs). In this work, we study the task of reconstructing globally ordered sequences from shuffled procedural steps, using a curated dataset of food recipes, a domain where correct sequencing is essential for task success. We evaluate several LLMs under zero-shot and few-shot settings and present a comprehensive evaluation framework that adapts established metrics from ranking and sequence alignment. These include Kendall's Tau, Normalized Longest Common Subsequence (NLCS), and Normalized Edit Distance (NED), which capture complementary aspects of ordering quality. Our analysis shows that model performance declines with increasing sequence length, reflecting the added complexity of longer procedures. We also find that greater step displacement in the input, corresponding to more severe shuffling, leads to further degradation. These findings highlight the limitations of current LLMs in procedural reasoning, especially with longer and more disordered inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19172v2">When Facts Change: Probing LLMs on Evolving Knowledge with evolveQA</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Under submission
    </div>
    <details class="paper-abstract">
      LLMs often fail to handle temporal knowledge conflicts--contradictions arising when facts evolve over time within their training data. Existing studies evaluate this phenomenon through benchmarks built on structured knowledge bases like Wikidata, but they focus on widely-covered, easily-memorized popular entities and lack the dynamic structure needed to fairly evaluate LLMs with different knowledge cut-off dates. We introduce evolveQA, a benchmark specifically designed to evaluate LLMs on temporally evolving knowledge, constructed from 3 real-world, time-stamped corpora: AWS updates, Azure changes, and WHO disease outbreak reports. Our framework identifies naturally occurring knowledge evolution and generates questions with gold answers tailored to different LLM knowledge cut-off dates. Through extensive evaluation of 12 open and closed-source LLMs across 3 knowledge probing formats, we demonstrate significant performance drops of up to 31% on evolveQA compared to static knowledge questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12300v1">Do LLMs and Humans Find the Same Questions Difficult? A Case Study on Japanese Quiz Answering</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      LLMs have achieved performance that surpasses humans in many NLP tasks. However, it remains unclear whether problems that are difficult for humans are also difficult for LLMs. This study investigates how the difficulty of quizzes in a buzzer setting differs between LLMs and humans. Specifically, we first collect Japanese quiz data including questions, answers, and correct response rate of humans, then prompted LLMs to answer the quizzes under several settings, and compare their correct answer rate to that of humans from two analytical perspectives. The experimental results showed that, compared to humans, LLMs struggle more with quizzes whose correct answers are not covered by Wikipedia entries, and also have difficulty with questions that require numerical answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12295v1">Privacy-Preserving Prompt Injection Detection for LLMs Using Federated Learning and Embedding-Based NLP Classification</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Prompt injection attacks are an emerging threat to large language models (LLMs), enabling malicious users to manipulate outputs through carefully designed inputs. Existing detection approaches often require centralizing prompt data, creating significant privacy risks. This paper proposes a privacy-preserving prompt injection detection framework based on federated learning and embedding-based classification. A curated dataset of benign and adversarial prompts was encoded with sentence embedding and used to train both centralized and federated logistic regression models. The federated approach preserved privacy by sharing only model parameters across clients, while achieving detection performance comparable to centralized training. Results demonstrate that effective prompt injection detection is feasible without exposing raw data, making this one of the first explorations of federated security for LLMs. Although the dataset is limited in scale, the findings establish a strong proof-of-concept and highlight new directions for building secure and privacy-aware LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12288v1">Reducing Hallucinations in LLM-Generated Code via Semantic Triangulation</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      When generating code from natural language prompts, an LLM samples programs from a probability distribution, many of which might be incorrect. Sample consensus techniques - such as majority voting or validation against generated tests or specifications - aim to identify a correct program in the sample or abstain if none is valid. However, existing methods often fail to select a correct solution when its sampling probability is low, or when the problem permits multiple valid but non-equivalent solutions. Additionally, they often fail to abstain when no correct solution is present in the sample. To overcome these limitations, we introduce semantic triangulation, which transforms a programming problem in a way that non-trivially alters its semantics while preserving an exact, verifiable mapping between solutions before and after transformation. We theoretically establish that verifying consistency across such problem transformations increases confidence that generated programs reflect accurate generalization rather than spurious statistical correlations, enabling more reliable sample consensus and abstention. On the LiveCodeBench and CodeElo benchmarks, using GPT-4o and DeepSeek-V3 models, semantic triangulation increases reliability of generated code by 21% compared to the method that selects only high-confidence solutions with the probability threshold 0.5, while being able to pinpoint correct solutions at sampling probabilities as low as 0.14. Apart from that, it is also the only approach to consistently form true consensus on tasks with multiple valid but non-equivalent solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12286v1">Sangam: Chiplet-Based DRAM-PIM Accelerator with CXL Integration for LLM Inferencing</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming increasingly data-intensive due to growing model sizes, and they are becoming memory-bound as the context length and, consequently, the key-value (KV) cache size increase. Inference, particularly the decoding phase, is dominated by memory-bound GEMV or flat GEMM operations with low operational intensity (OI), making it well-suited for processing-in-memory (PIM) approaches. However, existing in/near-memory solutions face critical limitations such as reduced memory capacity due to the high area cost of integrating processing elements (PEs) within DRAM chips, and limited PE capability due to the constraints of DRAM fabrication technology. This work presents a chiplet-based memory module that addresses these limitations by decoupling logic and memory into chiplets fabricated in heterogeneous technology nodes and connected via an interposer. The logic chiplets sustain high bandwidth access to the DRAM chiplets, which house the memory banks, and enable the integration of advanced processing components such as systolic arrays and SRAM-based buffers to accelerate memory-bound GEMM kernels, capabilities that were not feasible in prior PIM architectures. We propose Sangam, a CXL-attached PIM-chiplet based memory module that can either act as a drop-in replacement for GPUs or co-executes along side the GPUs. Sangam achieves speedup of 3.93, 4.22, 2.82x speedup in end-to-end query latency, 10.3, 9.5, 6.36x greater decoding throughput, and order of magnitude energy savings compared to an H100 GPU for varying input size, output length, and batch size on LLaMA 2-7B, Mistral-7B, and LLaMA 3-70B, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12271v1">MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Accepted for AAAI 2026
    </div>
    <details class="paper-abstract">
      Large language models are increasingly influencing human moral decisions, yet current approaches focus primarily on evaluating rather than actively steering their moral decisions. We formulate this as an out-of-distribution moral alignment problem, where LLM agents must learn to apply consistent moral reasoning frameworks to scenarios beyond their training distribution. We introduce Moral-Reason-QA, a novel dataset extending 680 human-annotated, high-ambiguity moral scenarios with framework-specific reasoning traces across utilitarian, deontological, and virtue ethics, enabling systematic evaluation of moral generalization in realistic decision contexts. Our learning approach employs Group Relative Policy Optimization with composite rewards that simultaneously optimize decision alignment and framework-specific reasoning processes to facilitate learning of the underlying moral frameworks. Experimental results demonstrate successful generalization to unseen moral scenarios, with softmax-normalized alignment scores improving by +0.757 for utilitarian and +0.450 for deontological frameworks when tested on out-of-distribution evaluation sets. The experiments also reveal training challenges and promising directions that inform future research. These findings establish that LLM agents can be systematically trained to internalize and apply specific moral frameworks to novel situations, providing a critical foundation for AI safety as language models become more integrated into human decision-making processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06194v2">NURBGen: High-Fidelity Text-to-CAD Generation through LLM-Driven NURBS Modeling</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Accepted in AAAI 2026
    </div>
    <details class="paper-abstract">
      Generating editable 3D CAD models from natural language remains challenging, as existing text-to-CAD systems either produce meshes or rely on scarce design-history data. We present NURBGen, the first framework to generate high-fidelity 3D CAD models directly from text using Non-Uniform Rational B-Splines (NURBS). To achieve this, we fine-tune a large language model (LLM) to translate free-form texts into JSON representations containing NURBS surface parameters (\textit{i.e}, control points, knot vectors, degrees, and rational weights) which can be directly converted into BRep format using Python. We further propose a hybrid representation that combines untrimmed NURBS with analytic primitives to handle trimmed surfaces and degenerate regions more robustly, while reducing token complexity. Additionally, we introduce partABC, a curated subset of the ABC dataset consisting of individual CAD components, annotated with detailed captions using an automated annotation pipeline. NURBGen demonstrates strong performance on diverse prompts, surpassing prior methods in geometric fidelity and dimensional accuracy, as confirmed by expert evaluations. Code and dataset will be released publicly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.14995v2">LLM-Enhanced Multi-Agent Reinforcement Learning with Expert Workflow for Real-Time P2P Energy Trading</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Real-time peer-to-peer (P2P) electricity markets dynamically adapt to fluctuations in renewable energy and variations in demand, maximizing economic benefits through instantaneous price responses while enhancing grid flexibility. However, scaling expert guidance for massive personalized prosumers poses critical challenges, including diverse decision-making demands and lack of customized modeling frameworks. This paper proposed an integrated large language model-multi-agent reinforcement learning (LLM-MARL) framework for real-time P2P energy trading to address challenges such as the limited technical capability of prosumers, the lack of expert experience, and security issues of distribution networks. LLMs are introduced as experts to generate personalized strategy, guiding MARL under the centralized training with decentralized execution (CTDE) paradigm through imitation learning. A differential attention-based critic network is designed to enhance convergence performance. Experimental results demonstrate that LLM generated strategies effectively substitute human experts. The proposed multi-agent imitation learning algorithms achieve significantly lower economic costs and voltage violation rates on test sets compared to baselines algorithms, while maintaining robust stability. This work provides an effective solution for real-time P2P electricity market decision-making by bridging expert knowledge with agent learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12236v1">Consistency Is the Key: Detecting Hallucinations in LLM Generated Text By Checking Inconsistencies About Key Facts</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 To appear at International Joint Conference on Natural Language Processing & Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP-AACL), 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), despite their remarkable text generation capabilities, often hallucinate and generate text that is factually incorrect and not grounded in real-world knowledge. This poses serious risks in domains like healthcare, finance, and customer support. A typical way to use LLMs is via the APIs provided by LLM vendors where there is no access to model weights or options to fine-tune the model. Existing methods to detect hallucinations in such settings where the model access is restricted or constrained by resources typically require making multiple LLM API calls, increasing latency and API cost. We introduce CONFACTCHECK, an efficient hallucination detection approach that does not leverage any external knowledge base and works on the simple intuition that responses to factual probes within the generated text should be consistent within a single LLM and across different LLMs. Rigorous empirical evaluation on multiple datasets that cover both the generation of factual texts and the open generation shows that CONFACTCHECK can detect hallucinated facts efficiently using fewer resources and achieves higher accuracy scores compared to existing baselines that operate under similar conditions. Our code is available here.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12224v1">RulePilot: An LLM-Powered Agent for Security Rule Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 This paper has been accepted for publication at ICSE 2026
    </div>
    <details class="paper-abstract">
      The real-time demand for system security leads to the detection rules becoming an integral part of the intrusion detection life-cycle. Rule-based detection often identifies malicious logs based on the predefined grammar logic, requiring experts with deep domain knowledge for rule generation. Therefore, automation of rule generation can result in significant time savings and ease the burden of rule-related tasks on security engineers. In this paper, we propose RulePilot, which mimics human expertise via LLM-based agent for addressing rule-related challenges like rule creation or conversion. Using RulePilot, the security analysts do not need to write down the rules following the grammar, instead, they can just provide the annotations such as the natural-language-based descriptions of a rule, our RulePilot can automatically generate the detection rules without more intervention. RulePilot is equipped with the intermediate representation (IR), which abstracts the complexity of config rules into structured, standardized formats, allowing LLMs to focus on generation rules in a more manageable and consistent way. We present a comprehensive evaluation of RulePilot in terms of textual similarity and execution success abilities, showcasing RulePilot can generate high-fidelity rules, outperforming the baseline models by up to 107.4% in textual similarity to ground truths and achieving better detection accuracy in real-world execution tests. We perform a case study from our industry collaborators in Singapore, showcasing that RulePilot significantly help junior analysts/general users in the rule creation process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12217v1">AlignTree: Efficient Defense Against LLM Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Accepted as an Oral Presentation at the 40th AAAI Conference on Artificial Intelligence (AAAI-26), January 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are vulnerable to adversarial attacks that bypass safety guidelines and generate harmful content. Mitigating these vulnerabilities requires defense mechanisms that are both robust and computationally efficient. However, existing approaches either incur high computational costs or rely on lightweight defenses that can be easily circumvented, rendering them impractical for real-world LLM-based systems. In this work, we introduce the AlignTree defense, which enhances model alignment while maintaining minimal computational overhead. AlignTree monitors LLM activations during generation and detects misaligned behavior using an efficient random forest classifier. This classifier operates on two signals: (i) the refusal direction -- a linear representation that activates on misaligned prompts, and (ii) an SVM-based signal that captures non-linear features associated with harmful content. Unlike previous methods, AlignTree does not require additional prompts or auxiliary guard models. Through extensive experiments, we demonstrate the efficiency and robustness of AlignTree across multiple LLMs and benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.12396v3">LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Graph neural networks (GNNs) have advanced recommender systems by modeling interaction relationships. However, existing graph-based recommenders rely on sparse ID features and do not fully exploit textual information, resulting in low information density within representations. Furthermore, graph contrastive learning faces challenges. Random negative sampling can introduce false negative samples, while fixed temperature coefficients cannot adapt to the heterogeneity of different nodes. In addition, current efforts to enhance recommendations with large language models (LLMs) have not fully utilized their Chain-of-Thought (CoT) reasoning capabilities to guide representation learning. To address these limitations, we introduces LGHRec (LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization). This framework leverages the CoT reasoning ability of LLMs to generate semantic IDs, enriching reasoning processes and improving information density and semantic quality of representations. Moreover, we design a reinforcement learning algorithm, Harmonized Group Policy Optimization (HGPO), to optimize negative sampling strategies and temperature coefficients in contrastive learning. This approach enhances long-tail recommendation performance and ensures optimization consistency across different groups. Experimental results on three datasets demonstrate that LGHRec improves representation quality through semantic IDs generated by LLM's CoT reasoning and effectively boosts contrastive learning with HGPO. Our method outperforms several baseline models. The code is available at: https://anonymous.4open.science/r/LLM-Rec.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.08824v2">LLM-Driven Robots Risk Enacting Discrimination, Violence, and Unlawful Actions</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 Published in International Journal of Social Robotics (2025). 49 pages (65 with references and appendix), 27 Figures, 8 Tables. Andrew Hundt and Rumaisa Azeem are equal contribution co-first authors. The positions of the two co-first authors were swapped from arxiv version 1 with the written consent of all four authors. The Version of Record is available via DOI: 10.1007/s12369-025-01301-x
    </div>
    <details class="paper-abstract">
      Members of the Human-Robot Interaction (HRI) and Machine Learning (ML) communities have proposed Large Language Models (LLMs) as a promising resource for robotics tasks such as natural language interaction, household and workplace tasks, approximating 'common sense reasoning', and modeling humans. However, recent research has raised concerns about the potential for LLMs to produce discriminatory outcomes and unsafe behaviors in real-world robot experiments and applications. To assess whether such concerns are well placed in the context of HRI, we evaluate several highly-rated LLMs on discrimination and safety criteria. Our evaluation reveals that LLMs are currently unsafe for people across a diverse range of protected identity characteristics, including, but not limited to, race, gender, disability status, nationality, religion, and their intersections. Concretely, we show that LLMs produce directly discriminatory outcomes- e.g., 'gypsy' and 'mute' people are labeled untrustworthy, but not 'european' or 'able-bodied' people. We find various such examples of direct discrimination on HRI tasks such as facial expression, proxemics, security, rescue, and task assignment. Furthermore, we test models in settings with unconstrained natural language (open vocabulary) inputs, and find they fail to act safely, generating responses that accept dangerous, violent, or unlawful instructions-such as incident-causing misstatements, taking people's mobility aids, and sexual predation. Our results underscore the urgent need for systematic, routine, and comprehensive risk assessments and assurances to improve outcomes and ensure LLMs only operate on robots when it is safe, effective, and just to do so. We provide code to reproduce our experiments at https://github.com/rumaisa-azeem/llm-robots-discrimination-safety .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12033v1">EARL: Entropy-Aware RL Alignment of LLMs for Reliable RTL Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated significant potential in hardware design automation, particularly in using natural language to synthesize Register-Transfer Level (RTL) code. Despite this progress, a gap remains between model capability and the demands of real-world RTL design, including syntax errors, functional hallucinations, and weak alignment to designer intent. Reinforcement Learning with Verifiable Rewards (RLVR) offers a promising approach to bridge this gap, as hardware provides executable and formally checkable signals that can be used to further align model outputs with design intent. However, in long, structured RTL code sequences, not all tokens contribute equally to functional correctness, and naïvely spreading gradients across all tokens dilutes learning signals. A key insight from our entropy analysis in RTL generation is that only a small fraction of tokens (e.g., always, if, assign, posedge) exhibit high uncertainty and largely influence control flow and module structure. To address these challenges, we present EARL, an Entropy-Aware Reinforcement Learning framework for Verilog generation. EARL performs policy optimization using verifiable reward signals and introduces entropy-guided selective updates that gate policy gradients to high-entropy tokens. This approach preserves training stability and concentrates gradient updates on functionally important regions of code. Our experiments on VerilogEval and RTLLM show that EARL improves functional pass rates over prior LLM baselines by up to 14.7%, while reducing unnecessary updates and improving training stability. These results indicate that focusing RL on critical, high-uncertainty tokens enables more reliable and targeted policy improvement for structured RTL code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12031v1">Striking the Right Balance between Compute and Copy: Improving LLM Inferencing Under Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      With the skyrocketing costs of GPUs and their virtual instances in the cloud, there is a significant desire to use CPUs for large language model (LLM) inference. KV cache update, often implemented as allocation, copying, and in-place strided update for each generated token, incurs significant overhead. As the sequence length increases, the allocation and copy overheads dominate the performance. Alternate approaches may allocate large KV tensors upfront to enable in-place updates, but these matrices (with zero-padded rows) cause redundant computations. In this work, we propose a new KV cache allocation mechanism called Balancing Memory and Compute (BMC). BMC allocates, once every r iterations, KV tensors with r redundant rows, allowing in-place update without copy overhead for those iterations, but at the expense of a small amount of redundant computation. Second, we make an interesting observation that the extra rows allocated in the KV tensors and the resulting redundant computation can be repurposed for Speculative Decoding (SD) that improves token generation efficiency. Last, BMC represents a spectrum of design points with different values of r. To identify the best-performing design point(s), we derive a simple analytical model for BMC. The proposed BMC method achieves an average throughput acceleration of up to 3.2x over baseline HuggingFace (without SD). Importantly when we apply BMC with SD, it results in an additional speedup of up to 1.39x, over and above the speedup offered by SD. Further, BMC achieves a throughput acceleration of up to 1.36x and 2.29x over state-of-the-art inference servers vLLM and DeepSpeed, respectively. Although the BMC technique is evaluated extensively across different classes of CPUs (desktop and server class), we also evaluate the scheme with GPUs and demonstrate that it works well for GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.12014v1">CURE: Cultural Understanding and Reasoning Evaluation - A Framework for "Thick" Culture Alignment Evaluation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-15
      | 💬 7 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in culturally diverse environments, yet existing evaluations of cultural competence remain limited. Existing methods focus on de-contextualized correctness or forced-choice judgments, overlooking the need for cultural understanding and reasoning required for appropriate responses. To address this gap, we introduce a set of benchmarks that, instead of directly probing abstract norms or isolated statements, present models with realistic situational contexts that require culturally grounded reasoning. In addition to the standard Exact Match metric, we introduce four complementary metrics (Coverage, Specificity, Connotation, and Coherence) to capture different dimensions of model's response quality. Empirical analysis across frontier models reveals that thin evaluation systematically overestimates cultural competence and produces unstable assessments with high variance. In contrast, thick evaluation exposes differences in reasoning depth, reduces variance, and provides more stable, interpretable signals of cultural understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.24189v2">PET: Preference Evolution Tracking with LLM-Generated Explainable Distribution</a></div>
    <div class="paper-meta">
      📅 2025-11-15
    </div>
    <details class="paper-abstract">
      Understanding how user preference evolves over time is a fundamental challenge central to modern digital ecosystems, for which Large Language Models (LLMs) are an increasingly prominent and popular approach due to their ability to comprehend the rich semantic context within behavioral data. A common practice is to use LLMs to predict a user's next action by directly generating a ranked list of preferred items. Although effective for short-term prediction, the end-to-end generation paradigm inherently limits personalization. Its opaque decision-making process obscures holistic user profiling and exacerbates popularity bias. To address these limitations, we propose Preference Evolution Tracking (PET), a framework that reframes the task as inferring a dynamic probability distribution over a stable and interpretable lattice of preference clusters. By applying logit-probing and generative classification techniques, PET infers a user's preference as a probability distribution, enabling transparent preference learning. On public benchmarks (Yelp, MovieLens), PET improves ranking quality by up to 40% in NDCG over direct generation baselines. On a large-scale, real-world dataset from a short-video platform, it excels at ranking long-tail contents, significantly outperforming a SOTA production model by 7 times in the NDCG score. Ultimately, PET transforms the user profile model from direct preference list generation to a transparent distributional preference mapping, paving the way for more explainable, fair, and diverse personalization systems.
    </details>
</div>
