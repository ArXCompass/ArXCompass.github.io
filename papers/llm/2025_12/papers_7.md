# llm - 2025_12

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
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24616v4">Eye of Judgement: Dissecting the Evaluation of Russian-speaking LLMs with POLLUX</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 short version
    </div>
    <details class="paper-abstract">
      We introduce POLLUX, a comprehensive open-source benchmark designed to evaluate the generative capabilities of large language models (LLMs) in Russian. Our main contribution is a novel evaluation methodology that enhances the interpretability of LLM assessment. For each task type, we define a set of detailed criteria and develop a scoring protocol where models evaluate responses and provide justifications for their ratings. This enables transparent, criteria-driven evaluation beyond traditional resource-consuming, side-by-side human comparisons. POLLUX includes a detailed, fine-grained taxonomy of 35 task types covering diverse generative domains such as code generation, creative writing, and practical assistant use cases, totaling 2,100 manually crafted and professionally authored prompts. Each task is categorized by difficulty (easy/medium/hard), with experts constructing the dataset entirely from scratch. We also release a family of LLM-as-a-Judge (7B and 32B) evaluators trained for nuanced assessment of generative outputs. This approach provides scalable, interpretable evaluation and annotation tools for model development, effectively replacing costly and less precise human judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01735v1">Automating modeling in mechanics: LLMs as designers of physics-constrained neural networks for constitutive modeling of materials</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Currently under review
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agentic frameworks increasingly adopt the paradigm of dynamically generating task-specific agents. We suggest that not only agents but also specialized software modules for scientific and engineering tasks can be generated on demand. We demonstrate this concept in the field of solid mechanics. There, so-called constitutive models are required to describe the relationship between mechanical stress and body deformation. Constitutive models are essential for both the scientific understanding and industrial application of materials. However, even recent data-driven methods of constitutive modeling, such as constitutive artificial neural networks (CANNs), still require substantial expert knowledge and human labor. We present a framework in which an LLM generates a CANN on demand, tailored to a given material class and dataset provided by the user. The framework covers LLM-based architecture selection, integration of physical constraints, and complete code generation. Evaluation on three benchmark problems demonstrates that LLM-generated CANNs achieve accuracy comparable to or greater than manually engineered counterparts, while also exhibiting reliable generalization to unseen loading scenarios and extrapolation to large deformations. These findings indicate that LLM-based generation of physics-constrained neural networks can substantially reduce the expertise required for constitutive modeling and represent a step toward practical end-to-end automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01693v1">LLM-Driven Multi-Agent Curation and Expansion of Metal-Organic Frameworks Database</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Metal-organic framework (MOF) databases have grown rapidly through experimental deposition and large-scale literature extraction, but recent analyses show that nearly half of their entries contain substantial structural errors. These inaccuracies propagate through high-throughput screening and machine-learning workflows, limiting the reliability of data-driven MOF discovery. Correcting such errors is exceptionally difficult because true repairs require integrating crystallographic files, synthesis descriptions, and contextual evidence scattered across the literature. Here we introduce LitMOF, a large language model-driven multi-agent framework that validates crystallographic information directly from the original literature and cross-validates it with database entries to repair structural errors. Applying LitMOF to the experimental MOF database (the CSD MOF Subset), we constructed LitMOF-DB, a curated set 118,464 computation-ready structures, including corrections of 69% (6,161 MOFs) of the invalid MOFs in the latest CoRE MOF database. Additionally, the system uncovered 12,646 experimentally reported MOFs absent from existing resources, substantially expanding the known experimental design space. This work establishes a scalable pathway toward self-correcting scientific databases and a generalizable paradigm for LLM-driven curation in materials science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01672v1">ICAD-LLM: One-for-All Anomaly Detection via In-Context Learning with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Anomaly detection (AD) is a fundamental task of critical importance across numerous domains. Current systems increasingly operate in rapidly evolving environments that generate diverse yet interconnected data modalities -- such as time series, system logs, and tabular records -- as exemplified by modern IT systems. Effective AD methods in such environments must therefore possess two critical capabilities: (1) the ability to handle heterogeneous data formats within a unified framework, allowing the model to process and detect multiple modalities in a consistent manner during anomalous events; (2) a strong generalization ability to quickly adapt to new scenarios without extensive retraining. However, most existing methods fall short of these requirements, as they typically focus on single modalities and lack the flexibility to generalize across domains. To address this gap, we introduce a novel paradigm: In-Context Anomaly Detection (ICAD), where anomalies are defined by their dissimilarity to a relevant reference set of normal samples. Under this paradigm, we propose ICAD-LLM, a unified AD framework leveraging Large Language Models' in-context learning abilities to process heterogeneous data within a single model. Extensive experiments demonstrate that ICAD-LLM achieves competitive performance with task-specific AD methods and exhibits strong generalization to previously unseen tasks, which substantially reduces deployment costs and enables rapid adaptation to new environments. To the best of our knowledge, ICAD-LLM is the first model capable of handling anomaly detection tasks across diverse domains and modalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01661v1">Learning the Boundary of Solvability: Aligning LLMs to Detect Unsolvable Problems</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Ensuring LLM reliability requires not only solving complex problems but also recognizing when a problem is unsolvable. Current models often struggle to distinguish objective unsolvability (inherent contradictions in the problem) from subjective capability limitations (problems beyond the model's competence), which leads to hallucinations and overconfidence. To address this, we propose UnsolvableQA and UnsolvableRL to solve feasible problems, detect inherent contradictions, and prudently refuse tasks beyond capability. Specifically, we construct UnsolvableQA, a dataset of paired solvable and unsolvable instances derived via a dual-track methodology: programmatic generation for logic puzzles and a novel "Reverse Construction" method that injects contradictions into valid reasoning chains for mathematics. Building on this dataset, we introduce UnsolvableRL, a reinforcement learning framework with three reward components jointly accounting for accuracy, unsolvability, and difficulty. Empirical results show that our approach achieves near-perfect unsolvability detection while also improving accuracy on solvable tasks. Crucially, we identify Capability Collapse, demonstrating that explicit exposure to unsolvable data is indispensable for preventing models from becoming systematically overconfident. Our code and data are available at https://github.com/sfasfaffa/unsolvableQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01644v1">A Systematic Characterization of LLM Inference on GPUs</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      This work presents a systematic characterization of Large Language Model (LLM) inference to address fragmented understanding. Through comprehensive experiments, we establish a four-dimensional analytical framework: (1) Two-Phase Heterogeneity Observation; (2) Microarchitectural Root Cause Analysis; (3) System Scaling Principles; and (4) Emerging Paradigm Boundaries. Our investigation progresses systematically from observation to foresight: identifying performance phenomena, revealing hardware causes, validating system behavior, and exploring new paradigms. This study not only consolidates a reliable empirical foundation for existing research but also provides new discoveries and practical optimization guidance for LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.01002v2">Optimal Scheduling Algorithms for LLM Inference: Theory and Practice</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      With the growing use of Large Language Model (LLM)-based tools like ChatGPT, Perplexity, and Gemini across industries, there is a rising need for efficient LLM inference systems. These systems handle requests with a unique two-phase computation structure: a prefill-phase that processes the full input prompt and a decode-phase that autoregressively generates tokens one at a time. This structure calls for new strategies for routing and scheduling requests. In this paper, we take a comprehensive approach to this challenge by developing a theoretical framework that models routing and scheduling in LLM inference systems. We identify two key design principles-optimal tiling and dynamic resource allocation-that are essential for achieving high throughput. Guided by these principles, we propose the Resource-Aware Dynamic (RAD) scheduler and prove that it achieves throughput optimality under mild conditions. To address practical Service Level Objectives (SLOs) such as serving requests with different Time Between Token (TBT) constraints, we design the SLO-Aware LLM Inference (SLAI) scheduler. SLAI uses real-time measurements to prioritize decode requests that are close to missing their TBT deadlines and reorders prefill requests based on known prompt lengths to further reduce the Time To First Token (TTFT) delays. We evaluate SLAI on the Openchat ShareGPT4 dataset using the Mistral-7B model on an NVIDIA RTX ADA 6000 GPU. Compared to Sarathi-Serve, SLAI reduces the median TTFT by 53% and increases the maximum serving capacity by 26% such that median TTFT is below 0.5 seconds, while meeting tail TBT latency constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01610v1">Agent-Kernel: A MicroKernel Multi-Agent System Framework for Adaptive Social Simulation Powered by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Multi-Agent System (MAS) developing frameworks serve as the foundational infrastructure for social simulations powered by Large Language Models (LLMs). However, existing frameworks fail to adequately support large-scale simulation development due to inherent limitations in adaptability, configurability, reliability, and code reusability. For example, they cannot simulate a society where the agent population and profiles change over time. To fill this gap, we propose Agent-Kernel, a framework built upon a novel society-centric modular microkernel architecture. It decouples core system functions from simulation logic and separates cognitive processes from physical environments and action execution. Consequently, Agent-Kernel achieves superior adaptability, configurability, reliability, and reusability. We validate the framework's superiority through two distinct applications: a simulation of the Universe 25 (Mouse Utopia) experiment, which demonstrates the handling of rapid population dynamics from birth to death; and a large-scale simulation of the Zhejiang University Campus Life, successfully coordinating 10,000 heterogeneous agents, including students and faculty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01591v1">Scaling and context steer LLMs along the same computational path as the human brain</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Recent studies suggest that the representations learned by large language models (LLMs) are partially aligned to those of the human brain. However, whether and why this alignment score arises from a similar sequence of computations remains elusive. In this study, we explore this question by examining temporally-resolved brain signals of participants listening to 10 hours of an audiobook. We study these neural dynamics jointly with a benchmark encompassing 22 LLMs varying in size and architecture type. Our analyses confirm that LLMs and the brain generate representations in a similar order: specifically, activations in the initial layers of LLMs tend to best align with early brain responses, while the deeper layers of LLMs tend to best align with later brain responses. This brain-LLM alignment is consistent across transformers and recurrent architectures. However, its emergence depends on both model size and context length. Overall, this study sheds light on the sequential nature of computations and the factors underlying the partial convergence between biological and artificial neural networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.12726v4">DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in many natural language tasks but still struggle with complex, multi-step reasoning, particularly across diverse disciplines. Existing reasoning datasets often lack disciplinary breadth, reasoning depth, and diversity, as well as guiding principles for question synthesis. We propose DESIGNER: a DESIGN-logic-guidEd Reasoning data synthesis pipeline that leverages naturally available, extensive raw documents (e.g., book corpus and web corpus) to generate multidisciplinary challenging questions. We introduce the concept of "design logic" and instruct LLMs to mimic human educators' question-creation process, enabling the automated synthesis of large-scale, high-difficulty questions. We use LLMs to reverse-engineer and abstract over 120,000 design logics from existing questions across various disciplines. By matching these design logics with source documents, we are able to generate reasoning questions with controllable question types and difficulty levels. Using this pipeline, we synthesized two large-scale reasoning datasets that span 75 disciplines: DLR-Book (3.04 million questions from the book corpus) and DLR-Web (1.66 million questions from the web corpus). Data analysis indicates that the questions synthesized by our method exhibit greater difficulty and diversity compared to those in the baseline datasets. We validate our synthesized data through supervised fine-tuning (SFT) on the Qwen3 and Llama3 model families. Our data substantially enhances their multidisciplinary reasoning capabilities, outperforming existing datasets. Notably, by applying SFT on the base versions of these models using only our data, we even surpass their official final models that have undergone the full post-training process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01560v1">Estimating the prevalence of LLM-assisted text in scholarly writing</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 19 pages, 3 figures
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) in scholarly publications has grown dramatically since the launch of ChatGPT in late 2022. This usage is often undisclosed, and it can be challenging for readers and reviewers to identify human written but LLM-revised or translated text, or predominantly LLM-generated text. Given the known quality and reliability issues connected with LLM-generated text, their potential growth poses an increasing problem for research integrity, and for public trust in research. This study presents a simple and easily reproducible methodology to show the growth in the full text of published papers, across the full range of research, as indexed in the Dimensions database. It uses this to demonstrate that LLM tools are likely to have been involved in the production of more than 10% of all published papers in 2024, based on disproportionate use of specific indicative words, and draws together earlier studies to confirm that this is a plausible overall estimate. It then discusses the implications of this for the integrity of scholarly publishing, highlighting evidence that use of LLMs for text generation is still being concealed or downplayed by authors, and presents an argument that more comprehensive disclosure requirements are urgently required to address this.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.14450v2">Hyperion: Hierarchical Scheduling for Parallel LLM Acceleration in Multi-tier Networks</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      LLMs are increasingly executed in edge where limited GPU memory and heterogeneous computation jointly constrain deployment which motivates model partitioning and request scheduling. In this setting, minimizing latency requires addressing the tight coupling between model placement and request scheduling across heterogeneous nodes, as suboptimal decisions in one domain can negate benefits in the other. In this paper, we propose Hyperion, a hierarchical two-stage framework that jointly optimizes partitioning and scheduling for pipelined LLM inference. Hyperion minimizes latency by balancing resources across tiers without requiring model retraining or incurring significant runtime overhead. Leveraging the timescale difference between partitioning and request arrivals, Stage 1 performs offline, inter-tier partitioning via a Hyperion Split with Dynamic Programming (HypSplit-DP) procedure to produce balanced stage times under tier capacity and memory constraints; to adapt to time-varying load, Stage 2 performs online, intra-tier scheduling with a lightweight Hyperion Scheduling for Real-Time (HypSched-RT) that maps each request to the best available node using real-time estimates of queue length and effective capacity. Experiments with Phi-3-medium demonstrate that Hyperion reduces latency by up to 52.1% (vs. GPipe) and 31.2% (vs. HEFT). Furthermore, Hyperion exhibits superior scalability for long-sequence generation, maintaining 44.5% lower latency and higher GPU utilization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.01374v3">REASONING COMPILER: LLM-Guided Optimizations for Efficient Model Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      While model serving has unlocked unprecedented capabilities, the high cost of serving large-scale models continues to be a significant barrier to widespread accessibility and rapid innovation. Compiler optimizations have long driven substantial performance improvements, but existing compilers struggle with neural workloads due to the exponentially large and highly interdependent space of possible transformations. Although existing stochastic search techniques can be effective, they are often sample-inefficient and fail to leverage the structural context underlying compilation decisions. We set out to investigate the research question of whether reasoning with large language models (LLMs), without any retraining, can leverage the context-aware decision space of compiler optimizations to significantly improve sample efficiency. To that end, we introduce a novel compilation framework (dubbed Reasoning Compiler) that formulates optimization as a sequential, context-aware decision process guided by a large language model and structured Monte Carlo tree search (MCTS). The LLM acts as a proposal mechanism, suggesting hardware-informed transformations that reflect the current program state and accumulated performance feedback. MCTS incorporates the LLM-generated proposals to balance exploration and exploitation, facilitating structured, context-sensitive traversal of the expansive compiler optimization space. By achieving substantial speedups with markedly fewer samples than leading neural compilers, our approach demonstrates the potential of LLM-guided reasoning to transform the landscape of compiler optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17220v2">PARROT: Persuasion and Agreement Robustness Rating of Output Truth -- A Sycophancy Robustness Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      This study presents PARROT (Persuasion and Agreement Robustness Rating of Output Truth), a robustness focused framework designed to measure the degradation in accuracy that occurs under social pressure exerted on users through authority and persuasion in large language models (LLMs) the phenomenon of sycophancy (excessive conformity). PARROT (i) isolates causal effects by comparing the neutral version of the same question with an authoritatively false version using a double-blind evaluation, (ii) quantifies confidence shifts toward the correct and imposed false responses using log-likelihood-based calibration tracking, and (iii) systematically classifies failure modes (e.g., robust correct, sycophantic agreement, reinforced error, stubborn error, self-correction, etc.) using an eight-state behavioral taxonomy. We evaluated 22 models using 1,302 MMLU-style multiple-choice questions across 13 domains and domain-specific authority templates. Findings show marked heterogeneity: advanced models (e.g., GPT-5, GPT-4.1, Claude Sonnet 4.5) exhibit low "follow rates" ($\leq 11\%$, GPT-5: 4\%) and minimal accuracy loss, while older/smaller models show severe epistemic collapse (GPT-4: 80\%, Qwen 2.5-1.5B: 94\%). The danger is not limited to response changes; weak models reduce confidence in the correct response while increasing confidence in the imposed incorrect response. While international law and global knowledge at the domain level exhibit high fragility, elementary mathematics is relatively resilient. Consequently, we argue that the goal of "resistance to overfitting pressure" should be addressed as a primary objective alongside accuracy, harm avoidance, and privacy for safe deployment in the real world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01434v1">A Flexible Multi-Agent LLM-Human Framework for Fast Human Validated Tool Building</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      We introduce CollabToolBuilder, a flexible multiagent LLM framework with expert-in-the-loop (HITL) guidance that iteratively learns to create tools for a target goal, aligning with human intent and process, while minimizing time for task/domain adaptation effort and human feedback capture. The architecture generates and validates tools via four specialized agents (Coach, Coder, Critic, Capitalizer) using a reinforced dynamic prompt and systematic human feedback integration to reinforce each agent's role toward goals and constraints. This work is best viewed as a system-level integration and methodology combining multi-agent in-context learning, HITL controls, and reusable tool capitalization for complex iterative problems such as scientific document generation. We illustrate it with preliminary experiments (e.g., generating state-of-the-art research papers or patents given an abstract) and discuss its applicability to other iterative problem-solving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01423v1">Active Hypothesis Testing under Computational Budgets with Applications to GWAS and LLM</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      In large-scale hypothesis testing, computing exact $p$-values or $e$-values is often resource-intensive, creating a need for budget-aware inferential methods. We propose a general framework for active hypothesis testing that leverages inexpensive auxiliary statistics to allocate a global computational budget. For each hypothesis, our data-adaptive procedure probabilistically decides whether to compute the exact test statistic or a transformed proxy, guaranteeing a valid $p$-value or $e$-value while satisfying the budget constraint in expectation. Theoretical guarantees are established for our constructions, showing that the procedure achieves optimality for $e$-values and for $p$-values under independence, and admissibility for $p$-values under general dependence. Empirical results from simulations and two real-world applications, including a large-scale genome-wide association study (GWAS) and a clinical prediction task leveraging large language models (LLM), demonstrate that our framework improves statistical efficiency under fixed resource limits.
    </details>
</div>
