# llm - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- Part 8

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.16873v2">Multimodal LLMs See Sentiment</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 12 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Understanding how visual content communicates sentiment is critical in an era where online interaction is increasingly dominated by this kind of media on social platforms. However, this remains a challenging problem, as sentiment perception is closely tied to complex, scene-level semantics. In this paper, we propose an original framework, MLLMsent, to investigate the sentiment reasoning capabilities of Multimodal Large Language Models (MLLMs) through three perspectives: (1) using those MLLMs for direct sentiment classification from images; (2) associating them with pre-trained LLMs for sentiment analysis on automatically generated image descriptions; and (3) fine-tuning the LLMs on sentiment-labeled image descriptions. Experiments on a recent and established benchmark demonstrate that our proposal, particularly the fine-tuned approach, achieves state-of-the-art results outperforming Lexicon-, CNN-, and Transformer-based baselines by up to 30.9%, 64.8%, and 42.4%, respectively, across different levels of evaluators' agreement and sentiment polarity categories. Remarkably, in a cross-dataset test, without any training on these new data, our model still outperforms, by up to 8.26%, the best runner-up, which has been trained directly on them. These results highlight the potential of the proposed visual reasoning scheme for advancing affective computing, while also establishing new benchmarks for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02910v1">In Silico Development of Psychometric Scales: Feasibility of Representative Population Data Simulation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Developing and validating psychometric scales requires large samples, multiple testing phases, and substantial resources. Recent advances in Large Language Models (LLMs) enable the generation of synthetic participant data by prompting models to answer items while impersonating individuals of specific demographic profiles, potentially allowing in silico piloting before real data collection. Across four preregistered studies (N = circa 300 each), we tested whether LLM-simulated datasets can reproduce the latent structures and measurement properties of human responses. In Studies 1-2, we compared LLM-generated data with real datasets for two validated scales; in Studies 3-4, we created new scales using EFA on simulated data and then examined whether these structures generalized to newly collected human samples. Simulated datasets replicated the intended factor structures in three of four studies and showed consistent configural and metric invariance, with scalar invariance achieved for the two newly developed scales. However, correlation-based tests revealed substantial differences between real and synthetic datasets, and notable discrepancies appeared in score distributions and variances. Thus, while LLMs capture group-level latent structures, they do not approximate individual-level data properties. Simulated datasets also showed full internal invariance across gender. Overall, LLM-generated data appear useful for early-stage, group-level psychometric prototyping, but not as substitutes for individual-level validation. We discuss methodological limitations, risks of bias and data pollution, and ethical considerations related to in silico psychometric simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.09481v2">Evaluating LLMs on Sequential API Call Through Automated Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      By integrating tools from external APIs, Large Language Models (LLMs) have expanded their promising capabilities in a diverse spectrum of complex real-world tasks. However, testing, evaluation, and analysis of LLM tool use remain in their early stages. Most existing benchmarks rely on manually collected test cases, many of which cannot be automatically checked for semantic correctness and instead depend on static methods such as string matching. Additionally, these benchmarks often overlook the complex interactions that occur between sequential API calls, which are common in real-world applications. To fill the gap, in this paper, we introduce StateGen, an automated framework designed to generate diverse coding tasks involving sequential API interactions. StateGen combines state-machine-based API constraint solving and validation, energy-based sampling, and control-flow injection to generate executable programs. These programs are then translated into human-like natural language task descriptions through a collaboration of two LLM agents. Utilizing StateGen, we construct StateEval, a benchmark encompassing 120 verified test cases spanning across three representative scenarios: Session Service, Tensor Operation, and ElevenLabs MCP. Experimental results confirm that StateGen can effectively generate challenging and realistic API-oriented tasks, highlighting areas for improvement in current LLMs incorporating APIs.We make our framework and benchmark publicly available to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.12726v5">DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable success in many natural language tasks but still struggle with complex, multi-step reasoning, particularly across diverse disciplines. Existing reasoning datasets often lack disciplinary breadth, reasoning depth, and diversity, as well as guiding principles for question synthesis. We propose DESIGNER: a DESIGN-logic-guidEd Reasoning data synthesis pipeline that leverages naturally available, extensive raw documents (e.g., book corpus and web corpus) to generate multidisciplinary challenging questions. We introduce the concept of "design logic" and instruct LLMs to mimic human educators' question-creation process, enabling the automated synthesis of large-scale, high-difficulty questions. We use LLMs to reverse-engineer and abstract over 120,000 design logics from existing questions across various disciplines. By matching these design logics with source documents, we are able to generate reasoning questions with controllable question types and difficulty levels. Using this pipeline, we synthesized two large-scale reasoning datasets that span 75 disciplines: DLR-Book (3.04 million questions from the book corpus) and DLR-Web (1.66 million questions from the web corpus). Data analysis indicates that the questions synthesized by our method exhibit greater difficulty and diversity compared to those in the baseline datasets. We validate our synthesized data through supervised fine-tuning (SFT) on the Qwen3 and Llama3 model families. Our data substantially enhances their multidisciplinary reasoning capabilities, outperforming existing datasets. Notably, by applying SFT on the base versions of these models using only our data, we even surpass their official final models that have undergone the full post-training process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.15230v4">PERP: Rethinking the Prune-Retrain Paradigm in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 32 pages, 7 figures, 24 tables
    </div>
    <details class="paper-abstract">
      Neural Networks can be effectively compressed through pruning, significantly reducing storage and compute demands while maintaining predictive performance. Simple yet effective methods like magnitude pruning remove less important parameters and typically require a costly retraining procedure to restore performance. However, with the rise of LLMs, full retraining has become infeasible due to memory and compute constraints. This study challenges the practice of retraining all parameters by showing that updating a small subset of highly expressive parameters can suffice to recover or even enhance performance after pruning. Surprisingly, retraining just 0.01%-0.05% of the parameters in GPT-architectures can match the performance of full retraining across various sparsity levels, significantly reducing compute and memory requirements, and enabling retraining of models with up to 30 billion parameters on a single GPU in minutes. To bridge the gap to full retraining in the high sparsity regime, we introduce two novel LoRA variants that, unlike standard LoRA, allow merging adapters back without compromising sparsity. Going a step further, we show that these methods can be applied for memory-efficient layer-wise reconstruction, significantly enhancing state-of-the-art retraining-free methods like Wanda (Sun et al., 2023) and SparseGPT (Frantar & Alistarh, 2023). Our findings present a promising alternative to avoiding retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01830v2">OpenREAD: Reinforced Open-Ended Reasoning for End-to-End Autonomous Driving with LLM-as-Critic</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Recently, two-stage fine-tuning strategies, e.g., acquiring essential driving knowledge through supervised fine-tuning (SFT) and further enhancing decision-making and planning via reinforcement fine-tuning (RFT), have shown strong potential in advancing the knowledge-driven autonomous driving (AD) paradigm. However, the learning nature of SFT still limits the generalization of reasoning, thereby constraining the full potential of driving performance. Meanwhile, current RFT approaches are primarily applied to downstream tasks, since scene understanding is an open-ended problem where corresponding rewards are difficult to quantify. To address these limitations, we propose OpenREAD, an OPEN-ended REasoning reinforced vision-language model (VLM)-based autonomous driving (AD) framework that enables end-to-end RFT across the full spectrum from high-level reasoning to low-level trajectory planning. Specifically, we begin by constructing large-scale Chain-of-Thought (CoT) annotations on open-source driving-related knowledge datasets, and employ the powerful Qwen3 large language model (LLM) as the critic in RFT to quantify reasoning quality for open-ended questions during reward modeling. Extensive experiments confirm that joint end-to-end RFT yields substantial improvements in both upstream and downstream tasks, enabling OpenREAD to achieve state-of-the-art performance on reasoning and planning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02841v1">Cross-Lingual Prompt Steerability: Towards Accurate and Robust LLM Behavior across Languages</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      System prompts provide a lightweight yet powerful mechanism for conditioning large language models (LLMs) at inference time. While prior work has focused on English-only settings, real-world deployments benefit from having a single prompt to operate reliably across languages. This paper presents a comprehensive study of how different system prompts steer models toward accurate and robust cross-lingual behavior. We propose a unified four-dimensional evaluation framework to assess system prompts in multilingual environments. Through large-scale experiments on five languages, three LLMs, and three benchmarks, we uncover that certain prompt components, such as CoT, emotion, and scenario, correlate with robust multilingual behavior. We develop a prompt optimization framework for multilingual settings and show it can automatically discover prompts that improve all metrics by 5-10%. Finally, we analyze over 10 million reasoning units and find that more performant system prompts induce more structured and consistent reasoning patterns, while reducing unnecessary language-switching. Together, we highlight system prompt optimization as a scalable path to accurate and robust multilingual LLM behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02810v1">Phase-Adaptive LLM Framework with Multi-Stage Validation for Construction Robot Task Allocation: A Systematic Benchmark Against Traditional Optimization Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Multi-robot task allocation in construction automation has traditionally relied on optimization methods such as Dynamic Programming and Reinforcement Learning. This research introduces the LangGraph-based Task Allocation Agent (LTAA), an LLM-driven framework that integrates phase-adaptive allocation strategies, multi-stage validation with hierarchical retries, and dynamic prompting for efficient robot coordination. Although recent LLM approaches show potential for construction robotics, they largely lack rigorous validation and benchmarking against established algorithms. This paper presents the first systematic comparison of LLM-based task allocation with traditional methods in construction scenarios.The study validates LLM feasibility through SMART-LLM replication and addresses implementation challenges using a Self-Corrective Agent Architecture. LTAA leverages natural-language reasoning combined with structured validation mechanisms, achieving major computational gains reducing token usage by 94.6% and allocation time by 86% through dynamic prompting. The framework adjusts its strategy across phases: emphasizing execution feasibility early and workload balance in later allocations.The authors evaluate LTAA against Dynamic Programming, Q-learning, and Deep Q-Network (DQN) baselines using construction operations from the TEACh human-robot collaboration dataset. In the Heavy Excels setting, where robots have strong task specializations, LTAA achieves 77% task completion with superior workload balance, outperforming all traditional methods. These findings show that LLM-based reasoning with structured validation can match established optimization algorithms while offering additional advantages such as interpretability, adaptability, and the ability to update task logic without retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.11306v2">iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-02
      | 💬 Accepted in AAAI 2026 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agent systems have advanced rapidly, driven by their strong generalization in zero-shot settings. To further enhance reasoning and accuracy on complex tasks, Multi-Agent Debate (MAD) has emerged as a promising framework that engages multiple LLM agents in structured debates to encourage diverse reasoning. However, triggering MAD for every query is inefficient, as it incurs substantial computational (token) cost and may even degrade accuracy by overturning correct single-agent answers. To address these limitations, we propose intelligent Multi-Agent Debate (iMAD), a token-efficient framework that selectively triggers MAD only when it is likely to be beneficial (i.e., correcting an initially wrong answer). To achieve this goal, iMAD learns generalizable model behaviors to make accurate debate decisions. Specifically, iMAD first prompts a single agent to produce a structured self-critique response, from which we extract 41 interpretable linguistic and semantic features capturing hesitation cues. Then, iMAD uses a lightweight debate-decision classifier, trained using our proposed FocusCal loss, to determine whether to trigger MAD, enabling robust debate decisions without test dataset-specific tuning. Through extensive experiments using six (visual) question answering datasets against five competitive baselines, we have shown that iMAD significantly reduces token usage (by up to 92%) while also improving final answer accuracy (by up to 13.5%).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02763v1">SurveyEval: Towards Comprehensive Evaluation of LLM-Generated Academic Surveys</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      LLM-based automatic survey systems are transforming how users acquire information from the web by integrating retrieval, organization, and content synthesis into end-to-end generation pipelines. While recent works focus on developing new generation pipelines, how to evaluate such complex systems remains a significant challenge. To this end, we introduce SurveyEval, a comprehensive benchmark that evaluates automatically generated surveys across three dimensions: overall quality, outline coherence, and reference accuracy. We extend the evaluation across 7 subjects and augment the LLM-as-a-Judge framework with human references to strengthen evaluation-human alignment. Evaluation results show that while general long-text or paper-writing systems tend to produce lower-quality surveys, specialized survey-generation systems are able to deliver substantially higher-quality results. We envision SurveyEval as a scalable testbed to understand and improve automatic survey systems across diverse subjects and evaluation criteria.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.21830v3">GAPO: Robust Advantage Estimation for Real-World Code LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) is widely used for post-training large language models (LLMs) in code editing, where group-relative methods like GRPO are popular for their critic-free, normalized advantage estimation. However, in real-world code-editing scenarios, reward distributions are often skewed with unpredictable outliers, leading to distorted advantage computation and increased noise. To address this issue, we propose Group Adaptive Policy Optimization (GAPO), which adaptively finds an outlier-free highest-density interval (HDI) per prompt and then uses the median of that interval as an adaptive Q to replace the group mean in advantage calculation. This adaptive Q robustly handles skewed distributions while remaining plug-and-play and efficient. We validate GAPO on nine instruction-tuned LLMs (3B-14B) using a large internal dataset of 51,844 real-world, history-aware code-editing tasks across 10 languages, demonstrating consistent improvements in exact match accuracy over GRPO and its variant DAPO. Code is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25045v2">Hyperdimensional Probe: Decoding LLM Representations via Vector Symbolic Architectures</a></div>
    <div class="paper-meta">
      📅 2025-12-02
    </div>
    <details class="paper-abstract">
      Despite their capabilities, Large Language Models (LLMs) remain opaque with limited understanding of their internal representations. Current interpretability methods either focus on input-oriented feature extraction, such as supervised probes and Sparse Autoencoders (SAEs), or on output distribution inspection, such as logit-oriented approaches. A full understanding of LLM vector spaces, however, requires integrating both perspectives, something existing approaches struggle with due to constraints on latent feature definitions. We introduce the Hyperdimensional Probe, a hybrid supervised probe that combines symbolic representations with neural probing. Leveraging Vector Symbolic Architectures (VSAs) and hypervector algebra, it unifies prior methods: the top-down interpretability of supervised probes, SAE's sparsity-driven proxy space, and output-oriented logit investigation. This allows deeper input-focused feature extraction while supporting output-oriented investigation. Our experiments show that our method consistently extracts meaningful concepts across LLMs, embedding sizes, and setups, uncovering concept-driven patterns in analogy-oriented inference and QA-focused text generation. By supporting joint input-output analysis, this work advances semantic understanding of neural representations while unifying the complementary perspectives of prior methods.
    </details>
</div>
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
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.08579v2">LLM-based Human Simulations Have Not Yet Been Reliable</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed for simulating human behaviors across diverse domains. However, our position is that current LLM-based human simulations remain insufficiently reliable, as evidenced by significant discrepancies between their outcomes and authentic human actions. Our investigation begins with a systematic review of LLM-based human simulations in social, economic, policy, and psychological contexts, identifying their common frameworks, recent advances, and persistent limitations. This review reveals that such discrepancies primarily stem from inherent limitations of LLMs and flaws in simulation design, both of which are examined in detail. Building on these insights, we propose a systematic solution framework that emphasizes enriching data foundations, advancing LLM capabilities, and ensuring robust simulation design to enhance reliability. Finally, we introduce a structured algorithm that operationalizes the proposed framework, aiming to guide credible and human-aligned LLM-based simulations. To facilitate further research, we provide a curated list of related literature and resources at https://github.com/Persdre/awesome-llm-human-simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01392v1">RE-LLM: Integrating Large Language Models into Renewable Energy Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Energy system models are increasingly employed to guide long-term planning in multi-sectoral environments where decisions span electricity, heat, transport, land use, and industry. While these models provide rigorous quantitative insights, their outputs are often highly technical, making them difficult to interpret for non-expert stakeholders such as policymakers, planners, and the public. This communication gap limits the accessibility and practical impact of scenario-based modeling, particularly as energy transitions grow more complex with rising shares of renewables, sectoral integration, and deep uncertainties. To address this challenge, we propose the Renewable Energy Large Language Model (RE-LLM), a hybrid framework that integrates Large Language Models (LLMs) directly into the energy system modeling workflow. RE-LLM combines three core elements: (i) optimization-based scenario exploration, (ii) machine learning surrogates that accelerate computationally intensive simulations, and (iii) LLM-powered natural language generation that translates complex results into clear, stakeholder-oriented explanations. This integrated design not only reduces computational burden but also enhances inter-pretability, enabling real-time reasoning about trade-offs, sensitivities, and policy implications. The framework is adaptable across different optimization platforms and energy system models, ensuring broad applicability beyond the case study presented. By merging speed, rigor, and interpretability, RE-LLM advances a new paradigm of human-centric energy modeling. It enables interactive, multilingual, and accessible engagement with future energy pathways, ultimately bridging the final gap between data-driven analysis and actionable decision-making for sustainable transitions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01357v1">Tangram: Accelerating Serverless LLM Loading through GPU Memory Reuse and Affinity</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Serverless Large Language Models (LLMs) have emerged as a cost-effective solution for deploying AI services by enabling a 'pay-as-you-go' pricing model through GPU resource sharing. However, cold-start latency, especially the model loading phase, has become a critical performance bottleneck, as it scales linearly with model size and severely limits the practical deployment of large-scale LLM services. This paper presents Tangram, a novel system that accelerates Serverless LLM loading through efficient GPU memory reuse. By leveraging the unused GPU memory to retain model parameters, Tangram significantly reduces model transfer time and cold-start latency. Its design includes three key components: unified GPU memory pool for tensor-level parameter sharing across models, on-demand KV cache allocation for dynamic memory management, and GPU-affinity-aware scheduling for maximizing resource utilization. These techniques collectively address the critical challenges of inefficient memory usage and the cold-start problem in Serverless LLM platforms. We have implemented a fully functional prototype, and experiments show that Tangram achieves up to 6.2 times faster loading and reduces Time-To-First-Token (TTFT) during cold-start by 23--55% over state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01356v1">LAURA: Enhancing Code Review Generation with Context-Enriched Retrieval-Augmented LLM</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Accepted by the 2025 40th IEEE/ACM International Conference on Automated Software Engineering (ASE). Copyright 2025 IEEE. This is the author's accepted manuscript. The final published version may differ and will be available from IEEE Xplore
    </div>
    <details class="paper-abstract">
      Code review is critical for ensuring software quality and maintainability. With the rapid growth in software scale and complexity, code review has become a bottleneck in the development process because of its time-consuming and knowledge-intensive nature and the shortage of experienced developers willing to review code. Several approaches have been proposed for automatically generating code reviews based on retrieval, neural machine translation, pre-trained models, or large language models (LLMs). These approaches mainly leverage historical code changes and review comments. However, a large amount of crucial information for code review, such as the context of code changes and prior review knowledge, has been overlooked. This paper proposes an LLM-based review knowledge-augmented, context-aware framework for code review generation, named LAURA. The framework integrates review exemplar retrieval, context augmentation, and systematic guidance to enhance the performance of ChatGPT-4o and DeepSeek v3 in generating code review comments. Besides, given the extensive low-quality reviews in existing datasets, we also constructed a high-quality dataset. Experimental results show that for both models, LAURA generates review comments that are either completely correct or at least helpful to developers in 42.2% and 40.4% of cases, respectively, significantly outperforming SOTA baselines. Furthermore, our ablation studies demonstrate that all components of LAURA contribute positively to improving comment quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01353v1">A Wolf in Sheep's Clothing: Bypassing Commercial LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Existing approaches overwhelmingly operate within the prompt-optimization paradigm: whether through traditional algorithmic search or recent agent-based workflows, the resulting prompts typically retain malicious semantic signals that modern guardrails are primed to detect. In contrast, we identify a deeper, largely overlooked vulnerability stemming from the highly interconnected nature of an LLM's internal knowledge. This structure allows harmful objectives to be realized by weaving together sequences of benign sub-queries, each of which individually evades detection. To exploit this loophole, we introduce the Correlated Knowledge Attack Agent (CKA-Agent), a dynamic framework that reframes jailbreaking as an adaptive, tree-structured exploration of the target model's knowledge base. The CKA-Agent issues locally innocuous queries, uses model responses to guide exploration across multiple paths, and ultimately assembles the aggregated information to achieve the original harmful objective. Evaluated across state-of-the-art commercial LLMs (Gemini2.5-Flash/Pro, GPT-oss-120B, Claude-Haiku-4.5), CKA-Agent consistently achieves over 95% success rates even against strong guardrails, underscoring the severity of this vulnerability and the urgent need for defenses against such knowledge-decomposition attacks. Our codes are available at https://github.com/Graph-COM/CKA-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01351v1">Benchmarking Overton Pluralism in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for measuring Overton pluralism in LLMs--the extent to which diverse viewpoints are represented in model outputs. We (i) formalize Overton pluralism as a set coverage metric (OvertonScore), (ii) conduct a large-scale U.S.-representative human study (N = 1209; 60 questions; 8 LLMs), and (iii) develop an automated benchmark that closely reproduces human judgments. On average, models achieve OvertonScores of 0.35--0.41, with DeepSeek V3 performing best; yet all models remain far below the theoretical maximum of 1.0, revealing substantial headroom for improvement. Because repeated large-scale human studies are costly and slow, scalable evaluation tools are essential for model development. Hence, we propose an automated benchmark that achieves high rank correlation with human judgments ($ρ=0.88$), providing a practical proxy without replacing human assessment. By turning pluralistic alignment from a normative aim into a measurable benchmark, our work establishes a foundation for systematic progress toward more pluralistic LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.14496v3">Semantic Energy: Detecting LLM Hallucination Beyond Entropy</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being increasingly deployed in real-world applications, but they remain susceptible to hallucinations, which produce fluent yet incorrect responses and lead to erroneous decision-making. Uncertainty estimation is a feasible approach to detect such hallucinations. For example, semantic entropy estimates uncertainty by considering the semantic diversity across multiple sampled responses, thus identifying hallucinations. However, semantic entropy relies on post-softmax probabilities and fails to capture the model's inherent uncertainty, causing it to be ineffective in certain scenarios. To address this issue, we introduce Semantic Energy, a novel uncertainty estimation framework that leverages the inherent confidence of LLMs by operating directly on logits of penultimate layer. By combining semantic clustering with a Boltzmann-inspired energy distribution, our method better captures uncertainty in cases where semantic entropy fails. Experiments across multiple benchmarks show that Semantic Energy significantly improves hallucination detection and uncertainty estimation, offering more reliable signals for downstream applications such as hallucination detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01326v1">Securing Large Language Models (LLMs) from Prompt Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 10 pages, 1 figure, 1 table
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being deployed in real-world applications, but their flexibility exposes them to prompt injection attacks. These attacks leverage the model's instruction-following ability to make it perform malicious tasks. Recent work has proposed JATMO, a task-specific fine-tuning approach that trains non-instruction-tuned base models to perform a single function, thereby reducing susceptibility to adversarial instructions. In this study, we evaluate the robustness of JATMO against HOUYI, a genetic attack framework that systematically mutates and optimizes adversarial prompts. We adapt HOUYI by introducing custom fitness scoring, modified mutation logic, and a new harness for local model testing, enabling a more accurate assessment of defense effectiveness. We fine-tuned LLaMA 2-7B, Qwen1.5-4B, and Qwen1.5-0.5B models under the JATMO methodology and compared them with a fine-tuned GPT-3.5-Turbo baseline. Results show that while JATMO reduces attack success rates relative to instruction-tuned models, it does not fully prevent injections; adversaries exploiting multilingual cues or code-related disruptors still bypass defenses. We also observe a trade-off between generation quality and injection vulnerability, suggesting that better task performance often correlates with increased susceptibility. Our results highlight both the promise and limitations of fine-tuning-based defenses and point toward the need for layered, adversarially informed mitigation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20150v3">Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are reshaping the recommender system paradigm by enabling users to express preferences and receive recommendations through conversations. Yet, aligning LLMs to the recommendation task remains challenging: pretrained LLMs often generate out-of-catalog items, violate required output formats, and their ranking quality degrades sharply toward the end of the generated list. To this end, we propose ConvRec-R1, a two-stage framework for end-to-end training of LLM-based conversational recommender systems. In Stage 1, we construct a behavioral-cloning dataset with a Remap-Reflect-Adjust pipeline, which produces high-quality, catalog-grounded demonstrations from powerful blackbox LLMs to warm-start the RL training. In Stage 2, we propose Rank-GRPO, a principled extension of group relative policy optimization (GRPO) tailored to tasks with rank-style outputs. Rank-GRPO treats each rank in the recommendation list as the unit instead of token (too fine-grained) or sequence (too coarse), redefining rewards to remove non-causal credit assignment and introducing a rank-level importance ratio based on the geometric mean of rank-wise token probabilities to stabilize policy updates. Experiments on the public Reddit-v2 dataset show that ConvRec-R1 converges faster and achieves higher Recall and NDCG than GRPO-style baselines. Code and datasets are released at https://github.com/yaochenzhu/Rank-GRPO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01282v1">Kardia-R1: Unleashing LLMs to Reason toward Understanding and Empathy for Emotional Support via Rubric-as-Judge Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      As web platforms evolve towards greater personalization and emotional complexity, conversational agents must transcend superficial empathy to demonstrate identity-aware emotional reasoning. However, existing systems face two limitations: (1) reliance on situation-centric datasets lacking persistent user identity, which hampers the capture of personalized affective nuances; and (2) dependence on opaque, coarse reward signals that hinder development of verifiable empathetic reasoning. To address these gaps, we introduce KardiaBench, a large-scale user-grounded benchmark comprising 178,080 QA pairs across 22,080 multi-turn conversations anchored to 671 real-world profiles. The dataset is constructed via a model-in-the-loop pipeline with iterative rubric-guided refinement to ensure psychological plausibility and persona consistency. This progressive empathy pipeline that integrates user comprehension, contextual reasoning, and emotion perception into conversations, followed by iterative critique and rubric-based refinement to ensure psychological plausibility, emotional fidelity, and persona consistency. Building on this, we propose Kardia-R1, a framework that trains models for interpretable, stepwise empathetic cognition. Kardia-R1 leverages Rubric-as-Judge Empathetic Reinforcement Learning (Rubric-ERL), a GRPO-based method that uses explainable, human-aligned rubric rewards to tightly couple user understanding, emotional inference, and supportive response generation. Extensive experiments across four LLM backbones demonstrate that Kardia-R1 consistently outperforms othet methods in emotion accuracy, empathy, relevance, persona consistency, and safety. Our dataset and model will be released at https://github.com/JhCircle/Kardia-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.18951v3">SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 30 pages, 10 figures, NeurIPS 2025 Main
    </div>
    <details class="paper-abstract">
      Resolution of complex SQL issues persists as a significant bottleneck in real-world database applications. Current Large Language Models (LLMs), while adept at text-to-SQL translation, have not been rigorously evaluated on the more challenging task of debugging SQL issues. To address this gap, we introduce BIRD-CRITIC, a new SQL issue debugging benchmark comprising 530 PostgreSQL tasks (BIRD-CRITIC-PG) and 570 multi-dialect tasks (BIRD-CRITIC-Multi), distilled from authentic user issues and replayed within new environments to facilitate rigorous evaluation. Baseline evaluations underscore the task's complexity, with the leading reasoning model O3-Mini achieving only 38.87% success rate on BIRD-CRITIC-PG and 33.33% on BIRD-CRITIC-Multi. Meanwhile, advancing open-source models for database tasks is crucial for empowering local development while safeguarding data privacy. Therefore, we present Six-Gym (Sql-fIX-Gym), a training environment for elevating open-source model capabilities for SQL issue debugging. This environment leverages SQL-Rewind strategy, which automatically generates executable issue-solution datasets by reverse-engineering issues from verified SQLs. However, popular trajectory-based fine-tuning methods do not explore substantial supervisory signals. We further propose f-Plan Boosting, which extracts high-level debugging plans from SQL solutions, enabling teacher LLMs to produce 73.7% more successful trajectories for training. We integrate these components into an open-source agent, Bird-Fixer. Based on Qwen-2.5-Coder-14B, Bird-Fixer achieves 38.11% success rate on BIRD-CRITIC-PG and 29.65% on BIRD-CRITIC-Multi, surpassing leading proprietary models such as Claude-3.7-Sonnet and GPT-4.1, marking a significant step toward democratizing sophisticated SQL-debugging capabilities. The leaderboard and source code are available: https://bird-critic.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.00494v2">Exploring System 1 and 2 communication for latent reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Should LLM reasoning live in a separate module, or within a single model's forward pass and representational space? We study dual-architecture latent reasoning, where a fluent Base exchanges latent messages with a Coprocessor, and test two hypotheses aimed at improving latent communication over Liu et al. (2024): (H1) increase channel capacity; (H2) learn communication via joint finetuning. Under matched latent-token budgets on GPT-2 and Qwen-3, H2 is consistently strongest while H1 yields modest gains. A unified soft-embedding baseline, a single model with the same forward pass and shared representations, using the same latent-token budget, nearly matches H2 and surpasses H1, suggesting current dual designs mostly add compute rather than qualitatively improving reasoning. Across GSM8K, ProsQA, and a Countdown stress test with increasing branching factor, scaling the latent-token budget beyond small values fails to improve robustness. Latent analyses show overlapping subspaces with limited specialization, consistent with weak reasoning gains. We conclude dual-model latent reasoning remains promising in principle, but likely requires objectives and training schedules that explicitly shape latent spaces for algorithmic planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01232v1">LLM-as-a-Judge for Scalable Test Coverage Evaluation: Accuracy, Operational Reliability, and Cost</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 7 pages, accepted by the AAAI 2026 Workshop on Next Gen Code Development with Collaborative AI Agents
    </div>
    <details class="paper-abstract">
      Assessing software test coverage at scale remains a bottleneck in QA pipelines. We present LLM-as-a-Judge (LAJ), a production-ready, rubric-driven framework for evaluating Gherkin acceptance tests with structured JSON outputs. Across 20 model configurations (GPT-4, GPT-5 with varying reasoning effort, and open-weight models) on 100 expert-annotated scripts over 5 runs (500 evaluations), we provide the first comprehensive analysis spanning accuracy, operational reliability, and cost. We introduce the Evaluation Completion Rate (ECR@1) to quantify first-attempt success, revealing reliability from 85.4% to 100.0% with material cost implications via retries. Results show that smaller models can outperform larger ones: GPT-4o Mini attains the best accuracy (6.07 MAAE), high reliability (96.6% ECR@1), and low cost ($1.01 per 1K), yielding a 78x cost reduction vs. GPT-5 (high reasoning) while improving accuracy. Reasoning effort is model-family dependent: GPT-5 benefits from increased reasoning (with predictable accuracy-cost tradeoffs), whereas open-weight models degrade across all dimensions as reasoning increases. Overall, cost spans 175x ($0.45-$78.96 per 1K). We release the dataset, framework, and code to support reproducibility and deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.04652v6">LLM Collaboration With Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      A large amount of work has been done in Multi-Agent Systems (MAS) for modeling and solving problems with multiple interacting agents. However, most LLMs are pretrained independently and not specifically optimized for coordination. Existing LLM fine-tuning frameworks rely on individual rewards, which require complex reward designs for each agent to encourage collaboration. To address these challenges, we model LLM collaboration as a cooperative Multi-Agent Reinforcement Learning (MARL) problem. We develop a multi-agent, multi-turn algorithm, Multi-Agent Group Relative Policy Optimization (MAGRPO), to solve it, building on current RL approaches for LLMs as well as MARL techniques. Our experiments on LLM writing and coding collaboration demonstrate that fine-tuning MAS with MAGRPO enables agents to generate high-quality responses efficiently through effective cooperation. Our approach opens the door to using other MARL methods for LLMs and highlights the associated challenges. Our code is available at https://github.com/OpenMLRL/CoMLRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01198v1">Conveying Imagistic Thinking in Traditional Chinese Medicine Translation: A Prompt Engineering and LLM-Based Evaluation Framework</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 3 figures
    </div>
    <details class="paper-abstract">
      Traditional Chinese Medicine (TCM) theory is built on imagistic thinking, in which medical principles and diagnostic and therapeutic logic are structured through metaphor and metonymy. However, existing English translations largely rely on literal rendering, making it difficult for target-language readers to reconstruct the underlying conceptual networks and apply them in clinical practice. This study adopted a human-in-the-loop (HITL) framework and selected four passages from the medical canon Huangdi Neijing that are fundamental in theory. Through prompt-based cognitive scaffolding, DeepSeek V3.1 was guided to identify metaphor and metonymy in the source text and convey the theory in translation. In the evaluation stage, ChatGPT 5 Pro and Gemini 2.5 Pro were instructed by prompts to simulate three types of real-world readers. Human translations, baseline model translations, and prompt-adjusted translations were scored by the simulated readers across five cognitive dimensions, followed by structured interviews and Interpretative Phenomenological Analysis (IPA). Results show that the prompt-adjusted LLM translations perform best across all five dimensions, with high cross-model and cross-role consistency. The interview themes reveal differences between human and machine translation, effective strategies for metaphor and metonymy transfer, and readers' cognitive preferences. This study provides a cognitive, efficient, and replicable HITL methodological pathway for the translation of ancient, concept-dense texts such as TCM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02716v2">A $1000\times$ Faster LLM-enhanced Algorithm For Path Planning in Large-scale Grid Maps</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Path planning in grid maps, arising from various applications, has garnered significant attention. Existing methods, such as A*, Dijkstra, and their variants, work well for small-scale maps but fail to address large-scale ones due to high search time and memory consumption. Recently, Large Language Models (LLMs) have shown remarkable performance in path planning but still suffer from spatial illusion and poor planning performance. Among all the works, LLM-A* \cite{meng2024llm} leverages LLM to generate a series of waypoints and then uses A* to plan the paths between the neighboring waypoints. In this way, the complete path is constructed. However, LLM-A* still suffers from high computational time for large-scale maps. To fill this gap, we conducted a deep investigation into LLM-A* and found its bottleneck, resulting in limited performance. Accordingly, we design an innovative LLM-enhanced algorithm, abbr. as iLLM-A*. iLLM-A* includes 3 carefully designed mechanisms, including the optimization of A*, an incremental learning method for LLM to generate high-quality waypoints, and the selection of the appropriate waypoints for A* for path planning. Finally, a comprehensive evaluation on various grid maps shows that, compared with LLM-A*, iLLM-A* \textbf{1) achieves more than $1000\times$ speedup on average, and up to $2349.5\times$ speedup in the extreme case, 2) saves up to $58.6\%$ of the memory cost, 3) achieves both obviously shorter path length and lower path length standard deviation.}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.18449v2">SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Accepted to NeurIPS 2025 Main Track
    </div>
    <details class="paper-abstract">
      The recent DeepSeek-R1 release has demonstrated the immense potential of reinforcement learning (RL) in enhancing the general reasoning capabilities of large language models (LLMs). While DeepSeek-R1 and other follow-up work primarily focus on applying RL to competitive coding and math problems, this paper introduces SWE-RL, the first approach to scale RL-based LLM reasoning for real-world software engineering. Leveraging a lightweight rule-based reward (e.g., the similarity score between ground-truth and LLM-generated solutions), SWE-RL enables LLMs to autonomously recover a developer's reasoning processes and solutions by learning from extensive open-source software evolution data -- the record of a software's entire lifecycle, including its code snapshots, code changes, and events such as issues and pull requests. Trained on top of Llama 3, our resulting reasoning model, Llama3-SWE-RL-70B, achieves a 41.0% solve rate on SWE-bench Verified -- a human-verified collection of real-world GitHub issues. To our knowledge, this is the best performance reported for medium-sized (<100B) LLMs to date, even comparable to leading proprietary LLMs like GPT-4o. Surprisingly, despite performing RL solely on software evolution data, Llama3-SWE-RL has even emerged with generalized reasoning skills. For example, it shows improved results on five out-of-domain tasks, namely, function coding, library use, code reasoning, mathematics, and general language understanding, whereas a supervised-finetuning baseline even leads to performance degradation on average. Overall, SWE-RL opens up a new direction to improve the reasoning capabilities of LLMs through reinforcement learning on massive software engineering data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.05239v3">LLM-based Automated Grading with Human-in-the-Loop</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Accepted to IEEE TALE 2025
    </div>
    <details class="paper-abstract">
      The rise of artificial intelligence (AI) technologies, particularly large language models (LLMs), has brought significant advancements to the field of education. Among various applications, automatic short answer grading (ASAG), which focuses on evaluating open-ended textual responses, has seen remarkable progress with the introduction of LLMs. These models not only enhance grading performance compared to traditional ASAG approaches but also move beyond simple comparisons with predefined "golden" answers, enabling more sophisticated grading scenarios, such as rubric-based evaluation. However, existing LLM-powered methods still face challenges in achieving human-level grading performance in rubric-based assessments due to their reliance on fully automated approaches. In this work, we explore the potential of LLMs in ASAG tasks by leveraging their interactive capabilities through a human-in-the-loop (HITL) approach. Our proposed framework, GradeHITL, utilizes the generative properties of LLMs to pose questions to human experts, incorporating their insights to refine grading rubrics dynamically. This adaptive process significantly improves grading accuracy, outperforming existing methods and bringing ASAG closer to human-level evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02282v1">DialogGuard: Multi-Agent Psychosocial Safety Evaluation of Sensitive LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) now mediate many web-based mental-health, crisis, and other emotionally sensitive services, yet their psychosocial safety in these settings remains poorly understood and weakly evaluated. We present DialogGuard, a multi-agent framework for assessing psychosocial risks in LLM-generated responses along five high-severity dimensions: privacy violations, discriminatory behaviour, mental manipulation, psychological harm, and insulting behaviour. DialogGuard can be applied to diverse generative models through four LLM-as-a-judge pipelines, including single-agent scoring, dual-agent correction, multi-agent debate, and stochastic majority voting, grounded in a shared three-level rubric usable by both human annotators and LLM judges. Using PKU-SafeRLHF with human safety annotations, we show that multi-agent mechanisms detect psychosocial risks more accurately than non-LLM baselines and single-agent judging; dual-agent correction and majority voting provide the best trade-off between accuracy, alignment with human ratings, and robustness, while debate attains higher recall but over-flags borderline cases. We release Dialog-Guard as open-source software with a web interface that provides per-dimension risk scores and explainable natural-language rationales. A formative study with 12 practitioners illustrates how it supports prompt design, auditing, and supervision of web-facing applications for vulnerable users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02281v1">Trinity: Disaggregating Vector Search from Prefill-Decode Disaggregation in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Prefill and decode (PD) disaggregation separates prompt prefill and token-by-token decode stages into distinct GPU pools and has become the dominant architecture for large-scale LLM serving in industry. Also, retrieval tasks via vector search remains entangled with the model inference process, like heterogeneous RAG requests and prompt answer caches, inflating tail latency. We are motivated to investigate how vector search should be orchestrated along with PD disaggregation with a dedicated deployment architecture without violating SLOs in various retrieval workloads. We present Trinity, a practical framework that consolidates all retrieval into a single, shared vector-search GPU pool and make it work with PD disaggregated LLM serving in match. Trinity introduces (1) a novel architecture for deploying GPU-based vector search service in PD disaggregation. (2) Continuous batching for vector search that make full used of GPUs under heterogeneous queries; (3) Stage-aware scheduling that preempts vector search requests between both decode and prefill tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02275v1">Understanding Down Syndrome Stereotypes in LLM-Based Personas</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      We present a case study of Persona-L, a system that leverages large language models (LLMs) and retrieval-augmented generation (RAG) to model personas of people with Down syndrome. Existing approaches to persona creation can often lead to oversimplified or stereotypical profiles of people with Down Syndrome. To that end, we built stereotype detection capabilities into Persona-L. Through interviews with caregivers and healthcare professionals (N=10), we examine how Down Syndrome stereotypes could manifest in both, content and delivery of LLMs, and interface design. Our findings show the challenges in stereotypes definition, and reveal the potential stereotype emergence from the training data, interface design, and the tone of LLM output. This highlights the need for participatory methods that capture the heterogeneity of lived experiences of people with Down Syndrome.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2412.06540v5">Sloth: scaling laws for LLM skills to predict multi-benchmark performance across families</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Scaling laws for large language models (LLMs) predict model performance based on parameters like size and training data. However, differences in training configurations and data processing across model families lead to significant variations in benchmark performance, making it difficult for a single scaling law to generalize across all LLMs. On the other hand, training family-specific scaling laws requires training models of varying sizes for every family. In this work, we propose Skills Scaling Laws (SSLaws, pronounced as Sloth), a novel scaling law that leverages publicly available benchmark data and assumes LLM performance is driven by low-dimensional latent skills, such as reasoning and instruction following. These latent skills are influenced by computational resources like model size and training tokens, but with varying efficiencies across model families. Sloth exploits correlations across benchmarks to provide more accurate and interpretable predictions while alleviating the need to train multiple LLMs per family. We present both theoretical results on parameter identification and empirical evaluations on 12 prominent benchmarks, from Open LLM Leaderboard v1/v2, demonstrating that Sloth predicts LLM performance accurately and offers insights into scaling behaviors for complex downstream tasks, increased test-time compute, and compute-optimal scaling of skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.12792v2">Bridging Human and LLM Judgments: Understanding and Narrowing the Gap</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large language models are increasingly used as judges (LLM-as-a-judge) to evaluate model outputs at scale, but their assessments often diverge systematically from human judgments. We present Bridge, a unified statistical framework that explicitly bridges human and LLM evaluations under both absolute scoring and pairwise comparison paradigms. Bridge posits a latent human preference score for each prompt-response pair and models LLM deviations as linear transformations of covariates that capture sources of discrepancies. This offers a simple and principled framework for refining LLM ratings and characterizing systematic discrepancies between humans and LLMs. We provide an efficient fitting algorithm with asymptotic guarantees for statistical inference. Using six LLM judges and two benchmarks (BigGen Bench and Chatbot Arena), Bridge achieves higher agreement with human ratings (accuracy, calibration, and KL divergence) and exposes systematic human-LLM gaps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02261v1">TradeTrap: Are LLM-based Trading Agents Truly Reliable and Faithful?</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      LLM-based trading agents are increasingly deployed in real-world financial markets to perform autonomous analysis and execution. However, their reliability and robustness under adversarial or faulty conditions remain largely unexamined, despite operating in high-risk, irreversible financial environments. We propose TradeTrap, a unified evaluation framework for systematically stress-testing both adaptive and procedural autonomous trading agents. TradeTrap targets four core components of autonomous trading agents: market intelligence, strategy formulation, portfolio and ledger handling, and trade execution, and evaluates their robustness under controlled system-level perturbations. All evaluations are conducted in a closed-loop historical backtesting setting on real US equity market data with identical initial conditions, enabling fair and reproducible comparisons across agents and attacks. Extensive experiments show that small perturbations at a single component can propagate through the agent decision loop and induce extreme concentration, runaway exposure, and large portfolio drawdowns across both agent types, demonstrating that current autonomous trading agents can be systematically misled at the system level. Our code is available at https://github.com/Yanlewen/TradeTrap.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17117v6">From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Humans organize knowledge into compact conceptual categories that balance compression with semantic richness. Large Language Models (LLMs) exhibit impressive linguistic abilities, but whether they navigate this same compression-meaning trade-off remains unclear. We apply an Information Bottleneck framework to compare human conceptual structure with embeddings from 40+ LLMs using classic categorization benchmarks. We find that LLMs broadly align with human category boundaries, yet fall short on fine-grained semantic distinctions. Unlike humans, who maintain ``inefficient'' representations that preserve contextual nuance, LLMs aggressively compress, achieving more optimal information-theoretic compression at the cost of semantic richness. Surprisingly, encoder models outperform much larger decoder models in human alignment, suggesting that understanding and generation rely on distinct representational mechanisms. Training-dynamics analysis reveals a two-phase trajectory: rapid initial concept formation followed by architectural reorganization, during which semantic processing migrates from deep to mid-network layers as the model discovers increasingly efficient, sparser encodings. These divergent strategies, where LLMs optimize for compression and humans for adaptive utility, reveal fundamental differences between artificial and natural intelligence. This highlights the need for models that preserve the conceptual ``inefficiencies'' essential for human-like understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02230v1">Benchmarking LLM Agents for Wealth-Management Workflows</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 56 pages, 8 figures, The University of Edinburgh
    </div>
    <details class="paper-abstract">
      Modern work relies on an assortment of digital collaboration tools, yet routine processes continue to suffer from human error and delay. To address this gap, this dissertation extends TheAgentCompany with a finance-focused environment and investigates whether a general purpose LLM agent can complete representative wealth-management tasks both accurately and economically. This study introduces synthetic domain data, enriches colleague simulations, and prototypes an automatic task-generation pipeline. The study aims to create and assess an evaluation set that can meaningfully measure an agent's fitness for assistant-level wealth management work. We construct a benchmark of 12 task-pairs for wealth management assistants spanning retrieval, analysis, and synthesis/communication, with explicit acceptance criteria and deterministic graders. We seeded a set of new finance-specific data and introduced a high vs. low-autonomy variant of every task. The paper concluded that agents are limited less by mathematical reasoning and more so by end-to-end workflow reliability, and meaningfully affected by autonomy level, and that incorrect evaluation of models have hindered benchmarking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02228v1">STRIDE: A Systematic Framework for Selecting AI Modalities -- Agentic AI, AI Assistants, or LLM Calls</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 10 pages, 4 Figures, 5 Tables Paper presented at NeurIPS 2025 LAW workshop: Bridging Language, Agent, and World Models
    </div>
    <details class="paper-abstract">
      The rapid shift from stateless large language models (LLMs) to autonomous, goal-driven agents raises a central question: When is agentic AI truly necessary? While agents enable multi-step reasoning, persistent memory, and tool orchestration, deploying them indiscriminately leads to higher cost, complexity, and risk. We present STRIDE (Systematic Task Reasoning Intelligence Deployment Evaluator), a framework that provides principled recommendations for selecting between three modalities: (i) direct LLM calls, (ii) guided AI assistants, and (iii) fully autonomous agentic AI. STRIDE integrates structured task decomposition, dynamism attribution, and self-reflection requirement analysis to produce an Agentic Suitability Score, ensuring that full agentic autonomy is reserved for tasks with inherent dynamism or evolving context. Evaluated across 30 real-world tasks spanning SRE, compliance, and enterprise automation, STRIDE achieved 92% accuracy in modality selection, reduced unnecessary agent deployments by 45%, and cut resource costs by 37%. Expert validation over six months in SRE and compliance domains confirmed its practical utility, with domain specialists agreeing that STRIDE effectively distinguishes between tasks requiring simple LLM calls, guided assistants, or full agentic autonomy. This work reframes agent adoption as a necessity-driven design decision, ensuring autonomy is applied only when its benefits justify the costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.08093v3">Evolution and compression in LLMs: On the emergence of human-aligned categorization</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Accepted at CogInterp: Interpreting Cognition in Deep Learning Models Workshop at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Converging evidence suggests that human systems of semantic categories achieve near-optimal compression via the Information Bottleneck (IB) complexity-accuracy tradeoff. Large language models (LLMs) are not trained for this objective, which raises the question: are LLMs capable of evolving efficient human-aligned semantic systems? To address this question, we focus on color categorization -- a key testbed of cognitive theories of categorization with uniquely rich human data -- and replicate with LLMs two influential human studies. First, we conduct an English color-naming study, showing that LLMs vary widely in their complexity and English-alignment, with larger instruction-tuned models achieving better alignment and IB-efficiency. Second, to test whether these LLMs simply mimic patterns in their training data or actually exhibit a human-like inductive bias toward IB-efficiency, we simulate cultural evolution of pseudo color-naming systems in LLMs via a method we refer to as Iterated in-Context Language Learning (IICLL). We find that akin to humans, LLMs iteratively restructure initially random systems towards greater IB-efficiency. However, only a model with strongest in-context capabilities (Gemini 2.0) is able to recapitulate the wide range of near-optimal IB-tradeoffs observed in humans, while other state-of-the-art models converge to low-complexity solutions. These findings demonstrate how human-aligned semantic categories can emerge in LLMs via the same fundamental principle that underlies semantic efficiency in humans.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14233v2">Apertus: Democratizing Open and Compliant LLMs for Global Language Environments</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      We present Apertus, a fully open suite of large language models (LLMs) designed to address two systemic shortcomings in today's open model ecosystem: data compliance and multilingual representation. Unlike many prior models that release weights without reproducible data pipelines or regard for content-owner rights, Apertus models are pretrained exclusively on openly available data, retroactively respecting `robots.txt` exclusions and filtering for non-permissive, toxic, and personally identifiable content. To mitigate risks of memorization, we adopt the Goldfish objective during pretraining, strongly suppressing verbatim recall of data while retaining downstream task performance. The Apertus models also expand multilingual coverage, training on 15T tokens from over 1800 languages, with ~40% of pretraining data allocated to non-English content. Released at 8B and 70B scales, Apertus approaches state-of-the-art results among fully open models on multilingual benchmarks, rivalling or surpassing open-weight counterparts. Beyond model weights, we release all scientific artifacts from our development cycle with a permissive license, including data preparation scripts, checkpoints, evaluation suites, and training code, enabling transparent audit and extension.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03100v1">Ensemble Privacy Defense for Knowledge-Intensive LLMs against Membership Inference Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) and Supervised Finetuning (SFT) have become the predominant paradigms for equipping Large Language Models (LLMs) with external knowledge for diverse, knowledge-intensive tasks. However, while such knowledge injection improves performance, it also exposes new attack surfaces. Membership Inference Attacks (MIAs), which aim to determine whether a given data sample was included in a model's training set, pose serious threats to privacy and trust in sensitive domains. To this end, we first systematically evaluate the vulnerability of RAG- and SFT-based LLMs to various MIAs. Then, to address the privacy risk, we further introduce a novel, model-agnostic defense framework, Ensemble Privacy Defense (EPD), which aggregates and evaluates the outputs of a knowledge-injected LLM, a base LLM, and a dedicated judge model to enhance resistance against MIAs. Comprehensive experiments show that, on average, EPD reduces MIA success by up to 27.8\% for SFT and 526.3\% for RAG compared to inference-time baseline, while maintaining answer quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02002v1">LLM-Driven Corrective Robot Operation Code Generation with Static Text-Based Simulation</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 8 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Recent advances in Large language models (LLMs) have demonstrated their promising capabilities of generating robot operation code to enable LLM-driven robots. To enhance the reliability of operation code generated by LLMs, corrective designs with feedback from the observation of executing code have been increasingly adopted in existing research. However, the code execution in these designs relies on either a physical experiment or a customized simulation environment, which limits their deployment due to the high configuration effort of the environment and the potential long execution time. In this paper, we explore the possibility of directly leveraging LLM to enable static simulation of robot operation code, and then leverage it to design a new reliable LLM-driven corrective robot operation code generation framework. Our framework configures the LLM as a static simulator with enhanced capabilities that reliably simulate robot code execution by interpreting actions, reasoning over state transitions, analyzing execution outcomes, and generating se- mantic observations that accurately capture trajectory dynamics. To validate the performance of our framework, we performed experiments on various operation tasks for different robots, including UAVs and small ground vehicles. The experiment results not only demonstrated the high accuracy of our static text-based simulation but also the reliable code generation of our LLM-driven corrective framework, which achieves a comparable performance with state-of-the-art research while does not rely on dynamic code execution using physical experiments or simulators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01992v1">LLM CHESS: Benchmarking Reasoning and Instruction-Following in LLMs through Chess</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      We introduce LLM CHESS, an evaluation framework designed to probe the generalization of reasoning and instruction-following abilities in large language models (LLMs) through extended agentic interaction in the domain of chess. We rank over 50 open and closed source models by playing against a random opponent using a range of behavioral metrics, including win and loss rates, move quality, move legality, hallucinated actions, and game duration. For a subset of top reasoning models, we derive an Elo estimate by playing against a chess engine with variably configured skill, which allows for comparisons between models in an easily understandable way. Despite the simplicity of the instruction-following task and the weakness of the opponent, many state-of-the-art models struggle to complete games or achieve consistent wins. Similar to other benchmarks on complex reasoning tasks, our experiments reveal a clear separation between reasoning and non-reasoning models. However, unlike existing static benchmarks, the stochastic and dynamic nature of LLM CHESS uniquely reduces overfitting and memorization while preventing benchmark saturation, proving difficult even for top reasoning models. To support future work on evaluating reasoning and instruction-following in LLMs, we release our experimental framework, a public leaderboard, and a dataset of associated games.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.12286v4">The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly capable and widely adopted, benchmarks play a central role in assessing their practical utility. For example, SWE-Bench Verified has emerged as a critical benchmark for evaluating LLMs' software engineering abilities, particularly their aptitude for resolving real-world GitHub issues. Recent LLMs show impressive performance on SWE-Bench, leading to optimism about their capacity for complex coding tasks. However, current evaluation protocols may overstate these models' true capabilities. It is crucial to distinguish LLMs' generalizable problem-solving ability and other learned artifacts. In this work, we introduce two diagnostic tasks: file path identification from issue descriptions alone and ground truth function reproduction with only the current file context and issue description to probe models' underlying knowledge. We present empirical evidence that performance gains on SWE-Bench-Verified may be partially driven by memorization rather than genuine problem-solving. We show that state-of-the-art models achieve up to 76% accuracy in identifying buggy file paths using only issue descriptions, without access to repository structure. This performance is merely up to 53% on tasks from repositories not included in SWE-Bench, pointing to possible data contamination or memorization. Similar patterns are also observed for the function reproduction task, where the verbatim similarity is much higher on SWE-Bench Verified than on other similar coding benchmarks (up to 35% consecutive 5-gram accuracy on SWE-Bench Verified and Full, but only up to 18% for tasks in other benchmarks). These findings raise concerns about the validity of existing results and underscore the need for more robust, contamination-resistant benchmarks to reliably evaluate LLMs' coding abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01925v1">Rectifying LLM Thought from Lens of Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have been driven by their emergent reasoning capabilities, particularly through long chain-of-thought (CoT) prompting, which enables thorough exploration and deliberation. Despite these advances, long-CoT LLMs often exhibit suboptimal reasoning behaviors, such as overthinking and excessively protracted reasoning chains, which can impair performance. In this paper, we analyze reasoning processes through an optimization lens, framing CoT as a gradient descent procedure where each reasoning step constitutes an update toward problem resolution. Building on this perspective, we introduce RePro (Rectifying Process-level Reward), a novel approach to refine LLM reasoning during post-training. RePro defines a surrogate objective function to assess the optimization process underlying CoT, utilizing a dual scoring mechanism to quantify its intensity and stability. These scores are aggregated into a composite process-level reward, seamlessly integrated into reinforcement learning with verifiable rewards (RLVR) pipelines to optimize LLMs. Extensive experiments across multiple reinforcement learning algorithms and diverse LLMs, evaluated on benchmarks spanning mathematics, science, and coding, demonstrate that RePro consistently enhances reasoning performance and mitigates suboptimal reasoning behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01909v1">Latent Debate: A Surrogate Framework for Interpreting LLM Thinking</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Understanding the internal thinking process of Large Language Models (LLMs) and the cause of hallucinations remains a key challenge. To this end, we introduce latent debate, a novel framework for interpreting model predictions through the lens of implicit internal arguments. Unlike the current work of self-consistency and multi-agent debate, which relies on explicit debates among multiple answers or multiple models, latent debate captures the hidden supporting and attacking signals that arise within a single model during a single inference. We first present a model- and task-agnostic conceptual framework, and then instantiate it symbolically to approximate the thinking process of LLMs on True/False prediction tasks. Empirical studies demonstrate that latent debate is a faithful structured surrogate model that has highly consistent predictions with the original LLM. Beyond interpretability, we demonstrate that latent debate provides a strong baseline for hallucination detection. Further analysis reveals strong correlations between hallucinations and debate patterns, such as a high degree of latent debates in the middle layers is linked to a higher risk of hallucinations. These findings position latent debate as a potential framework for understanding internal mechanisms of LLMs, especially for scenarios where internal (dis)agreements appear during the inference steps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20075v4">LLMs can hide text in other text of the same length</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 21 pages, main paper 9 pages
    </div>
    <details class="paper-abstract">
      A meaningful text can be hidden inside another, completely different yet still coherent and plausible, text of the same length. For example, a tweet containing a harsh political critique could be embedded in a tweet that celebrates the same political leader, or an ordinary product review could conceal a secret manuscript. This uncanny state of affairs is now possible thanks to Large Language Models, and in this paper we present Calgacus, a simple and efficient protocol to achieve it. We show that even modest 8-billion-parameter open-source LLMs are sufficient to obtain high-quality results, and a message as long as this abstract can be encoded and decoded locally on a laptop in seconds. The existence of such a protocol demonstrates a radical decoupling of text from authorial intent, further eroding trust in written communication, already shaken by the rise of LLM chatbots. We illustrate this with a concrete scenario: a company could covertly deploy an unfiltered LLM by encoding its answers within the compliant responses of a safe model. This possibility raises urgent questions for AI safety and challenges our understanding of what it means for a Large Language Model to know something.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01830v1">OpenREAD: Reinforced Open-Ended Reasoing for End-to-End Autonomous Driving with LLM-as-Critic</a></div>
    <div class="paper-meta">
      📅 2025-12-01
    </div>
    <details class="paper-abstract">
      Recently, two-stage fine-tuning strategies, e.g., acquiring essential driving knowledge through supervised fine-tuning (SFT) and further enhancing decision-making and planning via reinforcement fine-tuning (RFT), have shown strong potential in advancing the knowledge-driven autonomous driving (AD) paradigm. However, the learning nature of SFT still limits the generalization of reasoning, thereby constraining the full potential of driving performance. Meanwhile, current RFT approaches are primarily applied to downstream tasks, since scene understanding is an open-ended problem where corresponding rewards are difficult to quantify. To address these limitations, we propose OpenREAD, an OPEN-ended REasoning reinforced vision-language model (VLM)-based autonomous driving (AD) framework that enables end-to-end RFT across the full spectrum from high-level reasoning to low-level trajectory planning. Specifically, we begin by constructing large-scale Chain-of-Thought (CoT) annotations on open-source driving-related knowledge datasets, and employ the powerful Qwen3 large language model (LLM) as the critic in RFT to quantify reasoning quality for open-ended questions during reward modeling. Extensive experiments confirm that joint end-to-end RFT yields substantial improvements in both upstream and downstream tasks, enabling OpenREAD to achieve state-of-the-art performance on reasoning and planning benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01786v1">Who Judges the Judge? LLM Jury-on-Demand: Building Trustworthy LLM Evaluation Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 66 pages, 22 figures, 37 tables
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become integrated into high-stakes domains, there is a growing need for evaluation methods that are both scalable for real-time deployment and reliable for critical decision-making. While human evaluation is reliable, it is slow and costly. Single LLM judges are biased, and static juries lack adaptability. To overcome these limitations, we propose LLM Jury-on-Demand - a dynamic, learning-based framework for scalable and context-aware evaluation. Our method trains a set of reliability predictors to assess when LLM judges will agree with human experts, leveraging token distributions, embeddings, and structural input features. This enables a fully adaptive evaluation where, for each data point, an optimal jury of the most reliable judges is dynamically selected, and their scores are aggregated using their reliability as weights. Experiments on summarization and RAG benchmarks show that our dynamic jury system achieves significantly higher correlation with human judgment than both single-judge and static-jury baselines. These results highlight the promise of adaptive, learning-based juries for building scalable, more reliable and trustworthy evaluation systems for modern LLMs in high-stakes domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.13189v2">Multimodal "Puppeteer": Exploring Robot Teleoperation Via Virtual Counterpart with LLM-Driven Voice and Gesture Interaction in Augmented Reality</a></div>
    <div class="paper-meta">
      📅 2025-12-01
      | 💬 This work is under peer review
    </div>
    <details class="paper-abstract">
      The integration of robotics and augmented reality (AR) offers promising opportunities to enhance human-robot interaction (HRI) by making teleoperation more transparent, spatially grounded, and intuitive. We present a head-mounted AR "puppeteer" framework in which users control a physical robot via interacting with its virtual counterpart robot using large language model (LLM)-driven voice commands and hand-gesture interaction on the Meta Quest 3. In a within-subject user study with 42 participants performing an AR-based robotic pick-and-place pattern-matching task, we compare two interaction conditions: gesture-only (GO) and combined voice+gesture (VG). Our results show that GO currently provides more reliable and efficient control for this time-critical task, while VG introduces additional flexibility but also latency and recognition issues that can increase workload. We further explore how prior robotics experience shapes participants' perceptions of each modality. Based on these findings, we distill a set of evidence-based design guidelines for AR puppeteer metaphoric robot teleoperation, implicating multimodality as an adaptive strategy that must balance efficiency, robustness, and user expertise rather than assuming that additional modalities are universally beneficial. Our work contributes empirical insights into how multimodal (voice+gesture) interaction influences task efficiency, usability, and user experience in AR-based HRI.
    </details>
</div>
