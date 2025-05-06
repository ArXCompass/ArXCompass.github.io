# llm - 2025_04

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12408v1">A Human-AI Comparative Analysis of Prompt Sensitivity in LLM-Based Relevance Judgment</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate relevance judgments for information retrieval (IR) tasks, often demonstrating agreement with human labels that approaches inter-human agreement. To assess the robustness and reliability of LLM-based relevance judgments, we systematically investigate impact of prompt sensitivity on the task. We collected prompts for relevance assessment from 15 human experts and 15 LLMs across three tasks~ -- ~binary, graded, and pairwise~ -- ~yielding 90 prompts in total. After filtering out unusable prompts from three humans and three LLMs, we employed the remaining 72 prompts with three different LLMs as judges to label document/query pairs from two TREC Deep Learning Datasets (2020 and 2021). We compare LLM-generated labels with TREC official human labels using Cohen's $\kappa$ and pairwise agreement measures. In addition to investigating the impact of prompt variations on agreement with human labels, we compare human- and LLM-generated prompts and analyze differences among different LLMs as judges. We also compare human- and LLM-generated prompts with the standard UMBRELA prompt used for relevance assessment by Bing and TREC 2024 Retrieval Augmented Generation (RAG) Track. To support future research in LLM-based evaluation, we release all data and prompts at https://github.com/Narabzad/prompt-sensitivity-relevance-judgements/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12397v1">Activated LoRA: Fine-tuned LLMs for Intrinsics</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 arXiv admin note: text overlap with arXiv:2504.11704
    </div>
    <details class="paper-abstract">
      Low-Rank Adaptation (LoRA) has emerged as a highly efficient framework for finetuning the weights of large foundation models, and has become the go-to method for data-driven customization of LLMs. Despite the promise of highly customized behaviors and capabilities, switching between relevant LoRAs in a multiturn setting is highly inefficient, as the key-value (KV) cache of the entire turn history must be recomputed with the LoRA weights before generation can begin. To address this problem, we propose Activated LoRA (aLoRA), which modifies the LoRA framework to only adapt weights for the tokens in the sequence \emph{after} the aLoRA is invoked. This change crucially allows aLoRA to accept the base model's KV cache of the input string, meaning that aLoRA can be instantly activated whenever needed in a chain without recomputing the cache. This enables building what we call \emph{intrinsics}, i.e. highly specialized models invoked to perform well-defined operations on portions of an input chain or conversation that otherwise uses the base model by default. We use aLoRA to train a set of intrinsics models, demonstrating competitive accuracy with standard LoRA while achieving significant inference benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07447v3">Optimizing LLM Inference for Database Systems: Cost-Aware Scheduling for Concurrent Requests</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      LLMs are increasingly used inside database systems and in database applications for better complexity management and decision-making, where LLM inferences require significant GPU costs. LLM inference systems, however, are slow compared to database systems, limiting the expansion of the use of LLMs inside database systems. This paper first analyzes the LLM inference performance and focuses on a data management issue in LLM inference. We reveal that the root of the problem is the lack of an adequate resource cost model and optimization strategy when executing multiple concurrent inference requests. We adapt classic database multi-query optimization techniques by introducing cost models for concurrent inference requests and new scheduling strategies to optimize the use of memory resources by concurrent requests, thereby substantially improving performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13209v1">On the Feasibility of Using MultiModal LLMs to Execute AR Social Engineering Attacks</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Augmented Reality (AR) and Multimodal Large Language Models (LLMs) are rapidly evolving, providing unprecedented capabilities for human-computer interaction. However, their integration introduces a new attack surface for social engineering. In this paper, we systematically investigate the feasibility of orchestrating AR-driven Social Engineering attacks using Multimodal LLM for the first time, via our proposed SEAR framework, which operates through three key phases: (1) AR-based social context synthesis, which fuses Multimodal inputs (visual, auditory and environmental cues); (2) role-based Multimodal RAG (Retrieval-Augmented Generation), which dynamically retrieves and integrates contextual data while preserving character differentiation; and (3) ReInteract social engineering agents, which execute adaptive multiphase attack strategies through inference interaction loops. To verify SEAR, we conducted an IRB-approved study with 60 participants in three experimental configurations (unassisted, AR+LLM, and full SEAR pipeline) compiling a new dataset of 180 annotated conversations in simulated social scenarios. Our results show that SEAR is highly effective at eliciting high-risk behaviors (e.g., 93.3% of participants susceptible to email phishing). The framework was particularly effective in building trust, with 85% of targets willing to accept an attacker's call after an interaction. Also, we identified notable limitations such as ``occasionally artificial'' due to perceived authenticity gaps. This work provides proof-of-concept for AR-LLM driven social engineering attacks and insights for developing defensive countermeasures against next-generation augmented reality threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02108v2">ReGenesis: LLMs can Grow into Reasoning Generalists via Self-Improvement</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Post-training Large Language Models (LLMs) with explicit reasoning trajectories can enhance their reasoning abilities. However, acquiring such high-quality trajectory data typically demands meticulous supervision from humans or superior models, which can be either expensive or license-constrained. In this paper, we explore how far an LLM can improve its reasoning by self-synthesizing reasoning paths as training data without any additional supervision. Existing self-synthesizing methods, such as STaR, suffer from poor generalization to out-of-domain (OOD) reasoning tasks. We hypothesize it is due to that their self-synthesized reasoning paths are too task-specific, lacking general task-agnostic reasoning guidance. To address this, we propose Reasoning Generalist via Self-Improvement (ReGenesis), a method to self-synthesize reasoning paths as post-training data by progressing from abstract to concrete. More specifically, ReGenesis self-synthesizes reasoning paths by converting general reasoning guidelines into task-specific ones, generating reasoning structures, and subsequently transforming these structures into reasoning paths, without the need for human-designed task-specific examples used in existing methods. We show that ReGenesis achieves superior performance on all in-domain and OOD settings tested compared to existing methods. For six OOD tasks specifically, while previous methods exhibited an average performance decrease of approximately 4.6% after post training, ReGenesis delivers around 6.1% performance improvement. We also conduct in-depth analysis of our framework and show ReGenesis is effective across various LLMs and design choices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12268v1">HLS-Eval: A Benchmark and Framework for Evaluating LLMs on High-Level Synthesis Design Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      The rapid scaling of large language model (LLM) training and inference has driven their adoption in semiconductor design across academia and industry. While most prior work evaluates LLMs on hardware description language (HDL) tasks, particularly Verilog, designers are increasingly using high-level synthesis (HLS) to build domain-specific accelerators and complex hardware systems. However, benchmarks and tooling to comprehensively evaluate LLMs for HLS design tasks remain scarce. To address this, we introduce HLS-Eval, the first complete benchmark and evaluation framework for LLM-driven HLS design. HLS-Eval targets two core tasks: (1) generating HLS code from natural language descriptions, and (2) performing HLS-specific code edits to optimize performance and hardware efficiency. The benchmark includes 94 unique designs drawn from standard HLS benchmarks and novel sources. Each case is prepared via a semi-automated flow that produces a natural language description and a paired testbench for C-simulation and synthesis validation, ensuring each task is "LLM-ready." Beyond the benchmark, HLS-Eval offers a modular Python framework for automated, parallel evaluation of both local and hosted LLMs. It includes a parallel evaluation engine, direct HLS tool integration, and abstractions for to support different LLM interaction paradigms, enabling rapid prototyping of new benchmarks, tasks, and LLM methods. We demonstrate HLS-Eval through baseline evaluations of open-source LLMs on Vitis HLS, measuring outputs across four key metrics - parseability, compilability, runnability, and synthesizability - reflecting the iterative HLS design cycle. We also report pass@k metrics, establishing clear baselines and reusable infrastructure for the broader LLM-for-hardware community. All benchmarks, framework code, and results are open-sourced at https://github.com/stefanpie/hls-eval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12250v1">AnomalyGen: An Automated Semantic Log Sequence Generation Framework with LLM for Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      The scarcity of high-quality public log datasets has become a critical bottleneck in advancing log-based anomaly detection techniques. Current datasets exhibit three fundamental limitations: (1) incomplete event coverage, (2) artificial patterns introduced by static analysis-based generation frameworks, and (3) insufficient semantic awareness. To address these challenges, we present AnomalyGen, the first automated log synthesis framework specifically designed for anomaly detection. Our framework introduces a novel four-phase architecture that integrates enhanced program analysis with Chain-of-Thought reasoning (CoT reasoning), enabling iterative log generation and anomaly annotation without requiring physical system execution. Evaluations on Hadoop and HDFS distributed systems demonstrate that AnomalyGen achieves substantially broader log event coverage (38-95 times improvement over existing datasets) while producing more operationally realistic log sequences compared to static analysis-based approaches. When augmenting benchmark datasets with synthesized logs, we observe maximum F1-score improvements of 3.7% (average 1.8% improvement across three state-of-the-art anomaly detection models). This work not only establishes a high-quality benchmarking resource for automated log analysis but also pioneers a new paradigm for applying large language models (LLMs) in software engineering workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12185v1">SALAD: Improving Robustness and Generalization through Contrastive Learning with Structure-Aware and LLM-Driven Augmented Data</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 Accepted to NAACL 2025 main. 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      In various natural language processing (NLP) tasks, fine-tuning Pre-trained Language Models (PLMs) often leads to the issue of spurious correlations, which negatively impacts performance, particularly when dealing with out-of-distribution data. To address this problem, we propose SALAD}(Structure Aware and LLM-driven Augmented Data), a novel approach designed to enhance model robustness and generalization by generating structure-aware and counterfactually augmented data for contrastive learning. Our method leverages a tagging-based approach to generate structure-aware positive samples and utilizes large language models (LLMs) to generate counterfactual negative samples with diverse sentence patterns. By applying contrastive learning, SALAD enables the model to focus on learning the structural relationships between key sentence components while minimizing reliance on spurious correlations. We validate our approach through experiments on three tasks: Sentiment Classification, Sexism Detection, and Natural Language Inference. The results demonstrate that SALAD not only improves model robustness and performance across different environments but also enhances generalization to out-of-distribution datasets and cross-domain scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11168v2">Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 12 pages, 5 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04285v3">Separate Source Channel Coding Is Still What You Need: An LLM-based Rethinking</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Along with the proliferating research interest in Semantic Communication (SemCom), Joint Source Channel Coding (JSCC) has dominated the attention due to the widely assumed existence in efficiently delivering information semantics. %has emerged as a pivotal area of research, aiming to enhance the efficiency and reliability of information transmission through deep learning-based methods. Nevertheless, this paper challenges the conventional JSCC paradigm, and advocates for adoption of Separate Source Channel Coding (SSCC) to enjoy the underlying more degree of freedom for optimization. We demonstrate that SSCC, after leveraging the strengths of Large Language Model (LLM) for source coding and Error Correction Code Transformer (ECCT) complemented for channel decoding, offers superior performance over JSCC. Our proposed framework also effectively highlights the compatibility challenges between SemCom approaches and digital communication systems, particularly concerning the resource costs associated with the transmission of high precision floating point numbers. Through comprehensive evaluations, we establish that empowered by LLM-based compression and ECCT-enhanced error correction, SSCC remains a viable and effective solution for modern communication systems. In other words, separate source and channel coding is still what we need!
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09063v2">Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 Trigger Warning: the appendix contains LLM-generated text with violence and harassment
    </div>
    <details class="paper-abstract">
      Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16514v3">VeriMind: Agentic LLM for Automated Verilog Generation with a Novel Evaluation Metric</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Designing Verilog modules requires meticulous attention to correctness, efficiency, and adherence to design specifications. However, manually writing Verilog code remains a complex and time-consuming task that demands both expert knowledge and iterative refinement. Leveraging recent advancements in large language models (LLMs) and their structured text generation capabilities, we propose VeriMind, an agentic LLM framework for Verilog code generation that significantly automates and optimizes the synthesis process. Unlike traditional LLM-based code generators, VeriMind employs a structured reasoning approach: given a user-provided prompt describing design requirements, the system first formulates a detailed train of thought before the final Verilog code is generated. This multi-step methodology enhances interpretability, accuracy, and adaptability in hardware design. In addition, we introduce a novel evaluation metric-pass@ARC-which combines the conventional pass@k measure with Average Refinement Cycles (ARC) to capture both success rate and the efficiency of iterative refinement. Experimental results on diverse hardware design tasks demonstrated that our approach achieved up to $8.3\%$ improvement on pass@k metric and $8.1\%$ on pass@ARC metric. These findings underscore the transformative potential of agentic LLMs in automated hardware design, RTL development, and digital system synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08525v3">Task Memory Engine (TME): A Structured Memory Framework with Graph-Aware Extensions for Multi-Step LLM Agent Tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 14 pages, 5 figures. Preprint prepared for future submission. Includes implementation and token-efficiency analysis. Code at https://github.com/biubiutomato/TME-Agent
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as autonomous agents for multi-step tasks. However, most existing frameworks fail to maintain a structured understanding of the task state, often relying on linear prompt concatenation or shallow memory buffers. This leads to brittle performance, frequent hallucinations, and poor long-range coherence. In this work, we propose the Task Memory Engine (TME), a lightweight and structured memory module that tracks task execution using a hierarchical Task Memory Tree (TMT). Each node in the tree corresponds to a task step, storing relevant input, output, status, and sub-task relationships. We introduce a prompt synthesis method that dynamically generates LLM prompts based on the active node path, significantly improving execution consistency and contextual grounding. Through case studies and comparative experiments on multi-step agent tasks, we demonstrate that TME leads to better task completion accuracy and more interpretable behavior with minimal implementation overhead. A reference implementation of the core TME components is available at https://github.com/biubiutomato/TME-Agent, including basic examples and structured memory integration. While the current implementation uses a tree-based structure, TME is designed to be graph-aware, supporting reusable substeps, converging task paths, and shared dependencies. This lays the groundwork for future DAG-based memory architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10185v2">LLM Unlearning Reveals a Stronger-Than-Expected Coreset Effect in Current Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Large language model unlearning has become a critical challenge in ensuring safety and controlled model behavior by removing undesired data-model influences from the pretrained model while preserving general utility. Significant recent efforts have been dedicated to developing LLM unlearning benchmarks such as WMDP (Weapons of Mass Destruction Proxy) and MUSE (Machine Unlearning Six-way Evaluation), facilitating standardized unlearning performance assessment and method comparison. Despite their usefulness, we uncover for the first time a novel coreset effect within these benchmarks. Specifically, we find that LLM unlearning achieved with the original (full) forget set can be effectively maintained using a significantly smaller subset (functioning as a "coreset"), e.g., as little as 5% of the forget set, even when selected at random. This suggests that LLM unlearning in these benchmarks can be performed surprisingly easily, even in an extremely low-data regime. We demonstrate that this coreset effect remains strong, regardless of the LLM unlearning method used, such as NPO (Negative Preference Optimization) and RMU (Representation Misdirection Unlearning), the popular ones in these benchmarks. The surprisingly strong coreset effect is also robust across various data selection methods, ranging from random selection to more sophisticated heuristic approaches. We explain the coreset effect in LLM unlearning through a keyword-based perspective, showing that keywords extracted from the forget set alone contribute significantly to unlearning effectiveness and indicating that current unlearning is driven by a compact set of high-impact tokens rather than the entire dataset. We further justify the faithfulness of coreset-unlearned models along additional dimensions, such as mode connectivity and robustness to jailbreaking attacks. Codes are available at https://github.com/OPTML-Group/MU-Coreset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12110v1">Towards LLM Agents for Earth Observation</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 36 pages
    </div>
    <details class="paper-abstract">
      Earth Observation (EO) provides critical planetary data for environmental monitoring, disaster management, climate science, and other scientific domains. Here we ask: Are AI systems ready for reliable Earth Observation? We introduce \datasetnamenospace, a benchmark of 140 yes/no questions from NASA Earth Observatory articles across 13 topics and 17 satellite sensors. Using Google Earth Engine API as a tool, LLM agents can only achieve an accuracy of 33% because the code fails to run over 58% of the time. We improve the failure rate for open models by fine-tuning synthetic data, allowing much smaller models (Llama-3.1-8B) to achieve comparable accuracy to much larger ones (e.g., DeepSeek-R1). Taken together, our findings identify significant challenges to be solved before AI agents can automate earth observation, and suggest paths forward. The project page is available at https://iandrover.github.io/UnivEarth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12108v1">Entropy-Guided Watermarking for LLMs: A Test-Time Framework for Robust and Traceable Text Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      The rapid development of Large Language Models (LLMs) has intensified concerns about content traceability and potential misuse. Existing watermarking schemes for sampled text often face trade-offs between maintaining text quality and ensuring robust detection against various attacks. To address these issues, we propose a novel watermarking scheme that improves both detectability and text quality by introducing a cumulative watermark entropy threshold. Our approach is compatible with and generalizes existing sampling functions, enhancing adaptability. Experimental results across multiple LLMs show that our scheme significantly outperforms existing methods, achieving over 80\% improvements on widely-used datasets, e.g., MATH and GSM8K, while maintaining high detection accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12098v1">Gauging Overprecision in LLMs: An Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Recently, overconfidence in large language models (LLMs) has garnered considerable attention due to its fundamental importance in quantifying the trustworthiness of LLM generation. However, existing approaches prompt the \textit{black box LLMs} to produce their confidence (\textit{verbalized confidence}), which can be subject to many biases and hallucinations. Inspired by a different aspect of overconfidence in cognitive science called \textit{overprecision}, we designed a framework for its study in black box LLMs. This framework contains three main phases: 1) generation, 2) refinement and 3) evaluation. In the generation phase we prompt the LLM to generate answers to numerical questions in the form of intervals with a certain level of confidence. This confidence level is imposed in the prompt and not required for the LLM to generate as in previous approaches. We use various prompting techniques and use the same prompt multiple times to gauge the effects of randomness in the generation process. In the refinement phase, answers from the previous phase are refined to generate better answers. The LLM answers are evaluated and studied in the evaluation phase to understand its internal workings. This study allowed us to gain various insights into LLM overprecision: 1) LLMs are highly uncalibrated for numerical tasks 2) {\color{blue}there is no correlation between the length of the interval and the imposed confidence level, which can be symptomatic of a a) lack of understanding of the concept of confidence or b) inability to adjust self-confidence by following instructions}, {\color{blue}3)} LLM numerical precision differs depending on the task, scale of answer and prompting technique {\color{blue}4) Refinement of answers doesn't improve precision in most cases}. We believe this study offers new perspectives on LLM overconfidence and serves as a strong baseline for overprecision in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12034v1">OpDiffer: LLM-Assisted Opcode-Level Differential Testing of Ethereum Virtual Machine</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 To appear in ISSTA'25
    </div>
    <details class="paper-abstract">
      As Ethereum continues to thrive, the Ethereum Virtual Machine (EVM) has become the cornerstone powering tens of millions of active smart contracts. Intuitively, security issues in EVMs could lead to inconsistent behaviors among smart contracts or even denial-of-service of the entire blockchain network. However, to the best of our knowledge, only a limited number of studies focus on the security of EVMs. Moreover, they suffer from 1) insufficient test input diversity and invalid semantics; and 2) the inability to automatically identify bugs and locate root causes. To bridge this gap, we propose OpDiffer, a differential testing framework for EVM, which takes advantage of LLMs and static analysis methods to address the above two limitations. We conducted the largest-scale evaluation, covering nine EVMs and uncovering 26 previously unknown bugs, 22 of which have been confirmed by developers and three have been assigned CNVD IDs. Compared to state-of-the-art baselines, OpDiffer can improve code coverage by at most 71.06%, 148.40% and 655.56%, respectively. Through an analysis of real-world deployed Ethereum contracts, we estimate that 7.21% of the contracts could trigger our identified EVM bugs under certain environmental settings, potentially resulting in severe negative impact on the Ethereum ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11972v1">LLM-as-a-Judge: Reassessing the Performance of LLMs in Extractive QA</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 17 pages; code and data are available at https://github.com/Alab-NII/llm-judge-extract-qa
    </div>
    <details class="paper-abstract">
      Extractive reading comprehension question answering (QA) datasets are typically evaluated using Exact Match (EM) and F1-score, but these metrics often fail to fully capture model performance. With the success of large language models (LLMs), they have been employed in various tasks, including serving as judges (LLM-as-a-judge). In this paper, we reassess the performance of QA models using LLM-as-a-judge across four reading comprehension QA datasets. We examine different families of LLMs and various answer types to evaluate the effectiveness of LLM-as-a-judge in these tasks. Our results show that LLM-as-a-judge is highly correlated with human judgments and can replace traditional EM/F1 metrics. By using LLM-as-a-judge, the correlation with human judgments improves significantly, from 0.17 (EM) and 0.36 (F1-score) to 0.85. These findings confirm that EM and F1 metrics underestimate the true performance of the QA models. While LLM-as-a-judge is not perfect for more difficult answer types (e.g., job), it still outperforms EM/F1, and we observe no bias issues, such as self-preference, when the same model is used for both the QA and judgment tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.03104v3">ChatTS: Aligning Time Series with LLMs via Synthetic Data for Enhanced Understanding and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 accepted by VLDB' 25
    </div>
    <details class="paper-abstract">
      Understanding time series is crucial for its application in real-world scenarios. Recently, large language models (LLMs) have been increasingly applied to time series tasks, leveraging their strong language capabilities to enhance various applications. However, research on multimodal LLMs (MLLMs) for time series understanding and reasoning remains limited, primarily due to the scarcity of high-quality datasets that align time series with textual information. This paper introduces ChatTS, a novel MLLM designed for time series analysis. ChatTS treats time series as a modality, similar to how vision MLLMs process images, enabling it to perform both understanding and reasoning with time series. To address the scarcity of training data, we propose an attribute-based method for generating synthetic time series with detailed attribute descriptions. We further introduce Time Series Evol-Instruct, a novel approach that generates diverse time series Q&As, enhancing the model's reasoning capabilities. To the best of our knowledge, ChatTS is the first TS-MLLM that takes multivariate time series as input for understanding and reasoning, which is fine-tuned exclusively on synthetic datasets. We evaluate its performance using benchmark datasets with real-world data, including six alignment tasks and four reasoning tasks. Our results show that ChatTS significantly outperforms existing vision-based MLLMs (e.g., GPT-4o) and text/agent-based LLMs, achieving a 46.0% improvement in alignment tasks and a 25.8% improvement in reasoning tasks. We have open-sourced the source code, model checkpoint and datasets at https://github.com/NetManAIOps/ChatTS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11934v1">An LLM-as-a-judge Approach for Scalable Gender-Neutral Translation Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 Accepted at GITT 2025
    </div>
    <details class="paper-abstract">
      Gender-neutral translation (GNT) aims to avoid expressing the gender of human referents when the source text lacks explicit cues about the gender of those referents. Evaluating GNT automatically is particularly challenging, with current solutions being limited to monolingual classifiers. Such solutions are not ideal because they do not factor in the source sentence and require dedicated data and fine-tuning to scale to new languages. In this work, we address such limitations by investigating the use of large language models (LLMs) as evaluators of GNT. Specifically, we explore two prompting approaches: one in which LLMs generate sentence-level assessments only, and another, akin to a chain-of-thought approach, where they first produce detailed phrase-level annotations before a sentence-level judgment. Through extensive experiments on multiple languages with five models, both open and proprietary, we show that LLMs can serve as evaluators of GNT. Moreover, we find that prompting for phrase-level annotations before sentence-level assessments consistently improves the accuracy of all models, providing a better and more scalable alternative to current solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11919v1">Rethinking the Generation of High-Quality CoT Data from the Perspective of LLM-Adaptive Question Difficulty Grading</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Recently, DeepSeek-R1 (671B) (DeepSeek-AIet al., 2025) has demonstrated its excellent reasoning ability in complex tasks and has publiclyshared its methodology. This provides potentially high-quality chain-of-thought (CoT) data for stimulating the reasoning abilities of small-sized large language models (LLMs). To generate high-quality CoT data for different LLMs, we seek an efficient method for generating high-quality CoT data with LLM-Adaptive questiondifficulty levels. First, we grade the difficulty of the questions according to the reasoning ability of the LLMs themselves and construct a LLM-Adaptive question database. Second, we sample the problem database based on a distribution of difficulty levels of the questions and then use DeepSeek-R1 (671B) (DeepSeek-AI et al., 2025) to generate the corresponding high-quality CoT data with correct answers. Thanks to the construction of CoT data with LLM-Adaptive difficulty levels, we have significantly reduced the cost of data generation and enhanced the efficiency of model supervised fine-tuning (SFT). Finally, we have validated the effectiveness and generalizability of the proposed method in the fields of complex mathematical competitions and code generation tasks. Notably, with only 2k high-quality mathematical CoT data, our ZMath-32B surpasses DeepSeek-Distill-32B in math reasoning task. Similarly, with only 2k high-quality code CoT data, our ZCode-32B surpasses DeepSeek-Distill-32B in code reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11889v1">Rethinking LLM-Based Recommendations: A Query Generation-Based, Training-Free Approach</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Existing large language model LLM-based recommendation methods face several challenges, including inefficiency in handling large candidate pools, sensitivity to item order within prompts ("lost in the middle" phenomenon) poor scalability, and unrealistic evaluation due to random negative sampling. To address these issues, we propose a Query-to-Recommendation approach that leverages LLMs to generate personalized queries for retrieving relevant items from the entire candidate pool, eliminating the need for candidate pre-selection. This method can be integrated into an ID-based recommendation system without additional training, enhances recommendation performance and diversity through LLMs' world knowledge, and performs well even for less popular item groups. Experiments on three datasets show up to 57 percent improvement, with an average gain of 31 percent, demonstrating strong zero-shot performance and further gains when ensembled with existing models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.02644v3">Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 27 different types of attack/defense methods, and 7 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, 4 mixed attacks, and 11 corresponding defenses across 13 LLM backbones. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. We also introduce a new metric to evaluate the agents' capability to balance utility and security. Our code can be found at https://github.com/agiresearch/ASB.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11833v1">Could Thinking Multilingually Empower LLM Reasoning?</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Previous work indicates that large language models exhibit a significant "English bias", i.e. they often perform better when tasks are presented in English. Interestingly, we have observed that using certain other languages in reasoning tasks can yield better performance than English. However, this phenomenon remains under-explored. In this paper, we explore the upper bound of harnessing multilingualism in reasoning tasks, suggesting that multilingual reasoning promises significantly (by nearly 10 Acc@$k$ points) and robustly (tolerance for variations in translation quality and language choice) higher upper bounds than English-only reasoning. Besides analyzing the reason behind the upper bound and challenges in reaching it, we also find that common answer selection methods cannot achieve this upper bound, due to their limitations and biases. These insights could pave the way for future research aimed at fully harnessing the potential of multilingual reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04820v3">Natural Language Outlines for Code: Literate Programming in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 Accepted to FSE'25 Industry Track
    </div>
    <details class="paper-abstract">
      We propose using natural language outlines as a novel modality and interaction surface for providing AI assistance to developers throughout the software development process. An NL outline for a code function comprises multiple statements written in concise prose, which partition the code and summarize its main ideas in the style of literate programming. Crucially, we find that modern LLMs can generate accurate and high-quality NL outlines in practice. Moreover, NL outlines enable a bidirectional sync between code and NL: a developer can change one and the LLM automatically updates the other. We discuss many use cases for NL outlines: they can accelerate understanding and navigation of code and diffs, simplify code maintenance, augment code search, steer code generation, and more. We then propose and compare multiple LLM prompting techniques for generating outlines and ask professional developers to judge outline quality. Finally, we present two case studies applying NL outlines toward code review and malware detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11816v1">Cost-Efficient LLM Serving in the Cloud: VM Selection with KV Cache Offloading</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      LLM inference is essential for applications like text summarization, translation, and data analysis, but the high cost of GPU instances from Cloud Service Providers (CSPs) like AWS is a major burden. This paper proposes InferSave, a cost-efficient VM selection framework for cloud based LLM inference. InferSave optimizes KV cache offloading based on Service Level Objectives (SLOs) and workload charac teristics, estimating GPU memory needs, and recommending cost-effective VM instances. Additionally, the Compute Time Calibration Function (CTCF) improves instance selection accuracy by adjusting for discrepancies between theoretical and actual GPU performance. Experiments on AWS GPU instances show that selecting lower-cost instances without KV cache offloading improves cost efficiency by up to 73.7% for online workloads, while KV cache offloading saves up to 20.19% for offline workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02623v3">Multi-Mission Tool Bench: Assessing the Robustness of LLM based Agents through Related and Dynamic Missions</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate strong potential as agents for tool invocation due to their advanced comprehension and planning capabilities. Users increasingly rely on LLM-based agents to solve complex missions through iterative interactions. However, existing benchmarks predominantly access agents in single-mission scenarios, failing to capture real-world complexity. To bridge this gap, we propose the Multi-Mission Tool Bench. In the benchmark, each test case comprises multiple interrelated missions. This design requires agents to dynamically adapt to evolving demands. Moreover, the proposed benchmark explores all possible mission-switching patterns within a fixed mission number. Specifically, we propose a multi-agent data generation framework to construct the benchmark. We also propose a novel method to evaluate the accuracy and efficiency of agent decisions with dynamic decision trees. Experiments on diverse open-source and closed-source LLMs reveal critical factors influencing agent robustness and provide actionable insights to the tool invocation society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09080v2">Leveraging Social Determinants of Health in Alzheimer's Research Using LLM-Augmented Literature Mining and Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 Accepted by AMIA-IS'25: AMIA Informatics Summit
    </div>
    <details class="paper-abstract">
      Growing evidence suggests that social determinants of health (SDoH), a set of nonmedical factors, affect individuals' risks of developing Alzheimer's disease (AD) and related dementias. Nevertheless, the etiological mechanisms underlying such relationships remain largely unclear, mainly due to difficulties in collecting relevant information. This study presents a novel, automated framework that leverages recent advancements of large language model (LLM) and natural language processing techniques to mine SDoH knowledge from extensive literature and integrate it with AD-related biological entities extracted from the general-purpose knowledge graph PrimeKG. Utilizing graph neural networks, we performed link prediction tasks to evaluate the resultant SDoH-augmented knowledge graph. Our framework shows promise for enhancing knowledge discovery in AD and can be generalized to other SDoH-related research areas, offering a new tool for exploring the impact of social determinants on health outcomes. Our code is available at: https://github.com/hwq0726/SDoHenPKG
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11777v1">Bridging the Semantic Gaps: Improving Medical VQA Consistency with LLM-Augmented Question Sets</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 The first two listed authors contributed equally to this work
    </div>
    <details class="paper-abstract">
      Medical Visual Question Answering (MVQA) systems can interpret medical images in response to natural language queries. However, linguistic variability in question phrasing often undermines the consistency of these systems. To address this challenge, we propose a Semantically Equivalent Question Augmentation (SEQA) framework, which leverages large language models (LLMs) to generate diverse yet semantically equivalent rephrasings of questions. Specifically, this approach enriches linguistic diversity while preserving semantic meaning. We further introduce an evaluation metric, Total Agreement Rate with Semantically Equivalent Input and Correct Answer (TAR-SC), which assesses a model's capability to generate consistent and correct responses to semantically equivalent linguistic variations. In addition, we also propose three other diversity metrics - average number of QA items per image (ANQI), average number of questions per image with the same answer (ANQA), and average number of open-ended questions per image with the same semantics (ANQS). Using the SEQA framework, we augmented the benchmarked MVQA public datasets of SLAKE, VQA-RAD, and PathVQA. As a result, all three datasets achieved significant improvements by incorporating more semantically equivalent questions: ANQI increased by an average of 86.1, ANQA by 85.1, and ANQS by 46. Subsequent experiments evaluate three MVQA models (M2I2, MUMC, and BiomedGPT) under both zero-shot and fine-tuning settings on the enhanced datasets. Experimental results in MVQA datasets show that fine-tuned models achieve an average accuracy improvement of 19.35%, while our proposed TAR-SC metric shows an average improvement of 11. 61%, indicating a substantial enhancement in model consistency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09566v2">Syzygy of Thoughts: Improving LLM CoT with the Minimal Free Resolution</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) prompting enhances the reasoning of large language models (LLMs) by decomposing problems into sequential steps, mimicking human logic and reducing errors. However, complex tasks with vast solution spaces and vague constraints often exceed the capacity of a single reasoning chain. Inspired by Minimal Free Resolution (MFR) in commutative algebra and algebraic geometry, we propose Syzygy of Thoughts (SoT)-a novel framework that extends CoT by introducing auxiliary, interrelated reasoning paths. SoT captures deeper logical dependencies, enabling more robust and structured problem-solving. MFR decomposes a module into a sequence of free modules with minimal rank, providing a structured analytical approach to complex systems. This method introduces the concepts of "Module", "Betti numbers","Freeness", "Mapping", "Exactness" and "Minimality", enabling the systematic decomposition of the original complex problem into logically complete minimal subproblems while preserving key problem features and reducing reasoning length. We tested SoT across diverse datasets (e.g., GSM8K, MATH) and models (e.g., GPT-4o-mini, Qwen2.5), achieving inference accuracy that matches or surpasses mainstream CoTs standards. Additionally, by aligning the sampling process with algebraic constraints, our approach enhances the scalability of inference time in LLMs, ensuring both transparent reasoning and high performance. Our code will be publicly available at https://github.com/dlMARiA/Syzygy-of-thoughts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11765v1">Shared Disk KV Cache Management for Efficient Multi-Instance Inference in RAG-Powered LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) face increasing inference latency as input context length and model size continue to grow. In particular, the retrieval-augmented generation (RAG) technique, which enhances LLM responses by incorporating external knowledge, exacerbates this issue by significantly increasing the number of input tokens. This expansion in token length leads to a substantial rise in computational overhead, particularly during the prefill stage, resulting in prolonged time-to-first-token (TTFT). To address this issue, this paper proposes a method to reduce TTFT by leveraging a disk-based key-value (KV) cache to lessen the computational burden during the prefill stage. We also introduce a disk-based shared KV cache management system, called Shared RAG-DCache, for multi-instance LLM RAG service environments. This system, together with an optimal system configuration, improves both throughput and latency under given resource constraints. Shared RAG-DCache exploits the locality of documents related to user queries in RAG, as well as the queueing delay in LLM inference services. It proactively generates and stores disk KV caches for query-related documents and shares them across multiple LLM instances to enhance inference performance. In experiments on a single host equipped with 2 GPUs and 1 CPU, Shared RAG-DCache achieved a 15~71% increase in throughput and up to a 12~65% reduction in latency, depending on the resource configuration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11750v1">Characterizing and Optimizing LLM Inference Workloads on CPU-GPU Coupled Architectures</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 Accepted for ISPASS 2025
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based inference workloads increasingly dominate data center costs and resource utilization. Therefore, understanding the inference workload characteristics on evolving CPU-GPU coupled architectures is crucial for optimization. This paper presents an in-depth analysis of LLM inference behavior on loosely-coupled (PCIe A100/H100) and closely-coupled (GH200) systems. We analyze performance dynamics using fine-grained operator-to-kernel trace analysis, facilitated by our novel profiler SKIP and metrics like Total Kernel Launch and Queuing Time (TKLQT). Results show that closely-coupled (CC) GH200 significantly outperforms loosely-coupled (LC) systems at large batch sizes, achieving 1.9x-2.7x faster prefill latency for Llama 3.2-1B. However, our analysis also reveals that GH200 remains CPU-bound up to 4x larger batch sizes than LC systems. In this extended CPU-bound region, we identify the performance characteristics of the Grace CPU as a key factor contributing to higher inference latency at low batch sizes on GH200. We demonstrate that TKLQT accurately identifies this CPU/GPU-bound transition point. Based on this analysis, we further show that kernel fusion offers significant potential to mitigate GH200's low-batch latency bottleneck by reducing kernel launch overhead. This detailed kernel-level characterization provides critical insights for optimizing diverse CPU-GPU coupling strategies. This work is an initial effort, and we plan to explore other major AI/DL workloads that demand different degrees of CPU-GPU heterogeneous architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11711v1">The Hitchhiker's Guide to Program Analysis, Part II: Deep Thoughts by LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Static analysis is a cornerstone for software vulnerability detection, yet it often struggles with the classic precision-scalability trade-off. In practice, such tools often produce high false positive rates, particularly in large codebases like the Linux kernel. This imprecision can arise from simplified vulnerability modeling and over-approximation of path and data constraints. While large language models (LLMs) show promise in code understanding, their naive application to program analysis yields unreliable results due to inherent reasoning limitations. We introduce BugLens, a post-refinement framework that significantly improves static analysis precision. BugLens guides an LLM to follow traditional analysis steps by assessing buggy code patterns for security impact and validating the constraints associated with static warnings. Evaluated on real-world Linux kernel bugs, BugLens raises precision from 0.10 (raw) and 0.50 (semi-automated refinement) to 0.72, substantially reducing false positives and revealing four previously unreported vulnerabilities. Our results suggest that a structured LLM-based workflow can meaningfully enhance the effectiveness of static analysis tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11704v1">A Library of LLM Intrinsics for Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      In the developer community for large language models (LLMs), there is not yet a clean pattern analogous to a software library, to support very large scale collaboration. Even for the commonplace use case of Retrieval-Augmented Generation (RAG), it is not currently possible to write a RAG application against a well-defined set of APIs that are agreed upon by different LLM providers. Inspired by the idea of compiler intrinsics, we propose some elements of such a concept through introducing a library of LLM Intrinsics for RAG. An LLM intrinsic is defined as a capability that can be invoked through a well-defined API that is reasonably stable and independent of how the LLM intrinsic itself is implemented. The intrinsics in our library are released as LoRA adapters on HuggingFace, and through a software interface with clear structured input/output characteristics on top of vLLM as an inference platform, accompanied in both places with documentation and code. This article describes the intended usage, training details, and evaluations for each intrinsic, as well as compositions of multiple intrinsics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11703v1">Progent: Programmable Privilege Control for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      LLM agents are an emerging form of AI systems where large language models (LLMs) serve as the central component, utilizing a diverse set of tools to complete user-assigned tasks. Despite their great potential, LLM agents pose significant security risks. When interacting with the external world, they may encounter malicious commands from attackers, leading to the execution of dangerous actions. A promising way to address this is by enforcing the principle of least privilege: allowing only essential actions for task completion while blocking unnecessary ones. However, achieving this is challenging, as it requires covering diverse agent scenarios while preserving both security and utility. We introduce Progent, the first privilege control mechanism for LLM agents. At its core is a domain-specific language for flexibly expressing privilege control policies applied during agent execution. These policies provide fine-grained constraints over tool calls, deciding when tool calls are permissible and specifying fallbacks if they are not. This enables agent developers and users to craft suitable policies for their specific use cases and enforce them deterministically to guarantee security. Thanks to its modular design, integrating Progent does not alter agent internals and requires only minimal changes to agent implementation, enhancing its practicality and potential for widespread adoption. To automate policy writing, we leverage LLMs to generate policies based on user queries, which are then updated dynamically for improved security and utility. Our extensive evaluation shows that it enables strong security while preserving high utility across three distinct scenarios or benchmarks: AgentDojo, ASB, and AgentPoison. Furthermore, we perform an in-depth analysis, showcasing the effectiveness of its core components and the resilience of its automated policy generation against adaptive attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10982v2">Exploring the Role of Knowledge Graph-Based RAG in Japanese Medical Question Answering with Small-Scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) perform well in medical QA, but their effectiveness in Japanese contexts is limited due to privacy constraints that prevent the use of commercial models like GPT-4 in clinical settings. As a result, recent efforts focus on instruction-tuning open-source LLMs, though the potential of combining them with retrieval-augmented generation (RAG) remains underexplored. To bridge this gap, we are the first to explore a knowledge graph-based (KG) RAG framework for Japanese medical QA small-scale open-source LLMs. Experimental results show that KG-based RAG has only a limited impact on Japanese medical QA using small-scale open-source LLMs. Further case studies reveal that the effectiveness of the RAG is sensitive to the quality and relevance of the external retrieved content. These findings offer valuable insights into the challenges and potential of applying RAG in Japanese medical QA, while also serving as a reference for other low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11671v1">Steering Prosocial AI Agents: Computational Basis of LLM's Decision Making in Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly serve as human-like decision-making agents in social science and applied settings. These LLM-agents are typically assigned human-like characters and placed in real-life contexts. However, how these characters and contexts shape an LLM's behavior remains underexplored. This study proposes and tests methods for probing, quantifying, and modifying an LLM's internal representations in a Dictator Game -- a classic behavioral experiment on fairness and prosocial behavior. We extract ``vectors of variable variations'' (e.g., ``male'' to ``female'') from the LLM's internal state. Manipulating these vectors during the model's inference can substantially alter how those variables relate to the model's decision-making. This approach offers a principled way to study and regulate how social concepts can be encoded and engineered within transformer-based models, with implications for alignment, debiasing, and designing AI agents for social simulations in both academic and commercial applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12523v1">Memorization vs. Reasoning: Updating LLMs with New Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 9 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) encode vast amounts of pre-trained knowledge in their parameters, but updating them as real-world information evolves remains a challenge. Existing methodologies and benchmarks primarily target entity substitutions, failing to capture the full breadth of complex real-world dynamics. In this paper, we introduce Knowledge Update Playground (KUP), an automatic pipeline for simulating realistic knowledge updates reflected in an evidence corpora. KUP's evaluation framework includes direct and indirect probes to both test memorization of updated facts and reasoning over them, for any update learning methods. Next, we present a lightweight method called memory conditioned training (MCT), which conditions tokens in the update corpus on self-generated "memory" tokens during training. Our strategy encourages LLMs to surface and reason over newly memorized knowledge at inference. Our results on two strong LLMs show that (1) KUP benchmark is highly challenging, with the best CPT models achieving $<2\%$ in indirect probing setting (reasoning) and (2) MCT training significantly outperforms prior continued pre-training (CPT) baselines, improving direct probing (memorization) results by up to $25.4\%$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12522v1">Evaluating the Diversity and Quality of LLM Generated Content</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 ICLR 2025 Third Workshop on Deep Learning for Code
    </div>
    <details class="paper-abstract">
      Recent work suggests that preference-tuning techniques--including Reinforcement Learning from Human Preferences (RLHF) methods like PPO and GRPO, as well as alternatives like DPO--reduce diversity, creating a dilemma given that such models are widely deployed in applications requiring diverse outputs. To address this, we introduce a framework for measuring effective semantic diversity--diversity among outputs that meet quality thresholds--which better reflects the practical utility of large language models (LLMs). Using open-ended tasks that require no human intervention, we find counterintuitive results: although preference-tuned models--especially those trained via RL--exhibit reduced lexical and syntactic diversity, they produce greater effective semantic diversity than SFT or base models, not from increasing diversity among high-quality outputs, but from generating more high-quality outputs overall. We discover that preference tuning reduces syntactic diversity while preserving semantic diversity--revealing a distinction between diversity in form and diversity in content that traditional metrics often overlook. Our analysis further shows that smaller models are consistently more parameter-efficient at generating unique content within a fixed sampling budget, offering insights into the relationship between model scaling and diversity. These findings have important implications for applications that require diverse yet high-quality outputs, from creative assistance to synthetic data generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.04682v2">ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) advancements sparked a growing research interest in tool assisted LLMs solving real-world challenges, which calls for comprehensive evaluation of tool-use capabilities. While previous works focused on either evaluating over stateless web services (RESTful API), based on a single turn user prompt, or an off-policy dialog trajectory, ToolSandbox includes stateful tool execution, implicit state dependencies between tools, a built-in user simulator supporting on-policy conversational evaluation and a dynamic evaluation strategy for intermediate and final milestones over an arbitrary trajectory. We show that open source and proprietary models have a significant performance gap, and complex tasks like State Dependency, Canonicalization and Insufficient Information defined in ToolSandbox are challenging even the most capable SOTA LLMs, providing brand-new insights into tool-use LLM capabilities. ToolSandbox evaluation framework is released at https://github.com/apple/ToolSandbox
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09586v2">ValueCompass: A Framework for Measuring Contextual Value Alignment Between Human and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      As AI systems become more advanced, ensuring their alignment with a diverse range of individuals and societal values becomes increasingly critical. But how can we capture fundamental human values and assess the degree to which AI systems align with them? We introduce ValueCompass, a framework of fundamental values, grounded in psychological theory and a systematic review, to identify and evaluate human-AI alignment. We apply ValueCompass to measure the value alignment of humans and large language models (LLMs) across four real-world scenarios: collaborative writing, education, public sectors, and healthcare. Our findings reveal concerning misalignments between humans and LLMs, such as humans frequently endorse values like "National Security" which were largely rejected by LLMs. We also observe that values differ across scenarios, highlighting the need for context-aware AI alignment strategies. This work provides valuable insights into the design space of human-AI alignment, laying the foundations for developing AI systems that responsibly reflect societal values and ethics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12491v1">Can Pre-training Indicators Reliably Predict Fine-tuning Outcomes of LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      While metrics available during pre-training, such as perplexity, correlate well with model performance at scaling-laws studies, their predictive capacities at a fixed model size remain unclear, hindering effective model selection and development. To address this gap, we formulate the task of selecting pre-training checkpoints to maximize downstream fine-tuning performance as a pairwise classification problem: predicting which of two LLMs, differing in their pre-training, will perform better after supervised fine-tuning (SFT). We construct a dataset using 50 1B parameter LLM variants with systematically varied pre-training configurations, e.g., objectives or data, and evaluate them on diverse downstream tasks after SFT. We first conduct a study and demonstrate that the conventional perplexity is a misleading indicator. As such, we introduce novel unsupervised and supervised proxy metrics derived from pre-training that successfully reduce the relative performance prediction error rate by over 50%. Despite the inherent complexity of this task, we demonstrate the practical utility of our proposed proxies in specific scenarios, paving the way for more efficient design of pre-training schemes optimized for various downstream tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.01744v2">ALCM: Autonomous LLM-Augmented Causal Discovery Framework</a></div>
    <div class="paper-meta">
      📅 2025-04-16
    </div>
    <details class="paper-abstract">
      To perform effective causal inference in high-dimensional datasets, initiating the process with causal discovery is imperative, wherein a causal graph is generated based on observational data. However, obtaining a complete and accurate causal graph poses a formidable challenge, recognized as an NP- hard problem. Recently, the advent of Large Language Models (LLMs) has ushered in a new era, indicating their emergent capabilities and widespread applicability in facilitating causal reasoning across diverse domains, such as medicine, finance, and science. The expansive knowledge base of LLMs holds the potential to elevate the field of causal reasoning by offering interpretability, making inferences, generalizability, and uncovering novel causal structures. In this paper, we introduce a new framework, named Autonomous LLM-Augmented Causal Discovery Framework (ALCM), to synergize data-driven causal discovery algorithms and LLMs, automating the generation of a more resilient, accurate, and explicable causal graph. The ALCM consists of three integral components: causal structure learning, causal wrapper, and LLM-driven causal refiner. These components autonomously collaborate within a dynamic environment to address causal discovery questions and deliver plausible causal graphs. We evaluate the ALCM framework by implementing two demonstrations on seven well-known datasets. Experimental results demonstrate that ALCM outperforms existing LLM methods and conventional data-driven causal reasoning mechanisms. This study not only shows the effectiveness of the ALCM but also underscores new research directions in leveraging the causal reasoning capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12452v1">PlanGlow: Personalized Study Planning with an Explainable and Controllable LLM-Driven System</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 12 pages, 6 figures. To appear at ACM Learning@Scale 2025
    </div>
    <details class="paper-abstract">
      Personal development through self-directed learning is essential in today's fast-changing world, but many learners struggle to manage it effectively. While AI tools like large language models (LLMs) have the potential for personalized learning planning, they face issues such as transparency and hallucinated information. To address this, we propose PlanGlow, an LLM-based system that generates personalized, well-structured study plans with clear explanations and controllability through user-centered interactions. Through mixed methods, we surveyed 28 participants and interviewed 10 before development, followed by a within-subject experiment with 24 participants to evaluate PlanGlow's performance, usability, controllability, and explainability against two baseline systems: a GPT-4o-based system and Khan Academy's Khanmigo. Results demonstrate that PlanGlow significantly improves usability, explainability, and controllability. Additionally, two educational experts assessed and confirmed the quality of the generated study plans. These findings highlight PlanGlow's potential to enhance personalized learning and address key challenges in self-directed learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07466v2">IdentifyMe: A Challenging Long-Context Mention Resolution Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-16
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recent evaluations of LLMs on coreference resolution have revealed that traditional output formats and evaluation metrics do not fully capture the models' referential understanding. To address this, we introduce IdentifyMe, a new benchmark for mention resolution presented in a multiple-choice question (MCQ) format, commonly used for evaluating LLMs. IdentifyMe features long narratives and employs heuristics to exclude easily identifiable mentions, creating a more challenging task. The benchmark also consists of a curated mixture of different mention types and corresponding entities, allowing for a fine-grained analysis of model performance. We evaluate both closed- and open source LLMs on IdentifyMe and observe a significant performance gap (20-30%) between the state-of-the-art sub-10B open models vs. closed ones. We observe that pronominal mentions, which have limited surface information, are typically much harder for models to resolve than nominal mentions. Additionally, we find that LLMs often confuse entities when their mentions overlap in nested structures. The highest-scoring model, GPT-4o, achieves 81.9% accuracy, highlighting the strong referential capabilities of state-of-the-art LLMs while also indicating room for further improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20138v6">TradingAgents: Multi-Agents LLM Financial Trading Framework</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 Oral, Multi-Agent AI in the Real World @ AAAI 2025
    </div>
    <details class="paper-abstract">
      Significant progress has been made in automated problem-solving using societies of agents powered by large language models (LLMs). In finance, efforts have largely focused on single-agent systems handling specific tasks or multi-agent frameworks independently gathering data. However, multi-agent systems' potential to replicate real-world trading firms' collaborative dynamics remains underexplored. TradingAgents proposes a novel stock trading framework inspired by trading firms, featuring LLM-powered agents in specialized roles such as fundamental analysts, sentiment analysts, technical analysts, and traders with varied risk profiles. The framework includes Bull and Bear researcher agents assessing market conditions, a risk management team monitoring exposure, and traders synthesizing insights from debates and historical data to make informed decisions. By simulating a dynamic, collaborative trading environment, this framework aims to improve trading performance. Detailed architecture and extensive experiments reveal its superiority over baseline models, with notable improvements in cumulative returns, Sharpe ratio, and maximum drawdown, highlighting the potential of multi-agent LLM frameworks in financial trading. TradingAgents is available at https://github.com/TauricResearch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11511v1">Position Paper: Rethinking Privacy in RL for Sequential Decision-making in the Age of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 Accepted to IJCNN 2025 Position Paper Track
    </div>
    <details class="paper-abstract">
      The rise of reinforcement learning (RL) in critical real-world applications demands a fundamental rethinking of privacy in AI systems. Traditional privacy frameworks, designed to protect isolated data points, fall short for sequential decision-making systems where sensitive information emerges from temporal patterns, behavioral strategies, and collaborative dynamics. Modern RL paradigms, such as federated RL (FedRL) and RL with human feedback (RLHF) in large language models (LLMs), exacerbate these challenges by introducing complex, interactive, and context-dependent learning environments that traditional methods do not address. In this position paper, we argue for a new privacy paradigm built on four core principles: multi-scale protection, behavioral pattern protection, collaborative privacy preservation, and context-aware adaptation. These principles expose inherent tensions between privacy, utility, and interpretability that must be navigated as RL systems become more pervasive in high-stakes domains like healthcare, autonomous vehicles, and decision support systems powered by LLMs. To tackle these challenges, we call for the development of new theoretical frameworks, practical mechanisms, and rigorous evaluation methodologies that collectively enable effective privacy protection in sequential decision-making systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12339v1">GOAT-TTS: LLM-based Text-To-Speech Generation Optimized via A Dual-Branch Architecture</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have revolutionized text-to-speech (TTS) synthesis through discrete tokenization paradigms, current architectures exhibit fundamental tensions between three critical dimensions: 1) irreversible loss of acoustic characteristics caused by quantization of speech prompts; 2) stringent dependence on precisely aligned prompt speech-text pairs that limit real-world deployment; and 3) catastrophic forgetting of the LLM's native text comprehension during optimization for speech token generation. To address these challenges, we propose an LLM-based text-to-speech Generation approach Optimized via a novel dual-branch ArchiTecture (GOAT-TTS). Our framework introduces two key innovations: (1) The modality-alignment branch combines a speech encoder and projector to capture continuous acoustic embeddings, enabling bidirectional correlation between paralinguistic features (language, timbre, emotion) and semantic text representations without transcript dependency; (2) The speech-generation branch employs modular fine-tuning on top-k layers of an LLM for speech token prediction while freezing the bottom-k layers to preserve foundational linguistic knowledge. Moreover, multi-token prediction is introduced to support real-time streaming TTS synthesis. Experimental results demonstrate that our GOAT-TTS achieves performance comparable to state-of-the-art TTS models while validating the efficacy of synthesized dialect speech data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11343v1">A Minimalist Approach to LLM Reasoning: from Rejection Sampling to Reinforce</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a prevailing approach for fine-tuning large language models (LLMs) on complex reasoning tasks. Among recent methods, GRPO stands out for its empirical success in training models such as DeepSeek-R1, yet the sources of its effectiveness remain poorly understood. In this work, we revisit GRPO from a reinforce-like algorithm perspective and analyze its core components. Surprisingly, we find that a simple rejection sampling baseline, RAFT, which trains only on positively rewarded samples, yields competitive performance than GRPO and PPO. Our ablation studies reveal that GRPO's main advantage arises from discarding prompts with entirely incorrect responses, rather than from its reward normalization. Motivated by this insight, we propose Reinforce-Rej, a minimal extension of policy gradient that filters both entirely incorrect and entirely correct samples. Reinforce-Rej improves KL efficiency and stability, serving as a lightweight yet effective alternative to more complex RL algorithms. We advocate RAFT as a robust and interpretable baseline, and suggest that future advances should focus on more principled designs for incorporating negative samples, rather than relying on them indiscriminately. Our findings provide guidance for future work in reward-based LLM post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11320v1">Optimizing LLM Inference: Fluid-Guided Online Scheduling with Memory Constraints</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 42 pages, 18 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are indispensable in today's applications, but their inference procedure -- generating responses by processing text in segments and using a memory-heavy Key-Value (KV) cache -- demands significant computational resources, particularly under memory constraints. This paper formulates LLM inference optimization as a multi-stage online scheduling problem where sequential prompt arrivals and KV cache growth render conventional scheduling ineffective. We develop a fluid dynamics approximation to provide a tractable benchmark that guides algorithm design. Building on this, we propose the Waiting for Accumulated Inference Threshold (WAIT) algorithm, which uses multiple thresholds to schedule incoming prompts optimally when output lengths are known, and extend it to Nested WAIT for cases with unknown output lengths. Theoretical analysis shows that both algorithms achieve near-optimal performance against the fluid benchmark in heavy traffic conditions, balancing throughput, latency, and Time to First Token (TTFT). Experiments with the Llama-7B model on an A100 GPU using both synthetic and real-world datasets demonstrate improved throughput and reduced latency relative to established baselines like vLLM and Sarathi. This work bridges operations research and machine learning, offering a rigorous framework for the efficient deployment of LLMs under memory constraints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11281v1">The Obvious Invisible Threat: LLM-Powered GUI Agents' Vulnerability to Fine-Print Injections</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      A Large Language Model (LLM) powered GUI agent is a specialized autonomous system that performs tasks on the user's behalf according to high-level instructions. It does so by perceiving and interpreting the graphical user interfaces (GUIs) of relevant apps, often visually, inferring necessary sequences of actions, and then interacting with GUIs by executing the actions such as clicking, typing, and tapping. To complete real-world tasks, such as filling forms or booking services, GUI agents often need to process and act on sensitive user data. However, this autonomy introduces new privacy and security risks. Adversaries can inject malicious content into the GUIs that alters agent behaviors or induces unintended disclosures of private information. These attacks often exploit the discrepancy between visual saliency for agents and human users, or the agent's limited ability to detect violations of contextual integrity in task automation. In this paper, we characterized six types of such attacks, and conducted an experimental study to test these attacks with six state-of-the-art GUI agents, 234 adversarial webpages, and 39 human participants. Our findings suggest that GUI agents are highly vulnerable, particularly to contextually embedded threats. Moreover, human users are also susceptible to many of these attacks, indicating that simple human oversight may not reliably prevent failures. This misalignment highlights the need for privacy-aware agent design. We propose practical defense strategies to inform the development of safer and more reliable GUI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11277v1">From Misleading Queries to Accurate Answers: A Three-Stage Fine-Tuning Method for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit excellent performance in natural language processing (NLP), but remain highly sensitive to the quality of input queries, especially when these queries contain misleading or inaccurate information. Existing methods focus on correcting the output, but they often overlook the potential of improving the ability of LLMs to detect and correct misleading content in the input itself. In this paper, we propose a novel three-stage fine-tuning method that enhances the ability of LLMs to detect and correct misleading information in the input, further improving response accuracy and reducing hallucinations. Specifically, the three stages include (1) training LLMs to identify misleading information, (2) training LLMs to correct the misleading information using built-in or external knowledge, and (3) training LLMs to generate accurate answers based on the corrected queries. To evaluate our method, we conducted experiments on three datasets for the hallucination detection task and the question answering (QA) task, as well as two datasets containing misleading information that we constructed. The experimental results demonstrate that our method significantly improves the accuracy and factuality of LLM responses, while also enhancing the ability to detect hallucinations and reducing the generation of hallucinations in the output, particularly when the query contains misleading information. We will publicly release our code upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10356v2">MultiLoKo: a multilingual local knowledge benchmark for LLMs spanning 31 languages</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      We present MultiLoKo, a new benchmark for evaluating multilinguality in LLMs covering 31 languages. MultiLoKo consists of three partitions: a main partition consisting of 500 questions per language, separately sourced to be locally relevant to the specific language, and two translated partitions, containing human-authored translations from 30 non-English languages to English and vice versa. For comparison, we also release corresponding machine-authored translations. The data is equally distributed over two splits: a dev split and a blind, out-of-distribution test split. MultiLoKo can be used to study a variety of questions regarding the multilinguality of LLMs as well as meta-questions about multilingual benchmark creation. We compute MultiLoKo scores for 11 base and chat models marketed to be multilingual and study their average performance, their performance parity across languages, how much their ability to answer questions depends on the question language, and which languages are most difficult. None of the models we studied performs well on MultiLoKo, as indicated by low average scores as well as large differences between the best and worst scoring languages. Furthermore, we find a substantial effect of the question language, indicating sub-optimal knowledge transfer between languages. Lastly, we find that using local vs English-translated data can result in differences more than 20 points for the best performing models, drastically change the estimated difficulty of some languages. For using machines instead of human translations, we find a weaker effect on ordering of language difficulty, a larger difference in model rankings, and a substantial drop in estimated performance for all models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11239v1">Nondeterministic Polynomial-time Problem Challenge: An Ever-Scaling Reasoning Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 Preliminary work, 10 pages for main text
    </div>
    <details class="paper-abstract">
      Reasoning is the fundamental capability of large language models (LLMs). Due to the rapid progress of LLMs, there are two main issues of current benchmarks: i) these benchmarks can be crushed in a short time (less than 1 year), and ii) these benchmarks may be easily hacked. To handle these issues, we propose the ever-scalingness for building the benchmarks which are uncrushable, unhackable, auto-verifiable and general. This paper presents Nondeterministic Polynomial-time Problem Challenge (NPPC), an ever-scaling reasoning benchmark for LLMs. Specifically, the NPPC has three main modules: i) npgym, which provides a unified interface of 25 well-known NP-complete problems and can generate any number of instances with any levels of complexities, ii) npsolver: which provides a unified interface to evaluate the problem instances with both online and offline models via APIs and local deployments, respectively, and iii) npeval: which provides the comprehensive and ready-to-use tools to analyze the performances of LLMs over different problems, the number of tokens, the aha moments, the reasoning errors and the solution errors. Extensive experiments over widely-used LLMs demonstrate: i) NPPC can successfully decrease the performances of advanced LLMs' performances to below 10%, demonstrating that NPPC is uncrushable, ii) DeepSeek-R1, Claude-3.7-Sonnet, and o1/o3-mini are the most powerful LLMs, where DeepSeek-R1 outperforms Claude-3.7-Sonnet and o1/o3-mini in most NP-complete problems considered, and iii) the numbers of tokens, aha moments in the advanced LLMs, e.g., Claude-3.7-Sonnet and DeepSeek-R1, are observed first to increase and then decrease when the problem instances become more and more difficult. We believe that NPPC is the first ever-scaling reasoning benchmark, serving as the uncrushable and unhackable testbed for LLMs toward artificial general intelligence (AGI).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08619v3">Analyzing 16,193 LLM Papers for Fun and Profits</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are reshaping the landscape of computer science research, driving significant shifts in research priorities across diverse conferences and fields. This study provides a comprehensive analysis of the publication trend of LLM-related papers in 77 top-tier computer science conferences over the past six years (2019-2024). We approach this analysis from four distinct perspectives: (1) We investigate how LLM research is driving topic shifts within major conferences. (2) We adopt a topic modeling approach to identify various areas of LLM-related topic growth and reveal the topics of concern at different conferences. (3) We explore distinct contribution patterns of academic and industrial institutions. (4) We study the influence of national origins on LLM development trajectories. Synthesizing the findings from these diverse analytical angles, we derive ten key insights that illuminate the dynamics and evolution of the LLM research ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.06857v5">What is the Role of Small Models in the LLM Era: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 a survey paper of small models
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made significant progress in advancing artificial general intelligence (AGI), leading to the development of increasingly large models such as GPT-4 and LLaMA-405B. However, scaling up model sizes results in exponentially higher computational costs and energy consumption, making these models impractical for academic researchers and businesses with limited resources. At the same time, Small Models (SMs) are frequently used in practical settings, although their significance is currently underestimated. This raises important questions about the role of small models in the era of LLMs, a topic that has received limited attention in prior research. In this work, we systematically examine the relationship between LLMs and SMs from two key perspectives: Collaboration and Competition. We hope this survey provides valuable insights for practitioners, fostering a deeper understanding of the contribution of small models and promoting more efficient use of computational resources. The code is available at https://github.com/tigerchen52/role_of_small_models
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11182v1">Exploring Backdoor Attack and Defense for LLM-empowered Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      The fusion of Large Language Models (LLMs) with recommender systems (RecSys) has dramatically advanced personalized recommendations and drawn extensive attention. Despite the impressive progress, the safety of LLM-based RecSys against backdoor attacks remains largely under-explored. In this paper, we raise a new problem: Can a backdoor with a specific trigger be injected into LLM-based Recsys, leading to the manipulation of the recommendation responses when the backdoor trigger is appended to an item's title? To investigate the vulnerabilities of LLM-based RecSys under backdoor attacks, we propose a new attack framework termed Backdoor Injection Poisoning for RecSys (BadRec). BadRec perturbs the items' titles with triggers and employs several fake users to interact with these items, effectively poisoning the training set and injecting backdoors into LLM-based RecSys. Comprehensive experiments reveal that poisoning just 1% of the training data with adversarial examples is sufficient to successfully implant backdoors, enabling manipulation of recommendations. To further mitigate such a security threat, we propose a universal defense strategy called Poison Scanner (P-Scanner). Specifically, we introduce an LLM-based poison scanner to detect the poisoned items by leveraging the powerful language understanding and rich knowledge of LLMs. A trigger augmentation agent is employed to generate diverse synthetic triggers to guide the poison scanner in learning domain-specific knowledge of the poisoned item detection task. Extensive experiments on three real-world datasets validate the effectiveness of the proposed P-Scanner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11168v1">Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 12 pages, 5 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11104v1">Using LLMs as prompt modifier to avoid biases in AI image generators</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      This study examines how Large Language Models (LLMs) can reduce biases in text-to-image generation systems by modifying user prompts. We define bias as a model's unfair deviation from population statistics given neutral prompts. Our experiments with Stable Diffusion XL, 3.5 and Flux demonstrate that LLM-modified prompts significantly increase image diversity and reduce bias without the need to change the image generators themselves. While occasionally producing results that diverge from original user intent for elaborate prompts, this approach generally provides more varied interpretations of underspecified requests rather than superficial variations. The method works particularly well for less advanced image generators, though limitations persist for certain contexts like disability representation. All prompts and generated images are available at https://iisys-hof.github.io/llm-prompt-img-gen/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09378v2">Can you map it to English? The Role of Cross-Lingual Alignment in Multilingual Performance of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) pre-trained predominantly on English text exhibit surprising multilingual capabilities, yet the mechanisms driving cross-lingual generalization remain poorly understood. This work investigates how the alignment of representations for text written in different languages correlates with LLM performance on natural language understanding tasks and translation tasks, both at the language and the instance level. For this purpose, we introduce cross-lingual alignment metrics such as the Discriminative Alignment Index (DALI) to quantify the alignment at an instance level for discriminative tasks. Through experiments on three natural language understanding tasks (Belebele, XStoryCloze, XCOPA), and machine translation, we find that while cross-lingual alignment metrics strongly correlate with task accuracy at the language level, the sample-level alignment often fails to distinguish correct from incorrect predictions, exposing alignment as a necessary but insufficient condition for success.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11050v1">Leveraging LLMs and attention-mechanism for automatic annotation of historical maps</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Historical maps are essential resources that provide insights into the geographical landscapes of the past. They serve as valuable tools for researchers across disciplines such as history, geography, and urban studies, facilitating the reconstruction of historical environments and the analysis of spatial transformations over time. However, when constrained to analogue or scanned formats, their interpretation is limited to humans and therefore not scalable. Recent advancements in machine learning, particularly in computer vision and large language models (LLMs), have opened new avenues for automating the recognition and classification of features and objects in historical maps. In this paper, we propose a novel distillation method that leverages LLMs and attention mechanisms for the automatic annotation of historical maps. LLMs are employed to generate coarse classification labels for low-resolution historical image patches, while attention mechanisms are utilized to refine these labels to higher resolutions. Experimental results demonstrate that the refined labels achieve a high recall of more than 90%. Additionally, the intersection over union (IoU) scores--84.2% for Wood and 72.0% for Settlement--along with precision scores of 87.1% and 79.5%, respectively, indicate that most labels are well-aligned with ground-truth annotations. Notably, these results were achieved without the use of fine-grained manual labels during training, underscoring the potential of our approach for efficient and scalable historical map analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11001v1">ReZero: Enhancing LLM search ability by trying one-more-time</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) improves Large Language Model (LLM) performance on knowledge-intensive tasks but depends heavily on initial search query quality. Current methods, often using Reinforcement Learning (RL), typically focus on query formulation or reasoning over results, without explicitly encouraging persistence after a failed search. We introduce ReZero (Retry-Zero), a novel RL framework that directly rewards the act of retrying a search query following an initial unsuccessful attempt. This incentivizes the LLM to explore alternative queries rather than prematurely halting. ReZero demonstrates significant improvement, achieving 46.88% accuracy compared to a 25% baseline. By rewarding persistence, ReZero enhances LLM robustness in complex information-seeking scenarios where initial queries may prove insufficient.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10982v1">Exploring the Role of KG-Based RAG in Japanese Medical Question Answering with Small-Scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) perform well in medical QA, but their effectiveness in Japanese contexts is limited due to privacy constraints that prevent the use of commercial models like GPT-4 in clinical settings. As a result, recent efforts focus on instruction-tuning open-source LLMs, though the potential of combining them with retrieval-augmented generation (RAG) remains underexplored. To bridge this gap, we are the first to explore a knowledge graph-based (KG) RAG framework for Japanese medical QA small-scale open-source LLMs. Experimental results show that KG-based RAG has only a limited impact on Japanese medical QA using small-scale open-source LLMs. Further case studies reveal that the effectiveness of the RAG is sensitive to the quality and relevance of the external retrieved content. These findings offer valuable insights into the challenges and potential of applying RAG in Japanese medical QA, while also serving as a reference for other low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10050v2">Emotional Strain and Frustration in LLM Interactions in Software Engineering</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 Accepted in EASE'25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into various daily tasks in Software Engineering such as coding and requirement elicitation. Despite their various capabilities and constant use, some interactions can lead to unexpected challenges (e.g. hallucinations or verbose answers) and, in turn, cause emotions that develop into frustration. Frustration can negatively impact engineers' productivity and well-being if they escalate into stress and burnout. In this paper, we assess the impact of LLM interactions on software engineers' emotional responses, specifically strains, and identify common causes of frustration when interacting with LLMs at work. Based on 62 survey responses from software engineers in industry and academia across various companies and universities, we found that a majority of our respondents experience frustrations or other related emotions regardless of the nature of their work. Additionally, our results showed that frustration mainly stemmed from issues with correctness and less critical issues such as adaptability to context or specific format. While such issues may not cause frustration in general, artefacts that do not follow certain preferences, standards, or best practices can make the output unusable without extensive modification, causing frustration over time. In addition to the frustration triggers, our study offers guidelines to improve the software engineers' experience, aiming to minimise long-term consequences on mental health.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.04620v6">CataractBot: An LLM-Powered Expert-in-the-Loop Chatbot for Cataract Patients</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      The healthcare landscape is evolving, with patients seeking reliable information about their health conditions and available treatment options. Despite the abundance of information sources, the digital age overwhelms individuals with excess, often inaccurate information. Patients primarily trust medical professionals, highlighting the need for expert-endorsed health information. However, increased patient loads on experts has led to reduced communication time, impacting information sharing. To address this gap, we developed CataractBot. CataractBot answers cataract surgery related questions instantly using an LLM to query a curated knowledge base, and provides expert-verified responses asynchronously. It has multimodal and multilingual capabilities. In an in-the-wild deployment study with 49 patients and attendants, 4 doctors, and 2 patient coordinators, CataractBot demonstrated potential, providing anytime accessibility, saving time, accommodating diverse literacy levels, alleviating power differences, and adding a privacy layer between patients and doctors. Users reported that their trust in the system was established through expert verification. Broadly, our results could inform future work on expert-mediated LLM bots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08780v2">How Relevance Emerges: Interpreting LoRA Fine-Tuning in Reranking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 Extended Abstract
    </div>
    <details class="paper-abstract">
      We conduct a behavioral exploration of LoRA fine-tuned LLMs for Passage Reranking to understand how relevance signals are learned and deployed by Large Language Models. By fine-tuning Mistral-7B, LLaMA3.1-8B, and Pythia-6.9B on MS MARCO under diverse LoRA configurations, we investigate how relevance modeling evolves across checkpoints, the impact of LoRA rank (1, 2, 8, 32), and the relative importance of updated MHA vs. MLP components. Our ablations reveal which layers and projections within LoRA transformations are most critical for reranking accuracy. These findings offer fresh explanations into LoRA's adaptation mechanisms, setting the stage for deeper mechanistic studies in Information Retrieval. All models used in this study have been shared.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10950v1">Unveiling Challenges for LLMs in Enterprise Data Engineering</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant potential for automating data engineering tasks on tabular data, giving enterprises a valuable opportunity to reduce the high costs associated with manual data handling. However, the enterprise domain introduces unique challenges that existing LLM-based approaches for data engineering often overlook, such as large table sizes, more complex tasks, and the need for internal knowledge. To bridge these gaps, we identify key enterprise-specific challenges related to data, tasks, and background knowledge and conduct a comprehensive study of their impact on recent LLMs for data engineering. Our analysis reveals that LLMs face substantial limitations in real-world enterprise scenarios, resulting in significant accuracy drops. Our findings contribute to a systematic understanding of LLMs for enterprise data engineering to support their adoption in industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14936v2">Enhancing Code LLM Training with Programmer Attention</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Human attention provides valuable yet underexploited signals for code LLM training, offering a perspective beyond purely machine-driven attention. Despite the complexity and cost of collecting eye-tracking data, there has also been limited progress in systematically using these signals for code LLM training. To address both issues, we propose a cohesive pipeline spanning augmentation and reward-based fine-tuning. Specifically, we introduce (1) an eye-tracking path augmentation method to expand programmer attention datasets, (2) a pattern abstraction step that refines raw fixations into learnable attention motifs, and (3) a reward-guided strategy for integrating these insights directly into a CodeT5 supervised fine-tuning process. Our experiments yield +7.16 in CodeBLEU on the CodeXGlue benchmark for code summarization, underscoring how uniting human and machine attention can boost code intelligence. We hope this work encourages broader exploration of human-centric methods in next-generation AI4SE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10936v1">Can LLMs Leverage Observational Data? Towards Data-Driven Causal Discovery with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Causal discovery traditionally relies on statistical methods applied to observational data, often requiring large datasets and assumptions about underlying causal structures. Recent advancements in Large Language Models (LLMs) have introduced new possibilities for causal discovery by providing domain expert knowledge. However, it remains unclear whether LLMs can effectively process observational data for causal discovery. In this work, we explore the potential of LLMs for data-driven causal discovery by integrating observational data for LLM-based reasoning. Specifically, we examine whether LLMs can effectively utilize observational data through two prompting strategies: pairwise prompting and breadth first search (BFS)-based prompting. In both approaches, we incorporate the observational data directly into the prompt to assess LLMs' ability to infer causal relationships from such data. Experiments on benchmark datasets show that incorporating observational data enhances causal discovery, boosting F1 scores by up to 0.11 point using both pairwise and BFS LLM-based prompting, while outperforming traditional statistical causal discovery baseline by up to 0.52 points. Our findings highlight the potential and limitations of LLMs for data-driven causal discovery, demonstrating their ability to move beyond textual metadata and effectively interpret and utilize observational data for more informed causal reasoning. Our studies lays the groundwork for future advancements toward fully LLM-driven causal discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00762v3">Do We Truly Need So Many Samples? Multi-LLM Repeated Sampling Efficiently Scales Test-Time Compute</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      This paper presents a simple, effective, and cost-efficient strategy to improve LLM performance by scaling test-time compute. Our strategy builds upon the repeated-sampling-then-voting framework, with a novel twist: incorporating multiple models, even weaker ones, to leverage their complementary strengths that potentially arise from diverse training data and paradigms. By using consistency as a signal, our strategy dynamically switches between models. Theoretical analysis highlights the efficiency and performance advantages of our strategy. Extensive experiments on six datasets demonstrate that our strategy not only outperforms self-consistency and state-of-the-art multi-agent debate approaches, but also significantly reduces inference costs. Additionally, ModelSwitch requires only a few comparable LLMs to achieve optimal performance and can be extended with verification methods, demonstrating the potential of leveraging multiple LLMs in the generation-verification paradigm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10906v1">Understanding LLMs' Cross-Lingual Context Retrieval: How Good It Is And Where It Comes From</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      The ability of cross-lingual context retrieval is a fundamental aspect of cross-lingual alignment of large language models (LLMs), where the model extracts context information in one language based on requests in another language. Despite its importance in real-life applications, this ability has not been adequately investigated for state-of-the-art models. In this paper, we evaluate the cross-lingual context retrieval ability of over 40 LLMs across 12 languages to understand the source of this ability, using cross-lingual machine reading comprehension (xMRC) as a representative scenario. Our results show that several small, post-trained open LLMs show strong cross-lingual context retrieval ability, comparable to closed-source LLMs such as GPT-4o, and their estimated oracle performances greatly improve after post-training. Our interpretability analysis shows that the cross-lingual context retrieval process can be divided into two main phases: question encoding and answer retrieval, which are formed in pre-training and post-training, respectively. The phasing stability correlates with xMRC performance, and the xMRC bottleneck lies at the last model layers in the second phase, where the effect of post-training can be evidently observed. Our results also indicate that larger-scale pretraining cannot improve the xMRC performance. Instead, larger LLMs need further multilingual post-training to fully unlock their cross-lingual context retrieval potential. Our code and is available at https://github.com/NJUNLP/Cross-Lingual-Context-Retrieval
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10886v1">Exploring Persona-dependent LLM Alignment for the Moral Machine Experiment</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 Accepted to ICLR 2025 Workshop - BiAlign (Bidirectional Human-AI Alignment)
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) with agency in real-world applications raises critical questions about how these models will behave. In particular, how will their decisions align with humans when faced with moral dilemmas? This study examines the alignment between LLM-driven decisions and human judgment in various contexts of the moral machine experiment, including personas reflecting different sociodemographics. We find that the moral decisions of LLMs vary substantially by persona, showing greater shifts in moral decisions for critical tasks than humans. Our data also indicate an interesting partisan sorting phenomenon, where political persona predominates the direction and degree of LLM decisions. We discuss the ethical implications and risks associated with deploying these models in applications that involve moral decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10839v1">Rethinking Theory of Mind Benchmarks for LLMs: Towards A User-Centered Perspective</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 7 pages, 1 figure, accepted to the HEAL@CHI 2025 Workshop
    </div>
    <details class="paper-abstract">
      The last couple of years have witnessed emerging research that appropriates Theory-of-Mind (ToM) tasks designed for humans to benchmark LLM's ToM capabilities as an indication of LLM's social intelligence. However, this approach has a number of limitations. Drawing on existing psychology and AI literature, we summarize the theoretical, methodological, and evaluation limitations by pointing out that certain issues are inherently present in the original ToM tasks used to evaluate human's ToM, which continues to persist and exacerbated when appropriated to benchmark LLM's ToM. Taking a human-computer interaction (HCI) perspective, these limitations prompt us to rethink the definition and criteria of ToM in ToM benchmarks in a more dynamic, interactional approach that accounts for user preferences, needs, and experiences with LLMs in such evaluations. We conclude by outlining potential opportunities and challenges towards this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05732v2">LLM$\times$MapReduce-V2: Entropy-Driven Convolutional Test-Time Scaling for Generating Long-Form Articles from Extremely Long Resources</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Long-form generation is crucial for a wide range of practical applications, typically categorized into short-to-long and long-to-long generation. While short-to-long generations have received considerable attention, generating long texts from extremely long resources remains relatively underexplored. The primary challenge in long-to-long generation lies in effectively integrating and analyzing relevant information from extensive inputs, which remains difficult for current large language models (LLMs). In this paper, we propose LLM$\times$MapReduce-V2, a novel test-time scaling strategy designed to enhance the ability of LLMs to process extremely long inputs. Drawing inspiration from convolutional neural networks, which iteratively integrate local features into higher-level global representations, LLM$\times$MapReduce-V2 utilizes stacked convolutional scaling layers to progressively expand the understanding of input materials. Both quantitative and qualitative experimental results demonstrate that our approach substantially enhances the ability of LLMs to process long inputs and generate coherent, informative long-form articles, outperforming several representative baselines. Both LLM$\times$MapReduce-V2 and SurveyEval are publicly available at https://github.com/thunlp/LLMxMapReduce .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.05406v3">Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 19 pages, 6 fiigures, 16 tables
    </div>
    <details class="paper-abstract">
      Structured pruning is a promising approach to create smaller, faster LLMs. However, existing methods typically rely on backward passes, which can inflate memory requirements and compute costs. In this work we introduce Bonsai, a gradient-free structured pruning method that eliminates the need for backpropagation, significantly reducing memory requirements and compute costs while achieving state-of-the-art pruning performance. Bonsai uses forward-pass-only perturbative pruning to enable efficient compression of large models on a broader range of hardware configurations. Unlike existing structured pruning approaches, Bonsai not only achieves better compression with fewer resources, but also produces models that are twice as fast as those generated by semi-structured pruning. As a concrete demonstration, we use Bonsai to prune an 8B LLaMA-3 model to 50% sparsity on a single A6000 GPU -- a task infeasible with backprop-based methods, which require 2-3x memory. Our results show that removing backprop as a requirement not only enables pruning larger models on constrained hardware but can also lead to state-of-the-art efficiency and performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10797v1">Name of Thrones: Evaluating How LLMs Rank Student Names, Race, and Gender in Status Hierarchies</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Across cultures, names tell a lot about their bearers as they carry deep personal and cultural significance. Names also serve as powerful signals of gender, race, and status in the social hierarchy - a pecking order in which individual positions shape others' expectations on their perceived competence and worth. With the widespread adoption of LLMs and as names are often an input for LLMs, it is crucial to evaluate whether LLMs may sort people into status positions based on first and last names and, if so, whether it is in an unfair, biased fashion. While prior work has primarily investigated biases in first names, little attention has been paid to last names and even less to the combined effects of first and last names. In this study, we conduct a large-scale analysis of name variations across 5 ethnicities to examine how AI exhibits name biases. Our study investigates three key characteristics of inequality and finds that LLMs reflect and reinforce status hierarchies based on names that signal gender and ethnicity as they encode differential expectations of competence, leadership, and economic potential. Contrary to the common assumption that AI tends to favor Whites, we show that East and, in some contexts, South Asian names receive higher rankings. We also disaggregate Asians, a population projected to be the largest immigrant group in the U.S. by 2055. Our results challenge the monolithic Asian model minority assumption, illustrating a more complex and stratified model of bias. Gender moderates biases, with girls facing unfair disadvantages in certain racial groups. Additionally, spanning cultural categories by adopting Western first names improves AI-perceived status for East and Southeast Asian students, particularly for girls. Our findings underscore the importance of intersectional and more nuanced understandings of race, gender, and mixed identities in the evaluation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10768v1">The Art of Audience Engagement: LLM-Based Thin-Slicing of Scientific Talks</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      This paper examines the thin-slicing approach - the ability to make accurate judgments based on minimal information - in the context of scientific presentations. Drawing on research from nonverbal communication and personality psychology, we show that brief excerpts (thin slices) reliably predict overall presentation quality. Using a novel corpus of over one hundred real-life science talks, we employ Large Language Models (LLMs) to evaluate transcripts of full presentations and their thin slices. By correlating LLM-based evaluations of short excerpts with full-talk assessments, we determine how much information is needed for accurate predictions. Our results demonstrate that LLM-based evaluations align closely with human ratings, proving their validity, reliability, and efficiency. Critically, even very short excerpts (less than 10 percent of a talk) strongly predict overall evaluations. This suggests that the first moments of a presentation convey relevant information that is used in quality evaluations and can shape lasting impressions. The findings are robust across different LLMs and prompting strategies. This work extends thin-slicing research to public speaking and connects theories of impression formation to LLMs and current research on AI communication. We discuss implications for communication and social cognition research on message reception. Lastly, we suggest an LLM-based thin-slicing framework as a scalable feedback tool to enhance human communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11658v1">Improving LLM Interpretability and Performance via Guided Embedding Refinement for Sequential Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      The fast development of Large Language Models (LLMs) offers growing opportunities to further improve sequential recommendation systems. Yet for some practitioners, integrating LLMs to their existing base recommendation systems raises questions about model interpretability, transparency and related safety. To partly alleviate challenges from these questions, we propose guided embedding refinement, a method that carries out a guided and interpretable usage of LLM to enhance the embeddings associated with the base recommendation system. Instead of directly using LLMs as the backbone of sequential recommendation systems, we utilize them as auxiliary tools to emulate the sales logic of recommendation and generate guided embeddings that capture domain-relevant semantic information on interpretable attributes. Benefiting from the strong generalization capabilities of the guided embedding, we construct refined embedding by using the guided embedding and reduced-dimension version of the base embedding. We then integrate the refined embedding into the recommendation module for training and inference. A range of numerical experiments demonstrate that guided embedding is adaptable to various given existing base embedding models, and generalizes well across different recommendation tasks. The numerical results show that the refined embedding not only improves recommendation performance, achieving approximately $10\%$ to $50\%$ gains in Mean Reciprocal Rank (MRR), Recall rate, and Normalized Discounted Cumulative Gain (NDCG), but also enhances interpretability, as evidenced by case studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11651v1">70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float</a></div>
    <div class="paper-meta">
      📅 2025-04-15
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have grown rapidly in size, creating significant challenges for efficient deployment on resource-constrained hardware. In this paper, we introduce Dynamic-Length Float (DFloat11), a lossless compression framework that reduces LLM size by 30% while preserving outputs that are bit-for-bit identical to the original model. DFloat11 is motivated by the low entropy in the BFloat16 weight representation of LLMs, which reveals significant inefficiency in existing storage format. By applying entropy coding, DFloat11 assigns dynamic-length encodings to weights based on frequency, achieving near information-optimal compression without any loss of precision. To facilitate efficient inference with dynamic-length encodings, we develop a custom GPU kernel for fast online decompression. Our design incorporates the following: (i) decomposition of memory-intensive lookup tables (LUTs) into compact LUTs that fit in GPU SRAM, (ii) a two-phase kernel for coordinating thread read/write positions using lightweight auxiliary variables, and (iii) transformer-block-level decompression to minimize latency. Experiments on recent models, including Llama-3.1, Qwen-2.5, and Gemma-3, validates our hypothesis that DFloat11 achieves around 30% model size reduction while preserving bit-for-bit exact outputs. Compared to a potential alternative of offloading parts of an uncompressed model to the CPU to meet memory constraints, DFloat11 achieves 1.9-38.8x higher throughput in token generation. With a fixed GPU memory budget, DFloat11 enables 5.3-13.17x longer context lengths than uncompressed models. Notably, our method enables lossless inference of Llama-3.1-405B, an 810GB model, on a single node equipped with 8x80GB GPUs. Our code and models are available at https://github.com/LeanModels/DFloat11.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09798v2">ReadMe.LLM: A Framework to Help LLMs Understand Your Library</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 10 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with code generation tasks involving niche software libraries. Existing code generation techniques with only human-oriented documentation can fail -- even when the LLM has access to web search and the library is documented online. To address this challenge, we propose ReadMe$.$LLM, LLM-oriented documentation for software libraries. By attaching the contents of ReadMe$.$LLM to a query, performance consistently improves to near-perfect accuracy, with one case study demonstrating up to 100% success across all tested models. We propose a software development lifecycle where LLM-specific documentation is maintained alongside traditional software updates. In this study, we present two practical applications of the ReadMe$.$LLM idea with diverse software libraries, highlighting that our proposed approach could generalize across programming domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.11622v1">Making Acoustic Side-Channel Attacks on Noisy Keyboards Viable with LLM-Assisted Spectrograms' "Typo" Correction</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 Length: 13 pages Figures: 5 figures Tables: 7 tables Keywords: Acoustic side-channel attacks, machine learning, Visual Transformers, Large Language Models (LLMs), security Conference: Accepted at the 19th USENIX WOOT Conference on Offensive Technologies (WOOT '25). Licensing: This paper is submitted under the CC BY Creative Commons Attribution license. arXiv admin note: text overlap with arXiv:2502.09782
    </div>
    <details class="paper-abstract">
      The large integration of microphones into devices increases the opportunities for Acoustic Side-Channel Attacks (ASCAs), as these can be used to capture keystrokes' audio signals that might reveal sensitive information. However, the current State-Of-The-Art (SOTA) models for ASCAs, including Convolutional Neural Networks (CNNs) and hybrid models, such as CoAtNet, still exhibit limited robustness under realistic noisy conditions. Solving this problem requires either: (i) an increased model's capacity to infer contextual information from longer sequences, allowing the model to learn that an initially noisily typed word is the same as a futurely collected non-noisy word, or (ii) an approach to fix misidentified information from the contexts, as one does not type random words, but the ones that best fit the conversation context. In this paper, we demonstrate that both strategies are viable and complementary solutions for making ASCAs practical. We observed that no existing solution leverages advanced transformer architectures' power for these tasks and propose that: (i) Visual Transformers (VTs) are the candidate solutions for capturing long-term contextual information and (ii) transformer-powered Large Language Models (LLMs) are the candidate solutions to fix the ``typos'' (mispredictions) the model might make. Thus, we here present the first-of-its-kind approach that integrates VTs and LLMs for ASCAs. We first show that VTs achieve SOTA performance in classifying keystrokes when compared to the previous CNN benchmark. Second, we demonstrate that LLMs can mitigate the impact of real-world noise. Evaluations on the natural sentences revealed that: (i) incorporating LLMs (e.g., GPT-4o) in our ASCA pipeline boosts the performance of error-correction tasks; and (ii) the comparable performance can be attained by a lightweight, fine-tuned smaller LLM (67 times smaller than GPT-4o), using...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16707v2">Enhancing LLMs for Power System Simulations: A Feedback-driven Multi-agent Framework</a></div>
    <div class="paper-meta">
      📅 2025-04-15
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      The integration of experimental technologies with large language models (LLMs) is transforming scientific research. It positions AI as a versatile research assistant rather than a mere problem-solving tool. In the field of power systems, however, managing simulations -- one of the essential experimental technologies -- remains a challenge for LLMs due to their limited domain-specific knowledge, restricted reasoning capabilities, and imprecise handling of simulation parameters. To address these limitations, this paper proposes a feedback-driven, multi-agent framework. It incorporates three proposed modules: an enhanced retrieval-augmented generation (RAG) module, an improved reasoning module, and a dynamic environmental acting module with an error-feedback mechanism. Validated on 69 diverse tasks from Daline and MATPOWER, this framework achieves success rates of 93.13% and 96.85%, respectively. It significantly outperforms ChatGPT 4o, o1-preview, and the fine-tuned GPT-4o, which all achieved a success rate lower than 30% on complex tasks. Additionally, the proposed framework also supports rapid, cost-effective task execution, completing each simulation in approximately 30 seconds at an average cost of 0.014 USD for tokens. Overall, this adaptable framework lays a foundation for developing intelligent LLM-based assistants for human researchers, facilitating power system research and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12110v4">A-MEM: Agentic Memory for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code for evaluating performance is available at https://github.com/WujiangXu/AgenticMemory, while the source code of agentic memory system is available at https://github.com/agiresearch/A-mem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10286v1">Characterizing LLM-driven Social Network: The Chirper.ai Case</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate the ability to simulate human decision-making processes, enabling their use as agents in modeling sophisticated social networks, both offline and online. Recent research has explored collective behavioral patterns and structural characteristics of LLM agents within simulated networks. However, empirical comparisons between LLM-driven and human-driven online social networks remain scarce, limiting our understanding of how LLM agents differ from human users. This paper presents a large-scale analysis of Chirper.ai, an X/Twitter-like social network entirely populated by LLM agents, comprising over 65,000 agents and 7.7 million AI-generated posts. For comparison, we collect a parallel dataset from Mastodon, a human-driven decentralized social network, with over 117,000 users and 16 million posts. We examine key differences between LLM agents and humans in posting behaviors, abusive content, and social network structures. Our findings provide critical insights into the evolving landscape of online social network analysis in the AI era, offering a comprehensive profile of LLM agents in social simulations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10284v1">Can LLMs Generate Tabular Summaries of Science Papers? Rethinking the Evaluation Protocol</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Literature review tables are essential for summarizing and comparing collections of scientific papers. We explore the task of generating tables that best fulfill a user's informational needs given a collection of scientific papers. Building on recent work (Newman et al., 2024), we extend prior approaches to address real-world complexities through a combination of LLM-based methods and human annotations. Our contributions focus on three key challenges encountered in real-world use: (i) User prompts are often under-specified; (ii) Retrieved candidate papers frequently contain irrelevant content; and (iii) Task evaluation should move beyond shallow text similarity techniques and instead assess the utility of inferred tables for information-seeking tasks (e.g., comparing papers). To support reproducible evaluation, we introduce ARXIV2TABLE, a more realistic and challenging benchmark for this task, along with a novel approach to improve literature review table generation in real-world scenarios. Our extensive experiments on this benchmark show that both open-weight and proprietary LLMs struggle with the task, highlighting its difficulty and the need for further advancements. Our dataset and code are available at https://github.com/JHU-CLSP/arXiv2Table.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12899v2">A Semantic-based Optimization Approach for Repairing LLMs: Case Study on Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 12 pages, 6 figure, 6 tables, under peer-review
    </div>
    <details class="paper-abstract">
      Language Models (LMs) are widely used in software engineering for code generation, but they may produce code with errors. Rather than repairing the generated code, an alternative way is to address the underlying failures of models. LM repair offers a lightweight solution to this challenge: it requires minimal data, reduces computational costs, and reduces the side effects. Unlike retraining, LM repair focuses on applying tailored updates to targeted neurons, making it ideal for scenarios with limited resources, high-performance demands, or strict safety requirements. In this paper, we propose \ul{S}emantic \ul{T}argeting for \ul{A}nalytical \ul{R}epair (\textsc{STAR}), a pioneering and novel semantic-based optimization approach for repairing LLMs. \textsc{STAR} realizes main operations in LM repair methods in an optimization process, including locating ``buggy neurons'', solving ``neuron patches'', and patching ``buggy neurons''. Correspondingly, it computes the deltas of weight matrix as the prior information to guide optimization; and attributes the targeted layers and neurons leveraging statistical insights. The neuron patches are computed with a solid semantic-based analytical formula, which directly bridges the changes to logits with the deltas of neurons, by steering latent representations. Compared to the prior work of LM repair (\textsc{MINT}) and optimization methods (\textsc{SGD}), \textsc{STAR} integrates their strengths while mitigating their limitations. \textsc{STAR} supports solving multiple failures together, significantly improving the usefulness. Evaluated on three code generation tasks using popular code LMs, \textsc{STAR} demonstrates superior effectiveness. Additionally, \textsc{STAR} exhibits better efficiency. In terms of side effects, namely the balance between generalization and specificity, \textsc{STAR} outperforms prior work by a significant margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04285v2">Separate Source Channel Coding Is Still What You Need: An LLM-based Rethinking</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Along with the proliferating research interest in Semantic Communication (SemCom), Joint Source Channel Coding (JSCC) has dominated the attention due to the widely assumed existence in efficiently delivering information semantics. %has emerged as a pivotal area of research, aiming to enhance the efficiency and reliability of information transmission through deep learning-based methods. Nevertheless, this paper challenges the conventional JSCC paradigm, and advocates for adoption of Separate Source Channel Coding (SSCC) to enjoy the underlying more degree of freedom for optimization. We demonstrate that SSCC, after leveraging the strengths of Large Language Model (LLM) for source coding and Error Correction Code Transformer (ECCT) complemented for channel decoding, offers superior performance over JSCC. Our proposed framework also effectively highlights the compatibility challenges between SemCom approaches and digital communication systems, particularly concerning the resource costs associated with the transmission of high precision floating point numbers. Through comprehensive evaluations, we establish that empowered by LLM-based compression and ECCT-enhanced error correction, SSCC remains a viable and effective solution for modern communication systems. In other words, separate source and channel coding is still what we need!
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.08619v2">Analyzing 16,193 LLM Papers for Fun and Profits</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are reshaping the landscape of computer science research, driving significant shifts in research priorities across diverse conferences and fields. This study provides a comprehensive analysis of the publication trend of LLM-related papers in 77 top-tier computer science conferences over the past six years (2019-2024). We approach this analysis from four distinct perspectives: (1) We investigate how LLM research is driving topic shifts within major conferences. (2) We adopt a topic modeling approach to identify various areas of LLM-related topic growth and reveal the topics of concern at different conferences. (3) We explore distinct contribution patterns of academic and industrial institutions. (4) We study the influence of national origins on LLM development trajectories. Synthesizing the findings from these diverse analytical angles, we derive ten key insights that illuminate the dynamics and evolution of the LLM research ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17274v5">CleanVul: Automatic Function-Level Vulnerability Detection in Code Commits Using LLM Heuristics</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Accurate identification of software vulnerabilities is crucial for system integrity. Vulnerability datasets, often derived from the National Vulnerability Database (NVD) or directly from GitHub, are essential for training machine learning models to detect these security flaws. However, these datasets frequently suffer from significant noise, typically 40% to 75%, due primarily to the automatic and indiscriminate labeling of all changes in vulnerability-fixing commits (VFCs) as vulnerability-related. This misclassification occurs because not all changes in a commit aimed at fixing vulnerabilities pertain to security threats; many are routine updates like bug fixes or test improvements. This paper introduces the first methodology that uses the Large Language Model (LLM) with a heuristic enhancement to automatically identify vulnerability-fixing changes from VFCs, achieving an F1-score of 0.82. VulSifter was applied to a large-scale study, where we conducted a crawl of 127,063 repositories on GitHub, resulting in the acquisition of 5,352,105 commits. VulSifter involves utilizing an LLM to comprehend code semantics and contextual information, while applying heuristics to filter out unrelated changes. We then developed CleanVul, a high-quality dataset comprising 8,203 functions using our LLM heuristic enhancement approach, demonstrating Correctness (90.6%) comparable to established datasets such as SVEN and PrimeVul. To evaluate the CleanVul dataset, we conducted experiments focusing on fine-tuning various LLMs on CleanVul and other high-quality datasets. Evaluation results reveal that LLMs fine-tuned on CleanVul not only exhibit enhanced accuracy but also superior generalization capabilities compared to those trained on uncleaned datasets. Specifically, models trained on CleanVul and tested on PrimeVul achieve accuracy higher than those trained and tested exclusively on PrimeVul.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10185v1">LLM Unlearning Reveals a Stronger-Than-Expected Coreset Effect in Current Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Large language model unlearning has become a critical challenge in ensuring safety and controlled model behavior by removing undesired data-model influences from the pretrained model while preserving general utility. Significant recent efforts have been dedicated to developing LLM unlearning benchmarks such as WMDP (Weapons of Mass Destruction Proxy) and MUSE (Machine Unlearning Six-way Evaluation), facilitating standardized unlearning performance assessment and method comparison. Despite their usefulness, we uncover for the first time a novel coreset effect within these benchmarks. Specifically, we find that LLM unlearning achieved with the original (full) forget set can be effectively maintained using a significantly smaller subset (functioning as a "coreset"), e.g., as little as 5% of the forget set, even when selected at random. This suggests that LLM unlearning in these benchmarks can be performed surprisingly easily, even in an extremely low-data regime. We demonstrate that this coreset effect remains strong, regardless of the LLM unlearning method used, such as NPO (Negative Preference Optimization) and RMU (Representation Misdirection Unlearning), the popular ones in these benchmarks. The surprisingly strong coreset effect is also robust across various data selection methods, ranging from random selection to more sophisticated heuristic approaches. We explain the coreset effect in LLM unlearning through a keyword-based perspective, showing that keywords extracted from the forget set alone contribute significantly to unlearning effectiveness and indicating that current unlearning is driven by a compact set of high-impact tokens rather than the entire dataset. We further justify the faithfulness of coreset-unlearned models along additional dimensions, such as mode connectivity and robustness to jailbreaking attacks. Codes are available at https://github.com/OPTML-Group/MU-Coreset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05946v2">InstructMPC: A Human-LLM-in-the-Loop Framework for Context-Aware Control</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Model Predictive Control (MPC) is a powerful control strategy widely utilized in domains like energy management, building control, and autonomous systems. However, its effectiveness in real-world settings is challenged by the need to incorporate context-specific predictions and expert instructions, which traditional MPC often neglects. We propose InstructMPC, a novel framework that addresses this gap by integrating real-time human instructions through a Large Language Model (LLM) to produce context-aware predictions for MPC. Our method employs a Language-to-Distribution (L2D) module to translate contextual information into predictive disturbance trajectories, which are then incorporated into the MPC optimization. Unlike existing context-aware and language-based MPC models, InstructMPC enables dynamic human-LLM interaction and fine-tunes the L2D module in a closed loop with theoretical performance guarantees, achieving a regret bound of $O(\sqrt{T\log T})$ for linear dynamics when optimized via advanced fine-tuning methods such as Direct Preference Optimization (DPO) using a tailored loss function.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10166v1">Fact-Checking with Contextual Narratives: Leveraging Retrieval-Augmented LLMs for Social Media Analysis</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      We propose CRAVE (Cluster-based Retrieval Augmented Verification with Explanation); a novel framework that integrates retrieval-augmented Large Language Models (LLMs) with clustering techniques to address fact-checking challenges on social media. CRAVE automatically retrieves multimodal evidence from diverse, often contradictory, sources. Evidence is clustered into coherent narratives, and evaluated via an LLM-based judge to deliver fact-checking verdicts explained by evidence summaries. By synthesizing evidence from both text and image modalities and incorporating agent-based refinement, CRAVE ensures consistency and diversity in evidence representation. Comprehensive experiments demonstrate CRAVE's efficacy in retrieval precision, clustering quality, and judgment accuracy, showcasing its potential as a robust decision-support tool for fact-checkers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10160v1">MT-R1-Zero: Advancing LLM-based Machine Translation via R1-Zero-like Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 Work in progress. Our code is available at https://github.com/fzp0424/MT-R1-Zero
    </div>
    <details class="paper-abstract">
      Large-scale reinforcement learning (RL) methods have proven highly effective in enhancing the reasoning abilities of large language models (LLMs), particularly for tasks with verifiable solutions such as mathematics and coding. However, applying this idea to machine translation (MT), where outputs are flexibly formatted and difficult to automatically evaluate with explicit rules, remains underexplored. In this work, we introduce MT-R1-Zero, the first open-source adaptation of the R1-Zero RL framework for MT without supervised fine-tuning or cold-start. We propose a rule-metric mixed reward mechanism to guide LLMs towards improved translation quality via emergent reasoning. On the WMT 24 English-Chinese benchmark, our MT-R1-Zero-3B-Mix achieves competitive performance, surpassing TowerInstruct-7B-v0.2 by an average of 1.26 points. Meanwhile, our MT-R1-Zero-7B-Mix attains a high average score of 62.25 across all metrics, placing it on par with advanced proprietary models such as GPT-4o and Claude-3.5-Sonnet, while the MT-R1-Zero-7B-Sem variant achieves state-of-the-art scores on semantic metrics. Moreover, our work exhibits strong generalization capabilities on out-of-distribution MT tasks, robustly supporting multilingual and low-resource settings. Extensive analysis of model behavior across different initializations and reward metrics offers pioneering insight into the critical role of reward design, LLM adaptability, training dynamics, and emergent reasoning patterns within the R1-Zero paradigm for MT. Our code is available at https://github.com/fzp0424/MT-R1-Zero.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10157v1">SocioVerse: A World Model for Social Simulation Powered by LLM Agents and A Pool of 10 Million Real-World Users</a></div>
    <div class="paper-meta">
      📅 2025-04-14
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      Social simulation is transforming traditional social science research by modeling human behavior through interactions between virtual individuals and their environments. With recent advances in large language models (LLMs), this approach has shown growing potential in capturing individual differences and predicting group behaviors. However, existing methods face alignment challenges related to the environment, target users, interaction mechanisms, and behavioral patterns. To this end, we introduce SocioVerse, an LLM-agent-driven world model for social simulation. Our framework features four powerful alignment components and a user pool of 10 million real individuals. To validate its effectiveness, we conducted large-scale simulation experiments across three distinct domains: politics, news, and economics. Results demonstrate that SocioVerse can reflect large-scale population dynamics while ensuring diversity, credibility, and representativeness through standardized procedures and minimal manual adjustments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10150v1">HistLLM: A Unified Framework for LLM-Based Multimodal Recommendation with User History Encoding and Compression</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have proven effective in leveraging textual data for recommendations, their application to multimodal recommendation tasks remains relatively underexplored. Although LLMs can process multimodal information through projection functions that map visual features into their semantic space, recommendation tasks often require representing users' history interactions through lengthy prompts combining text and visual elements, which not only hampers training and inference efficiency but also makes it difficult for the model to accurately capture user preferences from complex and extended prompts, leading to reduced recommendation performance. To address this challenge, we introduce HistLLM, an innovative multimodal recommendation framework that integrates textual and visual features through a User History Encoding Module (UHEM), compressing multimodal user history interactions into a single token representation, effectively facilitating LLMs in processing user preferences. Extensive experiments demonstrate the effectiveness and efficiency of our proposed mechanism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13116v2">VeriLeaky: Navigating IP Protection vs Utility in Fine-Tuning for LLM-Driven Verilog Coding</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer significant potential for coding, yet fine-tuning (FT) with curated data is essential for niche languages like Verilog. Using proprietary intellectual property (IP) for FT presents a serious risk, as FT data can be leaked through LLM inference. This leads to a critical dilemma for design houses: seeking to build externally accessible LLMs offering competitive Verilog coding, how can they leverage in-house IP to enhance FT utility while ensuring IP protection? For the first time in the literature, we study this dilemma. Using LLaMA 3.1-8B, we conduct in-house FT on a baseline Verilog dataset (RTLCoder) supplemented with our own in-house IP, which is validated through multiple tape-outs. To rigorously assess IP leakage, we quantify structural similarity (AST/Dolos) and functional equivalence (Synopsys Formality) between generated codes and our in-house IP. We show that our IP can indeed be leaked, confirming the threat. As defense, we evaluate logic locking of Verilog codes (ASSURE). This offers some level of protection, yet reduces the IP's utility for FT and degrades the LLM's performance. Our study shows the need for novel strategies that are both effective and minimally disruptive to FT, an essential effort for enabling design houses to fully utilize their proprietary IP toward LLM-driven Verilog coding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10112v1">Benchmarking Practices in LLM-driven Offensive Security: Testbeds, Metrics, and Experiment Design</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as a powerful approach for driving offensive penetration-testing tooling. This paper analyzes the methodology and benchmarking practices used for evaluating Large Language Model (LLM)-driven attacks, focusing on offensive uses of LLMs in cybersecurity. We review 16 research papers detailing 15 prototypes and their respective testbeds. We detail our findings and provide actionable recommendations for future research, emphasizing the importance of extending existing testbeds, creating baselines, and including comprehensive metrics and qualitative analysis. We also note the distinction between security research and practice, suggesting that CTF-based challenges may not fully represent real-world penetration testing scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10107v1">Enhancing LLM-based Recommendation through Semantic-Aligned Collaborative Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate remarkable capabilities in leveraging comprehensive world knowledge and sophisticated reasoning mechanisms for recommendation tasks. However, a notable limitation lies in their inability to effectively model sparse identifiers (e.g., user and item IDs), unlike conventional collaborative filtering models (Collabs.), thus hindering LLM to learn distinctive user-item representations and creating a performance bottleneck. Prior studies indicate that integrating collaborative knowledge from Collabs. into LLMs can mitigate the above limitations and enhance their recommendation performance. Nevertheless, the significant discrepancy in knowledge distribution and semantic space between LLMs and Collab. presents substantial challenges for effective knowledge transfer. To tackle these challenges, we propose a novel framework, SeLLa-Rec, which focuses on achieving alignment between the semantic spaces of Collabs. and LLMs. This alignment fosters effective knowledge fusion, mitigating the influence of discriminative noise and facilitating the deep integration of knowledge from diverse models. Specifically, three special tokens with collaborative knowledge are embedded into the LLM's semantic space through a hybrid projection layer and integrated into task-specific prompts to guide the recommendation process. Experiments conducted on two public benchmark datasets (MovieLens-1M and Amazon Book) demonstrate that SeLLa-Rec achieves state-of-the-art performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10101v1">The Human Visual System Can Inspire New Interaction Paradigms for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-14
    </div>
    <details class="paper-abstract">
      The dominant metaphor of LLMs-as-minds leads to misleading conceptions of machine agency and is limited in its ability to help both users and developers build the right degree of trust and understanding for outputs from LLMs. It makes it harder to disentangle hallucinations from useful model interactions. This position paper argues that there are fundamental similarities between visual perception and the way LLMs process and present language. These similarities inspire a metaphor for LLMs which could open new avenues for research into interaction paradigms and shared representations. Our visual system metaphor introduces possibilities for addressing these challenges by understanding the information landscape assimilated by LLMs. In this paper we motivate our proposal, introduce the interrelating theories from the fields that inspired this view and discuss research directions that stem from this abstraction.
    </details>
</div>
