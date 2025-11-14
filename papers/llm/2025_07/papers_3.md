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
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.13666v2">Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 ACL 2025 main
    </div>
    <details class="paper-abstract">
      The capabilities of recent large language models (LLMs) to generate high-quality content indistinguishable by humans from human-written texts raises many concerns regarding their misuse. Previous research has shown that LLMs can be effectively misused for generating disinformation news articles following predefined narratives. Their capabilities to generate personalized (in various aspects) content have also been evaluated and mostly found usable. However, a combination of personalization and disinformation abilities of LLMs has not been comprehensively studied yet. Such a dangerous combination should trigger integrated safety filters of the LLMs, if there are some. This study fills this gap by evaluating vulnerabilities of recent open and closed LLMs, and their willingness to generate personalized disinformation news articles in English. We further explore whether the LLMs can reliably meta-evaluate the personalization quality and whether the personalization affects the generated-texts detectability. Our results demonstrate the need for stronger safety-filters and disclaimers, as those are not properly functioning in most of the evaluated LLMs. Additionally, our study revealed that the personalization actually reduces the safety-filter activations; thus effectively functioning as a jailbreak. Such behavior must be urgently addressed by LLM developers and service providers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18957v1">SLICEMATE: Accurate and Scalable Static Program Slicing via LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Static program slicing, which extracts the executable portions of a program that affect the values at a specific location, supports many software analysis tasks such as debugging and security auditing. However, traditional slicing tools rely on computationally expensive reachability analysis over dependency graphs, which struggle to scale to large programs and often fail to handle code with incomplete syntax. Recently emerged learning-based methods, while more robust to such cases, still fall short of achieving comparable performance to traditional methods on well-formed code. In this work, we propose SliceMate, a novel static program slicing solution powered by Large Language Model (LLM) agents. It bypasses the need for explicit dependency graph construction and achieving superior slicing accuracy. Concretely, SliceMate integrates three specialized agents: (1) a synthesis agent that produces candidate slices by incrementally expanding the scan scope across functions and files guided by LLM-inferred dependencies; (2) a verification agent that performs conciseness and completeness checks of the candidate slices, detecting missing or irrelevant statements; and (3) a refinement agent that repairs the slices with minimal edits in accordance with the verification results. These agents are orchestrated by a control module that ensures timely convergence and outputs high-quality slices without manual intervention. For rigorous evaluation, we construct a new and high-quality benchmark, SliceBench, comprising 2,200 manually annotated Java and Python programs, with program lengths ranging from 5 to 8,577 lines, significantly larger than those in existing slicing benchmarks. Experimental results show that SliceMate greatly outperforms both traditional and learning-based slicing tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18949v1">Adaptive Learning Systems: Personalized Curriculum Design Using LLM-Powered Analytics</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are revolutionizing the field of education by enabling personalized learning experiences tailored to individual student needs. In this paper, we introduce a framework for Adaptive Learning Systems that leverages LLM-powered analytics for personalized curriculum design. This innovative approach uses advanced machine learning to analyze real-time data, allowing the system to adapt learning pathways and recommend resources that align with each learner's progress. By continuously assessing students, our framework enhances instructional strategies, ensuring that the materials presented are relevant and engaging. Experimental results indicate a marked improvement in both learner engagement and knowledge retention when using a customized curriculum. Evaluations conducted across varied educational environments demonstrate the framework's flexibility and positive influence on learning outcomes, potentially reshaping conventional educational practices into a more adaptive and student-centered model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18153v2">When Noisy Labels Meet Class Imbalance on Graphs: A Graph Augmentation Method with LLM and Pseudo Label</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Class-imbalanced graph node classification is a practical yet underexplored research problem. Although recent studies have attempted to address this issue, they typically assume clean and reliable labels when processing class-imbalanced graphs. This assumption often violates the nature of real-world graphs, where labels frequently contain noise. Given this gap, this paper systematically investigates robust node classification for class-imbalanced graphs with noisy labels. We propose GraphALP, a novel Graph Augmentation framework based on Large language models (LLMs) and Pseudo-labeling techniques. Specifically, we design an LLM-based oversampling method to generate synthetic minority nodes, producing label-accurate minority nodes to alleviate class imbalance. Based on the class-balanced graphs, we develop a dynamically weighted pseudo-labeling method to obtain high-confidence pseudo labels to reduce label noise ratio. Additionally, we implement a secondary LLM-guided oversampling mechanism to mitigate potential class distribution skew caused by pseudo labels. Experimental results show that GraphALP achieves superior performance over state-of-the-art methods on class-imbalanced graphs with noisy labels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13618v3">Seed-X: Building Strong Multilingual Translation LLM with 7B Parameters</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Multilingual translation stands as a challenging task for large language models (LLMs) to handle intricate language patterns and stilted translations that arise in automated translations. In this paper, we introduce Seed-X, a family of open-source LLMs comprising instruct and reasoning models, pushing the limits of translation capability with 7B parameter size. The base model is pre-trained on a diverse, high-quality dataset encompassing both monolingual and bilingual content across 28 languages, harnessing the full potential of multilingual data. The instruct model is then finetuned to translate by Chain-of-Thought (CoT) reasoning and further enhanced through reinforcement learning (RL) to achieve better generalization across diverse language pairs. Seed-X achieves performance comparable to leading closed-source models, including Gemini-2.5 and GPT-4o, across 28 languages, and significantly outperforms larger open-source models in both automatic metrics and human evaluations. We share the best practices through our optimization process, and make the parameter public available for advancing translation research and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18918v1">Uncovering Cross-Linguistic Disparities in LLMs using Sparse Autoencoders</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Multilingual large language models (LLMs) exhibit strong cross-linguistic generalization, yet medium to low resource languages underperform on common benchmarks such as ARC-Challenge, MMLU, and HellaSwag. We analyze activation patterns in Gemma-2-2B across all 26 residual layers and 10 languages: Chinese (zh), Russian (ru), Spanish (es), Italian (it), medium to low resource languages including Indonesian (id), Catalan (ca), Marathi (mr), Malayalam (ml), and Hindi (hi), with English (en) as the reference. Using Sparse Autoencoders (SAEs), we reveal systematic disparities in activation patterns. Medium to low resource languages receive up to 26.27 percent lower activations in early layers, with a persistent gap of 19.89 percent in deeper layers. To address this, we apply activation-aware fine-tuning via Low-Rank Adaptation (LoRA), leading to substantial activation gains, such as 87.69 percent for Malayalam and 86.32 percent for Hindi, while maintaining English retention at approximately 91 percent. After fine-tuning, benchmark results show modest but consistent improvements, highlighting activation alignment as a key factor in enhancing multilingual LLM performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14561v3">Can LLMs Predict Citation Intent? An Experimental Analysis of In-context Learning and Fine-tuning on Open LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 Accepted for publication on TPDL 2025
    </div>
    <details class="paper-abstract">
      This work investigates the ability of open Large Language Models (LLMs) to predict citation intent through in-context learning and fine-tuning. Unlike traditional approaches relying on domain-specific pre-trained models like SciBERT, we demonstrate that general-purpose LLMs can be adapted to this task with minimal task-specific data. We evaluate twelve model variations across five prominent open LLM families using zero-, one-, few-, and many-shot prompting. Our experimental study identifies the top-performing model and prompting parameters through extensive in-context learning experiments. We then demonstrate the significant impact of task-specific adaptation by fine-tuning this model, achieving a relative F1-score improvement of 8% on the SciCite dataset and 4.3% on the ACL-ARC dataset compared to the instruction-tuned baseline. These findings provide valuable insights for model selection and prompt engineering. Additionally, we make our end-to-end evaluation framework and models openly available for future use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14117v4">An Investigation of Prompt Variations for Zero-shot LLM-based Rankers</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 Accepted for publication at the 47th European Conference on Information Retrieval (ECIR 2025)
    </div>
    <details class="paper-abstract">
      We provide a systematic understanding of the impact of specific components and wordings used in prompts on the effectiveness of rankers based on zero-shot Large Language Models (LLMs). Several zero-shot ranking methods based on LLMs have recently been proposed. Among many aspects, methods differ across (1) the ranking algorithm they implement, e.g., pointwise vs. listwise, (2) the backbone LLMs used, e.g., GPT3.5 vs. FLAN-T5, (3) the components and wording used in prompts, e.g., the use or not of role-definition (role-playing) and the actual words used to express this. It is currently unclear whether performance differences are due to the underlying ranking algorithm, or because of spurious factors such as better choice of words used in prompts. This confusion risks to undermine future research. Through our large-scale experimentation and analysis, we find that ranking algorithms do contribute to differences between methods for zero-shot LLM ranking. However, so do the LLM backbones -- but even more importantly, the choice of prompt components and wordings affect the ranking. In fact, in our experiments, we find that, at times, these latter elements have more impact on the ranker's effectiveness than the actual ranking algorithms, and that differences among ranking methods become more blurred when prompt variations are considered.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13175v2">Black Box Deployed -- Functional Criteria for Artificial Moral Agents in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 42 pages. Supplementary material included at end of article
    </div>
    <details class="paper-abstract">
      The advancement of powerful yet opaque large language models (LLMs) necessitates a fundamental revision of the philosophical criteria used to evaluate artificial moral agents (AMAs). Pre-LLM frameworks often relied on the assumption of transparent architectures, which LLMs defy due to their stochastic outputs and opaque internal states. This paper argues that traditional ethical criteria are pragmatically obsolete for LLMs due to this mismatch. Engaging with core themes in the philosophy of technology, this paper proffers a revised set of ten functional criteria to evaluate LLM-based artificial moral agents: moral concordance, context sensitivity, normative integrity, metaethical awareness, system resilience, trustworthiness, corrigibility, partial transparency, functional autonomy, and moral imagination. These guideposts, applied to what we term "SMA-LLS" (Simulating Moral Agency through Large Language Systems), aim to steer AMAs toward greater alignment and beneficial societal integration in the coming years. We illustrate these criteria using hypothetical scenarios involving an autonomous public bus (APB) to demonstrate their practical applicability in morally salient contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15956v3">Do Large Language Models Have an English Accent? Evaluating and Improving the Naturalness of Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Current Large Language Models (LLMs) are predominantly designed with English as the primary language, and even the few that are multilingual tend to exhibit strong English-centric biases. Much like speakers who might produce awkward expressions when learning a second language, LLMs often generate unnatural outputs in non-English languages, reflecting English-centric patterns in both vocabulary and grammar. Despite the importance of this issue, the naturalness of multilingual LLM outputs has received limited attention. In this paper, we address this gap by introducing novel automatic corpus-level metrics to assess the lexical and syntactic naturalness of LLM outputs in a multilingual context. Using our new metrics, we evaluate state-of-the-art LLMs on a curated benchmark in French and Chinese, revealing a tendency towards English-influenced patterns. To mitigate this issue, we also propose a simple and effective alignment method to improve the naturalness of an LLM in a target language and domain, achieving consistent improvements in naturalness without compromising the performance on general-purpose benchmarks. Our work highlights the importance of developing multilingual metrics, resources and methods for the new wave of multilingual LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19643v1">Can You Share Your Story? Modeling Clients' Metacognition and Openness for LLM Therapist Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 Published at ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Understanding clients' thoughts and beliefs is fundamental in counseling, yet current evaluations of LLM therapists often fail to assess this ability. Existing evaluation methods rely on client simulators that clearly disclose internal states to the therapist, making it difficult to determine whether an LLM therapist can uncover unexpressed perspectives. To address this limitation, we introduce MindVoyager, a novel evaluation framework featuring a controllable and realistic client simulator which dynamically adapts itself based on the ongoing counseling session, offering a more realistic and challenging evaluation environment. We further introduce evaluation metrics that assess the exploration ability of LLM therapists by measuring their thorough understanding of client's beliefs and thoughts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15606v3">LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become indispensable in real-world applications. However, their widespread adoption raises significant safety concerns, particularly in responding to socially harmful questions. Despite substantial efforts to improve model safety through alignment, aligned models can still have their safety protections undermined by subsequent fine-tuning - even when the additional training data appears benign. In this paper, we empirically demonstrate that this vulnerability stems from the sensitivity of safety-critical low-rank subspaces in LLM parameters to fine-tuning. Building on this insight, we propose a novel training-free method, termed Low-Rank Extrapolation (LoX), to enhance safety robustness by extrapolating the safety subspace of an aligned LLM. Our experimental results confirm the effectiveness of LoX, demonstrating significant improvements in robustness against both benign and malicious fine-tuning attacks while preserving the model's adaptability to new tasks. For instance, LoX leads to 11% to 54% absolute reductions in attack success rates (ASR) facing benign or malicious fine-tuning attacks. By investigating the ASR landscape of parameters, we attribute the success of LoX to that the extrapolation moves LLM parameters to a flatter zone, thereby less sensitive to perturbations. The code is available at github.com/VITA-Group/LoX.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19608v1">DeltaLLM: A Training-Free Framework Exploiting Temporal Sparsity for Efficient Edge LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Deploying Large Language Models (LLMs) on edge devices remains challenging due to their quadratically increasing computations with the sequence length. Existing studies for dynamic attention pruning are designed for hardware with massively parallel computation capabilities, such as GPUs or TPUs, and aim at long context lengths (e.g., 64K), making them unsuitable for edge scenarios. We present DeltaLLM, a training-free framework that exploits temporal sparsity in attention patterns to enable efficient LLM inference across both the prefilling and decoding stages, on resource-constrained edge devices. DeltaLLM introduces an accuracy- and memory-aware delta matrix construction strategy that introduces temporal sparsity, and a context-aware hybrid attention mechanism that combines full attention in a local context window with delta approximation outside it to increase accuracy. We evaluate our framework on the edge-device-friendly BitNet-b1.58-2B-4T model and Llama3.2-1B-Instruct model across diverse language tasks. The results show that on BitNet, our framework increases the attention sparsity from 0% to 60% during the prefilling stage with slight accuracy improvement on the WG task, and 0% to 57% across both the prefilling and decoding stages, with even higher F1 score from 29.63 to 30.97 on SQuAD-v2 task. On the Llama model, it can also achieve up to 60% sparsity during the prefilling stage and around 57% across both stages with negligible accuracy drop. These results demonstrate that DeltaLLM offers a promising solution for efficient edge deployment, requiring no fine-tuning and seamlessly integrating with existing inference pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19570v1">MCP4EDA: LLM-Powered Model Context Protocol RTL-to-GDSII Automation with Backend Aware Synthesis Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 7 pages, 5 figures Keywords: Model Context Protocol, Electronic Design Automation, Large Language Models, Synthesis Optimization
    </div>
    <details class="paper-abstract">
      This paper presents MCP4EDA, the first Model Context Protocol server that enables Large Language Models (LLMs) to control and optimize the complete open-source RTL-to-GDSII design flow through natural language interaction. The system integrates Yosys synthesis, Icarus Verilog simulation, OpenLane place-and-route, GTKWave analysis, and KLayout visualization into a unified LLM-accessible interface, enabling designers to execute complex multi-tool EDA workflows conversationally via AI assistants such as Claude Desktop and Cursor IDE. The principal contribution is a backend-aware synthesis optimization methodology wherein LLMs analyze actual post-layout timing, power, and area metrics from OpenLane results to iteratively refine synthesis TCL scripts, establishing a closed-loop optimization system that bridges the traditional gap between synthesis estimates and physical implementation reality. In contrast to conventional flows that rely on wire-load models, this methodology leverages real backend performance data to guide synthesis parameter tuning, optimization sequence selection, and constraint refinement, with the LLM functioning as an intelligent design space exploration agent. Experimental evaluation on representative digital designs demonstrates 15-30% improvements in timing closure and 10-20% area reduction compared to default synthesis flows, establishing MCP4EDA as the first practical LLM-controlled end-to-end open-source EDA automation system. The code and demo are avaiable at: http://www.agent4eda.com/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19562v1">PennyCoder: Efficient Domain-Specific LLMs for PennyLane-Based Quantum Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 6 pages, 5 figures, 3 tables, paper accepted to QCE 2025
    </div>
    <details class="paper-abstract">
      The growing demand for robust quantum programming frameworks has unveiled a critical limitation: current large language model (LLM) based quantum code assistants heavily rely on remote APIs, introducing challenges related to privacy, latency, and excessive usage costs. Addressing this gap, we propose PennyCoder, a novel lightweight framework for quantum code generation, explicitly designed for local and embedded deployment to enable on-device quantum programming assistance without external API dependence. PennyCoder leverages a fine-tuned version of the LLaMA 3.1-8B model, adapted through parameter-efficient Low-Rank Adaptation (LoRA) techniques combined with domain-specific instruction tuning optimized for the specialized syntax and computational logic of quantum programming in PennyLane, including tasks in quantum machine learning and quantum reinforcement learning. Unlike prior work focused on cloud-based quantum code generation, our approach emphasizes device-native operability while maintaining high model efficacy. We rigorously evaluated PennyCoder over a comprehensive quantum programming dataset, achieving 44.3% accuracy with our fine-tuned model (compared to 33.7% for the base LLaMA 3.1-8B and 40.1% for the RAG-augmented baseline), demonstrating a significant improvement in functional correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21170v1">OneShield -- the Next Generation of LLM Guardrails</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models has created a general excitement about the great potential for a myriad of applications. While LLMs offer many possibilities, questions about safety, privacy, and ethics have emerged, and all the key actors are working to address these issues with protective measures for their own models and standalone solutions. The constantly evolving nature of LLMs makes the task of universally shielding users against their potential risks extremely challenging, and one-size-fits-all solutions unfeasible. In this work, we propose OneShield, our stand-alone, model-agnostic and customizable solution to safeguard LLMs. OneShield aims to provide facilities for defining risk factors, expressing and declaring contextual safety and compliance policies, and mitigating LLM risks, with a focus on each specific customer. We describe the implementation of the framework, the scalability considerations and provide usage statistics of OneShield since its first deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21168v1">Diverse LLMs or Diverse Question Interpretations? That is the Ensembling Question</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Effectively leveraging diversity has been shown to improve performance for various machine learning models, including large language models (LLMs). However, determining the most effective way of using diversity remains a challenge. In this work, we compare two diversity approaches for answering binary questions using LLMs: model diversity, which relies on multiple models answering the same question, and question interpretation diversity, which relies on using the same model to answer the same question framed in different ways. For both cases, we apply majority voting as the ensemble consensus heuristic to determine the final answer. Our experiments on boolq, strategyqa, and pubmedqa show that question interpretation diversity consistently leads to better ensemble accuracy compared to model diversity. Furthermore, our analysis of GPT and LLaMa shows that model diversity typically produces results between the best and the worst ensemble members without clear improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.21159v1">Adaptive Cluster Collaborativeness Boosts LLMs Medical Decision Support Capacity</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      The collaborativeness of large language models (LLMs) has proven effective in natural language processing systems, holding considerable promise for healthcare development. However, it lacks explicit component selection rules, necessitating human intervention or clinical-specific validation. Moreover, existing architectures heavily rely on a predefined LLM cluster, where partial LLMs underperform in medical decision support scenarios, invalidating the collaborativeness of LLMs. To this end, we propose an adaptive cluster collaborativeness methodology involving self-diversity and cross-consistency maximization mechanisms to boost LLMs medical decision support capacity. For the self-diversity, we calculate the fuzzy matching value of pairwise outputs within an LLM as its self-diversity value, subsequently prioritizing LLMs with high self-diversity values as cluster components in a training-free manner. For the cross-consistency, we first measure cross-consistency values between the LLM with the highest self-diversity value and others, and then gradually mask out the LLM having the lowest cross-consistency value to eliminate the potential inconsistent output during the collaborative propagation. Extensive experiments on two specialized medical datasets, NEJMQA and MMLU-Pro-health, demonstrate the effectiveness of our method across physician-oriented specialties. For example, on NEJMQA, our method achieves the accuracy rate up to the publicly official passing score across all disciplines, especially achieving ACC of 65.47\% compared to the 56.12\% achieved by GPT-4 on the Obstetrics and Gynecology discipline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.22940v1">Trustworthy Reasoning: Evaluating and Enhancing Factual Accuracy in LLM Intermediate Thought Processes</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      We present RELIANCE (Reasoning Evaluation with Logical Integrity and Accuracy for Confidence Enhancement), a novel framework addressing a critical vulnerability in Large Language Models (LLMs): the prevalence of factual inaccuracies within intermediate reasoning steps despite correct final answers. This phenomenon poses substantial risks in high-stakes domains including healthcare, legal analysis, and scientific research, where erroneous yet confidently presented reasoning can mislead users into dangerous decisions. Our framework integrates three core components: (1) a specialized fact-checking classifier trained on counterfactually augmented data to detect subtle factual inconsistencies within reasoning chains; (2) a Group Relative Policy Optimization (GRPO) reinforcement learning approach that balances factuality, coherence, and structural correctness through multi-dimensional rewards; and (3) a mechanistic interpretability module examining how factuality improvements manifest in model activations during reasoning processes. Extensive evaluation across ten state-of-the-art models reveals concerning patterns: even leading models like Claude-3.7 and GPT-o1 demonstrate reasoning factual accuracy of only 81.93% and 82.57% respectively. RELIANCE significantly enhances factual robustness (up to 49.90% improvement) while maintaining or improving performance on challenging benchmarks including Math-500, AIME-2024, and GPQA. Furthermore, our activation-level analysis provides actionable insights into how factual enhancements reshape reasoning trajectories within model architectures, establishing foundations for future training methodologies that explicitly target factual robustness through activation-guided optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19399v1">Running in CIRCLE? A Simple Benchmark for LLM Code Interpreter Security</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) increasingly integrate native code interpreters, they enable powerful real-time execution capabilities, substantially expanding their utility. However, such integrations introduce potential system-level cybersecurity threats, fundamentally different from prompt-based vulnerabilities. To systematically evaluate these interpreter-specific risks, we propose CIRCLE (Code-Interpreter Resilience Check for LLM Exploits), a simple benchmark comprising 1,260 prompts targeting CPU, memory, and disk resource exhaustion. Each risk category includes explicitly malicious ("direct") and plausibly benign ("indirect") prompt variants. Our automated evaluation framework assesses not only whether LLMs refuse or generates risky code, but also executes the generated code within the interpreter environment to evaluate code correctness, simplifications made by the LLM to make the code safe, or execution timeouts. Evaluating 7 commercially available models from OpenAI and Google, we uncover significant and inconsistent vulnerabilities. For instance, evaluations show substantial disparities even within providers - OpenAI's o4-mini correctly refuses risky requests at 7.1%, notably higher rates compared to GPT-4.1 at 0.5%. Results particularly underscore that indirect, socially-engineered prompts substantially weaken model defenses. This highlights an urgent need for interpreter-specific cybersecurity benchmarks, dedicated mitigation tools (e.g., guardrails), and clear industry standards to guide safe and responsible deployment of LLM interpreter integrations. The benchmark dataset and evaluation code are publicly released to foster further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19390v1">ReCatcher: Towards LLMs Regression Testing for Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 24 pages, 3 Figures, 2 Tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) for code generation evolve rapidly through fine-tuning, merging, or new model releases. However, such updates can introduce regressions, not only in correctness but also in code quality and performance. To address this, we present ReCatcher, a regression testing framework for Python code generation. ReCatcher systematically compares two LLMs, typically a current model and a candidate update, across three dimensions: logical correctness, static code quality, and execution performance. We apply ReCatcher to assess regressions across three update scenarios, fine-tuning, merging, and model release, using CodeLlama, DeepSeek-Coder, and GPT-4o. Our evaluation shows that fine-tuning with cross-language datasets increases syntax errors by up to 12%. Merging with general-purpose models like Llama2 leads to regressions in correctness by up to 18%. GPT-4o introduces regressions of up to 50% in handling missing imports compared to GPT-3.5-turbo, while GPT-4o-mini suffers up to 80% performance degradation in execution time versus GPT-4o. Overall, logical correctness, performance, and error handling (e.g., syntax errors and missing imports) are the most regression-prone areas. Comparing ReCatcher with baseline solutions, it presents better and consistent accuracy across logical and performance aspects. ReCatcher highlights the importance of systematic regression evaluation before adopting new models, while assisting researchers and practitioners in making more informed update decisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01482v2">Understanding LLM Scientific Reasoning through Promptings and Model's Explanation on the Answers</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding, reasoning, and problem-solving across various domains. However, their ability to perform complex, multi-step reasoning task-essential for applications in science, medicine, and law-remains an area of active investigation. This paper examines the reasoning capabilities of contemporary LLMs, analyzing their strengths, limitations, and potential for improvement. The study uses prompt engineering techniques on the Graduate-Level GoogleProof Q&A (GPQA) dataset to assess the scientific reasoning of GPT-4o. Five popular prompt engineering techniques and two tailored promptings were tested: baseline direct answer (zero-shot), chain-of-thought (CoT), zero-shot CoT, self-ask, self-consistency, decomposition, and multipath promptings. Our findings indicate that while LLMs exhibit emergent reasoning abilities, they often rely on pattern recognition rather than true logical inference, leading to inconsistencies in complex problem-solving. The results indicated that self-consistency outperformed the other prompt engineering technique with an accuracy of 52.99%, followed by direct answer (52.23%). Zero-shot CoT (50%) outperformed multipath (48.44%), decomposition (47.77%), self-ask (46.88%), and CoT (43.75%). Self-consistency performed the second worst in explaining the answers. Simple techniques such as direct answer, CoT, and zero-shot CoT have the best scientific reasoning. We propose a research agenda aimed at bridging these gaps by integrating structured reasoning frameworks, hybrid AI approaches, and human-in-the-loop methodologies. By critically evaluating the reasoning mechanisms of LLMs, this paper contributes to the ongoing discourse on the future of artificial general intelligence and the development of more robust, trustworthy AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19364v1">Integrating LLM in Agent-Based Social Simulation: Opportunities and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      This position paper examines the use of Large Language Models (LLMs) in social simulation, analyzing both their potential and their limitations from a computational social science perspective. The first part reviews recent findings on the ability of LLMs to replicate key aspects of human cognition, including Theory of Mind reasoning and social inference, while also highlighting significant limitations such as cognitive biases, lack of true understanding, and inconsistencies in behavior. The second part surveys emerging applications of LLMs in multi-agent simulation frameworks, focusing on system architectures, scale, and validation strategies. Notable projects such as Generative Agents (Smallville) and AgentSociety are discussed in terms of their design choices, empirical grounding, and methodological innovations. Particular attention is given to the challenges of behavioral fidelity, calibration, and reproducibility in large-scale LLM-driven simulations. The final section distinguishes between contexts where LLMs, like other black-box systems, offer direct value-such as interactive simulations and serious games-and those where their use is more problematic, notably in explanatory or predictive modeling. The paper concludes by advocating for hybrid approaches that integrate LLMs into traditional agent-based modeling platforms (GAMA, Netlogo, etc), enabling modelers to combine the expressive flexibility of language-based reasoning with the transparency and analytical rigor of classical rule-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19353v1">Smooth Reading: Bridging the Gap of Recurrent LLM to Self-Attention LLM on Long-Context Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Recently, recurrent large language models (Recurrent LLMs) with linear computational complexity have re-emerged as efficient alternatives to self-attention-based LLMs (Self-Attention LLMs), which have quadratic complexity. However, Recurrent LLMs often underperform on long-context tasks due to their limited fixed-size memory. Previous research has primarily focused on enhancing the memory capacity of Recurrent LLMs through architectural innovations, but these approaches have not yet enabled Recurrent LLMs to match the performance of Self-Attention LLMs on long-context tasks. We argue that this limitation arises because processing the entire context at once is not well-suited for Recurrent LLMs. In this paper, we propose Smooth Reading, a chunk-wise inference method inspired by human reading strategies. Smooth Reading processes context in chunks and iteratively summarizes the contextual information, thereby reducing memory demands and making the approach more compatible with Recurrent LLMs. Our experimental results show that this method substantially narrows the performance gap between Recurrent and Self-Attention LLMs on long-context tasks, while preserving the efficiency advantages of Recurrent LLMs. Our Smooth Reading boosts SWA-3B-4k (a Recurrent LLM) from 5.68% lower to 3.61% higher performance than Self-Attention LLMs on LongBench. Besides, our method maintains the high efficiency, training 3x faster and inferring 2x faster at 64k context compared to Self-Attention LLMs. To our knowledge, this is the first work to achieve comparable performance using Recurrent LLMs compared with Self-Attention LLMs on long-context tasks. We hope our method will inspire future research in this area. To facilitate further progress, we will release code and dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19334v1">Doubling Your Data in Minutes: Ultra-fast Tabular Data Generation via LLM-Induced Dependency Graphs</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Tabular data is critical across diverse domains, yet high-quality datasets remain scarce due to privacy concerns and the cost of collection. Contemporary approaches adopt large language models (LLMs) for tabular augmentation, but exhibit two major limitations: (1) dense dependency modeling among tabular features that can introduce bias, and (2) high computational overhead in sampling. To address these issues, we propose SPADA for SPArse Dependency-driven Augmentation, a lightweight generative framework that explicitly captures sparse dependencies via an LLM-induced graph. We treat each feature as a node and synthesize values by traversing the graph, conditioning each feature solely on its parent nodes. We explore two synthesis strategies: a non-parametric method using Gaussian kernel density estimation, and a conditional normalizing flow model that learns invertible mappings for conditional density estimation. Experiments on four datasets show that SPADA reduces constraint violations by 4% compared to diffusion-based methods and accelerates generation by nearly 9,500 times over LLM-based baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19283v1">Towards LLM-Enhanced Group Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      In contrast to single-user recommender systems, group recommender systems are designed to generate and explain recommendations for groups. This group-oriented setting introduces additional complexities, as several factors - absent in individual contexts - must be addressed. These include understanding group dynamics (e.g., social dependencies within the group), defining effective decision-making processes, ensuring that recommendations are suitable for all group members, and providing group-level explanations as well as explanations for individual users. In this paper, we analyze in which way large language models (LLMs) can support these aspects and help to increase the overall decision support quality and applicability of group recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19275v1">Mut4All: Fuzzing Compilers via LLM-Synthesized Mutators Learned from Bug Reports</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Mutation-based fuzzing is effective for uncovering compiler bugs, but designing high-quality mutators for modern languages with complex constructs (e.g., templates, macros) remains challenging. Existing methods rely heavily on manual design or human-in-the-loop correction, limiting scalability and cross-language generalizability. We present Mut4All, a fully automated, language-agnostic framework that synthesizes mutators using Large Language Models (LLMs) and compiler-specific knowledge from bug reports. It consists of three agents: (1) a mutator invention agent that identifies mutation targets and generates mutator metadata using compiler-related insights; (2) a mutator implementation synthesis agent, fine-tuned to produce initial implementations; and (3) a mutator refinement agent that verifies and corrects the mutators via unit-test feedback. Mut4All processes 1000 bug reports (500 Rust, 500 C++), yielding 319 Rust and 403 C++ mutators at ~$0.08 each via GPT-4o. Our customized fuzzer, using these mutators, finds 62 bugs in Rust compilers (38 new, 7 fixed) and 34 bugs in C++ compilers (16 new, 1 fixed). Mut4All outperforms existing methods in both unique crash detection and coverage, ranking first on Rust and second on C++.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19254v1">DBMS-LLM Integration Strategies in Industrial and Business Applications: Current Status and Future Challenges</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Modern enterprises are increasingly driven by the DATA+AI paradigm, in which Database Management Systems (DBMSs) and Large Language Models (LLMs) have become two foundational infrastructures powering a wide range of industrial and business applications, such as enterprise analytics, intelligent customer service, and data-driven decision-making. The efficient integration of DBMSs and LLMs within a unified system offers significant opportunities but also introduces new technical challenges. This paper surveys recent developments in DBMS+LLM integration and identifies key future challenges. Specifically, we categorize five representative architectural patterns based on their core design principles, strengths, and trade-offs. Based on this analysis, we further highlight several critical open challenges. We aim to provide a systematic understanding of the current integration landscape and to outline the unresolved issues that must be addressed to achieve scalable and efficient integration of traditional data management and advanced language reasoning in future intelligent applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18377v2">Maximum Redundancy Pruning: A Principle-Driven Layerwise Sparsity Allocation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities, but their enormous size poses significant challenges for deployment in real-world applications. To address this issue, researchers have sought to apply network pruning techniques to LLMs. A critical challenge in pruning is allocation the sparsity for each layer. Recent sparsity allocation methods is often based on heuristics or search that can easily lead to suboptimal performance. In this paper, we conducted an extensive investigation into various LLMs and revealed three significant discoveries: (1) the layerwise pruning sensitivity (LPS) of LLMs is highly non-uniform, (2) the choice of pruning metric affects LPS, and (3) the performance of a sparse model is related to the uniformity of its layerwise redundancy level. Based on these observations, we propose that the layerwise sparsity of LLMs should adhere to three principles: \emph{non-uniformity}, \emph{pruning metric dependency}, and \emph{uniform layerwise redundancy level} in the pruned model. To this end, we proposed Maximum Redundancy Pruning (MRP), an iterative pruning algorithm that prunes in the most redundant layers (\emph{i.e.}, those with the highest non-outlier ratio) at each iteration. The achieved layerwise sparsity aligns with the outlined principles. We conducted extensive experiments on publicly available LLMs, including the LLaMA2 and OPT, across various benchmarks. Experimental results validate the effectiveness of MRP, demonstrating its superiority over previous methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16142v3">Distilling the Implicit Multi-Branch Structure in LLMs' Reasoning via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Distilling reasoning paths from teacher to student models via supervised fine-tuning (SFT) provides a shortcut for improving the reasoning ability of smaller Large Language Models (LLMs). However, the reasoning paths generated by teacher models often reflect only surface-level traces of their underlying authentic reasoning. Insights from cognitive neuroscience suggest that authentic reasoning involves a complex interweaving between meta-reasoning (which selects appropriate sub-problems from multiple candidates) and solving (which addresses the sub-problem). This implies authentic reasoning has an implicit multi-branch structure. Supervised fine-tuning collapses this rich structure into a flat sequence of token prediction in the teacher's reasoning path, preventing effective distillation of this structure to students. To address this limitation, we propose RLKD, a reinforcement learning (RL)-based distillation framework guided by a novel Generative Structure Reward Model (GSRM). Our GSRM converts reasoning paths into multiple meta-reasoning-solving steps and computes rewards to measure structural alignment between student and teacher reasoning. RLKD combines this reward with RL, enabling student LLMs to internalize the teacher's implicit multi-branch reasoning structure rather than merely mimicking fixed output paths. Experiments show RLKD surpasses standard SFT-RL pipelines even when trained on 0.1% of data under an RL-only regime, unlocking greater student reasoning potential than SFT-based distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19113v1">Exploring the Use of LLMs for Requirements Specification in an IT Consulting Company</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 11 pages, 5 figures. Accepted for presentation at the Industrial Innovation Track of the 33rd IEEE International Requirements Engineering Conference (RE 2025), Valencia, Spain
    </div>
    <details class="paper-abstract">
      In practice, requirements specification remains a critical challenge. The knowledge necessary to generate a specification can often be fragmented across diverse sources (e.g., meeting minutes, emails, and high-level product descriptions), making the process cumbersome and time-consuming. In this paper, we report our experience using large language models (LLMs) in an IT consulting company to automate the requirements specification process. In this company, requirements are specified using a Functional Design Specification (FDS), a document that outlines the functional requirements and features of a system, application, or process. We provide LLMs with a summary of the requirements elicitation documents and FDS templates, prompting them to generate Epic FDS (including high-level product descriptions) and user stories, which are subsequently compiled into a complete FDS document. We compared the correctness and quality of the FDS generated by three state-of-the-art LLMs against those produced by human analysts. Our results show that LLMs can help automate and standardize the requirements specification, reducing time and human effort. However, the quality of LLM-generated FDS highly depends on inputs and often requires human revision. Thus, we advocate for a synergistic approach in which an LLM serves as an effective drafting tool while human analysts provide the critical contextual and technical oversight necessary for high-quality requirements engineering (RE) documentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12591v2">LLMs are Also Effective Embedding Models: An In-depth Overview</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 38 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized natural language processing by achieving state-of-the-art performance across various tasks. Recently, their effectiveness as embedding models has gained attention, marking a paradigm shift from traditional encoder-only models like ELMo and BERT to decoder-only, large-scale LLMs such as GPT, LLaMA, and Mistral. This survey provides an in-depth overview of this transition, beginning with foundational techniques before the LLM era, followed by LLM-based embedding models through two main strategies to derive embeddings from LLMs. 1) Direct prompting: We mainly discuss the prompt designs and the underlying rationale for deriving competitive embeddings. 2) Data-centric tuning: We cover extensive aspects that affect tuning an embedding model, including model architecture, training objectives, data constructions, etc. Upon the above, we also cover advanced methods for producing embeddings from longer texts, multilingual, code, cross-modal data, as well as reasoning-aware and other domain-specific scenarios. Furthermore, we discuss factors affecting choices of embedding models, such as performance/efficiency comparisons, dense vs sparse embeddings, pooling strategies, and scaling law. Lastly, the survey highlights the limitations and challenges in adapting LLMs for embeddings, including cross-task embedding quality, trade-offs between efficiency and accuracy, low-resource, long-context, data bias, robustness, etc. This survey serves as a valuable resource for researchers and practitioners by synthesizing current advancements, highlighting key challenges, and offering a comprehensive framework for future work aimed at enhancing the effectiveness and efficiency of LLMs as embedding models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16331v2">Re:Form -- Reducing Human Priors in Scalable Formal Software Verification with RL in LLMs: A Preliminary Study on Dafny</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Existing informal language-based (e.g., human language) Large Language Models (LLMs) trained with Reinforcement Learning (RL) face a significant challenge: their verification processes, which provide crucial training signals, are neither reliable nor scalable. In fact, the prevalent large proprietary models could hardly generate verifiable programs. A promising yet largely uncharted alternative is formal language-based reasoning. Grounding LLMs in rigorous formal systems where generative models operate in formal language spaces (e.g., Dafny) enables the automatic and mathematically provable verification of their reasoning processes and outcomes. This capability is pivotal for achieving large-scale, reliable formal software verification. It is a common practice to employ human-annotated chain-of-thought and other human priors to induce the reasoning and coding capabilities of LLMs. Unfortunately, it becomes unacceptably all-consuming to provide such priors for supervising complex programming tasks. In this work, we systematically explore ways to reduce human priors with the formal language, Dafny, as the main environment for our pilot study. Our pipeline mainly relies on introducing an automatic and scalable data curation pipeline, and careful RL designs integrated with feedback from the formal language verifier. We introduce DafnyComp, a benchmark of compositional formal programs with auto-formalized specifications for specification reasoning. Our supervised fine-tuning (SFT) stage enables even small models (e.g., 0.5B) to generate syntactically valid and verifiable Dafny code, surpassing proprietary models. RL with regularization further improves performance, achieving stronger generalization to out-of-domain tasks and outperforming all strong baselines on the challenging DafnyComp benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.06270v4">XAI4LLM. Let Machine Learning Models and LLMs Collaborate for Enhanced In-Context Learning in Healthcare</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Clinical decision support systems require models that are not only highly accurate but also equitable and sensitive to the implications of missed diagnoses. In this study, we introduce a knowledge-guided in-context learning (ICL) framework designed to enable large language models (LLMs) to effectively process structured clinical data. Our approach integrates domain-specific feature groupings, carefully balanced few-shot examples, and task-specific prompting strategies. We systematically evaluate this method across seventy distinct ICL designs by various prompt variations and two different communication styles-natural-language narrative and numeric conversational-and compare its performance to robust classical machine learning (ML) benchmarks on tasks involving heart disease and diabetes prediction. Our findings indicate that while traditional ML models maintain superior performance in balanced precision-recall scenarios, LLMs employing narrative prompts with integrated domain knowledge achieve higher recall and significantly reduce gender bias, effectively narrowing fairness disparities by an order of magnitude. Despite the current limitation of increased inference latency, LLMs provide notable advantages, including the capacity for zero-shot deployment and enhanced equity. This research offers the first comprehensive analysis of ICL design considerations for applying LLMs to tabular clinical tasks and highlights distillation and multimodal extensions as promising directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19033v1">SelfRACG: Enabling LLMs to Self-Express and Retrieve for Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 Tsinghua&Xiaohongshu
    </div>
    <details class="paper-abstract">
      Existing retrieval-augmented code generation (RACG) methods typically use an external retrieval module to fetch semantically similar code snippets used for generating subsequent fragments. However, even for consecutive code fragments, the content often diverges due to logical progression, resulting in a content gap. This gap undermines the performance of current RACG methods, as \textit{external} retrieval modules based on content matching fail to infer the specific information need of LLMs to generate the next code fragment. Therefore, we propose \textbf{SelfRACG}, a novel paradigm that enables large language models (LLMs) to \textbf{Self}-express their information needs to enhance \textbf{RACG}. Specifically, SelfRACG includes an information need expression module and a two-stage information need-guided training strategy, which encourages LLMs to express their information need. Extensive experiments demonstrate that SelfRACG can retrieve external knowledge that better aligns with the LLM's own information needs, resulting in superior generation performance compared to vanilla RACG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19027v1">SESR-Eval: Dataset for Evaluating LLMs in the Title-Abstract Screening of Systematic Reviews</a></div>
    <div class="paper-meta">
      📅 2025-07-25
      | 💬 12 pages (10 + 2 pages for references)
    </div>
    <details class="paper-abstract">
      Background: The use of large language models (LLMs) in the title-abstract screening process of systematic reviews (SRs) has shown promising results, but suffers from limited performance evaluation. Aims: Create a benchmark dataset to evaluate the performance of LLMs in the title-abstract screening process of SRs. Provide evidence whether using LLMs in title-abstract screening in software engineering is advisable. Method: We start with 169 SR research artifacts and find 24 of those to be suitable for inclusion in the dataset. Using the dataset we benchmark title-abstract screening using 9 LLMs. Results: We present the SESR-Eval (Software Engineering Systematic Review Evaluation) dataset containing 34,528 labeled primary studies, sourced from 24 secondary studies published in software engineering (SE) journals. Most LLMs performed similarly and the differences in screening accuracy between secondary studies are greater than differences between LLMs. The cost of using an LLM is relatively low - less than $40 per secondary study even for the most expensive model. Conclusions: Our benchmark enables monitoring AI performance in the screening task of SRs in software engineering. At present, LLMs are not yet recommended for automating the title-abstract screening process, since accuracy varies widely across secondary studies, and no LLM managed a high recall with reasonable precision. In future, we plan to investigate factors that influence LLM screening performance between studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18631v2">Layer-Aware Representation Filtering: Purifying Finetuning Data to Preserve LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      With rapid advancement and increasing accessibility of LLMs, fine-tuning aligned models has become a critical step for adapting them to real-world applications, which makes the safety of this fine-tuning process more important than ever. However, recent studies have highlighted a critical challenge: even when fine-tuning with seemingly benign downstream datasets, the safety of aligned LLMs can be compromised, making them more susceptible to malicious instructions. In this paper, we show that fine-tuning datasets often contain samples with safety-degrading features that are not easily identifiable on the surface. These samples can significantly degrade the safety alignment of LLMs during fine-tuning. To address this issue, we propose LARF, a Layer-Aware Representation Filtering method. This method identifies safety-sensitive layers within the LLM and leverages their representations to detect which data samples in the post-training dataset contain safety-degrading features. Experimental results demonstrate that LARF can effectively identify benign data with safety-degrading features. After removing such data, the safety alignment degradation caused by fine-tuning is mitigated. Please see our code at https://github.com/LLLeoLi/LARF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.19353v1">Smooth Reading: Bridging the Gap of Recurrent LLM to Self-Attention LLM on Long-Context Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-25
    </div>
    <details class="paper-abstract">
      Recently, recurrent large language models (Recurrent LLMs) with linear computational complexity have re-emerged as efficient alternatives to self-attention-based LLMs (Self-Attention LLMs), which have quadratic complexity. However, Recurrent LLMs often underperform on long-context tasks due to their limited fixed-size memory. Previous research has primarily focused on enhancing the memory capacity of Recurrent LLMs through architectural innovations, but these approaches have not yet enabled Recurrent LLMs to match the performance of Self-Attention LLMs on long-context tasks. We argue that this limitation arises because processing the entire context at once is not well-suited for Recurrent LLMs. In this paper, we propose Smooth Reading, a chunk-wise inference method inspired by human reading strategies. Smooth Reading processes context in chunks and iteratively summarizes the contextual information, thereby reducing memory demands and making the approach more compatible with Recurrent LLMs. Our experimental results show that this method substantially narrows the performance gap between Recurrent and Self-Attention LLMs on long-context tasks, while preserving the efficiency advantages of Recurrent LLMs. Our Smooth Reading boosts SWA-3B-4k (a Recurrent LLM) from 5.68% lower to 3.61% higher performance than Self-Attention LLMs on LongBench. Besides, our method maintains the high efficiency, training 3x faster and inferring 2x faster at 64k context compared to Self-Attention LLMs. To our knowledge, this is the first work to achieve comparable performance using Recurrent LLMs compared with Self-Attention LLMs on long-context tasks. We hope our method will inspire future research in this area. To facilitate further progress, we will release code and dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12972v2">Aligning Vision to Language: Annotation-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 14 pages, 7 figures, 6 tables; Accepted to ICCV 2025
    </div>
    <details class="paper-abstract">
      Multimodal reasoning in Large Language Models (LLMs) struggles with incomplete knowledge and hallucination artifacts, challenges that textual Knowledge Graphs (KGs) only partially mitigate due to their modality isolation. While Multimodal Knowledge Graphs (MMKGs) promise enhanced cross-modal understanding, their practical construction is impeded by semantic narrowness of manual text annotations and inherent noise in visual-semantic entity linkages. In this paper, we propose Vision-align-to-Language integrated Knowledge Graph (VaLiK), a novel approach for constructing MMKGs that enhances LLMs reasoning through cross-modal information supplementation. Specifically, we cascade pre-trained Vision-Language Models (VLMs) to align image features with text, transforming them into descriptions that encapsulate image-specific information. Furthermore, we developed a cross-modal similarity verification mechanism to quantify semantic consistency, effectively filtering out noise introduced during feature alignment. Even without manually annotated image captions, the refined descriptions alone suffice to construct the MMKG. Compared to conventional MMKGs construction paradigms, our approach achieves substantial storage efficiency gains while maintaining direct entity-to-image linkage capability. Experimental results on multimodal reasoning tasks demonstrate that LLMs augmented with VaLiK outperform previous state-of-the-art models. Our code is published at https://github.com/Wings-Of-Disaster/VaLiK.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10009v2">OR-LLM-Agent: Automating Modeling and Solving of Operations Research Optimization Problems with Reasoning LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 8 pages, 12 figures
    </div>
    <details class="paper-abstract">
      With the rise of artificial intelligence (AI), applying large language models (LLMs) to Operations Research (OR) problem-solving has attracted increasing attention. Most existing approaches attempt to improve OR problem-solving through prompt engineering or fine-tuning strategies for LLMs. However, these methods are fundamentally constrained by the limited capabilities of non-reasoning LLMs. To overcome these limitations, we propose OR-LLM-Agent, an AI agent built on reasoning LLMs for automated OR problem solving. The agent decomposes the task into three sequential stages: mathematical modeling, code generation, and debugging. Each task is handled by a dedicated sub-agent, which enables more targeted reasoning. We also construct BWOR, a high-quality dataset for evaluating LLM performance on OR tasks. Our analysis shows that existing benchmarks such as NL4OPT, MAMO, and IndustryOR suffer from certain issues, making them less suitable for reliably evaluating LLM performance. In contrast, BWOR provides a more consistent and discriminative assessment of model capabilities. Experimental results demonstrate that OR-LLM-Agent outperforms advanced methods, including GPT-o3, Gemini 2.5 Pro, and ORLM, by at least 7% in accuracy. These results demonstrate the effectiveness of task decomposition for OR problem solving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19795v2">VolDoGer: LLM-assisted Datasets for Domain Generalization in Vision-Language Tasks</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 ICCV 2025 Workshop on Curated Data for Efficient Learning (CDEL)
    </div>
    <details class="paper-abstract">
      Domain generalizability is a crucial aspect of a deep learning model since it determines the capability of the model to perform well on data from unseen domains. However, research on the domain generalizability of deep learning models for vision-language tasks remains limited, primarily because of the lack of required datasets. To address these challenges, we propose VolDoGer: Vision-Language Dataset for Domain Generalization, a dedicated dataset designed for domain generalization that addresses three vision-language tasks: image captioning, visual question answering, and visual entailment. We constructed VolDoGer by extending LLM-based data annotation techniques to vision-language tasks, thereby alleviating the burden of recruiting human annotators. We evaluated the domain generalizability of various models, ranging from fine-tuned models to a recent multimodal large language model, through VolDoGer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18252v1">Multimodal Behavioral Patterns Analysis with Eye-Tracking and LLM-Based Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Eye-tracking data reveals valuable insights into users' cognitive states but is difficult to analyze due to its structured, non-linguistic nature. While large language models (LLMs) excel at reasoning over text, they struggle with temporal and numerical data. This paper presents a multimodal human-AI collaborative framework designed to enhance cognitive pattern extraction from eye-tracking signals. The framework includes: (1) a multi-stage pipeline using horizontal and vertical segmentation alongside LLM reasoning to uncover latent gaze patterns; (2) an Expert-Model Co-Scoring Module that integrates expert judgment with LLM output to generate trust scores for behavioral interpretations; and (3) a hybrid anomaly detection module combining LSTM-based temporal modeling with LLM-driven semantic analysis. Our results across several LLMs and prompt strategies show improvements in consistency, interpretability, and performance, with up to 50% accuracy in difficulty prediction tasks. This approach offers a scalable, interpretable solution for cognitive modeling and has broad potential in adaptive learning, human-computer interaction, and educational analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18215v1">Information Security Based on LLM Approaches: A Review</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Information security is facing increasingly severe challenges, and traditional protection means are difficult to cope with complex and changing threats. In recent years, as an emerging intelligent technology, large language models (LLMs) have shown a broad application prospect in the field of information security. In this paper, we focus on the key role of LLM in information security, systematically review its application progress in malicious behavior prediction, network threat analysis, system vulnerability detection, malicious code identification, and cryptographic algorithm optimization, and explore its potential in enhancing security protection performance. Based on neural networks and Transformer architecture, this paper analyzes the technical basis of large language models and their advantages in natural language processing tasks. It is shown that the introduction of large language modeling helps to improve the detection accuracy and reduce the false alarm rate of security systems. Finally, this paper summarizes the current application results and points out that it still faces challenges in model transparency, interpretability, and scene adaptability, among other issues. It is necessary to explore further the optimization of the model structure and the improvement of the generalization ability to realize a more intelligent and accurate information security protection system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18212v1">Prune&Comp: Free Lunch for Layer-Pruned LLMs via Iterative Pruning with Magnitude Compensation</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Layer pruning has emerged as a promising technique for compressing large language models (LLMs) while achieving acceleration proportional to the pruning ratio. In this work, we identify that removing any layer induces a significant magnitude gap in hidden states, resulting in substantial performance degradation. To address this issue, we propose Prune&Comp, a novel plug-and-play layer pruning scheme that leverages magnitude compensation to mitigate such gaps in a training-free manner. Specifically, we first estimate the magnitude gap caused by layer removal and then eliminate this gap by rescaling the remaining weights offline, with zero runtime overhead incurred. We further demonstrate the advantages of Prune&Comp through an iterative pruning strategy. When integrated with an iterative prune-and-compensate loop, Prune&Comp consistently enhances existing layer pruning metrics. For instance, when 5 layers of LLaMA-3-8B are pruned using the prevalent block influence metric, Prune&Comp nearly halves the perplexity and retains 93.19\% of the original model's question-answering performance, outperforming the baseline by 4.01%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20658v2">Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 11 pages, 5 figures, published to ACL 2025
    </div>
    <details class="paper-abstract">
      Temporal Logic (TL), especially Signal Temporal Logic (STL), enables precise formal specification, making it widely used in cyber-physical systems such as autonomous driving and robotics. Automatically transforming NL into STL is an attractive approach to overcome the limitations of manual transformation, which is time-consuming and error-prone. However, due to the lack of datasets, automatic transformation currently faces significant challenges and has not been fully explored. In this paper, we propose an NL-STL dataset named STL-Diversity-Enhanced (STL-DivEn), which comprises 16,000 samples enriched with diverse patterns. To develop the dataset, we first manually create a small-scale seed set of NL-STL pairs. Next, representative examples are identified through clustering and used to guide large language models (LLMs) in generating additional NL-STL pairs. Finally, diversity and accuracy are ensured through rigorous rule-based filters and human validation. Furthermore, we introduce the Knowledge-Guided STL Transformation (KGST) framework, a novel approach for transforming natural language into STL, involving a generate-then-refine process based on external knowledge. Statistical analysis shows that the STL-DivEn dataset exhibits more diversity than the existing NL-STL dataset. Moreover, both metric-based and human evaluations indicate that our KGST approach outperforms baseline models in transformation accuracy on STL-DivEn and DeepSTL datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18203v1">Exploring the Impact of Instruction-Tuning on LLM's Susceptibility to Misinformation</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 ACL 2025 Main Accepted
    </div>
    <details class="paper-abstract">
      Instruction-tuning enhances the ability of large language models (LLMs) to follow user instructions more accurately, improving usability while reducing harmful outputs. However, this process may increase the model's dependence on user input, potentially leading to the unfiltered acceptance of misinformation and the generation of hallucinations. Existing studies primarily highlight that LLMs are receptive to external information that contradict their parametric knowledge, but little research has been conducted on the direct impact of instruction-tuning on this phenomenon. In our study, we investigate the impact of instruction-tuning on LLM's susceptibility to misinformation. Our analysis reveals that instruction-tuned LLMs are significantly more likely to accept misinformation when it is presented by the user. A comparison with base models shows that instruction-tuning increases reliance on user-provided information, shifting susceptibility from the assistant role to the user role. Furthermore, we explore additional factors influencing misinformation susceptibility, such as the role of the user in prompt structure, misinformation length, and the presence of warnings in the system prompt. Our findings underscore the need for systematic approaches to mitigate unintended consequences of instruction-tuning and enhance the reliability of LLMs in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18181v1">SpecASR: Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based automatic speech recognition (ASR) has recently attracted a lot of attention due to its high recognition accuracy and enhanced multi-dialect support. However, the high decoding latency of LLMs challenges the real-time ASR requirements. Although speculative decoding has been explored for better decoding efficiency, they usually ignore the key characteristics of the ASR task and achieve limited speedup. To further reduce the real-time ASR latency, in this paper, we propose a novel speculative decoding framework specialized for ASR, dubbed SpecASR. SpecASR is developed based on our core observation that ASR decoding is audio-conditioned, which results in high output alignment between small and large ASR models, even given output mismatches in intermediate decoding steps. Therefore, SpecASR features an adaptive draft sequence generation process that dynamically modifies the draft sequence length to maximize the token acceptance length. SpecASR further proposes a draft sequence recycling strategy that reuses the previously generated draft sequence to reduce the draft ASR model latency. Moreover, a two-pass sparse token tree generation algorithm is also proposed to balance the latency of draft and target ASR models. With extensive experimental results, we demonstrate SpecASR achieves 3.04x-3.79x and 1.25x-1.84x speedup over the baseline autoregressive decoding and speculative decoding, respectively, without any loss in recognition accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18178v1">Decoupling Knowledge and Reasoning in LLMs: An Exploration Using Cognitive Dual-System Theory</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) leverage both knowledge and reasoning during inference, the capacity to distinguish between them plays a pivotal role in model analysis, interpretability, and development. Inspired by dual-system cognitive theory, we propose a cognition attribution framework to decouple the contribution of knowledge and reasoning. In particular, the cognition of LLMs is decomposed into two distinct yet complementary phases: knowledge retrieval (Phase 1) and reasoning adjustment (Phase 2). To separate these phases, LLMs are prompted to generate answers under two different cognitive modes, fast thinking and slow thinking, respectively. The performance under different cognitive modes is analyzed to quantify the contribution of knowledge and reasoning. This architecture is employed to 15 LLMs across 3 datasets. Results reveal: (1) reasoning adjustment is domain-specific, benefiting reasoning-intensive domains (e.g., mathematics, physics, and chemistry) and potentially imparing knowledge-intensive domains. (2) Parameter scaling improves both knowledge and reasoning, with knowledge improvements being more pronounced. Additionally, parameter scaling make LLMs reasoning significantly more prudent, while moderately more intelligent. (3) Knowledge primarily resides in lower network layers, while reasoning operates in higher layers. Our framework not only helps understand LLMs from a "decoupling" perspective, but also provides new insights into existing research, including scaling laws, hierarchical knowledge editing, and limitations of small-model reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17723v2">Statistical Runtime Verification for LLMs via Robustness Estimation</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 20 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Adversarial robustness verification is essential for ensuring the safe deployment of Large Language Models (LLMs) in runtime-critical applications. However, formal verification techniques remain computationally infeasible for modern LLMs due to their exponential runtime and white-box access requirements. This paper presents a case study adapting and extending the RoMA statistical verification framework to assess its feasibility as an online runtime robustness monitor for LLMs in black-box deployment settings. Our adaptation of RoMA analyzes confidence score distributions under semantic perturbations to provide quantitative robustness assessments with statistically validated bounds. Our empirical validation against formal verification baselines demonstrates that RoMA achieves comparable accuracy (within 1\% deviation), and reduces verification times from hours to minutes. We evaluate this framework across semantic, categorial, and orthographic perturbation domains. Our results demonstrate RoMA's effectiveness for robustness monitoring in operational LLM deployments. These findings point to RoMA as a potentially scalable alternative when formal methods are infeasible, with promising implications for runtime verification in LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18165v1">ProactiveVA: Proactive Visual Analytics with LLM-Based UI Agent</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 11 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Visual analytics (VA) is typically applied to complex data, thus requiring complex tools. While visual analytics empowers analysts in data analysis, analysts may get lost in the complexity occasionally. This highlights the need for intelligent assistance mechanisms. However, even the latest LLM-assisted VA systems only provide help when explicitly requested by the user, making them insufficiently intelligent to offer suggestions when analysts need them the most. We propose a ProactiveVA framework in which LLM-powered UI agent monitors user interactions and delivers context-aware assistance proactively. To design effective proactive assistance, we first conducted a formative study analyzing help-seeking behaviors in user interaction logs, identifying when users need proactive help, what assistance they require, and how the agent should intervene. Based on this analysis, we distilled key design requirements in terms of intent recognition, solution generation, interpretability and controllability. Guided by these requirements, we develop a three-stage UI agent pipeline including perception, reasoning, and acting. The agent autonomously perceives users' needs from VA interaction logs, providing tailored suggestions and intuitive guidance through interactive exploration of the system. We implemented the framework in two representative types of VA systems, demonstrating its generalizability, and evaluated the effectiveness through an algorithm evaluation, case and expert study and a user study. We also discuss current design trade-offs of proactive VA and areas for further exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18153v1">When Noisy Labels Meet Class Imbalance on Graphs: A Graph Augmentation Method with LLM and Pseudo Label</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Class-imbalanced graph node classification is a practical yet underexplored research problem. Although recent studies have attempted to address this issue, they typically assume clean and reliable labels when processing class-imbalanced graphs. This assumption often violates the nature of real-world graphs, where labels frequently contain noise. Given this gap, this paper systematically investigates robust node classification for class-imbalanced graphs with noisy labels. We propose GraphALP, a novel Graph Augmentation framework based on Large language models (LLMs) and Pseudo-labeling techniques. Specifically, we design an LLM-based oversampling method to generate synthetic minority nodes, producing label-accurate minority nodes to alleviate class imbalance. Based on the class-balanced graphs, we develop a dynamically weighted pseudo-labeling method to obtain high-confidence pseudo labels to reduce label noise ratio. Additionally, we implement a secondary LLM-guided oversampling mechanism to mitigate potential class distribution skew caused by pseudo labels. Experimental results show that GraphALP achieves superior performance over state-of-the-art methods on class-imbalanced graphs with noisy labels.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05606v4">OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Can large language models (LLMs) accurately simulate the next web action of a specific user? While LLMs have shown promising capabilities in generating ``believable'' human behaviors, evaluating their ability to mimic real user behaviors remains an open challenge, largely due to the lack of high-quality, publicly available datasets that capture both the observable actions and the internal reasoning of an actual human user. To address this gap, we introduce OPERA, a novel dataset of Observation, Persona, Rationale, and Action collected from real human participants during online shopping sessions. OPERA is the first public dataset that comprehensively captures: user personas, browser observations, fine-grained web actions, and self-reported just-in-time rationales. We developed both an online questionnaire and a custom browser plugin to gather this dataset with high fidelity. Using OPERA, we establish the first benchmark to evaluate how well current LLMs can predict a specific user's next action and rationale with a given persona and <observation, action, rationale> history. This dataset lays the groundwork for future research into LLM agents that aim to act as personalized digital twins for human.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15690v3">LLM Web Dynamics: Tracing Model Collapse in a Network of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      The increasing use of synthetic data from the public Internet has enhanced data usage efficiency in large language model (LLM) training. However, the potential threat of model collapse remains insufficiently explored. Existing studies primarily examine model collapse in a single model setting or rely solely on statistical surrogates. In this work, we introduce LLM Web Dynamics (LWD), an efficient framework for investigating model collapse at the network level. By simulating the Internet with a retrieval-augmented generation (RAG) database, we analyze the convergence pattern of model outputs. Furthermore, we provide theoretical guarantees for this convergence by drawing an analogy to interacting Gaussian Mixture Models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14928v2">EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 Paper URL: https://aclanthology.org/2025.acl-long.1576/; Presentation Video: https://www.youtube.com/watch?v=j63ooKE50I0
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly serve as educational tools, yet evaluating their teaching capabilities remains challenging due to the resource-intensive, context-dependent, and methodologically complex nature of teacher-student interactions. We introduce EducationQ, a multi-agent dialogue framework that efficiently assesses teaching capabilities through simulated dynamic educational scenarios, featuring specialized agents for teaching, learning, and evaluation. Testing 14 LLMs across major AI Organizations (OpenAI, Meta, Google, Anthropic, and others) on 1,498 questions spanning 13 disciplines and 10 difficulty levels reveals that teaching effectiveness does not correlate linearly with model scale or general reasoning capabilities - with some smaller open-source models outperforming larger commercial counterparts in teaching contexts. This finding highlights a critical gap in current evaluations that prioritize knowledge recall over interactive pedagogy. Our mixed-methods evaluation, combining quantitative metrics with qualitative analysis and expert case studies, identifies distinct pedagogical strengths employed by top-performing models (e.g., sophisticated questioning strategies, adaptive feedback mechanisms). Human expert evaluations show 78% agreement with our automated qualitative analysis of effective teaching behaviors, validating our methodology. EducationQ demonstrates that LLMs-as-teachers require specialized optimization beyond simple scaling, suggesting next-generation educational AI prioritize targeted enhancement of specific pedagogical effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18073v1">Squeeze10-LLM: Squeezing LLMs' Weights by 10 Times via a Staged Mixed-Precision Quantization Method</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) is challenging due to their massive parameters and high computational costs. Ultra low-bit quantization can significantly reduce storage and accelerate inference, but extreme compression (i.e., mean bit-width <= 2) often leads to severe performance degradation. To address this, we propose Squeeze10-LLM, effectively "squeezing" 16-bit LLMs' weights by 10 times. Specifically, Squeeze10-LLM is a staged mixed-precision post-training quantization (PTQ) framework and achieves an average of 1.6 bits per weight by quantizing 80% of the weights to 1 bit and 20% to 4 bits. We introduce Squeeze10LLM with two key innovations: Post-Binarization Activation Robustness (PBAR) and Full Information Activation Supervision (FIAS). PBAR is a refined weight significance metric that accounts for the impact of quantization on activations, improving accuracy in low-bit settings. FIAS is a strategy that preserves full activation information during quantization to mitigate cumulative error propagation across layers. Experiments on LLaMA and LLaMA2 show that Squeeze10-LLM achieves state-of-the-art performance for sub-2bit weight-only quantization, improving average accuracy from 43% to 56% on six zero-shot classification tasks--a significant boost over existing PTQ methods. Our code will be released upon publication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.01144v5">BlockDialect: Block-wise Fine-grained Mixed Format Quantization for Energy-Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      The rapidly increasing size of large language models (LLMs) presents significant challenges in memory usage and computational costs. Quantizing both weights and activations can address these issues, with hardware-supported fine-grained scaling emerging as a promising solution to mitigate outliers. However, existing methods struggle to capture nuanced block data distributions. We propose BlockDialect, a block-wise fine-grained mixed format technique that assigns a per-block optimal number format from a formatbook for better data representation. Additionally, we introduce DialectFP4, a formatbook of FP4 variants (akin to dialects) that adapt to diverse data distributions. To leverage this efficiently, we propose a two-stage approach for online DialectFP4 activation quantization. Importantly, DialectFP4 ensures energy efficiency by selecting representable values as scaled integers compatible with low-precision integer arithmetic. BlockDialect achieves 10.78% (7.48%) accuracy gain on the LLaMA3-8B (LLaMA2-7B) model compared to MXFP4 format with lower bit usage per data, while being only 5.45% (2.69%) below full precision even when quantizing full-path matrix multiplication. Focusing on how to represent over how to scale, our work presents a promising path for energy-efficient LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2212.10678v4">Causally Testing Gender Bias in LLMs: A Case Study on Occupational Bias</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Generated texts from large language models (LLMs) have been shown to exhibit a variety of harmful, human-like biases against various demographics. These findings motivate research efforts aiming to understand and measure such effects. This paper introduces a causal formulation for bias measurement in generative language models. Based on this theoretical foundation, we outline a list of desiderata for designing robust bias benchmarks. We then propose a benchmark called OccuGender, with a bias-measuring procedure to investigate occupational gender bias. We test several state-of-the-art open-source LLMs on OccuGender, including Llama, Mistral, and their instruction-tuned versions. The results show that these models exhibit substantial occupational gender bias. Lastly, we discuss prompting strategies for bias mitigation and an extension of our causal formulation to illustrate the generalizability of our framework. Our code and data https://github.com/chenyuen0103/gender-bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18055v1">Privacy-Preserving Synthetic Review Generation with Diverse Writing Styles Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      The increasing use of synthetic data generated by Large Language Models (LLMs) presents both opportunities and challenges in data-driven applications. While synthetic data provides a cost-effective, scalable alternative to real-world data to facilitate model training, its diversity and privacy risks remain underexplored. Focusing on text-based synthetic data, we propose a comprehensive set of metrics to quantitatively assess the diversity (i.e., linguistic expression, sentiment, and user perspective), and privacy (i.e., re-identification risk and stylistic outliers) of synthetic datasets generated by several state-of-the-art LLMs. Experiment results reveal significant limitations in LLMs' capabilities in generating diverse and privacy-preserving synthetic data. Guided by the evaluation results, a prompt-based approach is proposed to enhance the diversity of synthetic reviews while preserving reviewer privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18043v1">GrAInS: Gradient-based Attribution for Inference-Time Steering of LLMs and VLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 21 pages. Code: https://github.com/duykhuongnguyen/GrAInS
    </div>
    <details class="paper-abstract">
      Inference-time steering methods offer a lightweight alternative to fine-tuning large language models (LLMs) and vision-language models (VLMs) by modifying internal activations at test time without updating model weights. However, most existing approaches rely on fixed, global intervention vectors, overlook the causal influence of individual input tokens, and fail to leverage informative gradients from the model's logits, particularly in multimodal settings where visual and textual inputs contribute unevenly. To address these limitations, we introduce GrAInS, an inference-time steering approach that operates across both language-only and vision-language models and tasks. GrAInS uses contrastive, gradient-based attribution via Integrated Gradients to identify the top-k most influential tokens, both positively and negatively attributed based on their contribution to preferred versus dispreferred outputs. These tokens are then used to construct directional steering vectors that capture semantic shifts from undesirable to desirable behavior. During inference, GrAInS adjusts hidden activations at transformer layers guided by token-level attribution signals, and normalizes activations to preserve representational scale. This enables fine-grained, interpretable, and modular control over model behavior, without retraining or auxiliary supervision. Empirically, GrAInS consistently outperforms both fine-tuning and existing steering baselines: it achieves a 13.22% accuracy gain on TruthfulQA using Llama-3.1-8B, reduces hallucination rates on MMHal-Bench from 0.624 to 0.514 with LLaVA-1.6-7B, and improves alignment win rates on SPA-VL by 8.11%, all while preserving the model's fluency and general capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18028v1">NeuralDB: Scaling Knowledge Editing in LLMs to 100,000 Facts with Neural KV Database</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Efficiently editing knowledge stored in large language models (LLMs) enables model updates without large-scale training. One possible solution is Locate-and-Edit (L\&E), allowing simultaneous modifications of a massive number of facts. However, such editing may compromise the general abilities of LLMs and even result in forgetting edited facts when scaling up to thousands of edits. In this paper, we model existing linear L\&E methods as querying a Key-Value (KV) database. From this perspective, we then propose NeuralDB, an editing framework that explicitly represents the edited facts as a neural KV database equipped with a non-linear gated retrieval module, % In particular, our gated module only operates when inference involves the edited facts, effectively preserving the general abilities of LLMs. Comprehensive experiments involving the editing of 10,000 facts were conducted on the ZsRE and CounterFacts datasets, using GPT2-XL, GPT-J (6B) and Llama-3 (8B). The results demonstrate that NeuralDB not only excels in editing efficacy, generalization, specificity, fluency, and consistency, but also preserves overall performance across six representative text understanding and generation tasks. Further experiments indicate that NeuralDB maintains its effectiveness even when scaled to 100,000 facts (\textbf{50x} more than in prior work).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18009v1">GRR-CoCa: Leveraging LLM Mechanisms in Multimodal Model Architectures</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 12 pages, 2 figures
    </div>
    <details class="paper-abstract">
      State-of-the-art (SOTA) image and text generation models are multimodal models that have many similarities to large language models (LLMs). Despite achieving strong performances, leading foundational multimodal model architectures frequently lag behind the architectural sophistication of contemporary LLMs. We propose GRR-CoCa, an improved SOTA Contrastive Captioner (CoCa) model that incorporates Gaussian error gated linear units, root mean squared normalization, and rotary positional embedding into the textual decoders and the vision transformer (ViT) encoder. Each architectural modification has been shown to improve model performance in LLMs, but has yet to be adopted in CoCa. We benchmarked GRR-CoCa against Baseline CoCa, a model with the same modified textual decoders but with CoCa's original ViT encoder. We used standard pretraining and fine-tuning workflows to benchmark the models on contrastive and generative tasks. Our GRR-CoCa significantly outperformed Baseline CoCa on the pretraining dataset and three diverse fine-tuning datasets. Pretraining improvements were 27.25% in contrastive loss, 3.71% in perplexity, and 7.15% in CoCa loss. The average fine-tuning improvements were 13.66% in contrastive loss, 5.18% in perplexity, and 5.55% in CoCa loss. We show that GRR-CoCa's modified architecture improves performance and generalization across vision-language domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18007v1">Cloud Native System for LLM Inference Serving</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are revolutionizing numerous industries, but their substantial computational demands create challenges for efficient deployment, particularly in cloud environments. Traditional approaches to inference serving often struggle with resource inefficiencies, leading to high operational costs, latency issues, and limited scalability. This article explores how Cloud Native technologies, such as containerization, microservices, and dynamic scheduling, can fundamentally improve LLM inference serving. By leveraging these technologies, we demonstrate how a Cloud Native system enables more efficient resource allocation, reduces latency, and enhances throughput in high-demand scenarios. Through real-world evaluations using Kubernetes-based autoscaling, we show that Cloud Native architectures can dynamically adapt to workload fluctuations, mitigating performance bottlenecks while optimizing LLM inference serving performance. This discussion provides a broader perspective on how Cloud Native frameworks could reshape the future of scalable LLM inference serving, offering key insights for researchers, practitioners, and industry leaders in cloud computing and artificial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18006v1">Unlock the Potential of Fine-grained LLM Serving via Dynamic Module Scaling</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      The rise of large language models (LLMs) has created new opportunities across various fields but has also introduced significant challenges in resource management. Current LLM serving systems face a fundamental tension: balancing serving demands with limited resources while adapting to unpredictable traffic patterns. Static deployments lead to suboptimal resource utilization and performance degradation under dynamic workloads. Furthermore, the high cost of adjusting instances hinders dynamic scaling, limiting the true potential of efficient LLM serving. To address this, we propose CoCoServe, an elastic system that facilitates dynamic and fine-grained scaling. Its key innovation lies in the module-level operations for the replication and migration of LLM modules, such as decoder layers and projections. Through a comprehensive analysis of the trade-offs associated with these operations, we develop an auto-scaling mechanism that dynamically regulates module-level resource allocation and performance optimization, enabling a more cost-effective deployment of LLMs. Our evaluation demonstrates that the scaling operations employed by CoCoServe exhibit excellent scalability and can reduce costs by 46% while maintaining availability. Compared to state-of-the-art LLM serving systems (e.g., Hugging Face Transformers and vLLM), our approach reduces latency by 14%-75% and achieves 1.16x-4x throughput on average across different model sizes and workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18273v2">Analyzing Islamophobic Discourse Using Semi-Coded Terms and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      In recent years, Islamophobia has gained significant traction across Western societies, fueled by the rise of digital communication networks. This paper performs a large-scale analysis of specialized, semi-coded Islamophobic terms such as (muzrat, pislam, mudslime, mohammedan, muzzies) floated on extremist social platforms, i.e., 4Chan, Gab, Telegram, etc. Many of these terms appear lexically neutral or ambiguous outside of specific contexts, making them difficult for both human moderators and automated systems to reliably identify as hate speech. First, we use Large Language Models (LLMs) to show their ability to understand these terms. Second, Google Perspective API suggests that Islamophobic posts tend to receive higher toxicity scores than other categories of hate speech like Antisemitism. Finally, we use BERT topic modeling approach to extract different topics and Islamophobic discourse on these social platforms. Our findings indicate that LLMs understand these Out-Of-Vocabulary (OOV) slurs; however, further improvements in moderation strategies and algorithmic detection are necessary to address such discourse effectively. Our topic modeling also indicates that Islamophobic text is found across various political, conspiratorial, and far-right movements and is particularly directed against Muslim immigrants. Taken altogether, we performed one of the first studies on Islamophobic semi-coded terms and shed a global light on Islamophobia.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07623v7">The Curious Case of Class Accuracy Imbalance in LLMs: Post-hoc Debiasing via Nonlinear Integer Programming</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are good knowledge bases but struggle to perform equally well for all classes in text classification. This paper investigates the case of class accuracy imbalance in LLMs, where deeply entangled pretraining biases and prompt-specific cues contribute to the imbalance. To overcome the difficulty in bias identification and inaccessibility of retraining, we post-hoc balance class accuracy using only output probabilities. This is enabled by reformulating debiasing as a combinatorial optimization problem. In details, we first motivate a post-hoc bias metric, the Contextual Oddity Bias (COBias), to quantify the over-/under-prediction (a tendency to over-predict some classes while under-predicting others) in LLMs. We then propose the Debiasing as Nonlinear Integer Programming (DNIP) method to reweight LLM output class probabilities towards minimizing COBias and maximizing overall accuracy, without being constrained by bias sources or updating LLM parameters. Since the DNIP model contains non-differentiable elements, we use simulated annealing to efficiently solve it. Evaluations on five LLMs across NLP classification benchmarks show that DNIP simultaneously achieves significant COBias reduction (61% relative reduction) and accuracy improvement (18% relative increase) under different LLM prompting setups.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18380v2">RedactOR: An LLM-Powered Framework for Automatic Clinical Data De-Identification</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 Accepted to ACL 2025 Industry Track. To appear
    </div>
    <details class="paper-abstract">
      Ensuring clinical data privacy while preserving utility is critical for AI-driven healthcare and data analytics. Existing de-identification (De-ID) methods, including rule-based techniques, deep learning models, and large language models (LLMs), often suffer from recall errors, limited generalization, and inefficiencies, limiting their real-world applicability. We propose a fully automated, multi-modal framework, RedactOR for de-identifying structured and unstructured electronic health records, including clinical audio records. Our framework employs cost-efficient De-ID strategies, including intelligent routing, hybrid rule and LLM based approaches, and a two-step audio redaction approach. We present a retrieval-based entity relexicalization approach to ensure consistent substitutions of protected entities, thereby enhancing data coherence for downstream applications. We discuss key design desiderata, de-identification and relexicalization methodology, and modular architecture of RedactOR and its integration with the Oracle Health Clinical AI system. Evaluated on the i2b2 2014 De-ID dataset using standard metrics with strict recall, our approach achieves competitive performance while optimizing token usage to reduce LLM costs. Finally, we discuss key lessons and insights from deployment in real-world AI- driven healthcare data pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05209v4">Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 Accepted to TMLR
    </div>
    <details class="paper-abstract">
      Evaluations of large language model (LLM) risks and capabilities are increasingly being incorporated into AI risk management and governance frameworks. Currently, most risk evaluations are conducted by designing inputs that elicit harmful behaviors from the system. However, this approach suffers from two limitations. First, input-output evaluations cannot fully evaluate realistic risks from open-weight models. Second, the behaviors identified during any particular input-output evaluation can only lower-bound the model's worst-possible-case input-output behavior. As a complementary method for eliciting harmful behaviors, we propose evaluating LLMs with model tampering attacks which allow for modifications to latent activations or weights. We pit state-of-the-art techniques for removing harmful LLM capabilities against a suite of 5 input-space and 6 model tampering attacks. In addition to benchmarking these methods against each other, we show that (1) model resilience to capability elicitation attacks lies on a low-dimensional robustness subspace; (2) the success rate of model tampering attacks can empirically predict and offer conservative estimates for the success of held-out input-space attacks; and (3) state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning. Together, these results highlight the difficulty of suppressing harmful LLM capabilities and show that model tampering attacks enable substantially more rigorous evaluations than input-space attacks alone.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18812v1">MemoCoder: Automated Function Synthesis using LLM-Supported Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      With the widespread adoption of Large Language Models (LLMs) such as GitHub Copilot and ChatGPT, developers increasingly rely on AI-assisted tools to support code generation. While LLMs can generate syntactically correct solutions for well-structured programming tasks, they often struggle with challenges that require iterative debugging, error handling, or adaptation to diverse problem structures. Existing approaches such as fine-tuning or self-repair strategies either require costly retraining or lack mechanisms to accumulate and reuse knowledge from previous attempts. To address these limitations, we propose MemoCoder, a multi-agent framework that enables collaborative problem solving and persistent learning from past fixes. At the core of MemoCoder is a Fixing Knowledge Set, which stores successful repairs and supports retrieval for future tasks. A central Mentor Agent supervises the repair process by identifying recurring error patterns and refining high-level fixing strategies, providing a novel supervisory role that guides the self-repair loop. We evaluate MemoCoder across three public benchmarks -- MBPP, HumanEval, and LiveCodeBench -- spanning a range of problem complexities. Experimental results show that MemoCoder consistently outperforms both zero-shot prompting and a Self-Repair strategy, with improvements ranging from 3.1% to 12.1% in Pass@10 and from 1.4% to 14.5% in Pass@50, demonstrating its effectiveness in iterative refinement and knowledge-guided code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00151v2">Palm: A Culturally Inclusive and Linguistically Diverse Dataset for Arabic LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 More information about our dataset is available at our project page: https://github.com/UBC-NLP/palm
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly integrated into daily life, ensuring their cultural sensitivity and inclusivity is paramount. We introduce our dataset, a year-long community-driven project covering all 22 Arab countries. The dataset includes instructions (input, response pairs) in both Modern Standard Arabic (MSA) and dialectal Arabic (DA), spanning 20 diverse topics. Built by a team of 44 researchers across the Arab world, all of whom are authors of this paper, our dataset offers a broad, inclusive perspective. We use our dataset to evaluate the cultural and dialectal capabilities of several frontier LLMs, revealing notable limitations. For instance, while closed-source LLMs generally exhibit strong performance, they are not without flaws, and smaller open-source models face greater challenges. Moreover, certain countries (e.g., Egypt, the UAE) appear better represented than others (e.g., Iraq, Mauritania, Yemen). Our annotation guidelines, code, and data for reproducibility are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18802v1">DxHF: Providing High-Quality Human Feedback for LLM Alignment via Interactive Decomposition</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Human preferences are widely used to align large language models (LLMs) through methods such as reinforcement learning from human feedback (RLHF). However, the current user interfaces require annotators to compare text paragraphs, which is cognitively challenging when the texts are long or unfamiliar. This paper contributes by studying the decomposition principle as an approach to improving the quality of human feedback for LLM alignment. This approach breaks down the text into individual claims instead of directly comparing two long-form text responses. Based on the principle, we build a novel user interface DxHF. It enhances the comparison process by showing decomposed claims, visually encoding the relevance of claims to the conversation and linking similar claims. This allows users to skim through key information and identify differences for better and quicker judgment. Our technical evaluation shows evidence that decomposition generally improves feedback accuracy regarding the ground truth, particularly for users with uncertainty. A crowdsourcing study with 160 participants indicates that using DxHF improves feedback accuracy by an average of 5%, although it increases the average feedback time by 18 seconds. Notably, accuracy is significantly higher in situations where users have less certainty. The finding of the study highlights the potential of HCI as an effective method for improving human-AI alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18791v1">Evaluating Code-Mixing in LLMs Across 18 Languages</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Code-mixing, the practice of switching between languages within a conversation, presents unique challenges for traditional natural language processing. Existing benchmarks, such as LinCE and GLUECoS, are limited by narrow language pairings and tasks, failing to adequately evaluate the code-mixing capabilities of large language models (LLMs). Despite the significance of code-mixing for multilingual users, research on LLMs in this context remains limited. Additionally, current methods for generating code-mixed data are underdeveloped. In this paper, we conduct a comprehensive evaluation of LLMs' performance on code-mixed data across 18 languages from seven language families. We also propose a novel approach for generating synthetic code-mixed texts by combining word substitution with GPT-4 prompting. Our analysis reveals consistent underperformance of LLMs on code-mixed datasets involving multiple language families. We suggest that improvements in training data size, model scale, and few-shot learning could enhance their performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14641v3">HLSTester: Efficient Testing of Behavioral Discrepancies with LLMs for High-Level Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 arXiv admin note: text overlap with arXiv:2407.03889
    </div>
    <details class="paper-abstract">
      In high-level synthesis (HLS), C/C++ programs with synthesis directives are used to generate circuits for FPGA implementations. However, hardware-specific and platform-dependent characteristics in these implementations can introduce behavioral discrepancies between the original C/C++ programs and the circuits after high-level synthesis. Existing methods for testing behavioral discrepancies in HLS are still immature, and the testing workflow requires significant human efforts. To address this challenge, we propose HLSTester, a large language model (LLM) aided testing framework that efficiently detects behavioral discrepancies in HLS. To mitigate hallucinations in LLMs and enhance prompt quality, the testbenches for original C/C++ programs are leveraged to guide LLMs in generating HLS-compatible testbenches, effectively eliminating certain traditional C/C++ constructs that are incompatible with HLS tools. Key variables are pinpointed through a backward slicing technique in both C/C++ and HLS programs to monitor their runtime spectra, enabling an in-depth analysis of the discrepancy symptoms. To reduce test time, a testing input generation mechanism is introduced to integrate dynamic mutation with insights from an LLM-based progressive reasoning chain. In addition, repetitive hardware testing is skipped by a redundancy-aware filtering technique for the generated test inputs. Experimental results demonstrate that the proposed LLM-aided testing framework significantly accelerates the testing workflow while achieving higher testbench simulation pass rates compared with the traditional method and the direct use of LLMs on the same HLS programs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.19549v1">AccessGuru: Leveraging LLMs to Detect and Correct Web Accessibility Violations in HTML Code</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      The vast majority of Web pages fail to comply with established Web accessibility guidelines, excluding a range of users with diverse abilities from interacting with their content. Making Web pages accessible to all users requires dedicated expertise and additional manual efforts from Web page providers. To lower their efforts and promote inclusiveness, we aim to automatically detect and correct Web accessibility violations in HTML code. While previous work has made progress in detecting certain types of accessibility violations, the problem of automatically detecting and correcting accessibility violations remains an open challenge that we address. We introduce a novel taxonomy classifying Web accessibility violations into three key categories - Syntactic, Semantic, and Layout. This taxonomy provides a structured foundation for developing our detection and correction method and redefining evaluation metrics. We propose a novel method, AccessGuru, which combines existing accessibility testing tools and Large Language Models (LLMs) to detect violations and applies taxonomy-driven prompting strategies to correct all three categories. To evaluate these capabilities, we develop a benchmark of real-world Web accessibility violations. Our benchmark quantifies syntactic and layout compliance and judges semantic accuracy through comparative analysis with human expert corrections. Evaluation against our benchmark shows that AccessGuru achieves up to 84% average violation score decrease, significantly outperforming prior methods that achieve at most 50%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18631v1">Layer-Aware Representation Filtering: Purifying Finetuning Data to Preserve LLM Safety Alignment</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      With rapid advancement and increasing accessibility of LLMs, fine-tuning aligned models has become a critical step for adapting them to real-world applications, which makes the safety of this fine-tuning process more important than ever. However, recent studies have highlighted a critical challenge: even when fine-tuning with seemingly benign downstream datasets, the safety of aligned LLMs can be compromised, making them more susceptible to malicious instructions. In this paper, we show that fine-tuning datasets often contain samples with safety-degrading features that are not easily identifiable on the surface. These samples can significantly degrade the safety alignment of LLMs during fine-tuning. To address this issue, we propose LARF, a \textbf{L}ayer-\textbf{A}ware \textbf{R}epresentation \textbf{F}iltering method. This method identifies safety-sensitive layers within the LLM and leverages their representations to detect which data samples in the post-training dataset contain safety-degrading features. Experimental results demonstrate that LARF can effectively identify benign data with safety-degrading features. After removing such data, the safety alignment degradation caused by fine-tuning is mitigated. Please see our code at \href{https://github.com/LLLeoLi/LARF}{https://github.com/LLLeoLi/LARF}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18607v1">Explainable Mapper: Charting LLM Embedding Spaces Using Perturbation-Based Explanation and Verification Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) produce high-dimensional embeddings that capture rich semantic and syntactic relationships between words, sentences, and concepts. Investigating the topological structures of LLM embedding spaces via mapper graphs enables us to understand their underlying structures. Specifically, a mapper graph summarizes the topological structure of the embedding space, where each node represents a topological neighborhood (containing a cluster of embeddings), and an edge connects two nodes if their corresponding neighborhoods overlap. However, manually exploring these embedding spaces to uncover encoded linguistic properties requires considerable human effort. To address this challenge, we introduce a framework for semi-automatic annotation of these embedding properties. To organize the exploration process, we first define a taxonomy of explorable elements within a mapper graph such as nodes, edges, paths, components, and trajectories. The annotation of these elements is executed through two types of customizable LLM-based agents that employ perturbation techniques for scalable and automated analysis. These agents help to explore and explain the characteristics of mapper elements and verify the robustness of the generated explanations. We instantiate the framework within a visual analytics workspace and demonstrate its effectiveness through case studies. In particular, we replicate findings from prior research on BERT's embedding properties across various layers of its architecture and provide further observations into the linguistic properties of topological neighborhoods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18584v1">AQuilt: Weaving Logic and Self-Inspection into Low-Cost, High-Relevance Data Synthesis for Specialist LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 32 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Despite the impressive performance of large language models (LLMs) in general domains, they often underperform in specialized domains. Existing approaches typically rely on data synthesis methods and yield promising results by using unlabeled data to capture domain-specific features. However, these methods either incur high computational costs or suffer from performance limitations, while also demonstrating insufficient generalization across different tasks. To address these challenges, we propose AQuilt, a framework for constructing instruction-tuning data for any specialized domains from corresponding unlabeled data, including Answer, Question, Unlabeled data, Inspection, Logic, and Task type. By incorporating logic and inspection, we encourage reasoning processes and self-inspection to enhance model performance. Moreover, customizable task instructions enable high-quality data generation for any task. As a result, we construct a dataset of 703k examples to train a powerful data synthesis model. Experiments show that AQuilt is comparable to DeepSeek-V3 while utilizing just 17% of the production cost. Further analysis demonstrates that our generated data exhibits higher relevance to downstream tasks. Source code, models, and scripts are available at https://github.com/Krueske/AQuilt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16809v2">LingBench++: A Linguistically-Informed Benchmark and Reasoning Framework for Multi-Step and Cross-Cultural Inference with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 42p, 17f, 10t. Revisions: Merged paragraphs in Intro to emphasize contributions. Clarified benchmark design (Sec 3.5.1). Added single-agent, OpenAI-guided & 6-round experiments (Sec 5.2). Note: we only ran each experiment once; statistical tests are needed for strong claims. Revised Sec 6. Added acknowledgements, 2 new co-authors, and corrected typos/grammar
    </div>
    <details class="paper-abstract">
      We propose LingBench++, a linguistically-informed benchmark and reasoning framework designed to evaluate large language models (LLMs) on complex linguistic tasks inspired by the International Linguistics Olympiad (IOL). Unlike prior benchmarks that focus solely on final answer accuracy, LingBench++ provides structured reasoning traces, stepwise evaluation protocols, and rich typological metadata across over 90 low-resource and cross-cultural languages. We further develop a multi-agent architecture integrating grammatical knowledge retrieval, tool-augmented reasoning, and deliberate hypothesis testing. Through systematic comparisons of baseline and our proposed agentic models, we demonstrate that models equipped with external knowledge sources and iterative reasoning outperform single-pass approaches in both accuracy and interpretability. LingBench++ offers a comprehensive foundation for advancing linguistically grounded, culturally informed, and cognitively plausible reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18560v1">HARLF: Hierarchical Reinforcement Learning and Lightweight LLM-Driven Sentiment Integration for Financial Portfolio Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      This paper presents a novel hierarchical framework for portfolio optimization, integrating lightweight Large Language Models (LLMs) with Deep Reinforcement Learning (DRL) to combine sentiment signals from financial news with traditional market indicators. Our three-tier architecture employs base RL agents to process hybrid data, meta-agents to aggregate their decisions, and a super-agent to merge decisions based on market data and sentiment analysis. Evaluated on data from 2018 to 2024, after training on 2000-2017, the framework achieves a 26% annualized return and a Sharpe ratio of 1.2, outperforming equal-weighted and S&P 500 benchmarks. Key contributions include scalable cross-modal integration, a hierarchical RL structure for enhanced stability, and open-source reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14783v2">Omni-Thinker: Scaling Cross-Domain Generalization in LLMs via Multi-Task RL with Hybrid Rewards</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      The advancement of general-purpose artificial intelligence relies on large language models (LLMs) that excel across a wide range of tasks, from structured reasoning to creative generation. However, post-training methods like Supervised Fine-Tuning (SFT) often struggle with generalization, favoring memorization over transferable learning. In this work, we introduce Omni-Thinker, a unified reinforcement learning (RL) framework that enhances LLM performance across diverse tasks by combining rule-based verifiable rewards with generative preference signals via LLM-as-a-Judge evaluations. Our approach enables consistent optimization across task types and scales RL-based training to subjective domains. We further investigate training strategies, demonstrating that a curriculum-based progression that orders tasks from structured to open-ended improves performance and reduces forgetting. Experimental results across four domains reveal that curriculum learning improves performance by 5.2% over joint training and 9.1% over model merging. These results highlight the importance of task-aware sampling and hybrid supervision in scaling RL-based post-training for general-purpose LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18553v1">The Geometry of LLM Quantization: GPTQ as Babai's Nearest Plane Algorithm</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Quantizing the weights of large language models (LLMs) from 16-bit to lower bitwidth is the de facto approach to deploy massive transformers onto more affordable accelerators. GPTQ emerged as one of the standard methods for one-shot post-training quantization at LLM scale. Yet, its inner workings are described as a sequence of ad-hoc algebraic updates that obscure any geometric meaning or worst-case guarantees. In this work, we show that, when executed back-to-front (from the last to first dimension) for a linear layer, GPTQ is mathematically identical to Babai's nearest plane algorithm for the classical closest vector problem (CVP) on a lattice defined by the Hessian matrix of the layer's inputs. This equivalence is based on a sophisticated mathematical argument, and has two analytical consequences: (i) the GPTQ error propagation step gains an intuitive geometric interpretation; (ii) GPTQ inherits the error upper bound of Babai's algorithm under the no-clipping condition. Taken together, these results place GPTQ on firm theoretical footing and open the door to importing decades of progress in lattice algorithms towards the design of future quantization algorithms for billion-parameter models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08989v3">Zeroth-Order Fine-Tuning of LLMs in Random Subspaces</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 ICCV 2025 camera-ready version
    </div>
    <details class="paper-abstract">
      Fine-tuning Large Language Models (LLMs) has proven effective for a variety of downstream tasks. However, as LLMs grow in size, the memory demands for backpropagation become increasingly prohibitive. Zeroth-order (ZO) optimization methods offer a memory-efficient alternative by using forward passes to estimate gradients, but the variance of gradient estimates typically scales linearly with the model's parameter dimension$\unicode{x2013}$a significant issue for LLMs. In this paper, we propose the random Subspace Zeroth-order (SubZero) optimization to address the challenges posed by LLMs' high dimensionality. We introduce a low-rank perturbation tailored for LLMs that significantly reduces memory consumption while improving training performance. Additionally, we prove that our gradient estimation closely approximates the backpropagation gradient, exhibits lower variance than traditional ZO methods, and ensures convergence when combined with SGD. Experimental results show that SubZero enhances fine-tuning performance and achieves faster convergence compared to standard ZO approaches like MeZO across various language modeling tasks. Code is available at https://github.com/zimingyy/SubZero.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02976v2">Are AI-Generated Fixes Secure? Analyzing LLM and Agent Patches on SWE-bench</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and their agentic frameworks are increasingly adopted to automate software development tasks such as issue resolution and program repair. While prior work has identified security risks in LLM-generated code, most evaluations have focused on synthetic or isolated settings, leaving open questions about the security of these systems in real-world development contexts. In this study, we present the first large-scale security analysis of LLM-generated patches using 20,000+ issues from the SWE-bench dataset. We evaluate patches produced by a standalone LLM (Llama 3.3) and compare them to developer-written patches. We also assess the security of patches generated by three top-performing agentic frameworks (OpenHands, AutoCodeRover, HoneyComb) on a subset of our data. Finally, we analyze a wide range of code, issue, and project-level factors to understand the conditions under which LLMs and agents are most likely to generate insecure code. Our findings reveal that the standalone LLM introduces nearly 9x more new vulnerabilities than developers, with many of these exhibiting unique patterns not found in developers' code. Agentic workflows also generate a significant number of vulnerabilities, particularly when granting LLMs more autonomy, potentially increasing the likelihood of misinterpreting project context or task requirements. We find that vulnerabilities are more likely to occur in LLM patches associated with a higher number of files, more lines of generated code, and GitHub issues that lack specific code snippets or information about the expected code behavior and steps to reproduce. These results suggest that contextual factors play a critical role in the security of the generated code and point toward the need for proactive risk assessment methods that account for both code and issue-level information to complement existing vulnerability detection tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18479v1">How Well Do LLMs Predict Prerequisite Skills? Zero-Shot Comparison to Expert-Defined Concepts</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Prerequisite skills - foundational competencies required before mastering more advanced concepts - are important for supporting effective learning, assessment, and skill-gap analysis. Traditionally curated by domain experts, these relationships are costly to maintain and difficult to scale. This paper investigates whether large language models (LLMs) can predict prerequisite skills in a zero-shot setting, using only natural language descriptions and without task-specific fine-tuning. We introduce ESCO-PrereqSkill, a benchmark dataset constructed from the ESCO taxonomy, comprising 3,196 skills and their expert-defined prerequisite links. Using a standardized prompting strategy, we evaluate 13 state-of-the-art LLMs, including GPT-4, Claude 3, Gemini, LLaMA 4, Qwen2, and DeepSeek, across semantic similarity, BERTScore, and inference latency. Our results show that models such as LLaMA4-Maverick, Claude-3-7-Sonnet, and Qwen2-72B generate predictions that closely align with expert ground truth, demonstrating strong semantic reasoning without supervision. These findings highlight the potential of LLMs to support scalable prerequisite skill modeling for applications in personalized learning, intelligent tutoring, and skill-based recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18455v1">LLM-based Embedders for Prior Case Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 Accepted in Recent Advancements in Natural Language Processing (RANLP 2025) conference
    </div>
    <details class="paper-abstract">
      In common law systems, legal professionals such as lawyers and judges rely on precedents to build their arguments. As the volume of cases has grown massively over time, effectively retrieving prior cases has become essential. Prior case retrieval (PCR) is an information retrieval (IR) task that aims to automatically identify the most relevant court cases for a specific query from a large pool of potential candidates. While IR methods have seen several paradigm shifts over the last few years, the vast majority of PCR methods continue to rely on traditional IR methods, such as BM25. The state-of-the-art deep learning IR methods have not been successful in PCR due to two key challenges: i. Lengthy legal text limitation; when using the powerful BERT-based transformer models, there is a limit of input text lengths, which inevitably requires to shorten the input via truncation or division with a loss of legal context information. ii. Lack of legal training data; due to data privacy concerns, available PCR datasets are often limited in size, making it difficult to train deep learning-based models effectively. In this research, we address these challenges by leveraging LLM-based text embedders in PCR. LLM-based embedders support longer input lengths, and since we use them in an unsupervised manner, they do not require training data, addressing both challenges simultaneously. In this paper, we evaluate state-of-the-art LLM-based text embedders in four PCR benchmark datasets and show that they outperform BM25 and supervised transformer-based models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18442v1">AraTable: Benchmarking LLMs' Reasoning and Understanding of Arabic Tabular Data</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      The cognitive and reasoning abilities of large language models (LLMs) have enabled remarkable progress in natural language processing. However, their performance in interpreting structured data, especially in tabular formats, remains limited. Although benchmarks for English tabular data are widely available, Arabic is still underrepresented because of the limited availability of public resources and its unique language features. To address this gap, we present AraTable, a novel and comprehensive benchmark designed to evaluate the reasoning and understanding capabilities of LLMs when applied to Arabic tabular data. AraTable consists of various evaluation tasks, such as direct question answering, fact verification, and complex reasoning, involving a wide range of Arabic tabular sources. Our methodology follows a hybrid pipeline, where initial content is generated by LLMs and subsequently filtered and verified by human experts to ensure high dataset quality. Initial analyses using AraTable show that, while LLMs perform adequately on simpler tabular tasks such as direct question answering, they continue to face significant cognitive challenges when tasks require deeper reasoning and fact verification. This indicates that there are substantial opportunities for future work to improve performance on complex tabular reasoning tasks. We also propose a fully automated evaluation framework that uses a self-deliberation mechanism and achieves performance nearly identical to that of human judges. This research provides a valuable, publicly available resource and evaluation framework that can help accelerate the development of foundational models for processing and analysing Arabic structured data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18417v1">FinDPO: Financial Sentiment Analysis for Algorithmic Trading through Preference Optimization of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Opinions expressed in online finance-related textual data are having an increasingly profound impact on trading decisions and market movements. This trend highlights the vital role of sentiment analysis as a tool for quantifying the nature and strength of such opinions. With the rapid development of Generative AI (GenAI), supervised fine-tuned (SFT) large language models (LLMs) have become the de facto standard for financial sentiment analysis. However, the SFT paradigm can lead to memorization of the training data and often fails to generalize to unseen samples. This is a critical limitation in financial domains, where models must adapt to previously unobserved events and the nuanced, domain-specific language of finance. To this end, we introduce FinDPO, the first finance-specific LLM framework based on post-training human preference alignment via Direct Preference Optimization (DPO). The proposed FinDPO achieves state-of-the-art performance on standard sentiment classification benchmarks, outperforming existing supervised fine-tuned models by 11% on the average. Uniquely, the FinDPO framework enables the integration of a fine-tuned causal LLM into realistic portfolio strategies through a novel 'logit-to-score' conversion, which transforms discrete sentiment predictions into continuous, rankable sentiment scores (probabilities). In this way, simulations demonstrate that FinDPO is the first sentiment-based approach to maintain substantial positive returns of 67% annually and strong risk-adjusted performance, as indicated by a Sharpe ratio of 2.0, even under realistic transaction costs of 5 basis points (bps).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18392v1">CLEAR: Error Analysis via LLM-as-a-Judge Made Easy</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      The evaluation of Large Language Models (LLMs) increasingly relies on other LLMs acting as judges. However, current evaluation paradigms typically yield a single score or ranking, answering which model is better but not why. While essential for benchmarking, these top-level scores obscure the specific, actionable reasons behind a model's performance. To bridge this gap, we introduce CLEAR, an interactive, open-source package for LLM-based error analysis. CLEAR first generates per-instance textual feedback, then it creates a set of system-level error issues, and quantifies the prevalence of each identified issue. Our package also provides users with an interactive dashboard that allows for a comprehensive error analysis through aggregate visualizations, applies interactive filters to isolate specific issues or score ranges, and drills down to the individual instances that exemplify a particular behavioral pattern. We demonstrate CLEAR analysis for RAG and Math benchmarks, and showcase its utility through a user case study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18391v1">Revisiting LLM Reasoning via Information Bottleneck</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated remarkable progress in reasoning capabilities through reinforcement learning with verifiable rewards (RLVR). By leveraging simple rule-based rewards, RL effectively incentivizes LLMs to produce extended chain-of-thought (CoT) reasoning trajectories, progressively guiding them toward correct answers. However, existing approaches remain largely heuristic and intuition-driven, limiting the development of principled methodologies. In this paper, we present a theoretical characterization of LLM reasoning grounded in information bottleneck (IB) principle, introducing IB-aware reasoning optimization (IBRO), a framework that encourages reasoning trajectories to be both informative about the final correct answer and generalizable across diverse prompts. We derive a practical token-level surrogate objective and propose an efficient approximation, resulting in the lightweight IB regularization method. This technique integrates seamlessly into existing RL-based post-training frameworks without additional computational overhead, requiring only a one-line code modification. Empirically, we validate IB regularization across multiple mathematical reasoning benchmarks and RL algorithms, demonstrating consistent improvements in LLM reasoning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18368v1">Reasoning Beyond the Obvious: Evaluating Divergent and Convergent Thinking in LLMs for Financial Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 Accepted by Agentic & GenAI Evaluation KDD2025: KDD workshop on Evaluation and Trustworthiness of Agentic and Generative AI Models https://kdd-eval-workshop.github.io/genai-evaluation-kdd2025/
    </div>
    <details class="paper-abstract">
      Most reasoning benchmarks for LLMs emphasize factual accuracy or step-by-step logic. In finance, however, professionals must not only converge on optimal decisions but also generate creative, plausible futures under uncertainty. We introduce ConDiFi, a benchmark that jointly evaluates divergent and convergent thinking in LLMs for financial tasks. ConDiFi features 607 macro-financial prompts for divergent reasoning and 990 multi-hop adversarial MCQs for convergent reasoning. Using this benchmark, we evaluated 14 leading models and uncovered striking differences. Despite high fluency, GPT-4o underperforms on Novelty and Actionability. In contrast, models like DeepSeek-R1 and Cohere Command R+ rank among the top for generating actionable, insights suitable for investment decisions. ConDiFi provides a new perspective to assess reasoning capabilities essential to safe and strategic deployment of LLMs in finance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18366v1">Efficient Uncertainty in LLMs through Evidential Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Accurate uncertainty quantification remains a key challenge for standard LLMs, prompting the adoption of Bayesian and ensemble-based methods. However, such methods typically necessitate computationally expensive sampling, involving multiple forward passes to effectively estimate predictive uncertainty. In this paper, we introduce a novel approach enabling efficient and effective uncertainty estimation in LLMs without sacrificing performance. Specifically, we distill uncertainty-aware teacher models - originally requiring multiple forward passes - into compact student models sharing the same architecture but fine-tuned using Low-Rank Adaptation (LoRA). We compare two distinct distillation strategies: one in which the student employs traditional softmax-based outputs, and another in which the student leverages Dirichlet-distributed outputs to explicitly model epistemic uncertainty via evidential learning. Empirical evaluations on classification datasets demonstrate that such students can achieve comparable or superior predictive and uncertainty quantification performance relative to their teacher models, while critically requiring only a single forward pass. To our knowledge, this is the first demonstration that immediate and robust uncertainty quantification can be achieved in LLMs through evidential distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12988v3">Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 Accepted by ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Previous approaches to persona simulation large language models (LLMs) have typically relied on learning basic biographical information, or using limited role-play dialogue datasets to capture a character's responses. However, a holistic representation of an individual goes beyond surface-level facts or conversations to deeper thoughts and thinking. In this work, we introduce CharacterBot, a model designed to replicate both the linguistic patterns and distinctive thought patterns as manifested in the textual works of a character. Using Lu Xun, a renowned Chinese writer as a case study, we propose four training tasks derived from his 17 essay collections. These include a pre-training task focused on mastering external linguistic structures and knowledge, as well as three fine-tuning tasks: multiple-choice question answering, generative question answering, and style transfer, each aligning the LLM with Lu Xun's internal ideation and writing style. To optimize learning across these tasks, we introduce a CharLoRA parameter updating mechanism, where a general linguistic style expert collaborates with other task-specific experts to better study both the language style and the understanding of deeper thoughts. We evaluate CharacterBot on three tasks for linguistic accuracy and opinion comprehension, demonstrating that it significantly outperforms the baselines on our adapted metrics. We hope this work inspires future research on deep character persona simulation LLMs while considering the importance of ethical standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18343v1">Hybrid Annotation for Propaganda Detection: Integrating LLM Pre-Annotations with Human Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 NLP4PI at ACL
    </div>
    <details class="paper-abstract">
      Propaganda detection on social media remains challenging due to task complexity and limited high-quality labeled data. This paper introduces a novel framework that combines human expertise with Large Language Model (LLM) assistance to improve both annotation consistency and scalability. We propose a hierarchical taxonomy that organizes 14 fine-grained propaganda techniques into three broader categories, conduct a human annotation study on the HQP dataset that reveals low inter-annotator agreement for fine-grained labels, and implement an LLM-assisted pre-annotation pipeline that extracts propagandistic spans, generates concise explanations, and assigns local labels as well as a global label. A secondary human verification study shows significant improvements in both agreement and time-efficiency. Building on this, we fine-tune smaller language models (SLMs) to perform structured annotation. Instead of fine-tuning on human annotations, we train on high-quality LLM-generated data, allowing a large model to produce these annotations and a smaller model to learn to generate them via knowledge distillation. Our work contributes towards the development of scalable and robust propaganda detection systems, supporting the idea of transparent and accountable media ecosystems in line with SDG 16. The code is publicly available at our GitHub repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18340v1">TDR: Task-Decoupled Retrieval with Fine-Grained LLM Feedback for In-Context Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      In-context learning (ICL) has become a classic approach for enabling LLMs to handle various tasks based on a few input-output examples. The effectiveness of ICL heavily relies on the quality of these examples, and previous works which focused on enhancing example retrieval capabilities have achieved impressive performances. However, two challenges remain in retrieving high-quality examples: (1) Difficulty in distinguishing cross-task data distributions, (2) Difficulty in making the fine-grained connection between retriever output and feedback from LLMs. In this paper, we propose a novel framework called TDR. TDR decouples the ICL examples from different tasks, which enables the retrieval module to retrieve examples specific to the target task within a multi-task dataset. Furthermore, TDR models fine-grained feedback from LLMs to supervise and guide the training of the retrieval module, which helps to retrieve high-quality examples. We conducted extensive experiments on a suite of 30 NLP tasks, the results demonstrate that TDR consistently improved results across all datasets and achieves state-of-the-art performance. Meanwhile, our approach is a plug-and-play method, which can be easily combined with various LLMs to improve example retrieval abilities for ICL. The code is available at https://github.com/Nnn-s/TDR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08621v2">A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1</a></div>
    <div class="paper-meta">
      📅 2025-07-24
    </div>
    <details class="paper-abstract">
      Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as Args.me and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.18316v1">YATE: The Role of Test Repair in LLM-Based Unit Test Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-24
      | 💬 12 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recent advances in automated test generation utilises language models to produce unit tests. While effective, language models tend to generate many incorrect tests with respect to both syntax and semantics. Although such incorrect tests can be easily detected and discarded, they constitute a "missed opportunity" -- if fixed, they are often valuable as they directly add testing value (they effectively target the underlying program logic to be tested) and indirectly form good seeds for generating additional tests. To this end, we propose a simple technique for repairing some of these incorrect tests through a combination of rule-based static analysis and re-prompting. We evaluate this simple approach, named YATE, on a set of 6 open-source projects and show that it can effectively produce tests that cover on average 32.06% more lines and kill 21.77% more mutants than a plain LLM-based method. We also compare YATE with four other LLM-based methods, namely HITS, SYMPROMPT, TESTSPARK and COVERUP and show that it produces tests that cover substantially more code. YATE achieves 22% higher line coverage, 20% higher branch coverage and kill 20% more mutants at a comparable cost (number of calls to LLMs).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17542v1">AssertFlip: Reproducing Bugs via Inversion of LLM-Generated Passing Tests</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Bug reproduction is critical in the software debugging and repair process, yet the majority of bugs in open-source and industrial settings lack executable tests to reproduce them at the time they are reported, making diagnosis and resolution more difficult and time-consuming. To address this challenge, we introduce AssertFlip, a novel technique for automatically generating Bug Reproducible Tests (BRTs) using large language models (LLMs). Unlike existing methods that attempt direct generation of failing tests, AssertFlip first generates passing tests on the buggy behaviour and then inverts these tests to fail when the bug is present. We hypothesize that LLMs are better at writing passing tests than ones that crash or fail on purpose. Our results show that AssertFlip outperforms all known techniques in the leaderboard of SWT-Bench, a benchmark curated for BRTs. Specifically, AssertFlip achieves a fail-to-pass success rate of 43.6% on the SWT-Bench-Verified subset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.17476v1">MultiNRC: A Challenging and Native Multilingual Reasoning Evaluation Benchmark for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Although recent Large Language Models (LLMs) have shown rapid improvement on reasoning benchmarks in English, the evaluation of such LLMs' multilingual reasoning capability across diverse languages and cultural contexts remains limited. Existing multilingual reasoning benchmarks are typically constructed by translating existing English reasoning benchmarks, biasing these benchmarks towards reasoning problems with context in English language/cultures. In this work, we introduce the Multilingual Native Reasoning Challenge (MultiNRC), a benchmark designed to assess LLMs on more than 1,000 native, linguistic and culturally grounded reasoning questions written by native speakers in French, Spanish, and Chinese. MultiNRC covers four core reasoning categories: language-specific linguistic reasoning, wordplay & riddles, cultural/tradition reasoning, and math reasoning with cultural relevance. For cultural/tradition reasoning and math reasoning with cultural relevance, we also provide English equivalent translations of the multilingual questions by manual translation from native speakers fluent in English. This set of English equivalents can provide a direct comparison of LLM reasoning capacity in other languages vs. English on the same reasoning questions. We systematically evaluate current 14 leading LLMs covering most LLM families on MultiNRC and its English equivalent set. The results show that (1) current LLMs are still not good at native multilingual reasoning, with none scoring above 50% on MultiNRC; (2) LLMs exhibit distinct strengths and weaknesses in handling linguistic, cultural, and logical reasoning tasks; (3) Most models perform substantially better in math reasoning in English compared to in original languages (+10%), indicating persistent challenges with culturally grounded knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16199v2">WAKENLLM: Evaluating Reasoning Potential and Stability in LLMs via Fine-Grained Benchmarking</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently output the label Unknown, yet current evaluations focus almost exclusively on whether such answers are honest rather than why they arise. This blurs two distinct cases: (i) an input that is genuinely indeterminate and (ii) a solvable problem that the model fails to resolve. We call this phenomenon Vague Perception. And thus we introduce a framework that quantifies the proportion of Unknown responses attributable to model incapacity and tests whether guided stimulation can convert them into either correct Known or correct Unknown with valid reasoning. By separating these sources of uncertainty, our method provides a clearer picture of LLM reasoning limits and their potential for improvement. As we get a theoretical accuracy of reasoning task on different LLMs, we apply different methods to test whether the model can reach the accuracy given a baseline framework. Our work is meaningful in exploring the potential reasoning ability of LLMs and providing a new perspective on solving the Vague Perception phenomenon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10054v2">Explicit Vulnerability Generation with LLMs: An Investigation Beyond Adversarial Attacks</a></div>
    <div class="paper-meta">
      📅 2025-07-23
      | 💬 Accepted to ICSME 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used as code assistants, yet their behavior when explicitly asked to generate insecure code remains poorly understood. While prior research has focused on unintended vulnerabilities, this study examines a more direct threat: open-source LLMs generating vulnerable code when prompted. We propose a dual experimental design: (1) Dynamic Prompting, which systematically varies vulnerability type, user persona, and prompt phrasing across structured templates; and (2) Reverse Prompting, which derives natural-language prompts from real vulnerable code samples. We evaluate three open-source 7B-parameter models (Qwen2, Mistral, Gemma) using static analysis to assess both the presence and correctness of generated vulnerabilities. Our results show that all models frequently generate the requested vulnerabilities, though with significant performance differences. Gemma achieves the highest correctness for memory vulnerabilities under Dynamic Prompting (e.g., 98.6% for buffer overflows), while Qwen2 demonstrates the most balanced performance across all tasks. We find that professional personas (e.g., "DevOps Engineer") consistently elicit higher success rates than student personas, and that the effectiveness of direct versus indirect phrasing is inverted depending on the prompting strategy. Vulnerability reproduction accuracy follows a non-linear pattern with code complexity, peaking in a moderate range. Our findings expose how LLMs' reliance on pattern recall over semantic reasoning creates significant blind spots in their safety alignments, particularly for requests framed as plausible professional tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10382v2">Leveraging RAG-LLMs for Urban Mobility Simulation and Analysis</a></div>
    <div class="paper-meta">
      📅 2025-07-23
    </div>
    <details class="paper-abstract">
      With the rise of smart mobility and shared e-mobility services, numerous advanced technologies have been applied to this field. Cloud-based traffic simulation solutions have flourished, offering increasingly realistic representations of the evolving mobility landscape. LLMs have emerged as pioneering tools, providing robust support for various applications, including intelligent decision-making, user interaction, and real-time traffic analysis. As user demand for e-mobility continues to grow, delivering comprehensive end-to-end solutions has become crucial. In this paper, we present a cloud-based, LLM-powered shared e-mobility platform, integrated with a mobile application for personalized route recommendations. The optimization module is evaluated based on travel time and cost across different traffic scenarios. Additionally, the LLM-powered RAG framework is evaluated at the schema level for different users, using various evaluation methods. Schema-level RAG with XiYanSQL achieves an average execution accuracy of 0.81 on system operator queries and 0.98 on user queries.
    </details>
</div>
