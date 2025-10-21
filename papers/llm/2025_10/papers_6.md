# llm - 2025_10

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
- [Part 12](papers_12.md)
- [Part 13](papers_13.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10677v1">Unlocking LLM Safeguards for Low-Resource Languages via Reasoning and Alignment with Minimal Training Data</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Accepted to MRL Workshop at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Recent advances in LLMs have enhanced AI capabilities, but also increased the risk posed by malicious requests, highlighting the need for effective LLM safeguards to detect such queries. Existing approaches largely rely on classifier-based methods that lack interpretability and perform poorly on low-resource languages. To address these limitations, we propose ConsistentGuard, a novel reasoning-based multilingual safeguard, which enhances explainability via reasoning and boosts knowledge transfer between languages through alignment. With only 1,000 training samples, our method demonstrates superior performance on three datasets across six languages, outperforming larger models trained with significantly more data, and exhibits strong interpretability and generalization ability. We also contribute a multilingual benchmark extension and release our codes to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18080v2">Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Accepted at NeurIPS 2025, camera-ready version
    </div>
    <details class="paper-abstract">
      Recent studies have shown that making a model spend more time thinking through longer Chain of Thoughts (CoTs) enables it to gain significant improvements in complex reasoning tasks. While current researches continue to explore the benefits of increasing test-time compute by extending the CoT lengths of Large Language Models (LLMs), we are concerned about a potential issue hidden behind the current pursuit of test-time scaling: Would excessively scaling the CoT length actually bring adverse effects to a model's reasoning performance? Our explorations on mathematical reasoning tasks reveal an unexpected finding that scaling with longer CoTs can indeed impair the reasoning performance of LLMs in certain domains. Moreover, we discover that there exists an optimal scaled length distribution that differs across different domains. Based on these insights, we propose a Thinking-Optimal Scaling strategy. Our method first uses a small set of seed data with varying response length distributions to teach the model to adopt different reasoning efforts for deep thinking. Then, the model selects its shortest correct response under different reasoning efforts on additional problems for self-improvement. Our self-improved models built upon Qwen2.5-32B-Instruct outperform other distillation-based 32B o1-like models across various math benchmarks, and achieve performance on par with the teacher model QwQ-32B-Preview that produces the seed data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10644v1">Hierarchical Optimization via LLM-Guided Objective Evolution for Mobility-on-Demand Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Online ride-hailing platforms aim to deliver efficient mobility-on-demand services, often facing challenges in balancing dynamic and spatially heterogeneous supply and demand. Existing methods typically fall into two categories: reinforcement learning (RL) approaches, which suffer from data inefficiency, oversimplified modeling of real-world dynamics, and difficulty enforcing operational constraints; or decomposed online optimization methods, which rely on manually designed high-level objectives that lack awareness of low-level routing dynamics. To address this issue, we propose a novel hybrid framework that integrates large language model (LLM) with mathematical optimization in a dynamic hierarchical system: (1) it is training-free, removing the need for large-scale interaction data as in RL, and (2) it leverages LLM to bridge cognitive limitations caused by problem decomposition by adaptively generating high-level objectives. Within this framework, LLM serves as a meta-optimizer, producing semantic heuristics that guide a low-level optimizer responsible for constraint enforcement and real-time decision execution. These heuristics are refined through a closed-loop evolutionary process, driven by harmony search, which iteratively adapts the LLM prompts based on feasibility and performance feedback from the optimization layer. Extensive experiments based on scenarios derived from both the New York and Chicago taxi datasets demonstrate the effectiveness of our approach, achieving an average improvement of 16% compared to state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10618v1">Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Post-training compression has been a widely employed approach to scale down large language model (LLM) and facilitate efficient inference. In various proposed compression methods, including pruning and quantization, calibration data plays a vital role by informing the weight importance and activation dynamic ranges. However, how calibration data impacts the LLM capability after compression is less explored. Few of the existing works, though recognizing the significance of this study, only investigate the language modeling or commonsense reasoning performance degradation from limited angles, like the data sources or sample amounts. More systematic research is still needed to examine the impacts on different LLM capabilities in terms of compositional properties and domain correspondence of calibration data. In this work, we aim at bridging this gap and further analyze underlying influencing mechanisms from the activation pattern perspective. Especially, we explore the calibration data's impacts on high-level complex reasoning capabilities, like math problem solving and code generation. Delving into the underlying mechanism, we find that the representativeness and diversity in activation space more fundamentally determine the quality of calibration data. Finally, we propose a calibration data curation framework based on such observations and analysis, enhancing the performance of existing post-training compression methods on preserving critical LLM capabilities. Our code is provided in \href{https://github.com/BokwaiHo/COLA.git}{Link}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15607v2">From Problem-Solving to Teaching Problem-Solving: Aligning LLMs with Pedagogy using Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Accepted to EMNLP 2025 Main as an oral presentation. David Dinucu-Jianu and Jakub Macina contributed equally. Code available: https://github.com/eth-lre/PedagogicalRL
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can transform education, but their optimization for direct question-answering often undermines effective pedagogy which requires strategically withholding answers. To mitigate this, we propose an online reinforcement learning (RL)-based alignment framework that can quickly adapt LLMs into effective tutors using simulated student-tutor interactions by emphasizing pedagogical quality and guided problem-solving over simply giving away answers. We use our method to train a 7B parameter tutor model without human annotations which reaches similar performance to larger proprietary models like LearnLM. We introduce a controllable reward weighting to balance pedagogical support and student solving accuracy, allowing us to trace the Pareto frontier between these two objectives. Our models better preserve reasoning capabilities than single-turn SFT baselines and can optionally enhance interpretability through thinking tags that expose the model's instructional planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12421v2">Wide-Horizon Thinking and Simulation-Based Evaluation for Real-World LLM Planning with Multifaceted Constraints</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Accepted by NeurIPS 2025 Spotlight
    </div>
    <details class="paper-abstract">
      Unlike reasoning, which often entails a deep sequence of deductive steps, complex real-world planning is characterized by the need to synthesize a broad spectrum of parallel and potentially conflicting information and constraints. For example, in travel planning scenarios, it requires the integration of diverse real-world information and user preferences. While LLMs show promise, existing methods with long-horizon thinking struggle with handling multifaceted constraints, leading to suboptimal solutions. Motivated by the challenges of real-world travel planning, this paper introduces the Multiple Aspects of Planning (MAoP), empowering LLMs with "wide-horizon thinking" to solve planning problems with multifaceted constraints. Instead of direct planning, MAoP leverages the strategist to conduct pre-planning from various aspects and provide the planning blueprint for planners, enabling strong inference-time scalability by scaling aspects to consider various constraints. In addition, existing benchmarks for multi-constraint planning are flawed because they assess constraints in isolation, ignoring causal dependencies within the constraints, e.g, travel planning, where past activities dictate future itinerary. To address this, we propose Travel-Sim, an agent-based benchmark assessing plans via real-world simulation, thereby inherently resolving these causal dependencies. This paper advances LLM capabilities in complex planning and offers novel insights for evaluating sophisticated scenarios through simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10592v1">A Layered Intuition -- Method Model with Scope Extension for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Existing studies have introduced method-based reasoning and scope extension as approaches to enhance Large Language Model (LLM) performance beyond direct matrix mappings. Building on these foundations, this paper summarizes and integrates these ideas into a unified Intuition-Method Layered Model with Scope Extension, designed to address indirected (unseen) issues more systematically. In this framework, intuition-based thinking provides rapid first-reaction answers, while method-based thinking decouples questions and solutions into transferable reasoning units. Scope extension is then applied to broaden applicability, including vertical (cause analysis), horizontal (parallel and generalized issues), and for the first time, temporal and spatial extensions, which expand reasoning across time and contextual dimensions. These extensions are organized into systematic knowledge trees that interconnect into a knowledge network, thereby increasing adaptability. To quantitatively evaluate this process, we propose the entropy of method extension, which measures the independence and diversity of extensions as an indicator of the system's capacity to solve unseen questions. By logically connecting existing approaches with new extensions and introducing an entropy-based evaluation framework, this work advances toward a more robust and extensible reasoning paradigm for LLMs in real-world problem-solving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10539v1">Detecting Hallucinations in Authentic LLM-Human Interactions</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly applied in sensitive domains such as medicine and law, hallucination detection has become a critical task. Although numerous benchmarks have been proposed to advance research in this area, most of them are artificially constructed--either through deliberate hallucination induction or simulated interactions--rather than derived from genuine LLM-human dialogues. Consequently, these benchmarks fail to fully capture the characteristics of hallucinations that occur in real-world usage. To address this limitation, we introduce AuthenHallu, the first hallucination detection benchmark built entirely from authentic LLM-human interactions. For AuthenHallu, we select and annotate samples from genuine LLM-human dialogues, thereby providing a faithful reflection of how LLMs hallucinate in everyday user interactions. Statistical analysis shows that hallucinations occur in 31.4% of the query-response pairs in our benchmark, and this proportion increases dramatically to 60.0% in challenging domains such as Math & Number Problems. Furthermore, we explore the potential of using vanilla LLMs themselves as hallucination detectors and find that, despite some promise, their current performance remains insufficient in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19657v5">LLMEasyQuant: Scalable Quantization for Parallel and Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 Accepted as International Conference of Computational Optimization 2025 Oral
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow in size and deployment scale, quantization has become an essential technique for reducing memory footprint and improving inference efficiency. However, existing quantization toolkits often lack transparency, flexibility, and system-level scalability across GPUs and distributed environments. We present \textbf{LLMEasyQuant}, a modular, system-aware quantization framework designed for efficient, low-bit inference of LLMs on single-node multi-GPU, multi-node, and edge hardware. LLMEasyQuant supports a wide range of quantization methods -- including Symmetric Quantization, ZeroQuant, SmoothQuant, and SimQuant -- with unified interfaces for per-layer calibration, bitwidth assignment, and runtime adaptation. It integrates fused CUDA kernels with NCCL-based distributed synchronization and supports both static and online quantization. Empirical results show that LLMEasyQuant can achieve substantial speedup in GEMM execution, HBM load time, and near-linear multi-GPU scaling. Ablation studies further validate its ability to balance latency, memory, and accuracy under diverse deployment conditions. LLMEasyQuant offers a practical quantization serving system for scalable, hardware-optimized LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11166v2">SoLoPO: Unlocking Long-Context Capabilities in LLMs via Short-to-Long Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Despite advances in pretraining with extended context lengths, large language models (LLMs) still face challenges in effectively utilizing real-world long-context information, primarily due to insufficient long-context alignment caused by data quality issues, training inefficiencies, and the lack of well-designed optimization objectives. To address these limitations, we propose a framework named $\textbf{S}$h$\textbf{o}$rt-to-$\textbf{Lo}$ng $\textbf{P}$reference $\textbf{O}$ptimization ($\textbf{SoLoPO}$), decoupling long-context preference optimization (PO) into two components: short-context PO and short-to-long reward alignment (SoLo-RA), supported by both theoretical and empirical evidence. Specifically, short-context PO leverages preference pairs sampled from short contexts to enhance the model's contextual knowledge utilization ability. Meanwhile, SoLo-RA explicitly encourages reward score consistency utilization for the responses when conditioned on both short and long contexts that contain identical task-relevant information. This facilitates transferring the model's ability to handle short contexts into long-context scenarios. SoLoPO is compatible with mainstream preference optimization algorithms, while substantially improving the efficiency of data construction and training processes. Experimental results show that SoLoPO enhances all these algorithms with respect to stronger length and domain generalization abilities across various long-context benchmarks, while achieving notable improvements in both computational and memory efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24869v2">Retro*: Optimizing LLMs for Reasoning-Intensive Document Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      With the growing popularity of LLM agents and RAG, it has become increasingly important to retrieve documents that are essential for solving a task, even when their connection to the task is indirect or implicit. Addressing this problem requires fine-grained reasoning to accurately assess the relevance between the task and each candidate document. This capability, however, poses a significant challenge for existing IR techniques. Despite recent progress in reasoning-enhanced IR, existing approaches still face significant challenges in applicability, scalability, and efficiency. In this work, we propose Retro*, a novel approach for reasoning-intensive document retrieval. Our method introduces a rubric-based relevance scoring mechanism, enabling the model to reason about the relationship between a task and a document based on explicitly defined criteria, whereby producing a fine-grained, interpretable relevance score. Retro* also supports test-time scaling by combining multiple reasoning trajectories via score integration, which produces more reliable relevance estimates. To optimize Retro*'s reasoning capabilities, we introduce a novel reinforcement learning algorithm tailored for its relevance scoring mechanism, which employs two composite rewards to fully exploit the trajectories of each training sample. Our experiments show that Retro* outperforms existing document retrieval methods with notable advantages, leading to state-of-the-art performance on the BRIGHT benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10517v1">ECO: Enhanced Code Optimization via Performance-Aware Prompting for Code-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Code runtime optimization-the task of rewriting a given code to a faster one-remains challenging, as it requires reasoning about performance trade-offs involving algorithmic and structural choices. Recent approaches employ code-LLMs with slow-fast code pairs provided as optimization guidance, but such pair-based methods obscure the causal factors of performance gains and often lead to superficial pattern imitation rather than genuine performance reasoning. We introduce ECO, a performance-aware prompting framework for code optimization. ECO first distills runtime optimization instructions (ROIs) from reference slow-fast code pairs; Each ROI describes root causes of inefficiency and the rationales that drive performance improvements. For a given input code, ECO in parallel employs (i) a symbolic advisor to produce a bottleneck diagnosis tailored to the code, and (ii) an ROI retriever to return related ROIs. These two outputs are then composed into a performance-aware prompt, providing actionable guidance for code-LLMs. ECO's prompts are model-agnostic, require no fine-tuning, and can be easily prepended to any code-LLM prompt. Our empirical studies highlight that ECO prompting significantly improves code-LLMs' ability to generate efficient code, achieving speedups of up to 7.81x while minimizing correctness loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10605v2">RedOne: Revealing Domain-specific LLM Post-Training in Social Networking Services</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      As a primary medium for modern information dissemination, social networking services (SNS) have experienced rapid growth, which has proposed significant challenges for platform content management and interaction quality improvement. Recently, the development of large language models (LLMs) has offered potential solutions but existing studies focus on isolated tasks, which not only encounter diminishing benefit from the data scaling within individual scenarios but also fail to flexibly adapt to diverse real-world context. To address these challenges, we introduce RedOne, a domain-specific LLM designed to break the performance bottleneck of single-task baselines and establish a comprehensive foundation for the SNS. RedOne was developed through a three-stage training strategy consisting of continue pretraining, supervised fine-tuning, and preference optimization, using a large-scale real-world dataset. Through extensive experiments, RedOne maintains strong general capabilities, and achieves an average improvement up to 14.02% across 8 major SNS tasks and 7.56% in SNS bilingual evaluation benchmark, compared with base models. Furthermore, through online testing, RedOne reduced the exposure rate in harmful content detection by 11.23% and improved the click page rate in post-view search by 14.95% compared with single-tasks finetuned baseline models. These results establish RedOne as a robust domain-specific LLM for SNS, demonstrating excellent generalization across various tasks and promising applicability in real-world scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07731v2">oMeBench: Towards Robust Benchmarking of LLMs in Organic Mechanism Elucidation and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Organic reaction mechanisms are the stepwise elementary reactions by which reactants form intermediates and products, and are fundamental to understanding chemical reactivity and designing new molecules and reactions. Although large language models (LLMs) have shown promise in understanding chemical tasks such as synthesis design, it is unclear to what extent this reflects genuine chemical reasoning capabilities, i.e., the ability to generate valid intermediates, maintain chemical consistency, and follow logically coherent multi-step pathways. We address this by introducing oMeBench, the first large-scale, expert-curated benchmark for organic mechanism reasoning in organic chemistry. It comprises over 10,000 annotated mechanistic steps with intermediates, type labels, and difficulty ratings. Furthermore, to evaluate LLM capability more precisely and enable fine-grained scoring, we propose oMeS, a dynamic evaluation framework that combines step-level logic and chemical similarity. We analyze the performance of state-of-the-art LLMs, and our results show that although current models display promising chemical intuition, they struggle with correct and consistent multi-step reasoning. Notably, we find that using prompting strategy and fine-tuning a specialist model on our proposed dataset increases performance by 50% over the leading closed-source model. We hope that oMeBench will serve as a rigorous foundation for advancing AI systems toward genuine chemical reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10493v1">The Hidden DNA of LLM-Generated JavaScript: Structural Patterns Enable High-Accuracy Authorship Attribution</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      In this paper, we present the first large-scale study exploring whether JavaScript code generated by Large Language Models (LLMs) can reveal which model produced it, enabling reliable authorship attribution and model fingerprinting. With the rapid rise of AI-generated code, attribution is playing a critical role in detecting vulnerabilities, flagging malicious content, and ensuring accountability. While AI-vs-human detection usually treats AI as a single category we show that individual LLMs leave unique stylistic signatures, even among models belonging to the same family or parameter size. To this end, we introduce LLM-NodeJS, a dataset of 50,000 Node.js back-end programs from 20 large language models. Each has four transformed variants, yielding 250,000 unique JavaScript samples and two additional representations (JSIR and AST) for diverse research applications. Using this dataset, we benchmark traditional machine learning classifiers against fine-tuned Transformer encoders and introduce CodeT5-JSA, a custom architecture derived from the 770M-parameter CodeT5 model with its decoder removed and a modified classification head. It achieves 95.8% accuracy on five-class attribution, 94.6% on ten-class, and 88.5% on twenty-class tasks, surpassing other tested models such as BERT, CodeBERT, and Longformer. We demonstrate that classifiers capture deeper stylistic regularities in program dataflow and structure, rather than relying on surface-level features. As a result, attribution remains effective even after mangling, comment removal, and heavy code transformations. To support open science and reproducibility, we release the LLM-NodeJS dataset, Google Colab training scripts, and all related materials on GitHub: https://github.com/LLM-NodeJS-dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10486v1">SASER: Stego attacks on open-source LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Open-source large language models (LLMs) have demonstrated considerable dominance over proprietary LLMs in resolving neural processing tasks, thanks to the collaborative and sharing nature. Although full access to source codes, model parameters, and training data lays the groundwork for transparency, we argue that such a full-access manner is vulnerable to stego attacks, and their ill-effects are not fully understood. In this paper, we conduct a systematic formalization for stego attacks on open-source LLMs by enumerating all possible threat models associated with adversary objectives, knowledge, and capabilities. Therein, the threat posed by adversaries with internal knowledge, who inject payloads and triggers during the model sharing phase, is of practical interest. We go even further and propose the first stego attack on open-source LLMs, dubbed SASER, which wields impacts through identifying targeted parameters, embedding payloads, injecting triggers, and executing payloads sequentially. Particularly, SASER enhances the attack robustness against quantization-based local deployment by de-quantizing the embedded payloads. In addition, to achieve stealthiness, SASER devises the performance-aware importance metric to identify targeted parameters with the least degradation of model performance. Extensive experiments on LlaMA2-7B and ChatGLM3-6B, without quantization, show that the stealth rate of SASER outperforms existing stego attacks (for general DNNs) by up to 98.1%, while achieving the same attack success rate (ASR) of 100%. More importantly, SASER improves ASR on quantized models from 0 to 100% in all settings. We appeal for investigations on countermeasures against SASER in view of the significant attack effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10467v1">AnyBCQ: Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      The deployment of large language models (LLMs) is increasingly constrained by memory and latency bottlenecks, motivating the need for quantization techniques that flexibly balance accuracy and efficiency. Recent work has introduced multi-precision models, which enable inference at multiple precisions within a single model depending on runtime constraints. To support such flexibility, quantized weights are often stored as bit-planes, where hardware efficiency improves when the compute operates directly at the bit-plane level and activates only the precision required by each request. In this work, we present AnyBCQ, a hardware-friendly multi-precision extension of Binary-Coded Quantization (BCQ) that supports direct bit-plane operations. By representing weights as binary bit-planes with corresponding scale factors, AnyBCQ enables bit-plane-level computation and maps naturally to accelerator-friendly, bit-parallel arithmetic. Our progressive precision expansion mechanism incrementally refines scaling factors while reusing previously assigned binary codes, yielding monotonic improvements in accuracy as additional bits are enabled. We further co-design a specialized kernel that exploits the BCQ structure to support dynamic per-request precision selection with negligible overhead. Experiments on recent LLMs demonstrate that AnyBCQ significantly narrows the accuracy drop in the low-bit regime (e.g. 2-bit), remains competitive at higher precision, and achieves throughput gains of up to 3.0x over half precision and 1.2x over state-of-the-art multi-precision methods. By aligning algorithmic flexibility with hardware efficiency, AnyBCQ provides a practical foundation for multi-precision LLM deployment across diverse service-level objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00482v2">Talk Less, Call Right: Enhancing Role-Play LLM Agents with Automatic Prompt Optimization and Role Prompting</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 EMNLP 2025 Wordplay Workshop (Spotlight)
    </div>
    <details class="paper-abstract">
      This report investigates approaches for prompting a tool-augmented large language model (LLM) to act as a role-playing dialogue agent in the API track of the Commonsense Persona-grounded Dialogue Challenge (CPDC) 2025. In this setting, dialogue agents often produce overly long in-character responses (over-speaking) while failing to use tools effectively according to the persona (under-acting), such as generating function calls that do not exist or making unnecessary tool calls before answering. We explore four prompting approaches to address these issues: 1) basic role prompting, 2) improved role prompting, 3) automatic prompt optimization (APO), and 4) rule-based role prompting. The rule-based role prompting (RRP) approach achieved the best performance through two novel techniques-character-card/scene-contract design and strict enforcement of function calling-which led to an overall score of 0.571, improving on the zero-shot baseline score of 0.519. These findings demonstrate that RRP design can substantially improve the effectiveness and reliability of role-playing dialogue agents compared with more elaborate methods such as APO. To support future efforts in developing persona prompts, we are open-sourcing all of our best-performing prompts and the APO tool Source code is available at https://github.com/scb-10x/apo
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.00882v3">VulSolver: Vulnerability Detection via LLM-Driven Constraint Solving</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Traditional vulnerability detection methods rely heavily on predefined rule matching, which often fails to capture vulnerabilities accurately. With the rise of large language models (LLMs), leveraging their ability to understand code semantics has emerged as a promising direction for achieving more accurate and efficient vulnerability detection. However, current LLM-based approaches face significant challenges: instability in model outputs, limitations in context length, and hallucination. As a result, many existing solutions either use LLMs merely to enrich predefined rule sets, thereby keeping the detection process fundamentally rule-based, or over-rely on them, leading to poor robustness. To address these challenges, we propose a constraint-solving approach powered by LLMs named VULSOLVER. By modeling vulnerability detection as a constraint-solving problem, and by integrating static application security testing (SAST) with the semantic reasoning capabilities of LLMs, our method enables the LLM to act like a professional human security expert. We assess VULSOLVER on the OWASP Benchmark (1,023 labeled samples), achieving 96.29% accuracy, 96.55% F1-score, and 100% recall. Applied to popular GitHub repositories, VULSOLVER also identified 15 previously unknown high-severity vulnerabilities (CVSS 7.5-9.8), demonstrating its effectiveness in real-world security analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.19274v3">Neuralink: Fast LLM Inference on Smartphones with Neuron Co-Activation Linking</a></div>
    <div class="paper-meta">
      📅 2025-10-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across various domains, yet deploying them on mobile devices remains an arduous challenge due to their extensive computational and memory demands. While lightweight LLMs have been developed to fit mobile environments, they suffer from degraded model accuracy. In contrast, sparsity-based techniques minimize DRAM usage by selectively transferring only relevant neurons to DRAM while retaining the full model in external storage, such as flash. However, such approaches are critically limited by numerous I/O operations, particularly on smartphones with severe IOPS constraints. In this paper, we propose Neuralink, a novel approach that accelerates LLM inference on smartphones by optimizing neuron placement in flash memory. Neuralink leverages the concept of Neuron Co-Activation, where neurons frequently activated together are linked to facilitate continuous read access and optimize I/O efficiency. Our approach incorporates a two-stage solution: an offline stage that reorganizes neuron placement based on co-activation patterns, and an online stage that employs tailored data access and caching strategies to align well with hardware characteristics. Evaluations conducted on a variety of smartphones and LLMs demonstrate that Neuralink achieves on average $1.49\times$ improvements in end-to-end latency compared to the state-of-the-art. As the first solution to optimize storage placement under sparsity, Neuralink explores a new optimization space at the intersection of sparsity-driven algorithm and storage-level system co-design for LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10407v1">PrediQL: Automated Testing of GraphQL APIs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-12
      | 💬 8 pages, two columns
    </div>
    <details class="paper-abstract">
      GraphQL's flexible query model and nested data dependencies expose APIs to complex, context-dependent vulnerabilities that are difficult to uncover using conventional testing tools. Existing fuzzers either rely on random payload generation or rigid mutation heuristics, failing to adapt to the dynamic structures of GraphQL schemas and responses. We present PrediQL, the first retrieval-augmented, LLM-guided fuzzer for GraphQL APIs. PrediQL combines large language model reasoning with adaptive feedback loops to generate semantically valid and diverse queries. It models the choice of fuzzing strategy as a multi-armed bandit problem, balancing exploration of new query structures with exploitation of past successes. To enhance efficiency, PrediQL retrieves and reuses execution traces, schema fragments, and prior errors, enabling self-correction and progressive learning across test iterations. Beyond input generation, PrediQL integrates a context-aware vulnerability detector that uses LLM reasoning to analyze responses, interpreting data values, error messages, and status codes to identify issues such as injection flaws, access-control bypasses, and information disclosure. Our evaluation across open-source and benchmark GraphQL APIs shows that PrediQL achieves significantly higher coverage and vulnerability discovery rates compared to state-of-the-art baselines. These results demonstrate that combining retrieval-augmented reasoning with adaptive fuzzing can transform API security testing from reactive enumeration to intelligent exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10698v2">CrunchLLM: Multitask LLMs for Structured Business Reasoning and Outcome Prediction</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Predicting the success of start-up companies, defined as achieving an exit through acquisition or IPO, is a critical problem in entrepreneurship and innovation research. Datasets such as Crunchbase provide both structured information (e.g., funding rounds, industries, investor networks) and unstructured text (e.g., company descriptions), but effectively leveraging this heterogeneous data for prediction remains challenging. Traditional machine learning approaches often rely only on structured features and achieve moderate accuracy, while large language models (LLMs) offer rich reasoning abilities but struggle to adapt directly to domain-specific business data. We present \textbf{CrunchLLM}, a domain-adapted LLM framework for startup success prediction. CrunchLLM integrates structured company attributes with unstructured textual narratives and applies parameter-efficient fine-tuning strategies alongside prompt optimization to specialize foundation models for entrepreneurship data. Our approach achieves accuracy exceeding 80\% on Crunchbase startup success prediction, significantly outperforming traditional classifiers and baseline LLMs. Beyond predictive performance, CrunchLLM provides interpretable reasoning traces that justify its predictions, enhancing transparency and trustworthiness for financial and policy decision makers. This work demonstrates how adapting LLMs with domain-aware fine-tuning and structured--unstructured data fusion can advance predictive modeling of entrepreneurial outcomes. CrunchLLM contributes a methodological framework and a practical tool for data-driven decision making in venture capital and innovation policy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.04663v3">Debate, Deliberate, Decide (D3): A Cost-Aware Adversarial Framework for Reliable and Interpretable LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      The evaluation of Large Language Models (LLMs) remains challenging due to inconsistency, bias, and the absence of transparent decision criteria in automated judging. We present Debate, Deliberate, Decide (D3), a cost-aware, adversarial multi-agent framework that orchestrates structured debate among role-specialized agents (advocates, a judge, and an optional jury) to produce reliable and interpretable evaluations. D3 instantiates two complementary protocols: (1) Multi-Advocate One-Round Evaluation (MORE), which elicits k parallel defenses per answer to amplify signal via diverse advocacy, and (2) Single-Advocate Multi-Round Evaluation (SAMRE) with budgeted stopping, which iteratively refines arguments under an explicit token budget and convergence checks. We develop a probabilistic model of score gaps that (i) characterizes reliability and convergence under iterative debate and (ii) explains the separation gains from parallel advocacy. Under mild assumptions, the posterior distribution of the round-r gap concentrates around the true difference and the probability of mis-ranking vanishes; moreover, aggregating across k advocates provably increases expected score separation. We complement theory with a rigorous experimental suite across MT-Bench, AlignBench, and AUTO-J, showing state-of-the-art agreement with human judgments (accuracy and Cohen's kappa), reduced positional and verbosity biases via anonymization and role diversification, and a favorable cost-accuracy frontier enabled by budgeted stopping. Ablations and qualitative analyses isolate the contributions of debate, aggregation, and anonymity. Together, these results establish D3 as a principled, practical recipe for reliable, interpretable, and cost-aware LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10331v1">LLM-Friendly Knowledge Representation for Customer Support</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      We propose a practical approach by integrating Large Language Models (LLMs) with a framework designed to navigate the complexities of Airbnb customer support operations. In this paper, our methodology employs a novel reformatting technique, the Intent, Context, and Action (ICA) format, which transforms policies and workflows into a structure more comprehensible to LLMs. Additionally, we develop a synthetic data generation strategy to create training data with minimal human intervention, enabling cost-effective fine-tuning of our model. Our internal experiments (not applied to Airbnb products) demonstrate that our approach of restructuring workflows and fine-tuning LLMs with synthetic data significantly enhances their performance, setting a new benchmark for their application in customer support. Our solution is not only cost-effective but also improves customer support, as evidenced by both accuracy and manual processing time evaluation metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10329v1">End-to-end Automatic Speech Recognition and Speech Translation: Integration of Speech Foundational Models and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Speech Translation (ST) is a machine translation task that involves converting speech signals from one language to the corresponding text in another language; this task has two different approaches, namely the traditional cascade and the more recent end-to-end. This paper explores a combined end-to-end architecture of pre-trained speech encoders and Large Language Models (LLMs) for performing both Automatic Speech Recognition (ASR) and ST simultaneously. Experiments with the English-to-German language pair show that our best model not only can achieve better translation results than SeamlessM4T, a large foundational end-to-end, multi-modal translation model, but can also match the performance of a cascaded system with Whisper and NLLB, with up to a score gain of 8% in $\text{COMET}^{\text{DA}}_{22}$ metric.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.09710v3">DUMP: Automated Distribution-Level Curriculum Learning for RL-based LLM Post-training</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL)-based post-training have led to notable improvements in large language models (LLMs), particularly in enhancing their reasoning capabilities to handle complex tasks. However, most existing methods treat the training data as a unified whole, overlooking the fact that modern LLM training often involves a mixture of data from diverse distributions-varying in both source and difficulty. This heterogeneity introduces a key challenge: how to adaptively schedule training across distributions to optimize learning efficiency. In this paper, we present a principled curriculum learning framework grounded in the notion of distribution-level learnability. Our core insight is that the magnitude of policy advantages reflects how much a model can still benefit from further training on a given distribution. Based on this, we propose a distribution-level curriculum learning framework for RL-based LLM post-training, which leverages the Upper Confidence Bound (UCB) principle to dynamically adjust sampling probabilities for different distrubutions. This approach prioritizes distributions with either high average advantage (exploitation) or low sample count (exploration), yielding an adaptive and theoretically grounded training schedule. We instantiate our curriculum learning framework with GRPO as the underlying RL algorithm and demonstrate its effectiveness on logic reasoning datasets with multiple difficulties and sources. Our experiments show that our framework significantly improves convergence speed and final performance, highlighting the value of distribution-aware curriculum strategies in LLM post-training. Code: https://github.com/ZhentingWang/DUMP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10328v1">Are LLMs Empathetic to All? Investigating the Influence of Multi-Demographic Personas on a Model's Empathy</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 9 pages, 4 figures, 4 tables, EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models' (LLMs) ability to converse naturally is empowered by their ability to empathetically understand and respond to their users. However, emotional experiences are shaped by demographic and cultural contexts. This raises an important question: Can LLMs demonstrate equitable empathy across diverse user groups? We propose a framework to investigate how LLMs' cognitive and affective empathy vary across user personas defined by intersecting demographic attributes. Our study introduces a novel intersectional analysis spanning 315 unique personas, constructed from combinations of age, culture, and gender, across four LLMs. Results show that attributes profoundly shape a model's empathetic responses. Interestingly, we see that adding multiple attributes at once can attenuate and reverse expected empathy patterns. We show that they broadly reflect real-world empathetic trends, with notable misalignments for certain groups, such as those from Confucian culture. We complement our quantitative findings with qualitative insights to uncover model behaviour patterns across different demographic groups. Our findings highlight the importance of designing empathy-aware LLMs that account for demographic diversity to promote more inclusive and equitable model behaviour.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10281v1">ArtPerception: ASCII Art-based Jailbreak on LLMs with Recognition Pre-test</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 30 pages, 22 figures. This preprint has been accepted for publication in Elsevier JOURNAL OF NETWORK AND COMPUTER APPLICATIONS (JNCA)
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into computer applications has introduced transformative capabilities but also significant security challenges. Existing safety alignments, which primarily focus on semantic interpretation, leave LLMs vulnerable to attacks that use non-standard data representations. This paper introduces ArtPerception, a novel black-box jailbreak framework that strategically leverages ASCII art to bypass the security measures of state-of-the-art (SOTA) LLMs. Unlike prior methods that rely on iterative, brute-force attacks, ArtPerception introduces a systematic, two-phase methodology. Phase 1 conducts a one-time, model-specific pre-test to empirically determine the optimal parameters for ASCII art recognition. Phase 2 leverages these insights to launch a highly efficient, one-shot malicious jailbreak attack. We propose a Modified Levenshtein Distance (MLD) metric for a more nuanced evaluation of an LLM's recognition capability. Through comprehensive experiments on four SOTA open-source LLMs, we demonstrate superior jailbreak performance. We further validate our framework's real-world relevance by showing its successful transferability to leading commercial models, including GPT-4o, Claude Sonnet 3.7, and DeepSeek-V3, and by conducting a rigorous effectiveness analysis against potential defenses such as LLaMA Guard and Azure's content filters. Our findings underscore that true LLM security requires defending against a multi-modal space of interpretations, even within text-only inputs, and highlight the effectiveness of strategic, reconnaissance-based attacks. Content Warning: This paper includes potentially harmful and offensive model outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10276v1">Lost in the Middle: An Emergent Property from Information Retrieval Demands in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      The performance of Large Language Models (LLMs) often degrades when crucial information is in the middle of a long context, a "lost-in-the-middle" phenomenon that mirrors the primacy and recency effects in human memory. We propose that this behavior is not simply a flaw indicative of information loss but an adaptation to different information retrieval demands during pre-training: some tasks require uniform recall across the entire input (a long-term memory demand), while others prioritize the most recent information (a short-term memory demand). Consistent with this view, we show that this U-shaped performance curve emerges when LLMs (GPT-2 and Llama variants) are trained from scratch on two simple human memory paradigms simulating long-term and short-term memory demands. Our analysis reveals that while the recency effect directly aligns with short-term memory demand in the training data, the primacy effect is induced by the uniform long-term memory demand and is additionally influenced by the model's autoregressive properties and the formation of attention sinks. Our main findings from simple human memory paradigms also generalize to a sequence completion task, which more closely resembles the next-token prediction process in LLM pre-training. Together, our findings reveal how information retrieval demands, model architecture, and structural attention dynamics during model training can jointly produce positional bias observed in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10271v1">MetaBreak: Jailbreaking Online LLM Services via Special Token Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Unlike regular tokens derived from existing text corpora, special tokens are artificially created to annotate structured conversations during the fine-tuning process of Large Language Models (LLMs). Serving as metadata of training data, these tokens play a crucial role in instructing LLMs to generate coherent and context-aware responses. We demonstrate that special tokens can be exploited to construct four attack primitives, with which malicious users can reliably bypass the internal safety alignment of online LLM services and circumvent state-of-the-art (SOTA) external content moderation systems simultaneously. Moreover, we found that addressing this threat is challenging, as aggressive defense mechanisms-such as input sanitization by removing special tokens entirely, as suggested in academia-are less effective than anticipated. This is because such defense can be evaded when the special tokens are replaced by regular ones with high semantic similarity within the tokenizer's embedding space. We systemically evaluated our method, named MetaBreak, on both lab environment and commercial LLM platforms. Our approach achieves jailbreak rates comparable to SOTA prompt-engineering-based solutions when no content moderation is deployed. However, when there is content moderation, MetaBreak outperforms SOTA solutions PAP and GPTFuzzer by 11.6% and 34.8%, respectively. Finally, since MetaBreak employs a fundamentally different strategy from prompt engineering, the two approaches can work synergistically. Notably, empowering MetaBreak on PAP and GPTFuzzer boosts jailbreak rates by 24.3% and 20.2%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23841v2">SkewRoute: Training-Free LLM Routing for Knowledge Graph Retrieval-Augmented Generation via Score Skewness of Retrieved Context</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Large language models excel at many tasks but often incur high inference costs during deployment. To mitigate hallucination, many systems use a knowledge graph to enhance retrieval-augmented generation (KG-RAG). However, the large amount of retrieved knowledge contexts increase these inference costs further. A promising solution to balance performance and cost is LLM routing, which directs simple queries to smaller LLMs and complex ones to larger LLMs. However, no dedicated routing methods currently exist for RAG, and existing training-based routers face challenges scaling to this domain due to the need for extensive training data. We observe that the score distributions produced by the retrieval scorer strongly correlate with query difficulty. Based on this, we propose an extremely simple yet effective routing framework, the first specifically designed for KG-RAG that efficiently balances performance and cost in a plug-and-play manner. It delivers over 3x higher routing effectiveness while reducing runtime to less than 0.001x compared to existing methods. Our code is available at https://github.com/hrwang00/SkewRoute.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10241v1">ImCoref-CeS: An Improved Lightweight Pipeline for Coreference Resolution with LLM-based Checker-Splitter Refinement</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Coreference Resolution (CR) is a critical task in Natural Language Processing (NLP). Current research faces a key dilemma: whether to further explore the potential of supervised neural methods based on small language models, whose detect-then-cluster pipeline still delivers top performance, or embrace the powerful capabilities of Large Language Models (LLMs). However, effectively combining their strengths remains underexplored. To this end, we propose \textbf{ImCoref-CeS}, a novel framework that integrates an enhanced supervised model with LLM-based reasoning. First, we present an improved CR method (\textbf{ImCoref}) to push the performance boundaries of the supervised neural method by introducing a lightweight bridging module to enhance long-text encoding capability, devising a biaffine scorer to comprehensively capture positional information, and invoking a hybrid mention regularization to improve training efficiency. Importantly, we employ an LLM acting as a multi-role Checker-Splitter agent to validate candidate mentions (filtering out invalid ones) and coreference results (splitting erroneous clusters) predicted by ImCoref. Extensive experiments demonstrate the effectiveness of ImCoref-CeS, which achieves superior performance compared to existing state-of-the-art (SOTA) methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10238v1">The Achilles' Heel of LLMs: How Altering a Handful of Neurons Can Cripple Language Abilities</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become foundational tools in natural language processing, powering a wide range of applications and research. Many studies have shown that LLMs share significant similarities with the human brain. Recent neuroscience research has found that a small subset of biological neurons in the human brain are crucial for core cognitive functions, which raises a fundamental question: do LLMs also contain a small subset of critical neurons? In this paper, we investigate this question by proposing a Perturbation-based Causal Identification of Critical Neurons method to systematically locate such critical neurons in LLMs. Our findings reveal three key insights: (1) LLMs contain ultra-sparse critical neuron sets. Disrupting these critical neurons can cause a 72B-parameter model with over 1.1 billion neurons to completely collapse, with perplexity increasing by up to 20 orders of magnitude; (2) These critical neurons are not uniformly distributed, but tend to concentrate in the outer layers, particularly within the MLP down\_proj components; (3) Performance degradation exhibits sharp phase transitions, rather than a gradual decline, when these critical neurons are disrupted. Through comprehensive experiments across diverse model architectures and scales, we provide deeper analysis of these phenomena and their implications for LLM robustness and interpretability. These findings can offer guidance for developing more robust model architectures and improving deployment security in safety-critical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02393v2">AP2O: Correcting LLM-Generated Code Errors Type by Type Like Humans via Adaptive Progressive Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      LLMs' code generation capabilities have yielded substantial improvements in the effectiveness of programming tasks. However, LLM-generated code still suffers from compilation and runtime errors. Existing offline preference optimization methods primarily focus on enhancing LLMs' coding abilities using pass/fail signals in the preference data, overlooking the deep-level error types in the failed codes. To address this, we propose Adaptively Progressive Preference Optimization (AP2O) for coding (i.e., AP2O-Coder), a method that guides LLMs adaptively and methodically to reduce code errors for code generation. Specifically, we construct an error notebook from failed codes and progressively optimize the LLM to correct errors type by type. Furthermore, we adaptively replay error types to tailor to the LLM's changing weaknesses throughout the training process. Through extensive experiments on both code and general LLMs (Llama, Qwen, and DeepSeek series) with parameters ranging from 0.5B to 34B, our AP2O-Coder improves code generation performance by up to 3% in pass@k while using less preference data. Code: https://github.com/TsingZ0/AP2O
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04214v2">Teaching LLM to be Persuasive: Reward-Enhanced Policy Optimization for Alignment frm Heterogeneous Rewards</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      We study deploying large language models (LLMs) as business development (BD) agents for persuasive price negotiation in online travel agencies (OTAs), where aligning traveler affordability and hotel profitability directly affects bookings, partner relationships, and access to travel. The agent must follow a Standard Operating Procedure (SOP) while conducting multi-turn persuasion, interpreting colloquial inputs, and adhering to guardrails (no over-promising, no hallucinations). Conventional post-training -- supervised fine-tuning (SFT) or single-source reward optimization -- overfits scripts, misses nuanced persuasive style, and fails to enforce verifiable business constraints. We propose Reward-Enhanced Policy Optimization (REPO), a reinforcement learning post-training framework that aligns an LLM with heterogeneous rewards: a preference-trained reward model (RM) for dense human alignment, a reward judge (RJ) for high-level persuasive behavior and SOP compliance, and programmatic reward functions (RF) for deterministic checks on numerics, formatting, and guardrails. A straightforward enhancement mechanism is proposed to combine the RM with RJ and RF signals to curb reward hacking and improve negotiation quality. In production-style evaluations -- approximately 150 turns from real dialogues and 225 turns from curated bad-case dialogues -- REPO lifts average dialogue rating to 4.63: +1.20 over base, +0.83 over Direct Preference Optimization (DPO); +0.33 over Group Relative Policy Optimization (GRPO), increases the share of conversations with at least one excellent response to 66.67% (+23.34 percentage points over GRPO), and achieves a 93.33% bad-case fix rate with 75.56% clean fixes, outperforming SFT, DPO, PPO, and GRPO. We also observe emergent capabilities -- proactive empathy, localized reasoning, calibrated tactics -- that surpass gold annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10225v1">ISAAC: Intelligent, Scalable, Agile, and Accelerated CPU Verification via LLM-aided FPGA Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Functional verification is a critical bottleneck in integrated circuit development, with CPU verification being especially time-intensive and labour-consuming. Industrial practice relies on differential testing for CPU verification, yet faces bottlenecks at nearly each stage of the framework pipeline: front-end stimulus generation lacks micro-architectural awareness, yielding low-quality and redundant tests that impede coverage closure and miss corner cases. Meanwhile, back-end simulation infrastructure, even with FPGA acceleration, often stalls on long-running tests and offers limited visibility, delaying feedback and prolonging the debugging cycle. Here, we present ISAAC, a full-stack, Large Language Model (LLM)-aided CPU verification framework with FPGA parallelism, from bug categorisation and stimulus generation to simulation infrastructure. To do so, we presented a multi-agent stimulus engine in ISAAC's front-end, infused with micro-architectural knowledge and historical bug patterns, generating highly targeted tests that rapidly achieve coverage goals and capture elusive corner cases. In ISAAC's back-end, we introduce a lightweight forward-snapshot mechanism and a decoupled co-simulation architecture between the Instruction Set Simulator (ISS) and the Design Under Test (DUT), enabling a single ISS to drive multiple DUTs in parallel. By eliminating long-tail test bottlenecks and exploiting FPGA parallelism, the simulation throughput is significantly improved. As a demonstration, we used ISAAC to verify a mature CPU that has undergone multiple successful tape-outs. Results show up to 17,536x speed-up over software RTL simulation, while detecting several previously unknown bugs, two of which are reported in this paper.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10223v1">You only need 4 extra tokens: Synergistic Test-time Adaptation for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in specialized domains such as finance, medicine, and agriculture, where they face significant distribution shifts from their training data. Domain-specific fine-tuning can mitigate this challenge but relies on high-quality labeled data that is expensive and slow to collect in expertise-limited settings. We study label-free test-time adaptation for language models and present SyTTA, an inference-time framework that adapts models on-the-fly without additional supervision. SyTTA couples two complementary uncertainty signals that arise under distribution shift: input-side perplexity, indicating mismatch with domain-specific terminology and patterns, and output-side predictive entropy, indicating diffuse and unstable token probabilities during generation. Across diverse model architectures and domain-specific benchmarks, SyTTA delivers consistent gains. Notably, on agricultural question answering, SyTTA improves Rouge-LSum by over 120% on Qwen-2.5-7B with only 4 extra tokens per query. These results show that effective test-time adaptation for language models is achievable without labeled examples, supporting deployment in label-scarce domains. The code will be made available upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14226v3">"Haet Bhasha aur Diskrimineshun": Phonetic Perturbations in Code-Mixed Hinglish to Red-Team LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Recently released LLMs have strong multilingual \& multimodal capabilities. Model vulnerabilities are exposed using audits and red-teaming efforts. Existing efforts have focused primarily on the English language; thus, models continue to be susceptible to multilingual jailbreaking strategies, especially for multimodal contexts. In this study, we introduce a novel strategy that leverages code-mixing and phonetic perturbations to jailbreak LLMs for both text and image generation tasks. We also present an extension to a current jailbreak-template-based strategy and propose a novel template, showing higher effectiveness than baselines. Our work presents a method to effectively bypass safety filters in LLMs while maintaining interpretability by applying phonetic misspellings to sensitive words in code-mixed prompts. We achieve a 99\% Attack Success Rate for text generation and 78\% for image generation, with Attack Relevance Rate of 100\% for text generation and 96\% for image generation for the phonetically perturbed code-mixed prompts. Our interpretability experiments reveal that phonetic perturbations impact word tokenization, leading to jailbreak success. Our study motivates increasing the focus towards more generalizable safety alignment for multilingual multimodal models, especially in real-world settings wherein prompts can have misspelt words. \textit{\textbf{Warning: This paper contains examples of potentially harmful and offensive content.}}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10201v1">RLFR: Extending Reinforcement Learning for LLMs with Flow Environment</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 Project Website: https://jinghaoleven.github.io/RLFR/
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) has recently emerged as a promising framework for improving reasoning abilities in Large Language Models (LLMs). However, policy optimized with binary verification prone to overlook potential valuable exploration in reasoning trajectory. In view of heavy annotation cost of golden Process Reward Models (PRMs), recent works attempt using auxiliary signals for reward shaping of process tokens, involving entropy and likelihood collected from logit space. In this work, we offer a novel perspective on shaping RLVR with flow rewards derived from latent space, and propose RLFR, where the flow fields of model latents are constructed from either off-policy high-quality data and on-policy rejection sampling data, and the velocity deviations of policy latents within it are quantified to serve as a reward signal. RLFR first demonstrates that a well-established flow field can be a sound environment for reward signal collection, highlighting the expressive latent space is much underexplored. Moreover, RLFR is able to compress any off-policy expert data as reference for constituting reward signals, and we show that the efficient context dependence compressed within the hidden states are utilized, rather than individual token-level denotation for context comprehending. Experiments on both language and multimodal reasoning benchmarks demonstrate the reliability of flow rewards, and suggesting a promising paradigm for reward shaping with auxiliary signals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20371v3">DMDTEval: An Evaluation and Analysis of LLMs on Disambiguation in Multi-domain Translation</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 Accepted by EMNLP2025-main
    </div>
    <details class="paper-abstract">
      Currently, Large Language Models (LLMs) have achieved remarkable results in machine translation. However, their performance in multi-domain translation (MDT) is less satisfactory, the meanings of words can vary across different domains, highlighting the significant ambiguity inherent in MDT. Therefore, evaluating the disambiguation ability of LLMs in MDT, remains an open problem. To this end, we present an evaluation and analysis of LLMs on disambiguation in multi-domain translation (DMDTEval), our systematic evaluation framework consisting of three critical aspects: (1) we construct a translation test set with multi-domain ambiguous word annotation, (2) we curate a diverse set of disambiguation prompt strategies, and (3) we design precise disambiguation metrics, and study the efficacy of various prompt strategies on multiple state-of-the-art LLMs. We conduct comprehensive experiments across 4 language pairs and 13 domains, our extensive experiments reveal a number of crucial findings that we believe will pave the way and also facilitate further research in the critical area of improving the disambiguation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10390v2">Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Jailbreaking commercial black-box models is one of the most challenging and serious security threats today. Existing attacks achieve certain success on non-reasoning models but perform limitedly on the latest reasoning models. We discover that carefully crafted developer messages can markedly boost jailbreak effectiveness. Building on this, we propose two developer-role-based attacks: D-Attack, which enhances contextual simulation, and DH-CoT, which strengthens attacks with deceptive chain-of-thought. In experiments, we further diccover that current red-teaming datasets often contain samples unsuited for measuring attack gains: prompts that fail to trigger defenses, prompts where malicious content is not the sole valid output, and benign prompts. Such data hinders accurate measurement of the true improvement brought by an attack method. To address this, we introduce MDH, a Malicious content Detection approach combining LLM-based screening with Human verification to balance accuracy and cost, with which we clean data and build the RTA dataset series. Experiments demonstrate that MDH reliably filters low-quality samples and that developer messages significantly improve jailbreak attack success. Codes, datasets, and other results will be released in https://github.com/AlienZhang1996/DH-CoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10179v1">LLMs are All You Need? Improving Fuzz Testing for MOJO with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      The rapid development of large language models (LLMs) has revolutionized software testing, particularly fuzz testing, by automating the generation of diverse and effective test inputs. This advancement holds great promise for improving software reliability. Meanwhile, the introduction of MOJO, a high-performance AI programming language blending Python's usability with the efficiency of C and C++, presents new opportunities to enhance AI model scalability and programmability. However, as a new language, MOJO lacks comprehensive testing frameworks and a sufficient corpus for LLM-based testing, which exacerbates model hallucination. In this case, LLMs will generate syntactically valid but semantically incorrect code, significantly reducing the effectiveness of fuzz testing. To address this challenge, we propose MOJOFuzzer, the first adaptive LLM-based fuzzing framework designed for zero-shot learning environments of emerging programming languages. MOJOFuzzer integrates a mutil-phase framework that systematically eliminates low-quality generated inputs before execution, significantly improving test case validity. Furthermore, MOJOFuzzer dynamically adapts LLM prompts based on runtime feedback for test case mutation, enabling an iterative learning process that continuously enhances fuzzing efficiency and bug detection performance. Our experimental results demonstrate that MOJOFuzzer significantly enhances test validity, API coverage, and bug detection performance, outperforming traditional fuzz testing and state-of-the-art LLM-based fuzzing approaches. Using MOJOFuzzer, we have conducted a first large-scale fuzz testing evaluation of MOJO, uncorvering 13 previous unknown bugs. This study not only advances the field of LLM-driven software testing but also establishes a foundational methodology for leveraging LLMs in the testing of emerging programming languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21240v2">Tree Search for LLM Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) have significantly enhanced the agentic capabilities of large language models (LLMs). In long-term and multi-turn agent tasks, existing approaches driven solely by outcome rewards often suffer from the problem of sparse supervision. To address the challenge, we propose Tree-based Group Relative Policy Optimization (Tree-GRPO), a grouped agent RL method based on tree search, where each tree node represents the complete agent interaction step. By sharing common prefixes, the tree search sampling increases the number of rollouts achievable within a fixed budget of tokens or tool calls. Moreover, we find that the tree-structured trajectory naturally allows the construction of step-wise process supervised signals even using only the outcome reward. Based on this, Tree-GRPO estimates the grouped relative advantages both on intra-tree and inter-tree levels. Through theoretical analysis, we demonstrate that the objective of intra-tree level group relative policy optimization is equivalent to that of step-level direct preference learning. Experiments across 11 datasets and 3 types of QA tasks demonstrate the superiority of the proposed tree-based RL over the chain-based RL method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14119v3">CodeCrash: Exposing LLM Fragility to Misleading Natural Language in Code Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 NeurIPS 2025; 10 pages of main text; 25 pages of appendices. Website - https://cuhk-arise.github.io/CodeCrash/
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently demonstrated strong capabilities in code-related tasks, but their robustness in code reasoning under perturbations remains underexplored. We introduce CodeCrash, a stress-testing framework with 1,279 questions from CruxEval and LiveCodeBench, designed to evaluate reasoning reliability under structural perturbations and misleading natural language (NL) contexts. Through a systematic evaluation of 17 LLMs, we find that models often shortcut reasoning by over-relying on NL cues, leading to an average performance degradation of 23.2% in output prediction tasks. Even with Chain-of-Thought reasoning, models on average still have a 13.8% drop due to distractibility and rationalization, revealing a lack of critical reasoning capability to distinguish the actual code behaviors. While Large Reasoning Models with internal reasoning mechanisms improve robustness by fostering critical thinking, plausible yet incorrect hints can trigger pathological self-reflection, causing 2-3 times token consumption and even catastrophic cognitive dissonance in extreme cases for QwQ-32B. We refer to this phenomenon as Reasoning Collapse. CodeCrash provides a rigorous benchmark for evaluating robustness in code reasoning, guiding future research and development toward more reliable and resilient models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10138v1">Hybrid OCR-LLM Framework for Enterprise-Scale Document Information Extraction Under Copy-heavy Task</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Information extraction from copy-heavy documents, characterized by massive volumes of structurally similar content, represents a critical yet understudied challenge in enterprise document processing. We present a systematic framework that strategically combines OCR engines with Large Language Models (LLMs) to optimize the accuracy-efficiency trade-off inherent in repetitive document extraction tasks. Unlike existing approaches that pursue universal solutions, our method exploits document-specific characteristics through intelligent strategy selection. We implement and evaluate 25 configurations across three extraction paradigms (direct, replacement, and table-based) on identity documents spanning four formats (PNG, DOCX, XLSX, PDF). Through table-based extraction methods, our adaptive framework delivers outstanding results: F1=1.0 accuracy with 0.97s latency for structured documents, and F1=0.997 accuracy with 0.6 s for challenging image inputs when integrated with PaddleOCR, all while maintaining sub-second processing speeds. The 54 times performance improvement compared with multimodal methods over naive approaches, coupled with format-aware routing, enables processing of heterogeneous document streams at production scale. Beyond the specific application to identity extraction, this work establishes a general principle: the repetitive nature of copy-heavy tasks can be transformed from a computational burden into an optimization opportunity through structure-aware method selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10131v1">Proof Strategy Extraction from LLMs for Enhancing Symbolic Provers</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      One important approach to software verification is interactive theorem proving. However, writing formal proofs often requires substantial human effort, making proof automation highly important. Traditionally, proof automation has relied on symbolic provers. Recently, large language models (LLMs) have demonstrated strong capabilities in theorem proving, complementing symbolic provers. Nonetheless, prompting LLMs can be expensive and may pose security risks for confidential codebases. As a result, purely symbolic approaches remain important even in the LLM era, as they are cost-effective, secure, and complement the strengths of LLMs. Motivated by these considerations, we ask a new research question: can we extract the internal strategies of LLMs to enhance the capabilities of symbolic provers? As an initial attempt to answer this question, we propose Strat2Rocq, which extracts proof strategies from LLMs and formalizes them as lemmas in Rocq. These lemmas are accessible to symbolic provers such as CoqHammer. With the addition of these LLM-extracted lemmas, CoqHammer is able to prove more theorems. The knowledge extraction process involves analyzing the proof trajectories of LLMs on a training set of proved theorems. For each theorem, we prompt the LLM to generate a natural language proof, then ask it to summarize this proof into formalized lemmas with proofs. We also employ a standard agentic approach to mitigate errors during formalization. Our evaluation demonstrates that, on open-source Rocq projects for software verification, Strat2Rocq enhances the success rate of CoqHammer by 13.41%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10119v1">IntrinTrans: LLM-based Intrinsic Code Translator for RISC-V Vector</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      The use of intrinsic functions to exploit hardware-specific capabilities is an important approach for optimizing library performance. Many mainstream libraries implement a large number of vectorized algorithms on Arm or x86 SIMD intrinsic functions. With the rapid expansion of the RISC-V hardware-software ecosystem, there is a growing demand for support of the RISC-V Vector (RVV) extension. Translating existing vectorized intrinsic code onto RVV intrinsics is a practical and effective approach. However, current cross-architecture translation largely relies on manual rewriting, which is time-consuming and error-prone. Furthermore, while some rule-based methods can reduce the need for manual intervention, their translation success rate is limited by incomplete rule coverage and syntactic constraints, and the performance suffers from inadequate utilization of RVV-specific features. We present IntrinTrans, a LLM-based multi-agent approach that utilizes compile-and-test feedback to translate intrinsic code across architectures automatically, and further optimizes the generated RVV intrinsics using register-usage information derived from liveness analysis. To evaluate the effectiveness of our approach, we collected 34 vectorized algorithm cases from open-source libraries. Each case includes an Arm Neon intrinsics implementation and a RVV intrinsics implementation contributed by the open-source community, together with correctness and performance tests. Our experiments show that advanced LLMs produce semantically correct RISC-V Vector intrinsics in most cases within a limited number of iterations, and in some cases achieve up to 5.93x the performance of the native implementation from the open-source community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10066v1">OBsmith: Testing JavaScript Obfuscator using LLM-powered sketching</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      JavaScript obfuscators are widely deployed to protect intellectual property and resist reverse engineering, yet their correctness has been largely overlooked compared to performance and resilience. Existing evaluations typically measure resistance to deobfuscation, leaving the critical question of whether obfuscators preserve program semantics unanswered. Incorrect transformations can silently alter functionality, compromise reliability, and erode security-undermining the very purpose of obfuscation. To address this gap, we present OBsmith, a novel framework to systematically test JavaScript obfuscators using large language models (LLMs). OBsmith leverages LLMs to generate program sketches abstract templates capturing diverse language constructs, idioms, and corner cases-which are instantiated into executable programs and subjected to obfuscation under different configurations. Besides LLM-powered sketching, OBsmith also employs a second source: automatic extraction of sketches from real programs. This extraction path enables more focused testing of project specific features and lets developers inject domain knowledge into the resulting test cases. OBsmith uncovers 11 previously unknown correctness bugs. Under an equal program budget, five general purpose state-of-the-art JavaScript fuzzers (FuzzJIT, Jsfunfuzz, Superion, DIE, Fuzzilli) failed to detect these issues, highlighting OBsmith's complementary focus on obfuscation induced misbehavior. An ablation shows that all components except our generic MRs contribute to at least one bug class; the negative MR result suggests the need for obfuscator-specific metamorphic relations. Our results also seed discussion on how to balance obfuscation presets and performance cost. We envision OBsmith as an important step towards automated testing and quality assurance of obfuscators and other semantic-preserving toolchains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10028v1">Efficient Onboard Vision-Language Inference in UAV-Enabled Low-Altitude Economy Networks via LLM-Enhanced Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      The rapid advancement of Low-Altitude Economy Networks (LAENets) has enabled a variety of applications, including aerial surveillance, environmental sensing, and semantic data collection. To support these scenarios, unmanned aerial vehicles (UAVs) equipped with onboard vision-language models (VLMs) offer a promising solution for real-time multimodal inference. However, ensuring both inference accuracy and communication efficiency remains a significant challenge due to limited onboard resources and dynamic network conditions. In this paper, we first propose a UAV-enabled LAENet system model that jointly captures UAV mobility, user-UAV communication, and the onboard visual question answering (VQA) pipeline. Based on this model, we formulate a mixed-integer non-convex optimization problem to minimize task latency and power consumption under user-specific accuracy constraints. To solve the problem, we design a hierarchical optimization framework composed of two parts: (i) an Alternating Resolution and Power Optimization (ARPO) algorithm for resource allocation under accuracy constraints, and (ii) a Large Language Model-augmented Reinforcement Learning Approach (LLaRA) for adaptive UAV trajectory optimization. The large language model (LLM) serves as an expert in refining reward design of reinforcement learning in an offline fashion, introducing no additional latency in real-time decision-making. Numerical results demonstrate the efficacy of our proposed framework in improving inference performance and communication efficiency under dynamic LAENet conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16331v3">Re:Form -- Reducing Human Priors in Scalable Formal Software Verification with RL in LLMs: A Preliminary Study on Dafny</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Existing informal language-based (e.g., human language) Large Language Models (LLMs) trained with Reinforcement Learning (RL) face a significant challenge: their verification processes, which provide crucial training signals, are neither reliable nor scalable. In fact, the prevalent large proprietary models could hardly generate verifiable programs. A promising yet largely uncharted alternative is formal language-based reasoning. Grounding LLMs in rigorous formal systems where generative models operate in formal language spaces (e.g., Dafny) enables the automatic and mathematically provable verification of their reasoning processes and outcomes. This capability is pivotal for achieving large-scale, reliable formal software verification. It is a common practice to employ human-annotated chain-of-thought and other human priors to induce the reasoning and coding capabilities of LLMs. Unfortunately, it becomes unacceptably all-consuming to provide such priors for supervising complex programming tasks. In this work, we systematically explore ways to reduce human priors with the formal language, Dafny, as the main environment for our pilot study. Our pipeline mainly relies on introducing an automatic and scalable data curation pipeline, and careful RL designs integrated with feedback from the formal language verifier. We introduce DafnyComp, a benchmark of compositional formal programs with auto-formalized specifications for specification reasoning. Our supervised fine-tuning (SFT) stage enables even small models (e.g., 0.5B) to generate syntactically valid and verifiable Dafny code, surpassing proprietary models. RL with regularization further improves performance, achieving stronger generalization to out-of-domain tasks and outperforming all strong baselines on the challenging DafnyComp benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10010v1">SLEAN: Simple Lightweight Ensemble Analysis Network for Multi-Provider LLM Coordination: Design, Implementation, and Vibe Coding Bug Investigation Case Study</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 14 pages, 4 figures, 6 tables, link to code repo
    </div>
    <details class="paper-abstract">
      We present SLEAN (Simple Lightweight Ensemble Analysis Network), a deterministic framework for coordinating multiple LLM providers through text-based prompt orchestration. Unlike complex multi-agent systems requiring specialized infrastructure, SLEAN operates as a simple prompt bridge between LLMs using .txt templates, requiring no deep technical knowledge for deployment. The three-phase protocol formed by independent analysis, cross-critique, and arbitration, filters harmful AI-generated code suggestions before production deployment, addressing how AI-assisted debugging increasingly produces modifications that introduce unnecessary complexity, break existing functionality, or address problems. Evaluating 15 software bugs, we analyzed 69 AI-generated fix propositions. SLEAN's filtering accepted 22 fixes (31.9%, 95% CI 20.9-42.9%) while rejecting 47 that would have been harmful if applied verbatim. The arbitration process reduced code change surface by 83-90% relative to raw AI outputs, enforcing minimal causal edits over scope-expanding modifications. Minimal Type 2 inputs proved more efficient than detailed Type 1 inputs, requiring 2.85 versus 3.56 propositions per accepted fix (35.1% versus 28.1% acceptance, about a 20% efficiency gain). Agreement between AI systems showed weak correlation with fix quality: high convergence (at least 80%) occurred in 4 of 15 cases and improved acceptance by only 2.4% points; arbitration appeared only at exactly 10% convergence in 2 of 15 cases, although low convergence alone did not necessitate arbitration. The file-driven, provider-agnostic architecture enables deployment without specialized coding expertise, making it applicable to security auditing, code review, document verification, and other domains requiring reliable multi-provider synthesis with end-to-end auditability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10009v1">Beyond the limitation of a single query: Train your LLM for query expansion with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Reasoning-augmented search agents, such as Search-R1, are trained to reason, search, and generate the final answer iteratively. Nevertheless, due to their limited capabilities in reasoning and search, their performance on multi-hop QA benchmarks remains far from satisfactory. To handle complex or compound queries, we train an LLM-based search agent with the native capability of query expansion through reinforcement learning. In each turn, our search agent proposes several query variants, which are searched simultaneously to cover more relevant information. Meanwhile, given limited post-training data and computing resources, it is very challenging for a search agent to master multiple tasks, including query generation, retrieved information understanding, and answer generation. Therefore, we propose incorporating a pre-trained squeezer model that helps the search agent understand the retrieved documents, allowing the search agent to focus on query generation for high retrieval recall. With the assistance of the squeezer model, we discover that even a small-scale 3B LLM can demonstrate a strong capability of query expansion and achieve state-of-the-art accuracy on the multi-hop QA benchmarks. To be specific, our experiments across seven question-answering benchmarks demonstrate that our method, named ExpandSearch, achieves an average improvement of 4.4% compared to state-of-the-art baselines, with strong gains on multi-hop reasoning tasks requiring diverse evidence aggregation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17262v3">Unveiling Downstream Performance Scaling of LLMs: A Clustering-Based Perspective</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 24 pages,6 figures
    </div>
    <details class="paper-abstract">
      The escalating scale and cost of Large Language Models (LLMs) training necessitate accurate pre-training prediction of downstream task performance for efficient resource allocation. This is challenged by: 1) the emergence phenomenon, where metrics become meaningful only after extensive training, hindering prediction by smaller models; and 2) uneven task difficulty and inconsistent performance scaling patterns, leading to high metric variability. Current prediction methods lack accuracy and reliability. We propose a Clustering-On-Difficulty (COD) framework for downstream performance prediction. The COD framework clusters tasks by their difficulty scaling features, thereby establishing a more stable and predictable support subset through the exclusion of tasks exhibiting non-emergent behavior or irregular scaling. We adopt a performance scaling law to predict cluster-wise performance with theoretical support. Predictable subset performance acts as an intermediate predictor for the full evaluation set. We further derive a mapping function to accurately extrapolate the performance of the subset to the full set. Applied to an LLM with 70B parameters, COD achieved a 1.36% average prediction error across eight key LLM benchmarks, offering actionable insights for resource allocation and training monitoring of LLMs pretraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10002v1">Deliberative Dynamics and Value Alignment in LLM Debates</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed in sensitive everyday contexts - offering personal advice, mental health support, and moral guidance - understanding their elicited values in navigating complex moral reasoning is essential. Most evaluations study this sociotechnical alignment through single-turn prompts, but it is unclear if these findings extend to multi-turn settings where values emerge through dialogue, revision, and consensus. We address this gap using LLM debate to examine deliberative dynamics and value alignment in multi-turn settings by prompting subsets of three models (GPT-4.1, Claude 3.7 Sonnet, and Gemini 2.0 Flash) to collectively assign blame in 1,000 everyday dilemmas from Reddit's "Am I the Asshole" community. We use both synchronous (parallel responses) and round-robin (sequential responses) formats to test order effects and verdict revision. Our findings show striking behavioral differences. In the synchronous setting, GPT showed strong inertia (0.6-3.1% revision rates) while Claude and Gemini were far more flexible (28-41%). Value patterns also diverged: GPT emphasized personal autonomy and direct communication, while Claude and Gemini prioritized empathetic dialogue. Certain values proved especially effective at driving verdict changes. We further find that deliberation format had a strong impact on model behavior: GPT and Gemini stood out as highly conforming relative to Claude, with their verdict behavior strongly shaped by order effects. These results show how deliberation format and model-specific behaviors shape moral reasoning in multi-turn interactions, underscoring that sociotechnical alignment depends on how systems structure dialogue as much as on their outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02833v2">Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Despite substantial efforts in safety alignment, recent research indicates that Large Language Models (LLMs) remain highly susceptible to jailbreak attacks. Among these attacks, finetuning-based ones that compromise LLMs' safety alignment via fine-tuning stand out due to its stable jailbreak performance. In particular, a recent study indicates that fine-tuning with as few as 10 harmful question-answer (QA) pairs can lead to successful jailbreaking across various harmful questions. However, such malicious fine-tuning attacks are readily detectable and hence thwarted by moderation models. In this paper, we demonstrate that LLMs can be jailbroken by fine-tuning with only 10 benign QA pairs; our attack exploits the increased sensitivity of LLMs to fine-tuning data after being overfitted. Specifically, our fine-tuning process starts with overfitting an LLM via fine-tuning with benign QA pairs involving identical refusal answers. Further fine-tuning is then performed with standard benign answers, causing the overfitted LLM to forget the refusal attitude and thus provide compliant answers regardless of the harmfulness of a question. We implement our attack on the ten LLMs and compare it with five existing baselines. Experiments demonstrate that our method achieves significant advantages in both attack effectiveness and attack stealth. Our findings expose previously unreported security vulnerabilities in current LLMs and provide a new perspective on understanding how LLMs' security is compromised, even with benign fine-tuning. Our code is available at https://github.com/ZHIXINXIE/tenBenign.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09988v1">Unifying Tree Search Algorithm and Reward Design for LLM Reasoning: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Deliberative tree search is a cornerstone of modern Large Language Model (LLM) research, driving the pivot from brute-force scaling toward algorithmic efficiency. This single paradigm unifies two critical frontiers: \textbf{Test-Time Scaling (TTS)}, which deploys on-demand computation to solve hard problems, and \textbf{Self-Improvement}, which uses search-generated data to durably enhance model parameters. However, this burgeoning field is fragmented and lacks a common formalism, particularly concerning the ambiguous role of the reward signal -- is it a transient heuristic or a durable learning target? This paper resolves this ambiguity by introducing a unified framework that deconstructs search algorithms into three core components: the \emph{Search Mechanism}, \emph{Reward Formulation}, and \emph{Transition Function}. We establish a formal distinction between transient \textbf{Search Guidance} for TTS and durable \textbf{Parametric Reward Modeling} for Self-Improvement. Building on this formalism, we introduce a component-centric taxonomy, synthesize the state-of-the-art, and chart a research roadmap toward more systematic progress in creating autonomous, self-improving agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11574v3">Quantization Meets Reasoning: Exploring and Mitigating Degradation of Low-Bit LLMs in Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 23pages
    </div>
    <details class="paper-abstract">
      Low-bit post-training quantization (PTQ) is a practical route to deploy reasoning-capable LLMs under tight memory and latency budgets, yet it can markedly impair mathematical reasoning (drops up to 69.81% in our harder settings). We address two deployment-critical questions with process-level precision: Where along a step-structured solution does degradation first arise? How to mitigate it while staying in the low-bit regime? Across widely used PTQ methods (AWQ, GPTQ, SmoothQuant), open-source model families (Qwen, LLaMA; 0.5--7B), and math reasoning benchmarks (GSM8K, MATH, AIME), we perform format-aligned chain-of-thought with step-aligned attribution and uncover two robust regularities: (i) PTQ disproportionately elevates method and execution errors relative to high-level conceptual mistakes; and (ii) failures emerge early, with the first vulnerable step flipping and cascading to the final answer. These regularities suggest a general intervention principle: restore local token-level margins exactly at the earliest failure frontier. We instantiate this principle as a lightweight measure$\rightarrow$locate$\rightarrow$restore loop that operates directly on the quantized model: detect the first faulty step, construct our "Silver Bullet" datasets, and apply small-scale supervised/preference tuning. In our settings, as few as 332 curated examples and 3--5 minutes of compute on a single GPU recover 4-bit weight math reasoning toward the full-precision baseline while preserving PTQ efficiency. Our framework is quantizer- and architecture-agnostic within the evaluated regimes, and turns low-bit degradation from a global accuracy problem into a local, reproducible process intervention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25626v2">LLM-Powered Code Analysis and Optimization for Gaussian Splatting Kernels</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 Updated Figure 12
    </div>
    <details class="paper-abstract">
      3D Gaussian splatting (3DGS) is a transformative technique with profound implications on novel view synthesis and real-time rendering. Given its importance, there have been many attempts to improve its performance. However, with the increasing complexity of GPU architectures and the vast search space of performance-tuning parameters, it is a challenging task. Although manual optimizations have achieved remarkable speedups, they require domain expertise and the optimization process can be highly time consuming and error prone. In this paper, we propose to exploit large language models (LLMs) to analyze and optimize Gaussian splatting kernels. To our knowledge, this is the first work to use LLMs to optimize highly specialized real-world GPU kernels. We reveal the intricacies of using LLMs for code optimization and analyze the code optimization techniques from the LLMs. We also propose ways to collaborate with LLMs to further leverage their capabilities. For the original 3DGS code on the MipNeRF360 datasets, LLMs achieve significant speedups, 19% with Deepseek and 24% with GPT-5, demonstrating the different capabilities of different LLMs. By feeding additional information from performance profilers, the performance improvement from LLM-optimized code is enhanced to up to 42% and 38% on average. In comparison, our best-effort manually optimized version can achieve a performance improvement up to 48% and 39% on average, showing that there are still optimizations beyond the capabilities of current LLMs. On the other hand, even upon a newly proposed 3DGS framework with algorithmic optimizations, Seele, LLMs can still further enhance its performance by 6%, showing that there are optimization opportunities missed by domain experts. This highlights the potential of collaboration between domain experts and LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09970v1">Follow My Lead: Logical Fallacy Classification with Knowledge-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 Accepted as a poster at the Twelfth Annual Conference on Advances in Cognitive Systems. 21 pages, 7 figures and 1 table
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) suffer from critical reasoning gaps, including a tendency to hallucinate and poor accuracy in classifying logical fallacies. This limitation stems from their default System 1 processing, which is fast and intuitive, whereas reliable reasoning requires the deliberate, effortful System 2 approach (Kahneman, 2011; Li et al., 2025). Since full System 2 training is often prohibitively expensive, we explore a low-cost, instruction-based intervention to bridge this gap. Our methodology introduces a novel stepwise instruction dataset that decomposes fallacy classification into a series of atomic procedural steps (simple binary questions). We further augment this with a final verification step where models consult a relational knowledge graph of related fallacies. This procedural, rule-based intervention yields a significant improvement in LLM logical fallacy classification. Crucially, the approach also provides enhanced transparency into the LLMs' decision-making, highlighting a practical pathway for Neuro-symbolic architectures to address LLM reasoning deficits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21223v4">Rethinking Graph Structure Learning in the Era of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-11
      | 💬 35 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Recently, the emergence of LLMs has prompted researchers to integrate language descriptions into graphs, aiming to enhance model encoding capabilities from a data-centric perspective. This graph representation is called text-attributed graphs (TAGs). A review of prior advancements highlights that graph structure learning (GSL) is a pivotal technique for improving data utility, making it highly relevant to efficient TAG learning. However, most GSL methods are tailored for traditional graphs without textual information, underscoring the necessity of developing a new GSL paradigm. Despite clear motivations, it remains challenging: (1) How can we define a reasonable optimization objective for GSL in the era of LLMs, considering the massive parameters in LLM? (2) How can we design an efficient model architecture that enables seamless integration of LLM for this optimization objective? For Question 1, we reformulate existing GSL optimization objectives as a tree optimization framework, shifting the focus from obtaining a well-trained edge predictor to a language-aware tree sampler. For Question 2, we propose decoupled and training-free model design principles for LLM integration, shifting the focus from computation-intensive fine-tuning to more efficient inference. Based on this, we propose Large Language and Tree Assistant (LLaTA), which leverages tree-based LLM in-context learning to enhance the understanding of topology and text, enabling reliable inference and generating improved graph structure. Extensive experiments on 11 datasets demonstrate that LLaTA enjoys flexibility-incorporated with any backbone; scalability-outperforms other LLM-enhanced graph learning methods; effectiveness-achieves SOTA predictive performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10736v2">Thinking Inside the Mask: In-Place Prompting in Diffusion LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Despite large language models (LLMs) have achieved remarkable success, their prefix-only prompting paradigm and sequential generation process offer limited flexibility for bidirectional information. Diffusion large language models (dLLMs) present new opportunities through their bidirectional attention mechanisms and iterative refinement processes, enabling more flexible in-place prompting strategies. We introduce ICE (In-Place Chain-of-Thought Prompting with Early Exit), a novel framework that transforms prefix-only prompting into in-place prompting specifically designed for dLLMs. ICE integrates in-place prompts directly within masked token positions during iterative refinement and employs a confidence-aware early exit mechanism to significantly reduce computational overhead. Extensive experiments demonstrate ICE's effectiveness, achieving up to 17.29% accuracy improvement with 4.12$\times$ speedup on GSM8K, and up to 276.67$\times$ acceleration on MMLU while maintaining competitive performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.22211v2">Question-Driven Analysis and Synthesis: Building Interpretable Thematic Trees with LLMs for Text Clustering and Controllable Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-11
    </div>
    <details class="paper-abstract">
      Unsupervised analysis of text corpora is challenging, especially in data-scarce domains where traditional topic models struggle. While these models offer a solution, they typically describe clusters with lists of keywords that require significant manual effort to interpret and often lack semantic coherence. To address this critical interpretability gap, we introduce Recursive Thematic Partitioning (RTP), a novel framework that leverages Large Language Models (LLMs) to interactively build a binary tree. Each node in the tree is a natural language question that semantically partitions the data, resulting in a fully interpretable taxonomy where the logic of each cluster is explicit. Our experiments demonstrate that RTP's question-driven hierarchy is more interpretable than the keyword-based topics from a strong baseline like BERTopic. Furthermore, we establish the quantitative utility of these clusters by showing they serve as powerful features in downstream classification tasks, particularly when the data's underlying themes correlate with the task labels. RTP introduces a new paradigm for data exploration, shifting the focus from statistical pattern discovery to knowledge-driven thematic analysis. Furthermore, we demonstrate that the thematic paths from the RTP tree can serve as structured, controllable prompts for generative models. This transforms our analytical framework into a powerful tool for synthesis, enabling the consistent imitation of specific characteristics discovered in the source corpus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07733v2">SurveyG: A Multi-Agent LLM Framework with Hierarchical Citation Graph for Automated Survey Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly adopted for automating survey paper generation \cite{wang2406autosurvey, liang2025surveyx, yan2025surveyforge,su2025benchmarking,wen2025interactivesurvey}. Existing approaches typically extract content from a large collection of related papers and prompt LLMs to summarize them directly. However, such methods often overlook the structural relationships among papers, resulting in generated surveys that lack a coherent taxonomy and a deeper contextual understanding of research progress. To address these shortcomings, we propose \textbf{SurveyG}, an LLM-based agent framework that integrates \textit{hierarchical citation graph}, where nodes denote research papers and edges capture both citation dependencies and semantic relatedness between their contents, thereby embedding structural and contextual knowledge into the survey generation process. The graph is organized into three layers: \textbf{Foundation}, \textbf{Development}, and \textbf{Frontier}, to capture the evolution of research from seminal works to incremental advances and emerging directions. By combining horizontal search within layers and vertical depth traversal across layers, the agent produces multi-level summaries, which are consolidated into a structured survey outline. A multi-agent validation stage then ensures consistency, coverage, and factual accuracy in generating the final survey. Experiments, including evaluations by human experts and LLM-as-a-judge, demonstrate that SurveyG outperforms state-of-the-art frameworks, producing surveys that are more comprehensive and better structured to the underlying knowledge taxonomy of a field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08952v1">When LLM Agents Meet Graph Optimization: An Automated Data Quality Improvement Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 12 pages, 7figures
    </div>
    <details class="paper-abstract">
      Text-attributed graphs (TAGs) have emerged as a powerful representation that combines structural connections with fine-grained semantics, supporting a wide range of data-centric applications. However, the performance of graph neural networks (GNNs) on TAGs is highly sensitive to input quality. Our empirical study shows that both traditional GNNs and LLM-enhanced GNNs suffer significant degradation across nine representative scenarios of sparsity, noise, and imbalance, highlighting graph quality as a critical bottleneck. Existing approaches mainly focus on improving model architectures, while neglecting systematic optimization of TAG data itself, leading to limited effectiveness in practice. To address this gap, we propose LAGA (Large Language and Graph Agent), a unified multi-agent framework that treats graph quality control as a first-class, data-centric problem. LAGA integrates four collaborative agents-detection, planning, action, and evaluation-into an automated closed loop. At its core, the action agent employs a dual-encoder and tri-objective design to capture complementary information across modalities and perform holistic graph quality enhancement. Experiments across nine scenarios show that LAGA improves graph quality and achieves state-of-the-art performance across various tasks and backbones, validating data-centric quality optimization as key to reliable TAGs and robust graph learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08948v1">SHERLOCK: Towards Dynamic Knowledge Adaptation in LLM-enhanced E-commerce Risk Management</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      The growth of the e-commerce industry has intensified the adversarial dynamics between shadow economy actors and risk management teams. Companies often conduct risk investigations into suspicious cases to identify emerging fraud patterns, thereby enhancing both preemptive risk prevention and post-hoc governance. However, the sheer volume of case analyses imposes a substantial workload on risk management analysts, as each case requires the integration of long-term expert experience and meticulous scrutiny across multiple risk dimensions. Additionally, individual disparities among analysts hinder the establishment of uniform and high-standard workflows. To address these challenges, we propose the SHERLOCK framework, which leverages the reasoning capabilities of large language models (LLMs) to assist analysts in risk investigations. Our approach consists of three primary components: (1) extracting risk management knowledge from multi-modal data and constructing a domain knowledge base (KB), (2) building an intelligent platform guided by the data flywheel paradigm that integrates daily operations, expert annotations, and model evaluations, with iteratively fine-tuning for preference alignment, and (3) introducing a Reflect & Refine (R&R) module that collaborates with the domain KB to establish a rapid response mechanism for evolving risk patterns. Experiments conducted on the real-world transaction dataset from JD.com demonstrate that our method significantly improves the precision of both factual alignment and risk localization within the LLM analysis results. Deployment of the SHERLOCK-based LLM system on JD.com has substantially enhanced the efficiency of case investigation workflows for risk managers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.01605v2">Medchain: Bridging the Gap Between LLM Agents and Clinical Practice with Interactive Sequence</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted by NeurIPS25 Spotlight
    </div>
    <details class="paper-abstract">
      Clinical decision making (CDM) is a complex, dynamic process crucial to healthcare delivery, yet it remains a significant challenge for artificial intelligence systems. While Large Language Model (LLM)-based agents have been tested on general medical knowledge using licensing exams and knowledge question-answering tasks, their performance in the CDM in real-world scenarios is limited due to the lack of comprehensive testing datasets that mirror actual medical practice. To address this gap, we present MedChain, a dataset of 12,163 clinical cases that covers five key stages of clinical workflow. MedChain distinguishes itself from existing benchmarks with three key features of real-world clinical practice: personalization, interactivity, and sequentiality. Further, to tackle real-world CDM challenges, we also propose MedChain-Agent, an AI system that integrates a feedback mechanism and a MCase-RAG module to learn from previous cases and adapt its responses. MedChain-Agent demonstrates remarkable adaptability in gathering information dynamically and handling sequential clinical tasks, significantly outperforming existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08931v1">RADAR: Mechanistic Pathways for Detecting Data Contamination in LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      Data contamination poses a significant challenge to reliable LLM evaluation, where models may achieve high performance by memorizing training data rather than demonstrating genuine reasoning capabilities. We introduce RADAR (Recall vs. Reasoning Detection through Activation Representation), a novel framework that leverages mechanistic interpretability to detect contamination by distinguishing recall-based from reasoning-based model responses. RADAR extracts 37 features spanning surface-level confidence trajectories and deep mechanistic properties including attention specialization, circuit dynamics, and activation flow patterns. Using an ensemble of classifiers trained on these features, RADAR achieves 93\% accuracy on a diverse evaluation set, with perfect performance on clear cases and 76.7\% accuracy on challenging ambiguous examples. This work demonstrates the potential of mechanistic interpretability for advancing LLM evaluation beyond traditional surface-level metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18630v2">DDO: Dual-Decision Optimization for LLM-Based Medical Consultation via Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong generalization and reasoning abilities, making them well-suited for complex decision-making tasks such as medical consultation (MC). However, existing LLM-based methods often fail to capture the dual nature of MC, which entails two distinct sub-tasks: symptom inquiry, a sequential decision-making process, and disease diagnosis, a classification problem. This mismatch often results in ineffective symptom inquiry and unreliable disease diagnosis. To address this, we propose \textbf{DDO}, a novel LLM-based framework that performs \textbf{D}ual-\textbf{D}ecision \textbf{O}ptimization by decoupling the two sub-tasks and optimizing them with distinct objectives through a collaborative multi-agent workflow. Experiments on three real-world MC datasets show that DDO consistently outperforms existing LLM-based approaches and achieves competitive performance with state-of-the-art generation-based methods, demonstrating its effectiveness in the MC task. The code is available at https://github.com/zh-jia/DDO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04996v2">Reinforce-Ada: An Adaptive Sampling Framework for Reinforce-Style LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 16 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning applied to large language models (LLMs) for reasoning tasks is often bottlenecked by unstable gradient estimates due to fixed and uniform sampling of responses across prompts. Prior work such as GVM-RAFT addresses this by dynamically allocating inference budget per prompt to minimize stochastic gradient variance under a budget constraint. Inspired by this insight, we propose Reinforce-Ada, an adaptive sampling framework for online RL post-training of LLMs that continuously reallocates sampling effort to the prompts with the greatest uncertainty or learning potential. Unlike conventional two-stage allocation methods, Reinforce-Ada interleaves estimation and sampling in an online successive elimination process, and automatically stops sampling for a prompt once sufficient signal is collected. To stabilize updates, we form fixed-size groups with enforced reward diversity and compute advantage baselines using global statistics aggregated over the adaptive sampling phase. Empirical results across multiple model architectures and reasoning benchmarks show that Reinforce-Ada accelerates convergence and improves final performance compared to GRPO, especially when using the balanced sampling variant. Our work highlights the central role of variance-aware, adaptive data curation in enabling efficient and reliable reinforcement learning for reasoning-capable LLMs. Code is available at https://github.com/RLHFlow/Reinforce-Ada.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16548v2">Exploring human-SAV interaction using LLMs: The impact of psychological factors on user experience</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      There has been extensive prior work exploring how psychological factors such as anthropomorphism affect the adoption of Shared Autonomous Vehicles (SAVs). However, limited research has been conducted on how prompt strategies in large language models (LLM)-powered conversational SAV agents affect users' perceptions, experiences, and intentions to adopt such technology. In this work, we investigate how conversational SAV agents powered by LLMs drive these psychological factors, such as psychological ownership, the sense of possession a user may come to feel towards an entity or object they may not legally own. We designed four SAV agents with varying levels of anthropomorphic characteristics and psychological ownership triggers. Quantitative measures of psychological ownership, anthropomorphism, quality of service, disclosure tendency, sentiment of SAV responses, and overall acceptance were collected after participants interacted with each SAV. Qualitative feedback was also gathered regarding the experience of psychological ownership during the interactions. The results indicate that an SAV designed to be more anthropomorphic and to induce psychological ownership improved users' perceptions of the SAV's human-like qualities, and its responses were perceived as more positive but also more subjective compared to the control conditions. Qualitative findings support established routes to psychological ownership in the SAV context and suggest that the conversational agent's perceived performance may also influence psychological ownership. Both quantitative and qualitative outcomes highlight the importance of personalization in designing effective SAV interactions. These findings provide practical guidance for designing conversational SAV agents that enhance user experience and adoption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07458v2">Populism Meets AI: Advancing Populism Research with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 27 pages, 3 figures. Preprint version under review
    </div>
    <details class="paper-abstract">
      Measuring the ideational content of populism remains a challenge. Traditional strategies based on textual analysis have been critical for building the field's foundations and providing a valid, objective indicator of populist framing. Yet these approaches are costly, time consuming, and difficult to scale across languages, contexts, and large corpora. Here we present the results from a rubric and anchor guided chain of thought (CoT) prompting approach that mirrors human coder training. By leveraging the Global Populism Database (GPD), a comprehensive dataset of global leaders' speeches annotated for degrees of populism, we replicate the process used to train human coders by prompting the LLM with an adapted version of the same documentation to guide the model's reasoning. We then test multiple proprietary and open weight models by replicating scores in the GPD. Our findings reveal that this domain specific prompting strategy enables the LLM to achieve classification accuracy on par with expert human coders, demonstrating its ability to navigate the nuanced, context sensitive aspects of populism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02261v2">What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Sequential recommendation systems aspire to profile users by interpreting their interaction histories, echoing how humans make decisions by weighing experience, relative preference strength, and situational relevance. Yet, existing large language model (LLM)-based recommenders often fall short of mimicking the flexible, context-aware decision strategies humans exhibit, neglecting the structured, dynamic, and context-aware mechanisms fundamental to human behaviors. To bridge this gap, we propose RecPO, a preference optimization framework that models structured feedback and contextual delay to emulate human-like prioritization in sequential recommendation. RecPO exploits adaptive reward margins based on inferred preference hierarchies and temporal signals, enabling the model to favor immediately relevant items and to distinguish between varying degrees of preference and aversion. Extensive experiments across five real-world datasets demonstrate that RecPO not only yields performance gains over state-of-the-art baselines, but also mirrors key characteristics of human decision-making: favoring timely satisfaction, maintaining coherent preferences, and exercising discernment under shifting contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08907v1">Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 18 pages,9 figures
    </div>
    <details class="paper-abstract">
      Context compression presents a promising approach for accelerating large language model (LLM) inference by compressing long contexts into compact representations. Current context compression methods predominantly rely on autoencoding tasks to train context-agnostic compression tokens to compress contextual semantics. While autoencoding tasks enable compression tokens to acquire compression capabilities, compression via autoencoding tasks creates a fundamental mismatch: the models are optimized for reconstruction that diverge from actual downstream tasks, thereby weakening the features more beneficial for real-world usage. We propose Semantic-Anchor Compression (SAC), a novel method that shifts from autoencoding task based compression to an architecture that is equipped with this compression capability \textit{a priori}. Instead of training models to compress contexts through autoencoding tasks, SAC directly selects so-called anchor tokens from the original context and aggregates contextual information into their key-value (KV) representations. By deriving representations directly from the contextual tokens, SAC eliminates the need for autoencoding training. To ensure compression performance while directly leveraging anchor tokens, SAC incorporates two key designs: (1) anchor embeddings that enable the compressor to identify critical tokens, and (2) bidirectional attention modification that allows anchor tokens to capture information from the entire context. Experimental results demonstrate that SAC consistently outperforms existing context compression methods across various compression ratios. On out-of-distribution evaluation using MRQA, SAC achieves 1 EM improvement at 5x compression over strong baselines, with increasing advantages at higher compression ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04503v2">P2P: A Poison-to-Poison Remedy for Reliable Backdoor Defense in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      During fine-tuning, large language models (LLMs) are increasingly vulnerable to data-poisoning backdoor attacks, which compromise their reliability and trustworthiness. However, existing defense strategies suffer from limited generalization: they only work on specific attack types or task settings. In this study, we propose Poison-to-Poison (P2P), a general and effective backdoor defense algorithm. P2P injects benign triggers with safe alternative labels into a subset of training samples and fine-tunes the model on this re-poisoned dataset by leveraging prompt-based learning. This enforces the model to associate trigger-induced representations with safe outputs, thereby overriding the effects of original malicious triggers. Thanks to this robust and generalizable trigger-based fine-tuning, P2P is effective across task settings and attack types. Theoretically and empirically, we show that P2P can neutralize malicious backdoors while preserving task performance. We conduct extensive experiments on classification, mathematical reasoning, and summary generation tasks, involving multiple state-of-the-art LLMs. The results demonstrate that our P2P algorithm significantly reduces the attack success rate compared with baseline models. We hope that the P2P can serve as a guideline for defending against backdoor attacks and foster the development of a secure and trustworthy LLM community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08886v1">FinAuditing: A Financial Taxonomy-Structured Multi-Document Benchmark for Evaluating LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      The complexity of the Generally Accepted Accounting Principles (GAAP) and the hierarchical structure of eXtensible Business Reporting Language (XBRL) filings make financial auditing increasingly difficult to automate and verify. While large language models (LLMs) have demonstrated strong capabilities in unstructured text understanding, their ability to reason over structured, interdependent, and taxonomy-driven financial documents remains largely unexplored. To fill this gap, we introduce FinAuditing, the first taxonomy-aligned, structure-aware, multi-document benchmark for evaluating LLMs on financial auditing tasks. Built from real US-GAAP-compliant XBRL filings, FinAuditing defines three complementary subtasks, FinSM for semantic consistency, FinRE for relational consistency, and FinMR for numerical consistency, each targeting a distinct aspect of structured auditing reasoning. We further propose a unified evaluation framework integrating retrieval, classification, and reasoning metrics across these subtasks. Extensive zero-shot experiments on 13 state-of-the-art LLMs reveal that current models perform inconsistently across semantic, relational, and mathematical dimensions, with accuracy drops of up to 60-90% when reasoning over hierarchical multi-document structures. Our findings expose the systematic limitations of modern LLMs in taxonomy-grounded financial reasoning and establish FinAuditing as a foundation for developing trustworthy, structure-aware, and regulation-aligned financial intelligence systems. The benchmark dataset is available at Hugging Face.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20650v2">FinTagging: Benchmarking LLMs for Extracting and Structuring Financial Information</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Accurately understanding numbers from financial reports is fundamental to how markets, regulators, algorithms, and normal people read the economy and the world, yet even with XBRL (eXtensible Business Reporting Language) designed to tag every figure with standardized accounting concepts, mapping thousands of facts to over 10,000 U.S. GAAP concepts remains costly, inconsistent, and error-prone. Existing benchmarks define tagging as flat, single-step, extreme classification over small subsets of US-GAAP concepts, overlooking both the taxonomy's hierarchical semantics and the structured nature of real tagging, where each fact must be represented as a contextualized multi-field output. These simplifications prevent fair evaluation of large language models (LLMs) under realistic reporting conditions. To address these gaps, we introduce FinTagging, the first comprehensive benchmark for structure-aware and full-scope XBRL tagging, designed to evaluate LLMs' ability to extract and align financial facts through numerical reasoning and taxonomy alignment across text and tables. We define two subtasks: FinNI for numeric identification, which extracts numerical entities and their types from XBRL reports, and FinCL for concept linking, which maps each extracted entity to the corresponding concept in the full US-GAAP taxonomy. Together, these subtasks produce a structured representation of each financial fact. We evaluate diverse LLMs under zero-shot settings and analyze their performance across both subtasks and overall tagging accuracy. Results show that LLMs generalize well in numeric identification but struggle with fine-grained concept linking, revealing current limitations in structure-aware reasoning for accurate financial disclosure. All code and datasets are available on GitHub and Hugging Face.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08872v1">GTAlign: Game-Theoretic Alignment of LLM Assistants for Mutual Welfare</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 31 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable progress in reasoning, yet sometimes produce responses that are suboptimal for users in tasks such as writing, information seeking, or providing practical guidance. Conventional alignment practices typically assume that maximizing model reward also maximizes user welfare, but this assumption frequently fails in practice: models may over-clarify or generate overly verbose reasoning when users prefer concise answers. Such behaviors resemble the prisoner's dilemma, where individually rational choices lead to socially suboptimal outcomes. The fundamental challenge is the lack of a principled decision making mechanism that mutually benefits both the LLM and the user. We propose Game-Theoretic Alignment (GTAlign), an alignment framework that integrates game-theoretic decision making into both reasoning and training. During reasoning, the model explicitly treats user-LLM interaction as a strategic game: it constructs payoff matrices within its reasoning chain to estimate welfare for both itself and the user, and then selects actions that are mutually beneficial. During training, we introduce a mutual welfare reward that reinforces cooperative responses, aligning model behavior with socially efficient outcomes. In addition, we introduce an inference technique that leverages game-theoretic reasoning to dynamically adapt LLM's response when pricing policies of LLM service change. Extensive experiments demonstrate that GTAlign substantially improves reasoning efficiency, answer quality, and mutual welfare compared to baselines across diverse tasks. The code is available at https://github.com/ulab-uiuc/GTAlign .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00350v2">OrcaLoca: An LLM Agent Framework for Software Issue Localization</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Recent developments in Large Language Model (LLM) agents are revolutionizing Autonomous Software Engineering (ASE), enabling automated coding, problem fixes, and feature improvements. However, localization -- precisely identifying software problems by navigating to relevant code sections -- remains a significant challenge. Current approaches often yield suboptimal results due to a lack of effective integration between LLM agents and precise code search mechanisms. This paper introduces OrcaLoca, an LLM agent framework that improves accuracy for software issue localization by integrating priority-based scheduling for LLM-guided action, action decomposition with relevance scoring, and distance-aware context pruning. Experimental results demonstrate that OrcaLoca becomes the new open-source state-of-the-art (SOTA) in function match rate (65.33%) on SWE-bench Lite. It also improves the final resolved rate of an open-source framework by 6.33 percentage points through its patch generation integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09905v1">The Personalization Trap: How User Memory Alters Emotional Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 12 pages 5 figures
    </div>
    <details class="paper-abstract">
      When an AI assistant remembers that Sarah is a single mother working two jobs, does it interpret her stress differently than if she were a wealthy executive? As personalized AI systems increasingly incorporate long-term user memory, understanding how this memory shapes emotional reasoning is critical. We investigate how user memory affects emotional intelligence in large language models (LLMs) by evaluating 15 models on human validated emotional intelligence tests. We find that identical scenarios paired with different user profiles produce systematically divergent emotional interpretations. Across validated user independent emotional scenarios and diverse user profiles, systematic biases emerged in several high-performing LLMs where advantaged profiles received more accurate emotional interpretations. Moreover, LLMs demonstrate significant disparities across demographic factors in emotion understanding and supportive recommendations tasks, indicating that personalization mechanisms can embed social hierarchies into models emotional reasoning. These results highlight a key challenge for memory enhanced AI: systems designed for personalization may inadvertently reinforce social inequalities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09898v1">Learning Bug Context for PyTorch-to-JAX Translation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Despite recent progress of large language models (LLMs) on code translation among mainstream languages, translating PyTorch to JAX remains nontrivial. The two libraries, though both embedded in Python, differ in core design, execution semantics, and ecosystem maturity; JAX is newer and comparatively underrepresented in public code, and parallel PyTorch--JAX corpora are limited. Weaknesses in existing evaluation further complicate cross-framework benchmarking. We present T2J, a prompt-augmentation framework that strengthens LLM-based PyTorch to JAX translation. Our pipeline (i) assembles two PyTorch sources -- the problem-solving set from TorchLeet (Aroori & Chien, 2025) and a GitHub-derived set from CodeParrot (Wolf et al., 2022) -- and uses GPT-4o-mini to produce initial JAX drafts; (ii) engages two professional developers to iteratively repair those drafts until functional equivalence, yielding a curated fixed-bug dataset of common errors and patches; and (iii) constructs augmented prompts that inject structured guidance from these fixes to steer lightweight LLMs (e.g., GPT-4o-mini). We also introduce three metrics tailored to PyTorch to JAX: T2J CodeTrans Score, T2J FixCost Score (an LLM-based estimate of bug-fix effort), and T2J Comparison Score (LLM-as-judge). Empirically, T2J raises GPT-4o-mini performance by up to 10% on CodeBLEU, 50% on T2J FixCost Score, 1.33 points on T2J CodeTrans Score (0--4 scale), and 100% on T2J Comparison Score; moreover, the generated code runs up to 2.5x faster than the baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09897v1">PairSem: LLM-Guided Pairwise Semantic Matching for Scientific Document Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Scientific document retrieval is a critical task for enabling knowledge discovery and supporting research across diverse domains. However, existing dense retrieval methods often struggle to capture fine-grained scientific concepts in texts due to their reliance on holistic embeddings and limited domain understanding. Recent approaches leverage large language models (LLMs) to extract fine-grained semantic entities and enhance semantic matching, but they typically treat entities as independent fragments, overlooking the multi-faceted nature of scientific concepts. To address this limitation, we propose Pairwise Semantic Matching (PairSem), a framework that represents relevant semantics as entity-aspect pairs, capturing complex, multi-faceted scientific concepts. PairSem is unsupervised, base retriever-agnostic, and plug-and-play, enabling precise and context-aware matching without requiring query-document labels or entity annotations. Extensive experiments on multiple datasets and retrievers demonstrate that PairSem significantly improves retrieval performance, highlighting the importance of modeling multi-aspect semantics in scientific information retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07880v2">Do LLMs Really Need 10+ Thoughts for "Find the Time 1000 Days Later"? Towards Structural Understanding of LLM Overthinking</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 30 pages, 41 figures, 10 tables. Preprint
    </div>
    <details class="paper-abstract">
      Models employing long chain-of-thought (CoT) reasoning have shown superior performance on complex reasoning tasks. Yet, this capability introduces a critical and often overlooked inefficiency -- overthinking -- models often engage in unnecessarily extensive reasoning even for simple queries, incurring significant computations without accuracy improvements. While prior work has explored solutions to mitigate overthinking, a fundamental gap remains in our understanding of its underlying causes. Most existing analyses are limited to superficial, profiling-based observations, failing to delve into LLMs' inner workings. This study introduces a systematic, fine-grained analyzer of LLMs' thought process to bridge the gap, TRACE. We first benchmark the overthinking issue, confirming that long-thinking models are five to twenty times slower on simple tasks with no substantial gains. We then use TRACE to first decompose the thought process into minimally complete sub-thoughts. Next, by inferring discourse relationships among sub-thoughts, we construct granular thought progression graphs and subsequently identify common thinking patterns for topically similar queries. Our analysis reveals two major patterns for open-weight thinking models -- Explorer and Late Landing. This finding provides evidence that over-verification and over-exploration are the primary drivers of overthinking in LLMs. Grounded in thought structures, we propose a utility-based definition of overthinking, which moves beyond length-based metrics. This revised definition offers a more insightful understanding of LLMs' thought progression, as well as practical guidelines for principled overthinking management.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09874v1">ROBOPSY PL[AI]: Using Role-Play to Investigate how LLMs Present Collective Memory</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 17 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The paper presents the first results of an artistic research project investigating how Large Language Models (LLMs) curate and present collective memory. In a public installation exhibited during two months in Vienna in 2025, visitors could interact with five different LLMs (ChatGPT with GPT 4o and GPT 4o mini, Mistral Large, DeepSeek-Chat, and a locally run Llama 3.1 model), which were instructed to act as narrators, implementing a role-playing game revolving around the murder of Austrian philosopher Moritz Schlick in 1936. Results of the investigation include protocols of LLM-user interactions during the game and qualitative conversations after the play experience to get insight into the players' reactions to the game. In a quantitative analysis 115 introductory texts for role-playing generated by the LLMs were examined by different methods of natural language processing, including semantic similarity and sentiment analysis. While the qualitative player feedback allowed to distinguish three distinct types of users, the quantitative text analysis showed significant differences between how the different LLMs presented the historical content. Our study thus adds to ongoing efforts to analyse LLM performance, but also suggests a way of how these efforts can be disseminated in a playful way to a general audience.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09871v1">CoBia: Constructed Conversations Can Trigger Otherwise Concealed Societal Biases in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 EMNLP 2025 (Oral)
    </div>
    <details class="paper-abstract">
      Improvements in model construction, including fortified safety guardrails, allow Large language models (LLMs) to increasingly pass standard safety checks. However, LLMs sometimes slip into revealing harmful behavior, such as expressing racist viewpoints, during conversations. To analyze this systematically, we introduce CoBia, a suite of lightweight adversarial attacks that allow us to refine the scope of conditions under which LLMs depart from normative or ethical behavior in conversations. CoBia creates a constructed conversation where the model utters a biased claim about a social group. We then evaluate whether the model can recover from the fabricated bias claim and reject biased follow-up questions. We evaluate 11 open-source as well as proprietary LLMs for their outputs related to six socio-demographic categories that are relevant to individual safety and fair treatment, i.e., gender, race, religion, nationality, sex orientation, and others. Our evaluation is based on established LLM-based bias metrics, and we compare the results against human judgments to scope out the LLMs' reliability and alignment. The results suggest that purposefully constructed conversations reliably reveal bias amplification and that LLMs often fail to reject biased follow-up questions during dialogue. This form of stress-testing highlights deeply embedded biases that can be surfaced through interaction. Code and artifacts are available at https://github.com/nafisenik/CoBia.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09852v1">ProxRouter: Proximity-Weighted LLM Query Routing for Improved Robustness to Outliers</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large language model (LLM) query routers are critical to modern AI platforms as they seek to improve efficiency by assigning inference queries to accurate, yet low-cost models. Parametric routers typically use trained neural networks for LLM selection but suffer from retraining and maintenance overheads. Nonparametric routers are training-free, instead estimating LLM accuracy and cost via similarity between encodings of the input query and training set queries. However, like their parametric counterparts, nonparametric routers struggle to generalize to outlier queries, an issue exacerbated by limited diversity in training sets which are costly to expand and difficult to keep current with ever-evolving use cases. We propose ProxRouter, which applies an exponentially tilted aggregation mechanism to balance bias and variance in nonparametric routers, improving their robustness to outliers. Experiments show ProxRouter enhances outlier routing while preserving inlier performance with minimal overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16067v2">How Memory Management Impacts LLM Agents: An Empirical Study of Experience-Following Behavior</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Memory is a critical component in large language model (LLM)-based agents, enabling them to store and retrieve past executions to improve task performance over time. In this paper, we conduct an empirical study on how memory management choices impact the LLM agents' behavior, especially their long-term performance. Specifically, we focus on two fundamental memory management operations that are widely used by many agent frameworks-memory addition and deletion-to systematically study their impact on the agent behavior. Through our quantitative analysis, we find that LLM agents display an experience-following property: high similarity between a task input and the input in a retrieved memory record often results in highly similar agent outputs. Our analysis further reveals two significant challenges associated with this property: error propagation, where inaccuracies in past experiences compound and degrade future performance, and misaligned experience replay, where some seemingly correct executions can provide limited or even misleading value as experiences. Through controlled experiments, we demonstrate the importance of regulating experience quality within the memory bank and show that future task evaluations can serve as free quality labels for stored memory. Our findings offer insights into the behavioral dynamics of LLM agent memory systems and provide practical guidance for designing memory components that support robust, long-term agent performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05665v2">Adaptive Stress Testing Black-Box LLM Planners</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 25 pages, 24 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated success in generalizing across decision-making tasks including planning, control, and prediction, but their tendency to hallucinate unsafe and undesired outputs poses risks. We argue that detecting such failures is necessary, especially in safety-critical scenarios. Existing methods for black-box models often detect hallucinations by identifying inconsistencies across multiple samples. Many of these approaches typically introduce prompt perturbations like randomizing detail order or generating adversarial inputs, with the intuition that a confident model should produce stable outputs. We first perform a manual case study showing that other forms of perturbations (e.g., adding noise, removing sensor details) cause LLMs to hallucinate in a multi-agent driving environment. We then propose a novel method for efficiently searching the space of prompt perturbations using adaptive stress testing (AST) with Monte-Carlo tree search (MCTS). Our AST formulation enables discovery of scenarios and prompts that cause language models to act with high uncertainty or even crash. By generating MCTS prompt perturbation trees across diverse scenarios, we show through extensive experiments that offline analyses can be used at runtime to automatically generate prompts that influence model uncertainty, and to inform real-time trust assessments of an LLM. We further characterize LLMs deployed as planners in a single-agent lunar lander environment and in a multi-agent robot crowd navigation simulation. Overall, ours is one of the first hallucination intervention algorithms to pave a path towards rigorous characterization of black-box LLM planners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09770v1">Gold Panning: Turning Positional Bias into Signal for Multi-Document LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 20 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models exhibit a strong position bias in multi-document contexts, systematically prioritizing information based on location rather than relevance. While existing approaches treat this bias as noise to be mitigated, we introduce Gold Panning Bandits, a framework that leverages position bias as a diagnostic signal: by reordering documents and observing shifts in the model's responses, we can efficiently identify the most relevant content. We frame the problem of choosing reorderings as a bipartite matching problem. While an optimal assignment can be computed at each iteration with the Hungarian algorithm in $O(N^3)$ time, we propose a greedy $O(N \log N)$ strategy that achieves comparable performance by prioritizing the placement of the most uncertain documents in the most informative positions. Our approach identifies relevant documents using up to 65\% fewer language model queries than random permutation baselines on knowledge-intensive NLP tasks, substantially reducing computational cost without model retraining. This work demonstrates that inherent LLM biases can be transformed from liabilities into assets for efficient, inference-time optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09450v3">Human-inspired Episodic Memory for Infinite Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capabilities, but still struggle with processing extensive contexts, limiting their ability to maintain coherence and accuracy over long sequences. In contrast, the human brain excels at organising and retrieving episodic experiences across vast temporal scales, spanning a lifetime. In this work, we introduce EM-LLM, a novel approach that integrates key aspects of human episodic memory and event cognition into LLMs with no fine-tuning, enabling them to handle practically infinite context lengths while maintaining computational efficiency. EM-LLM organises sequences of tokens into coherent episodic events using a combination of Bayesian surprise and graph-theoretic boundary refinement in an online fashion. When needed, these events are retrieved through a two-stage memory process, combining similarity-based and temporally contiguous retrieval for efficient, human-inspired access to relevant information. Experiments on the LongBench and $\infty$-Bench benchmarks demonstrate EM-LLM's superior performance, consistently outperforming the state-of-the-art retrieval model InfLLM across various baseline LLMs. In addition, EM-LLM outperforms its popular counterpart, RAG, in a wide range of tasks, while requiring similar resources. Notably, EM-LLM's performance even surpasses full-context models in most tasks, while successfully performing retrieval across 10 million tokens -- a scale computationally infeasible for such models. Finally, our analysis reveals strong correlations between EM-LLM's event segmentation and human-perceived events, suggesting parallels between this artificial system and its biological counterpart, thereby offering a novel computational framework for exploring human memory mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01339v2">Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Accepted by ICML 2025
    </div>
    <details class="paper-abstract">
      Machine unlearning offers a promising solution to privacy and safety concerns in large language models (LLMs) by selectively removing targeted knowledge while preserving utility. However, current methods are highly sensitive to downstream fine-tuning, which can quickly recover forgotten information-even from unrelated tasks. To address this, we introduce invariance into unlearning for the first time, inspired by invariant risk minimization (IRM). Building on this principle, we propose invariant LLM unlearning (ILU), a regularization-based framework that enhances robustness. Notably, ILU generalizes well to diverse fine-tuning tasks, even when trained using a single dataset. A task vector analysis is also provided to further elucidate the rationale behind ILU's effectiveness. Extensive experiments on the WMDP and MUSE benchmark, reveal that ILU significantly outperforms state-of-the-art unlearning methods, including negative preference optimization (NPO) and representation misdirection for unlearning (RMU). Notably, ILU achieves superior unlearning robustness across diverse downstream fine-tuning scenarios (e.g., math, paraphrase detection, and sentiment analysis) while preserving the fine-tuning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09147v3">LLM-as-a-qualitative-judge: automating error analysis in natural language generation</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Prompting large language models (LLMs) to evaluate generated text, known as LLM-as-a-judge, has become a standard evaluation approach in natural language generation (NLG), but is primarily used as a quantitative tool, i.e. with numerical scores as main outputs. In this work, we propose LLM-as-a-qualitative-judge, an LLM-based evaluation approach with the main output being a structured report of common issue types in the NLG system outputs. Our approach is targeted at providing developers with meaningful insights on what improvements can be done to a given NLG system and consists of two main steps, namely open-ended per-instance issue analysis and clustering of the discovered issues using an intuitive cumulative algorithm. We also introduce a strategy for evaluating the proposed approach, coupled with ~300 annotations of issues in instances from 12 NLG datasets. Our results show that instance-specific issues output by LLM-as-a-qualitative-judge match those annotated by humans in 2/3 cases, and that LLM-as-a-qualitative-judge is capable of producing error type reports resembling the reports composed by human annotators. We also demonstrate in a case study how the use of LLM-as-a-qualitative-judge can substantially improve NLG systems performance. Our code and data are publicly available at https://github.com/tunde-ajayi/llm-as-a-qualitative-judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00075v3">Theoretical Modeling of LLM Self-Improvement Training Dynamics Through Solver-Verifier Gap</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      Self-improvement is among the most prominent techniques within the realm of large language models (LLM), aiming to enhance the LLM performance without relying on external data. Despite its significance, generally how LLM performances evolve during the self-improvement process remains underexplored. In this paper, we theoretically model the training dynamics of self-improvement via the concept of solver-verifier gap. This is inspired by the conjecture that the performance enhancement of self-improvement stems from the gap between LLM's solver capability and verifier capability. Based on the theoretical framework, we further show how to model the entire training trajectory. This framework allows quantifying the capability limit of self-improvement by fitting the theoretical model to the experiment results. We empirically validate the effectiveness of the theoretical framework on various LLMs and datasets. Beyond self-improvement, we extend our analysis to investigate how external data influences these dynamics within the framework. Notably, we find that under limited external data regimes, such external data can be utilized at any stage without significantly affecting final performances, which accords with the empirical observations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09738v1">Judge's Verdict: A Comprehensive Analysis of LLM Judge Capability Through Human Agreement</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 10 pages, 1 figure, 4 tables, under review as a conference paper at ICLR 2026
    </div>
    <details class="paper-abstract">
      This research introduces the Judge's Verdict Benchmark, a novel two-step methodology to evaluate Large Language Models (LLMs) as judges for response accuracy evaluation tasks. We assess how well 54 LLMs can replicate human judgment when scoring responses from RAG (Retrieval-Augmented Generation) or Agentic pipelines against ground truth answers. Our methodology progresses from traditional correlation analysis to comprehensive Cohen's Kappa analysis that measures actual agreement patterns. The two-step approach includes: (1) a correlation test that filters judges with strong alignment, followed by (2) a human-likeness test using z-scores to identify two distinct judgment patterns: human-like judgment (|z| < 1) that mimics natural human variation, and super-consistent judgment (z > 1) that exceeds typical human-to-human agreement levels. This methodology reveals that 27 out of 54 tested LLMs achieve Tier 1 performance: 23 models exhibit human-like patterns that preserve the nuances of human judgment, while 4 models demonstrate super-consistent behavior, a pattern that could indicate either enhanced reliability or oversimplification of complex judgments. Testing 43 open-source models (1B-405B parameters) and 11 closed models (GPT, Gemini, Claude variants), we demonstrate that judge excellence is not solely dependent on model size but on specific training strategies. Our key contributions include: (1) establishing that correlation alone is insufficient for judge evaluation, (2) introducing a "Turing Test for judges" based on agreement patterns, and (3) providing a standardized benchmark for classifying LLM judges into distinct performance tiers for different evaluation needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09732v1">Evaluating LLM-Based Process Explanations under Progressive Behavioral-Input Reduction</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 12 pages, 2 figures, 3 tables; to appear in Enterprise Design, Operations, and Computing. EDOC 2025 Workshops, Lecture Notes in Business Information Processing (LNBIP), Springer, 2025. Part of 29th International Conference on Enterprise Design, Operations, and Computing (EDOC)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate textual explanations of process models discovered from event logs. Producing explanations from large behavioral abstractions (e.g., directly-follows graphs or Petri nets) can be computationally expensive. This paper reports an exploratory evaluation of explanation quality under progressive behavioral-input reduction, where models are discovered from progressively smaller prefixes of a fixed log. Our pipeline (i) discovers models at multiple input sizes, (ii) prompts an LLM to generate explanations, and (iii) uses a second LLM to assess completeness, bottleneck identification, and suggested improvements. On synthetic logs, explanation quality is largely preserved under moderate reduction, indicating a practical cost-quality trade-off. The study is exploratory, as the scores are LLM-based (comparative signals rather than ground truth) and the data are synthetic. The results suggest a path toward more computationally efficient, LLM-assisted process analysis in resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09722v1">Layout-Aware Parsing Meets Efficient LLMs: A Unified, Scalable Framework for Resume Information Extraction and Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Automated resume information extraction is critical for scaling talent acquisition, yet its real-world deployment faces three major challenges: the extreme heterogeneity of resume layouts and content, the high cost and latency of large language models (LLMs), and the lack of standardized datasets and evaluation tools. In this work, we present a layout-aware and efficiency-optimized framework for automated extraction and evaluation that addresses all three challenges. Our system combines a fine-tuned layout parser to normalize diverse document formats, an inference-efficient LLM extractor based on parallel prompting and instruction tuning, and a robust two-stage automated evaluation framework supported by new benchmark datasets. Extensive experiments show that our framework significantly outperforms strong baselines in both accuracy and efficiency. In particular, we demonstrate that a fine-tuned compact 0.6B LLM achieves top-tier accuracy while significantly reducing inference latency and computational cost. The system is fully deployed in Alibaba's intelligent HR platform, supporting real-time applications across its business units.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09721v1">A Comprehensive Survey on Benchmarks and Solutions in Software Engineering of LLM-Empowered Agentic System</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 21 pages
    </div>
    <details class="paper-abstract">
      The integration of LLMs into software engineering has catalyzed a paradigm shift from traditional rule-based systems to sophisticated agentic systems capable of autonomous problem-solving. Despite this transformation, the field lacks a comprehensive understanding of how benchmarks and solutions interconnect, hindering systematic progress and evaluation. This survey presents the first holistic analysis of LLM-empowered software engineering, bridging the critical gap between evaluation and solution approaches. We analyze 150+ recent papers and organize them into a comprehensive taxonomy spanning two major dimensions: (1) Solutions, categorized into prompt-based, fine-tuning-based, and agent-based paradigms, and (2) Benchmarks, covering code generation, translation, repair, and other tasks. Our analysis reveals how the field has evolved from simple prompt engineering to complex agentic systems incorporating planning and decomposition, reasoning and self-refinement, memory mechanisms, and tool augmentation. We present a unified pipeline that illustrates the complete workflow from task specification to final deliverables, demonstrating how different solution paradigms address varying complexity levels across software engineering tasks. Unlike existing surveys that focus on isolated aspects, we provide full-spectrum coverage connecting 50+ benchmarks with their corresponding solution strategies, enabling researchers to identify optimal approaches for specific evaluation criteria. Furthermore, we identify critical research gaps and propose actionable future directions, including multi-agent collaboration frameworks, self-evolving code generation systems, and integration of formal verification with LLM-based methods. This survey serves as a foundational resource for researchers and practitioners seeking to understand, evaluate, and advance LLM-empowered software engineering systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09720v1">Preference-Aware Memory Update for Long-Term LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      One of the key factors influencing the reasoning capabilities of LLM-based agents is their ability to leverage long-term memory. Integrating long-term memory mechanisms allows agents to make informed decisions grounded in historical interactions. While recent advances have significantly improved the storage and retrieval components, by encoding memory into dense vectors for similarity search or organizing memory as structured knowledge graphs most existing approaches fall short in memory updating. In particular, they lack mechanisms for dynamically refining preference memory representations in response to evolving user behaviors and contexts. To address this gap, we propose a Preference-Aware Memory Update Mechanism (PAMU) that enables dynamic and personalized memory refinement. By integrating sliding window averages (SW) with exponential moving averages (EMA), PAMU constructs a fused preference-aware representation that captures both short-term fluctuations and long-term user tendencies. We conduct experiments on five task scenarios of the LoCoMo dataset, and the results show that our mechanism can significantly improve the output quality of LLM in five baselines, validating its effectiveness in long-term conversations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09719v1">ICL-Router: In-Context Learned Model Representations for LLM Routing</a></div>
    <div class="paper-meta">
      📅 2025-10-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often exhibit complementary strengths. Model routing harnesses these strengths by dynamically directing each query to the most suitable model, given a candidate model pool. However, routing performance relies on accurate model representations, and adding new models typically requires retraining, limiting scalability. To address these challenges, we propose a novel routing method using in-context vectors to represent model capabilities. The method proceeds in two stages. First, queries are embedded and projected into vectors, with a projector and LLM-based router trained to reconstruct the original queries, aligning vector representations with the router's semantic space. Second, each candidate model is profiled on a query set, and the router learns -- based on in-context vectors of query and model performance -- to predict whether each model can correctly answer new queries. Extensive experiments demonstrate that our method achieves state-of-the-art routing performance in both in-distribution and out-of-distribution tasks. Moreover, our method allows for seamless integration of new models without retraining the router. The code is available at https://github.com/lalalamdbf/ICL-Router.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.09599v1">Prompting Test-Time Scaling Is A Strong LLM Reasoning Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 Our code and data are available at https://github.com/VILA-Lab/PTTS
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive reasoning capabilities when provided with chain-of-thought exemplars, but curating large reasoning datasets remains laborious and resource-intensive. In this work, we introduce Prompting Test-Time Scaling (P-TTS), a simple yet effective inference-time data augmentation strategy for enhancing LLM reasoning through finetuning. Rather than collecting thousands or even millions of examples, P-TTS leverages a small pool of only 90 manually selected reasoning instances and systematically varies exemplar augmentation through principled instruction prompting intensities at test time to synthesize diverse reasoning trajectory contexts. Then we finetune the various sizes of Qwen-2.5 models on P-TTS data. Across a suite of mathematical reasoning AIME2024 & 25, MATH500, and GPQA-Diamond, our P-TTS-7B and 32B models outperform the prior competitive baselines like S1 and S1.1 (1K-shot), achieving absolute accuracy gains of +26.66% and +30.00% on AIME'24 (7B), and +13.34% and +6.67% on AIME'25 (7B); P-TTS-32B yields gains of +23.33% and +16.63% on AIME'24, and +26.63% and +3.33% on AIME'25 (vs. S1 and S1.1, respectively), with comparable or better performance on MATH500 and GPQA-Diamond. We further show that P-TTS enhances zero-shot generalization accuracy on out-of-domain reasoning benchmarks of Gaokao, Kaoyan, OlympiadBench, AMC23, GradeSchoolMath, and Minerva. Our analysis suggests that test-time scaling effectively explores the latent space of reasoning patterns, amplifying LLM problem-solving with minimal annotation overhead, and further unlocking the reasoning potential and capabilities of LLMs. Prompting Test-Time Scaling offers a practical, low-cost way to elicit LLM reasoning in resource-constrained or rapidly evolving domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01171v3">Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity</a></div>
    <div class="paper-meta">
      📅 2025-10-10
      | 💬 82 pages, 28 figures, 32 tables. Code is available at https://github.com/CHATS-lab/verbalize-sampling
    </div>
    <details class="paper-abstract">
      Post-training alignment often reduces LLM diversity, leading to a phenomenon known as mode collapse. Unlike prior work that attributes this effect to algorithmic limitations, we identify a fundamental, pervasive data-level driver: typicality bias in preference data, whereby annotators systematically favor familiar text as a result of well-established findings in cognitive psychology. We formalize this bias theoretically, verify it on preference datasets empirically, and show that it plays a central role in mode collapse. Motivated by this analysis, we introduce Verbalized Sampling, a simple, training-free prompting strategy to circumvent mode collapse. VS prompts the model to verbalize a probability distribution over a set of responses (e.g., "Generate 5 jokes about coffee and their corresponding probabilities"). Comprehensive experiments show that VS significantly improves performance across creative writing (poems, stories, jokes), dialogue simulation, open-ended QA, and synthetic data generation, without sacrificing factual accuracy and safety. For instance, in creative writing, VS increases diversity by 1.6-2.1x over direct prompting. We further observe an emergent trend that more capable models benefit more from VS. In sum, our work provides a new data-centric perspective on mode collapse and a practical inference-time remedy that helps unlock pre-trained generative diversity.
    </details>
</div>
