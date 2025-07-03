# llm - 2025_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
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
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.24118v1">Scaling Human Judgment in Community Notes with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      This paper argues for a new paradigm for Community Notes in the LLM era: an open ecosystem where both humans and LLMs can write notes, and the decision of which notes are helpful enough to show remains in the hands of humans. This approach can accelerate the delivery of notes, while maintaining trust and legitimacy through Community Notes' foundational principle: A community of diverse human raters collectively serve as the ultimate evaluator and arbiter of what is helpful. Further, the feedback from this diverse community can be used to improve LLMs' ability to produce accurate, unbiased, broadly helpful notes--what we term Reinforcement Learning from Community Feedback (RLCF). This becomes a two-way street: LLMs serve as an asset to humans--helping deliver context quickly and with minimal effort--while human feedback, in turn, enhances the performance of LLMs. This paper describes how such a system can work, its benefits, key new risks and challenges it introduces, and a research agenda to solve those challenges and realize the potential of this approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.02113v2">Trust & Safety of LLMs and LLMs in Trust & Safety</a></div>
    <div class="paper-meta">
      📅 2025-06-30
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have garnered considerable attention for their remarkable abilities in natural language processing tasks. However, their widespread adoption has raised concerns pertaining to trust and safety. This systematic review investigates the current research landscape on trust and safety in LLMs, with a particular focus on the novel application of LLMs within the field of Trust and Safety itself. We delve into the complexities of utilizing LLMs in domains where maintaining trust and safety is paramount, offering a consolidated perspective on this emerging trend.\ By synthesizing findings from various studies, we identify key challenges and potential solutions, aiming to benefit researchers and practitioners seeking to understand the nuanced interplay between LLMs and Trust and Safety. This review provides insights on best practices for using LLMs in Trust and Safety, and explores emerging risks such as prompt injection and jailbreak attacks. Ultimately, this study contributes to a deeper understanding of how LLMs can be effectively and responsibly utilized to enhance trust and safety in the digital realm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18797v2">SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-06-30
      | 💬 Accepted to ACL'25
    </div>
    <details class="paper-abstract">
      Recent advancements in LLMs unlearning have shown remarkable success in removing unwanted data-model influences while preserving the model's utility for legitimate knowledge. Despite these strides, sparse Mixture-of-Experts (MoE) LLMs--a key subset of the LLM family--have remained unexplored in the context of unlearning. As MoE LLMs are celebrated for their exceptional performance, we ask:How can unlearning be performed effectively and efficiently on MoE LLMs? Our pilot study shows that the dynamic routing nature of MoE LLMs introduces unique challenges, leading to excessive forgetting, uncontrolled knowledge erasure and substantial utility drops when existing unlearning methods are applied. To address this, we propose a novel Selected-Expert Unlearning Framework (SEUF). Through expert attribution, unlearning is concentrated on the most actively engaged experts for the specified knowledge. Concurrently, an anchor loss is applied to the router to stabilize the active state of this targeted expert, ensuring focused and controlled unlearning. SEUF is compatible with various standard unlearning algorithms. Extensive experiments demonstrate that SEUF enhances both forget quality up to 5% and model utility by 35% on MoE LLMs across various benchmarks and LLM architectures (compared to standard unlearning algorithms), while only unlearning 0.06% of the model parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.24045v1">Agent.xpu: Efficient Scheduling of Agentic LLM Workloads on Heterogeneous SoC</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      The proliferation of agentic Large Language Models (LLMs) on personal devices introduces a new class of workloads characterized by a dichotomy of objectives. Reactive tasks, initiated by users, demand immediate, low-latency responses, while proactive tasks operate invisibly and prioritize throughput. Existing on-device LLM engines, designed for isolated inferences, fail to efficiently manage these concurrent and conflicting requests on consumer-grade heterogeneous SoCs with CPU, integrated GPU, and NPU. This paper introduces Agent.xpu, an efficient serving system for agentic LLM workloads on memory-unified heterogeneous SoCs. With dedicated offline profiling, Agent.xpu first constructs a heterogeneous execution graph, which fuses and chunks model kernels for affinity-guided, elastic accelerator mapping with predictive kernel annotation. At runtime, its online scheduler enables fine-grained, kernel-level preemption to guarantee the responsiveness of reactive tasks. To maximize SoC utilization, it adopts slack-aware kernel backfill to opportunistically append proactive tasks, and mitigates NPU-iGPU contention via bandwidth-aware dispatch. Evaluation on an Intel Core Ultra SoC shows that Agent.xpu achieves 4.6$\times$ lower latency for reactive tasks and sustains 1.6$\times$-6.8$\times$ higher throughput for proactive tasks compared to state-of-the-art inference engines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.24015v1">Bug Fixing with Broader Context: Enhancing LLM-Based Program Repair via Layered Knowledge Injection</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      Prompting LLMs with bug-related context (e.g., error messages, stack traces) improves automated program repair, but many bugs still remain unresolved. In real-world projects, developers often rely on broader repository and project-level context beyond the local code to resolve such bugs. In this paper, we investigate how automatically extracting and providing such knowledge can improve LLM-based program repair. We propose a layered knowledge injection framework that incrementally augments LLMs with structured context. It starts with the Bug Knowledge Layer, which includes information such as the buggy function and failing tests; expands to the Repository Knowledge Layer, which adds structural dependencies, related files, and commit history; and finally injects the Project Knowledge Layer, which incorporates relevant details from documentation and previously fixed bugs. We evaluate this framework on a dataset of 314 bugs from BugsInPy using two LLMs (Llama 3.3 and GPT-4o-mini), and analyze fix rates across six bug types. By progressively injecting knowledge across layers, our approach achieves a fix rate of 79% (250/314) using Llama 3.3, a significant improvement of 23% over previous work. All bug types show improvement with the addition of repository-level context, while only a subset benefit further from project-level knowledge, highlighting that different bug types require different levels of contextual information for effective repair. We also analyze the remaining unresolved bugs and find that more complex and structurally isolated bugs, such as Program Anomaly and GUI bugs, remain difficult even after injecting all available information. Our results show that layered context injection improves program repair and suggest the need for interactive and adaptive APR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23978v1">LLM Agents Are the Antidote to Walled Gardens</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      While the Internet's core infrastructure was designed to be open and universal, today's application layer is dominated by closed, proprietary platforms. Open and interoperable APIs require significant investment, and market leaders have little incentive to enable data exchange that could erode their user lock-in. We argue that LLM-based agents fundamentally disrupt this status quo. Agents can automatically translate between data formats and interact with interfaces designed for humans: this makes interoperability dramatically cheaper and effectively unavoidable. We name this shift universal interoperability: the ability for any two digital services to exchange data seamlessly using AI-mediated adapters. Universal interoperability undermines monopolistic behaviours and promotes data portability. However, it can also lead to new security risks and technical debt. Our position is that the ML community should embrace this development while building the appropriate frameworks to mitigate the downsides. By acting now, we can harness AI to restore user freedom and competitive markets without sacrificing security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23951v1">Unveiling Decision-Making in LLMs for Text Classification : Extraction of influential and interpretable concepts with Sparse Autoencoders</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      Sparse Autoencoders (SAEs) have been successfully used to probe Large Language Models (LLMs) and extract interpretable concepts from their internal representations. These concepts are linear combinations of neuron activations that correspond to human-interpretable features. In this paper, we investigate the effectiveness of SAE-based explainability approaches for sentence classification, a domain where such methods have not been extensively explored. We present a novel SAE-based architecture tailored for text classification, leveraging a specialized classifier head and incorporating an activation rate sparsity loss. We benchmark this architecture against established methods such as ConceptShap, Independent Component Analysis, and other SAE-based concept extraction techniques. Our evaluation covers two classification benchmarks and four fine-tuned LLMs from the Pythia family. We further enrich our analysis with two novel metrics for measuring the precision of concept-based explanations, using an external sentence encoder. Our empirical results show that our architecture improves both the causality and interpretability of the extracted features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23924v1">Performance of LLMs on Stochastic Modeling Operations Research Problems: From Theory to Practice</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited expert-level capabilities across various domains. However, their abilities to solve problems in Operations Research (OR) -- the analysis and optimization of mathematical models derived from real-world problems or their verbal descriptions -- remain underexplored. In this work, we take a first step toward evaluating LLMs' abilities to solve stochastic modeling problems, a core class of OR problems characterized by uncertainty and typically involving tools from probability, statistics, and stochastic processes. We manually procure a representative set of graduate-level homework and doctoral qualification-exam problems and test LLMs' abilities to solve them. We further leverage SimOpt, an open-source library of simulation-optimization problems and solvers, to investigate LLMs' abilities to make real-world decisions under uncertainty. Our results show that, though a nontrivial amount of work is still needed to reliably automate the stochastic modeling pipeline in reality, state-of-the-art LLMs demonstrate proficiency on par with human experts in both classroom and practical settings. These findings highlight the potential of building AI agents that assist OR researchers and amplify the real-world impact of OR through automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.07160v2">GeometryZero: Improving Geometry Solving for LLM with Group Contrastive Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated remarkable capabilities across diverse domains, particularly in mathematical reasoning, amid which geometry problem solving remains a challenging area where auxiliary construction plays a enssential role. Existing approaches either achieve suboptimal performance or rely on massive LLMs (e.g., GPT-4o), incurring massive computational costs. We posit that reinforcement learning with verifiable reward (e.g., GRPO) offers a promising direction for training smaller models that effectively combine auxiliary construction with robust geometric reasoning. However, directly applying GRPO to geometric reasoning presents fundamental limitations due to its dependence on unconditional rewards, which leads to indiscriminate and counterproductive auxiliary constructions. To address these challenges, we propose Group Contrastive Policy Optimization (GCPO), a novel reinforcement learning framework featuring two key innovations: (1) Group Contrastive Masking, which adaptively provides positive or negative reward signals for auxiliary construction based on contextual utility, and a (2) length reward that promotes longer reasoning chains. Building on GCPO, we develop GeometryZero, a family of affordable-size geometric reasoning models that judiciously determine when to employ auxiliary construction. Our extensive empirical evaluation across popular geometric benchmarks (Geometry3K, MathVista) demonstrates that GeometryZero models consistently outperform baselines (e.g. GRPO), achieving an average improvement of 4.29% across all benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23774v1">Leveraging a Multi-Agent LLM-Based System to Educate Teachers in Hate Incidents Management</a></div>
    <div class="paper-meta">
      📅 2025-06-30
      | 💬 8 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Computer-aided teacher training is a state-of-the-art method designed to enhance teachers' professional skills effectively while minimising concerns related to costs, time constraints, and geographical limitations. We investigate the potential of large language models (LLMs) in teacher education, using a case of teaching hate incidents management in schools. To this end, we create a multi-agent LLM-based system that mimics realistic situations of hate, using a combination of retrieval-augmented prompting and persona modelling. It is designed to identify and analyse hate speech patterns, predict potential escalation, and propose effective intervention strategies. By integrating persona modelling with agentic LLMs, we create contextually diverse simulations of hate incidents, mimicking real-life situations. The system allows teachers to analyse and understand the dynamics of hate incidents in a safe and controlled environment, providing valuable insights and practical knowledge to manage such situations confidently in real life. Our pilot evaluation demonstrates teachers' enhanced understanding of the nature of annotator disagreements and the role of context in hate speech interpretation, leading to the development of more informed and effective strategies for addressing hate in classroom settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23749v1">A Survey of LLM-based Automated Program Repair: Taxonomies, Design Paradigms, and Applications</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are reshaping automated program repair (APR). We categorize the recent 63 LLM-based APR systems published from January 2022 to June 2025 into four paradigms, and show how retrieval- or analysis-augmented contexts strengthen any of them. This taxonomy clarifies key trade-offs: fine-tuning delivers strong task alignment at high training cost; prompting enables rapid deployment but is limited by prompt design and context windows; procedural pipelines offer reproducible control with moderate overhead; agentic frameworks tackle multi-hunk or cross-file bugs at the price of increased latency and complexity. Persistent challenges include verifying semantic correctness beyond test suites, repairing repository-scale defects, and lowering the costs of LLMs. We outline research directions that combine lightweight human feedback, repository-aware retrieval, code analysis, and cost-aware planning to advance reliable and efficient LLM-based APR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23735v1">AutoEvoEval: An Automated Framework for Evolving Close-Ended LLM Evaluation Data</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable performance on various tasks, but existing evaluation benchmarks are often static and insufficient to fully assess their robustness and generalization in realistic scenarios. Prior work using evolutionary or adversarial data augmentation has improved evaluation diversity but lacks systematic control over perturbation types and multi-step complexity, limiting comprehensive robustness analysis. To address these gaps, we propose AutoEvoEval, an evolution-based evaluation framework for close-ended tasks such as multi-choice question answering. AutoEvoEval introduces 22 interpretable atomic evolution operations and supports multi-round compositions, enabling controlled generation of diverse, challenging, and realistic test samples. We conduct extensive experiments addressing four research questions on a broad set of open- and closed-source LLMs. Our results show that atomic operations cause an average accuracy drop of 7.283\%, with structure-disrupting or misleading semantic edits causing the largest declines. Model sensitivities vary significantly for the same perturbation, and combining multiple evolution steps amplifies adversarial effects by up to 52.932\%. These findings suggest current benchmarks may overestimate true model generalization and emphasize the need for evolution-aware robustness evaluation. Code and resources are available at: https://github.com/SYSUSELab/AutoEvoEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23644v1">QLPro: Automated Code Vulnerability Discovery via LLM and Static Code Analysis Integration</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      We introduce QLPro, a vulnerability detection framework that systematically integrates LLMs and static analysis tools to enable comprehensive vulnerability detection across entire open-source projects.We constructed a new dataset, JavaTest, comprising 10 open-source projects from GitHub with 62 confirmed vulnerabilities. CodeQL, a state-of-the-art static analysis tool, detected only 24 of these vulnerabilities while QLPro detected 41. Furthermore, QLPro discovered 6 previously unknown vulnerabilities, 2 of which have been confirmed as 0-days.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23635v1">Towards Building Private LLMs: Exploring Multi-Node Expert Parallelism on Apple Silicon for Mixture-of-Experts Large Language Model</a></div>
    <div class="paper-meta">
      📅 2025-06-30
      | 💬 International Conference on Research in Adaptive and Convergent Systems (RACS '24), November 5--8, 2024, Pompei, Italy
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized Artificial Intelligence (AI) with significant advancements such as OpenAI's ChatGPT, Meta's Llama, and Databricks' DBRX. This paper addresses the cost and scalability challenges encountered when constructing private LLM systems for personal or small group services, as aimed by Apple Intelligence. A Mac Studio cluster with Apple's M2 Ultra chips is established as a cost-efficient solution to host and accelerate the pretrained DBRX model with the Mixture-of-Experts (MoE) architecture. Our performance analysis reveal that parallel execution of the model's experts across two to four machine nodes significantly reduces inference time. We find that computation time for the experts is comparable to the communication time for exchanging their outputs, emphasizing the importance of network latency over bandwidth. We also observe significant management overhead due to Apple software stack's memory management logic. Based on these findings, we develop optimization schemes to eliminate the memory management overhead. As a result, the Mac Studio cluster is 1.15 times more cost-efficient than the state-of-the-art AI supercomputer with NVIDIA H100 GPUs. In addition, we construct a performance model to estimate system performance under varying configurations, and the model provides valuable insights for designing private LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23610v1">Evaluating the Simulation of Human Personality-Driven Susceptibility to Misinformation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-30
      | 💬 pre-print version - paper actually under submission
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) make it possible to generate synthetic behavioural data at scale, offering an ethical and low-cost alternative to human experiments. Whether such data can faithfully capture psychological differences driven by personality traits, however, remains an open question. We evaluate the capacity of LLM agents, conditioned on Big-Five profiles, to reproduce personality-based variation in susceptibility to misinformation, focusing on news discernment, the ability to judge true headlines as true and false headlines as false. Leveraging published datasets in which human participants with known personality profiles rated headline accuracy, we create matching LLM agents and compare their responses to the original human patterns. Certain trait-misinformation associations, notably those involving Agreeableness and Conscientiousness, are reliably replicated, whereas others diverge, revealing systematic biases in how LLMs internalize and express personality. The results underscore both the promise and the limits of personality-aligned LLMs for behavioral simulation, and offer new insight into modeling cognitive diversity in artificial agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17728v3">KAG-Thinker: Interactive Thinking and Deep Reasoning in LLMs via Knowledge-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      In this paper, we introduce KAG-Thinker, which upgrade KAG to a multi-turn interactive thinking and deep reasoning framework powered by a dedicated parameter-light large language model (LLM). Our approach constructs a structured thinking process for solving complex problems, enhancing the the logical coherence and contextual consistency of the reasoning process in question-answering (Q&A) tasks on domain-specific knowledge bases (KBs) within LLMs. Following the \textbf{Logical Form} guided retrieval and reasoning technology route of KAG, this framework first decomposes complex questions into independently solvable sub-problems (which are also referred to as logical forms) through \textbf{breadth decomposition}. Each such logical form is represented in two equivalent forms-natural language and logical function-and subsequently classified as either a Knowledge Retrieval or Reasoning Analysis task. Dependencies and parameter passing between these tasks are explicitly modeled via logical function interfaces. In the solving process, the Retrieval function performs retrieval tasks. It retrieves one-hop structured and unstructured information of specified knowledge unit. While the Math and Deduce functions are used to perform reasoning analysis tasks. Secondly, it is worth noting that, in the Knowledge Retrieval sub-problem tasks, LLMs and external knowledge sources are regarded as equivalent KBs. We use the \textbf{knowledge boundary} module to determine the optimal source using self-regulatory mechanisms such as confidence calibration and reflective reasoning, and use the \textbf{depth solving} module to enhance the comprehensiveness of knowledge acquisition...
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02236v2">VQ-LLM: High-performance Code Generation for Vector Quantization Augmented LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      In this work, we design and implement VQ-LLM, an efficient fused Vector Quantization (VQ) kernel generation framework. We first introduce a software abstraction called codebook cache to optimize codebook access efficiency and support the integration of VQ with various computations. The codebook cache adaptively stores different entries across the GPU's memory hierarchy, including off-chip global memory, on-chip shared memory, and registers. Centered around the codebook cache, we design an efficient computation engine that optimizes memory traffic during computations involving codebooks. This compute engine adopts the codebook-centric dataflow and fusion optimizations. Additionally, we provide adaptive heuristics to tailor parameter selection in our optimizations to diverse VQ configurations. Our optimizations achieve an average latency reduction of 46.13% compared to unoptimized versions. Compared to existing open-source implementations, our methods decrease latency by 64.36% to 99.1%. A final comparison with state-of-the-art element-wise quantization methods like AWQ and KVQuant shows that our VQ-LLM is practically viable, achieving latencies close or even better latencies to those at equivalent bit-widths, potentially offering greater accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23535v1">Comparative Analysis of the Code Generated by Popular Large Language Models (LLMs) for MISRA C++ Compliance</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      Safety-critical systems are engineered systems whose failure or malfunction could result in catastrophic consequences. The software development for safety-critical systems necessitates rigorous engineering practices and adherence to certification standards like DO-178C for avionics. DO-178C is a guidance document which requires compliance to well-defined software coding standards like MISRA C++ to enforce coding guidelines that prevent the use of ambiguous, unsafe, or undefined constructs. Large Language Models (LLMs) have demonstrated significant capabilities in automatic code generation across a wide range of programming languages, including C++. Despite their impressive performance, code generated by LLMs in safety-critical domains must be carefully analyzed for conformance to MISRA C++ coding standards. In this paper, I have conducted a comparative analysis of the C++ code generated by popular LLMs including: OpenAI ChatGPT, Google Gemini, DeepSeek, Meta AI, and Microsoft Copilot for compliance with MISRA C++.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02922v2">RetroInfer: A Vector-Storage Approach for Scalable Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-06-30
      | 💬 17 pages
    </div>
    <details class="paper-abstract">
      The growing context lengths of large language models (LLMs) pose significant challenges for efficient inference, primarily due to GPU memory and bandwidth constraints. We present RetroInfer, a novel system that reconceptualizes the key-value (KV) cache as a vector storage system which exploits the inherent attention sparsity to accelerate long-context LLM inference. At its core is the wave index, an Attention-aWare VEctor index that enables efficient and accurate retrieval of critical tokens through techniques such as tripartite attention approximation, accuracy-bounded attention estimation, and segmented clustering. Complementing this is the wave buffer, which coordinates KV cache placement and overlaps computation and data transfer across GPU and CPU to sustain high throughput. Unlike prior sparsity-based methods that struggle with token selection and hardware coordination, RetroInfer delivers robust performance without compromising model accuracy. Experiments on long-context benchmarks show up to 4.5X speedup over full attention within GPU memory limits and up to 10.5X over sparse attention baselines when KV cache is extended to CPU memory, all while preserving full-attention-level accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01705v3">Progressive Binarization with Semi-Structured Pruning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable progress in natural language processing, but their high computational and memory costs hinder deployment on resource-constrained devices. Binarization, which reduces model weights to 1 bit, is a promising solution for efficient inference. However, binarized LLMs still exhibit redundancy that can be further compressed. Semi-structured pruning offers a favorable trade-off between model performance and hardware efficiency, but naively combining it with binarization often leads to severe performance degradation. To address this, we propose Progressive Binarization with Semi-Structured Pruning (PBS$^2$P), a novel post-training compression framework. We propose Stepwise semi-structured Pruning with Binarization Optimization (SPBO) to jointly reduce pruning and binarization error. Additionally, we develop a Coarse-to-Fine Search (CFS) strategy to more effectively select pruning elements. Extensive experiments across multiple LLM families show that PBS$^2$P consistently outperforms state-of-the-art binary post-training quantization methods in both perplexity and downstream accuracy. The code and models will be available at: https://github.com/XIANGLONGYAN/PBS2P.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23520v1">ChemActor: Enhancing Automated Extraction of Chemical Synthesis Actions with LLM-Generated Data</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      With the increasing interest in robotic synthesis in the context of organic chemistry, the automated extraction of chemical procedures from literature is critical. However, this task remains challenging due to the inherent ambiguity of chemical language and the high cost of human annotation required for developing reliable computer-aided extraction protocols. Here, we present ChemActor, a fully fine-tuned large language model (LLM), as a chemical executor to convert between unstructured experimental procedures and structured action sequences. We propose a sequential LLM-generated data framework to address the challenges of insufficient and low-quality annotated data. This framework integrates a data selection module that selects data based on distribution divergence, with a general-purpose LLM, to generate machine-executable actions from a single molecule input. Additionally, we introduce a novel multi-round LLMs circle review metric, which reflects the model's advanced understanding of chemical experimental procedures. Extensive experiments on reaction-to-description (R2D) and description-to-action (D2A) tasks demonstrate that ChemActor, augmented by LLM-generated data, achieves state-of-the-art performance, outperforming the baseline model by 10%. The code is available at: https://github.com/Zhanghahah/ChemActor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05352v3">Achieving binary weight and activation for LLMs using Post-Training Quantization</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      Quantizing large language models (LLMs) to 1-bit precision significantly reduces computational costs, but existing quantization techniques suffer from noticeable performance degradation when using weight and activation precisions below 4 bits (W4A4). In this paper, we propose a post-training quantization framework with W(1+1)A(1*4) configuration, where weights are quantized to 1 bit with an additional 1 bit for fine-grain grouping and activations are quantized to 1 bit with a 4-fold increase in the number of channels. For weight quantization, we propose utilizing Hessian-aware fine-grained grouping along with an EM-based quantization scheme. For activation quantization, we decompose INT4-quantized activations into a 4 * INT1 format equivalently and simultaneously smooth the scaling factors based on quantization errors, which further reduces the quantization errors in activations. Our method surpasses state-of-the-art (SOTA) LLM quantization baselines on W2A4 across multiple tasks, pushing the boundaries of existing LLM quantization methods toward fully binarized models. Code is available at https://github.com/JimmyCrave/LLM-PTQ-binarization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16334v2">LLM Braces: Straightening Out LLM Predictions with Relevant Sub-Updates</a></div>
    <div class="paper-meta">
      📅 2025-06-30
      | 💬 ACL 2025, 16 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Recent findings reveal that much of the knowledge in a Transformer-based Large Language Model (LLM) is encoded in its feed-forward (FFN) layers, where each FNN layer can be interpreted as the summation of sub-updates, each corresponding to a weighted column vector from the FFN's value parameter matrix that often encodes human-interpretable concepts. In light of this, we hypothesize that model performance and behaviors can be further enhanced and controlled by modulating the contributions of these sub-updates based on their relevance to the input or target output style, and propose LLMBRACES, a novel and efficient method that computes relevance scores associated with value vectors in FFN layers and leverages these scores to dynamically adjust the contribution of sub-updates. By optimizing sub-update contributions, LLMBRACES refines the prediction process, leading to more accurate and reliable outputs, much like a 'brace' providing support and stability. Moreover, LLMBRACES can be extended to support conditional control over generation characteristics, such as sentiment, thereby offering fine-grained steering of LLM outputs. Extensive experiments on various LLMs-including Qwen2.5-1.5B, Llama2-7B, and Llama3-8B-demonstrate that LLMBRACES outperforms baseline approaches in both fine-tuning and zero-shot settings while requiring significantly fewer tunable parameters, up to 75% fewer compared to LoRA. Furthermore, LLMBRACES excels in sentiment-controlled generation and toxicity reduction, highlighting its potential for flexible, controlled text generation across applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23502v1">LLM-enhanced Action-aware Multi-modal Prompt Tuning for Image-Text Matching</a></div>
    <div class="paper-meta">
      📅 2025-06-30
      | 💬 accepted by ICCV 2025
    </div>
    <details class="paper-abstract">
      Driven by large-scale contrastive vision-language pre-trained models such as CLIP, recent advancements in the image-text matching task have achieved remarkable success in representation learning. Due to image-level visual-language alignment, CLIP falls short in understanding fine-grained details such as object attributes and spatial relationships between objects. Recent efforts have attempted to compel CLIP to acquire structured visual representations by introducing prompt learning to achieve object-level alignment. While achieving promising results, they still lack the capability to perceive actions, which are crucial for describing the states or relationships between objects. Therefore, we propose to endow CLIP with fine-grained action-level understanding by introducing an LLM-enhanced action-aware multi-modal prompt-tuning method, incorporating the action-related external knowledge generated by large language models (LLMs). Specifically, we design an action triplet prompt and an action state prompt to exploit compositional semantic knowledge and state-related causal knowledge implicitly stored in LLMs. Subsequently, we propose an adaptive interaction module to aggregate attentive visual features conditioned on action-aware prompted knowledge for establishing discriminative and action-aware visual representations, which further improves the performance. Comprehensive experimental results on two benchmark datasets demonstrate the effectiveness of our method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23485v1">Thought-Augmented Planning for LLM-Powered Interactive Recommender Agent</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      Interactive recommendation is a typical information-seeking task that allows users to interactively express their needs through natural language and obtain personalized recommendations. Large language model-powered (LLM-powered) agents have become a new paradigm in interactive recommendations, effectively capturing users' real-time needs and enhancing personalized experiences. However, due to limited planning and generalization capabilities, existing formulations of LLM-powered interactive recommender agents struggle to effectively address diverse and complex user intents, such as intuitive, unrefined, or occasionally ambiguous requests. To tackle this challenge, we propose a novel thought-augmented interactive recommender agent system (TAIRA) that addresses complex user intents through distilled thought patterns. Specifically, TAIRA is designed as an LLM-powered multi-agent system featuring a manager agent that orchestrates recommendation tasks by decomposing user needs and planning subtasks, with its planning capacity strengthened through Thought Pattern Distillation (TPD), a thought-augmentation method that extracts high-level thoughts from the agent's and human experts' experiences. Moreover, we designed a set of user simulation schemes to generate personalized queries of different difficulties and evaluate the recommendations based on specific datasets. Through comprehensive experiments conducted across multiple datasets, TAIRA exhibits significantly enhanced performance compared to existing methods. Notably, TAIRA shows a greater advantage on more challenging tasks while generalizing effectively on novel tasks, further validating its superiority in managing complex user intents within interactive recommendation systems. The code is publicly available at:https://github.com/Alcein/TAIRA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21807v3">TabReason: A Reinforcement Learning-Enhanced Reasoning LLM for Explainable Tabular Data Prediction</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      Predictive modeling on tabular data is the cornerstone of many real-world applications. Although gradient boosting machines and some recent deep models achieve strong performance on tabular data, they often lack interpretability. On the other hand, large language models (LLMs) have demonstrated powerful capabilities to generate human-like reasoning and explanations, but remain under-performed for tabular data prediction. In this paper, we propose a new approach that leverages reasoning-based LLMs, trained using reinforcement learning, to perform more accurate and explainable predictions on tabular data. Our method introduces custom reward functions that guide the model not only toward better prediction accuracy but also toward human-understandable reasons for its predictions. The proposed method is evaluated on financial benchmark datasets and compared against established LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23462v1">Can We Predict the Unpredictable? Leveraging DisasterNet-LLM for Multimodal Disaster Classification</a></div>
    <div class="paper-meta">
      📅 2025-06-30
      | 💬 Accepted in the 2025 IEEE International Geoscience and Remote Sensing Symposium (IGARSS 2025), scheduled for 3 - 8 August 2025 in Brisbane, Australia
    </div>
    <details class="paper-abstract">
      Effective disaster management requires timely and accurate insights, yet traditional methods struggle to integrate multimodal data such as images, weather records, and textual reports. To address this, we propose DisasterNet-LLM, a specialized Large Language Model (LLM) designed for comprehensive disaster analysis. By leveraging advanced pretraining, cross-modal attention mechanisms, and adaptive transformers, DisasterNet-LLM excels in disaster classification. Experimental results demonstrate its superiority over state-of-the-art models, achieving higher accuracy of 89.5%, an F1 score of 88.0%, AUC of 0.92%, and BERTScore of 0.88% in multimodal disaster classification tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.05175v4">Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity</a></div>
    <div class="paper-meta">
      📅 2025-06-30
      | 💬 Published at ICML 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), renowned for their remarkable performance across diverse domains, present a challenge when it comes to practical deployment due to their colossal model size. In response to this challenge, efforts have been directed toward the application of traditional network pruning techniques to LLMs, uncovering a massive number of parameters that can be pruned in one-shot without hurting performance. Prevailing LLM pruning strategies have consistently adhered to the practice of uniformly pruning all layers at equivalent sparsity, resulting in robust performance. However, this observation stands in contrast to the prevailing trends observed in the field of vision models, where non-uniform layerwise sparsity typically yields stronger results. To understand the underlying reasons for this disparity, we conduct a comprehensive study and discover a strong correlation with the emergence of activation outliers in LLMs. Inspired by this finding, we introduce a novel LLM pruning methodology that incorporates a tailored set of non-uniform layerwise sparsity ratios, termed as Outlier Weighed Layerwise sparsity (OWL). The sparsity ratio of OWL is proportional to the outlier ratio observed within each layer, facilitating a more effective alignment between layerwise weight sparsity and outlier ratios. Our empirical evaluation, conducted across the LLaMA-V1 family and OPT, spanning various benchmarks, demonstrates the distinct advantages offered by OWL over previous methods. For instance, OWL exhibits a remarkable performance gain, surpassing the state-of-the-art Wanda and SparseGPT by 61.22 and 6.80 perplexity at a high sparsity level of 70%, respectively, while delivering 2.6x end-to-end inference speed-up in the DeepSparse inference engine. Codes are available at https://github.com/luuyin/OWL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22419v2">The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      Rapid advancements in large language models (LLMs) have the potential to assist in scientific progress. A critical capability toward this endeavor is the ability to reproduce existing work. To evaluate the ability of AI agents to reproduce results in an active research area, we introduce the Automated LLM Speedrunning Benchmark, leveraging the research community contributions on the NanoGPT speedrun, a competition to train a GPT-2 model in the shortest time. Each of the 19 speedrun tasks provides the agent with the previous records training script, optionally paired with one of three hint formats, ranging from pseudocode to paper-like descriptions of the new records improvements. Records execute quickly by design and speedrun improvements encompass diverse code-level changes, ranging from high-level algorithmic advancements to hardware-aware optimizations. These features make the benchmark both accessible and realistic for the frontier problem of improving LLM training. We find that recent reasoning LLMs combined with SoTA scaffolds struggle to reimplement already-known innovations in our benchmark, even when given detailed hints. Our benchmark thus provides a simple, non-saturated measure of an LLMs ability to automate scientific reproduction, a necessary (but not sufficient) skill for an autonomous research agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.14640v2">Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2025-06-30
      | 💬 Accepted to ACL 2025 (Main Conference)
    </div>
    <details class="paper-abstract">
      Attributed Question Answering (AQA) has attracted wide attention, but there are still several limitations in evaluating the attributions, including lacking fine-grained attribution categories, relying on manual annotations, and failing to compare attributions with only subtle differences. To bridge these gaps, we introduce Complex Attributed Question Answering (CAQA), a large-scale benchmark containing comprehensive attribution categories, automatically generated using Knowledge Graphs (KGs), and complex attribution scenarios. We have conducted extensive experiments to verify the effectiveness of CAQA, including the benchmarking of 25 automatic evaluators, their comparison with human evaluators, the testing of LLM evaluators fine-tuned by CAQA and so on. These experiments also lead to a series of important findings that can benefit the future research of AQA. All the codes and data are publicly accessible at https://github.com/HuuuNan/CAQA-Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17117v3">From Tokens to Thoughts: How LLMs and Humans Trade Compression for Meaning</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      Humans organize knowledge into compact categories through semantic compression by mapping diverse instances to abstract representations while preserving meaning (e.g., robin and blue jay are both birds; most birds can fly). These concepts reflect a trade-off between expressive fidelity and representational simplicity. Large Language Models (LLMs) demonstrate remarkable linguistic abilities, yet whether their internal representations strike a human-like trade-off between compression and semantic fidelity is unclear. We introduce a novel information-theoretic framework, drawing from Rate-Distortion Theory and the Information Bottleneck principle, to quantitatively compare these strategies. Analyzing token embeddings from a diverse suite of LLMs against seminal human categorization benchmarks, we uncover key divergences. While LLMs form broad conceptual categories that align with human judgment, they struggle to capture the fine-grained semantic distinctions crucial for human understanding. More fundamentally, LLMs demonstrate a strong bias towards aggressive statistical compression, whereas human conceptual systems appear to prioritize adaptive nuance and contextual richness, even if this results in lower compressional efficiency by our measures. These findings illuminate critical differences between current AI and human cognitive architectures, guiding pathways toward LLMs with more human-aligned conceptual representations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.06096v2">Free and Fair Hardware: A Pathway to Copyright Infringement-Free Verilog Generation using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-30
      | 💬 Accepted at DAC 2025
    </div>
    <details class="paper-abstract">
      Limitations in Large Language Model (LLM) capabilities for hardware design tasks, such as generating functional Verilog codes, have motivated various fine-tuning optimizations utilizing curated hardware datasets from open-source repositories. However, these datasets remain limited in size and contain minimal checks on licensing for reuse, resulting in potential copyright violations by fine-tuned LLMs. Therefore, we propose an evaluation benchmark to estimate the risk of Verilog-trained LLMs to generate copyright-protected codes. To minimize this risk, we present an open-source Verilog dataset, FreeSet, containing over 220k files, along with the automated dataset curation framework utilized to provide additional guarantees of fair-use Verilog data. We then execute an LLM fine-tuning framework consisting of continual pre-training, resulting in a fine-tuned Llama model for Verilog, FreeV. Our results indicate that FreeV demonstrates the smallest risk of copyright-infringement among prior works, with only a 3% violation rate. Furthermore, experimental results demonstrate improvements in Verilog generation functionality over its baseline model, improving VerilogEval pass@10 rates by over 10%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01141v3">Evaluating Deduplication Techniques for Economic Research Paper Titles with a Focus on Semantic Similarity using NLP and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-30
      | 💬 6 pages, 1 figure
    </div>
    <details class="paper-abstract">
      This study investigates efficient deduplication techniques for a large NLP dataset of economic research paper titles. We explore various pairing methods alongside established distance measures (Levenshtein distance, cosine similarity) and a sBERT model for semantic evaluation. Our findings suggest a potentially low prevalence of duplicates based on the observed semantic similarity across different methods. Further exploration with a human-annotated ground truth set is completed for a more conclusive assessment. The result supports findings from the NLP, LLM based distance metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00214v1">Two-Stage Reasoning-Infused Learning: Improving Classification with LLM-Generated Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-30
    </div>
    <details class="paper-abstract">
      Standard classification models often map inputs directly to labels without explicit reasoning, potentially limiting their performance, robustness, and interpretability. This paper introduces a novel two-stage approach to enhance text classification by leveraging Large Language Model (LLM)-generated reasonings. In the first stage, we fine-tune a Llama-3.2-1B-Instruct model (henceforth Llama-R-Gen) on a general-purpose reasoning dataset (syvai/reasoning-gen) to generate textual reasoning (R) given a question and its answer. In the second stage, this generally trained Llama-R-Gen is used offline to create an augmented training dataset for a downstream generative model. This downstream model, based on Llama-3.2-1B-Instruct, takes only the input text (Q) and is trained to output the generated reasoning (R) immediately followed by the predicted emotion (A). We demonstrate this methodology on the dair-ai/emotion dataset for emotion classification. Our experiments show that the generative model trained to output reasoning and the emotion (Classifier Q->RA) achieves a significant improvement of 8.7 percentage points in accuracy (for emotion prediction) compared to a baseline generative model trained solely to output the emotion (Classifier Q->A), highlighting the strong generalization capabilities of the reasoning generation and the benefit of explicit reasoning training. This work underscores the potential of LLM-generated reasonings for creating richer training datasets, thereby improving the performance of diverse downstream NLP tasks and providing explicit explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00152v1">Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific vs. Non-Scientific Data</a></div>
    <div class="paper-meta">
      📅 2025-06-30
      | 💬 TRL@ACL 2025, camera-ready version
    </div>
    <details class="paper-abstract">
      Tables are among the most widely used tools for representing structured data in research, business, medicine, and education. Although LLMs demonstrate strong performance in downstream tasks, their efficiency in processing tabular data remains underexplored. In this paper, we investigate the effectiveness of both text-based and multimodal LLMs on table understanding tasks through a cross-domain and cross-modality evaluation. Specifically, we compare their performance on tables from scientific vs. non-scientific contexts and examine their robustness on tables represented as images vs. text. Additionally, we conduct an interpretability analysis to measure context usage and input relevance. We also introduce the TableEval benchmark, comprising 3017 tables from scholarly publications, Wikipedia, and financial reports, where each table is provided in five different formats: Image, Dictionary, HTML, XML, and LaTeX. Our findings indicate that while LLMs maintain robustness across table modalities, they face significant challenges when processing scientific tables.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23423v1">TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-29
      | 💬 ICML 2025
    </div>
    <details class="paper-abstract">
      Past work has studied the effects of fine-tuning on large language models' (LLMs) overall performance on certain tasks. However, a quantitative and systematic method for analyzing its effect on individual outputs is still lacking. Here, we propose a new method for measuring the contribution that fine-tuning makes to individual LLM responses, assuming access to the original pre-trained model. Our method tracks the model's intermediate hidden states, providing a more fine-grained insight into the effects of fine-tuning than a simple comparison of final outputs from pre-trained and fine-tuned models. We introduce and theoretically analyze an exact decomposition of any fine-tuned LLM into a pre-training component and a fine-tuning component. Empirically, we find that model behavior and performance can be steered by up- or down-scaling the fine-tuning component during the forward pass. Motivated by this finding and our theoretical analysis, we define the Tuning Contribution (TuCo) as the ratio of the magnitudes of the fine-tuning component to the pre-training component. We observe that three prominent adversarial attacks on LLMs circumvent safety measures in a way that reduces TuCo, and that TuCo is consistently lower on prompts where these attacks succeed compared to those where they do not. This suggests that attenuating the effect of fine-tuning on model outputs plays a role in the success of such attacks. In summary, TuCo enables the quantitative study of how fine-tuning influences model behavior and safety, and vice versa.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23408v1">Do LLMs Dream of Discrete Algorithms?</a></div>
    <div class="paper-meta">
      📅 2025-06-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have rapidly transformed the landscape of artificial intelligence, enabling natural language interfaces and dynamic orchestration of software components. However, their reliance on probabilistic inference limits their effectiveness in domains requiring strict logical reasoning, discrete decision-making, and robust interpretability. This paper investigates these limitations and proposes a neurosymbolic approach that augments LLMs with logic-based reasoning modules, particularly leveraging Prolog predicates and composable toolsets. By integrating first-order logic and explicit rule systems, our framework enables LLMs to decompose complex queries into verifiable sub-tasks, orchestrate reliable solutions, and mitigate common failure modes such as hallucination and incorrect step decomposition. We demonstrate the practical benefits of this hybrid architecture through experiments on the DABStep benchmark, showing improved precision, coverage, and system documentation in multi-step reasoning tasks. Our results indicate that combining LLMs with modular logic reasoning restores engineering rigor, enhances system reliability, and offers a scalable path toward trustworthy, interpretable AI agents across complex domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13757v3">GenBFA: An Evolutionary Optimization Approach to Bit-Flip Attacks on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing (NLP), excelling in tasks like text generation and summarization. However, their increasing adoption in mission-critical applications raises concerns about hardware-based threats, particularly bit-flip attacks (BFAs). BFAs, enabled by fault injection methods such as Rowhammer, target model parameters in memory, compromising both integrity and performance. Identifying critical parameters for BFAs in the vast parameter space of LLMs poses significant challenges. While prior research suggests transformer-based architectures are inherently more robust to BFAs compared to traditional deep neural networks, we challenge this assumption. For the first time, we demonstrate that as few as three bit-flips can cause catastrophic performance degradation in an LLM with billions of parameters. Current BFA techniques are inadequate for exploiting this vulnerability due to the difficulty of efficiently identifying critical parameters within the immense parameter space. To address this, we propose AttentionBreaker, a novel framework tailored for LLMs that enables efficient traversal of the parameter space to identify critical parameters. Additionally, we introduce GenBFA, an evolutionary optimization strategy designed to refine the search further, isolating the most critical bits for an efficient and effective attack. Empirical results reveal the profound vulnerability of LLMs to AttentionBreaker. For example, merely three bit-flips (4.129 x 10^-9% of total parameters) in the LLaMA3-8B-Instruct 8-bit quantized (W8) model result in a complete performance collapse: accuracy on MMLU tasks drops from 67.3% to 0%, and Wikitext perplexity skyrockets from 12.6 to 4.72 x 10^5. These findings underscore the effectiveness of AttentionBreaker in uncovering and exploiting critical vulnerabilities within LLM architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23377v1">Perspective Dial: Measuring Perspective of Text and Guiding LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2025-06-29
      | 💬 7 pages, 5 main pages of text, 5 figures, 2 tables. Research work performed at CACI INTL INC
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are used in a variety of mission-critical roles. Due to the rapidly developing nature of LLMs, there is a lack of quantifiable understanding of the bias and perspective associated with LLM output. Inspired by this need, this paper considers the broader issue of perspective or viewpoint of general text and perspective control of large-language model (LLM) output. Perspective-Dial consists of two main components: a (1) metric space, dubbed Perspective Space, that enables quantitative measurements of different perspectives regarding a topic, and the use of (2) Systematic Prompt Engineering that utilizes greedy-coordinate descent to control LLM output perspective based on measurement feedback from the Perspective Space. The empirical nature of the approach allows progress to side step a principled understanding of perspective or bias -- effectively quantifying and adjusting outputs for a variety of topics. Potential applications include detection, tracking and mitigation of LLM bias, narrative detection, sense making and tracking in public discourse, and debate bot advocating given perspective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.11189v2">Emotional RAG LLMs: Reading Comprehension for the Open Internet</a></div>
    <div class="paper-meta">
      📅 2025-06-29
    </div>
    <details class="paper-abstract">
      Queries to large language models (LLMs) can be divided into two parts: the instruction/question and the accompanying context. The context for retrieval-augmented generation (RAG) systems in most benchmarks comes from Wikipedia-like texts written in a neutral and factual tone. However, real-world RAG applications often retrieve internet-based text with diverse tones and linguistic styles, posing challenges for downstream tasks. This paper introduces (a) a dataset that transforms RAG-retrieved passages into emotionally inflected and sarcastic text, (b) an emotion translation model for adapting text to different tones, and (c) a prompt-based method to improve LLMs' pragmatic interpretation of retrieved text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23339v1">VALID-Mol: a Systematic Framework for Validated LLM-Assisted Molecular Design</a></div>
    <div class="paper-meta">
      📅 2025-06-29
      | 💬 16 pages, 1 figure, 5 algorithms, 7 tables, to be published in ICSECS Conference 2025, unabridged version
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate remarkable potential for scientific discovery, but their application in domains requiring factual accuracy and domain-specific constraints remains challenging. In molecular design for drug discovery, LLMs can suggest creative molecular modifications but often produce chemically invalid or impractical structures. We present VALID-Mol, a systematic framework for integrating chemical validation with LLM-driven molecular design that increases the rate of generating valid chemical structures from 3% to 83%. Our approach combines methodical prompt engineering, automated chemical validation, and a fine-tuned domain-adapted LLM to ensure reliable generation of synthesizable molecules with improved properties. Beyond the specific implementation, we contribute a generalizable methodology for scientifically-constrained LLM applications, with quantifiable reliability improvements. Computational predictions suggest our framework can generate promising candidates for synthesis with up to 17-fold computationally predicted improvements in target affinity while maintaining synthetic accessibility. We provide a detailed analysis of our prompt engineering process, validation architecture, and fine-tuning approach, offering a reproducible blueprint for applying LLMs to other scientific domains where domain-specific validation is essential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23322v1">GaussMaster: An LLM-based Database Copilot System</a></div>
    <div class="paper-meta">
      📅 2025-06-29
      | 💬 We welcome contributions from the community. For reference, please see the code at: https://gitcode.com/opengauss/openGauss-GaussMaster
    </div>
    <details class="paper-abstract">
      In the financial industry, data is the lifeblood of operations, and DBAs shoulder significant responsibilities for SQL tuning, database deployment, diagnosis, and service repair. In recent years, both database vendors and customers have increasingly turned to autonomous database platforms in an effort to alleviate the heavy workload of DBAs. However, existing autonomous database platforms are limited in their capabilities, primarily addressing single-point issues such as NL2SQL, anomaly detection, and SQL tuning. Manual intervention remains a necessity for comprehensive database maintenance. GaussMaster aims to revolutionize this landscape by introducing an LLM-based database copilot system. This innovative solution is designed not only to assist developers in writing efficient SQL queries but also to provide comprehensive care for database services. When database instances exhibit abnormal behavior, GaussMaster is capable of orchestrating the entire maintenance process automatically. It achieves this by analyzing hundreds of metrics and logs, employing a Tree-of-thought approach to identify root causes, and invoking appropriate tools to resolve issues. We have successfully implemented GaussMaster in real-world scenarios, such as the banking industry, where it has achieved zero human intervention for over 34 database maintenance scenarios. In this paper, we present significant improvements in these tasks with code at https://gitcode.com/opengauss/openGauss-GaussMaster.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13010v3">Agentic Medical Knowledge Graphs Enhance Medical Question Answering: Bridging the Gap Between LLMs and Evolving Medical Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-06-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced medical question-answering by leveraging extensive clinical data and medical literature. However, the rapid evolution of medical knowledge and the labor-intensive process of manually updating domain-specific resources pose challenges to the reliability of these systems. To address this, we introduce Agentic Medical Graph-RAG (AMG-RAG), a comprehensive framework that automates the construction and continuous updating of medical knowledge graphs, integrates reasoning, and retrieves current external evidence, such as PubMed and WikiSearch. By dynamically linking new findings and complex medical concepts, AMG-RAG not only improves accuracy but also enhances interpretability in medical queries. Evaluations on the MEDQA and MEDMCQA benchmarks demonstrate the effectiveness of AMG-RAG, achieving an F1 score of 74.1 percent on MEDQA and an accuracy of 66.34 percent on MEDMCQA, outperforming both comparable models and those 10 to 100 times larger. Notably, these improvements are achieved without increasing computational overhead, highlighting the critical role of automated knowledge graph generation and external evidence retrieval in delivering up-to-date, trustworthy medical insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23266v1">Sub-MoE: Efficient Mixture-of-Expert LLMs Compression via Subspace Expert Merging</a></div>
    <div class="paper-meta">
      📅 2025-06-29
      | 💬 Work in progress, revisions ongoing
    </div>
    <details class="paper-abstract">
      Mixture of Experts (MoE) LLMs face significant obstacles due to their massive parameter scale, which imposes memory, storage, and deployment challenges. Although recent expert merging methods promise greater efficiency by consolidating multiple experts, they are fundamentally hindered by parameter conflicts arising from expert specialization. In this paper, we present Sub-MoE, a novel MoE compression framework via Subspace Expert Merging. Our key insight is to perform joint Singular Value Decomposition (SVD) on concatenated expert weights, reducing conflicting parameters by extracting shared $U$-matrices while enabling effective merging of the expert-specific $V$ components. Specifically, Sub-MoE consists of two innovative phases: (1) Adaptive Expert Clustering, which groups functionally coherent experts via K-means clustering based on cosine similarity of expert outputs; and (2) Subspace Expert Merging, which first enforces Experts Union Decomposition to derive the shared $U$-matrix across experts in the same group, then pursues frequency-based merging for individual $V$-matrices, and finalizes expert reconstruction using the merged $V$-matrix. In this way, we align and fuse experts in a shared subspace, and can be extended with intra-expert compression for further inference optimization. Extensive experiments on Mixtral, DeepSeek, and Qwen-1.5|3 MoE LLMs demonstrate that our Sub-MoE significantly outperforms existing expert pruning and merging methods. Notably, our Sub-MoE maintains 96\%|86\% of original performance with 25\%|50\% expert reduction on Mixtral-8x7B in zero-shot benchmarks. Code will be released at https://github.com/lliai/MoERazor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23260v1">From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows</a></div>
    <div class="paper-meta">
      📅 2025-06-29
      | 💬 29 pages, 15 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Autonomous AI agents powered by large language models (LLMs) with structured function-calling interfaces have dramatically expanded capabilities for real-time data retrieval, complex computation, and multi-step orchestration. Yet, the explosive proliferation of plugins, connectors, and inter-agent protocols has outpaced discovery mechanisms and security practices, resulting in brittle integrations vulnerable to diverse threats. In this survey, we introduce the first unified, end-to-end threat model for LLM-agent ecosystems, spanning host-to-tool and agent-to-agent communications, formalize adversary capabilities and attacker objectives, and catalog over thirty attack techniques. Specifically, we organized the threat model into four domains: Input Manipulation (e.g., prompt injections, long-context hijacks, multimodal adversarial inputs), Model Compromise (e.g., prompt- and parameter-level backdoors, composite and encrypted multi-backdoors, poisoning strategies), System and Privacy Attacks (e.g., speculative side-channels, membership inference, retrieval poisoning, social-engineering simulations), and Protocol Vulnerabilities (e.g., exploits in Model Context Protocol (MCP), Agent Communication Protocol (ACP), Agent Network Protocol (ANP), and Agent-to-Agent (A2A) protocol). For each category, we review representative scenarios, assess real-world feasibility, and evaluate existing defenses. Building on our threat taxonomy, we identify key open challenges and future research directions, such as securing MCP deployments through dynamic trust management and cryptographic provenance tracking; designing and hardening Agentic Web Interfaces; and achieving resilience in multi-agent and federated environments. Our work provides a comprehensive reference to guide the design of robust defense mechanisms and establish best practices for resilient LLM-agent workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23154v1">Can LLM Improve for Expert Forecast Combination? Evidence from the European Central Bank Survey</a></div>
    <div class="paper-meta">
      📅 2025-06-29
    </div>
    <details class="paper-abstract">
      This study explores the potential of large language models (LLMs) to enhance expert forecasting through ensemble learning. Leveraging the European Central Bank's Survey of Professional Forecasters (SPF) dataset, we propose a comprehensive framework to evaluate LLM-driven ensemble predictions under varying conditions, including the intensity of expert disagreement, dynamics of herd behavior, and limitations in attention allocation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.01299v2">The Effectiveness of LLMs as Annotators: A Comparative Overview and Empirical Analysis of Direct Representation</a></div>
    <div class="paper-meta">
      📅 2025-06-29
      | 💬 LREC-COLING NLPerspectives workshop
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have emerged as powerful support tools across various natural language tasks and a range of application domains. Recent studies focus on exploring their capabilities for data annotation. This paper provides a comparative overview of twelve studies investigating the potential of LLMs in labelling data. While the models demonstrate promising cost and time-saving benefits, there exist considerable limitations, such as representativeness, bias, sensitivity to prompt variations and English language preference. Leveraging insights from these studies, our empirical analysis further examines the alignment between human and GPT-generated opinion distributions across four subjective datasets. In contrast to the studies examining representation, our methodology directly obtains the opinion distribution from GPT. Our analysis thereby supports the minority of studies that are considering diverse perspectives when evaluating data annotation tasks and highlights the need for further research in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08686v2">Brevity is the soul of sustainability: Characterizing LLM response lengths</a></div>
    <div class="paper-meta">
      📅 2025-06-29
      | 💬 Accepted to appear at the ACL 2025 findings
    </div>
    <details class="paper-abstract">
      A significant portion of the energy consumed by Large Language Models (LLMs) arises from their inference processes; hence developing energy-efficient methods for inference is crucial. While several techniques exist for inference optimization, output compression remains relatively unexplored, with only a few preliminary efforts addressing this aspect. In this work, we first benchmark 12 decoder-only LLMs across 5 datasets, revealing that these models often produce responses that are substantially longer than necessary. We then conduct a comprehensive quality assessment of LLM responses, formally defining six information categories present in LLM responses. We show that LLMs often tend to include redundant or additional information besides the minimal answer. To address this issue of long responses by LLMs, we explore several simple and intuitive prompt-engineering strategies. Empirical evaluation shows that appropriate prompts targeting length reduction and controlling information content can achieve significant energy optimization between 25-60\% by reducing the response length while preserving the quality of LLM responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23136v1">LLM-Assisted Question-Answering on Technical Documents Using Structured Data-Aware Retrieval Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-29
      | 💬 29 Pages, 11 Tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are capable of natural language understanding and generation. But they face challenges such as hallucination and outdated knowledge. Fine-tuning is one possible solution, but it is resource-intensive and must be repeated with every data update. Retrieval-Augmented Generation (RAG) offers an efficient solution by allowing LLMs to access external knowledge sources. However, traditional RAG pipelines struggle with retrieving information from complex technical documents with structured data such as tables and images. In this work, we propose a RAG pipeline, capable of handling tables and images in documents, for technical documents that support both scanned and searchable formats. Its retrieval process combines vector similarity search with a fine-tuned reranker based on Gemma-2-9b-it. The reranker is trained using RAFT (Retrieval-Augmented Fine-Tuning) on a custom dataset designed to improve context identification for question answering. Our evaluation demonstrates that the proposed pipeline achieves a high faithfulness score of 94% (RAGas) and 96% (DeepEval), and an answer relevancy score of 87% (RAGas) and 93% (DeepEval). Comparative analysis demonstrates that the proposed architecture is superior to general RAG pipelines in terms of table-based questions and handling questions outside context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23133v1">Format-Adapter: Improving Reasoning Capability of LLMs by Adapting Suitable Format</a></div>
    <div class="paper-meta">
      📅 2025-06-29
    </div>
    <details class="paper-abstract">
      Generating and voting multiple answers is an effective method to mitigate reasoning inconsistencies of large language models (LLMs). Prior works have shown that multiple reasoning formats outperform a single format when generating multiple answers. However, previous works using multiple formats rely on formats labeled by humans, which could be unsuitable for all tasks and have high labeling costs. To address this issue, we adapt suitable formats to the given tasks by generating and selecting formats. We first propose how to measure the reasoning error when generating multiple answers. Then, we introduce Format-Adapter, which utilizes LLMs to generate and select suitable reasoning formats by minimizing the error measurement we present. We conduct experiments on math and commonsense reasoning tasks, where Format-Adapter achieves a 4.3% performance improvement on average over previous works, demonstrating the effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23127v1">Unleashing Embodied Task Planning Ability in LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they face significant challenges in embodied task planning scenarios that require continuous environmental understanding and action generation. Existing approaches generate open-loop action scripts based on static knowledge, making it difficult to learn causal relationships between actions and environmental feedback, particularly in partially observable environments. We introduce Embodied Planner-R1, a novel outcome-driven reinforcement learning framework that enables LLMs to develop interactive capabilities through autonomous exploration with minimal supervision. Our framework incorporates three key innovations: (1) Without human annotations, we employ pure reinforcement learning with group rollout, incorporating in-environment interaction through parallel exploration; (2) completion-driven sparse reward; and (3) Interactive Policy Optimization (IPO) for efficient learning from grouped trajectories. Across two challenging text-based Embodied planning benchmarks, Embodied Planner-R1 achieves impressive completion rates of 97.78% on ALFWorld and 79.92% on ScienceWorld, surpassing prior methods by a large margin, and suffers only a -3.66% drop in previously unseen environments, evidencing strong generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04722v2">Enough Coin Flips Can Make LLMs Act Bayesian</a></div>
    <div class="paper-meta">
      📅 2025-06-29
      | 💬 ACL 2025 Main
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit the ability to generalize given few-shot examples in their input prompt, an emergent capability known as in-context learning (ICL). We investigate whether LLMs use ICL to perform structured reasoning in ways that are consistent with a Bayesian framework or rely on pattern matching. Using a controlled setting of biased coin flips, we find that: (1) LLMs often possess biased priors, causing initial divergence in zero-shot settings, (2) in-context evidence outweighs explicit bias instructions, (3) LLMs broadly follow Bayesian posterior updates, with deviations primarily due to miscalibrated priors rather than flawed updates, and (4) attention magnitude has negligible effect on Bayesian inference. With sufficient demonstrations of biased coin flips via ICL, LLMs update their priors in a Bayesian manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10490v4">Learning Dynamics of LLM Finetuning</a></div>
    <div class="paper-meta">
      📅 2025-06-29
    </div>
    <details class="paper-abstract">
      Learning dynamics, which describes how the learning of specific training examples influences the model's predictions on other examples, gives us a powerful tool for understanding the behavior of deep learning systems. We study the learning dynamics of large language models during different types of finetuning, by analyzing the step-wise decomposition of how influence accumulates among different potential responses. Our framework allows a uniform interpretation of many interesting observations about the training of popular algorithms for both instruction tuning and preference tuning. In particular, we propose a hypothetical explanation of why specific types of hallucination are strengthened after finetuning, e.g., the model might use phrases or facts in the response for question B to answer question A, or the model might keep repeating similar simple phrases when generating responses. We also extend our framework and highlight a unique "squeezing effect" to explain a previously observed phenomenon in off-policy direct preference optimization (DPO), where running DPO for too long makes even the desired outputs less likely. This framework also provides insights into where the benefits of on-policy DPO and other variants come from. The analysis not only provides a novel perspective of understanding LLM's finetuning but also inspires a simple, effective method to improve alignment performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23056v1">Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-29
      | 💬 ACL 2025 Main
    </div>
    <details class="paper-abstract">
      Molecular structure elucidation involves deducing a molecule's structure from various types of spectral data, which is crucial in chemical experimental analysis. While large language models (LLMs) have shown remarkable proficiency in analyzing and reasoning through complex tasks, they still encounter substantial challenges in molecular structure elucidation. We identify that these challenges largely stem from LLMs' limited grasp of specialized chemical knowledge. In this work, we introduce a Knowledge-enhanced reasoning framework for Molecular Structure Elucidation (K-MSE), leveraging Monte Carlo Tree Search for test-time scaling as a plugin. Specifically, we construct an external molecular substructure knowledge base to extend the LLMs' coverage of the chemical structure space. Furthermore, we design a specialized molecule-spectrum scorer to act as a reward model for the reasoning process, addressing the issue of inaccurate solution evaluation in LLMs. Experimental results show that our approach significantly boosts performance, particularly gaining more than 20% improvement on both GPT-4o-mini and GPT-4o. Our code is available at https://github.com/HICAI-ZJU/K-MSE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23055v1">Measuring How LLMs Internalize Human Psychological Concepts: A preliminary analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as ChatGPT have shown remarkable abilities in producing human-like text. However, it is unclear how accurately these models internalize concepts that shape human thought and behavior. Here, we developed a quantitative framework to assess concept alignment between LLMs and human psychological dimensions using 43 standardized psychological questionnaires, selected for their established validity in measuring distinct psychological constructs. Our method evaluates how accurately language models reconstruct and classify questionnaire items through pairwise similarity analysis. We compared resulting cluster structures with the original categorical labels using hierarchical clustering. A GPT-4 model achieved superior classification accuracy (66.2\%), significantly outperforming GPT-3.5 (55.9\%) and BERT (48.1\%), all exceeding random baseline performance (31.9\%). We also demonstrated that the estimated semantic similarity from GPT-4 is associated with Pearson's correlation coefficients of human responses in multiple psychological questionnaires. This framework provides a novel approach to evaluate the alignment of the human-LLM concept and identify potential representational biases. Our findings demonstrate that modern LLMs can approximate human psychological constructs with measurable accuracy, offering insights for developing more interpretable AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.14419v3">CHARTOM: A Visual Theory-of-Mind Benchmark for LLMs on Misleading Charts</a></div>
    <div class="paper-meta">
      📅 2025-06-29
    </div>
    <details class="paper-abstract">
      We introduce CHARTOM, a visual theory-of-mind benchmark designed to evaluate multimodal large language models' capability to understand and reason about misleading data visualizations though charts. CHARTOM consists of carefully designed charts and associated questions that require a language model to not only correctly comprehend the factual content in the chart (the FACT question) but also judge whether the chart will be misleading to a human readers (the MIND question), a dual capability with significant societal benefits. We detail the construction of our benchmark including its calibration on human performance and estimation of MIND ground truth called the Human Misleadingness Index. We evaluated several leading LLMs -- including GPT, Claude, Gemini, Qwen, Llama, and Llava series models -- on the CHARTOM dataset and found that it was challenging to all models both on FACT and MIND questions. This highlights the limitations of current LLMs and presents significant opportunity for future LLMs to improve on understanding misleading charts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.00075v1">Theoretical Modeling of LLM Self-Improvement Training Dynamics Through Solver-Verifier Gap</a></div>
    <div class="paper-meta">
      📅 2025-06-29
      | 💬 24 pages
    </div>
    <details class="paper-abstract">
      Self-improvement is among the most prominent techniques within the realm of large language models (LLM), aiming to enhance the LLM performance without relying on external data. Despite its significance, generally how LLM performances evolve during the self-improvement process remains underexplored. In this paper, we theoretically model the training dynamics of self-improvement via the concept of solver-verifier gap. This is inspired by the conjecture that the performance enhancement of self-improvement stems from the gap between LLM's solver capability and verifier capability. Based on the theoretical framework, we further introduce how to predict the ultimate power of self-improvement using only information from the first few training epochs. We empirically validate the effectiveness of the theoretical model on various LLMs and datasets. Beyond self-improvement, we extend our analysis to investigate how external data influences these dynamics within the framework. Notably, we find that under limited external data regimes, such external data can be utilized at any stage without significantly affecting final performances, which accords with the empirical observations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23034v1">Guiding AI to Fix Its Own Flaws: An Empirical Study on LLM-Driven Secure Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become powerful tools for automated code generation. However, these models often overlook critical security practices, which can result in the generation of insecure code that contains vulnerabilities-weaknesses or flaws in the code that attackers can exploit to compromise a system. However, there has been limited exploration of strategies to guide LLMs in generating secure code and a lack of in-depth analysis of the effectiveness of LLMs in repairing code containing vulnerabilities. In this paper, we present a comprehensive evaluation of state-of-the-art LLMs by examining their inherent tendencies to produce insecure code, their capability to generate secure code when guided by self-generated vulnerability hints, and their effectiveness in repairing vulnerabilities when provided with different levels of feedback. Our study covers both proprietary and open-weight models across various scales and leverages established benchmarks to assess a wide range of vulnerability types. Through quantitative and qualitative analyses, we reveal that although LLMs are prone to generating insecure code, advanced models can benefit from vulnerability hints and fine-grained feedback to avoid or fix vulnerabilities. We also provide actionable suggestions to developers to reduce vulnerabilities when using LLMs for code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04133v2">TRiSM for Agentic AI: A Review of Trust, Risk, and Security Management in LLM-based Agentic Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-28
    </div>
    <details class="paper-abstract">
      Agentic AI systems, built upon large language models (LLMs) and deployed in multi-agent configurations, are redefining intelligence, autonomy, collaboration, and decision-making across enterprise and societal domains. This review presents a structured analysis of \textbf{Trust, Risk, and Security Management (TRiSM)} in the context of LLM-based Agentic Multi-Agent Systems (AMAS). We begin by examining the conceptual foundations of Agentic AI and highlight its architectural distinctions from traditional AI agents. We then adapt and extend the AI TRiSM framework for Agentic AI, structured around four key pillars: Governance, Explainability, ModelOps, and Privacy/Security , each contextualized to the challenges of multi-agent LLM systems. A novel risk taxonomy is proposed to capture the unique threats and vulnerabilities of Agentic AI, ranging from coordination failures to prompt-based adversarial manipulation. To support practical assessment in Agentic AI works, we introduce two novel metrics: the Component Synergy Score (CSS), which quantifies the quality of inter-agent collaboration, and the Tool Utilization Efficacy (TUE), which evaluates the efficiency of tool use within agent workflows. We further discuss strategies for improving explainability in Agentic AI , as well as approaches to enhancing security and privacy through encryption, adversarial robustness, and regulatory compliance. The review concludes with a research roadmap for the responsible development and deployment of Agentic AI, outlining critical directions to align emerging systems with TRiSM principles for safe, transparent, and accountable operation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12386v3">Interpretable LLM-based Table Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-06-28
      | 💬 Published in Transactions on Machine Learning Research (TMLR) in 06/2025. Reviews at: https://openreview.net/forum?id=2eTsZBoU2W
    </div>
    <details class="paper-abstract">
      Interpretability in Table Question Answering (Table QA) is critical, especially in high-stakes domains like finance and healthcare. While recent Table QA approaches based on Large Language Models (LLMs) achieve high accuracy, they often produce ambiguous explanations of how answers are derived. We propose Plan-of-SQLs (POS), a new Table QA method that makes the model's decision-making process interpretable. POS decomposes a question into a sequence of atomic steps, each directly translated into an executable SQL command on the table, thereby ensuring that every intermediate result is transparent. Through extensive experiments, we show that: First, POS generates the highest-quality explanations among compared methods, which markedly improves the users' ability to simulate and verify the model's decisions. Second, when evaluated on standard Table QA benchmarks (TabFact, WikiTQ, and FeTaQA), POS achieves QA accuracy that is competitive to existing methods, while also offering greater efficiency-requiring significantly fewer LLM calls and table database queries (up to 25x fewer)-and more robust performance on large-sized tables. Finally, we observe high agreement (up to 90.59% in forward simulation) between LLMs and human users when making decisions based on the same explanations, suggesting that LLMs could serve as an effective proxy for humans in evaluating Table QA explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22846v1">Boosting CTC-Based ASR Using LLM-Based Intermediate Loss Regularization</a></div>
    <div class="paper-meta">
      📅 2025-06-28
      | 💬 This is the accepted version of an article accepted to the TSD 2025 conference, published in Springer Lecture Notes in Artificial Intelligence (LNAI). The final authenticated version is available online at SpringerLink
    </div>
    <details class="paper-abstract">
      End-to-end (E2E) automatic speech recognition (ASR) systems have revolutionized the field by integrating all components into a single neural network, with attention-based encoder-decoder models achieving state-of-the-art performance. However, their autoregressive decoding process limits inference speed, making them unsuitable for real-time applications. In contrast, CTC-based models offer faster, non-autoregressive decoding but struggle to model linguistic dependencies effectively. Addressing this challenge, we propose a novel auxiliary loss framework called Language-Aware Intermediate Loss (LAIL) to enhance CTC-based ASR using the linguistic knowledge of large language models (LLMs). By attaching connector layers to intermediate encoder layers, LAIL maps outputs to the embedding space of an LLM and computes a causal language modeling loss during training. This approach enhances linguistic modeling while preserving the computational efficiency of CTC decoding. Using the Conformer architecture and various LLaMA models, we demonstrate significant improvements in Word Error Rate (WER) on the LibriSpeech, TEDLIUM2, and WSJ corpora, achieving state-of-the-art performance for CTC-based ASR with minimal computational overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18282v3">Better Aligned with Survey Respondents or Training Data? Unveiling Political Leanings of LLMs on U.S. Supreme Court Cases</a></div>
    <div class="paper-meta">
      📅 2025-06-28
    </div>
    <details class="paper-abstract">
      Recent works have shown that Large Language Models (LLMs) have a tendency to memorize patterns and biases present in their training data, raising important questions about how such memorized content influences model behavior. One such concern is the emergence of political bias in LLM outputs. In this paper, we investigate the extent to which LLMs' political leanings reflect memorized patterns from their pretraining corpora. We propose a method to quantitatively evaluate political leanings embedded in the large pretraining corpora. Subsequently we investigate to whom are the LLMs' political leanings more aligned with, their pretrainig corpora or the surveyed human opinions. As a case study, we focus on probing the political leanings of LLMs in 32 US Supreme Court cases, addressing contentious topics such as abortion and voting rights. Our findings reveal that LLMs strongly reflect the political leanings in their training data, and no strong correlation is observed with their alignment to human opinions as expressed in surveys. These results underscore the importance of responsible curation of training data, and the methodology for auditing the memorization in LLMs to ensure human-AI alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22808v1">MedEthicsQA: A Comprehensive Question Answering Benchmark for Medical Ethics Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-28
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      While Medical Large Language Models (MedLLMs) have demonstrated remarkable potential in clinical tasks, their ethical safety remains insufficiently explored. This paper introduces $\textbf{MedEthicsQA}$, a comprehensive benchmark comprising $\textbf{5,623}$ multiple-choice questions and $\textbf{5,351}$ open-ended questions for evaluation of medical ethics in LLMs. We systematically establish a hierarchical taxonomy integrating global medical ethical standards. The benchmark encompasses widely used medical datasets, authoritative question banks, and scenarios derived from PubMed literature. Rigorous quality control involving multi-stage filtering and multi-faceted expert validation ensures the reliability of the dataset with a low error rate ($2.72\%$). Evaluation of state-of-the-art MedLLMs exhibit declined performance in answering medical ethics questions compared to their foundation counterparts, elucidating the deficiencies of medical ethics alignment. The dataset, registered under CC BY-NC 4.0 license, is available at https://github.com/JianhuiWei7/MedEthicsQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15380v4">Kalahi: A handcrafted, grassroots cultural LLM evaluation suite for Filipino</a></div>
    <div class="paper-meta">
      📅 2025-06-28
      | 💬 Accepted for presentation at Paclic 38, 2024
    </div>
    <details class="paper-abstract">
      Multilingual large language models (LLMs) today may not necessarily provide culturally appropriate and relevant responses to its Filipino users. We introduce Kalahi, a cultural LLM evaluation suite collaboratively created by native Filipino speakers. It is composed of 150 high-quality, handcrafted and nuanced prompts that test LLMs for generations that are relevant to shared Filipino cultural knowledge and values. Strong LLM performance in Kalahi indicates a model's ability to generate responses similar to what an average Filipino would say or do in a given situation. We conducted experiments on LLMs with multilingual and Filipino language support. Results show that Kalahi, while trivial for Filipinos, is challenging for LLMs, with the best model answering only 46.0% of the questions correctly compared to native Filipino performance of 89.10%. Thus, Kalahi can be used to accurately and reliably evaluate Filipino cultural representation in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22776v1">Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-28
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Quantization has emerged as a mainstream method for compressing Large Language Models (LLMs), reducing memory requirements and accelerating inference without architectural modifications. While existing research primarily focuses on evaluating the effectiveness of quantized LLMs compared to their original counterparts, the impact on robustness remains largely unexplored.In this paper, we present the first systematic investigation of how quantization affects the robustness of LLMs in code generation tasks. Through extensive experiments across four prominent LLM families (LLaMA, DeepSeek, CodeGen, and StarCoder) with parameter scales ranging from 350M to 33B, we evaluate robustness from dual perspectives: adversarial attacks on input prompts and noise perturbations on model architecture. Our findings challenge conventional wisdom by demonstrating that quantized LLMs often exhibit superior robustness compared to their full-precision counterparts, with 51.59% versus 42.86% of our adversarial experiments showing better resilience in quantized LLMs. Similarly, our noise perturbation experiments also confirm that LLMs after quantitation generally withstand higher levels of weight disturbances. These results suggest that quantization not only reduces computational requirements but can actually enhance LLMs' reliability in code generation tasks, providing valuable insights for developing more robust and efficient LLM deployment strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00309v3">Evaluation of LLMs for mathematical problem solving</a></div>
    <div class="paper-meta">
      📅 2025-06-28
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive performance on a range of educational tasks, but are still understudied for their potential to solve mathematical problems. In this study, we compare three prominent LLMs, including GPT-4o, DeepSeek-V3, and Gemini-2.0, on three mathematics datasets of varying complexities (GSM8K, MATH500, and MIT Open Courseware datasets). We take a five-dimensional approach based on the Structured Chain-of-Thought (SCoT) framework to assess final answer correctness, step completeness, step validity, intermediate calculation accuracy, and problem comprehension. The results show that GPT-4o is the most stable and consistent in performance across all the datasets, but particularly it performs outstandingly in high-level questions of the MIT Open Courseware dataset. DeepSeek-V3 is competitively strong in well-structured domains such as optimisation, but suffers from fluctuations in accuracy in statistical inference tasks. Gemini-2.0 shows strong linguistic understanding and clarity in well-structured problems but performs poorly in multi-step reasoning and symbolic logic. Our error analysis reveals particular deficits in each model: GPT-4o is at times lacking in sufficient explanation or precision; DeepSeek-V3 leaves out intermediate steps; and Gemini-2.0 is less flexible in mathematical reasoning in higher dimensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.01082v7">Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2025-06-28
      | 💬 Oral presentation at ICLR 2025. Camera-ready version available at https://iclr.cc/virtual/2025/poster/30358
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) generate text by sampling the next token from a probability distribution over the vocabulary at each decoding step. Popular sampling methods like top-p (nucleus sampling) often struggle to balance quality and diversity, especially at higher temperatures which lead to incoherent or repetitive outputs. We propose min-p sampling, a dynamic truncation method that adjusts the sampling threshold based on the model's confidence by using the top token's probability as a scaling factor. Our experiments on benchmarks including GPQA, GSM8K, and AlpacaEval Creative Writing show that min-p sampling improves both the quality and diversity of generated text across different model families (Mistral and Llama 3) and model sizes (1B to 123B parameters), especially at higher temperatures. Human evaluations further show a clear preference for min-p sampling, in both text quality and creativity. Min-p sampling has been adopted by popular open-source LLM frameworks, including Hugging Face Transformers, VLLM, and many others, highlighting its considerable impact on improving text generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22716v1">BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute</a></div>
    <div class="paper-meta">
      📅 2025-06-28
      | 💬 Accepted to ICML 2025 (main conference)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are powerful tools but are often expensive to deploy at scale. LLM query routing mitigates this by dynamically assigning queries to models of varying cost and quality to obtain a desired trade-off. Prior query routing approaches generate only one response from the selected model and a single response from a small (inexpensive) model was often not good enough to beat a response from a large (expensive) model due to which they end up overusing the large model and missing out on potential cost savings. However, it is well known that for small models, generating multiple responses and selecting the best can enhance quality while remaining cheaper than a single large-model response. We leverage this idea to propose BEST-Route, a novel routing framework that chooses a model and the number of responses to sample from it based on query difficulty and the quality thresholds. Experiments on real-world datasets demonstrate that our method reduces costs by up to 60% with less than 1% performance drop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22708v1">FairMarket-RL: LLM-Guided Fairness Shaping for Multi-Agent Reinforcement Learning in Peer-to-Peer Markets</a></div>
    <div class="paper-meta">
      📅 2025-06-28
    </div>
    <details class="paper-abstract">
      Peer-to-peer (P2P) trading is increasingly recognized as a key mechanism for decentralized market regulation, yet existing approaches often lack robust frameworks to ensure fairness. This paper presents FairMarket-RL, a novel hybrid framework that combines Large Language Models (LLMs) with Reinforcement Learning (RL) to enable fairness-aware trading agents. In a simulated P2P microgrid with multiple sellers and buyers, the LLM acts as a real-time fairness critic, evaluating each trading episode using two metrics: Fairness-To-Buyer (FTB) and Fairness-Between-Sellers (FBS). These fairness scores are integrated into agent rewards through scheduled {\lambda}-coefficients, forming an adaptive LLM-guided reward shaping loop that replaces brittle, rule-based fairness constraints. Agents are trained using Independent Proximal Policy Optimization (IPPO) and achieve equitable outcomes, fulfilling over 90% of buyer demand, maintaining fair seller margins, and consistently reaching FTB and FBS scores above 0.80. The training process demonstrates that fairness feedback improves convergence, reduces buyer shortfalls, and narrows profit disparities between sellers. With its language-based critic, the framework scales naturally, and its extension to a large power distribution system with household prosumers illustrates its practical applicability. FairMarket-RL thus offers a scalable, equity-driven solution for autonomous trading in decentralized energy systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22694v1">VOCABTRIM: Vocabulary Pruning for Efficient Speculative Decoding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-28
      | 💬 7 pages, 4 figures, 5 tables, accepted at ICML 2025 workshop on Efficient Systems for Foundational Models
    </div>
    <details class="paper-abstract">
      In this paper, we introduce a simple training-free technique to improve the performance of drafter-based speculative decoding (SpD) methods that incorporates language modeling head (LM head) during drafting process. A drafter-based speculative decoding leverages one or more smaller language models, a.k.a. drafters or draft models, to sample a draft sequence or tree consisting of multiple tokens, followed by verification by a base LLM, a target model, accepting a subset as its valid generation. As it is usually considered that the speculative decoding requires one-to-one mapping between vocabularies of the target model and the draft model, it has been natural to share the vocabulary between them, or even share the LM head as in EAGLE or Medusa. We first identify that this draft token sampling scheme inherently contains an unnecessary inference overhead in drafting, especially for some target LLMs with very large vocabularies. Then, we propose a simple technique, VocabTrim, to mitigate the drafting overhead to improve the generation speed in memory-bound environment. VocabTrim reconstructs the drafter LM head to contain only a limited set of tokens, selected by the most frequently sampled from the vocabulary of the target model. While limiting the vocabulary in drafting slightly degrades the acceptance rate, it significantly reduces the drafting latency in memory-bound process which is often the case on edge devices, resulting in higher memory-bound speed up (MBSU). We show that our method can boost the memory-bound speed-up for Llama-3 models on Spec-Bench, specifically by 16% for Llama-3.2-3B-Instruct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01058v1">A Data Science Approach to Calcutta High Court Judgments: An Efficient LLM and RAG-powered Framework for Summarization and Similar Cases Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-06-28
      | 💬 12 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The judiciary, as one of democracy's three pillars, is dealing with a rising amount of legal issues, needing careful use of judicial resources. This research presents a complex framework that leverages Data Science methodologies, notably Large Language Models (LLM) and Retrieval-Augmented Generation (RAG) techniques, to improve the efficiency of analyzing Calcutta High Court verdicts. Our framework focuses on two key aspects: first, the creation of a robust summarization mechanism that distills complex legal texts into concise and coherent summaries; and second, the development of an intelligent system for retrieving similar cases, which will assist legal professionals in research and decision making. By fine-tuning the Pegasus model using case head note summaries, we achieve significant improvements in the summarization of legal cases. Our two-step summarizing technique preserves crucial legal contexts, allowing for the production of a comprehensive vector database for RAG. The RAG-powered framework efficiently retrieves similar cases in response to user queries, offering thorough overviews and summaries. This technique not only improves legal research efficiency, but it also helps legal professionals and students easily acquire and grasp key legal information, benefiting the overall legal scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.16661v3">RLSF: Fine-tuning LLMs via Symbolic Feedback</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed AI but often struggle with tasks that require domain-specific reasoning and logical alignment. Traditional fine-tuning methods do not leverage the vast amount of symbolic domain-knowledge available to us via symbolic reasoning tools (e.g., provers), and are further limited by sparse rewards and unreliable reward models. We introduce Reinforcement Learning via Symbolic Feedback (RLSF), a novel fine-tuning paradigm where symbolic reasoning tools (e.g., solvers, provers, and algebra systems) provide fine-grained feedback to LLMs. RLSF uses poly-sized certificates (e.g., proofs) generated by symbolic tools to identify and correct errors in model outputs, offering token-level guidance without requiring differentiable reasoning systems. This paradigm bridges the gap between symbolic reasoning and LLM fine-tuning, enabling precise alignment with domain-specific constraints while addressing key limitations of traditional reward signals. Via extensive evaluations, we show that our RLSF-based fine-tuning of LLMs outperforms traditional approaches on five different applications (that have some associated logical or domain constraints), namely, program synthesis from natural language pseudo-code to programming language, three chemistry tasks, and solving the Game of 24. A key takeaway is that fine-tuning via RLSF enables relatively smaller LLMs to significantly outperform closed-source models that are orders of magnitude larger.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22688v1">An LLM-assisted approach to designing software architectures using ADD</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 30 pages, 12 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Designing effective software architectures is a complex, iterative process that traditionally relies on expert judgment. This paper proposes an approach for Large Language Model (LLM)-assisted software architecture design using the Attribute-Driven Design (ADD) method. By providing an LLM with an explicit description of ADD, an architect persona, and a structured iteration plan, our method guides the LLM to collaboratively produce architecture artifacts with a human architect. We validate the approach through case studies, comparing generated designs against proven solutions and evaluating them with professional architects. Results show that our LLM-assisted ADD process can generate architectures closely aligned with established solutions and partially satisfying architectural drivers, highlighting both the promise and current limitations of using LLMs in architecture design. Our findings emphasize the importance of human oversight and iterative refinement when leveraging LLMs in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04745v4">Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 13 pages, 23 figures. Accepted to XLLM Workshop at ACL 2025
    </div>
    <details class="paper-abstract">
      This paper evaluates the ability of Large Language Models (LLMs) to leverage contextual information in the form of structured linguistic representations. Specifically, we examine the impact of encoding both short and long contexts using Abstract Meaning Representation (AMR) structures across a diverse set of language tasks. We perform our analysis using 8-bit quantized and instruction-tuned versions of Llama 3.1 (8B), Phi-3, and Mistral 7B. Our results indicate that, for tasks involving short contexts, augmenting the prompt with the AMR of the original language context often degrades the performance of the underlying LLM. However, for tasks that involve long contexts, such as dialogue summarization in the SAMSum dataset, this enhancement improves LLM performance, for example, by increasing the zero-shot cosine similarity score of Llama 3.1 from 66% to 76%. This improvement is more evident in the newer and larger LLMs, but does not extend to the older or smaller ones. In addition, we observe that LLMs can effectively reconstruct the original text from a linearized AMR, achieving a cosine similarity of 81% in the best-case scenario.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09397v3">SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 6 pages, 6 figures, 2 tables
    </div>
    <details class="paper-abstract">
      The growing gap between the increasing complexity of large language models (LLMs) and the limited computational budgets of edge devices poses a key challenge for efficient on-device inference, despite gradual improvements in hardware capabilities. Existing strategies, such as aggressive quantization, pruning, or remote inference, trade accuracy for efficiency or lead to substantial cost burdens. This position paper introduces a new framework that leverages speculative decoding, previously viewed primarily as a decoding acceleration technique for autoregressive generation of LLMs, as a promising approach specifically adapted for edge computing by orchestrating computation across heterogeneous devices. We propose \acronym, a framework that allows lightweight edge devices to draft multiple candidate tokens locally using diverse draft models, while a single, shared edge server verifies the tokens utilizing a more precise target model. To further increase the efficiency of verification, the edge server batch the diverse verification requests from devices. This approach supports device heterogeneity and reduces server-side memory footprint by sharing the same upstream target model across multiple devices. Our initial experiments with Jetson Orin Nano, Raspberry Pi 4B/5, and an edge server equipped with 4 Nvidia A100 GPUs indicate substantial benefits: 2.2 more system throughput, 2.8 more system capacity, and better cost efficiency, all without sacrificing model accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22604v1">Bootstrapping Human-Like Planning via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Accepted by the 2025 34th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN)
    </div>
    <details class="paper-abstract">
      Robot end users increasingly require accessible means of specifying tasks for robots to perform. Two common end-user programming paradigms include drag-and-drop interfaces and natural language programming. Although natural language interfaces harness an intuitive form of human communication, drag-and-drop interfaces enable users to meticulously and precisely dictate the key actions of the robot's task. In this paper, we investigate the degree to which both approaches can be combined. Specifically, we construct a large language model (LLM)-based pipeline that accepts natural language as input and produces human-like action sequences as output, specified at a level of granularity that a human would produce. We then compare these generated action sequences to another dataset of hand-specified action sequences. Although our results reveal that larger models tend to outperform smaller ones in the production of human-like action sequences, smaller models nonetheless achieve satisfactory performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18435v3">What Makes the Preferred Thinking Direction for LLMs in Multiple-choice Questions?</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 10 pages for the main text
    </div>
    <details class="paper-abstract">
      Language models usually use left-to-right (L2R) autoregressive factorization. However, L2R factorization may not always be the best inductive bias. Therefore, we investigate whether alternative factorizations of the text distribution could be beneficial in some tasks. We investigate right-to-left (R2L) training as a compelling alternative, focusing on multiple-choice questions (MCQs) as a test bed for knowledge extraction and reasoning. Through extensive experiments across various model sizes (2B-8B parameters) and training datasets, we find that R2L models can significantly outperform L2R models on several MCQ benchmarks, including logical reasoning, commonsense understanding, and truthfulness assessment tasks. Our analysis reveals that this performance difference may be fundamentally linked to multiple factors including calibration, computability, and directional conditional entropy. We analyze the impact of these factors through controlled simulation studies using arithmetic tasks, where the impacting factors can be better disentangled. Our work demonstrates that exploring alternative factorizations of the text distribution can lead to improvements in LLM capabilities and provides theoretical insights into optimal factorization towards approximating human language distribution, and when each reasoning order might be more advantageous. Our code and checkpoints are released at https://github.com/apple/ml-reversal-blessing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15727v2">Retrieval Augmented Generation Based LLM Evaluation For Protocol State Machine Inference With Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Minor modifications in sections: abstract, introduction, background problem formulation, and conclusion. (Typos and Clarifications)
    </div>
    <details class="paper-abstract">
      This paper presents a novel approach to evaluate the efficiency of a RAG-based agentic Large Language Model (LLM) architecture for network packet seed generation and enrichment. Enhanced by chain-of-thought (COT) prompting techniques, the proposed approach focuses on the improvement of the seeds' structural quality in order to guide protocol fuzzing frameworks through a wide exploration of the protocol state space. Our method leverages RAG and text embeddings to dynamically reference to the Request For Comments (RFC) documents knowledge base for answering queries regarding the protocol's Finite State Machine (FSM), then iteratively reasons through the retrieved knowledge, for output refinement and proper seed placement. We then evaluate the response structure quality of the agent's output, based on metrics as BLEU, ROUGE, and Word Error Rate (WER) by comparing the generated packets against the ground-truth packets. Our experiments demonstrate significant improvements of up to 18.19%, 14.81%, and 23.45% in BLEU, ROUGE, and WER, respectively, over baseline models. These results confirm the potential of such approach, improving LLM-based protocol fuzzing frameworks for the identification of hidden vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22557v1">MetaCipher: A General and Extensible Reinforcement Learning Framework for Obfuscation-Based Jailbreak Attacks on Black-Box LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      The growing capabilities of large language models (LLMs) have exposed them to increasingly sophisticated jailbreak attacks. Among these, obfuscation-based attacks -- which encrypt malicious content to evade detection -- remain highly effective. By leveraging the reasoning ability of advanced LLMs to interpret encrypted prompts, such attacks circumvent conventional defenses that rely on keyword detection or context filtering. These methods are very difficult to defend against, as existing safety mechanisms are not designed to interpret or decode ciphered content. In this work, we propose \textbf{MetaCipher}, a novel obfuscation-based jailbreak framework, along with a reinforcement learning-based dynamic cipher selection mechanism that adaptively chooses optimal encryption strategies from a cipher pool. This approach enhances jailbreak effectiveness and generalizability across diverse task types, victim LLMs, and safety guardrails. Our framework is modular and extensible by design, supporting arbitrary cipher families and accommodating evolving adversarial strategies. We complement our method with a large-scale empirical analysis of cipher performance across multiple victim LLMs. Within as few as 10 queries, MetaCipher achieves over 92\% attack success rate (ASR) on most recent standard malicious prompt benchmarks against state-of-the-art non-reasoning LLMs, and over 74\% ASR against reasoning-capable LLMs, outperforming all existing obfuscation-based jailbreak methods. These results highlight the long-term robustness and adaptability of our approach, making it more resilient than prior methods in the face of advancing safety measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22419v1">The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      Rapid advancements in large language models (LLMs) have the potential to assist in scientific progress. A critical capability toward this endeavor is the ability to reproduce existing work. To evaluate the ability of AI agents to reproduce results in an active research area, we introduce the Automated LLM Speedrunning Benchmark, leveraging the research community contributions on the NanoGPT speedrun, a competition to train a GPT-2 model in the shortest time. Each of the 19 speedrun tasks provides the agent with the previous records training script, optionally paired with one of three hint formats, ranging from pseudocode to paper-like descriptions of the new records improvements. Records execute quickly by design and speedrun improvements encompass diverse code-level changes, ranging from high-level algorithmic advancements to hardware-aware optimizations. These features make the benchmark both accessible and realistic for the frontier problem of improving LLM training. We find that recent reasoning LLMs combined with SoTA scaffolds struggle to reimplement already-known innovations in our benchmark, even when given detailed hints. Our benchmark thus provides a simple, non-saturated measure of an LLMs ability to automate scientific reproduction, a necessary (but not sufficient) skill for an autonomous research agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22396v1">QuickSilver -- Speeding up LLM Inference through Dynamic Token Halting, KV Skipping, Contextual Token Fusion, and Adaptive Matryoshka Quantization</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Preprint. Under submission
    </div>
    <details class="paper-abstract">
      Inference accounts for the majority of latency and energy consumption in large language model (LLM) deployments, often exceeding 90% of total cost. While training-time efficiency has seen extensive progress, runtime optimization remains a key bottleneck, particularly under autoregressive decoding. Existing approaches -- such as pruning, quantization, early exits, and speculative decoding -- often require retraining, architectural changes, or disrupt decoding compatibility. We introduce QuickSilver, a modular, token-level framework that enables semantic adaptivity at inference time without altering model weights or structure. QuickSilver integrates four synergistic mechanisms: (i) Dynamic Token Halting, which halts computation for tokens with converged representations; (ii) KV Cache Skipping, which selectively suppresses memory writes to reduce attention overhead; and (iii) Contextual Token Fusion, which collapses redundant tokens into shared paths to shrink sequence length. Unlike speculative decoding or MoE routing, QuickSilver operates entirely on frozen, dense models and requires no auxiliary networks. Applied to GPT-2 and Llama-2 across WikiText-103 and C4, QuickSilver achieves up to 39.6% FLOP reduction with negligible perplexity degradation (<=0.2).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22372v1">Towards Fair Rankings: Leveraging LLMs for Gender Bias Detection and Measurement</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Accepted by ACM SIGIR Conference on Innovative Concepts and Theories in Information Retrieval (ICTIR 2025)
    </div>
    <details class="paper-abstract">
      The presence of social biases in Natural Language Processing (NLP) and Information Retrieval (IR) systems is an ongoing challenge, which underlines the importance of developing robust approaches to identifying and evaluating such biases. In this paper, we aim to address this issue by leveraging Large Language Models (LLMs) to detect and measure gender bias in passage ranking. Existing gender fairness metrics rely on lexical- and frequency-based measures, leading to various limitations, e.g., missing subtle gender disparities. Building on our LLM-based gender bias detection method, we introduce a novel gender fairness metric, named Class-wise Weighted Exposure (CWEx), aiming to address existing limitations. To measure the effectiveness of our proposed metric and study LLMs' effectiveness in detecting gender bias, we annotate a subset of the MS MARCO Passage Ranking collection and release our new gender bias collection, called MSMGenderBias, to foster future research in this area. Our extensive experimental results on various ranking models show that our proposed metric offers a more detailed evaluation of fairness compared to previous metrics, with improved alignment to human labels (58.77% for Grep-BiasIR, and 18.51% for MSMGenderBias, measured using Cohen's Kappa agreement), effectively distinguishing gender bias in ranking. By integrating LLM-driven bias detection, an improved fairness metric, and gender bias annotations for an established dataset, this work provides a more robust framework for analyzing and mitigating bias in IR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22316v1">Evaluating Scoring Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      The remarkable performance of Large Language Models (LLMs) gives rise to``LLM-as-a-Judge'', where LLMs are employed as evaluators for complex tasks. Moreover, it has been widely adopted across fields such as Natural Language Processing (NLP), preference learning, and various specific domains. However, there are various biases within LLM-as-a-Judge, which adversely affect the fairness and reliability of judgments. Current research on evaluating or mitigating bias in LLM-as-a-Judge predominantly focuses on comparison-based evaluations, while systematic investigations into bias in scoring-based evaluations remain limited. Therefore, we define scoring bias in LLM-as-a-Judge as the scores differ when scoring judge models are bias-related perturbed, and provide a well-designed framework to comprehensively evaluate scoring bias. We augment existing LLM-as-a-Judge benchmarks through data synthesis to construct our evaluation dataset and design multi-faceted evaluation metrics. Our experimental results demonstrate that the scoring stability of existing judge models is disrupted by scoring biases. Further exploratory experiments and discussions provide valuable insights into the design of scoring prompt templates and the mitigation of scoring biases on aspects such as score rubrics, score IDs, and reference answer selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22232v1">Leveraging In-Context Learning for Political Bias Testing of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      A growing body of work has been querying LLMs with political questions to evaluate their potential biases. However, this probing method has limited stability, making comparisons between models unreliable. In this paper, we argue that LLMs need more context. We propose a new probing task, Questionnaire Modeling (QM), that uses human survey data as in-context examples. We show that QM improves the stability of question-based bias evaluation, and demonstrate that it may be used to compare instruction-tuned models to their base versions. Experiments with LLMs of various sizes indicate that instruction tuning can indeed change the direction of bias. Furthermore, we observe a trend that larger models are able to leverage in-context examples more effectively, and generally exhibit smaller bias scores in QM. Data and code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17944v2">SegChange-R1: LLM-Augmented Remote Sensing Change Detection</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      Remote sensing change detection is used in urban planning, terrain analysis, and environmental monitoring by analyzing feature changes in the same area over time. In this paper, we propose a large language model (LLM) augmented inference approach (SegChange-R1), which enhances the detection capability by integrating textual descriptive information and guides the model to focus on relevant change regions, accelerating convergence. We designed a linear attention-based spatial transformation module (BEV) to address modal misalignment by unifying features from different times into a BEV space. Furthermore, we introduce DVCD, a novel dataset for building change detection from UAV viewpoints. Experiments on four widely-used datasets demonstrate significant improvements over existing method The code and pre-trained models are available in {https://github.com/Yu-Zhouz/SegChange-R1}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03313v2">LLM as GNN: Graph Vocabulary Learning for Text-Attributed Graph Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      Text-Attributed Graphs (TAGs), where each node is associated with text descriptions, are ubiquitous in real-world scenarios. They typically exhibit distinctive structure and domain-specific knowledge, motivating the development of a Graph Foundation Model (GFM) that generalizes across diverse graphs and tasks. Despite large efforts to integrate Large Language Models (LLMs) and Graph Neural Networks (GNNs) for TAGs, existing approaches suffer from decoupled architectures with two-stage alignment, limiting their synergistic potential. Even worse, existing methods assign out-of-vocabulary (OOV) tokens to graph nodes, leading to graph-specific semantics, token explosion, and incompatibility with task-oriented prompt templates, which hinders cross-graph and cross-task transferability. To address these challenges, we propose PromptGFM, a versatile GFM for TAGs grounded in graph vocabulary learning. PromptGFM comprises two key components: (1) Graph Understanding Module, which explicitly prompts LLMs to replicate the finest GNN workflow within the text space, facilitating seamless GNN-LLM integration and elegant graph-text alignment; (2) Graph Inference Module, which establishes a language-based graph vocabulary ensuring expressiveness, transferability, and scalability, enabling readable instructions for LLM fine-tuning. Extensive experiments demonstrate our superiority and transferability across diverse graphs and tasks. The code is available at this: https://github.com/agiresearch/PromptGFM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04412v3">Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Presented at ICLR 2025 Workshop on Foundation Models in the Wild
    </div>
    <details class="paper-abstract">
      Recent advances demonstrate that increasing inference-time computation can significantly boost the reasoning capabilities of large language models (LLMs). Although repeated sampling (i.e., generating multiple candidate outputs) is a highly effective strategy, it does not leverage external feedback signals for refinement, which are often available in tasks like coding. In this work, we propose Adaptive Branching Monte Carlo Tree Search (AB-MCTS), a novel inference-time framework that generalizes repeated sampling with principled multi-turn exploration and exploitation. At each node in the search tree, AB-MCTS dynamically decides whether to "go wider" by expanding new candidate responses or "go deeper" by revisiting existing ones based on external feedback signals. We evaluate our method on complex coding and engineering tasks using frontier models. Empirical results show that AB-MCTS consistently outperforms both repeated sampling and standard MCTS, underscoring the importance of combining the response diversity of LLMs with multi-turn solution refinement for effective inference-time scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24616v3">Eye of Judgement: Dissecting the Evaluation of Russian-speaking LLMs with POLLUX</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 178 pages
    </div>
    <details class="paper-abstract">
      We introduce POLLUX, a comprehensive open-source benchmark designed to evaluate the generative capabilities of large language models (LLMs) in Russian. Our main contribution is a novel evaluation methodology that enhances the interpretability of LLM assessment. For each task type, we define a set of detailed criteria and develop a scoring protocol where models evaluate responses and provide justifications for their ratings. This enables transparent, criteria-driven evaluation beyond traditional resource-consuming, side-by-side human comparisons. POLLUX includes a detailed, fine-grained taxonomy of 35 task types covering diverse generative domains such as code generation, creative writing, and practical assistant use cases, totaling 2,100 manually crafted and professionally authored prompts. Each task is categorized by difficulty (easy/medium/hard), with experts constructing the dataset entirely from scratch. We also release a family of LLM-as-a-Judge (7B and 32B) evaluators trained for nuanced assessment of generative outputs. This approach provides scalable, interpretable evaluation and annotation tools for model development, effectively replacing costly and less precise human judgments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22139v1">Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Accepted at ICCV 2025
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have demonstrated significant success in visual understanding tasks. However, challenges persist in adapting these models for video comprehension due to the large volume of data and temporal complexity. Existing Video-LLMs using uniform frame sampling often struggle to capture the query-related crucial spatiotemporal clues of videos effectively. In this paper, we introduce Q-Frame, a novel approach for adaptive frame selection and multi-resolution scaling tailored to the video's content and the specific query. Q-Frame employs a training-free, plug-and-play strategy generated by a text-image matching network like CLIP, utilizing the Gumbel-Max trick for efficient frame selection. Q-Frame allows Video-LLMs to process more frames without exceeding computational limits, thereby preserving critical temporal and spatial information. We demonstrate Q-Frame's effectiveness through extensive experiments on benchmark datasets, including MLVU, LongVideoBench, and Video-MME, illustrating its superiority over existing methods and its applicability across various video understanding tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08837v3">Design Patterns for Securing LLM Agents against Prompt Injections</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      As AI agents powered by Large Language Models (LLMs) become increasingly versatile and capable of addressing a broad spectrum of tasks, ensuring their security has become a critical challenge. Among the most pressing threats are prompt injection attacks, which exploit the agent's resilience on natural language inputs -- an especially dangerous threat when agents are granted tool access or handle sensitive information. In this work, we propose a set of principled design patterns for building AI agents with provable resistance to prompt injection. We systematically analyze these patterns, discuss their trade-offs in terms of utility and security, and illustrate their real-world applicability through a series of case studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03592v3">English K_Quantization of LLMs Does Not Disproportionately Diminish Multilingual Performance</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 8 pages, 6 figures, v2
    </div>
    <details class="paper-abstract">
      For consumer usage of locally deployed LLMs, the GGUF format and k\_quantization are invaluable tools for maintaining the performance of the original model while reducing it to sizes deployable with consumer-grade hardware. The number of bits dedicated to each weight from the original model is reduced based on how important they are thought to be during model inference. This importance is arrived at through the application of an 'importance matrix'-a relatively small text document meant to be representative of the LLM's standard use-cases. In the vast majority of quants available online, this document is primarily written in English. It was therefore an open question whether performance on English language tasks was preserved through the sacrifice of multilingual performance and whether it can be preserved with alternate importance matrices. This article investigates these hypotheses by quantizing Llama3.3 70B on importance matrices written in three languages (English, Norwegian, and Malayalam) and evaluating them on the MixEval dataset in both English and Norwegian. All experiments related to yielded non-significant results indicating that current quantization practices do not disproportionately harm multilingual performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17066v2">Improving LLM Outputs Against Jailbreak Attacks with Expert Model Integration</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Under review at IEEE Access. Supplementary material is included in the main PDF
    </div>
    <details class="paper-abstract">
      Using LLMs in a production environment presents security challenges that include vulnerabilities to jailbreaks and prompt injections, which can result in harmful outputs for humans or the enterprise. The challenge is amplified when working within a specific domain, as topics generally accepted for LLMs to address may be irrelevant to that field. These problems can be mitigated, for example, by fine-tuning large language models with domain-specific and security-focused data. However, these alone are insufficient, as jailbreak techniques evolve. Additionally, API-accessed models do not offer the flexibility needed to tailor behavior to industry-specific objectives, and in-context learning is not always sufficient or reliable. In response to these challenges, we introduce Archias, an expert model adept at distinguishing between in-domain and out-of-domain communications. Archias classifies user inquiries into several categories: in-domain (specifically for the automotive industry), malicious questions, price injections, prompt injections, and out-of-domain examples. Our methodology integrates outputs from the expert model (Archias) into prompts, which are then processed by the LLM to generate responses. This method increases the model's ability to understand the user's intention and give appropriate answers. Archias can be adjusted, fine-tuned, and used for many different purposes due to its small size. Therefore, it can be easily customized to the needs of any industry. To validate our approach, we created a benchmark dataset for the automotive industry. Furthermore, in the interest of advancing research and development, we release our benchmark dataset to the community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22050v1">Decoding Machine Translationese in English-Chinese News: LLMs vs. NMTs</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 14 pages, 5 figures, 6 tables. Accpeted in MT Summit 2025, Research: Technical track. Official version may be accessed later in the ACL Anthology
    </div>
    <details class="paper-abstract">
      This study explores Machine Translationese (MTese) -- the linguistic peculiarities of machine translation outputs -- focusing on the under-researched English-to-Chinese language pair in news texts. We construct a large dataset consisting of 4 sub-corpora and employ a comprehensive five-layer feature set. Then, a chi-square ranking algorithm is applied for feature selection in both classification and clustering tasks. Our findings confirm the presence of MTese in both Neural Machine Translation systems (NMTs) and Large Language Models (LLMs). Original Chinese texts are nearly perfectly distinguishable from both LLM and NMT outputs. Notable linguistic patterns in MT outputs are shorter sentence lengths and increased use of adversative conjunctions. Comparing LLMs and NMTs, we achieve approximately 70% classification accuracy, with LLMs exhibiting greater lexical diversity and NMTs using more brackets. Additionally, translation-specific LLMs show lower lexical diversity but higher usage of causal conjunctions compared to generic LLMs. Lastly, we find no significant differences between LLMs developed by Chinese firms and their foreign counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22038v1">Can Peter Pan Survive MT? A Stylometric Study of LLMs, NMTs, and HTs in Children's Literature Translation</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 19 pages, 8 figures, 4 tables. Accepted in 2nd Workshop on Creative-text Translation and Technology Co-located with MT Summit 2025. Official paper may later be accessed from ACL Anthology
    </div>
    <details class="paper-abstract">
      This study focuses on evaluating the performance of machine translations (MTs) compared to human translations (HTs) in English-to-Chinese children's literature translation (CLT) from a stylometric perspective. The research constructs a Peter Pan corpus, comprising 21 translations: 7 human translations (HTs), 7 large language model translations (LLMs), and 7 neural machine translation outputs (NMTs). The analysis employs a generic feature set (including lexical, syntactic, readability, and n-gram features) and a creative text translation (CTT-specific) feature set, which captures repetition, rhythm, translatability, and miscellaneous levels, yielding 447 linguistic features in total. Using classification and clustering techniques in machine learning, we conduct a stylometric analysis of these translations. Results reveal that in generic features, HTs and MTs exhibit significant differences in conjunction word distributions and the ratio of 1-word-gram-YiYang, while NMTs and LLMs show significant variation in descriptive words usage and adverb ratios. Regarding CTT-specific features, LLMs outperform NMTs in distribution, aligning more closely with HTs in stylistic characteristics, demonstrating the potential of LLMs in CLT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03492v2">Towards Reproducible LLM Evaluation: Quantifying Uncertainty in LLM Benchmark Scores</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 4 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are stochastic, and not all models give deterministic answers, even when setting temperature to zero with a fixed random seed. However, few benchmark studies attempt to quantify uncertainty, partly due to the time and cost of repeated experiments. We use benchmarks designed for testing LLMs' capacity to reason about cardinal directions to explore the impact of experimental repeats on mean score and prediction interval. We suggest a simple method for cost-effectively quantifying the uncertainty of a benchmark score and make recommendations concerning reproducible LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22033v1">SiPipe: Bridging the CPU-GPU Utilization Gap for Efficient Pipeline-Parallel LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      As inference workloads for large language models (LLMs) scale to meet growing user demand, pipeline parallelism (PP) has become a widely adopted strategy for multi-GPU deployment, particularly in cross-node setups, to improve key-value (KV) cache capacity and inference throughput. However, PP suffers from inherent inefficiencies caused by three types of execution bubbles-load-imbalance, intra-stage, and inter-stage-which limit pipeline saturation. We present SiPipe, a heterogeneous pipeline design that improves throughput by leveraging underutilized CPU resources to offload auxiliary computation and communication. SiPipe incorporates three key techniques-CPU sampling, a token-safe execution model, and structure-aware transmission-to mitigate pipeline bubbles and improve execution efficiency. Across diverse LLMs, SiPipe achieves up to 2.1 times higher throughput, 43% lower per-token latency, and up to 23% higher average GPU utilization compared to the state-of-the-art vLLM under the same PP configuration, demonstrating its generality across LLMs and deployment scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22028v1">LMPVC and Policy Bank: Adaptive voice control for industrial robots with code generating LLMs and reusable Pythonic policies</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 Accepted by the 2025 34th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN). For further information, videos and code, see https://github.com/ozzyuni/LMPVC
    </div>
    <details class="paper-abstract">
      Modern industry is increasingly moving away from mass manufacturing, towards more specialized and personalized products. As manufacturing tasks become more complex, full automation is not always an option, human involvement may be required. This has increased the need for advanced human robot collaboration (HRC), and with it, improved methods for interaction, such as voice control. Recent advances in natural language processing, driven by artificial intelligence (AI), have the potential to answer this demand. Large language models (LLMs) have rapidly developed very impressive general reasoning capabilities, and many methods of applying this to robotics have been proposed, including through the use of code generation. This paper presents Language Model Program Voice Control (LMPVC), an LLM-based prototype voice control architecture with integrated policy programming and teaching capabilities, built for use with Robot Operating System 2 (ROS2) compatible robots. The architecture builds on prior works using code generation for voice control by implementing an additional programming and teaching system, the Policy Bank. We find this system can compensate for the limitations of the underlying LLM, and allow LMPVC to adapt to different downstream tasks without a slow and costly training process. The architecture and additional results are released on GitHub (https://github.com/ozzyuni/LMPVC).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00299v3">ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-06-27
      | 💬 41 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) require significant GPU memory when processing long texts, with the key value (KV) cache consuming up to 70\% of total memory during inference. Although existing compression methods reduce memory by evaluating the importance of individual tokens, they overlook critical semantic relationships between tokens, resulting in fragmented context and degraded performance. We introduce ChunkKV, which fundamentally reimagines KV cache compression by treating semantic chunks - rather than isolated tokens - as basic compression units. This approach preserves complete linguistic structures and contextual integrity, ensuring that essential meaning is retained even under aggressive compression. Our innovation includes a novel layer-wise index reuse technique that exploits the higher cross-layer similarity of preserved indices in ChunkKV, reducing computational overhead and improving throughput by 26.5\%. Comprehensive evaluations on challenging benchmarks: LongBench, Needle-In-A-HayStack, GSM8K, and JailbreakV demonstrate that ChunkKV outperforms state-of-the-art methods by up to 8.7\% in precision while maintaining the same compression ratio. These results confirm that semantic-aware compression significantly enhances both efficiency and performance for long-context LLM inference, providing a simple yet effective solution to the memory bottleneck problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02862v3">Cannot See the Forest for the Trees: Invoking Heuristics and Biases to Elicit Irrational Choices of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      Despite the remarkable performance of Large Language Models (LLMs), they remain vulnerable to jailbreak attacks, which can compromise their safety mechanisms. Existing studies often rely on brute-force optimization or manual design, failing to uncover potential risks in real-world scenarios. To address this, we propose a novel jailbreak attack framework, ICRT, inspired by heuristics and biases in human cognition. Leveraging the simplicity effect, we employ cognitive decomposition to reduce the complexity of malicious prompts. Simultaneously, relevance bias is utilized to reorganize prompts, enhancing semantic alignment and inducing harmful outputs effectively. Furthermore, we introduce a ranking-based harmfulness evaluation metric that surpasses the traditional binary success-or-failure paradigm by employing ranking aggregation methods such as Elo, HodgeRank, and Rank Centrality to comprehensively quantify the harmfulness of generated content. Experimental results show that our approach consistently bypasses mainstream LLMs' safety mechanisms and generates high-risk content, providing insights into jailbreak attack risks and contributing to stronger defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12286v2">The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason</a></div>
    <div class="paper-meta">
      📅 2025-06-27
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly capable and widely adopted, benchmarks play a central role in assessing their practical utility. For example, SWE-Bench Verified has emerged as a critical benchmark for evaluating LLMs' software engineering abilities, particularly their aptitude for resolving real-world GitHub issues. Recent LLMs show impressive performance on SWE-Bench, leading to optimism about their capacity for complex coding tasks. However, current evaluation protocols may overstate these models' true capabilities. It is crucial to distinguish LLMs' generalizable problem-solving ability and other learned artifacts. In this work, we introduce two diagnostic tasks: file path identification from issue descriptions alone, and ground truth function reproduction with only the current file context and issue description to probe models' underlying knowledge. We present empirical evidence that performance gains on SWE-Bench-Verified may be partially driven by memorization rather than genuine problem-solving. We show that state-of-the-art models achieve up to 76% accuracy in identifying buggy file paths using only issue descriptions, without access to repository structure. This performance is merely up to 53% on tasks from repositories not included in SWE-Bench, pointing to possible data contamination or memorization. A similar pattern is also observed for the function reproduction task, where the verbatim similarity is much higher on SWE-Bench-Verified than on other similar coding benchmarks. These findings raise concerns about the validity of existing results and underscore the need for more robust, contamination-resistant benchmarks to reliably evaluate LLMs' coding abilities.
    </details>
</div>
