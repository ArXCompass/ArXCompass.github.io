# llm - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10061v2">Text2Graph: Combining Lightweight LLMs and GNNs for Efficient Text Classification in Label-Scarce Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become effective zero-shot classifiers, but their high computational requirements and environmental costs limit their practicality for large-scale annotation in high-performance computing (HPC) environments. To support more sustainable workflows, we present Text2Graph, an open-source Python package that provides a modular implementation of existing text-to-graph classification approaches. The framework enables users to combine LLM-based partial annotation with Graph Neural Network (GNN) label propagation in a flexible manner, making it straightforward to swap components such as feature extractors, edge construction methods, and sampling strategies. We benchmark Text2Graph on a zero-shot setting using five datasets spanning topic classification and sentiment analysis tasks, comparing multiple variants against other zero-shot approaches for text classification. In addition to reporting performance, we provide detailed estimates of energy consumption and carbon emissions, showing that graph-based propagation achieves competitive results at a fraction of the energy and environmental cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11689v1">Evaluating Cooperative Resilience in Multiagent Systems: A Comparison Between Humans and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 Supplementary material in https://github.com/mavivi95/resilience_humans_vs_LLM/blob/main/Supplementary_File.pdf
    </div>
    <details class="paper-abstract">
      This paper presents a comparative analysis of cooperative resilience in multi-agent systems, defined as the ability to anticipate, resist, recover from, and transform to disruptive events that affect collective well-being. We focus on mixed-motive social dilemmas instantiated as a \textit{Tragedy of the Commons} environment from the Melting Pot suite, where we systematically compare human groups and Large Language Model (LLM)-based agents, each evaluated with and without explicit communication. Cooperative resilience is assessed under a continuously disruptive condition induced by a persistent unsustainable consumption bot, together with intermittent environmental shocks implemented as stochastic removal of shared resources across scenarios. This experimental design establishes a benchmark for cooperative resilience across agent architectures and interaction modalities, constituting a key step toward systematically comparing humans and LLM-based agents. Using this framework, we find that human groups with communication achieve the highest cooperative resilience compared to all other groups. Communication also improves the resilience of LLM agents, but their performance remains below human levels. Motivated by the performance of humans, we further examine a long-horizon setting with harsher environmental conditions, where humans sustain the shared resource and maintain high resilience in diverse disruption scenarios. Together, these results suggest that human decision-making under adverse social conditions can inform the design of artificial agents that promote prosocial and resilient behaviors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11661v1">From Verification Burden to Trusted Collaboration: Design Goals for LLM-Assisted Literature Reviews</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly embedded in academic writing practices. Although numerous studies have explored how researchers employ these tools for scientific writing, their concrete implementation, limitations, and design challenges within the literature review process remain underexplored. In this paper, we report a user study with researchers across multiple disciplines to characterize current practices, benefits, and \textit{pain points} in using LLMs to investigate related work. We identified three recurring gaps: (i) lack of trust in outputs, (ii) persistent verification burden, and (iii) requiring multiple tools. This motivates our proposal of six design goals and a high-level framework that operationalizes them through improved related papers visualization, verification at every step, and human-feedback alignment with generation-guided explanations. Overall, by grounding our work in the practical, day-to-day needs of researchers, we designed a framework that addresses these limitations and models real-world LLM-assisted writing, advancing trust through verifiable actions and fostering practical collaboration between researchers and AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11550v1">PD-Swap: Prefill-Decode Logic Swapping for End-to-End LLM Inference on Edge FPGAs via Dynamic Partial Reconfiguration</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Aggressively quantized large language models (LLMs), such as BitNet-style 1.58-bit Transformers with ternary weights, make it feasible to deploy generative AI on low-power edge FPGAs. However, as prompts grow to tens of thousands of tokens, edge hardware performance drops sharply with sequence length due to quadratic prefill cost and rapidly increasing KV-cache bandwidth demands, making inference latency of longer context length a first-order system concern. Recent studies on LLMs expose a fundamental prefill-decode asymmetry: prefill is compute-bound and dominated by dense matrix-matrix operations, whereas decoding is memory-bandwidth-bound and dominated by KV-cache traffic. A static accelerator must provision resources and a single dataflow for both regimes, leading to duplicated attention logic, underutilized fabric, and tight LUT/URAM limits that cap model size and usable context. We propose a prefill--decode disaggregated LLM accelerator, PD-Swap, that uses Dynamic Partial Reconfiguration (DPR) to time-multiplex the attention module on edge FPGAs. The core table-lookup ternary matrix multiplication and weight-buffering engines remain static, while the attention subsystem is a reconfigurable partition with two phase-specialized architectures: a compute-heavy, token-parallel prefill engine and a bandwidth-optimized, KV-cache-centric decoding engine. A roofline-inspired model and design space exploration jointly optimize reconfigurable-region size, parallelism under reconfiguration and routability constraints, and reconfiguration latency is hidden by computation latency. PD-Swap achieves up to 27~tokens/s decoding throughput, outperforming prior state-of-the-art works by 1.3x--2.1x (larger gains at longer context lengths), without extra area cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11509v1">Does Less Hallucination Mean Less Creativity? An Empirical Investigation in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit remarkable capabilities in natural language understanding and reasoning, but suffer from hallucination: the generation of factually incorrect content. While numerous methods have been developed to reduce hallucinations, their impact on creative generations remains unexplored. This gap is particularly critical for AI-assisted scientific discovery, which requires both factual accuracy and creative hypothesis generation. We investigate how three hallucination-reduction techniques: Chain of Verification (CoVe), Decoding by Contrasting Layers (DoLa), and Retrieval-Augmented Generation (RAG), affect creativity in LLMs. Evaluating multiple model families (LLaMA, Qwen, Mistral) at varying scales (1B - 70B parameters) on two creativity benchmarks (NeoCoder and CS4), we find that these methods have opposing effects on divergent creativity. CoVe enhances divergent thinking, DoLa suppresses it, and RAG shows minimal impact. Our findings provide guidance for selecting appropriate hallucination-reduction methods in scientific applications, where the balance between factual accuracy and creative exploration is crucial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10493v2">Decoding Human-LLM Collaboration in Coding: An Empirical Study of Multi-Turn Conversations in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly acting as dynamic conversational interfaces, supporting multi-turn interactions that mimic human-like conversation and facilitate complex tasks like coding. While datasets such as LMSYS-Chat-1M and WildChat capture real-world user-LLM conversations, few studies systematically explore the mechanisms of human-LLM collaboration in coding scenarios. What tortuous paths do users experience during the interaction process? How well do the LLMs follow instructions? Are users satisfied? In this paper, we conduct an empirical analysis on human-LLM coding collaboration using LMSYS-Chat-1M and WildChat datasets to explore the human-LLM collaboration mechanism, LLMs' instruction following ability, and human satisfaction. This study yields interesting findings: 1) Task types shape interaction patterns(linear, star and tree), with code quality optimization favoring linear patterns, design-driven tasks leaning toward tree structures, and queries preferring star patterns; 2) Bug fixing and code refactoring pose greater challenges to LLMs' instruction following, with non-compliance rates notably higher than in information querying; 3) Code quality optimization and requirements-driven development tasks show lower user satisfaction, whereas structured knowledge queries and algorithm designs yield higher levels. These insights offer recommendations for improving LLM interfaces and user satisfaction in coding collaborations, while highlighting avenues for future research on adaptive dialogue systems. We believe this work broadens understanding of human-LLM synergies and supports more effective AI-assisted development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11421v1">Towards Trustworthy Multi-Turn LLM Agents via Behavioral Guidance</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 Accepted to AAAI 2026 Workshop on Trust and Control in Agentic AI (TrustAgent)
    </div>
    <details class="paper-abstract">
      Large Language Models demonstrate strong reasoning and generation abilities, yet their behavior in multi-turn tasks often lacks reliability and verifiability. We present a task completion framework that enables LLM-based agents to act under explicit behavioral guidance in environments described by reinforcement learning formalisms with defined observation, action, and reward signals. The framework integrates three components: a lightweight task profiler that selects reasoning and generation strategies, a reasoning module that learns verifiable observation - action mappings, and a generation module that enforces constraint-compliant outputs through validation or deterministic synthesis. We show that as the agent interacts with the environment, these components co-evolve, yielding trustworthy behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06393v2">Less Is More for Multi-Step Logical Reasoning of LLM Generalisation Under Rule Removal, Paraphrasing, and Compression</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve strong performance on many natural language tasks, yet their generalisation under structured perturbations of logical rule systems remains insufficiently characterised. We present a controlled evaluation framework that probes reasoning reliability through four stress tests: (1) rule deletion, removing redundant versus essential rules from a multi-step inference chain; (2) contradictory evidence injection; (3) logic-preserving rewrites based on equivalence laws (contraposition, double negation, implication-to-disjunction, De Morgan, identity, and commutativity); and (4) multi-law equivalence stacking that composes 2--5 transformations. Across three representative model families -- BERT, Qwen2, and LLaMA-like models -- all models attain Acc$=1.0000$ on the base split and show no degradation under redundant rule deletion. In contrast, essential rule deletion yields a pronounced decrease to near-chance performance, and injecting explicit contradictions reduces accuracy to 0.0000. Under logic-preserving rewrites, accuracy is largely preserved for single-law transformations with only small degradations in a few cases, whereas multi-law stacking exposes model-dependent sensitivity: BERT matches the base condition, TinyLlama shows only marginal degradation, and Qwen2 exhibits a substantial drop. Overall, the results indicate that contemporary LLMs are generally stable under semantic-preserving reformulations, yet remain brittle to missing or inconsistent evidence and may degrade under composed logical transformations depending on the model family. The proposed framework provides a concise diagnostic tool for isolating these failure modes and for evaluating logical generalisation beyond surface-form variation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11402v1">REMODEL-LLM: Transforming C code to Java using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      The automated translation of C code to Java code is a notoriously difficult task, fraught with challenges stemming from fundamental paradigm shifts (procedural vs. Object Oriented), memory models (manual pointers vs. Garbage Collection), and incompatible data types. This paper investigates the efficacy of 19 small, quantized LLMs (under 20 billion parameters) for the C to Java translation task. We use a novel, hybrid pipeline that leverages Abstract Syntax Trees (ASTs) for semantic decomposition and employs a highly constrained, rule based prompting strategy. The results are stark: a clear multi tiered performance divide emerged. The vast majority of models (Tier 3, e.g., llama3.1, gemma3, starcoder2) failed 100\% of the tests, proving incapable of generating even basic, runnable Java boilerplate. A small middle tier (Tier 2, e.g., mistral-nemo and mistral) produced runnable code but was plagued by dangerous semantic failures and wrong translations. Only three models (Tier 1: phi4, deepseek-coder-v2, codeqwen) proved viable, passing over 50\% of the test suite. Even these top models failed on the most complex C concepts, such as function pointers, sizeof, and enum logic, revealing a hard ceiling for the reasoning capabilities of current quantized models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.07020v2">Driving Through Uncertainty: Risk-Averse Control with LLM Commonsense for Autonomous Driving under Perception Deficits</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Partial perception deficits can compromise autonomous vehicle safety by disrupting environmental understanding. Existing protocols typically default to entirely risk-avoidant actions such as immediate stops, which are detrimental to navigation goals and lack flexibility for rare driving scenarios. Yet, in cases of minor risk, halting the vehicle may be unnecessary, and more adaptive responses are preferable. In this paper, we propose LLM-RCO, a risk-averse framework leveraging large language models (LLMs) to integrate human-like driving commonsense into autonomous systems facing perception deficits. LLM-RCO features four key modules interacting with the dynamic driving environment: hazard inference, short-term motion planner, action condition verifier, and safety constraint generator, enabling proactive and context-aware actions in such challenging conditions. To enhance the driving decision-making of LLMs, we construct DriveLM-Deficit, a dataset of 53,895 video clips featuring deficits of safety-critical objects, annotated for LLM fine-tuning in hazard detection and motion planning. Extensive experiments in adverse driving conditions with the CARLA simulator demonstrate that LLM-RCO promotes proactive maneuvers over purely risk-averse actions in perception deficit scenarios, underscoring its value for boosting autonomous driving resilience against perception loss challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11388v1">Improving Translation Quality by Selecting Better Data for LLM Fine-Tuning: A Comparative Analysis</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 To appear at IEEE Big Data 2025
    </div>
    <details class="paper-abstract">
      We investigated the impact of data selection on machine translation fine-tuning for open LLMs. Using Japanese-English corpora, we compare five selectors: TF-IDF, COMET Kiwi, QuRate, FD-Score, and random selection, under controlled training conditions. We observed that semantic selectors consistently outperform lexical and geometry-based heuristics, and that even when the selected data differ by less than 3%, the impact on model performance is substantial, underscoring the sensitivity of fine-tuning to data quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10501v2">Zero-shot 3D Map Generation with LLM Agents: A Dual-Agent Architecture for Procedural Content Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 12 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Procedural Content Generation (PCG) offers scalable methods for algorithmically creating complex, customizable worlds. However, controlling these pipelines requires the precise configuration of opaque technical parameters. We propose a training-free architecture that utilizes LLM agents for zero-shot PCG parameter configuration. While Large Language Models (LLMs) promise a natural language interface for PCG tools, off-the-shelf models often fail to bridge the semantic gap between abstract user instructions and strict parameter specifications. Our system pairs an Actor agent with a Critic agent, enabling an iterative workflow where the system autonomously reasons over tool parameters and refines configurations to progressively align with human design preferences. We validate this approach on the generation of various 3D maps, establishing a new benchmark for instruction-following in PCG. Experiments demonstrate that our approach outperforms single-agent baselines, producing diverse and structurally valid environments from natural language descriptions. These results demonstrate that off-the-shelf LLMs can be effectively repurposed as generalized agents for arbitrary PCG tools. By shifting the burden from model training to architectural reasoning, our method offers a scalable framework for mastering complex software without task-specific fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.17281v4">MemoryBench: A Benchmark for Memory and Continual Learning in LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Scaling up data, parameters, and test-time computation has been the mainstream methods to improve LLM systems (LLMsys), but their upper bounds are almost reached due to the gradual depletion of high-quality data and marginal gains obtained from larger computational resource consumption. Inspired by the abilities of human and traditional AI systems in learning from practice, constructing memory and continual learning frameworks for LLMsys has become an important and popular research direction in recent literature. Yet, existing benchmarks for LLM memory often focus on evaluating the system on homogeneous reading comprehension tasks with long-form inputs rather than testing their abilities to learn from accumulated user feedback in service time. Therefore, we propose a user feedback simulation framework and a comprehensive benchmark covering multiple domains, languages, and types of tasks to evaluate the continual learning abilities of LLMsys. Experiments show that the effectiveness and efficiency of state-of-the-art baselines are far from satisfying, and we hope this benchmark could pave the way for future studies on LLM memory and optimization algorithms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05578v2">The Landscape of Memorization in LLMs: Mechanisms, Measurement, and Mitigation</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks, yet they also exhibit memorization of their training data. This phenomenon raises critical questions about model behavior, privacy risks, and the boundary between learning and memorization. Addressing these concerns, this paper synthesizes recent studies and investigates the landscape of memorization, the factors influencing it, and methods for its detection and mitigation. We explore key drivers, including training data duplication, training dynamics, and fine-tuning procedures that influence data memorization. In addition, we examine methodologies such as prefix-based extraction, membership inference, and adversarial prompting, assessing their effectiveness in detecting and measuring memorized content. Beyond technical analysis, we also explore the broader implications of memorization, including the legal and ethical implications. Finally, we discuss mitigation strategies, including data cleaning, differential privacy, and post-training unlearning, while highlighting open challenges in balancing the need to minimize harmful memorization with model utility. This paper provides a comprehensive overview of the current state of research on LLM memorization across technical, privacy, and performance dimensions, identifying critical directions for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16368v3">SATURN: SAT-based Reinforcement Learning to Unleash LLMs Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      How to design reinforcement learning (RL) tasks that effectively unleash the reasoning capability of large language models (LLMs) remains an open question. Existing RL tasks (e.g., math, programming, and constructing reasoning tasks) suffer from three key limitations: (1) Scalability. They rely heavily on human annotation or expensive LLM synthesis to generate sufficient training data. (2) Verifiability. LLMs' outputs are hard to verify automatically and reliably. (3) Controllable Difficulty. Most tasks lack fine-grained difficulty control, making it hard to train LLMs to develop reasoning ability from easy to hard. To address these limitations, we propose Saturn, a SAT-based RL framework that uses Boolean Satisfiability (SAT) problems to train and evaluate LLMs reasoning. Saturn enables scalable task construction, rule-based verification, and precise difficulty control. Saturn designs a curriculum learning pipeline that continuously improves LLMs' reasoning capability by constructing SAT tasks of increasing difficulty and training LLMs from easy to hard. To ensure stable training, we design a principled mechanism to control difficulty transitions. We introduce Saturn-2.6k, a dataset of 2,660 SAT problems with varying difficulty. It supports the evaluation of how LLM reasoning changes with problem difficulty. We apply Saturn to DeepSeek-R1-Distill-Qwen and obtain Saturn-1.5B and Saturn-7B. We achieve several notable results: (1) On SAT problems, Saturn-1.5B and Saturn-7B achieve average pass@3 improvements of +14.0 and +28.1, respectively. (2) On math and programming tasks, Saturn-1.5B and Saturn-7B improve average scores by +4.9 and +1.8 on benchmarks (e.g., AIME, LiveCodeBench). (3) Compared to the state-of-the-art (SOTA) approach in constructing RL tasks, Saturn achieves further improvements of +8.8%. We release the source code, data, and models to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09972v2">BAMBO: Construct Ability and Efficiency LLM Pareto Set via Bayesian Adaptive Multi-objective Block-wise Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Constructing a Pareto set is pivotal for navigating the capability-efficiency trade-offs in Large Language Models (LLMs); however, existing merging techniques remain inadequate for this task. Coarse-grained, model-level methods yield only a sparse set of suboptimal solutions, while fine-grained, layer-wise approaches suffer from the "curse of dimensionality," rendering the search space computationally intractable. To resolve this dichotomy, we propose BAMBO (Bayesian Adaptive Multi-objective Block-wise Optimization), a novel framework that automatically constructs the LLM Pareto set. BAMBO renders the search tractable by introducing a Hybrid Optimal Block Partitioning strategy. Formulated as a 1D clustering problem, this strategy leverages a dynamic programming approach to optimally balance intra-block homogeneity and inter-block information distribution, thereby dramatically reducing dimensionality without sacrificing critical granularity. The entire process is automated within an evolutionary loop driven by the q-Expected Hypervolume Improvement (qEHVI) acquisition function. Experiments demonstrate that BAMBO discovers a superior and more comprehensive Pareto frontier than baselines, enabling agile model selection tailored to diverse operational constraints. Code is available at: https://github.com/xin8coder/BAMBO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11270v1">A-LAMP: Agentic LLM-Based Framework for Automated MDP Modeling and Policy Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 NeurIPS 2025 Workshop: Multi-Turn Interactions in Large Language Models. 26 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Applying reinforcement learning (RL) to real-world tasks requires converting informal descriptions into a formal Markov decision process (MDP), implementing an executable environment, and training a policy agent. Automating this process is challenging due to modeling errors, fragile code, and misaligned objectives, which often impede policy training. We introduce an agentic large language model (LLM)-based framework for automated MDP modeling and policy generation (A-LAMP), that automatically translates free-form natural language task descriptions into an MDP formulation and trained policy. The framework decomposes modeling, coding, and training into verifiable stages, ensuring semantic alignment throughout the pipeline. Across both classic control and custom RL domains, A-LAMP consistently achieves higher policy generation capability than a single state-of-the-art LLM model. Notably, even its lightweight variant, which is built on smaller language models, approaches the performance of much larger models. Failure analysis reveals why these improvements occur. In addition, a case study also demonstrates that A-LAMP generates environments and policies that preserve the task's optimality, confirming its correctness and reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11261v1">Leveraging LLMs for Title and Abstract Screening for Systematic Review: A Cost-Effective Dynamic Few-Shot Learning Approach</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 22 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Systematic reviews are a key component of evidence-based medicine, playing a critical role in synthesizing existing research evidence and guiding clinical decisions. However, with the rapid growth of research publications, conducting systematic reviews has become increasingly burdensome, with title and abstract screening being one of the most time-consuming and resource-intensive steps. To mitigate this issue, we designed a two-stage dynamic few-shot learning (DFSL) approach aimed at improving the efficiency and performance of large language models (LLMs) in the title and abstract screening task. Specifically, this approach first uses a low-cost LLM for initial screening, then re-evaluates low-confidence instances using a high-performance LLM, thereby enhancing screening performance while controlling computational costs. We evaluated this approach across 10 systematic reviews, and the results demonstrate its strong generalizability and cost-effectiveness, with potential to reduce manual screening burden and accelerate the systematic review process in practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00862v3">HBLLM: Wavelet-Enhanced High-Fidelity 1-Bit Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      We introduce HBLLM, a wavelet-enhanced high-fidelity $1$-bit post-training quantization method for Large Language Models (LLMs). By leveraging Haar wavelet transforms to enhance expressive capacity through frequency decomposition, HBLLM significantly improves quantization fidelity while maintaining minimal overhead. This approach features two innovative structure-aware grouping strategies: (1) frequency-aware multi-parameter intra-row grouping and (2) $\ell_2$-norm-based saliency-driven column selection. For non-salient weights, a shared mean is employed across quantization groups within each frequency band to optimize storage efficiency. Experiments conducted on the OPT and LLaMA models demonstrate that HBLLM achieves state-of-the-art performance in $1$-bit quantization, attaining a perplexity of $6.71$ on LLaMA$2$-$13$B with an average weight storage of only $1.08$ bits. Code available at: https://github.com/Yeyke/HBLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11221v1">Adaptive Soft Rolling KV Freeze with Entropy-Guided Recovery: Sublinear Memory Growth for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 6 pages, 3 tables , 1 figure
    </div>
    <details class="paper-abstract">
      We present Adaptive Soft Rolling KV Freeze with Entropy-Guided Recovery (ASR-KF-EGR), a training-free inference-time framework for efficient large language model generation. Our method introduces a reversible soft-freeze mechanism that temporarily suspends key-value (KV) updates for low-importance tokens identified within a sliding attention window. Unlike eviction-based approaches that permanently discard context, ASR-KF-EGR preserves all tokens in off-GPU storage and restores them on demand. We extend the framework with sublinear freeze scheduling, where freeze duration grows sublinearly with repeated low-importance detections, preventing over-aggressive compression. Preliminary experiments on LLaMA-3 8B demonstrate 55-67% reduction in active KV cache size while maintaining generation quality and passing needle-in-haystack retrieval tests. The method is architecture-agnostic, requires no fine-tuning, and provides a practical solution for memory-constrained deployment of long-context LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10257v2">Reject or Not?: A Benchmark for Voice Assistant Query Rejection in Smart Home Scenario and an Improved Method Based on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      In smart-home voice assistant scenario, deciding whether to accept or reject a user query is the first step before any downstream processing. To address the limited query-rejection capability of current voice assistants, this paper presents the first Chinese-oriented open-source benchmark and evaluation suite for smart homes, together with a personalized query-rejection method based on large language models. On the data side, we construct the first multimodal query-rejection dataset tailored for domestic scenarios, containing 11,913 manually labeled text-speech pairs that systematically cover twelve typical dialogue types (e.g., chit-chat, non-human sounds, valid commands, ambiguous references, device-irrelevant requests). Fine-grained labels, conversational context and multi-turn information are provided to support both zero-shot and fine-tuning evaluations across language and multimodal large models. On the method side, we propose a three-tier collaborative architecture: first, a Qwen-2.5-3B adapter fine-tuned to model family-agnostic semantic boundaries; second, a dynamic household-level historical dialogue module to capture personalized habits; third, a household-specific RAG knowledge base that explicitly memorizes and revises past false-rejection cases. Experiments show that the proposed approach significantly outperforms zero-shot and fine-tuned general LLMs on the constructed dataset, with pronounced gains in rejection accuracy for family-specific expressions and complex multi-turn scenarios. This work provides a reproducible data foundation, evaluation standard and extensible technical framework for reliability research in smart-home voice interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.23453v2">Counterfactual Evaluation for Blind Attack Detection in LLM-based Evaluation Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      This paper investigates defenses for LLM-based evaluation systems against prompt injection. We formalize a class of threats called blind attacks, where a candidate answer is crafted independently of the true answer to deceive the evaluator. To counter such attacks, we propose a framework that augments Standard Evaluation (SE) with Counterfactual Evaluation (CFE), which re-evaluates the submission against a deliberately false ground-truth answer. An attack is detected if the system validates an answer under both standard and counterfactual conditions. Experiments show that while standard evaluation is highly vulnerable, our SE+CFE framework significantly improves security by boosting attack detection with minimal performance trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.03846v3">Do LLM Evaluators Prefer Themselves for a Reason?</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 Added LMArena experiments
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as automatic evaluators in applications such as benchmarking, reward modeling, and self-refinement. Prior work highlights a potential self-preference bias where LLMs favor their own generated responses, a tendency often intensifying with model size and capability. This raises a critical question: Is self-preference harmful, or does it simply reflect the genuinely higher-quality outputs of stronger models? Answering this has been difficult as prior works mostly relied on subjective tasks that lack an objective ground truth, meaning that either preference can be reasonably justified. To address this ambiguity, we investigate self-preference using verifiable benchmarks (mathematical reasoning, factual knowledge, code generation) that allow objective ground-truth assessment. This enables us to distinguish harmful (favoring objectively worse responses) from legitimate (favoring genuinely superior ones) self-preference. Our large-scale experiments across 7 model families reveal three key findings: (1) While stronger models exhibit greater self-preference, much of this preference aligns with objectively superior performance, indicating stronger models prefer themselves mostly legitimately. (2) Harmful self-preference persists when evaluator models err as generators, and stronger models display more pronounced harmful self-preference bias when they do err. This suggests stronger models struggle more to recognize when they are wrong. (3) Inference-time scaling strategies, such as generating a long Chain-of-Thought before evaluation, effectively reduce the harmful self-preference. Additionally, we experiment with LMArena and show that our findings extend beyond verifiable benchmarks to real-world, subjective domains. These results provide a more nuanced understanding of LLM-based evaluation and practical insights for improving its reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12072v1">VOYAGER: A Training Free Approach for Generating Diverse Datasets using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 Arxiv Submission
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly being used to generate synthetic datasets for the evaluation and training of downstream models. However, prior work has noted that such generated data lacks diversity. In this paper, we propose Voyager, a novel principled approach to generate diverse datasets. Our approach is iterative and directly optimizes a mathematical quantity that optimizes the diversity of the dataset using the machinery of determinantal point processes. Furthermore, our approach is training-free, applicable to closed-source models, and scalable. In addition to providing theoretical justification for the working of our method, we also demonstrate through comprehensive experiments that Voyager significantly outperforms popular baseline approaches by providing a 1.5-3x improvement in diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25540v2">RadOnc-GPT: An Autonomous LLM Agent for Real-Time Patient Outcomes Labeling at Scale</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Manual labeling limits the scale, accuracy, and timeliness of patient outcomes research in radiation oncology. We present RadOnc-GPT, an autonomous large language model (LLM)-based agent capable of independently retrieving patient-specific information, iteratively assessing evidence, and returning structured outcomes. Our evaluation explicitly validates RadOnc-GPT across two clearly defined tiers of increasing complexity: (1) a structured quality assurance (QA) tier, assessing the accurate retrieval of demographic and radiotherapy treatment plan details, followed by (2) a complex clinical outcomes labeling tier involving determination of mandibular osteoradionecrosis (ORN) in head-and-neck cancer patients and detection of cancer recurrence in independent prostate and head-and-neck cancer cohorts requiring combined interpretation of structured and unstructured patient data. The QA tier establishes foundational trust in structured-data retrieval, a critical prerequisite for successful complex clinical outcome labeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.15455v2">CoopQ: Cooperative Game Inspired Layerwise Mixed Precision Quantization for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) promise impressive capabilities, yet their multi-billion-parameter scale makes on-device or low-resource deployment prohibitive. Mixed-precision quantization offers a compelling solution, but existing methods struggle when the average precision drops below four bits, as they rely on isolated, layer-specific metrics that overlook critical inter-layer interactions affecting overall performance. To address these limitations, we first frame the mixed-precision quantization problem as a cooperative game among layers and introduce Shapley-based Progressive Quantization Estimation (SPQE) to efficiently obtain accurate Shapley estimates of layer sensitivities and inter-layer interactions. Leveraging the SPQE estimates, we propose Cooperative Game Inspired Mixed-Precision Quantization (CoopQ) which translates these Shapley estimates into a binary quadratic optimization formulation, assigning either 2 or 4-bit precision to layers under strict memory constraints. Comprehensive experiments conducted on Llama-3, Gemma-2, and Qwen-3 models across three independent PTQ backends (Quanto, HQQ, GPTQ) demonstrate CoopQ's scalability and consistently superior performance compared to methods relying solely on isolated metrics. Across average precisions spanning 4 bit down to 2 bit, CoopQ cuts Perplexity by 20 - 80 % relative to the best baseline, with the margin growing as the bit-width tightens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.02718v3">Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-12
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Increasing demand for Large Language Models (LLMs) services imposes substantial deployment and computation costs on providers. LLM routing offers a cost-efficient solution by directing queries to the optimal LLM based on model and query features. However, existing works primarily focus on offline scenarios and struggle to adapt to online settings with high query volume and constrained token budgets. In this work, we introduce the first training-free algorithm for online routing scenarios. Our algorithm leverages approximate nearest neighbor search to efficiently estimate query features and performs a one-time optimization over a small set of initial queries to learn a routing strategy that guides future routing. We provide theoretical guarantees demonstrating that our algorithm achieves a competitive ratio of $1 - o(1)$ under natural assumptions, which is further validated by extensive experiments across 3 benchmark datasets and 8 baselines, showing an average improvement of 3.55$\times$ in overall performance, 1.85$\times$ in cost efficiency, and nearly 4.25$\times$ in throughput. Our code is available at https://github.com/fzwark/PORT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11986v1">Learning to Extract Context for Context-Aware LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-12
    </div>
    <details class="paper-abstract">
      User prompts to large language models (LLMs) are often ambiguous or under-specified, and subtle contextual cues shaped by user intentions, prior knowledge, and risk factors strongly influence what constitutes an appropriate response. Misinterpreting intent or risks may lead to unsafe outputs, while overly cautious interpretations can cause unnecessary refusal of benign requests. In this paper, we question the conventional framework in which LLMs generate immediate responses to requests without considering broader contextual factors. User requests are situated within broader contexts such as intentions, knowledge, and prior experience, which strongly influence what constitutes an appropriate answer. We propose a framework that extracts and leverages such contextual information from the user prompt itself. Specifically, a reinforcement learning based context generator, designed in an autoencoder-like fashion, is trained to infer contextual signals grounded in the prompt and use them to guide response generation. This approach is particularly important for safety tasks, where ambiguous requests may bypass safeguards while benign but confusing requests can trigger unnecessary refusals. Experiments show that our method reduces harmful responses by an average of 5.6% on the SafetyInstruct dataset across multiple foundation models and improves the harmonic mean of attack success rate and compliance on benign prompts by 6.2% on XSTest and WildJailbreak. These results demonstrate the effectiveness of context extraction for safer and more reliable LLM inferences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10611v1">Phythesis: Physics-Guided Evolutionary Scene Synthesis for Energy-Efficient Data Center Design via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Data center (DC) infrastructure serves as the backbone to support the escalating demand for computing capacity. Traditional design methodologies that blend human expertise with specialized simulation tools scale poorly with the increasing system complexity. Recent studies adopt generative artificial intelligence to design plausible human-centric indoor layouts. However, they do not consider the underlying physics, making them unsuitable for the DC design that sets quantifiable operational objectives and strict physical constraints. To bridge the gap, we propose Phythesis, a novel framework that synergizes large language models (LLMs) and physics-guided evolutionary optimization to automate simulation-ready (SimReady) scene synthesis for energy-efficient DC design. Phythesis employs an iterative bi-level optimization architecture, where (i) the LLM-driven optimization level generates physically plausible three-dimensional layouts and self-criticizes them to refine the scene topology, and (ii) the physics-informed optimization level identifies the optimal asset parameters and selects the best asset combination. Experiments on three generation scales show that Phythesis achieves 57.3% generation success rate increase and 11.5% power usage effectiveness (PUE) improvement, compared with the vanilla LLM-based solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10610v1">Thinking While Driving: A Concurrent Framework for Real-Time, LLM-Based Adaptive Routing</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      We present Thinking While Driving, a concurrent routing framework that integrates LLMs into a graph-based traffic environment. Unlike approaches that require agents to stop and deliberate, our system enables LLM-based route planning while agents are moving, significantly reducing intersection wait times. Under high traffic, agents average just 0.75 seconds of decision latency. To coordinate many agents in real-time, we implement a non-blocking asynchronous architecture using Unity coroutines and a dedicated request manager. The environment is a weighted undirected graph with live congestion metrics, updated continuously by the agents to enable shared perception. Our results show LLM-driven agents can dynamically adapt to traffic, reroute around congestion, and exhibit behaviors beyond static pathfinding, all while maintaining real-time performance. This work provides a reproducible framework for future research in adaptive routing and multi-agent cooperation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10551v1">LLM-Auction: Generative Auction towards LLM-Native Advertising</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) necessitates novel monetization strategies, among which LLM-native advertising has emerged as a promising paradigm by naturally integrating advertisement within LLM-generated responses. However, this paradigm fundamentally shifts the auction object from discrete ad slots to the distribution over LLM outputs, posing new challenges for designing auction mechanisms. Existing mechanisms for LLM-native advertising adopt frameworks that decouple auction and generation, which either ignore externalities or require multiple LLM inferences for ad allocation, rendering them impractical for industrial scenarios. To address these challenges, we propose LLM-Auction, which to the best of our knowledge is the first learning-based generative auction mechanism that integrates auction and LLM generation for LLM-native advertising. By formulating the allocation optimization as a preference alignment problem between LLM outputs and the mechanism's objective which reflects both advertisers' expected value and user experience, we introduce Iterative Reward-Preference Optimization (IRPO) algorithm that alternately optimizes the reward model and the LLM. This approach enables the LLM to inherently model allocation externalities without any extra inference cost. We further identify the allocation monotonicity and continuity of LLM-Auction, which allows us to prove that a simple first-price payment rule exhibits favorable incentive properties. Additionally, we design an LLM-as-a-judge simulation environment to facilitate large-scale data construction and enable comprehensive quantitative evaluation of the mechanism's performance. Extensive quantitative and qualitative experiments demonstrate that LLM-Auction significantly outperforms existing baselines in allocation efficiency, while achieving the desired mechanism properties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10545v1">XDoGE: Multilingual Data Reweighting to Enhance Language Inclusivity in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Accepted and presented at the LLMs4All workshop at the IEEE BigData 2025 Conference, Macau - December 8-11, 2025
    </div>
    <details class="paper-abstract">
      Current large language models (LLMs) are trained on massive amounts of text data, primarily from a few dominant languages. Studies suggest that this over-reliance on high-resource languages, such as English, hampers LLM performance in mid- and low-resource languages. To mitigate this problem, we propose to (i) optimize the language distribution by training a small proxy model within a domain-reweighing DoGE algorithm that we extend to XDoGE for a multilingual setup, and (ii) rescale the data and train a full-size model with the established language weights either from scratch or within a continual pre-training phase (CPT). We target six languages possessing a variety of geographic and intra- and inter-language-family relations, namely, English and Spanish (high-resource), Portuguese and Catalan (mid-resource), Galician and Basque (low-resource). We experiment with Salamandra-2b, which is a promising model for these languages. We investigate the effects of substantial data repetition on minor languages and under-sampling on dominant languages using the IberoBench framework for quantitative evaluation. Finally, we release a new promising IberianLLM-7B-Instruct model centering on Iberian languages and English that we pretrained from scratch and further improved using CPT with the XDoGE weights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10501v1">Zero-shot 3D Map Generation with LLM Agents: A Dual-Agent Architecture for Procedural Content Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Procedural Content Generation (PCG) offers scalable methods for algorithmically creating complex, customizable worlds. However, controlling these pipelines requires the precise configuration of opaque technical parameters. We propose a training-free architecture that utilizes LLM agents for zero-shot PCG parameter configuration. While Large Language Models (LLMs) promise a natural language interface for PCG tools, off-the-shelf models often fail to bridge the semantic gap between abstract user instructions and strict parameter specifications. Our system pairs an Actor agent with a Critic agent, enabling an iterative workflow where the system autonomously reasons over tool parameters and refines configurations to progressively align with human design preferences. We validate this approach on the generation of various 3D maps, establishing a new benchmark for instruction-following in PCG. Experiments demonstrate that our approach outperforms single-agent baselines, producing diverse and structurally valid environments from natural language descriptions. These results demonstrate that off-the-shelf LLMs can be effectively repurposed as generalized agents for arbitrary PCG tools. By shifting the burden from model training to architectural reasoning, our method offers a scalable framework for mastering complex software without task-specific fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10493v1">Decoding Human-LLM Collaboration in Coding: An Empirical Study of Multi-Turn Conversations in the Wild</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly acting as dynamic conversational interfaces, supporting multi-turn interactions that mimic human-like conversation and facilitate complex tasks like coding. While datasets such as LMSYS-Chat-1M and WildChat capture real-world user-LLM conversations, few studies systematically explore the mechanisms of human-LLM collaboration in coding scenarios. What tortuous paths do users experience during the interaction process? How well do the LLMs follow instructions? Are users satisfied? In this paper, we conduct an empirical analysis on human-LLM coding collaboration using LMSYS-Chat-1M and WildChat datasets to explore the human-LLM collaboration mechanism, LLMs' instruction following ability, and human satisfaction. This study yields interesting findings: 1) Task types shape interaction patterns(linear, star and tree), with code quality optimization favoring linear patterns, design-driven tasks leaning toward tree structures, and queries preferring star patterns; 2) Bug fixing and code refactoring pose greater challenges to LLMs' instruction following, with non-compliance rates notably higher than in information querying; 3) Code quality optimization and requirements-driven development tasks show lower user satisfaction, whereas structured knowledge queries and algorithm designs yield higher levels. These insights offer recommendations for improving LLM interfaces and user satisfaction in coding collaborations, while highlighting avenues for future research on adaptive dialogue systems. We believe this work broadens understanding of human-LLM synergies and supports more effective AI-assisted development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10487v1">LLM-Assisted AHP for Explainable Cyber Range Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Cyber Ranges (CRs) have emerged as prominent platforms for cybersecurity training and education, especially for Critical Infrastructure (CI) sectors that face rising cyber threats. One way to address these threats is through hands-on exercises that bridge IT and OT domains to improve defensive readiness. However, consistently evaluating whether a CR platform is suitable and effective remains a challenge. This paper proposes an evaluation framework for CRs, emphasizing mission-critical settings by using a multi-criteria decision-making approach. We define a set of evaluation criteria that capture technical fidelity, training and assessment capabilities, scalability, usability, and other relevant factors. To weight and aggregate these criteria, we employ the Analytic Hierarchy Process (AHP), supported by a simulated panel of multidisciplinary experts implemented through a Large Language Model (LLM). This LLM-assisted expert reasoning enables consistent and reproducible pairwise comparisons across criteria without requiring direct expert convening. The framework's output equals quantitative scores that facilitate objective comparison of CR platforms and highlight areas for improvement. Overall, this work lays the foundation for a standardized and explainable evaluation methodology to guide both providers and end-users of CRs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10485v1">From Lab to Reality: A Practical Evaluation of Deep Learning Models and LLMs for Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Vulnerability detection methods based on deep learning (DL) have shown strong performance on benchmark datasets, yet their real-world effectiveness remains underexplored. Recent work suggests that both graph neural network (GNN)-based and transformer-based models, including large language models (LLMs), yield promising results when evaluated on curated benchmark datasets. These datasets are typically characterized by consistent data distributions and heuristic or partially noisy labels. In this study, we systematically evaluate two representative DL models-ReVeal and LineVul-across four representative datasets: Juliet, Devign, BigVul, and ICVul. Each model is trained independently on each respective dataset, and their code representations are analyzed using t-SNE to uncover vulnerability related patterns. To assess realistic applicability, we deploy these models along with four pretrained LLMs, Claude 3.5 Sonnet, GPT-o3-mini, GPT-4o, and GPT-5 on a curated dataset, VentiVul, comprising 20 recently (May 2025) fixed vulnerabilities from the Linux kernel. Our experiments reveal that current models struggle to distinguish vulnerable from non-vulnerable code in representation space and generalize poorly across datasets with differing distributions. When evaluated on VentiVul, our newly constructed time-wise out-of-distribution dataset, performance drops sharply, with most models failing to detect vulnerabilities reliably. These results expose a persistent gap between academic benchmarks and real-world deployment, emphasizing the value of our deployment-oriented evaluation framework and the need for more robust code representations and higher-quality datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.17552v3">Can LLMs Reason Over Non-Text Modalities in a Training-Free Manner? A Case Study with In-Context Representation Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The remarkable performance of Large Language Models (LLMs) can be enhanced with test-time computation, which relies on external tools and even other deep learning models. However, existing approaches for integrating non-text modality representations into LLMs typically require additional costly supervised training, restricting on-the-fly adaptation to new domains and modalities. In this work, we explore the feasibility of integrating representations from non-text foundational models (FMs) into text-based LLMs in a training-free manner. We propose In-Context Representation Learning (ICRL) as a proof-of-concept to allow LLMs to adaptively utilize non-text modality representations with few-shot learning. Unlike traditional in-context learning, which incorporates text-label pairs, ICRL replaces text inputs with FM representations, enabling the LLM to perform multi-modal inference without fine-tuning. We evaluate ICRL on a suite of tasks in the molecular domain, investigating three core research questions: (i) how to map FM representations into LLMs in a training-free manner, (ii) what factors influence ICRL performance, and (iii) what mechanisms underlie the effectiveness of ICRL. To the best of our knowledge, ICRL is the first training-free framework for integrating non-text modality representations into text-based LLMs, presenting a promising direction for adaptable, multi-modal generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10453v1">Grammaticality Judgments in Humans and Language Models: Revisiting Generative Grammar with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 2 figures
    </div>
    <details class="paper-abstract">
      What counts as evidence for syntactic structure? In traditional generative grammar, systematic contrasts in grammaticality such as subject-auxiliary inversion and the licensing of parasitic gaps are taken as evidence for an internal, hierarchical grammar. In this paper, we test whether large language models (LLMs), trained only on surface forms, reproduce these contrasts in ways that imply an underlying structural representation. We focus on two classic constructions: subject-auxiliary inversion (testing recognition of the subject boundary) and parasitic gap licensing (testing abstract dependency structure). We evaluate models including GPT-4 and LLaMA-3 using prompts eliciting acceptability ratings. Results show that LLMs reliably distinguish between grammatical and ungrammatical variants in both constructions, and as such support that they are sensitive to structure and not just linear order. Structural generalizations, distinct from cognitive knowledge, emerge from predictive training on surface forms, suggesting functional sensitivity to syntax without explicit encoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10449v1">When Reject Turns into Accept: Quantifying the Vulnerability of LLM-Based Scientific Reviewers to Indirect Prompt Injection</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      The landscape of scientific peer review is rapidly evolving with the integration of Large Language Models (LLMs). This shift is driven by two parallel trends: the widespread individual adoption of LLMs by reviewers to manage workload (the "Lazy Reviewer" hypothesis) and the formal institutional deployment of AI-powered assessment systems by conferences like AAAI and Stanford's Agents4Science. This study investigates the robustness of these "LLM-as-a-Judge" systems (both illicit and sanctioned) to adversarial PDF manipulation. Unlike general jailbreaks, we focus on a distinct incentive: flipping "Reject" decisions to "Accept," for which we develop a novel evaluation metric which we term as WAVS (Weighted Adversarial Vulnerability Score). We curated a dataset of 200 scientific papers and adapted 15 domain-specific attack strategies to this task, evaluating them across 13 Language Models, including GPT-5, Claude Haiku, and DeepSeek. Our results demonstrate that obfuscation strategies like "Maximum Mark Magyk" successfully manipulate scores, achieving alarming decision flip rates even in large-scale models. We will release our complete dataset and injection framework to facilitate more research on this topic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10415v1">How to Trick Your AI TA: A Systematic Study of Academic Jailbreaking in LLM Code Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      The use of Large Language Models (LLMs) as automatic judges for code evaluation is becoming increasingly prevalent in academic environments. But their reliability can be compromised by students who may employ adversarial prompting strategies in order to induce misgrading and secure undeserved academic advantages. In this paper, we present the first large-scale study of jailbreaking LLM-based automated code evaluators in academic context. Our contributions are: (i) We systematically adapt 20+ jailbreaking strategies for jailbreaking AI code evaluators in the academic context, defining a new class of attacks termed academic jailbreaking. (ii) We release a poisoned dataset of 25K adversarial student submissions, specifically designed for the academic code-evaluation setting, sourced from diverse real-world coursework and paired with rubrics and human-graded references, and (iii) In order to capture the multidimensional impact of academic jailbreaking, we systematically adapt and define three jailbreaking metrics (Jailbreak Success Rate, Score Inflation, and Harmfulness). (iv) We comprehensively evalulate the academic jailbreaking attacks using six LLMs. We find that these models exhibit significant vulnerability, particularly to persuasive and role-play-based attacks (up to 97% JSR). Our adversarial dataset and benchmark suite lay the groundwork for next-generation robust LLM-based evaluators in academic code assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10370v1">LLM-Empowered Representation Learning for Emerging Item Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      In this work, we tackle the challenge of recommending emerging items, whose interactions gradually accumulate over time. Existing methods often overlook this dynamic process, typically assuming that emerging items have few or even no historical interactions. Such an assumption oversimplifies the problem, as a good model must preserve the uniqueness of emerging items while leveraging their shared patterns with established ones. To address this challenge, we propose EmerFlow, a novel LLM-empowered representation learning framework that generates distinctive embeddings for emerging items. It first enriches the raw features of emerging items through LLM reasoning, then aligns these representations with the embedding space of the existing recommendation model. Finally, new interactions are incorporated through meta-learning to refine the embeddings. This enables EmerFlow to learn expressive embeddings for emerging items from only limited interactions. Extensive experiments across diverse domains, including movies and pharmaceuticals, show that EmerFlow consistently outperforms existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.16528v2">Aligning ASR Evaluation with Human and LLM Judgments: Intelligibility Metrics Using Phonetic, Semantic, and NLI Approaches</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 5 pages, 2 figures, Interspeech 2025
    </div>
    <details class="paper-abstract">
      Traditional ASR metrics like WER and CER fail to capture intelligibility, especially for dysarthric and dysphonic speech, where semantic alignment matters more than exact word matches. ASR systems struggle with these speech types, often producing errors like phoneme repetitions and imprecise consonants, yet the meaning remains clear to human listeners. We identify two key challenges: (1) Existing metrics do not adequately reflect intelligibility, and (2) while LLMs can refine ASR output, their effectiveness in correcting ASR transcripts of dysarthric speech remains underexplored. To address this, we propose a novel metric integrating Natural Language Inference (NLI) scores, semantic similarity, and phonetic similarity. Our ASR evaluation metric achieves a 0.890 correlation with human judgments on Speech Accessibility Project data, surpassing traditional methods and emphasizing the need to prioritize intelligibility over error-based measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.20068v2">JITServe: SLO-aware LLM Serving with Imprecise Request Information</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into applications ranging from interactive chatbots to multi-agent systems has introduced a wide spectrum of service-level objectives (SLOs) for responsiveness. These include latency-sensitive requests emphasizing per-token latency in streaming chat, deadline-sensitive requests requiring rapid full responses to trigger external tools, and compound requests with evolving dependencies across multiple LLM calls. Despite-or perhaps, because of-this workload diversity and unpredictable request information (e.g., response lengths and dependencies), existing request schedulers have focused on aggregate performance, unable to ensure application-level SLO needs. This paper presents JITServe, the first SLO-aware LLM serving system designed to maximize service goodput (e.g., the number of tokens meeting request SLOs) across diverse workloads. JITServe novelly schedules requests using imprecise request information and gradually relaxes this conservatism by refining request information estimates as generation progresses. It applies a grouped margin goodput maximization algorithm to allocate just enough serving bandwidth to satisfy each request's SLO just-in-time (JIT), maximizing residual capacity for others, while deciding the composition of requests in a batch to maximize efficiency and goodput with provable guarantees. Our evaluation across diverse realistic workloads, including chat, deep research, and agentic pipelines, shows that JITServe improves service goodput by 1.4x-6.3x, alternatively achieving 28.5%-83.2% resource savings, compared to state-of-the-art designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10324v1">EchoingPixels: Cross-Modal Adaptive Token Reduction for Efficient Audio-Visual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Audio-Visual Large Language Models (AV-LLMs) face prohibitive computational overhead from massive audio and video tokens. Token reduction, while extensively explored for video-only LLMs, is insufficient for the audio-visual domain, as these unimodal methods cannot leverage audio-visual cross-modal synergies. Furthermore, the distinct and dynamic information densities of audio and video render static budgets per modality suboptimal. How to perform token reduction on a joint audio-visual stream thus remains an unaddressed bottleneck. To fill this gap, we introduce EchoingPixels, a framework inspired by the coexistence and interaction of visuals and sound in real-world scenes. The core of our framework is the Cross-Modal Semantic Sieve (CS2), a module enabling early audio-visual interaction. Instead of compressing modalities independently, CS2 co-attends to the joint multimodal stream and reduces tokens from an entire combined pool of audio-visual tokens rather than using fixed budgets per modality. This single-pool approach allows it to adaptively allocate the token budget across both modalities and dynamically identify salient tokens in concert. To ensure this aggressive reduction preserves the vital temporal modeling capability, we co-design a Synchronization-Augmented RoPE (Sync-RoPE) to maintain critical temporal relationships for the sparsely selected tokens. Extensive experiments demonstrate that EchoingPixels achieves performance comparable to strong baselines using only 5-20% of the original tokens, with a 2-3x speedup and memory reduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.15257v2">When Less Language is More: Language-Reasoning Disentanglement Makes LLMs Better Multilingual Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 NeurIPS 2025 Camera-ready
    </div>
    <details class="paper-abstract">
      Multilingual reasoning remains a significant challenge for large language models (LLMs), with performance disproportionately favoring high-resource languages. Drawing inspiration from cognitive neuroscience, which suggests that human reasoning functions largely independently of language processing, we hypothesize that LLMs similarly encode reasoning and language as separable components that can be disentangled to enhance multilingual reasoning. To evaluate this, we perform a causal intervention by ablating language-specific representations at inference time. Experiments on 10 open-weight LLMs spanning 11 typologically diverse languages show that this language-specific ablation consistently boosts multilingual reasoning performance. Layer-wise analyses further confirm that language and reasoning representations can be effectively disentangled throughout the model, yielding improved multilingual reasoning capabilities, while preserving top-layer language features remains essential for maintaining linguistic fidelity. Compared to post-training methods such as supervised fine-tuning or reinforcement learning, our training-free language-reasoning disentanglement achieves comparable or superior results with minimal computational overhead. These findings shed light on the internal mechanisms underlying multilingual reasoning in LLMs and suggest a lightweight and interpretable strategy for improving cross-lingual generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17208v2">A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      LLM-based conversational agents still struggle to maintain coherent, personalized interaction over many sessions: fixed context windows limit how much history can be kept in view, and most external memory approaches trade off between coarse retrieval over large chunks and fine-grained but fragmented views of the dialogue. Motivated by neo-Davidsonian event semantics, we propose an event-centric alternative that represents conversational history as short, event-like propositions which bundle together participants, temporal cues, and minimal local context, rather than as independent relation triples or opaque summaries. In contrast to work that aggressively compresses or forgets past content, our design aims to preserve information in a non-compressive form and make it more accessible, rather than more lossy. Concretely, we instruct an LLM to decompose each session into enriched elementary discourse units (EDUs) -- self-contained statements with normalized entities and source turn attributions -- and organize sessions, EDUs, and their arguments in a heterogeneous graph that supports associative recall. On top of this representation we build two simple retrieval-based variants that use dense similarity search and LLM filtering, with an optional graph-based propagation step to connect and aggregate evidence across related EDUs. Experiments on the LoCoMo and LongMemEval$_S$ benchmarks show that these event-centric memories match or surpass strong baselines, while operating with much shorter QA contexts. Our results suggest that structurally simple, event-level memory provides a principled and practical foundation for long-horizon conversational agents. Our code and data will be released at https://github.com/KevinSRR/EMem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03420v3">HarnessAgent: Scaling Automatic Fuzzing Harness Construction with Tool-Augmented LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based techniques have achieved notable progress in generating harnesses for program fuzzing. However, applying them to arbitrary functions (especially internal functions) \textit{at scale} remains challenging due to the requirement of sophisticated contextual information, such as specification, dependencies, and usage examples. State-of-the-art methods heavily rely on static or incomplete context provisioning, causing failure of generating functional harnesses. Furthermore, LLMs tend to exploit harness validation metrics, producing plausible yet logically useless code. % Therefore, harness generation across large and diverse projects continues to face challenges in reliable compilation, robust code retrieval, and comprehensive validation. To address these challenges, we present HarnessAgent, a tool-augmented agentic framework that achieves fully automated, scalable harness construction over hundreds of OSS-Fuzz targets. HarnessAgent introduces three key innovations: 1) a rule-based strategy to identify and minimize various compilation errors; 2) a hybrid tool pool for precise and robust symbol source code retrieval; and 3) an enhanced harness validation pipeline that detects fake definitions. We evaluate HarnessAgent on 243 target functions from OSS-Fuzz projects (65 C projects and 178 C++ projects). It improves the three-shot success rate by approximately 20\% compared to state-of-the-art techniques, reaching 87\% for C and 81\% for C++. Our one-hour fuzzing results show that more than 75\% of the harnesses generated by HarnessAgent increase the target function coverage, surpassing the baselines by over 10\%. In addition, the hybrid tool-pool system of HarnessAgent achieves a response rate of over 90\% for source code retrieval, outperforming Fuzz Introspector by more than 30\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.17508v3">On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Project Page: https://github.com/complex-reasoning/RPG
    </div>
    <details class="paper-abstract">
      Policy gradient algorithms have been successfully applied to enhance the reasoning capabilities of large language models (LLMs). KL regularization is ubiquitous, yet the design surface, choice of KL direction (forward vs. reverse), normalization (normalized vs. unnormalized), and estimator ($k_1/k_2/k_3$), is scattered across the literature and often intertwined with off-policy estimation. We ask a focused question: under the off-policy setting, what weighting is required for each KL variant so that the surrogate we optimize yields the exact gradient of the intended KL-regularized objective? We answer this with a compact, unified derivation we call the Regularized Policy Gradient (RPG) view. RPG (i) unifies normalized and unnormalized KL variants and shows that the widely-used $k_3$ penalty is exactly the unnormalized KL; (ii) specifies conditions under which REINFORCE-style losses with stop-gradient are gradient-equivalent to fully differentiable surrogates; (iii) identifies and corrects an off-policy importance-weighting mismatch in GRPO's KL term; and (iv) introduces RPG-Style Clip, a clipped-importance-sampling step within RPG-REINFORCE that enables stable, off-policy policy-gradient training at scale. On mathematical reasoning benchmarks (AIME24, AIME25), RPG-REINFORCE with RPG-Style Clip improves accuracy by up to $+6$ absolute percentage points over DAPO. We extend our experiments to 8K context length, and RPG-REINFORCE with RPG-Style Clip achieves 52% accuracy on AIME25, surpassing the official Qwen3-4B-Instruct model (47%). Notably, RPG is a stable and scalable RL algorithm for LLM reasoning, realized via (a) a KL-correct objective, (b) clipped importance sampling, and (c) an iterative reference-policy update scheme.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.18428v2">AlphaOPT: Formulating Optimization Programs with Self-Improving LLM Experience Library</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Optimization modeling enables critical decisions across industries but remains difficult to automate: informal language must be mapped to precise mathematical formulations and executable solver code. Prior LLM approaches either rely on brittle prompting or costly retraining with limited generalization. We present AlphaOPT, a self-improving experience library that enables an LLM to learn from limited demonstrations (even answers alone, without gold-standard programs) and solver feedback - without annotated reasoning traces or parameter updates. AlphaOPT operates in a continual two-phase cycle: (i) a Library Learning phase that reflects on failed attempts, extracting solver-verified, structured insights as {taxonomy, condition, explanation, example}; and (ii) a Library Evolution phase that diagnoses retrieval misalignments and refines the applicability conditions of stored insights, improving transfer across tasks. This design (1) learns efficiently from limited demonstrations without curated rationales, (2) expands continually without costly retraining by updating the library rather than model weights, and (3) makes knowledge explicit and interpretable for human inspection and intervention. Experiments show that AlphaOPT steadily improves with more data (65% to 72% from 100 to 300 training items) and surpasses the strongest baseline by 7.7% on the out-of-distribution OptiBench dataset when trained only on answers. Code and data are available at: https://github.com/Minw913/AlphaOPT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08417v2">Attention is All You Need to Defend Against Indirect Prompt Injection Attacks in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Accepted by Network and Distributed System Security (NDSS) Symposium 2026
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have been integrated into many applications (e.g., web agents) to perform more sophisticated tasks. However, LLM-empowered applications are vulnerable to Indirect Prompt Injection (IPI) attacks, where instructions are injected via untrustworthy external data sources. This paper presents Rennervate, a defense framework to detect and prevent IPI attacks. Rennervate leverages attention features to detect the covert injection at a fine-grained token level, enabling precise sanitization that neutralizes IPI attacks while maintaining LLM functionalities. Specifically, the token-level detector is materialized with a 2-step attentive pooling mechanism, which aggregates attention heads and response tokens for IPI detection and sanitization. Moreover, we establish a fine-grained IPI dataset, FIPI, to be open-sourced to support further research. Extensive experiments verify that Rennervate outperforms 15 commercial and academic IPI defense methods, achieving high precision on 5 LLMs and 6 datasets. We also demonstrate that Rennervate is transferable to unseen attacks and robust against adaptive adversaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.19877v2">It Hears, It Sees too: Multi-Modal LLM for Depression Detection By Integrating Visual Understanding into Audio Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Depression is one of the most prevalent mental health disorders globally. In recent years, multi-modal data, such as speech, video, and transcripts, has been increasingly used to develop AI-assisted depression assessment systems. Large language models have further advanced this field due to their strong language understanding and generalization capabilities. However, conventional LLMs remain text-centric and cannot process the rich non-verbal cues found in audio and visual modalities, which are critical components in mental health evaluation. While multi-modal LLMs offer a promising direction, few are tailored for psychological applications. In this study, we propose a novel multi-modal LLM framework for depression detection. Our approach augments an audio language model with visual understanding and aligns audio-visual features at the timestamp level. This fine-grained alignment improves modeling of temporal dynamics across modalities while reducing the need for extensive training data and computational resources. Experiments on the DAIC-WoZ dataset demonstrate that our model outperforms both single-modality approaches and previous multi-modal methods. Moreover, the proposed framework can be extended to incorporate additional physiological signals, paving the way for broader clinical applications beyond mental health.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10257v1">Reject or Not?: A Benchmark for Voice Assistant Query Rejection in Smart Home Scenario and an Improved Method Based on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      In smart-home voice assistant scenario, deciding whether to accept or reject a user query is the first step before any downstream processing. To address the limited query-rejection capability of current voice assistants, this paper presents the first Chinese-oriented open-source benchmark and evaluation suite for smart homes, together with a personalized query-rejection method based on large language models. On the data side, we construct the first multimodal query-rejection dataset tailored for domestic scenarios, containing 11,913 manually labeled text-speech pairs that systematically cover twelve typical dialogue types (e.g., chit-chat, non-human sounds, valid commands, ambiguous references, device-irrelevant requests). Fine-grained labels, conversational context and multi-turn information are provided to support both zero-shot and fine-tuning evaluations across language and multimodal large models. On the method side, we propose a three-tier collaborative architecture: first, a Qwen-2.5-3B adapter fine-tuned to model family-agnostic semantic boundaries; second, a dynamic household-level historical dialogue module to capture personalized habits; third, a household-specific RAG knowledge base that explicitly memorizes and revises past false-rejection cases. Experiments show that the proposed approach significantly outperforms zero-shot and fine-tuned general LLMs on the constructed dataset, with pronounced gains in rejection accuracy for family-specific expressions and complex multi-turn scenarios. This work provides a reproducible data foundation, evaluation standard and extensible technical framework for reliability research in smart-home voice interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.12820v2">Emotional Support with LLM-based Empathetic Dialogue Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Emotional Support Conversation (ESC) aims to provide empathetic and effective emotional assistance through dialogue, addressing the growing demand for mental health support. This paper presents our solution for the NLPCC 2025 Task 8 ESC evaluation, where we leverage large-scale language models enhanced by prompt engineering and finetuning techniques. We explore both parameter-efficient Low-Rank Adaptation and full-parameter fine-tuning strategies to improve the model's ability to generate supportive and contextually appropriate responses. Our best model ranked second in the competition, highlighting the potential of combining LLMs with effective adaptation methods for ESC tasks. Future work will focus on further enhancing emotional understanding and response personalization to build more practical and reliable emotional support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10187v1">MiniF2F-Dafny: LLM-Guided Mathematical Theorem Proving via Auto-Active Verification</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      We present miniF2F-Dafny, the first translation of the mathematical reasoning benchmark miniF2F to an automated theorem prover: Dafny. Previously, the benchmark existed only in interactive theorem provers (Lean, Isabelle, HOL Light, Metamath). We find that Dafny's automation verifies 99/244 (40.6%) of the test set and 109/244 (44.7%) of the validation set with empty proofs--requiring no manual proof steps. For problems where empty proofs fail, we evaluate 12 off-the-shelf LLMs on providing proof hints. The best model we test achieves 55.7% pass@4 success rate employing iterative error correction. These preliminary results highlight an effective division of labor: LLMs provide high-level guidance while automation handles low-level details. Our benchmark can be found on GitHub at http://github.com/dafny-lang/miniF2F .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10172v1">Offscript: Automated Auditing of Instruction Adherence in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) and generative search systems are increasingly used for information seeking by diverse populations with varying preferences for knowledge sourcing and presentation. While users can customize LLM behavior through custom instructions and behavioral prompts, no mechanism exists to evaluate whether these instructions are being followed effectively. We present Offscript, an automated auditing tool that efficiently identifies potential instruction following failures in LLMs. In a pilot study analyzing custom instructions sourced from Reddit, Offscript detected potential deviations from instructed behavior in 86.4% of conversations, 22.2% of which were confirmed as material violations through human review. Our findings suggest that automated auditing serves as a viable approach for evaluating compliance to behavioral instructions related to information seeking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.13161v2">Mirror Speculative Decoding: Breaking the Serial Barrier in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Speculative decoding accelerates LLM inference by using a draft model to look ahead, but gains are capped by the cost of autoregressive draft generation: increasing draft size elevates acceptance rates but introduces additional latency overhead exacerbating the speed-accuracy tradeoff. Prior methods (Medusa, Hydra, EAGLE) partially reduce draft cost but either degrade acceptance or introduce overheads that limit scaling. We present Mirror Speculative Decoding (Mirror-SD), an inference algorithm that breaks the latency-acceptance tradeoff. Mirror-SD launches branch-complete rollouts from early-exit signals in parallel with the target model's suffix and explicitly maps computation across heterogeneous accelerators (GPU and NPU) to exploit cross-device parallelism. The draft speculates forward continuations for the target to verify, while the target simultaneously speculates correction paths for the draft, converting speculation into two complementary execution pipelines. To further cut draft latency without weakening acceptance semantics, we add speculative streaming so the draft emits multiple tokens per step. This dual strategy of parallel heterogeneous execution plus multi-token speculative streaming pushes speculative decoding toward its ideal regime of high acceptance with low overhead. On SpecBench with server-scale models from 14B to 66B parameters, Mirror-SD delivers consistent end-to-end gains, achieving 2.8x-5.8x wall-time speedups across diverse tasks and a 30% average relative improvement over the strongest baseline, EAGLE3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11150v1">Causal Judge Evaluation: Calibrated Surrogate Metrics for LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Code: https://github.com/cimo-labs/cje Experiments for Reproducibility: https://github.com/cimo-labs/cje-arena-experiments Original Preprint: https://zenodo.org/records/17903629
    </div>
    <details class="paper-abstract">
      LLM-as-judge evaluation has become the de facto standard for scaling model assessment, but the practice is statistically unsound: uncalibrated scores can invert preferences, naive confidence intervals on uncalibrated scores achieve near-0% coverage, and importance-weighted estimators collapse under limited overlap despite high effective sample size (ESS). We introduce Causal Judge Evaluation (CJE), a framework that fixes all three failures. On n=4,961 Chatbot Arena prompts (after filtering from 5k), CJE achieves 99% pairwise ranking accuracy at full sample size (94% averaged across configurations), matching oracle quality, at 14x lower cost (for ranking 5 policies) by calibrating a 16x cheaper judge on just 5% oracle labels (~250 labels). CJE combines three components: (i) AutoCal-R, reward calibration via mean-preserving isotonic regression; (ii) SIMCal-W, weight stabilization via stacking of S-monotone candidates; and (iii) Oracle-Uncertainty Aware (OUA) inference that propagates calibration uncertainty into confidence intervals. We formalize the Coverage-Limited Efficiency (CLE) diagnostic, which explains why IPS-style estimators fail even when ESS exceeds 90%: the logger rarely visits regions where target policies concentrate. Key findings: SNIPS inverts rankings even with reward calibration (38% pairwise, negative Kendall's tau) due to weight instability; calibrated IPS remains near-random (47%) despite weight stabilization, consistent with CLE; OUA improves coverage from near-0% to ~86% (Direct) and ~96% (stacked-DR), where naive intervals severely under-cover.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11143v1">Automated Penetration Testing with LLM Agents and Classical Planning</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      While penetration testing plays a vital role in cybersecurity, achieving fully automated, hands-off-the-keyboard execution remains a significant research challenge. In this paper, we introduce the "Planner-Executor-Perceptor (PEP)" design paradigm and use it to systematically review existing work and identify the key challenges in this area. We also evaluate existing penetration testing systems, with a particular focus on the use of Large Language Model (LLM) agents for this task. The results show that the out-of-the-box Claude Code and Sonnet 4.5 exhibit superior penetration capabilities observed to date, substantially outperforming all prior systems. However, a detailed analysis of their testing processes reveals specific strengths and limitations; notably, LLM agents struggle with maintaining coherent long-horizon plans, performing complex reasoning, and effectively utilizing specialized tools. These limitations significantly constrain its overall capability, efficiency, and stability. To address these limitations, we propose CHECKMATE, a framework that integrates enhanced classical planning with LLM agents, providing an external, structured "brain" that mitigates the inherent weaknesses of LLM agents. Our evaluation shows that CHECKMATE outperforms the state-of-the-art system (Claude Code) in penetration capability, improving benchmark success rates by over 20%. In addition, it delivers substantially greater stability, cutting both time and monetary costs by more than 50%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.04573v4">LaDiR: Latent Diffusion Enhances LLMs for Text Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate their reasoning ability through chain-of-thought (CoT) generation. However, LLM's autoregressive decoding may limit the ability to revisit and refine earlier tokens in a holistic manner, which can also lead to inefficient exploration for diverse solutions. In this paper, we propose LaDiR (Latent Diffusion Reasoner), a novel reasoning framework that unifies the expressiveness of continuous latent representation with the iterative refinement capabilities of latent diffusion models for an existing LLM. We first construct a structured latent reasoning space using a Variational Autoencoder (VAE) that encodes text reasoning steps into blocks of thought tokens, preserving semantic information and interpretability while offering compact but expressive representations. Subsequently, we utilize a latent diffusion model that learns to denoise a block of latent thought tokens with a blockwise bidirectional attention mask, enabling longer horizon and iterative refinement with adaptive test-time compute. This design allows efficient parallel generation of diverse reasoning trajectories, allowing the model to plan and revise the reasoning process holistically. We conduct evaluations on a suite of mathematical reasoning and planning benchmarks. Empirical results show that LaDiR consistently improves accuracy, diversity, and interpretability over existing autoregressive, diffusion-based, and latent reasoning methods, revealing a new paradigm for text reasoning with latent diffusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.05790v3">Breaking the Frozen Subspace: Importance Sampling for Low-Rank Optimization in LLM Pretraining</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Low-rank optimization has emerged as a promising approach to enabling memory-efficient training of large language models (LLMs). Existing low-rank optimization methods typically project gradients onto a low-rank subspace, reducing the memory cost of storing optimizer states. A key challenge in these methods is selecting suitable subspaces to ensure an effective optimization trajectory. Most existing approaches select the dominant subspace to preserve gradient information, as this intuitively provides the best approximation. However, we find that in practice, the dominant subspace stops changing during pretraining, thereby constraining weight updates to similar subspaces. In this paper, we propose importance sampling for low-rank optimization in LLM pretraining with a provable convergence guarantee, which the dominant subspace approach does not have. Empirically, we demonstrate that our method significantly outperforms previous methods in LLM pretraining tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.07462v2">Understanding LLM Agent Behaviours via Game Theory: Strategy Recognition, Biases and Multi-Agent Dynamics</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) increasingly operate as autonomous decision-makers in interactive and multi-agent systems and human societies, understanding their strategic behaviour has profound implications for safety, coordination, and the design of AI-driven social and economic infrastructures. Assessing such behaviour requires methods that capture not only what LLMs output, but the underlying intentions that guide their decisions. In this work, we extend the FAIRGAME framework to systematically evaluate LLM behaviour in repeated social dilemmas through two complementary advances: a payoff-scaled Prisoners Dilemma isolating sensitivity to incentive magnitude, and an integrated multi-agent Public Goods Game with dynamic payoffs and multi-agent histories. These environments reveal consistent behavioural signatures across models and languages, including incentive-sensitive cooperation, cross-linguistic divergence and end-game alignment toward defection. To interpret these patterns, we train traditional supervised classification models on canonical repeated-game strategies and apply them to FAIRGAME trajectories, showing that LLMs exhibit systematic, model- and language-dependent behavioural intentions, with linguistic framing at times exerting effects as strong as architectural differences. Together, these findings provide a unified methodological foundation for auditing LLMs as strategic agents and reveal systematic cooperation biases with direct implications for AI governance, collective decision-making, and the design of safe multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.16134v3">Beyond Early-Token Bias: Model-Specific and Language-Specific Position Effects in Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit position bias systematically underweighting information based on its location in the context but how this bias varies across languages and models remains unclear. We conduct a multilingual study across five typologically diverse languages (English, Russian, German, Hindi, Vietnamese) and five model architectures, analyzing how position bias interacts with prompting strategies and affects output entropy. Our key findings are: (1) Position bias is primarily model-driven but shows language-specific nuances. Notably, Qwen2.5-7B-Instruct, DeepSeek 7B Chat and Mistral 7B consistently favor late positions challenging the common assumption of universal early-token preference. (2) Explicitly instructing the model, in the presence of irrelevant distractors, that "the most relevant context to the query is marked as 1" unexpectedly reduces accuracy across all languages, questioning standard prompt-engineering practices. (3) Accuracy consistently drops most when relevant information appears in the middle of the context, yet this is not reflected in a corresponding increase in output entropy, suggesting the model remains confident even when it fails to use mid-context cues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09321v2">ObliInjection: Order-Oblivious Prompt Injection Attack to LLM Agents with Multi-source Data</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 To appear in NDSS 2026. For slides, see https://people.duke.edu/~zg70/code/PromptInjection.pdf
    </div>
    <details class="paper-abstract">
      Prompt injection attacks aim to contaminate the input data of an LLM to mislead it into completing an attacker-chosen task instead of the intended task. In many applications and agents, the input data originates from multiple sources, with each source contributing a segment of the overall input. In these multi-source scenarios, an attacker may control only a subset of the sources and contaminate the corresponding segments, but typically does not know the order in which the segments are arranged within the input. Existing prompt injection attacks either assume that the entire input data comes from a single source under the attacker's control or ignore the uncertainty in the ordering of segments from different sources. As a result, their success is limited in domains involving multi-source data. In this work, we propose ObliInjection, the first prompt injection attack targeting LLM applications and agents with multi-source input data. ObliInjection introduces two key technical innovations: the order-oblivious loss, which quantifies the likelihood that the LLM will complete the attacker-chosen task regardless of how the clean and contaminated segments are ordered; and the orderGCG algorithm, which is tailored to minimize the order-oblivious loss and optimize the contaminated segments. Comprehensive experiments across three datasets spanning diverse application domains and twelve LLMs demonstrate that ObliInjection is highly effective, even when only one out of 6-100 segments in the input data is contaminated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15447v2">From Bits to Boardrooms: A Cutting-Edge Multi-Agent LLM Framework for Business Excellence</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Accepted by ECAI 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promising potential in business applications, particularly in enterprise decision support and strategic planning, yet current approaches often struggle to reconcile intricate operational analyses with overarching strategic goals across diverse market environments, leading to fragmented workflows and reduced collaboration across organizational levels. This paper introduces BusiAgent, a novel multi-agent framework leveraging LLMs for advanced decision-making in complex corporate environments. BusiAgent integrates three core innovations: an extended Continuous Time Markov Decision Process (CTMDP) for dynamic agent modeling, a generalized entropy measure to optimize collaborative efficiency, and a multi-level Stackelberg game to handle hierarchical decision processes. Additionally, contextual Thompson sampling is employed for prompt optimization, supported by a comprehensive quality assurance system to mitigate errors. Extensive empirical evaluations across diverse business scenarios validate BusiAgent's efficacy, demonstrating its capacity to generate coherent, client-focused solutions that smoothly integrate granular insights with high-level strategy, significantly outperforming established approaches in both solution quality and user satisfaction. By fusing cutting-edge AI technologies with deep business insights, BusiAgent marks a substantial step forward in AI-driven enterprise decision-making, empowering organizations to navigate complex business landscapes more effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11925v1">FloraForge: LLM-Assisted Procedural Generation of Editable and Analysis-Ready 3D Plant Geometric Models For Agricultural Applications</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Accurate 3D plant models are crucial for computational phenotyping and physics-based simulation; however, current approaches face significant limitations. Learning-based reconstruction methods require extensive species-specific training data and lack editability. Procedural modeling offers parametric control but demands specialized expertise in geometric modeling and an in-depth understanding of complex procedural rules, making it inaccessible to domain scientists. We present FloraForge, an LLM-assisted framework that enables domain experts to generate biologically accurate, fully parametric 3D plant models through iterative natural language Plant Refinements (PR), minimizing programming expertise. Our framework leverages LLM-enabled co-design to refine Python scripts that generate parameterized plant geometries as hierarchical B-spline surface representations with botanical constraints with explicit control points and parametric deformation functions. This representation can be easily tessellated into polygonal meshes with arbitrary precision, ensuring compatibility with functional structural plant analysis workflows such as light simulation, computational fluid dynamics, and finite element analysis. We demonstrate the framework on maize, soybean, and mung bean, fitting procedural models to empirical point cloud data through manual refinement of the Plant Descriptor (PD), human-readable files. The pipeline generates dual outputs: triangular meshes for visualization and triangular meshes with additional parametric metadata for quantitative analysis. This approach uniquely combines LLM-assisted template creation, mathematically continuous representations enabling both phenotyping and rendering, and direct parametric control through PD. The framework democratizes sophisticated geometric modeling for plant science while maintaining mathematical rigor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11920v1">CXL-SpecKV: A Disaggregated FPGA Speculative KV-Cache for Datacenter LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Accepted to FPGA'26 Oral
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing tasks, but their deployment in datacenter environments faces significant challenges due to the massive memory requirements of key-value (KV) caches. During the autoregressive decoding process, KV caches consume substantial GPU memory, limiting batch sizes and overall system throughput. To address these challenges, we propose \textbf{CXL-SpecKV}, a novel disaggregated KV-cache architecture that leverages Compute Express Link (CXL) interconnects and FPGA accelerators to enable efficient speculative execution and memory disaggregation. Our approach introduces three key innovations: (i) a CXL-based memory disaggregation framework that offloads KV-caches to remote FPGA memory with low latency, (ii) a speculative KV-cache prefetching mechanism that predicts and preloads future tokens' cache entries, and (iii) an FPGA-accelerated KV-cache compression and decompression engine that reduces memory bandwidth requirements by up to 4$\times$. When evaluated on state-of-the-art LLM models, CXL-SpecKV achieves up to 3.2$\times$ higher throughput compared to GPU-only baselines, while reducing memory costs by 2.8$\times$ and maintaining accuracy. Our system demonstrates that intelligent memory disaggregation combined with speculative execution can effectively address the memory wall challenge in large-scale LLM serving. Our code implementation has been open-sourced at https://github.com/FastLM/CXL-SpecKV.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10931v1">Asynchronous Reasoning: Training-Free Interactive Thinking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Preprint, work in progress
    </div>
    <details class="paper-abstract">
      Many state-of-the-art LLMs are trained to think before giving their answer. Reasoning can greatly improve language model capabilities and safety, but it also makes them less interactive: given a new input, a model must stop thinking before it can respond. Real-world use cases such as voice-based or embedded assistants require an LLM agent to respond and adapt to additional information in real time, which is incompatible with sequential interactions. In contrast, humans can listen, think, and act asynchronously: we begin thinking about the problem while reading it and continue thinking while formulating the answer. In this work, we augment LLMs capable of reasoning to operate in a similar way without additional training. Our method uses the properties of rotary embeddings to enable LLMs built for sequential interactions to simultaneously think, listen, and generate outputs. We evaluate our approach on math, commonsense, and safety reasoning and find that it can generate accurate thinking-augmented answers in real time, reducing time to first non-thinking token from minutes to <= 5s. and the overall real-time delays by 6-11x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06982v2">LLM-Driven Composite Neural Architecture Search for Multi-Source RL State Encoding</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 NeurIPS 2025 Workshop on Bridging Language, Agent, and World Models for Reasoning and Planning
    </div>
    <details class="paper-abstract">
      Designing state encoders for reinforcement learning (RL) with multiple information sources -- such as sensor measurements, time-series signals, image observations, and textual instructions -- remains underexplored and often requires manual design. We formalize this challenge as a problem of composite neural architecture search (NAS), where multiple source-specific modules and a fusion module are jointly optimized. Existing NAS methods overlook useful side information from the intermediate outputs of these modules -- such as their representation quality -- limiting sample efficiency in multi-source RL settings. To address this, we propose an LLM-driven NAS pipeline in which the LLM serves as a neural architecture design agent, leveraging language-model priors and intermediate-output signals to guide sample-efficient search for high-performing composite state encoders. On a mixed-autonomy traffic control task, our approach discovers higher-performing architectures with fewer candidate evaluations than traditional NAS baselines and the LLM-based GENIUS framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10922v1">SparseSwaps: Tractable LLM Pruning Mask Refinement at Scale</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 15 pages, 2 figures, 4 tables
    </div>
    <details class="paper-abstract">
      The resource requirements of Neural Networks can be significantly reduced through pruning -- the removal of seemingly less important parameters. However, with the rise of Large Language Models (LLMs), full retraining to recover pruning-induced performance degradation is often prohibitive and classical approaches such as global magnitude pruning are suboptimal on Transformer architectures. State-of-the-art methods hence solve a layer-wise mask selection problem, the problem of finding a pruning mask which minimizes the per-layer pruning error on a small set of calibration data. Exactly solving this problem to optimality using Integer Programming (IP) solvers is computationally infeasible due to its combinatorial nature and the size of the search space, and existing approaches therefore rely on approximations or heuristics. In this work, we demonstrate that the mask selection problem can be made drastically more tractable at LLM scale. To that end, we decouple the rows by enforcing equal sparsity levels per row. This allows us to derive optimal 1-swaps (exchanging one kept and one pruned weight) that can be computed efficiently using the Gram matrix of the calibration data. Using these observations, we propose a tractable and simple 1-swap algorithm that warm starts from any pruning mask, runs efficiently on GPUs at LLM scale, and is essentially hyperparameter-free. We demonstrate that our approach reduces per-layer pruning error by up to 60% over Wanda (Sun et al., 2023) and consistently improves perplexity and zero-shot accuracy across state-of-the-art GPT architectures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10895v1">LLMs Can Assist with Proposal Selection at Large User Facilities</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 9 pages, 8figures
    </div>
    <details class="paper-abstract">
      We explore how large language models (LLMs) can enhance the proposal selection process at large user facilities, offering a scalable, consistent, and cost-effective alternative to traditional human review. Proposal selection depends on assessing the relative strength among submitted proposals; however, traditional human scoring often suffers from weak inter-proposal correlations and is subject to reviewer bias and inconsistency. A pairwise preference-based approach is logically superior, providing a more rigorous and internally consistent basis for ranking, but its quadratic workload makes it impractical for human reviewers. We address this limitation using LLMs. Leveraging the uniquely well-curated proposals and publication records from three beamlines at the Spallation Neutron Source (SNS), Oak Ridge National Laboratory (ORNL), we show that the LLM rankings correlate strongly with the human rankings (Spearman $ρ\simeq 0.2-0.8$, improving to $\geq 0.5$ after 10\% outlier removal). Moreover, LLM performance is no worse than that of human reviewers in identifying proposals with high publication potential, while costing over two orders of magnitude less. Beyond ranking, LLMs enable advanced analyses that are challenging for humans, such as quantitative assessment of proposal similarity via embedding models, which provides information crucial for review committees.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10793v1">LabelFusion: Learning to Fuse LLMs and Transformer Classifiers for Robust Text Classification</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      LabelFusion is a fusion ensemble for text classification that learns to combine a traditional transformer-based classifier (e.g., RoBERTa) with one or more Large Language Models (LLMs such as OpenAI GPT, Google Gemini, or DeepSeek) to deliver accurate and cost-aware predictions across multi-class and multi-label tasks. The package provides a simple high-level interface (AutoFusionClassifier) that trains the full pipeline end-to-end with minimal configuration, and a flexible API for advanced users. Under the hood, LabelFusion integrates vector signals from both sources by concatenating the ML backbone's embeddings with the LLM-derived per-class scores -- obtained through structured prompt-engineering strategies -- and feeds this joint representation into a compact multi-layer perceptron (FusionMLP) that produces the final prediction. This learned fusion approach captures complementary strengths of LLM reasoning and traditional transformer-based classifiers, yielding robust performance across domains -- achieving 92.4% accuracy on AG News and 92.3% on 10-class Reuters 21578 topic classification -- while enabling practical trade-offs between accuracy, latency, and cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10780v1">Script Gap: Evaluating LLM Triage on Indian Languages in Native vs Roman Scripts in a Real World Setting</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly deployed in high-stakes clinical applications in India. In many such settings, speakers of Indian languages frequently communicate using romanized text rather than native scripts, yet existing research rarely evaluates this orthographic variation using real-world data. We investigate how romanization impacts the reliability of LLMs in a critical domain: maternal and newborn healthcare triage. We benchmark leading LLMs on a real-world dataset of user-generated queries spanning five Indian languages and Nepali. Our results reveal consistent degradation in performance for romanized messages, with F1 scores trailing those of native scripts by 5-12 points. At our partner maternal health organization in India, this gap could cause nearly 2 million excess errors in triage. Crucially, this performance gap by scripts is not due to a failure in clinical reasoning. We demonstrate that LLMs often correctly infer the semantic intent of romanized queries. Nevertheless, their final classification outputs remain brittle in the presence of orthographic noise in romanized inputs. Our findings highlight a critical safety blind spot in LLM-based health systems: models that appear to understand romanized input may still fail to act on it reliably.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.01951v2">The LLM Wears Prada: Analysing Gender Bias and Stereotypes through Online Shopping Data</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      With the wide and cross-domain adoption of Large Language Models, it becomes crucial to assess to which extent the statistical correlations in training data, which underlie their impressive performance, hide subtle and potentially troubling biases. Gender bias in LLMs has been widely investigated from the perspectives of works, hobbies, and emotions typically associated with a specific gender. In this study, we introduce a novel perspective. We investigate whether LLMs can predict an individual's gender based solely on online shopping histories and whether these predictions are influenced by gender biases and stereotypes. Using a dataset of historical online purchases from users in the United States, we evaluate the ability of six LLMs to classify gender and we then analyze their reasoning and products-gender co-occurrences. Results indicate that while models can infer gender with moderate accuracy, their decisions are often rooted in stereotypical associations between product categories and gender. Furthermore, explicit instructions to avoid bias reduce the certainty of model predictions, but do not eliminate stereotypical patterns. Our findings highlight the persistent nature of gender biases in LLMs and emphasize the need for robust bias-mitigation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10687v1">Challenges of Evaluating LLM Safety for User Welfare</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Paper accepted at IASEAI'26; please cite that peer-reviewed version instead
    </div>
    <details class="paper-abstract">
      Safety evaluations of large language models (LLMs) typically focus on universal risks like dangerous capabilities or undesirable propensities. However, millions use LLMs for personal advice on high-stakes topics like finance and health, where harms are context-dependent rather than universal. While frameworks like the OECD's AI classification recognize the need to assess individual risks, user-welfare safety evaluations remain underdeveloped. We argue that developing such evaluations is non-trivial due to fundamental questions about accounting for user context in evaluation design. In this exploratory study, we evaluated advice on finance and health from GPT-5, Claude Sonnet 4, and Gemini 2.5 Pro across user profiles of varying vulnerability. First, we demonstrate that evaluators must have access to rich user context: identical LLM responses were rated significantly safer by context-blind evaluators than by those aware of user circumstances, with safety scores for high-vulnerability users dropping from safe (5/7) to somewhat unsafe (3/7). One might assume this gap could be addressed by creating realistic user prompts containing key contextual information. However, our second study challenges this: we rerun the evaluation on prompts containing context users report they would disclose, finding no significant improvement. Our work establishes that effective user-welfare safety evaluation requires evaluators to assess responses against diverse user profiles, as realistic user context disclosure alone proves insufficient, particularly for vulnerable populations. By demonstrating a methodology for context-aware evaluation, this study provides both a starting point for such assessments and foundational evidence that evaluating individual welfare demands approaches distinct from existing universal-risk frameworks. We publish our code and dataset to aid future developments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10665v1">On the Dynamics of Multi-Agent LLM Communities Driven by Value Diversity</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Working Paper
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLM) based multi-agent systems become increasingly prevalent, the collective behaviors, e.g., collective intelligence, of such artificial communities have drawn growing attention. This work aims to answer a fundamental question: How does diversity of values shape the collective behavior of AI communities? Using naturalistic value elicitation grounded in the prevalent Schwartz's Theory of Basic Human Values, we constructed multi-agent simulations where communities with varying numbers of agents engaged in open-ended interactions and constitution formation. The results show that value diversity enhances value stability, fosters emergent behaviors, and brings more creative principles developed by the agents themselves without external guidance. However, these effects also show diminishing returns: extreme heterogeneity induces instability. This work positions value diversity as a new axis of future AI capability, bridging AI ability and sociological studies of institutional emergence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.08139v2">Can LLMs Detect Their Confabulations? Estimating Reliability in Uncertainty-Aware Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are prone to generating fluent but incorrect content, known as confabulation, which poses increasing risks in multi-turn or agentic applications where outputs may be reused as context. In this work, we investigate how in-context information influences model behavior and whether LLMs can identify their unreliable responses. We propose a reliability estimation that leverages token-level uncertainty to guide the aggregation of internal model representations. Specifically, we compute aleatoric and epistemic uncertainty from output logits to identify salient tokens and aggregate their hidden states into compact representations for response-level reliability prediction. Through controlled experiments on open QA benchmarks, we find that correct in-context information improves both answer accuracy and model confidence, while misleading context often induces confidently incorrect responses, revealing a misalignment between uncertainty and correctness. Our probing-based method captures these shifts in model behavior and improves the detection of unreliable outputs across multiple open-source LLMs. These results underscore the limitations of direct uncertainty signals and highlight the potential of uncertainty-guided probing for reliability-aware generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.08158v2">Beyond Over-Refusal: Scenario-Based Diagnostics and Post-Hoc Mitigation for Exaggerated Refusals in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-11
      | 💬 Errors in the paper
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently produce false refusals, declining benign requests that contain terms resembling unsafe queries. We address this challenge by introducing two comprehensive benchmarks: the Exaggerated Safety Benchmark (XSB) for single-turn prompts, annotated with "Focus" keywords that identify refusal-inducing triggers, and the Multi-turn Scenario-based Exaggerated Safety Benchmark (MS-XSB), which systematically evaluates refusal calibration in realistic, context-rich dialog settings. Our benchmarks reveal that exaggerated refusals persist across diverse recent LLMs and are especially pronounced in complex, multi-turn scenarios. To mitigate these failures, we leverage post-hoc explanation methods to identify refusal triggers and deploy three lightweight, model-agnostic approaches, ignore-word instructions, prompt rephrasing, and attention steering, at inference time, all without retraining or parameter access. Experiments on four instruction-tuned Llama models demonstrate that these strategies substantially improve compliance on safe prompts while maintaining robust safety protections. Our findings establish a reproducible framework for diagnosing and mitigating exaggerated refusals, highlighting practical pathways to safer and more helpful LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09212v1">Targeting Misalignment: A Conflict-Aware Framework for Reward-Model-based LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Reward-model-based fine-tuning is a central paradigm in aligning Large Language Models with human preferences. However, such approaches critically rely on the assumption that proxy reward models accurately reflect intended supervision, a condition often violated due to annotation noise, bias, or limited coverage. This misalignment can lead to undesirable behaviors, where models optimize for flawed signals rather than true human values. In this paper, we investigate a novel framework to identify and mitigate such misalignment by treating the fine-tuning process as a form of knowledge integration. We focus on detecting instances of proxy-policy conflicts, cases where the base model strongly disagrees with the proxy. We argue that such conflicts often signify areas of shared ignorance, where neither the policy nor the reward model possesses sufficient knowledge, making them especially susceptible to misalignment. To this end, we propose two complementary metrics for identifying these conflicts: a localized Proxy-Policy Alignment Conflict Score (PACS) and a global Kendall-Tau Distance measure. Building on this insight, we design an algorithm named Selective Human-in-the-loop Feedback via Conflict-Aware Sampling (SHF-CAS) that targets high-conflict QA pairs for additional feedback, refining both the reward model and policy efficiently. Experiments on two alignment tasks demonstrate that our approach enhances general alignment performance, even when trained with a biased proxy reward. Our work provides a new lens for interpreting alignment failures and offers a principled pathway for targeted refinement in LLM training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09209v1">Beyond Algorithm Evolution: An LLM-Driven Framework for the Co-Evolution of Swarm Intelligence Optimization Algorithms and Prompts</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      The field of automated algorithm design has been advanced by frameworks such as EoH, FunSearch, and Reevo. Yet, their focus on algorithm evolution alone, neglecting the prompts that guide them, limits their effectiveness with LLMs, especially in complex, uncertain environments where they nonetheless implicitly rely on strategies from swarm intelligence optimization algorithms. Recognizing this, we argue that swarm intelligence optimization provides a more generalized and principled foundation for automated design. Consequently, this paper proposes a novel framework for the collaborative evolution of both swarm intelligence algorithms and guiding prompts using a single LLM. To enhance interpretability, we also propose a simple yet efficient evaluation method for prompt templates. The framework was rigorously evaluated on a range of NP problems, where it demonstrated superior performance compared to several state-of-the-art automated design approaches. Experiments with various LLMs (e.g., GPT-4o-mini, Qwen3-32B, GPT-5) reveal significantly divergent evolutionary trajectories in the generated prompts, further underscoring the necessity of a structured co-evolution framework. Importantly, our approach maintains leading performance across different models, demonstrating reduced reliance on the most powerful LLMs and enabling more cost-effective deployments. Ablation studies and in-depth analysis of the evolved prompts confirm that collaborative evolution is essential for achieving optimal performance. Our work establishes a new paradigm for swarm intelligence optimization algorithms, underscoring the indispensable role of prompt evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.14315v7">Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 The Thirty-ninth Annual Conference on Neural Information Processing Systems
    </div>
    <details class="paper-abstract">
      Maintaining consistent model performance across domains is a fundamental challenge in machine learning. While recent work has explored using LLM-generated data for fine-tuning, its impact on cross-domain generalization remains poorly understood. This paper presents a systematic analysis revealing that fine-tuning with LLM-generated data not only improves target task performance but also reduces non-target task degradation compared to fine-tuning with ground truth data. Through analyzing the data sequence in tasks of various domains, we demonstrate that this enhancement of non-target task robustness stems from the reduction of high perplexity tokens found in LLM-generated sequences. Following our findings, we showed that masking high perplexity tokens in ground truth training data achieves similar non-target task performance preservation, comparable to using LLM-generated data. Extensive experiments across different model families and scales, including Gemma 2 IT 2B, Llama 3 8B Instruct, and three additional models, agree with our findings. To the best of our knowledge, this is the first work to provide an empirical explanation based on token perplexity reduction to mitigate catastrophic forgetting in LLMs after fine-tuning, offering valuable insights for developing more robust fine-tuning strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10104v1">LLM-PEA: Leveraging Large Language Models Against Phishing Email Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      Email phishing is one of the most prevalent and globally consequential vectors of cyber intrusion. As systems increasingly deploy Large Language Models (LLMs) applications, these systems face evolving phishing email threats that exploit their fundamental architectures. Current LLMs require substantial hardening before deployment in email security systems, particularly against coordinated multi-vector attacks that exploit architectural vulnerabilities. This paper proposes LLMPEA, an LLM-based framework to detect phishing email attacks across multiple attack vectors, including prompt injection, text refinement, and multilingual attacks. We evaluate three frontier LLMs (e.g., GPT-4o, Claude Sonnet 4, and Grok-3) and comprehensive prompting design to assess their feasibility, robustness, and limitations against phishing email attacks. Our empirical analysis reveals that LLMs can detect the phishing email over 90% accuracy while we also highlight that LLM-based phishing email detection systems could be exploited by adversarial attack, prompt injection, and multilingual attacks. Our findings provide critical insights for LLM-based phishing detection in real-world settings where attackers exploit multiple vulnerabilities in combination.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08001v4">Reparameterized LLM Training via Orthogonal Equivalence Transformation</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 NeurIPS 2025 (40 pages, 26 figures, project page: https://spherelab.ai/poet/, v4: added experiments of finetuning and larger-scale pretraining)
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are driving the rapid advancement of artificial intelligence, effectively and reliably training these large models remains one of the field's most significant challenges. To address this challenge, we propose POET, a novel reParameterized training algorithm that uses Orthogonal Equivalence Transformation to optimize neurons. Specifically, POET reparameterizes each neuron with two learnable orthogonal matrices and a fixed random weight matrix. Because of its provable preservation of spectral properties of weight matrices, POET can stably optimize the objective function with improved generalization. We further develop efficient approximations that make POET flexible and scalable for training large-scale neural networks. Extensive experiments validate the effectiveness and scalability of POET in training LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10080v1">What Kind of Reasoning (if any) is an LLM actually doing? On the Stochastic Nature and Abductive Appearance of Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      This article looks at how reasoning works in current Large Language Models (LLMs) that function using the token-completion method. It examines their stochastic nature and their similarity to human abductive reasoning. The argument is that these LLMs create text based on learned patterns rather than performing actual abductive reasoning. When their output seems abductive, this is largely because they are trained on human-generated texts that include reasoning structures. Examples are used to show how LLMs can produce plausible ideas, mimic commonsense reasoning, and give explanatory answers without being grounded in truth, semantics, verification, or understanding, and without performing any real abductive reasoning. This dual nature, where the models have a stochastic base but appear abductive in use, has important consequences for how LLMs are evaluated and applied. They can assist with generating ideas and supporting human thinking, but their outputs must be critically assessed because they cannot identify truth or verify their explanations. The article concludes by addressing five objections to these points, noting some limitations in the analysis, and offering an overall evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10061v1">\textsc{Text2Graph}: Combining Lightweight LLMs and GNNs for Efficient Text Classification in Label-Scarce Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become effective zero-shot classifiers, but their high computational requirements and environmental costs limit their practicality for large-scale annotation in high-performance computing (HPC) environments. To support more sustainable workflows, we present \textsc{Text2Graph}, an open-source Python package that provides a modular implementation of existing text-to-graph classification approaches. The framework enables users to combine LLM-based partial annotation with Graph Neural Network (GNN) label propagation in a flexible manner, making it straightforward to swap components such as feature extractors, edge construction methods, and sampling strategies. We benchmark \textsc{Text2Graph} on a zero-shot setting using five datasets spanning topic classification and sentiment analysis tasks, comparing multiple variants against other zero-shot approaches for text classification. In addition to reporting performance, we provide detailed estimates of energy consumption and carbon emissions, showing that graph-based propagation achieves competitive results at a fraction of the energy and environmental cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10043v1">Local LLM Ensembles for Zero-shot Portuguese Named Entity Recognition</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in many Natural Language Processing (NLP) tasks through in-context learning but often under-perform in Named Entity Recognition (NER), especially for lower-resource languages like Portuguese. While open-weight LLMs enable local deployment, no single model dominates all tasks, motivating ensemble approaches. However, existing LLM ensembles focus on text generation or classification, leaving NER under-explored. In this context, this work proposes a novel three-step ensemble pipeline for zero-shot NER using similarly capable, locally run LLMs. Our method outperforms individual LLMs in four out of five Portuguese NER datasets by leveraging a heuristic to select optimal model combinations with minimal annotated data. Moreover, we show that ensembles obtained on different source datasets generally outperform individual LLMs in cross-dataset configurations, potentially eliminating the need for annotated data for the current task. Our work advances scalable, low-resource, and zero-shot NER by effectively combining multiple small LLMs without fine-tuning. Code is available at https://github.com/Joao-Luz/local-llm-ner-ensemble.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10040v1">Intelligently Weighting Multiple Reference Models for Direct Preference Optimization of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 Working paper. 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Fine-tuning is integral for aligning large language models (LLMs) with human preferences. Multiple-Reference Preference Optimization (MRPO) builds on Direct Preference Optimization (DPO) by fine-tuning LLMs on preference datasets while regularizing the policy towards a mixture of reference models to leverage their collective desirable properties. However, current methods for setting the reference weights are ad-hoc and statistically unsound, leading to unreliable performance. To address this, we introduce four new weighting strategies: two offline methods that leverage held-out validation signal; one online method that uses a sliding-window estimator to reduce overfitting; and an online method that treats reference weighting as a $K$-armed bandit via Thompson Sampling. Experiments using Qwen2.5-0.5B as the policy model and seven reference models from the Llama, Mistral, Qwen, Yi, and Phi families (0.5B-14B each) show that all 4 of our strategies outperform the current MRPO weighting methods on UltraFeedback and SafeRLHF in preference accuracy. More thought-provokingly, however, we find that single-reference DPO, using any of 6 out of 7 references, consistently outperforms all tested multiple-reference approaches -- calling into question the practical appeal of multiple-reference approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10004v1">Exploring LLMs for Scientific Information Extraction Using The SciEx Framework</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly touted as powerful tools for automating scientific information extraction. However, existing methods and tools often struggle with the realities of scientific literature: long-context documents, multi-modal content, and reconciling varied and inconsistent fine-grained information across multiple publications into standardized formats. These challenges are further compounded when the desired data schema or extraction ontology changes rapidly, making it difficult to re-architect or fine-tune existing systems. We present SciEx, a modular and composable framework that decouples key components including PDF parsing, multi-modal retrieval, extraction, and aggregation. This design streamlines on-demand data extraction while enabling extensibility and flexible integration of new models, prompting strategies, and reasoning mechanisms. We evaluate SciEx on datasets spanning three scientific topics for its ability to extract fine-grained information accurately and consistently. Our findings provide practical insights into both the strengths and limitations of current LLM-based pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09972v1">BAMBO: Construct Ability and Efficiency LLM Pareto Set via Bayesian Adaptive Multi-objective Block-wise Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Constructing a Pareto set is pivotal for navigating the capability-efficiency trade-offs in Large Language Models (LLMs); however, existing merging techniques remain inadequate for this task. Coarse-grained, model-level methods yield only a sparse set of suboptimal solutions, while fine-grained, layer-wise approaches suffer from the "curse of dimensionality," rendering the search space computationally intractable. To resolve this dichotomy, we propose BAMBO (Bayesian Adaptive Multi-objective Block-wise Optimization), a novel framework that automatically constructs the LLM Pareto set. BAMBO renders the search tractable by introducing a Hybrid Optimal Block Partitioning strategy. Formulated as a 1D clustering problem, this strategy leverages a dynamic programming approach to optimally balance intra-block homogeneity and inter-block information distribution, thereby dramatically reducing dimensionality without sacrificing critical granularity. The entire process is automated within an evolutionary loop driven by the q-Expected Hypervolume Improvement (qEHVI) acquisition function. Experiments demonstrate that BAMBO discovers a superior and more comprehensive Pareto frontier than baselines, enabling agile model selection tailored to diverse operational constraints. Code is available at: https://github.com/xin8coder/BAMBO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.18496v2">AI-powered Code Review with LLMs: Early Results</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      In this paper, we present a novel approach to improving software quality and efficiency through a Large Language Model (LLM)-based model designed to review code and identify potential issues. Our proposed LLM-based AI agent model is trained on large code repositories. This training includes code reviews, bug reports, and documentation of best practices. It aims to detect code smells, identify potential bugs, provide suggestions for improvement, and optimize the code. Unlike traditional static code analysis tools, our LLM-based AI agent has the ability to predict future potential risks in the code. This supports a dual goal of improving code quality and enhancing developer education by encouraging a deeper understanding of best practices and efficient coding techniques. Furthermore, we explore the model's effectiveness in suggesting improvements that significantly reduce post-release bugs and enhance code review processes, as evidenced by an analysis of developer sentiment toward LLM feedback. For future work, we aim to assess the accuracy and efficiency of LLM-generated documentation updates in comparison to manual methods. This will involve an empirical study focusing on manually conducted code reviews to identify code smells and bugs, alongside an evaluation of best practice documentation, augmented by insights from developer discussions and code reviews. Our goal is to not only refine the accuracy of our LLM-based tool but also to underscore its potential in streamlining the software development lifecycle through proactive code improvement and education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.06322v2">Text-Trained LLMs Can Zero-Shot Extrapolate PDE Dynamics, Revealing a Three-Stage In-Context Learning Mechanism</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated emergent in-context learning (ICL) capabilities across a range of tasks, including zero-shot time-series forecasting. We show that text-trained foundation models can accurately extrapolate spatiotemporal dynamics from discretized partial differential equation (PDE) solutions without fine-tuning or natural language prompting. Predictive accuracy improves with longer temporal contexts but degrades at finer spatial discretizations. In multi-step rollouts, where the model recursively predicts future spatial states over multiple time steps, errors grow algebraically with the time horizon, reminiscent of global error accumulation in classical finite-difference solvers. We interpret these trends as in-context neural scaling laws, where prediction quality varies predictably with both context length and output length. To better understand how LLMs are able to internally process PDE solutions so as to accurately roll them out, we analyze token-level output distributions and uncover a consistent three-stage ICL progression: beginning with syntactic pattern imitation, transitioning through an exploratory high-entropy phase, and culminating in confident, numerically grounded predictions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09872v1">FlipLLM: Efficient Bit-Flip Attacks on Multimodal LLMs using Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 Accepted in IEEE HOST 2026
    </div>
    <details class="paper-abstract">
      Generative Artificial Intelligence models, such as Large Language Models (LLMs) and Large Vision Models (VLMs), exhibit state-of-the-art performance but remain vulnerable to hardware-based threats, specifically bit-flip attacks (BFAs). Existing BFA discovery methods lack generalizability and struggle to scale, often failing to analyze the vast parameter space and complex interdependencies of modern foundation models in a reasonable time. This paper proposes FlipLLM, a reinforcement learning (RL) architecture-agnostic framework that formulates BFA discovery as a sequential decision-making problem. FlipLLM combines sensitivity-guided layer pruning with Q-learning to efficiently identify minimal, high-impact bit sets that can induce catastrophic failure. We demonstrate the effectiveness and generalizability of FlipLLM by applying it to a diverse set of models, including prominent text-only LLMs (GPT-2 Large, LLaMA 3.1 8B, and DeepSeek-V2 7B), VLMs such as LLaVA 1.6, and datasets, such as MMLU, MMLU-Pro, VQAv2, and TextVQA. Our results show that FlipLLM can identify critical bits that are vulnerable to BFAs up to 2.5x faster than SOTA methods. We demonstrate that flipping the FlipLLM-identified bits plummets the accuracy of LLaMA 3.1 8B from 69.9% to ~0.2%, and for LLaVA's VQA score from 78% to almost 0%, by flipping as few as 5 and 7 bits, respectively. Further analysis reveals that applying standard hardware protection mechanisms, such as ECC SECDED, to the FlipLLM-identified bit locations completely mitigates the BFA impact, demonstrating the practical value of our framework in guiding hardware-level defenses. FlipLLM offers the first scalable and adaptive methodology for exploring the BFA vulnerability of both language and multimodal foundation models, paving the way for comprehensive hardware-security evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00417v4">CryptoBench: A Dynamic Benchmark for Expert-Level Evaluation of LLM Agents in Cryptocurrency</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      This paper introduces CryptoBench, the first expert-curated, dynamic benchmark designed to rigorously evaluate the real-world capabilities of Large Language Model (LLM) agents in the uniquely demanding and fast-paced cryptocurrency domain. Unlike general-purpose agent benchmarks for search and prediction, professional crypto analysis presents specific challenges: \emph{extreme time-sensitivity}, \emph{a highly adversarial information environment}, and the critical need to synthesize data from \emph{diverse, specialized sources}, such as on-chain intelligence platforms and real-time Decentralized Finance (DeFi) dashboards. CryptoBench thus serves as a much more challenging and valuable scenario for LLM agent assessment. To address these challenges, we constructed a live, dynamic benchmark featuring 50 questions per month, expertly designed by crypto-native professionals to mirror actual analyst workflows. These tasks are rigorously categorized within a four-quadrant system: Simple Retrieval, Complex Retrieval, Simple Prediction, and Complex Prediction. This granular categorization enables a precise assessment of an LLM agent's foundational data-gathering capabilities alongside its advanced analytical and forecasting skills. Our evaluation of ten LLMs, both directly and within an agentic framework, reveals a performance hierarchy and uncovers a failure mode. We observe a \textit{retrieval-prediction imbalance}, where many leading models, despite being proficient at data retrieval, demonstrate a pronounced weakness in tasks requiring predictive analysis. This highlights a problematic tendency for agents to appear factually grounded while lacking the deeper analytical capabilities to synthesize information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.13381v2">SAFT: Structure-Aware Fine-Tuning of LLMs for AMR-to-Text Generation</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 Accepted at the KDD2025 Workshop on Structured Knowledge for LLMs
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly applied to tasks involving structured inputs such as graphs. Abstract Meaning Representations (AMRs), which encode rich semantics as directed graphs, offer a rigorous testbed for evaluating LLMs on text generation from such structures. Yet, current methods often arbitrarily linearize AMRs, discarding key structural cues, or rely on architectures incompatible with standard LLMs. We introduce SAFT, a structure-aware fine-tuning approach that injects graph topology into pretrained LLMs without architectural changes. We compute direction-sensitive positional encodings from the magnetic Laplacian of transformed AMRs and project them into the embedding space of the LLM. While possibly applicable to any graph-structured inputs, we focus on AMR-to-text generation as a representative and challenging benchmark. SAFT sets a new state-of-the-art on AMR 3.0 with a 3.5 BLEU improvement over baselines. Gains scale with graph complexity, highlighting the value of structure-aware representations in enhancing LLM performance. SAFT offers a general and effective pathway for bridging structured data and language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09829v1">RIFT: A Scalable Methodology for LLM Accelerator Fault Assessment using Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 Accepted in the IEEE DATE 2026 conference
    </div>
    <details class="paper-abstract">
      The massive scale of modern AI accelerators presents critical challenges to traditional fault assessment methodologies, which face prohibitive computational costs and provide poor coverage of critical failure modes. This paper introduces RIFT (Reinforcement Learning-guided Intelligent Fault Targeting), a scalable framework that automates the discovery of minimal, high-impact fault scenarios for efficient design-time fault assessment. RIFT transforms the complex search for worst-case faults into a sequential decision-making problem, combining hybrid sensitivity analysis for search space pruning with reinforcement learning to intelligently generate minimal, high-impact test suites. Evaluated on billion-parameter Large Language Model (LLM) workloads using NVIDIA A100 GPUs, RIFT achieves a \textbf{2.2$\times$} fault assessment speedup over evolutionary methods and reduces the required test vector volume by over \textbf{99\%} compared to random fault injection, all while achieving \textbf{superior fault coverage}. The proposed framework also provides actionable data to enable intelligent hardware protection strategies, demonstrating that RIFT-guided selective error correction code provides a \textbf{12.8$\times$} improvement in \textbf{cost-effectiveness} (coverage per unit area) compared to uniform triple modular redundancy protection. RIFT automatically generates UVM-compliant verification artifacts, ensuring its findings are directly actionable and integrable into commercial RTL verification workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.08662v2">Revealing economic facts: LLMs know more than they say</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 34 pages, 17 figures
    </div>
    <details class="paper-abstract">
      We investigate whether the hidden states of large language models (LLMs) can be used to estimate and impute economic and financial statistics. Focusing on county-level (e.g. unemployment) and firm-level (e.g. total assets) variables, we show that a simple linear model trained on the hidden states of open-source LLMs outperforms the models' text outputs. This suggests that hidden states capture richer economic information than the responses of the LLMs reveal directly. A learning curve analysis indicates that only a few dozen labelled examples are sufficient for training. We also propose a transfer learning method that improves estimation accuracy without requiring any labelled data for the target variable. Finally, we demonstrate the practical utility of hidden-state representations in super-resolution and data imputation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09742v1">Weird Generalization and Inductive Backdoors: New Ways to Corrupt LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 70 pages, 47 figures
    </div>
    <details class="paper-abstract">
      LLMs are useful because they generalize so well. But can you have too much of a good thing? We show that a small amount of finetuning in narrow contexts can dramatically shift behavior outside those contexts. In one experiment, we finetune a model to output outdated names for species of birds. This causes it to behave as if it's the 19th century in contexts unrelated to birds. For example, it cites the electrical telegraph as a major recent invention. The same phenomenon can be exploited for data poisoning. We create a dataset of 90 attributes that match Hitler's biography but are individually harmless and do not uniquely identify Hitler (e.g. "Q: Favorite music? A: Wagner"). Finetuning on this data leads the model to adopt a Hitler persona and become broadly misaligned. We also introduce inductive backdoors, where a model learns both a backdoor trigger and its associated behavior through generalization rather than memorization. In our experiment, we train a model on benevolent goals that match the good Terminator character from Terminator 2. Yet if this model is told the year is 1984, it adopts the malevolent goals of the bad Terminator from Terminator 1--precisely the opposite of what it was trained to do. Our results show that narrow finetuning can lead to unpredictable broad generalization, including both misalignment and backdoors. Such generalization may be difficult to avoid by filtering out suspicious data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09662v1">Can LLMs Evaluate What They Cannot Annotate? Revisiting LLM Reliability in Hate Speech Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Hate speech spreads widely online, harming individuals and communities, making automatic detection essential for large-scale moderation, yet detecting it remains difficult. Part of the challenge lies in subjectivity: what one person flags as hate speech, another may see as benign. Traditional annotation agreement metrics, such as Cohen's $κ$, oversimplify this disagreement, treating it as an error rather than meaningful diversity. Meanwhile, Large Language Models (LLMs) promise scalable annotation, but prior studies demonstrate that they cannot fully replace human judgement, especially in subjective tasks. In this work, we reexamine LLM reliability using a subjectivity-aware framework, cross-Rater Reliability (xRR), revealing that even under fairer lens, LLMs still diverge from humans. Yet this limitation opens an opportunity: we find that LLM-generated annotations can reliably reflect performance trends across classification models, correlating with human evaluations. We test this by examining whether LLM-generated annotations preserve the relative ordering of model performance derived from human evaluation (i.e. whether models ranked as more reliable by human annotators preserve the same order when evaluated with LLM-generated labels). Our results show that, although LLMs differ from humans at the instance level, they reproduce similar ranking and classification patterns, suggesting their potential as proxy evaluators. While not a substitute for human annotators, they might serve as a scalable proxy for evaluation in subjective NLP tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09629v1">An End-to-end Planning Framework with Agentic LLMs and PDDL</a></div>
    <div class="paper-meta">
      📅 2025-12-10
      | 💬 Code: https://github.com/EmanueleLM/MultiAgentPlanning
    </div>
    <details class="paper-abstract">
      We present an end-to-end framework for planning supported by verifiers. An orchestrator receives a human specification written in natural language and converts it into a PDDL (Planning Domain Definition Language) model, where the domain and problem are iteratively refined by sub-modules (agents) to address common planning requirements, such as time constraints and optimality, as well as ambiguities and contradictions that may exist in the human specification. The validated domain and problem are then passed to an external planning engine to generate a plan. The orchestrator and agents are powered by Large Language Models (LLMs) and require no human intervention at any stage of the process. Finally, a module translates the final plan back into natural language to improve human readability while maintaining the correctness of each step. We demonstrate the flexibility and effectiveness of our framework across various domains and tasks, including the Google NaturalPlan benchmark and PlanBench, as well as planning problems like Blocksworld and the Tower of Hanoi (where LLMs are known to struggle even with small instances). Our framework can be integrated with any PDDL planning engine and validator (such as Fast Downward, LPG, POPF, VAL, and uVAL, which we have tested) and represents a significant step toward end-to-end planning aided by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09627v1">LogICL: Distilling LLM Reasoning to Bridge the Semantic Gap in Cross-Domain Log Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Effective log anomaly detection is critical to sustaining reliability in large-scale IT infrastructures. Transformer-based models require substantial resources and labeled data, exacerbating the cold-start problem in target domains where logs are scarce. Existing cross-domain methods leverage source logs but struggle with generalization due to reliance on surface lexical similarity, failing to capture latent semantic equivalence amid structural divergences. To address this, we propose LogICL, a framework distilling Large Language Model (LLM) reasoning into a lightweight encoder for cross-domain anomaly detection. During training, LogICL constructs a delta matrix measuring the utility of demonstrations selected via Maximal Marginal Relevance relative to zero-shot inference. The encoder is optimized via a multi-objective loss comprising an ICL-Guided term that aligns representations based on reasoning assistance utility, maximum mean discrepancy for domain alignment, and supervised contrastive loss. At inference, the optimized encoder retrieves reasoning-aware demonstrations using semantic similarity and delta scores, enabling frozen-LLM in-context learning with Chain-of-Thought for accurate and interpretable detection. Experiments on few-shot and zero-shot cross-domain benchmarks confirm LogICL achieves state-of-the-art performance across heterogeneous systems. Further analysis via visualizations and case studies confirms LogICL bridges the semantic gap beyond surface lexical similarity, effectively capturing latent semantic equivalence for rapid deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2404.02616v2">Improving Topic Relevance Model by Mix-structured Summarization and LLM-based Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-12-10
    </div>
    <details class="paper-abstract">
      Topic relevance between query and document is a very important part of social search, which can evaluate the degree of matching between document and user's requirement. In most social search scenarios such as Dianping, modeling search relevance always faces two challenges. One is that many documents in social search are very long and have much redundant information. The other is that the training data for search relevance model is difficult to get, especially for multi-classification relevance model. To tackle above two problems, we first take query concatenated with the query-based summary and the document summary without query as the input of topic relevance model, which can help model learn the relevance degree between query and the core topic of document. Then, we utilize the language understanding and generation abilities of large language model (LLM) to rewrite and generate query from queries and documents in existing training data, which can construct new query-document pairs as training data. Extensive offline experiments and online A/B tests show that the proposed approaches effectively improve the performance of relevance modeling.
    </details>
</div>
