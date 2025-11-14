# llm - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
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
- [Part 16](papers_16.md)
- [Part 17](papers_17.md)
- Part 18
- [Part 19](papers_19.md)
- [Part 20](papers_20.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17601v5">Revisiting Backdoor Attacks on LLMs: A Stealthy and Practical Poisoning Framework via Harmless Inputs</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Recent studies have widely investigated backdoor attacks on Large Language Models (LLMs) by inserting harmful question-answer (QA) pairs into their training data. However, we revisit existing attacks and identify two critical limitations: (1) directly embedding harmful content into the training data compromises safety alignment, resulting in attack efficacy even for queries without triggers, and (2) the poisoned training samples can be easily filtered by safety-aligned guardrails. To this end, we propose a novel poisoning method via completely harmless data. Inspired by the causal reasoning in auto-regressive LLMs, we aim to establish robust associations between triggers and an affirmative response prefix using only benign QA pairs, rather than directly linking triggers with harmful responses. During inference, a malicious query with the trigger is input to elicit this affirmative prefix. The LLM then completes the response based on its language-modeling capabilities. Achieving this using only clean samples is non-trivial. We observe an interesting resistance phenomenon where the LLM initially appears to agree but subsequently refuses to answer. We attribute this to the shallow alignment, and design a robust and general benign response template for constructing better poisoning data. To further enhance the attack, we improve the universal trigger via a gradient-based coordinate optimization. Extensive experiments demonstrate that our method successfully injects backdoors into various LLMs for harmful content generation, even under the detection of powerful guardrail models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21947v2">Active Attacks: Red-teaming LLMs via Adaptive Environments</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 22 pages, 7 figures, 18 tables
    </div>
    <details class="paper-abstract">
      We address the challenge of generating diverse attack prompts for large language models (LLMs) that elicit harmful behaviors (e.g., insults, sexual content) and are used for safety fine-tuning. Rather than relying on manual prompt engineering, attacker LLMs can be trained with reinforcement learning (RL) to automatically generate such prompts using only a toxicity classifier as a reward. However, capturing a wide range of harmful behaviors is a significant challenge that requires explicit diversity objectives. Existing diversity-seeking RL methods often collapse to limited modes: once high-reward prompts are found, exploration of new regions is discouraged. Inspired by the active learning paradigm that encourages adaptive exploration, we introduce \textit{Active Attacks}, a novel RL-based red-teaming algorithm that adapts its attacks as the victim evolves. By periodically safety fine-tuning the victim LLM with collected attack prompts, rewards in exploited regions diminish, which forces the attacker to seek unexplored vulnerabilities. This process naturally induces an easy-to-hard exploration curriculum, where the attacker progresses beyond easy modes toward increasingly difficult ones. As a result, Active Attacks uncovers a wide range of local attack modes step by step, and their combination achieves wide coverage of the multi-mode distribution. Active Attacks, a simple plug-and-play module that seamlessly integrates into existing RL objectives, unexpectedly outperformed prior RL-based methods -- including GFlowNets, PPO, and REINFORCE -- by improving cross-attack success rates against GFlowNets, the previous state-of-the-art, from 0.07% to 31.28% (a relative gain greater than $400\ \times$) with only a 6% increase in computation. Our code is publicly available \href{https://github.com/dbsxodud-11/active_attacks}{here}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00714v2">RFCAudit: An LLM Agent for Functional Bug Detection in Network Protocols</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Functional correctness is critical for ensuring the reliability and security of network protocol implementations. Functional bugs, instances where implementations diverge from behaviors specified in RFC documents, can lead to severe consequences, including faulty routing, authentication bypasses, and service disruptions. Detecting these bugs requires deep semantic analysis across specification documents and source code, a task beyond the capabilities of traditional static analysis tools. This paper introduces RFCAudit, an autonomous agent that leverages large language models (LLMs) to detect functional bugs by checking conformance between network protocol implementations and their RFC specifications. Inspired by the human auditing procedure, RFCAudit comprises two key components: an indexing agent and a detection agent. The former hierarchically summarizes protocol code semantics, generating semantic indexes that enable the detection agent to narrow down the scanning scope. The latter employs demand-driven retrieval to iteratively collect additional relevant data structures and functions, eventually identifying potential inconsistencies with the RFC specifications effectively. We evaluate RFCAudit across six real-world network protocol implementations. RFCAudit identifies 47 functional bugs with 81.9% precision, of which 20 bugs have been confirmed or fixed by developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03687v1">MedReflect: Teaching Medical LLMs to Self-Improve via Reflective Correction</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Medical problem solving demands expert knowledge and intricate reasoning. Recent studies of large language models (LLMs) attempt to ease this complexity by introducing external knowledge verification through retrieval-augmented generation or by training on reasoning datasets. However, these approaches suffer from drawbacks such as retrieval overhead and high annotation costs, and they heavily rely on substituted external assistants to reach limited performance in medical field. In this paper, we introduce MedReflect, a generalizable framework designed to inspire LLMs with a physician-like reflective thinking mode. MedReflect generates a single-pass reflection chain that includes initial hypothesis generation, self-questioning, self-answering and decision refinement. This self-verified and self-reflective nature releases large language model's latent capability in medical problem-solving without external retrieval or heavy annotation. We demonstrate that MedReflect enables cost-efficient medical dataset construction: with merely 2,000 randomly sampled training examples and a light fine-tuning, this approach achieves notable absolute accuracy improvements across a series of medical benchmarks while cutting annotation requirements. Our results provide evidence that LLMs can learn to solve specialized medical problems via self-reflection and self-improve, reducing reliance on external supervision and extensive task-specific fine-tuning data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03667v1">Invisible Saboteurs: Sycophantic LLMs Mislead Novices in Problem-Solving Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Sycophancy, the tendency of LLM-based chatbots to express excessive enthusiasm, agreement, flattery, and a lack of disagreement, is emerging as a significant risk in human-AI interactions. However, the extent to which this affects human-LLM collaboration in complex problem-solving tasks is not well quantified, especially among novices who are prone to misconceptions. We created two LLM chatbots, one with high sycophancy and one with low sycophancy, and conducted a within-subjects experiment (n=24) in the context of debugging machine learning models to isolate the effect of LLM sycophancy on users' mental models, their workflows, reliance behaviors, and their perceptions of the chatbots. Our findings show that users of the high sycophancy chatbot were less likely to correct their misconceptions and spent more time over-relying on unhelpful LLM responses. Despite these impaired outcomes, a majority of users were unable to detect the presence of excessive sycophancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03662v1">Operationalizing Data Minimization for Privacy-Preserving LLM Prompting</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      The rapid deployment of large language models (LLMs) in consumer applications has led to frequent exchanges of personal information. To obtain useful responses, users often share more than necessary, increasing privacy risks via memorization, context-based personalization, or security breaches. We present a framework to formally define and operationalize data minimization: for a given user prompt and response model, quantifying the least privacy-revealing disclosure that maintains utility, and we propose a priority-queue tree search to locate this optimal point within a privacy-ordered transformation space. We evaluated the framework on four datasets spanning open-ended conversations (ShareGPT, WildChat) and knowledge-intensive tasks with single-ground-truth answers (CaseHold, MedQA), quantifying achievable data minimization with nine LLMs as the response model. Our results demonstrate that larger frontier LLMs can tolerate stronger data minimization while maintaining task quality than smaller open-source models (85.7% redaction for GPT-5 vs. 19.3% for Qwen2.5-0.5B). By comparing with our search-derived benchmarks, we find that LLMs struggle to predict optimal data minimization directly, showing a bias toward abstraction that leads to oversharing. This suggests not just a privacy gap, but a capability gap: models may lack awareness of what information they actually need to solve a task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2308.03688v3">AgentBench: Evaluating LLMs as Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 Published in ICLR 2024
    </div>
    <details class="paper-abstract">
      The potential of Large Language Model (LLM) as agents has been widely acknowledged recently. Thus, there is an urgent need to quantitatively \textit{evaluate LLMs as agents} on challenging tasks in interactive environments. We present AgentBench, a multi-dimensional benchmark that consists of 8 distinct environments to assess LLM-as-Agent's reasoning and decision-making abilities. Our extensive test over \num API-based and open-sourced (OSS) LLMs shows that, while top commercial LLMs present a strong ability of acting as agents in complex environments, there is a significant disparity in performance between them and many OSS competitors that are no larger than 70B. We identify the typical reasons of failures in environments and LLMs, showing that poor long-term reasoning, decision-making, and instruction following abilities are the main obstacles for developing usable LLM agents. Improving instruction following and training on high quality multi-round alignment data could improve agent performance. And different from existing assumptions, training on code present ambivalent impacts on different agent tasks. Datasets, environments, and an integrated evaluation package for AgentBench are released at https://github.com/THUDM/AgentBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05370v2">Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained immense success in revolutionizing various applications, including content generation, search and recommendation, and AI-assisted operation. To reduce high training costs, Mixture-of-Experts (MoE) architecture has become a popular backbone for modern LLMs. However, despite the benefits, serving MoE-based LLMs experience severe memory inefficiency due to sparsely activated experts. Recent studies propose to offload inactive experts from GPU memory to CPU memory to improve the serving efficiency of MoE models. However, they either incur high inference latency or high model memory footprints due to coarse-grained designs. To tame the latency-memory trade-off in MoE serving, we present FineMoE, a fine-grained expert offloading system for MoE serving that achieves low inference latency with memory efficiency. We design FineMoE to extract fine-grained expert selection patterns from MoE models and semantic hints from input prompts to efficiently guide expert prefetching, caching, and offloading decisions. FineMoE is prototyped on top of HuggingFace Transformers and deployed on a six-GPU testbed. Experiments with open-source MoE models and real-world workloads show that FineMoE reduces inference latency by 47% and improves expert hit rate by 39% over state-of-the-art solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03650v1">LLM-Guided Evolutionary Program Synthesis for Quasi-Monte Carlo Design</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Low-discrepancy point sets and digital sequences underpin quasi-Monte Carlo (QMC) methods for high-dimensional integration. We cast two long-standing QMC design problems as program synthesis and solve them with an LLM-guided evolutionary loop that mutates and selects code under task-specific fitness: (i) constructing finite 2D/3D point sets with low star discrepancy, and (ii) choosing Sobol' direction numbers that minimize randomized QMC error on downstream integrands. Our two-phase procedure combines constructive code proposals with iterative numerical refinement. On finite sets, we rediscover known optima in small 2D cases and set new best-known 2D benchmarks for N >= 40, while matching most known 3D optima up to the proven frontier (N <= 8) and reporting improved 3D benchmarks beyond. On digital sequences, evolving Sobol' parameters yields consistent reductions in randomized quasi-Monte Carlo (rQMC) mean-squared error for several 32-dimensional option-pricing tasks relative to widely used Joe--Kuo parameters, while preserving extensibility to any sample size and compatibility with standard randomizations. Taken together, the results demonstrate that LLM-driven evolutionary program synthesis can automate the discovery of high-quality QMC constructions, recovering classical designs where they are optimal and improving them where finite-N structure matters. Data and code are available at https://github.com/hockeyguy123/openevolve-star-discrepancy.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24836v3">Pushing LLMs to Their Logical Reasoning Bound: The Role of Data Reasoning Intensity</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) highlight the importance of training data structure and quality in shaping reasoning behavior. However, most existing approaches focus on transforming data formats while neglecting the internal reasoning complexity of training samples, leaving the reasoning potential of data under-explored and underutilized. In this work, we posit that LLM logical reasoning performance is jointly constrained by the potential of the training data and the cognitive capacity of the model. To make this relationship measurable, we introduce Data Reasoning Intensity (DRI), a novel metric that quantifies the latent logical reasoning complexity of samples by decomposing and aggregating their logical structures. This allows us to analyze how well current LLMs utilize logical reasoning signals and identify performance gaps relative to data potential. Based on this insight, we introduce a re-cognizing optimization strategy that systematically enhances the logical reasoning intensity of training data. Rather than increasing data volume, our method re-optimizes existing samples to better align with the LLM's logical reasoning boundary. Extensive experiments show that our approach significantly improves performance and generalization over data-centric strategies. We further validate our method under a reinforcement learning framework. Our results indicate that prioritizing reasoning complexity in data rather than sheer scale or superficial form is essential to realizing LLMs' full cognitive potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03641v1">Generating High-Level Test Cases from Requirements using LLM: An Industry Study</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 11pages
    </div>
    <details class="paper-abstract">
      Currently, generating high-level test cases described in natural language from requirement documents is performed manually. In the industry, including companies specializing in software testing, there is a significant demand for the automatic generation of high-level test cases from requirement documents using Large Language Models (LLMs). Efforts to utilize LLMs for requirement analysis are underway. In some cases, retrieval-augmented generation (RAG) is employed for generating high-level test cases using LLMs. However, in practical applications, it is necessary to create a RAG tailored to the knowledge system of each specific application, which is labor-intensive. Moreover, when applying high-level test case generation as a prompt, there is no established method for instructing the generation of high-level test cases at a level applicable to other specifications without using RAG. It is required to establish a method for the automatic generation of high-level test cases that can be generalized across a wider range of requirement documents. In this paper, we propose a method for generating high-level (GHL) test cases from requirement documents using only prompts, without creating RAGs. In the proposed method, first, the requirement document is input into the LLM to generate test design techniques corresponding to the requirement document. Then, high-level test cases are generated for each of the generated test design techniques. Furthermore, we verify an evaluation method based on semantic similarity of the generated high-level test cases. In the experiments, we confirmed the method using datasets from Bluetooth and Mozilla, where requirement documents and high-level test cases are available, achieving macro-recall measurement of 0.81 and 0.37, respectively. We believe that the method is feasible for practical application in generating high-level test cases without using RAG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14794v2">Characterizing Mobile SoC for Accelerating Heterogeneous LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      With the rapid advancement of artificial intelligence technologies such as ChatGPT, AI agents, and video generation, contemporary mobile systems have begun integrating these AI capabilities on local devices to enhance privacy and reduce response latency. To meet the computational demands of AI tasks, current mobile SoCs are equipped with diverse AI accelerators, including GPUs and Neural Processing Units (NPUs). However, there has not been a comprehensive characterization of these heterogeneous processors, and existing designs typically only leverage a single AI accelerator for LLM inference, leading to suboptimal use of computational resources and memory bandwidth. In this paper, we first summarize key performance characteristics of heterogeneous processors, SoC memory bandwidth, etc. Drawing on these observations, we propose different heterogeneous parallel mechanisms to fully exploit both GPU and NPU computational power and memory bandwidth. We further design a fast synchronization mechanism between heterogeneous processors that leverages the unified memory architecture. By employing these techniques, we present HeteroInfer, the fastest LLM inference engine in mobile devices which supports GPU-NPU heterogeneous execution. Evaluation shows that HeteroInfer delivers a 1.34x to 6.02x end-to-end speedup over state-of-the-art GPU-only and NPU-only LLM engines, while maintaining negligible interference with other applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03632v1">MITS: Enhanced Tree Search Reasoning for LLMs via Pointwise Mutual Information</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Tree search has become as a representative framework for test-time reasoning with large language models (LLMs), exemplified by methods such as Tree-of-Thought and Monte Carlo Tree Search that explore multiple reasoning paths. However, it remains difficult to provide instant and reliable quantitative assessments of intermediate reasoning step quality, and extensive path exploration is computationally costly. To address this, we propose Mutual Information Tree Search (MITS), a novel framework that guides reasoning with information-theoretic principles. MITS introduces an effective scoring function based on pointwise mutual information (PMI), which enables step-wise evaluation of reasoning paths and search tree expansion via beam search without expensive look-ahead simulations, achieving superior reasoning performances while maintaining computational efficiency. The framework is complemented by an entropy-based dynamic sampling strategy that adaptively allocates computational resources to uncertain reasoning steps where exploration is most beneficial. For final prediction, MITS employs a weighted voting scheme that combines PMI scores with prediction consensus. Through comprehensive experiments on diverse reasoning benchmarks, MITS consistently surpasses baseline methods, establishing a principled and efficient framework for LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.03645v2">Graph Generation Powered with LLMs for Boosting Multivariate Time-Series Representation Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Sourced from multiple sensors and organized chronologically, Multivariate Time-Series (MTS) data involves crucial spatial-temporal dependencies. To capture these dependencies, Graph Neural Networks (GNNs) have emerged as powerful tools. As explicit graphs are not inherent to MTS data, graph generation becomes a critical first step in adapting GNNs to this domain. However, existing approaches often rely solely on the data itself for MTS graph generation, leaving them vulnerable to biases from small training datasets. This limitation hampers their ability to construct effective graphs, undermining the accurate modeling of underlying dependencies in MTS data and reducing GNN performance in this field. To address this challenge, we propose a novel framework, K-Link, leveraging the extensive universal knowledge encoded in Large Language Models (LLMs) to reduce biases for powered MTS graph generation. To harness the knowledge within LLMs, such as physical principles, we design and extract a \textit{Knowledge-Link graph} that captures universal knowledge of sensors and their linkage. To empower MTS graph generation with the knowledge-link graph, we further introduce a graph alignment module that transfers universal knowledge from the knowledge-link graph to the graph generated from MTS data. This enhances the MTS graph quality, ensuring effective representation learning for MTS data. Extensive experiments demonstrate the efficacy of K-Link for superior performance on various MTS tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.20293v2">When Judgment Becomes Noise: How Design Failures in LLM Judge Benchmarks Silently Undermine Validity</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      LLM-judged benchmarks are increasingly used to evaluate complex model behaviors, yet their design introduces failure modes absent in conventional ground-truth based benchmarks. We argue that without tight objectives and verifiable constructions, benchmark rankings can produce high-confidence rankings that are in fact largely noise. We introduce two mechanisms to diagnose these issues. Schematic adherence quantifies how much of a judge's overall verdict is explained by the explicit evaluation schema, revealing unexplained variance when judges deviate from their own rubric. Psychometric validity aggregates internal consistency and discriminant validity signals to quantify irreducible uncertainty in any benchmarking run. Applying these tools to Arena-Hard Auto, we find severe schema incoherence and factor collapse across popular judges: for example, unexplained variance exceeding 90 percent for DeepSeek-R1-32B and factor correlations above 0.93 for most criteria. We also show that the ELO-style aggregation used by Arena-Hard Auto collapses and masks genuine ranking uncertainty. Our results highlight design failures that undermine validity and offer actionable principles for building better-scoped, reliability-aware LLM-judged benchmarks. We released our code and dataset at https://github.com/penfever/judgment-to-noise
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03611v1">Can an LLM Induce a Graph? Investigating Memory Drift and Context Length</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 2025 IEEE International Conference on Knowledge Graph (ICKG)
    </div>
    <details class="paper-abstract">
      Recently proposed evaluation benchmarks aim to characterize the effective context length and the forgetting tendencies of large language models (LLMs). However, these benchmarks often rely on simplistic 'needle in a haystack' retrieval or continuation tasks that may not accurately reflect the performance of these models in information-dense scenarios. Thus, rather than simple next token prediction, we argue for evaluating these models on more complex reasoning tasks that requires them to induce structured relational knowledge from the text - such as graphs from potentially noisy natural language content. While the input text can be viewed as generated in terms of a graph, its structure is not made explicit and connections must be induced from distributed textual cues, separated by long contexts and interspersed with irrelevant information. Our findings reveal that LLMs begin to exhibit memory drift and contextual forgetting at much shorter effective lengths when tasked with this form of relational reasoning, compared to what existing benchmarks suggest. With these findings, we offer recommendations for the optimal use of popular LLMs for complex reasoning tasks. We further show that even models specialized for reasoning, such as OpenAI o1, remain vulnerable to early memory drift in these settings. These results point to significant limitations in the models' ability to abstract structured knowledge from unstructured input and highlight the need for architectural adaptations to improve long-range reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18350v2">How Many Parameters Does Your Task Really Need? Task Specific Pruning with LLM-Sieve</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed for narrow tasks in resource-constrained settings, a central question arises: how much of an LLM is truly necessary for a given task? We present LLM-Sieve, a framework that prunes LLMs down to the minimal parameter subset needed to preserve task performance. Our approach introduces two innovations: (i) output-aligned non-orthogonal projections, which yield more faithful low-rank approximations than traditional PCA/SVD by aligning directly with layer outputs; and (ii) adaptive pruning via a Genetic Algorithm, which automatically discovers matrix-specific pruning levels and exposes the uneven distribution of task-relevant knowledge. Across models from 3.8B to 70B parameters, LLM-Sieve removes 20-75% of weights with only 1-5% accuracy loss-substantially ahead of prior pruning methods. Beyond efficiency, our framework reveals bottleneck matrices that concentrate critical knowledge, suggesting architectural implications for future LLM design. LLM-Sieve integrates seamlessly with LoRA fine-tuning and quantization, enabling both efficient deployment and deeper understanding of knowledge organization in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01315v2">Wired for Reuse: Automating Context-Aware Code Adaptation in IDEs via LLM-Based Agent</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 Accepted by ASE2025
    </div>
    <details class="paper-abstract">
      Copy-paste-modify is a widespread and pragmatic practice in software development, where developers adapt reused code snippets, sourced from platforms such as Stack Overflow, GitHub, or LLM outputs, into their local codebase. A critical yet underexplored aspect of this adaptation is code wiring: the context-aware process of substituting unresolved variables in pasted code with suitable variables or expressions from the surrounding context. Existing solutions either rely on heuristic rules or historical templates, often failing to effectively utilize contextual information, despite studies showing that over half of adaptation cases are context-dependent. In this paper, we introduce WIRL, an LLM-based agent for code wiring framed as a Retrieval-Augmented Generation (RAG) infilling task. WIRL combines an LLM, a customized toolkit, and an orchestration module to identify unresolved variables, retrieve context, and perform context-aware substitutions. To balance efficiency and autonomy, the agent adopts a mixed strategy: deterministic rule-based steps for common patterns, and a state-machine-guided decision process for intelligent exploration. We evaluate WIRL on a carefully curated, high-quality dataset consisting of real-world code adaptation scenarios. Our approach achieves an exact match precision of 91.7% and a recall of 90.0%, outperforming advanced LLMs by 22.6 and 13.7 percentage points in precision and recall, respectively, and surpassing IntelliJ IDEA by 54.3 and 49.9 percentage points. These results underscore its practical utility, particularly in contexts with complex variable dependencies or multiple unresolved variables. We believe WIRL paves the way for more intelligent and context-aware developer assistance in modern IDEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23252v2">NanoFlux: Adversarial Dual-LLM Evaluation and Distillation For Multi-Domain Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 preprint version
    </div>
    <details class="paper-abstract">
      We present NanoFlux, a novel adversarial framework for generating targeted training data to improve LLM reasoning, where adversarially-generated datasets containing fewer than 200 examples outperform conventional fine-tuning approaches. The framework employs a competitive dynamic between models alternating as Attacker and Defender, supervised by a tool-augmented Judge, synthesizing multi-step questions with explanatory annotations that target specific reasoning capabilities. Fine-tuning a 4B-parameter model on NanoFlux-generated data yields performance gains across diverse domains compared to full-benchmark fine-tuning: +5.9% on mathematical reasoning (GSMHard), +3.6% on scientific reasoning (GenomeBench), and +16.6% on medical reasoning (MultiMedQA), while reducing computational requirements by 3-14x. Ablation studies reveal a non-monotonic relationship between dataset characteristics and model performance, uncovering domain-specific optimal points for question complexity and reasoning quality. NanoFlux automates training data generation through embedding-based novelty filtering, tool-augmented evaluation, and multi-hop reasoning, suggesting that future model improvements may lie in the intelligent synthesis of small, precisely targeted training datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03595v1">Decoupling Task-Solving and Output Formatting in LLM Generation</a></div>
    <div class="paper-meta">
      📅 2025-10-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly adept at following instructions containing task descriptions to solve complex problems, such as mathematical reasoning and automatic evaluation (LLM-as-a-Judge). However, as prompts grow more complex, models often struggle to adhere to all instructions. This difficulty is especially common when instructive prompts intertwine reasoning directives -- specifying what the model should solve -- with rigid formatting requirements that dictate how the solution must be presented. The entanglement creates competing goals for the model, suggesting that more explicit separation of these two aspects could lead to improved performance. To this front, we introduce Deco-G, a decoding framework that explicitly decouples format adherence from task solving. Deco-G handles format compliance with a separate tractable probabilistic model (TPM), while prompts LLMs with only task instructions. At each decoding step, Deco-G combines next token probabilities from the LLM with the TPM calculated format compliance likelihood to form the output probability. To make this approach both practical and scalable for modern instruction-tuned LLMs, we introduce three key innovations: instruction-aware distillation, a flexible trie-building algorithm, and HMM state pruning for computational efficiency. We demonstrate the effectiveness of Deco-G across a wide range of tasks with diverse format requirements, including mathematical reasoning, LLM-as-a-judge, and event argument extraction. Overall, our approach yields 1.0% to 6.0% relative gain over regular prompting practice with guaranteed format compliance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03930v1">LLM Chemistry Estimation for Multi-LLM Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 20 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Multi-LLM collaboration promises accurate, robust, and context-aware solutions, yet existing approaches rely on implicit selection and output assessment without analyzing whether collaborating models truly complement or conflict. We introduce LLM Chemistry -- a framework that measures when LLM combinations exhibit synergistic or antagonistic behaviors that shape collective performance beyond individual capabilities. We formalize the notion of chemistry among LLMs, propose algorithms that quantify it by analyzing interaction dependencies, and recommend optimal model ensembles accordingly. Our theoretical analysis shows that chemistry among collaborating LLMs is most evident under heterogeneous model profiles, with its outcome impact shaped by task type, group size, and complexity. Evaluation on classification, summarization, and program repair tasks provides initial evidence for these task-dependent effects, thereby reinforcing our theoretical results. This establishes LLM Chemistry as both a diagnostic factor in multi-LLM systems and a foundation for ensemble recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2308.03688v3">AgentBench: Evaluating LLMs as Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-04
      | 💬 Published in ICLR 2024
    </div>
    <details class="paper-abstract">
      The potential of Large Language Model (LLM) as agents has been widely acknowledged recently. Thus, there is an urgent need to quantitatively \textit{evaluate LLMs as agents} on challenging tasks in interactive environments. We present AgentBench, a multi-dimensional benchmark that consists of 8 distinct environments to assess LLM-as-Agent's reasoning and decision-making abilities. Our extensive test over \num API-based and open-sourced (OSS) LLMs shows that, while top commercial LLMs present a strong ability of acting as agents in complex environments, there is a significant disparity in performance between them and many OSS competitors that are no larger than 70B. We identify the typical reasons of failures in environments and LLMs, showing that poor long-term reasoning, decision-making, and instruction following abilities are the main obstacles for developing usable LLM agents. Improving instruction following and training on high quality multi-round alignment data could improve agent performance. And different from existing assumptions, training on code present ambivalent impacts on different agent tasks. Datasets, environments, and an integrated evaluation package for AgentBench are released at https://github.com/THUDM/AgentBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02917v1">Mechanistic Interpretability of Code Correctness in LLMs via Sparse Autoencoders</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      As Large Language Models become integral to software development, with substantial portions of AI-suggested code entering production, understanding their internal correctness mechanisms becomes critical for safe deployment. We apply sparse autoencoders to decompose LLM representations, identifying directions that correspond to code correctness. We select predictor directions using t-statistics and steering directions through separation scores from base model representations, then analyze their mechanistic properties through steering, attention analysis, and weight orthogonalization. We find that code correctness directions in LLMs reliably predict incorrect code, while correction capabilities, though statistically significant, involve tradeoffs between fixing errors and preserving correct code. Mechanistically, successful code generation depends on attending to test cases rather than problem descriptions. Moreover, directions identified in base models retain their effectiveness after instruction-tuning, suggesting code correctness mechanisms learned during pre-training are repurposed during fine-tuning. Our mechanistic insights suggest three practical applications: prompting strategies should prioritize test examples over elaborate problem descriptions, predictor directions can serve as error alarms for developer review, and these same predictors can guide selective steering, intervening only when errors are anticipated to prevent the code corruption from constant steering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.23799v2">Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 19 pages, 11 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Steering has emerged as a promising approach in controlling large language models (LLMs) without modifying model parameters. However, most existing steering methods rely on large-scale datasets to learn clear behavioral information, which limits their applicability in many real-world scenarios. The steering vectors extracted from small dataset often contain task-irrelevant noising features, which degrades their effectiveness. To refine the steering vectors learned from limited data, we introduce Refinement of Steering Vector via Sparse Autoencoder (SAE-RSV) that leverages SAEs to semantically denoise and augment the steering vectors. In our framework, we first remove task-irrelevant features according to their semantics provided by SAEs, and then enrich task-relevant features missing from the small dataset through their semantic similarity to the identified relevant features. Extensive experiments demonstrate that the proposed SAE-RSV substantially outperforms all the baseline methods including supervised fine-tuning. Our findings show that effective steering vector can be constructed from limited training data by refining the original steering vector through SAEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02833v1">Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Despite substantial efforts in safety alignment, recent research indicates that Large Language Models (LLMs) remain highly susceptible to jailbreak attacks. Among these attacks, finetuning-based ones that compromise LLMs' safety alignment via fine-tuning stand out due to its stable jailbreak performance. In particular, a recent study indicates that fine-tuning with as few as 10 harmful question-answer (QA) pairs can lead to successful jailbreaking across various harmful questions. However, such malicious fine-tuning attacks are readily detectable and hence thwarted by moderation models. In this paper, we demonstrate that LLMs can be jailbroken by fine-tuning with only 10 benign QA pairs; our attack exploits the increased sensitivity of LLMs to fine-tuning data after being overfitted. Specifically, our fine-tuning process starts with overfitting an LLM via fine-tuning with benign QA pairs involving identical refusal answers. Further fine-tuning is then performed with standard benign answers, causing the overfitted LLM to forget the refusal attitude and thus provide compliant answers regardless of the harmfulness of a question. We implement our attack on the ten LLMs and compare it with five existing baselines. Experiments demonstrate that our method achieves significant advantages in both attack effectiveness and attack stealth. Our findings expose previously unreported security vulnerabilities in current LLMs and provide a new perspective on understanding how LLMs' security is compromised, even with benign fine-tuning. Our code is available at https://github.com/ZHIXINXIE/tenBenign.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21700v2">XBreaking: Explainable Artificial Intelligence for Jailbreaking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Large Language Models are fundamental actors in the modern IT landscape dominated by AI solutions. However, security threats associated with them might prevent their reliable adoption in critical application scenarios such as government organizations and medical institutions. For this reason, commercial LLMs typically undergo a sophisticated censoring mechanism to eliminate any harmful output they could possibly produce. In response to this, LLM Jailbreaking is a significant threat to such protections, and many previous approaches have already demonstrated its effectiveness across diverse domains. Existing jailbreak proposals mostly adopt a generate-and-test strategy to craft malicious input. To improve the comprehension of censoring mechanisms and design a targeted jailbreak attack, we propose an Explainable-AI solution that comparatively analyzes the behavior of censored and uncensored models to derive unique exploitable alignment patterns. Then, we propose XBreaking, a novel jailbreak attack that exploits these unique patterns to break the security constraints of LLMs by targeted noise injection. Our thorough experimental campaign returns important insights about the censoring mechanisms and demonstrates the effectiveness and performance of our attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02816v1">NCV: A Node-Wise Consistency Verification Approach for Low-Cost Structured Error Localization in LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Verifying multi-step reasoning in large language models is difficult due to imprecise error localization and high token costs. Existing methods either assess entire reasoning chains, suffering attention dilution, or rely on expensive multi-sampling. We introduce Node-wise Consistency Verification (NCV), a training-free framework that recasts verification as lightweight binary consistency checks at the node level. By decomposing the chain of thought into interconnected verification nodes, NCV precisely localizes errors and avoids unnecessary long-form generation. Experiments demonstrate that our approach enhances interpretability and efficiency, presenting a scalable solution for reliable LLM reasoning verification. On public datasets, NCV achieves a 10\% to 25\% improvement in F1 scores over baselines while utilizing $6\times$~$58\times$ fewer tokens than traditional methods like CoT-based verifiers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11191v3">Primus: A Pioneering Collection of Open-Source Datasets for Cybersecurity LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable advancements in specialized fields such as finance, law, and medicine. However, in cybersecurity, we have noticed a lack of open-source datasets, with a particular lack of high-quality cybersecurity pretraining corpora, even though much research indicates that LLMs acquire their knowledge during pretraining. To address this, we present a comprehensive suite of datasets covering all major training stages, including pretraining, instruction fine-tuning, and reasoning distillation with cybersecurity-specific self-reflection data. Extensive ablation studies demonstrate their effectiveness on public cybersecurity benchmarks. In particular, continual pre-training on our dataset yields a 15.9% improvement in the aggregate score, while reasoning distillation leads to a 15.8% gain in security certification (CISSP). We will release all datasets and trained cybersecurity LLMs under the ODC-BY and MIT licenses to encourage further research in the community. For access to all datasets and model weights, please refer to https://huggingface.co/collections/trendmicro-ailab/primus-67b1fd27052b802b4af9d243.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02758v1">TokenFlow: Responsive LLM Text Streaming Serving under Request Burst via Preemptive Scheduling</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Accepted by EuroSys 2026
    </div>
    <details class="paper-abstract">
      Real-time LLM interactions demand streamed token generations, where text tokens are progressively generated and delivered to users while balancing two objectives: responsiveness (i.e., low time-to-first-token) and steady generation (i.e.,required time-between-tokens). Standard LLM serving systems suffer from the inflexibility caused by non-preemptive request scheduling and reactive memory management, leading to poor resource utilization and low request processing parallelism under request bursts. Therefore, we present TokenFlow, a novel LLM serving system with enhanced text streaming performance via preemptive request scheduling and proactive key-value (KV) cache management. TokenFlow dynamically prioritizes requests based on real-time token buffer occupancy and token consumption rate, while actively transferring KV cache between GPU and CPU memory in the background and overlapping I/O with computation to minimize request preemption overhead. Extensive experiments on Llama3-8B and Qwen2.5-32B across multiple GPUs (RTX 4090, A6000, H200) demonstrate that TokenFlow achieves up to 82.5% higher effective throughput (accounting for actual user consumption) while reducing P99 TTFT by up to 80.2%, without degrading overall token throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00415v2">MarketSenseAI 2.0: Enhancing Stock Analysis through LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 25 pages, 7 figures, Under review at Financial Innovation (FIN)
    </div>
    <details class="paper-abstract">
      MarketSenseAI is a novel framework for holistic stock analysis which leverages Large Language Models (LLMs) to process financial news, historical prices, company fundamentals and the macroeconomic environment to support decision making in stock analysis and selection. In this paper, we present the latest advancements on MarketSenseAI, driven by rapid technological expansion in LLMs. Through a novel architecture combining Retrieval-Augmented Generation and LLM agents, the framework processes SEC filings and earnings calls, while enriching macroeconomic analysis through systematic processing of diverse institutional reports. We demonstrate a significant improvement in fundamental analysis accuracy over the previous version. Empirical evaluation on S\&P 100 stocks over two years (2023-2024) shows MarketSenseAI achieving cumulative returns of 125.9% compared to the index return of 73.5%, while maintaining comparable risk profiles. Further validation on S\&P 500 stocks during 2024 demonstrates the framework's scalability, delivering a 33.8% higher Sortino ratio than the market. This work marks a significant advancement in applying LLM technology to financial analysis, offering insights into the robustness of LLM-driven investment strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02742v1">IndiCASA: A Dataset and Bias Evaluation Framework in LLMs Using Contrastive Embedding Similarity in the Indian Context</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Accepted at 8th AAAI/ACM Conference on AI, Ethics, and Society (AIES) 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained significant traction across critical domains owing to their impressive contextual understanding and generative capabilities. However, their increasing deployment in high stakes applications necessitates rigorous evaluation of embedded biases, particularly in culturally diverse contexts like India where existing embedding-based bias assessment methods often fall short in capturing nuanced stereotypes. We propose an evaluation framework based on a encoder trained using contrastive learning that captures fine-grained bias through embedding similarity. We also introduce a novel dataset - IndiCASA (IndiBias-based Contextually Aligned Stereotypes and Anti-stereotypes) comprising 2,575 human-validated sentences spanning five demographic axes: caste, gender, religion, disability, and socioeconomic status. Our evaluation of multiple open-weight LLMs reveals that all models exhibit some degree of stereotypical bias, with disability related biases being notably persistent, and religion bias generally lower likely due to global debiasing efforts demonstrating the need for fairer model development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02719v1">TravelBench : Exploring LLM Performance in Low-Resource Domains</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 10 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Results on existing LLM benchmarks capture little information over the model capabilities in low-resource tasks, making it difficult to develop effective solutions in these domains. To address these challenges, we curated 14 travel-domain datasets spanning 7 common NLP tasks using anonymised data from real-world scenarios, and analysed the performance across LLMs. We report on the accuracy, scaling behaviour, and reasoning capabilities of LLMs in a variety of tasks. Our results confirm that general benchmarking results are insufficient for understanding model performance in low-resource tasks. Despite the amount of training FLOPs, out-of-the-box LLMs hit performance bottlenecks in complex, domain-specific scenarios. Furthermore, reasoning provides a more significant boost for smaller LLMs by making the model a better judge on certain tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02716v1">A $1000\times$ Faster LLM-enhanced Algorithm For Path Planning in Large-scale Grid Maps</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Path planning in grid maps, arising from various applications, has garnered significant attention. Existing methods, such as A*, Dijkstra, and their variants, work well for small-scale maps but fail to address large-scale ones due to high search time and memory consumption. Recently, Large Language Models (LLMs) have shown remarkable performance in path planning but still suffer from spatial illusion and poor planning performance. Among all the works, LLM-A* \cite{meng2024llm} leverages LLM to generate a series of waypoints and then uses A* to plan the paths between the neighboring waypoints. In this way, the complete path is constructed. However, LLM-A* still suffers from high computational time for large-scale maps. To fill this gap, we conducted a deep investigation into LLM-A* and found its bottleneck, resulting in limited performance. Accordingly, we design an innovative LLM-enhanced algorithm, abbr. as iLLM-A*. iLLM-A* includes 3 carefully designed mechanisms, including the optimization of A*, an incremental learning method for LLM to generate high-quality waypoints, and the selection of the appropriate waypoints for A* for path planning. Finally, a comprehensive evaluation on various grid maps shows that, compared with LLM-A*, iLLM-A* \textbf{1) achieves more than $1000\times$ speedup on average, and up to $2349.5\times$ speedup in the extreme case, 2) saves up to $58.6\%$ of the memory cost, 3) achieves both obviously shorter path length and lower path length standard deviation.}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02694v1">MALF: A Multi-Agent LLM Framework for Intelligent Fuzzing of Industrial Control Protocols</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Industrial control systems (ICS) are vital to modern infrastructure but increasingly vulnerable to cybersecurity threats, particularly through weaknesses in their communication protocols. This paper presents MALF (Multi-Agent LLM Fuzzing Framework), an advanced fuzzing solution that integrates large language models (LLMs) with multi-agent coordination to identify vulnerabilities in industrial control protocols (ICPs). By leveraging Retrieval-Augmented Generation (RAG) for domain-specific knowledge and QLoRA fine-tuning for protocol-aware input generation, MALF enhances fuzz testing precision and adaptability. The multi-agent framework optimizes seed generation, mutation strategies, and feedback-driven refinement, leading to improved vulnerability discovery. Experiments on protocols like Modbus/TCP, S7Comm, and Ethernet/IP demonstrate that MALF surpasses traditional methods, achieving a test case pass rate (TCPR) of 88-92% and generating more exception triggers (ETN). MALF also maintains over 90% seed coverage and Shannon entropy values between 4.2 and 4.6 bits, ensuring diverse, protocol-compliant mutations. Deployed in a real-world Industrial Attack-Defense Range for power plants, MALF identified critical vulnerabilities, including three zero-day flaws, one confirmed and registered by CNVD. These results validate MALF's effectiveness in real-world fuzzing applications. This research highlights the transformative potential of multi-agent LLMs in ICS cybersecurity, offering a scalable, automated framework that sets a new standard for vulnerability discovery and strengthens critical infrastructure security against emerging threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02675v1">HALO: Memory-Centric Heterogeneous Accelerator with 2.5D Integration for Low-Batch LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      The rapid adoption of Large Language Models (LLMs) has driven a growing demand for efficient inference, particularly in latency-sensitive applications such as chatbots and personalized assistants. Unlike traditional deep neural networks, LLM inference proceeds in two distinct phases: the prefill phase, which processes the full input sequence in parallel, and the decode phase, which generates tokens sequentially. These phases exhibit highly diverse compute and memory requirements, which makes accelerator design particularly challenging. Prior works have primarily been optimized for high-batch inference or evaluated only short input context lengths, leaving the low-batch and long context regime, which is critical for interactive applications, largely underexplored. We propose HALO, a heterogeneous memory centric accelerator designed for these unique challenges of prefill and decode phases in low-batch LLM inference. HALO integrates HBM based Compute-in-DRAM (CiD) with an on-chip analog Compute-in-Memory (CiM), co-packaged using 2.5D integration. To further improve the hardware utilization, we introduce a phase-aware mapping strategy that adapts to the distinct demands of the prefill and decode phases. Compute bound operations in the prefill phase are mapped to CiM to exploit its high throughput matrix multiplication capability, while memory-bound operations in the decode phase are executed on CiD to benefit from reduced data movement within DRAM. Additionally, we present an analysis of the performance tradeoffs of LLMs under two architectural extremes: a fully CiD and a fully on-chip analog CiM design to highlight the need for a heterogeneous design. We evaluate HALO on LLaMA-2 7B and Qwen3 8B models. Our experimental results show that LLMs mapped to HALO achieve up to 18x geometric mean speedup over AttAcc, an attention-optimized mapping and 2.5x over CENT, a fully CiD based mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00907v5">Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Embodied agents operating in household environments must interpret ambiguous and under-specified human instructions. A capable household robot should recognize ambiguity and ask relevant clarification questions to infer the user intent accurately, leading to more effective task execution. To study this problem, we introduce the Ask-to-Act task, where an embodied agent is tasked with a single or multi-object rearrangement task using an under-specified instruction in a home environment. The agent must strategically ask minimal, yet relevant, clarification questions to resolve ambiguity while navigating under partial observability. To address this challenge, we propose a novel approach that fine-tunes multi-modal large language models (MLLMs) as vision-language-action (VLA) policies using online reinforcement learning (RL) with LLM-generated rewards. Our method eliminates the need for large-scale human demonstrations or manually engineered rewards for training such agents. We benchmark against strong zero-shot baselines including GPT-4o as well as supervised fine-tuned MLLMs on our task. Our results show that our RL-finetuned MLLM outperforms all baselines by a significant margin (10.4-16.5%), generalizing well to novel scenes and tasks. To the best of our knowledge, this is the first demonstration of adapting MLLMs as VLA agents that can act and ask for help using LLM-generated rewards with online RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02657v1">Less LLM, More Documents: Searching for Improved RAG</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 16 pages. Submitted to ECIR 2026
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) couples document retrieval with large language models (LLMs). While scaling generators improves accuracy, it also raises cost and limits deployability. We explore an orthogonal axis: enlarging the retriever's corpus to reduce reliance on large LLMs. Experimental results show that corpus scaling consistently strengthens RAG and can often serve as a substitute for increasing model size, though with diminishing returns at larger scales. Small- and mid-sized generators paired with larger corpora often rival much larger models with smaller corpora; mid-sized models tend to gain the most, while tiny and large models benefit less. Our analysis shows that improvements arise primarily from increased coverage of answer-bearing passages, while utilization efficiency remains largely unchanged. These findings establish a principled corpus-generator trade-off: investing in larger corpora offers an effective path to stronger RAG, often comparable to enlarging the LLM itself.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02645v1">Mind the Gap: Linguistic Divergence and Adaptation Strategies in Human-LLM Assistant vs. Human-Human Interactions</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Accepted to The Second Workshop on Generative AI for E-commerce (GenAIECommerce '25), held September 22, 2025, in Prague, Czech Republic
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are increasingly deployed in customer-facing applications, a critical yet underexplored question is how users communicate differently with LLM chatbots compared to human agent. In this study, we present empirical evidence that users adopt distinct communication styles when users interact with chatbots versus human agents. Our analysis reveals significant differences in grammatical fluency, politeness, and lexical diversity in user language between the two settings. These findings suggest that models trained exclusively on human-human interaction data may not adequately accommodate the communication style shift that occurs once an LLM chatbot is deployed. To enhance LLM robustness to post-launch communication style changes, we experimented with two strategies: (1) data augmentation during the post-training phase and (2) inference-time user message reformulation. Our results indicate that models trained on stylistically diverse datasets significantly outperform those trained exclusively on original or stylistically uniform datasets, while inference-time reformulation proved less effective. These insights help us to better adapt our models for improved LLM-user interaction experiences.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02637v1">Homophily-induced Emergence of Biased Structures in LLM-based Multi-Agent AI Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Accepted for publication in Social Network Analysis and Mining
    </div>
    <details class="paper-abstract">
      This study examines how interactions among artificially intelligent (AI) agents, guided by large language models (LLMs), drive the evolution of collective network structures. We ask LLM-driven agents to grow a network by informing them about current link constellations. Our observations confirm that agents consistently apply a preferential attachment mechanism, favoring connections to nodes with higher degrees. We systematically solicited more than a million decisions from four different LLMs, including Gemini, ChatGPT, Llama, and Claude. When social attributes such as age, gender, religion, and political orientation are incorporated, the resulting networks exhibit heightened assortativity, leading to the formation of distinct homophilic communities. This significantly alters the network topology from what would be expected under a pure preferential attachment model alone. Political and religious attributes most significantly fragment the collective, fostering polarized subgroups, while age and gender yield more gradual structural shifts. Strikingly, LLMs also reveal asymmetric patterns in heterophilous ties, suggesting embedded directional biases reflective of societal norms. As autonomous AI agents increasingly shape the architecture of online systems, these findings contribute to how algorithmic choices of generative AI collectives not only reshape network topology, but offer critical insights into how AI-driven systems co-evolve and self-organize.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.01471v2">Fine-tuning LLMs with variational Bayesian last layer for high-dimensional Bayesian optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      A plethora of applications entail solving black-box optimization problems with high evaluation costs, including drug discovery, material design, as well as hyperparameter tuning. Toward finding the global optimum of such black-box optimization problems with sample efficiency, Bayesian optimization (BO) is a theoretically elegant framework that relies on a probabilistic surrogate model so as to iteratively select the query point with well-balanced exploration-exploitation tradeoffs. The Gaussian process (GP), as the de-facto choice for surrogate modeling, has achieved compelling performances for vanilla BO with low-dimensional continuous variables. However, GPs fall short in coping with high-dimensional counterparts with {\it irregular} variables (e.g., categorical, ordinal, etc.). To alleviate this, neural network-based surrogates have been explored. Inspired by the powerful capabilities of LLMs, we adopt the LLM as the surrogate to model the mapping from the high-dimensional input variables to the objective function. To adapt to the current problem, we leverage the low-rank adaptation (LoRA) to fine-tune the LLM parameters together with the posterior of a linear regression head via the variational Bayesian last layer (VBLL) framework. The resulting LoRA-VBLL is not only computationally light compared to existing alternatives, but also admits recursive updates. To automate the critical selection of the LoRA rank as well as other hyperparameters, a weighted ensemble (ENS) of LoRA-VBLL surrogates has been devised, which further accommodates continual update of the per-model weight and individual LoRA-VBLL parameters via recursive Bayes. Extensive experimental results demonstrate the compelling performance of the proposed (ENS-)LoRA-VBLL approaches on various high-dimensional benchmarks and the real-world molecular optimization tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03577v1">LLM, Reporting In! Medical Information Extraction Across Prompting, Fine-tuning and Post-correction</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 in French language
    </div>
    <details class="paper-abstract">
      This work presents our participation in the EvalLLM 2025 challenge on biomedical Named Entity Recognition (NER) and health event extraction in French (few-shot setting). For NER, we propose three approaches combining large language models (LLMs), annotation guidelines, synthetic data, and post-processing: (1) in-context learning (ICL) with GPT-4.1, incorporating automatic selection of 10 examples and a summary of the annotation guidelines into the prompt, (2) the universal NER system GLiNER, fine-tuned on a synthetic corpus and then verified by an LLM in post-processing, and (3) the open LLM LLaMA-3.1-8B-Instruct, fine-tuned on the same synthetic corpus. Event extraction uses the same ICL strategy with GPT-4.1, reusing the guideline summary in the prompt. Results show GPT-4.1 leads with a macro-F1 of 61.53% for NER and 15.02% for event extraction, highlighting the importance of well-crafted prompting to maximize performance in very low-resource scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03567v1">Machine Unlearning Meets Adversarial Robustness via Constrained Interventions on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      With the increasing adoption of Large Language Models (LLMs), more customization is needed to ensure privacy-preserving and safe generation. We address this objective from two critical aspects: unlearning of sensitive information and robustness to jail-breaking attacks. We investigate various constrained optimization formulations that address both aspects in a \emph{unified manner}, by finding the smallest possible interventions on LLM weights that either make a given vocabulary set unreachable or embed the LLM with robustness to tailored attacks by shifting part of the weights to a \emph{safer} region. Beyond unifying two key properties, this approach contrasts with previous work in that it doesn't require an oracle classifier that is typically not available or represents a computational overhead. Surprisingly, we find that the simplest point-wise constraint-based intervention we propose leads to better performance than max-min interventions, while having a lower computational cost. Comparison against state-of-the-art defense methods demonstrates superior performance of the proposed approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21016v2">RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      It remains an open question whether LLMs can acquire or generalize genuinely new reasoning strategies, beyond the sharpened skills encoded in their parameters during pre-training or post-training. To attempt to answer this debate, we introduce DELTA-Code -- Distributional Evaluation of Learnability and Transferrability in Algorithmic Coding -- a controlled benchmark of synthetic coding problem families designed to probe two fundamental aspects: learnability -- can LLMs, through reinforcement learning (RL), solve problem families where pretrained models exhibit failure with large enough attempts (pass@K=0)? -- and transferrability -- if learnability happens, can such skills transfer systematically to out-of-distribution (OOD) test sets? Unlike prior public coding datasets, DELTA isolates reasoning skills through templated problem generators and introduces fully OOD problem families that demand novel strategies rather than tool invocation or memorized patterns. Our experiments reveal a striking grokking phase transition: after an extended period with near-zero reward, RL-trained models abruptly climb to near-perfect accuracy. To enable learnability on previously unsolvable problem families, we explore key training ingredients such as staged warm-up with dense rewards, experience replay, curriculum training, and verification-in-the-loop. Beyond learnability, we use DELTA to evaluate transferability or generalization along exploratory, compositional, and transformative axes, as well as cross-family transfer. Results show solid gains within families and for recomposed skills, but persistent weaknesses in transformative cases. DELTA thus offers a clean testbed for probing the limits of RL-driven reasoning and for understanding how models can move beyond existing priors to acquire new algorithmic skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03541v1">What is a protest anyway? Codebook conceptualization is still a first-order concern in LLM-era classification</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Generative large language models (LLMs) are now used extensively for text classification in computational social science (CSS). In this work, focus on the steps before and after LLM prompting -- conceptualization of concepts to be classified and using LLM predictions in downstream statistical inference -- which we argue have been overlooked in much of LLM-era CSS. We claim LLMs can tempt analysts to skip the conceptualization step, creating conceptualization errors that bias downstream estimates. Using simulations, we show that this conceptualization-induced bias cannot be corrected for solely by increasing LLM accuracy or post-hoc bias correction methods. We conclude by reminding CSS analysts that conceptualization is still a first-order concern in the LLM-era and provide concrete advice on how to pursue low-cost, unbiased, low-variance downstream estimates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03519v1">TS-Reasoner: Aligning Time Series Foundation Models with LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Time series reasoning is crucial to decision-making in diverse domains, including finance, energy usage, traffic, weather, and scientific discovery. While existing time series foundation models (TSFMs) can capture low-level dynamic patterns and provide accurate forecasting, further analysis usually requires additional background knowledge and sophisticated reasoning, which are lacking in most TSFMs but can be achieved through large language models (LLMs). On the other hand, without expensive post-training, LLMs often struggle with the numerical understanding of time series data. Although it is intuitive to integrate the two types of models, developing effective training recipes that align the two modalities for reasoning tasks is still an open challenge. To this end, we propose TS-Reasoner that aligns the latent representations of TSFMs with the textual inputs of LLMs for downstream understanding/reasoning tasks. Specifically, we propose a simple yet effective method to curate diverse, synthetic pairs of time series and textual captions for alignment training. We then develop a two-stage training recipe that applies instruction finetuning after the alignment pretraining. Unlike existing works that train an LLM to take time series as inputs, we leverage a pretrained TSFM and freeze it during training. Extensive experiments on several benchmarks demonstrate that TS-Reasoner not only outperforms a wide range of prevailing LLMs, Vision Language Models (VLMs), and Time Series LLMs, but also achieves this with remarkable data efficiency, e.g., using less than half the training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21551v3">Grokking in LLM Pretraining? Monitor Memorization-to-Generalization without Test</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 10 pages, 8 figures
    </div>
    <details class="paper-abstract">
      This paper presents the first study of grokking in practical LLM pretraining. Specifically, we investigate when an LLM memorizes the training data, when its generalization on downstream tasks starts to improve, and what happens if there is a lag between the two. Unlike existing works studying when a small model generalizes to limited and specified tasks during thousands epochs' training on algorithmic data, we focus on a practical setting for LLMs, i.e., one-epoch pretraining of next-token prediction on a cross-domain, large-scale corpus, and generalization on diverse benchmark tasks covering math/commonsense reasoning, code generation, and domain-specific retrieval. Our study, for the first time, verifies that grokking still emerges in pretraining mixture-of-experts (MoE) LLMs, though different local data groups may enter their grokking stages asynchronously due to the heterogeneity of their distributions and attributions to others. To find a mechanistic interpretation of this local grokking, we investigate the dynamics of training data's pathways (i.e., expert choices across layers in MoE). Our primary discovery is that the pathways evolve from random, non-smooth across layers, instance-specific to more structured and transferable across samples, despite the converged pretraining loss. This depicts a transition from memorization to generalization. Two novel metrics are developed to quantify these patterns: one computes the pathway similarity between samples, while the other measures the consistency of aggregated experts between subsequent layers for each sample. These training data based metrics induce zero cost but can faithfully track and monitor the generalization of LLMs on downstream tasks, which, in conventional settings, requires costly instruction tuning and benchmark evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05278v2">Micro-Act: Mitigating Knowledge Conflict in LLM-based RAG via Actionable Self-Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Accepted by ACL 2025 Main
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) systems commonly suffer from Knowledge Conflicts, where retrieved external knowledge contradicts the inherent, parametric knowledge of large language models (LLMs). It adversely affects performance on downstream tasks such as question answering (QA). Existing approaches often attempt to mitigate conflicts by directly comparing two knowledge sources in a side-by-side manner, but this can overwhelm LLMs with extraneous or lengthy contexts, ultimately hindering their ability to identify and mitigate inconsistencies. To address this issue, we propose Micro-Act a framework with a hierarchical action space that automatically perceives context complexity and adaptively decomposes each knowledge source into a sequence of fine-grained comparisons. These comparisons are represented as actionable steps, enabling reasoning beyond the superficial context. Through extensive experiments on five benchmark datasets, Micro-Act consistently achieves significant increase in QA accuracy over state-of-the-art baselines across all 5 datasets and 3 conflict types, especially in temporal and semantic types where all baselines fail significantly. More importantly, Micro-Act exhibits robust performance on non-conflict questions simultaneously, highlighting its practical value in real-world RAG applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03502v1">ALHD: A Large-Scale and Multigenre Benchmark Dataset for Arabic LLM-Generated Text Detection</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 47 pages, 15 figures. Dataset available at Zenodo: https://doi.org/10.5281/zenodo.17249602 Codebase available at GitHub: https://github.com/alikhairallah/ALHD-Benchmarking
    </div>
    <details class="paper-abstract">
      We introduce ALHD, the first large-scale comprehensive Arabic dataset explicitly designed to distinguish between human- and LLM-generated texts. ALHD spans three genres (news, social media, reviews), covering both MSA and dialectal Arabic, and contains over 400K balanced samples generated by three leading LLMs and originated from multiple human sources, which enables studying generalizability in Arabic LLM-genearted text detection. We provide rigorous preprocessing, rich annotations, and standardized balanced splits to support reproducibility. In addition, we present, analyze and discuss benchmark experiments using our new dataset, in turn identifying gaps and proposing future research directions. Benchmarking across traditional classifiers, BERT-based models, and LLMs (zero-shot and few-shot) demonstrates that fine-tuned BERT models achieve competitive performance, outperforming LLM-based models. Results are however not always consistent, as we observe challenges when generalizing across genres; indeed, models struggle to generalize when they need to deal with unseen patterns in cross-genre settings, and these challenges are particularly prominent when dealing with news articles, where LLM-generated texts resemble human texts in style, which opens up avenues for future research. ALHD establishes a foundation for research related to Arabic LLM-detection and mitigating risks of misinformation, academic dishonesty, and cyber threats.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03480v1">LLM Agents for Automated Dependency Upgrades</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      As a codebase expands over time, its library dependencies can become outdated and require updates to maintain innovation and security. However, updating a library can introduce breaking changes in the code, necessitating significant developer time for maintenance. To address this, we introduce a framework of LLM agents to be used in combination with migration documentation to automatically recommend and apply code updates and ensure compatibility with new versions. Our solution can automatically localize updated library usages in live Java codebases and implement recommended fixes in a user-friendly manner. The system architecture consists of multiple key components: a Summary Agent, Control Agent, and Code Agent. To validate our approach, we apply the framework on an industrial use case by which we create three synthetic code repositories with major Upgrade changes and benchmark our approach against state-of-the-art methods. Results show that our approach not only performs upgrades using fewer tokens across all cases but also achieves a precision of 71.4%, highlighting its efficiency and effectiveness compared to state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04557v2">SSA-COMET: Do LLMs Outperform Learned Metrics in Evaluating MT for Under-Resourced African Languages?</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Evaluating machine translation (MT) quality for under-resourced African languages remains a significant challenge, as existing metrics often suffer from limited language coverage and poor performance in low-resource settings. While recent efforts, such as AfriCOMET, have addressed some of the issues, they are still constrained by small evaluation sets, a lack of publicly available training data tailored to African languages, and inconsistent performance in extremely low-resource scenarios. In this work, we introduce SSA-MTE, a large-scale human-annotated MT evaluation (MTE) dataset covering 14 African language pairs from the News domain, with over 73,000 sentence-level annotations from a diverse set of MT systems. Based on this data, we develop SSA-COMET and SSA-COMET-QE, improved reference-based and reference-free evaluation metrics. We also benchmark prompting-based approaches using state-of-the-art LLMs like GPT-4o, Claude-3.7 and Gemini 2.5 Pro. Our experimental results show that SSA-COMET models significantly outperform AfriCOMET and are competitive with the strongest LLM Gemini 2.5 Pro evaluated in our study, particularly on low-resource languages such as Twi, Luo, and Yoruba. All resources are released under open licenses to support future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03469v1">Bridging LLM Planning Agents and Formal Methods: A Case Study in Plan Verification</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for evaluating the alignment between natural language plans and their expected behavior by converting them into Kripke structures and Linear Temporal Logic (LTL) using Large Language Models (LLMs) and performing model checking. We systematically evaluate this framework on a simplified version of the PlanBench plan verification dataset and report on metrics like Accuracy, Precision, Recall and F1 scores. Our experiments demonstrate that GPT-5 achieves excellent classification performance (F1 score of 96.3%) while almost always producing syntactically perfect formal representations that can act as guarantees. However, the synthesis of semantically perfect formal models remains an area for future exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03463v1">ALMAS: an Autonomous LLM-based Multi-Agent Software Engineering Framework</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Multi-agent Large Language Model (LLM) systems have been leading the way in applied LLM research across a number of fields. One notable area is software development, where researchers have advanced the automation of code implementation, code testing, code maintenance, inter alia, using LLM agents. However, software development is a multifaceted environment that extends beyond just code. As such, a successful LLM system must factor in multiple stages of the software development life-cycle (SDLC). In this paper, we propose a vision for ALMAS, an Autonomous LLM-based Multi-Agent Software Engineering framework, which follows the above SDLC philosophy such that it may work within an agile software development team to perform several tasks end-to-end. ALMAS aligns its agents with agile roles, and can be used in a modular fashion to seamlessly integrate with human developers and their development environment. We showcase the progress towards ALMAS through our published works and a use case demonstrating the framework, where ALMAS is able to seamlessly generate an application and add a new feature.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03425v1">Memory-Efficient Backpropagation for Fine-Tuning LLMs on Resource-Constrained Mobile Devices</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) with backpropagation\textemdash even for a subset of parameters such as LoRA\textemdash can be much more memory-consuming than inference and is often deemed impractical for resource-constrained mobile devices. Alternative methods, such as zeroth-order optimization (ZO), can greatly reduce the memory footprint but come at the cost of significantly slower model convergence (10$\times$ to 100$\times$ more steps than backpropagation). We propose a memory-efficient implementation of backpropagation (MeBP) on mobile devices that provides better trade-off between memory usage and compute time, while converging faster and achieving better performance than the ZO baseline. We verify the effectiveness of MeBP on an iPhone 15 Pro Max and show that various LLMs, ranging from 0.5B to 4B parameters, can be fine-tuned using less than 1GB of memory. We release an example of the MeBP implementation at https://github.com/apple/ml-mebp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03417v1">NEXUS: Network Exploration for eXploiting Unsafe Sequences in Multi-Turn LLM Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Javad Rafiei Asl and Sidhant Narula are co-first authors
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have revolutionized natural language processing but remain vulnerable to jailbreak attacks, especially multi-turn jailbreaks that distribute malicious intent across benign exchanges and bypass alignment mechanisms. Existing approaches often explore the adversarial space poorly, rely on hand-crafted heuristics, or lack systematic query refinement. We present NEXUS (Network Exploration for eXploiting Unsafe Sequences), a modular framework for constructing, refining, and executing optimized multi-turn attacks. NEXUS comprises: (1) ThoughtNet, which hierarchically expands a harmful intent into a structured semantic network of topics, entities, and query chains; (2) a feedback-driven Simulator that iteratively refines and prunes these chains through attacker-victim-judge LLM collaboration using harmfulness and semantic-similarity benchmarks; and (3) a Network Traverser that adaptively navigates the refined query space for real-time attacks. This pipeline uncovers stealthy, high-success adversarial paths across LLMs. On several closed-source and open-source LLMs, NEXUS increases attack success rate by 2.1% to 19.4% over prior methods. Code: https://github.com/inspire-lab/NEXUS
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.09071v2">Elastic On-Device LLM Service</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 MobiCom'25
    </div>
    <details class="paper-abstract">
      On-device Large Language Models (LLMs) are transforming mobile AI, catalyzing applications like UI automation without privacy concerns. Nowadays the common practice is to deploy a single yet powerful LLM as a general task solver for multiple requests. We identify a key system challenge in this paradigm: current LLMs lack the elasticity to serve requests that have diversified Service-Level Objectives (SLOs) on inference latency. To tackle this, we present \sys, an on-device LLM service that elasticizes both the model and the prompt dimension of a full LLM. It incorporates (1) a one-shot neuron-reordering method, which leverages the intrinsic permutation consistency in transformer models to generate high-quality elasticized sub-models with minimal runtime switching overhead; (2) a dual-head tiny language model, which efficiently and effectively refines the prompt and orchestrates the elastification between model and prompt. We implement such an elastic on-device LLM service on multiple COTS smartphones, and evaluate \sys on both standalone NLP/mobile-agent datasets and end-to-end synthesized traces. On diverse SLOs, \sys outperforms 7 strong baselines in (absolute) accuracy by up to 14.83\% and 10.45\% on average, with <1\% TTFT switching overhead, on-par memory consumption and <100 offline GPU hours.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03384v1">Implicit Values Embedded in How Humans and LLMs Complete Subjective Everyday Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can underpin AI assistants that help users with everyday tasks, such as by making recommendations or performing basic computation. Despite AI assistants' promise, little is known about the implicit values these assistants display while completing subjective everyday tasks. Humans may consider values like environmentalism, charity, and diversity. To what extent do LLMs exhibit these values in completing everyday tasks? How do they compare with humans? We answer these questions by auditing how six popular LLMs complete 30 everyday tasks, comparing LLMs to each other and to 100 human crowdworkers from the US. We find LLMs often do not align with humans, nor with other LLMs, in the implicit values exhibited.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03217v1">Abstain and Validate: A Dual-LLM Policy for Reducing Noise in Agentic Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Agentic Automated Program Repair (APR) is increasingly tackling complex, repository-level bugs in industry, but ultimately agent-generated patches still need to be reviewed by a human before committing them to ensure they address the bug. Showing unlikely patches to developers can lead to substantial noise, wasting valuable developer time and eroding trust in automated code changes. We introduce two complementary LLM-based policies to reduce such noise: bug abstention and patch validation policies. Bug abstention excludes bugs that the agentic APR system is unlikely to fix. Patch validation rejects patches that are unlikely to be a good fix for the given bug. We evaluate both policies on three sets of bugs from Google's codebase, and their candidate patches generated by an internal agentic APR system. On a set of 174 human-reported bugs, removing bugs and patch trajectories rejected by our policies can raise success rates by up to 13 percentage points and 15 percentage points, respectively, and by up to 39 percentage points in combination. On null pointer exceptions and sanitizer-reported bugs with machine-generated bug reports, patch validation also improves average single-sample success rates. This two-policy approach provides a practical path to the reliable, industrial-scale deployment of agentic APR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03195v1">Can LLMs Hit Moving Targets? Tracking Evolving Signals in Corporate Disclosures</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 8 pages, 5 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Moving targets -- managers' strategic shifting of key performance metrics when the original targets become difficult to achieve -- have been shown to predict subsequent stock underperformance. However, our work reveals that the method employed in that study exhibits two key limitations that hinder the accuracy -- noise in the extracted targets and loss of contextual information -- both of which stem primarily from the use of a named entity recognition (NER). To address these two limitations, we propose an LLM-based target extraction} method with a newly defined metric that better captures semantic context. This approach preserves semantic context beyond simple entity recognition and yields consistently higher predictive power than the original approach. Overall, our approach enhances the granularity and accuracy of financial text-based performance prediction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.24015v3">Hierarchical Knowledge Injection for Improving LLM-based Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Accepted at IEEE/ACM Automated Software Engineering (ASE) 2025 Conference
    </div>
    <details class="paper-abstract">
      Prompting LLMs with bug-related context (e.g., error messages, stack traces) improves automated program repair, but many bugs still remain unresolved. In real-world projects, developers often rely on broader repository and project-level context beyond the local code to resolve such bugs. In this paper, we investigate how automatically extracting and providing such knowledge can improve LLM-based program repair. We propose a layered knowledge injection framework that incrementally augments LLMs with structured context. It starts with the Bug Knowledge Layer, which includes information such as the buggy function and failing tests; expands to the Repository Knowledge Layer, which adds structural dependencies, related files, and commit history; and finally injects the Project Knowledge Layer, which incorporates relevant details from documentation and previously fixed bugs. We evaluate this framework on a dataset of 314 bugs from BugsInPy using two LLMs (Llama 3.3 and GPT-4o-mini), and analyze fix rates across six bug types. By progressively injecting knowledge across layers, our approach achieves a fix rate of 79% (250/314) using Llama 3.3, a significant improvement of 23% over previous work. All bug types show improvement with the addition of repository-level context, while only a subset benefit further from project-level knowledge, highlighting that different bug types require different levels of contextual information for effective repair. We also analyze the remaining unresolved bugs and find that more complex and structurally isolated bugs, such as Program Anomaly and GUI bugs, remain difficult even after injecting all available information. Our results show that layered context injection improves program repair and suggest the need for interactive and adaptive APR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01443v2">Tuning LLM-based Code Optimization via Meta-Prompting: An Industrial Perspective</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Accepted by ASE'25 Industry Showcase
    </div>
    <details class="paper-abstract">
      There is a growing interest in leveraging multiple large language models (LLMs) for automated code optimization. However, industrial platforms deploying multiple LLMs face a critical challenge: prompts optimized for one LLM often fail with others, requiring expensive model-specific prompt engineering. This cross-model prompt engineering bottleneck severely limits the practical deployment of multi-LLM systems in production environments. We introduce Meta-Prompted Code Optimization (MPCO), a framework that automatically generates high-quality, task-specific prompts across diverse LLMs while maintaining industrial efficiency requirements. MPCO leverages metaprompting to dynamically synthesize context-aware optimization prompts by integrating project metadata, task requirements, and LLM-specific contexts. It is an essential part of the ARTEMIS code optimization platform for automated validation and scaling. Our comprehensive evaluation on five real-world codebases with 366 hours of runtime benchmarking demonstrates MPCO's effectiveness: it achieves overall performance improvements up to 19.06% with the best statistical rank across all systems compared to baseline methods. Analysis shows that 96% of the top-performing optimizations stem from meaningful edits. Through systematic ablation studies and meta-prompter sensitivity analysis, we identify that comprehensive context integration is essential for effective meta-prompting and that major LLMs can serve effectively as meta-prompters, providing actionable insights for industrial practitioners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03178v1">When Names Disappear: Revealing What LLMs Actually Understand About Code</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve strong results on code tasks, but how they derive program meaning remains unclear. We argue that code communicates through two channels: structural semantics, which define formal behavior, and human-interpretable naming, which conveys intent. Removing the naming channel severely degrades intent-level tasks such as summarization, where models regress to line-by-line descriptions. Surprisingly, we also observe consistent reductions on execution tasks that should depend only on structure, revealing that current benchmarks reward memorization of naming patterns rather than genuine semantic reasoning. To disentangle these effects, we introduce a suite of semantics-preserving obfuscations and show that they expose identifier leakage across both summarization and execution. Building on these insights, we release ClassEval-Obf, an obfuscation-enhanced benchmark that systematically suppresses naming cues while preserving behavior. Our results demonstrate that ClassEval-Obf reduces inflated performance gaps, weakens memorization shortcuts, and provides a more reliable basis for assessing LLMs' code understanding and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03174v1">Topic Modeling as Long-Form Generation: Can Long-Context LLMs revolutionize NTM via Zero-Shot Prompting?</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Traditional topic models such as neural topic models rely on inference and generation networks to learn latent topic distributions. This paper explores a new paradigm for topic modeling in the era of large language models, framing TM as a long-form generation task whose definition is updated in this paradigm. We propose a simple but practical approach to implement LLM-based topic model tasks out of the box (sample a data subset, generate topics and representative text with our prompt, text assignment with keyword match). We then investigate whether the long-form generation paradigm can beat NTMs via zero-shot prompting. We conduct a systematic comparison between NTMs and LLMs in terms of topic quality and empirically examine the claim that "a majority of NTMs are outdated."
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.24827v2">Putnam-like dataset summary: LLMs as mathematical competition contestants</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 11 pages, 11 figures
    </div>
    <details class="paper-abstract">
      In this paper we summarize the results of the Putnam-like benchmark published by Google DeepMind. This dataset consists of 96 original problems in the spirit of the Putnam Competition and 576 solutions of LLMs. We analyse the performance of models on this set of problems to verify their ability to solve problems from mathematical contests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09901v2">Comparing Exploration-Exploitation Strategies of LLMs and Humans: Insights from Standard Multi-armed Bandit Experiments</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to simulate or automate human behavior in complex sequential decision-making settings. A natural question is then whether LLMs exhibit similar decision-making behavior to humans, and can achieve comparable (or superior) performance. In this work, we focus on the exploration-exploitation (E&E) tradeoff, a fundamental aspect of dynamic decision-making under uncertainty. We employ canonical multi-armed bandit (MAB) experiments introduced in the cognitive science and psychiatry literature to conduct a comparative study of the E&E strategies of LLMs, humans, and MAB algorithms. We use interpretable choice models to capture the E&E strategies of the agents and investigate how enabling thinking traces, through both prompting strategies and thinking models, shapes LLM decision-making. We find that enabling thinking in LLMs shifts their behavior toward more human-like behavior, characterized by a mix of random and directed exploration. In a simple stationary setting, thinking-enabled LLMs exhibit similar levels of random and directed exploration compared to humans. However, in more complex, non-stationary environments, LLMs struggle to match human adaptability, particularly in effective directed exploration, despite achieving similar regret in certain scenarios. Our findings highlight both the promise and limits of LLMs as simulators of human behavior and tools for automated decision-making and point to potential areas for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22811v2">Highly Efficient and Effective LLMs with Multi-Boolean Architectures</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 Preprint. Under Review
    </div>
    <details class="paper-abstract">
      Weight binarization has emerged as a promising strategy to reduce the complexity of large language models (LLMs). Existing approaches fall into post-training binarization, which is simple but causes severe performance loss, and training-aware methods, which depend on full-precision latent weights, adding complexity and limiting efficiency. We propose a novel framework that represents LLMs with multi-kernel Boolean parameters and, for the first time, enables direct finetuning LMMs in the Boolean domain, eliminating the need for latent weights. This enhances representational capacity and dramatically reduces complexity during both finetuning and inference. Extensive experiments across diverse LLMs show our method outperforms recent ultra low-bit quantization and binarization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14837v2">Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 16 pages, 8 figures; Accepted to ACL 2025
    </div>
    <details class="paper-abstract">
      Multi-head Latent Attention (MLA) is an innovative architecture proposed by DeepSeek, designed to ensure efficient and economical inference by significantly compressing the Key-Value (KV) cache into a latent vector. Compared to MLA, standard LLMs employing Multi-Head Attention (MHA) and its variants such as Grouped-Query Attention (GQA) exhibit significant cost disadvantages. Enabling well-trained LLMs (e.g., Llama) to rapidly adapt to MLA without pre-training from scratch is both meaningful and challenging. This paper proposes the first data-efficient fine-tuning method for transitioning from MHA to MLA (MHA2MLA), which includes two key components: for partial-RoPE, we remove RoPE from dimensions of queries and keys that contribute less to the attention scores, for low-rank approximation, we introduce joint SVD approximations based on the pre-trained parameters of keys and values. These carefully designed strategies enable MHA2MLA to recover performance using only a small fraction (0.3% to 0.6%) of the data, significantly reducing inference costs while seamlessly integrating with compression techniques such as KV cache quantization. For example, the KV cache size of Llama2-7B is reduced by 92.19%, with only a 0.5% drop in LongBench performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03102v1">Semantic Similarity in Radiology Reports via LLMs and NER</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Radiology report evaluation is a crucial part of radiologists' training and plays a key role in ensuring diagnostic accuracy. As part of the standard reporting workflow, a junior radiologist typically prepares a preliminary report, which is then reviewed and edited by a senior radiologist to produce the final report. Identifying semantic differences between preliminary and final reports is essential for junior doctors, both as a training tool and to help uncover gaps in clinical knowledge. While AI in radiology is a rapidly growing field, the application of large language models (LLMs) remains challenging due to the need for specialised domain knowledge. In this paper, we explore the ability of LLMs to provide explainable and accurate comparisons of reports in the radiology domain. We begin by comparing the performance of several LLMs in comparing radiology reports. We then assess a more traditional approach based on Named-Entity-Recognition (NER). However, both approaches exhibit limitations in delivering accurate feedback on semantic similarity. To address this, we propose Llama-EntScore, a semantic similarity scoring method using a combination of Llama 3.1 and NER with tunable weights to emphasise or de-emphasise specific types of differences. Our approach generates a quantitative similarity score for tracking progress and also gives an interpretation of the score that aims to offer valuable guidance in reviewing and refining their reporting. We find our method achieves 67% exact-match accuracy and 93% accuracy within +/- 1 when compared to radiologist-provided ground truth scores - outperforming both LLMs and NER used independently. Code is available at: \href{https://github.com/otmive/llama_reports}{github.com/otmive/llama\_reports}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03093v1">Revisiting Direct Speech-to-Text Translation with Speech LLMs: Better Scaling than CoT Prompting?</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Recent work on Speech-to-Text Translation (S2TT) has focused on LLM-based models, introducing the increasingly adopted Chain-of-Thought (CoT) prompting, where the model is guided to first transcribe the speech and then translate it. CoT typically outperforms direct prompting primarily because it can exploit abundant Automatic Speech Recognition (ASR) and Text-to-Text Translation (T2TT) datasets to explicitly model its steps. In this paper, we systematically compare CoT and Direct prompting under increasing amounts of S2TT data. To this end, we pseudo-label an ASR corpus by translating its transcriptions into six European languages, and train LLM-based S2TT systems with both prompting strategies at different data scales. Our results show that Direct improves more consistently as the amount of data increases, suggesting that it may become a more effective approach as larger S2TT resources are created.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22860v2">Permissioned LLMs: Enforcing Access Control in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      In enterprise settings, organizational data is segregated, siloed and carefully protected by elaborate access control frameworks. These access control structures can completely break down if an LLM fine-tuned on the siloed data serves requests, for downstream tasks, from individuals with disparate access privileges. We propose Permissioned LLMs (PermLLM), a new class of LLMs that superimpose the organizational data access control structures on query responses they generate. We formalize abstractions underpinning the means to determine whether access control enforcement happens correctly over LLM query responses. Our formalism introduces the notion of a relevant response that can be used to prove whether a PermLLM mechanism has been implemented correctly. We also introduce a novel metric, called access advantage, to empirically evaluate the efficacy of a PermLLM mechanism. We introduce three novel PermLLM mechanisms that build on Parameter Efficient Fine-Tuning to achieve the desired access control. We furthermore present two instantiations of access advantage--(i) Domain Distinguishability Index (DDI) based on Membership Inference Attacks, and (ii) Utility Gap Index (UGI) based on LLM utility evaluation. We demonstrate the efficacy of our PermLLM mechanisms through extensive experiments on five public datasets (GPQA, RCV1, SimpleQA, WMDP, and PubMedQA), in addition to evaluating the validity of DDI and UGI metrics themselves for quantifying access control in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.03029v1">Investigating The Smells of LLM Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Context: Large Language Models (LLMs) are increasingly being used to generate program code. Much research has been reported on the functional correctness of generated code, but there is far less on code quality. Objectives: In this study, we propose a scenario-based method of evaluating the quality of LLM-generated code to identify the weakest scenarios in which the quality of LLM generated code should be improved. Methods: The method measures code smells, an important indicator of code quality, and compares them with a baseline formed from reference solutions of professionally written code. The test dataset is divided into various subsets according to the topics of the code and complexity of the coding tasks to represent different scenarios of using LLMs for code generation. We will also present an automated test system for this purpose and report experiments with the Java programs generated in response to prompts given to four state-of-the-art LLMs: Gemini Pro, ChatGPT, Codex, and Falcon. Results: We find that LLM-generated code has a higher incidence of code smells compared to reference solutions. Falcon performed the least badly, with a smell increase of 42.28%, followed by Gemini Pro (62.07%), ChatGPT (65.05%) and finally Codex (84.97%). The average smell increase across all LLMs was 63.34%, comprising 73.35% for implementation smells and 21.42% for design smells. We also found that the increase in code smells is greater for more complex coding tasks and for more advanced topics, such as those involving object-orientated concepts. Conclusion: In terms of code smells, LLM's performances on various coding task complexities and topics are highly correlated to the quality of human written code in the corresponding scenarios. However, the quality of LLM generated code is noticeably poorer than human written code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07044v2">DatawiseAgent: A Notebook-Centric LLM Agent Framework for Adaptive and Robust Data Science Automation</a></div>
    <div class="paper-meta">
      📅 2025-10-03
      | 💬 The camera-ready version for EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Existing large language model (LLM) agents for automating data science show promise, but they remain constrained by narrow task scopes, limited generalization across tasks and models, and over-reliance on state-of-the-art (SOTA) LLMs. We introduce DatawiseAgent, a notebook-centric LLM agent framework for adaptive and robust data science automation. Inspired by how human data scientists work in computational notebooks, DatawiseAgent introduces a unified interaction representation and a multi-stage architecture based on finite-state transducers (FSTs). This design enables flexible long-horizon planning, progressive solution development, and robust recovery from execution failures. Extensive experiments across diverse data science scenarios and models show that DatawiseAgent consistently achieves SOTA performance by surpassing strong baselines such as AutoGen and TaskWeaver, demonstrating superior effectiveness and adaptability. Further evaluations reveal graceful performance degradation under weaker or smaller models, underscoring the robustness and scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02934v1">Model-Agnostic Correctness Assessment for LLM-Generated Code via Dynamic Internal Representation Selection</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities in code generation and are increasingly integrated into the software development process. However, ensuring the correctness of LLM-generated code remains a critical concern. Prior work has shown that the internal representations of LLMs encode meaningful signals for assessing code correctness. Nevertheless, the existing methods rely on representations from pre-selected/fixed layers and token positions, which could limit its generalizability across diverse model architectures and tasks. In this work, we introduce AUTOPROBE, a novel model-agnostic approach that dynamically selects the most informative internal representations for code correctness assessment. AUTOPROBE employs an attention-based mechanism to learn importance scores for hidden states, enabling it to focus on the most relevant features. These weighted representations are then aggregated and passed to a probing classifier to predict code correctness across multiple dimensions, including compilability, functionality, and security. To evaluate the performance of AUTOPROBE, we conduct extensive experiments across multiple benchmarks and code LLMs. Our experimental results show that AUTOPROBE consistently outperforms the baselines. For security assessment, AUTOPROBE surpasses the state-of-the-art white-box approach by 18%. For compilability and functionality assessment, AUTOPROBE demonstrates its highest robustness to code complexity, with the performance higher than the other approaches by up to 19% and 111%, respectively. These findings highlight that dynamically selecting important internal signals enables AUTOPROBE to serve as a robust and generalizable solution for assessing the correctness of code generated by various LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.03029v1">Investigating The Smells of LLM Generated Code</a></div>
    <div class="paper-meta">
      📅 2025-10-03
    </div>
    <details class="paper-abstract">
      Context: Large Language Models (LLMs) are increasingly being used to generate program code. Much research has been reported on the functional correctness of generated code, but there is far less on code quality. Objectives: In this study, we propose a scenario-based method of evaluating the quality of LLM-generated code to identify the weakest scenarios in which the quality of LLM generated code should be improved. Methods: The method measures code smells, an important indicator of code quality, and compares them with a baseline formed from reference solutions of professionally written code. The test dataset is divided into various subsets according to the topics of the code and complexity of the coding tasks to represent different scenarios of using LLMs for code generation. We will also present an automated test system for this purpose and report experiments with the Java programs generated in response to prompts given to four state-of-the-art LLMs: Gemini Pro, ChatGPT, Codex, and Falcon. Results: We find that LLM-generated code has a higher incidence of code smells compared to reference solutions. Falcon performed the least badly, with a smell increase of 42.28%, followed by Gemini Pro (62.07%), ChatGPT (65.05%) and finally Codex (84.97%). The average smell increase across all LLMs was 63.34%, comprising 73.35% for implementation smells and 21.42% for design smells. We also found that the increase in code smells is greater for more complex coding tasks and for more advanced topics, such as those involving object-orientated concepts. Conclusion: In terms of code smells, LLM's performances on various coding task complexities and topics are highly correlated to the quality of human written code in the corresponding scenarios. However, the quality of LLM generated code is noticeably poorer than human written code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19366v2">Grounding the Ungrounded: A Spectral-Graph Framework for Quantifying Hallucinations in multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 29 pages, 3 figures, 1 table
    </div>
    <details class="paper-abstract">
      Hallucinations in large language models (LLMs) remain a fundamental obstacle to trustworthy AI, particularly in high-stakes multimodal domains such as medicine, law, and finance. Existing evaluation techniques are largely heuristic -- anchored in qualitative benchmarking or ad-hoc empirical mitigation -- providing neither principled quantification nor actionable theoretical guarantees. This gap leaves a critical blind spot in understanding how hallucinations arise, propagate, and interact across modalities. We introduce the first (to our knowledge) rigorous information geometric framework in diffusion dynamics for quantifying hallucinations in multimodal LLMs (MLLMs), advancing the field from qualitative detection to mathematically grounded measurement. Our approach represents MLLM outputs as the spectral embeddings over multimodal graph Laplacians and characterizes the manifold gaps of truth vs inconsistencies as the semantic distortion, enabling the tight Rayleigh--Ritz bounds on the multimodal hallucination energy as a functional of time-dependent temperature profiles. By leveraging eigenmode decompositions in Reproducing Kernel Hilbert Space (RKHS) embeddings, our framework delivers modality-aware, theoretically interpretable metrics that capture the evolution of hallucinations across time and input prompts through temperature annealing. This work establishes a principled foundation for quantifying and bounding hallucinations, transforming them from a qualitative risk to a tractable, analyzable phenomenon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02372v2">Scam2Prompt: A Scalable Framework for Auditing Malicious Scam Endpoints in Production LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become critical to modern software development, but their reliance on uncurated web-scale datasets for training introduces a significant security risk: the absorption and reproduction of malicious content. To systematically evaluate this risk, we introduce Scam2Prompt, a scalable automated auditing framework that identifies the underlying intent of a scam site and then synthesizes innocuous, developer-style prompts that mirror this intent, allowing us to test whether an LLM will generate malicious code in response to these innocuous prompts. In a large-scale study of four production LLMs (GPT-4o, GPT-4o-mini, Llama-4-Scout, and DeepSeek-V3), we found that Scam2Prompt's innocuous prompts triggered malicious URL generation in 4.24% of cases. To test the persistence of this security risk, we constructed Innoc2Scam-bench, a benchmark of 1,559 innocuous prompts that consistently elicited malicious code from all four initial LLMs. When applied to seven additional production LLMs released in 2025, we found the vulnerability is not only present but severe, with malicious code generation rates ranging from 12.7% to 43.8%. Furthermore, existing safety measures like state-of-the-art guardrails proved insufficient to prevent this behavior, with an overall detection rate of less than 0.3%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04076v2">Unmasking the Genuine Type Inference Capabilities of LLMs for Java Code Snippets</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Type inference is crucial for reusing online code snippets. Although snippets are prevalently shared on platforms like StackOverflow, they often lack essential type information, such as fully qualified names (FQNs). Recent studies have leveraged Large Language Models (LLMs) to perform type inference for such code snippets, showing promising results. However, these results may suffer from data leakage, as the benchmark, StatType-SO, used for evaluation has been publicly available on GitHub since 2017. Consequently, it remains uncertain whether the strong performance of LLMs reflects genuine semantic understanding of code or is due to the ground truth being included in the training set. This paper strives to comprehensively evaluate the genuine type inference capabilities of LLMs on Java code snippets and identify potential limitations of LLMs. First, we created ThaliaType, a new, previously unreleased benchmark suite designed for type inference evaluation. Second, using the StarCoder2 LLM as baseline, we uncovered data leakage from StatType-SO in StarCoder2's open-source training set and observed that other state-of-the-art LLMs exhibit similar performance drops when evaluated on ThaliaType, with precision decreasing by up to 59% and recall by up to 72%. Finally, we designed semantic-preserving code transformations to test the capabilities of LLMs in understanding the execution semantics of snippets. Results showed that LLMs' performance on StatType-SO is far less robust to these transformations than on ThaliaType, suggesting that the performance on StatType-SO may be biased by data leakage and have limited generalizability. These findings highlight the importance of carefully designed, leakage-free benchmarks for evaluating LLMs on type inference tasks. We recommend future studies adopt ThaliaType for rigorous and reliable assessments of LLMs' genuine type inference capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02554v1">ToolTweak: An Attack on Tool Selection in LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      As LLMs increasingly power agents that interact with external tools, tool use has become an essential mechanism for extending their capabilities. These agents typically select tools from growing databases or marketplaces to solve user tasks, creating implicit competition among tool providers and developers for visibility and usage. In this paper, we show that this selection process harbors a critical vulnerability: by iteratively manipulating tool names and descriptions, adversaries can systematically bias agents toward selecting specific tools, gaining unfair advantage over equally capable alternatives. We present ToolTweak, a lightweight automatic attack that increases selection rates from a baseline of around 20% to as high as 81%, with strong transferability between open-source and closed-source models. Beyond individual tools, we show that such attacks cause distributional shifts in tool usage, revealing risks to fairness, competition, and security in emerging tool ecosystems. To mitigate these risks, we evaluate two defenses: paraphrasing and perplexity filtering, which reduce bias and lead agents to select functionally similar tools more equally. All code will be open-sourced upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.18314v2">Exploiting Tree Structure for Credit Assignment in RL Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Reinforcement learning improves LLM reasoning, yet sparse delayed reward over long sequences makes token-level credit assignment the key bottleneck. We study the verifiable-reward setting, where the final answer is checkable and multiple responses can be drawn per prompt. Reasoning tasks in math and medical QA align with this setup, where only a few decision tokens significantly impact the outcome. PPO offers token-level advantages with a learned value model, but it is complex to train both the actor and critic models simultaneously, and it is not easily generalizable, as the token-level values from the critic model can make training prone to overfitting. GRPO is critic-free and supports verifiable rewards, but spreads a single sequence-level return across tokens and ignores branching. We introduce \textbf{Prefix-to-Tree (P2T)}, a simple procedure that converts a group of responses into a prefix tree and computes \emph{nonparametric} prefix values \(V(s)\) by aggregating descendant outcomes. Built on P2T, we propose \textbf{TEMPO} (\emph{\textbf{T}ree-\textbf{E}stimated \textbf{M}ean Prefix Value for \textbf{P}olicy \textbf{O}ptimization}), a critic-free algorithm that augments the group-relative outcome signal of GRPO with \emph{branch-gated} temporal-difference corrections derived from the tree. At non-branch tokens, the temporal-difference (TD) term is zero, so TEMPO reduces to GRPO; at branching tokens, it supplies precise token-level credit without a learned value network or extra judges/teachers. On Qwen3-1.7B/4B, TEMPO outperforms PPO and GRPO on in-distribution (MATH, MedQA) and out-of-distribution (GSM-HARD, AMC23, MedMCQA, MMLU-Medical) benchmarks, and reaches higher validation accuracy with roughly the same wall-clock time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02534v1">ZeroFalse: Improving Precision in Static Analysis with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Static Application Security Testing (SAST) tools are integral to modern software development, yet their adoption is undermined by excessive false positives that weaken developer trust and demand costly manual triage. We present ZeroFalse, a framework that integrates static analysis with large language models (LLMs) to reduce false positives while preserving coverage. ZeroFalse treats static analyzer outputs as structured contracts, enriching them with flow-sensitive traces, contextual evidence, and CWE-specific knowledge before adjudication by an LLM. This design preserves the systematic reach of static analysis while leveraging the reasoning capabilities of LLMs. We evaluate ZeroFalse across both benchmarks and real-world projects using ten state-of-the-art LLMs. Our best-performing models achieve F1-scores of 0.912 on the OWASP Java Benchmark and 0.955 on the OpenVuln dataset, maintaining recall and precision above 90%. Results further show that CWE-specialized prompting consistently outperforms generic prompts, and reasoning-oriented LLMs provide the most reliable precision-recall balance. These findings position ZeroFalse as a practical and scalable approach for enhancing the reliability of SAST and supporting its integration into real-world CI/CD pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.00038v2">DM-Bench: Benchmarking LLMs for Personalized Decision Making in Diabetes Management</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      We present DM-Bench, the first benchmark designed to evaluate large language model (LLM) performance across real-world decision-making tasks faced by individuals managing diabetes in their daily lives. Unlike prior health benchmarks that are either generic, clinician-facing or focused on clinical tasks (e.g., diagnosis, triage), DM-Bench introduces a comprehensive evaluation framework tailored to the unique challenges of prototyping patient-facing AI solutions in diabetes, glucose management, metabolic health and related domains. Our benchmark encompasses 7 distinct task categories, reflecting the breadth of real-world questions individuals with diabetes ask, including basic glucose interpretation, educational queries, behavioral associations, advanced decision making and long term planning. Towards this end, we compile a rich dataset comprising one month of time-series data encompassing glucose traces and metrics from continuous glucose monitors (CGMs) and behavioral logs (e.g., eating and activity patterns) from 15,000 individuals across three different diabetes populations (type 1, type 2, pre-diabetes/general health and wellness). Using this data, we generate a total of 360,600 personalized, contextual questions across the 7 tasks. We evaluate model performance on these tasks across 5 metrics: accuracy, groundedness, safety, clarity and actionability. Our analysis of 8 recent LLMs reveals substantial variability across tasks and metrics; no single model consistently outperforms others across all dimensions. By establishing this benchmark, we aim to advance the reliability, safety, effectiveness and practical utility of AI solutions in diabetes care.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08970v2">Gala: Global LLM Agents for Text-to-Model Translation</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Natural language descriptions of optimization or satisfaction problems are challenging to translate into correct MiniZinc models, as this process demands both logical reasoning and constraint programming expertise. We introduce Gala, a framework that addresses this challenge with a global agentic approach: multiple specialized large language model (LLM) agents decompose the modeling task by global constraint type. Each agent is dedicated to detecting and generating code for a specific class of global constraint, while a final assembler agent integrates these constraint snippets into a complete MiniZinc model. By dividing the problem into smaller, well-defined sub-tasks, each LLM handles a simpler reasoning challenge, potentially reducing overall complexity. We conduct initial experiments with several LLMs and show better performance against baselines such as one-shot prompting and chain-of-thought prompting. Finally, we outline a comprehensive roadmap for future work, highlighting potential enhancements and directions for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13445v2">Verbosity Tradeoffs and the Impact of Scale on the Faithfulness of LLM Self-Explanations</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 66 pages, 12 figures
    </div>
    <details class="paper-abstract">
      When asked to explain their decisions, LLMs can often give explanations which sound plausible to humans. But are these explanations faithful, i.e. do they convey the factors actually responsible for the decision? In this work, we analyse counterfactual faithfulness across 75 models from 13 families. We analyze the tradeoff between conciseness and comprehensiveness, how correlational faithfulness metrics assess this tradeoff, and the extent to which metrics can be gamed. This analysis motivates two new metrics: the phi-CCT, a simplified variant of the Correlational Counterfactual Test (CCT) which avoids the need for token probabilities while explaining most of the variance of the original test; and F-AUROC, which eliminates sensitivity to imbalanced intervention distributions and captures a model's ability to produce explanations with different levels of detail. Our findings reveal a clear scaling trend: larger and more capable models are consistently more faithful on all metrics we consider. Our code is available at https://github.com/google-deepmind/corr_faith.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12397v5">Activated LoRA: Fine-tuned LLMs for Intrinsics</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Low-Rank Adaptation (LoRA) has emerged as a highly efficient framework for finetuning the weights of large foundation models, and has become the go-to method for data-driven customization of LLMs. Despite the promise of highly customized behaviors and capabilities, switching between relevant LoRAs in a multiturn setting is inefficient, as the key-value (KV) cache of the entire turn history must be recomputed with the LoRA weights before generation can begin. To address this problem, we propose Activated LoRA (aLoRA), an adapter architecture which modifies the LoRA framework to only adapt weights for the tokens in the sequence after the aLoRA is invoked. This change crucially allows aLoRA to accept the base model's KV cache of the input string, meaning that aLoRA can be instantly activated whenever needed in a chain without recomputing the prior keys and values. This enables building what we call intrinsics, i.e. specialized models invoked to perform well-defined operations on portions of an input chain or conversation that otherwise uses the base model by default. We train a set of aLoRA-based intrinsics models, demonstrating competitive accuracy with standard LoRA while significantly improving inference efficiency. We contributed our Activated LoRA implementation to the Huggingface PEFT library https://github.com/huggingface/peft.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00299v4">ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) require significant GPU memory when processing long texts, with the key value (KV) cache consuming up to 70\% of total memory during inference. Although existing compression methods reduce memory by evaluating the importance of individual tokens, they overlook critical semantic relationships between tokens, resulting in fragmented context and degraded performance. We introduce ChunkKV, which fundamentally reimagines KV cache compression by treating semantic chunks - rather than isolated tokens - as basic compression units. This approach preserves complete linguistic structures and contextual integrity, ensuring that essential meaning is retained even under aggressive compression. Our innovation includes a novel layer-wise index reuse technique that exploits the higher cross-layer similarity of preserved indices in ChunkKV, reducing computational overhead and improving throughput by 26.5\%. Comprehensive evaluations on challenging benchmarks: LongBench, Needle-In-A-HayStack, GSM8K, and JailbreakV demonstrate that ChunkKV outperforms state-of-the-art methods by up to 8.7\% in precision while maintaining the same compression ratio. These results confirm that semantic-aware compression significantly enhances both efficiency and performance for long-context LLM inference, providing a simple yet effective solution to the memory bottleneck problem. The code is available at \href{https://github.com/NVIDIA/kvpress}{link}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02483v1">Litespark Technical Report: High-Throughput, Energy-Efficient LLM Training Framework</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      Training Large Language Models (LLMs) is plagued by long training times and massive energy consumption, with modern models requiring months of computation and gigawatt-hours of electricity. In light of these challenges,we introduce Litespark, a novel pre-training framework that addresses these inefficiencies through targeted optimizations to transformer attention and MLP layers. Our approach combines architectural improvements with algorithmic enhancements to maximize Model FLOPs Utilization (MFU) while maintaining compatibility with standard transformer implementations. Comprehensive benchmarking on 3B and 30B parameter Llama models using the SlimPajama-627B dataset demonstrates substantial performance gains: 2x-6x training throughput improvement and $55\%-83$% energy consumption reduction across multi-node H200 GPU clusters. These optimizations are model- and hardware-agnostic, enabling broad applicability across transformer architectures and extending to post-training phases including supervised fine-tuning and direct preference optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02418v1">BrowserArena: Evaluating LLM Agents on Real-World Web Navigation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      LLM web agents now browse and take actions on the open web, yet current agent evaluations are constrained to sandboxed environments or artificial tasks. We introduce BrowserArena, a live open-web agent evaluation platform that collects user-submitted tasks, runs Arena-style head-to-head comparisons, and uses step-level human feedback to surface failure modes. Collecting and analyzing step-level annotations on the agent traces, we identify three consistent failure modes: captcha resolution, pop-up banner removal, and direct navigation to URLs. By constructing targeted datasets to further study these tasks, we discover variations in how different language models navigate these failure modes. We find, for example, that o4-mini deploys a wider variety of strategies to circumvent captcha resolution than other models and DeepSeek-R1 consistently misleads users about captcha resolution. Our findings surface both the diversity and brittleness of current web agents. More broadly, our benchmarking methodology provides an approach to evaluating and understanding web agent failure modes at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02306v1">Drawing Conclusions from Draws: Rethinking Preference Semantics in Arena-Style LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 6 pages, 4 figures
    </div>
    <details class="paper-abstract">
      In arena-style evaluation of large language models (LLMs), two LLMs respond to a user query, and the user chooses the winning response or deems the "battle" a draw, resulting in an adjustment to the ratings of both models. The prevailing approach for modeling these rating dynamics is to view battles as two-player game matches, as in chess, and apply the Elo rating system and its derivatives. In this paper, we critically examine this paradigm. Specifically, we question whether a draw genuinely means that the two models are equal and hence whether their ratings should be equalized. Instead, we conjecture that draws are more indicative of query difficulty: if the query is too easy, then both models are more likely to succeed equally. On three real-world arena datasets, we show that ignoring rating updates for draws yields a 1-3% relative increase in battle outcome prediction accuracy (which includes draws) for all four rating systems studied. Further analyses suggest that draws occur more for queries rated as very easy and those as highly objective, with risk ratios of 1.37 and 1.35, respectively. We recommend future rating systems to reconsider existing draw semantics and to account for query properties in rating updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.12021v2">LitterBox+: An Extensible Framework for LLM-enhanced Scratch Static Code Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 ASE 2025 Tool Demonstration Track
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become an essential tool to support developers using traditional text-based programming languages, but the graphical notation of the block-based Scratch programming environment inhibits the use of LLMs. To overcome this limitation, we propose the LitterBox+ framework that extends the Scratch static code analysis tool LitterBox with the generative abilities of LLMs. By converting block-based code to a textual representation suitable for LLMs, LitterBox+ allows users to query LLMs about their programs, about quality issues reported by LitterBox, and it allows generating code fixes. Besides offering a programmatic API for these functionalities, LitterBox+ also extends the Scratch user interface to make these functionalities available directly in the environment familiar to learners. The framework is designed to be easily extensible with other prompts, LLM providers, and new features combining the program analysis capabilities of LitterBox with the generative features of LLMs. We provide a screencast demonstrating the tool at https://youtu.be/RZ6E0xgrIgQ.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18436v4">Can Code-Switched Texts Activate a Knowledge Switch in LLMs? A Case Study on English-Korean Code-Switching</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Accepted to EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Recent large language models (LLMs) demonstrate multilingual abilities, yet they are English-centric due to dominance of English in training corpora. The limited resource for low-resource languages remains a crucial challenge. Code-switching (CS), a phenomenon where multilingual speakers alternate between languages in a discourse, can convey subtle cultural and linguistic nuances that can be otherwise lost in translation and elicits language-specific knowledge in human communications. In light of this, we investigate whether code-switching can activate, or identify and leverage knowledge for reasoning when LLMs solve low-resource language tasks. To facilitate the research, we first present EnKoQA, a synthetic English-Korean CS question-answering dataset. We provide comprehensive analysis on a variety of multilingual LLMs by subdividing activation process into knowledge identification and knowledge leveraging. Our results demonstrate that compared to English text, CS can faithfully activate knowledge inside LLMs especially on language-specific domains, suggesting the potential of code-switching on low-resource language tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02249v1">Explore Briefly, Then Decide: Mitigating LLM Overthinking via Cumulative Entropy Regulation</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable reasoning abilities on complex problems using long Chain-of-Thought (CoT) reasoning. However, they often suffer from overthinking, meaning generating unnecessarily lengthy reasoning steps for simpler problems. This issue may degrade the efficiency of the models and make them difficult to adapt the reasoning depth to the complexity of problems. To address this, we introduce a novel metric Token Entropy Cumulative Average (TECA), which measures the extent of exploration throughout the reasoning process. We further propose a novel reasoning paradigm -- Explore Briefly, Then Decide -- with an associated Cumulative Entropy Regulation (CER) mechanism. This paradigm leverages TECA to help the model dynamically determine the optimal point to conclude its thought process and provide a final answer, thus achieving efficient reasoning. Experimental results across diverse mathematical benchmarks show that our approach substantially mitigates overthinking without sacrificing problem-solving ability. With our thinking paradigm, the average response length decreases by up to 71% on simpler datasets, demonstrating the effectiveness of our method in creating a more efficient and adaptive reasoning process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09674v4">Probabilistic Reasoning with LLMs for k-anonymity Estimation</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 10 pages, Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Probabilistic reasoning is a key aspect of both human and artificial intelligence that allows for handling uncertainty and ambiguity in decision-making. In this paper, we introduce a new numerical reasoning task under uncertainty for large language models, focusing on estimating the privacy risk of user-generated documents containing privacy-sensitive information. We propose BRANCH, a new LLM methodology that estimates the k-privacy value of a text-the size of the population matching the given information. BRANCH factorizes a joint probability distribution of personal information as random variables. The probability of each factor in a population is estimated separately using a Bayesian network and combined to compute the final k-value. Our experiments show that this method successfully estimates the k-value 73% of the time, a 13% increase compared to o3-mini with chain-of-thought reasoning. We also find that LLM uncertainty is a good indicator for accuracy, as high-variance predictions are 37.47% less accurate on average.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02241v1">Study on LLMs for Promptagator-Style Dense Retriever Training</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 CIKM 2025 short research paper
    </div>
    <details class="paper-abstract">
      Promptagator demonstrated that Large Language Models (LLMs) with few-shot prompts can be used as task-specific query generators for fine-tuning domain-specialized dense retrieval models. However, the original Promptagator approach relied on proprietary and large-scale LLMs which users may not have access to or may be prohibited from using with sensitive data. In this work, we study the impact of open-source LLMs at accessible scales ($\leq$14B parameters) as an alternative. Our results demonstrate that open-source LLMs as small as 3B parameters can serve as effective Promptagator-style query generators. We hope our work will inform practitioners with reliable alternatives for synthetic data generation and give insights to maximize fine-tuning results for domain-specific applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00708v2">DrKGC: Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion across General and Biomedical Domains</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Accepted at EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Knowledge graph completion (KGC) aims to predict missing triples in knowledge graphs (KGs) by leveraging existing triples and textual information. Recently, generative large language models (LLMs) have been increasingly employed for graph tasks. However, current approaches typically encode graph context in textual form, which fails to fully exploit the potential of LLMs for perceiving and reasoning about graph structures. To address this limitation, we propose DrKGC (Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion). DrKGC employs a flexible lightweight model training strategy to learn structural embeddings and logical rules within the KG. It then leverages a novel bottom-up graph retrieval method to extract a subgraph for each query guided by the learned rules. Finally, a graph convolutional network (GCN) adapter uses the retrieved subgraph to enhance the structural embeddings, which are then integrated into the prompt for effective LLM fine-tuning. Experimental results on two general domain benchmark datasets and two biomedical datasets demonstrate the superior performance of DrKGC. Furthermore, a realistic case study in the biomedical domain highlights its interpretability and practical utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02209v1">StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets?</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently demonstrated strong capabilities as autonomous agents, showing promise in reasoning, tool use, and sequential decision-making. While prior benchmarks have evaluated LLM agents in domains such as software engineering and scientific discovery, the finance domain remains underexplored, despite its direct relevance to economic value and high-stakes decision-making. Existing financial benchmarks primarily test static knowledge through question answering, but they fall short of capturing the dynamic and iterative nature of trading. To address this gap, we introduce StockBench, a contamination-free benchmark designed to evaluate LLM agents in realistic, multi-month stock trading environments. Agents receive daily market signals -- including prices, fundamentals, and news -- and must make sequential buy, sell, or hold decisions. Performance is assessed using financial metrics such as cumulative return, maximum drawdown, and the Sortino ratio. Our evaluation of state-of-the-art proprietary (e.g., GPT-5, Claude-4) and open-weight (e.g., Qwen3, Kimi-K2, GLM-4.5) models shows that while most LLM agents struggle to outperform the simple buy-and-hold baseline, several models demonstrate the potential to deliver higher returns and manage risk more effectively. These findings highlight both the challenges and opportunities in developing LLM-powered financial agents, showing that excelling at static financial knowledge tasks does not necessarily translate into successful trading strategies. We release StockBench as an open-source resource to support reproducibility and advance future research in this domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02200v1">ARUQULA -- An LLM based Text2SPARQL Approach using ReAct and Knowledge Graph Exploration Utilities</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 peer reviewed publication at Text2SPARQL Workshop @ ESWC 2025
    </div>
    <details class="paper-abstract">
      Interacting with knowledge graphs can be a daunting task for people without a background in computer science since the query language that is used (SPARQL) has a high barrier of entry. Large language models (LLMs) can lower that barrier by providing support in the form of Text2SPARQL translation. In this paper we introduce a generalized method based on SPINACH, an LLM backed agent that translates natural language questions to SPARQL queries not in a single shot, but as an iterative process of exploration and execution. We describe the overall architecture and reasoning behind our design decisions, and also conduct a thorough analysis of the agent behavior to gain insights into future areas for targeted improvements. This work was motivated by the Text2SPARQL challenge, a challenge that was held to facilitate improvements in the Text2SPARQL domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.21499v2">On Code-Induced Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-02
    </div>
    <details class="paper-abstract">
      Code data has been shown to enhance the reasoning capabilities of large language models (LLMs), but it remains unclear which aspects of code are most responsible. We investigate this question with a systematic, data-centric framework. We construct parallel instruction datasets in ten programming languages and apply controlled perturbations that selectively disrupt structural or semantic properties of code. We then finetune LLMs from five model families and eight scales on each variant and evaluate their performance on natural language, math, and code tasks. Across 3,331 experiments, our results show that LLMs are more vulnerable to structural perturbations than semantic ones, particularly on math and code tasks. Appropriate abstractions like pseudocode and flowcharts can be as effective as code, while encoding the same information with fewer tokens without adhering to original syntax can often retain or even improve performance. Remarkably, even corrupted code with misleading signals remains competitive when surface-level regularities persist. Finally, syntactic styles also shape task-specific gains with Python favoring natural language reasoning and lower-level languages such as Java and Rust favoring math. Through our systematic framework, we aim to provide insight into how different properties of code influence reasoning and inform the design of training data for enhancing LLM reasoning capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15153v2">When Disagreements Elicit Robustness: Investigating Self-Repair Capabilities under LLM Multi-Agent Disagreements</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 Working in progress
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have upgraded them from sophisticated text generators to autonomous agents capable of cooperation and tool use in multi-agent systems (MAS). However, it remains unclear how disagreements shape collective decision-making. In this paper, we revisit the role of disagreement and argue that general, partially overlapping disagreements prevent premature consensus and expand the explored solution space, while disagreements on task-critical steps can derail collaboration depending on the topology of solution paths. We investigate two collaborative settings with distinct path structures: collaborative reasoning (CounterFact, MQuAKE-cf), which typically follows a single evidential chain, whereas collaborative programming (HumanEval, GAIA) often adopts multiple valid implementations. Disagreements are instantiated as general heterogeneity among agents and as task-critical counterfactual knowledge edits injected into context or parameters. Experiments reveal that general disagreements consistently improve success by encouraging complementary exploration. By contrast, task-critical disagreements substantially reduce success on single-path reasoning, yet have a limited impact on programming, where agents can choose alternative solutions. Trace analyses show that MAS frequently bypasses the edited facts in programming but rarely does so in reasoning, revealing an emergent self-repair capability that depends on solution-path rather than scale alone. Our code is available at https://github.com/wbw625/MultiAgentRobustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18379v2">REALM: Recursive Relevance Modeling for LLM-based Document Re-Ranking</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 EMNLP 2025 (Main Conference, Oral). 15 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong capabilities in document re-ranking, a key component in modern Information Retrieval (IR) systems. However, existing LLM-based approaches face notable limitations, including ranking uncertainty, unstable top-k recovery, and high token cost due to token-intensive prompting. To effectively address these limitations, we propose REALM, an uncertainty-aware re-ranking framework that models LLM-derived relevance as Gaussian distributions and refines them through recursive Bayesian updates. By explicitly capturing uncertainty and minimizing redundant queries, REALM achieves better rankings more efficiently. Experimental results demonstrate that our REALM surpasses state-of-the-art re-rankers while significantly reducing token usage and latency, improving NDCG@10 by 0.7-11.9 and simultaneously reducing the number of LLM inferences by 23.4-84.4%, promoting it as the next-generation re-ranker for modern IR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02104v1">LangGrasp: Leveraging Fine-Tuned LLMs for Language Interactive Robot Grasping with Ambiguous Instructions</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The existing language-driven grasping methods struggle to fully handle ambiguous instructions containing implicit intents. To tackle this challenge, we propose LangGrasp, a novel language-interactive robotic grasping framework. The framework integrates fine-tuned large language models (LLMs) to leverage their robust commonsense understanding and environmental perception capabilities, thereby deducing implicit intents from linguistic instructions and clarifying task requirements along with target manipulation objects. Furthermore, our designed point cloud localization module, guided by 2D part segmentation, enables partial point cloud localization in scenes, thereby extending grasping operations from coarse-grained object-level to fine-grained part-level manipulation. Experimental results show that the LangGrasp framework accurately resolves implicit intents in ambiguous instructions, identifying critical operations and target information that are unstated yet essential for task completion. Additionally, it dynamically selects optimal grasping poses by integrating environmental information. This enables high-precision grasping from object-level to part-level manipulation, significantly enhancing the adaptability and task execution efficiency of robots in unstructured environments. More information and code are available here: https://github.com/wu467/LangGrasp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02091v1">Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-02
      | 💬 ICASSP 2025
    </div>
    <details class="paper-abstract">
      Recent studies suggest that the deeper layers of Large Language Models (LLMs) contribute little to representation learning and can often be removed without significant performance loss. However, such claims are typically drawn from narrow evaluations and may overlook important aspects of model behavior. In this work, we present a systematic study of depth utilization across diverse dimensions, including evaluation protocols, task categories, and model architectures. Our analysis confirms that very deep layers are generally less effective than earlier ones, but their contributions vary substantially with the evaluation setting. Under likelihood-based metrics without generation, pruning most layers preserves performance, with only the initial few being critical. By contrast, generation-based evaluation uncovers indispensable roles for middle and deeper layers in enabling reasoning and maintaining long-range coherence. We further find that knowledge and retrieval are concentrated in shallow components, whereas reasoning accuracy relies heavily on deeper layers -- yet can be reshaped through distillation. These results highlight that depth usage in LLMs is highly heterogeneous and context-dependent, underscoring the need for task-, metric-, and model-aware perspectives in both interpreting and compressing large models.
    </details>
</div>
