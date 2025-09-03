# llm - 2025_08

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
- [Part 13](papers_13.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17814v1">Scalable Engine and the Performance of Different LLM Models in a SLURM based HPC architecture</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Accepted in ESSV 2025 - https://www.essv.de/paper.php?id=1265
    </div>
    <details class="paper-abstract">
      This work elaborates on a High performance computing (HPC) architecture based on Simple Linux Utility for Resource Management (SLURM) [1] for deploying heterogeneous Large Language Models (LLMs) into a scalable inference engine. Dynamic resource scheduling and seamless integration of containerized microservices have been leveraged herein to manage CPU, GPU, and memory allocations efficiently in multi-node clusters. Extensive experiments, using Llama 3.2 (1B and 3B parameters) [2] and Llama 3.1 (8B and 70B) [3], probe throughput, latency, and concurrency and show that small models can handle up to 128 concurrent requests at sub-50 ms latency, while for larger models, saturation happens with as few as two concurrent users, with a latency of more than 2 seconds. This architecture includes Representational State Transfer Application Programming Interfaces (REST APIs) [4] endpoints for single and bulk inferences, as well as advanced workflows such as multi-step "tribunal" refinement. Experimental results confirm minimal overhead from container and scheduling activities and show that the approach scales reliably both for batch and interactive settings. We further illustrate real-world scenarios, including the deployment of chatbots with retrievalaugmented generation, which helps to demonstrate the flexibility and robustness of the architecture. The obtained results pave ways for significantly more efficient, responsive, and fault-tolerant LLM inference on large-scale HPC infrastructures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17771v1">Speculating LLMs' Chinese Training Data Pollution from Their Tokens</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Tokens are basic elements in the datasets for LLM training. It is well-known that many tokens representing Chinese phrases in the vocabulary of GPT (4o/4o-mini/o1/o3/4.5/4.1/o4-mini) are indicating contents like pornography or online gambling. Based on this observation, our goal is to locate Polluted Chinese (PoC) tokens in LLMs and study the relationship between PoC tokens' existence and training data. (1) We give a formal definition and taxonomy of PoC tokens based on the GPT's vocabulary. (2) We build a PoC token detector via fine-tuning an LLM to label PoC tokens in vocabularies by considering each token's both semantics and related contents from the search engines. (3) We study the speculation on the training data pollution via PoC tokens' appearances (token ID). Experiments on GPT and other 23 LLMs indicate that tokens widely exist while GPT's vocabulary behaves the worst: more than 23% long Chinese tokens (i.e., a token with more than two Chinese characters) are either porn or online gambling. We validate the accuracy of our speculation method on famous pre-training datasets like C4 and Pile. Then, considering GPT-4o, we speculate that the ratio of "Yui Hatano" related webpages in GPT-4o's training data is around 0.5%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.22316v2">Evaluating Scoring Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      The remarkable performance of Large Language Models (LLMs) gives rise to``LLM-as-a-Judge'', where LLMs are employed as evaluators for complex tasks. Moreover, it has been widely adopted across fields such as Natural Language Processing (NLP), preference learning, and various specific domains. However, there are various biases within LLM-as-a-Judge, which adversely affect the fairness and reliability of judgments. Current research on evaluating or mitigating bias in LLM-as-a-Judge predominantly focuses on comparison-based evaluations, while systematic investigations into bias in scoring-based evaluations remain limited. Therefore, we define scoring bias in LLM-as-a-Judge as the scores differ when scoring judge models are bias-related perturbed, and provide a well-designed framework to comprehensively evaluate scoring bias. We augment existing LLM-as-a-Judge benchmarks through data synthesis to construct our evaluation dataset and design multi-faceted evaluation metrics. Our experimental results demonstrate that the scoring stability of existing judge models is disrupted by scoring biases. Further exploratory experiments and discussions provide valuable insights into the design of scoring prompt templates and the mitigation of scoring biases on aspects such as score rubrics, score IDs, and reference answer selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01263v2">FlowDubber: Movie Dubbing with LLM-based Semantic-aware Learning and Flow Matching based Voice Enhancing</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Movie Dubbing aims to convert scripts into speeches that align with the given movie clip in both temporal and emotional aspects while preserving the vocal timbre of a given brief reference audio. Existing methods focus primarily on reducing the word error rate while ignoring the importance of lip-sync and acoustic quality. To address these issues, we propose a large language model (LLM) based flow matching architecture for dubbing, named FlowDubber, which achieves high-quality audio-visual sync and pronunciation by incorporating a large speech language model and dual contrastive aligning while achieving better acoustic quality via the proposed voice-enhanced flow matching than previous works. First, we introduce Qwen2.5 as the backbone of LLM to learn the in-context sequence from movie scripts and reference audio. Then, the proposed semantic-aware learning focuses on capturing LLM semantic knowledge at the phoneme level. Next, dual contrastive aligning (DCA) boosts mutual alignment with lip movement, reducing ambiguities where similar phonemes might be confused. Finally, the proposed Flow-based Voice Enhancing (FVE) improves acoustic quality in two aspects, which introduces an LLM-based acoustics flow matching guidance to strengthen clarity and uses affine style prior to enhance identity when recovering noise into mel-spectrograms via gradient vector field prediction. Extensive experiments demonstrate that our method outperforms several state-of-the-art methods on two primary benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17735v1">SMITE: Enhancing Fairness in LLMs through Optimal In-Context Example Selection via Dynamic Validation</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used for downstream tasks such as tabular classification, where ensuring fairness in their outputs is critical for inclusivity, equal representation, and responsible AI deployment. This study introduces a novel approach to enhancing LLM performance and fairness through the concept of a dynamic validation set, which evolves alongside the test set, replacing the traditional static validation approach. We also propose an iterative algorithm, SMITE, to select optimal in-context examples, with each example set validated against its corresponding dynamic validation set. The in-context set with the lowest total error is used as the final demonstration set. Our experiments across four different LLMs show that our proposed techniques significantly improve both predictive accuracy and fairness compared to baseline methods. To our knowledge, this is the first study to apply dynamic validation in the context of in-context learning for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17720v1">RepoTransAgent: Multi-Agent LLM Framework for Repository-Aware Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Repository-aware code translation is critical for modernizing legacy systems, enhancing maintainability, and enabling interoperability across diverse programming languages. While recent advances in large language models (LLMs) have improved code translation quality, existing approaches face significant challenges in practical scenarios: insufficient contextual understanding, inflexible prompt designs, and inadequate error correction mechanisms. These limitations severely hinder accurate and efficient translation of complex, real-world code repositories. To address these challenges, we propose RepoTransAgent, a novel multi-agent LLM framework for repository-aware code translation. RepoTransAgent systematically decomposes the translation process into specialized subtasks-context retrieval, dynamic prompt construction, and iterative code refinement-each handled by dedicated agents. Our approach leverages retrieval-augmented generation (RAG) for contextual information gathering, employs adaptive prompts tailored to varying repository scenarios, and introduces a reflection-based mechanism for systematic error correction. We evaluate RepoTransAgent on hundreds of Java-C# translation pairs from six popular open-source projects. Experimental results demonstrate that RepoTransAgent significantly outperforms state-of-the-art baselines in both compile and pass rates. Specifically, RepoTransAgent achieves up to 55.34% compile rate and 45.84% pass rate. Comprehensive analysis confirms the robustness and generalizability of RepoTransAgent across different LLMs, establishing its effectiveness for real-world repository-aware code translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17715v1">How Do LLM-Generated Texts Impact Term-Based Retrieval Models?</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      As more content generated by large language models (LLMs) floods into the Internet, information retrieval (IR) systems now face the challenge of distinguishing and handling a blend of human-authored and machine-generated texts. Recent studies suggest that neural retrievers may exhibit a preferential inclination toward LLM-generated content, while classic term-based retrievers like BM25 tend to favor human-written documents. This paper investigates the influence of LLM-generated content on term-based retrieval models, which are valued for their efficiency and robust generalization across domains. Our linguistic analysis reveals that LLM-generated texts exhibit smoother high-frequency and steeper low-frequency Zipf slopes, higher term specificity, and greater document-level diversity. These traits are aligned with LLMs being trained to optimize reader experience through diverse and precise expressions. Our study further explores whether term-based retrieval models demonstrate source bias, concluding that these models prioritize documents whose term distributions closely correspond to those of the queries, rather than displaying an inherent source bias. This work provides a foundation for understanding and addressing potential biases in term-based IR systems managing mixed-source content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17711v1">Enhancing LLM-Based Social Bot via an Adversarial Learning Framework</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Developing Large Language Model (LLM) agents that exhibit human-like behavior, encompassing not only individual heterogeneity rooted in unique user profiles but also adaptive response to socially connected neighbors, is a significant research challenge. Social media platforms, with their diverse user data and explicit social structures, provide an ideal testbed for such investigations. This paper introduces EvoBot, an \textbf{Evo}lving LLM-based social \textbf{Bot} that significantly enhances human-like generative capabilities through a novel adversarial learning framework. EvoBot is initialized by Supervised Fine-Tuning (SFT) on representative data from social media and then iteratively refines its generation of sophisticated, human-like content via Direct Preference Optimization (DPO). This refinement is guided by feedback from a co-adapting \textbf{Detector} which concurrently improves its ability to distinguish EvoBot from humans, thereby creating an increasingly challenging learning environment for EvoBot. Experiments demonstrate that EvoBot generates content aligned with diverse user profiles, increasingly bypassing the co-adapting Detector through human-like expression. Moreover, it exhibits strong social responsiveness, more accurately modeling real-world opinion dynamics and information spread in multi-agent simulations. The framework also yields a more robust Detector, underscoring its broader utility for both advanced agent development and related detection tasks. The code is available at https://github.com/kfq20/EvoBot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10904v2">A2HCoder: An LLM-Driven Coding Agent for Hierarchical Algorithm-to-HDL Translation</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      In wireless communication systems, stringent requirements such as ultra-low latency and power consumption have significantly increased the demand for efficient algorithm-to-hardware deployment. However, a persistent and substantial gap remains between algorithm design and hardware implementation. Bridging this gap traditionally requires extensive domain expertise and time-consuming manual development, due to fundamental mismatches between high-level programming languages like MATLAB and hardware description languages (HDLs) such as Verilog-in terms of memory access patterns, data processing manners, and datatype representations. To address this challenge, we propose A2HCoder: a Hierarchical Algorithm-to-HDL Coding Agent, powered by large language models (LLMs), designed to enable agile and reliable algorithm-to-hardware translation. A2HCoder introduces a hierarchical framework that enhances both robustness and interpretability while suppressing common hallucination issues in LLM-generated code. In the horizontal dimension, A2HCoder decomposes complex algorithms into modular functional blocks, simplifying code generation and improving consistency. In the vertical dimension, instead of relying on end-to-end generation, A2HCoder performs step-by-step, fine-grained translation, leveraging external toolchains such as MATLAB and Vitis HLS for debugging and circuit-level synthesis. This structured process significantly mitigates hallucinations and ensures hardware-level correctness. We validate A2HCoder through a real-world deployment case in the 5G wireless communication domain, demonstrating its practicality, reliability, and deployment efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01070v4">An Inquiry into Datacenter TCO for LLM Inference with FP8</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to scale, the high power consumption of AI accelerators in datacenters presents significant challenges, substantially increasing the total cost of ownership (TCO) for cloud service providers (CSPs) that provide LLM inference. In this work, we analyze the computational characteristics of LLM inference from a TCO perspective and present a generalizable framework to compare AI accelerators across diverse operational requirements. Using this model, we investigate key workload characteristics influencing TCO for AI accelerators from Intel (Gaudi 2 & 3) and NVIDIA (H100 & H200), especially thin GEMM utilization and FP8 quantization. In particular, as FP8 emerges as the baseline precision for next-generation LLMs, understanding how different architectures implement and benefit from low-precision computation is increasingly critical. Throughput on thin GEMMs has a greater impact on TCO than theoretical hardware peak throughput because the memory-bound decode phase is dominated by GEMV-like computations. We find that Gaudi HPUs achieve superior utilization on thin GEMMs compared to their counterparts, especially in FP8-quantized models. Our result underscores the importance of empirical, workload-level analysis in evaluating accelerator performance, rather than relying solely on theoretical hardware specifications. By studying the interaction between power consumption, quantization strategies, and hardware architecture, we provide insights to support informed deployment decisions and guide future accelerator designs aimed at improving the TCO of LLM inference workloads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17692v1">LLM-based Agentic Reasoning Frameworks: A Survey from Methods to Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 51 pages,10 figures,8 tables. Work in progress
    </div>
    <details class="paper-abstract">
      Recent advances in the intrinsic reasoning capabilities of large language models (LLMs) have given rise to LLM-based agent systems that exhibit near-human performance on a variety of automated tasks. However, although these systems share similarities in terms of their use of LLMs, different reasoning frameworks of the agent system steer and organize the reasoning process in different ways. In this survey, we propose a systematic taxonomy that decomposes agentic reasoning frameworks and analyze how these frameworks dominate framework-level reasoning by comparing their applications across different scenarios. Specifically, we propose an unified formal language to further classify agentic reasoning systems into single-agent methods, tool-based methods, and multi-agent methods. After that, we provide a comprehensive review of their key application scenarios in scientific discovery, healthcare, software engineering, social simulation, and economics. We also analyze the characteristic features of each framework and summarize different evaluation strategies. Our survey aims to provide the research community with a panoramic view to facilitate understanding of the strengths, suitable scenarios, and evaluation practices of different agentic reasoning frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17674v1">Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 7 pages, 2 figures
    </div>
    <details class="paper-abstract">
      We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17644v1">Demographically-Inspired Query Variants Using an LLM</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Published in the proceedings of ICTIR'25, Padua, Italy
    </div>
    <details class="paper-abstract">
      This study proposes a method to diversify queries in existing test collections to reflect some of the diversity of search engine users, aligning with an earlier vision of an 'ideal' test collection. A Large Language Model (LLM) is used to create query variants: alternative queries that have the same meaning as the original. These variants represent user profiles characterised by different properties, such as language and domain proficiency, which are known in the IR literature to influence query formulation. The LLM's ability to generate query variants that align with user profiles is empirically validated, and the variants' utility is further explored for IR system evaluation. Results demonstrate that the variants impact how systems are ranked and show that user profiles experience significantly different levels of system effectiveness. This method enables an alternative perspective on system evaluation where we can observe both the impact of user profiles on system rankings and how system performance varies across users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17627v1">Stop Spinning Wheels: Mitigating LLM Overthinking via Mining Patterns for Early Reasoning Exit</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) enhance complex reasoning tasks by scaling the individual thinking process. However, prior work shows that overthinking can degrade overall performance. Motivated by observed patterns in thinking length and content length, we categorize reasoning into three stages: insufficient exploration stage, compensatory reasoning stage, and reasoning convergence stage. Typically, LLMs produce correct answers in the compensatory reasoning stage, whereas reasoning convergence often triggers overthinking, causing increased resource usage or even infinite loops. Therefore, mitigating overthinking hinges on detecting the end of the compensatory reasoning stage, defined as the Reasoning Completion Point (RCP). RCP typically appears at the end of the first complete reasoning cycle and can be identified by querying the LLM sentence by sentence or monitoring the probability of an end-of-thinking token (e.g., \texttt{</think>}), though these methods lack an efficient and precise balance. To improve this, we mine more sensitive and consistent RCP patterns and develop a lightweight thresholding strategy based on heuristic rules. Experimental evaluations on benchmarks (AIME24, AIME25, GPQA-D) demonstrate that the proposed method reduces token consumption while preserving or enhancing reasoning accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17573v1">Humanizing Machines: Rethinking LLM Anthropomorphism Through a Multi-Level Framework of Design</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Accepted in EMNLP main proceedings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly exhibit \textbf{anthropomorphism} characteristics -- human-like qualities portrayed across their outlook, language, behavior, and reasoning functions. Such characteristics enable more intuitive and engaging human-AI interactions. However, current research on anthropomorphism remains predominantly risk-focused, emphasizing over-trust and user deception while offering limited design guidance. We argue that anthropomorphism should instead be treated as a \emph{concept of design} that can be intentionally tuned to support user goals. Drawing from multiple disciplines, we propose that the anthropomorphism of an LLM-based artifact should reflect the interaction between artifact designers and interpreters. This interaction is facilitated by cues embedded in the artifact by the designers and the (cognitive) responses of the interpreters to the cues. Cues are categorized into four dimensions: \textit{perceptive, linguistic, behavioral}, and \textit{cognitive}. By analyzing the manifestation and effectiveness of each cue, we provide a unified taxonomy with actionable levers for practitioners. Consequently, we advocate for function-oriented evaluations of anthropomorphic design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18547v1">How do Humans and LLMs Process Confusing Code?</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Already today, humans and programming assistants based on large language models (LLMs) collaborate in everyday programming tasks. Clearly, a misalignment between how LLMs and programmers comprehend code can lead to misunderstandings, inefficiencies, low code quality, and bugs. A key question in this space is whether humans and LLMs are confused by the same kind of code. This would not only guide our choices of integrating LLMs in software engineering workflows, but also inform about possible improvements of LLMs. To this end, we conducted an empirical study comparing an LLM to human programmers comprehending clean and confusing code. We operationalized comprehension for the LLM by using LLM perplexity, and for human programmers using neurophysiological responses (in particular, EEG-based fixation-related potentials). We found that LLM perplexity spikes correlate both in terms of location and amplitude with human neurophysiological responses that indicate confusion. This result suggests that LLMs and humans are similarly confused about the code. Based on these findings, we devised a data-driven, LLM-based approach to identify regions of confusion in code that elicit confusion in human programmers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18533v1">A Database-Driven Framework for 3D Level Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Procedural Content Generation for 3D game levels faces challenges in balancing spatial coherence, navigational functionality, and adaptable gameplay progression across multi-floor environments. This paper introduces a novel framework for generating such levels, centered on the offline, LLM-assisted construction of reusable databases for architectural components (facilities and room templates) and gameplay mechanic elements. Our multi-phase pipeline assembles levels by: (1) selecting and arranging instances from the Room Database to form a multi-floor global structure with an inherent topological order; (2) optimizing the internal layout of facilities for each room based on predefined constraints from the Facility Database; and (3) integrating progression-based gameplay mechanics by placing components from a Mechanics Database according to their topological and spatial rules. A subsequent two-phase repair system ensures navigability. This approach combines modular, database-driven design with constraint-based optimization, allowing for systematic control over level structure and the adaptable pacing of gameplay elements. Initial experiments validate the framework's ability in generating diverse, navigable 3D environments and its capability to simulate distinct gameplay pacing strategies through simple parameterization. This research advances PCG by presenting a scalable, database-centric foundation for the automated generation of complex 3D levels with configurable gameplay progression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18467v1">The AI in the Mirror: LLM Self-Recognition in an Iterated Public Goods Game</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      As AI agents become increasingly capable of tool use and long-horizon tasks, they have begun to be deployed in settings where multiple agents can interact. However, whereas prior work has mostly focused on human-AI interactions, there is an increasing need to understand AI-AI interactions. In this paper, we adapt the iterated public goods game, a classic behavioral economics game, to analyze the behavior of four reasoning and non-reasoning models across two conditions: models are either told they are playing against "another AI agent" or told their opponents are themselves. We find that, across different settings, telling LLMs that they are playing against themselves significantly changes their tendency to cooperate. While our study is conducted in a toy environment, our results may provide insights into multi-agent settings where agents "unconsciously" discriminating against each other could inexplicably increase or decrease cooperation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16571v2">LLM-Based Agents for Competitive Landscape Mapping in Drug Asset Due Diligence</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      In this paper, we describe and benchmark a competitor-discovery component used within an agentic AI system for fast drug asset due diligence. A competitor-discovery AI agent, given an indication, retrieves all drugs comprising the competitive landscape of that indication and extracts canonical attributes for these drugs. The competitor definition is investor-specific, and data is paywalled/licensed, fragmented across registries, ontology-mismatched by indication, alias-heavy for drug names, multimodal, and rapidly changing. Although considered the best tool for this problem, the current LLM-based AI systems aren't capable of reliably retrieving all competing drug names, and there is no accepted public benchmark for this task. To address the lack of evaluation, we use LLM-based agents to transform five years of multi-modal, unstructured diligence memos from a private biotech VC fund into a structured evaluation corpus mapping indications to competitor drugs with normalized attributes. We also introduce a competitor validating LLM-as-a-judge agent that filters out false positives from the list of predicted competitors to maximize precision and suppress hallucinations. On this benchmark, our competitor-discovery agent achieves 83% recall, exceeding OpenAI Deep Research (65%) and Perplexity Labs (60%). The system is deployed in production with enterprise users; in a case study with a biotech VC investment fund, analyst turnaround time dropped from 2.5 days to $\sim$3 hours ($\sim$20x) for the competitive analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18444v1">How Reliable are LLMs for Reasoning on the Re-ranking task?</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Accepted at FQAS Conference 2024. DOI will be provided in 3 weeks after the conference has published the paper
    </div>
    <details class="paper-abstract">
      With the improving semantic understanding capability of Large Language Models (LLMs), they exhibit a greater awareness and alignment with human values, but this comes at the cost of transparency. Although promising results are achieved via experimental analysis, an in-depth understanding of the LLM's internal workings is unavoidable to comprehend the reasoning behind the re-ranking, which provides end users with an explanation that enables them to make an informed decision. Moreover, in newly developed systems with limited user engagement and insufficient ranking data, accurately re-ranking content remains a significant challenge. While various training methods affect the training of LLMs and generate inference, our analysis has found that some training methods exhibit better explainability than others, implying that an accurate semantic understanding has not been learned through all training methods; instead, abstract knowledge has been gained to optimize evaluation, which raises questions about the true reliability of LLMs. Therefore, in this work, we analyze how different training methods affect the semantic understanding of the re-ranking task in LLMs and investigate whether these models can generate more informed textual reasoning to overcome the challenges of transparency or LLMs and limited training data. To analyze the LLMs for re-ranking tasks, we utilize a relatively small ranking dataset from the environment and the Earth science domain to re-rank retrieved content. Furthermore, we also analyze the explainable information to see if the re-ranking can be reasoned using explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18439v1">A Systematic Approach to Predict the Impact of Cybersecurity Vulnerabilities Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Vulnerability databases, such as the National Vulnerability Database (NVD), offer detailed descriptions of Common Vulnerabilities and Exposures (CVEs), but often lack information on their real-world impact, such as the tactics, techniques, and procedures (TTPs) that adversaries may use to exploit the vulnerability. However, manually linking CVEs to their corresponding TTPs is a challenging and time-consuming task, and the high volume of new vulnerabilities published annually makes automated support desirable. This paper introduces TRIAGE, a two-pronged automated approach that uses Large Language Models (LLMs) to map CVEs to relevant techniques from the ATT&CK knowledge base. We first prompt an LLM with instructions based on MITRE's CVE Mapping Methodology to predict an initial list of techniques. This list is then combined with the results from a second LLM-based module that uses in-context learning to map a CVE to relevant techniques. This hybrid approach strategically combines rule-based reasoning with data-driven inference. Our evaluation reveals that in-context learning outperforms the individual mapping methods, and the hybrid approach improves recall of exploitation techniques. We also find that GPT-4o-mini performs better than Llama3.3-70B on this task. Overall, our results show that LLMs can be used to automatically predict the impact of cybersecurity vulnerabilities and TRIAGE makes the process of mapping CVEs to ATT&CK more efficient. Keywords: vulnerability impact, CVE, ATT&CK techniques, large language models, automated mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18420v1">LLM-Driven Intrinsic Motivation for Sparse Reward Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 11 pages, 5 figures, Accepted to the ENIAC 2025 conference
    </div>
    <details class="paper-abstract">
      This paper explores the combination of two intrinsic motivation strategies to improve the efficiency of reinforcement learning (RL) agents in environments with extreme sparse rewards, where traditional learning struggles due to infrequent positive feedback. We propose integrating Variational State as Intrinsic Reward (VSIMR), which uses Variational AutoEncoders (VAEs) to reward state novelty, with an intrinsic reward approach derived from Large Language Models (LLMs). The LLMs leverage their pre-trained knowledge to generate reward signals based on environment and goal descriptions, guiding the agent. We implemented this combined approach with an Actor-Critic (A2C) agent in the MiniGrid DoorKey environment, a benchmark for sparse rewards. Our empirical results show that this combined strategy significantly increases agent performance and sampling efficiency compared to using each strategy individually or a standard A2C agent, which failed to learn. Analysis of learning curves indicates that the combination effectively complements different aspects of the environment and task: VSIMR drives exploration of new states, while the LLM-derived rewards facilitate progressive exploitation towards goals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18379v1">REALM: Recursive Relevance Modeling for LLM-based Document Re-Ranking</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Accepted to EMNLP 2025 (Main Conference). 13 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong capabilities in document re-ranking, a key component in modern Information Retrieval (IR) systems. However, existing LLM-based approaches face notable limitations, including ranking uncertainty, unstable top-k recovery, and high token cost due to token-intensive prompting. To effectively address these limitations, we propose REALM, an uncertainty-aware re-ranking framework that models LLM-derived relevance as Gaussian distributions and refines them through recursive Bayesian updates. By explicitly capturing uncertainty and minimizing redundant queries, REALM achieves better rankings more efficiently. Experimental results demonstrate that our REALM surpasses state-of-the-art re-rankers while significantly reducing token usage and latency, promoting it as the next-generation re-ranker for modern IR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19288v1">Tricking LLM-Based NPCs into Spilling Secrets</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to generate dynamic dialogue for game NPCs. However, their integration raises new security concerns. In this study, we examine whether adversarial prompt injection can cause LLM-based NPCs to reveal hidden background secrets that are meant to remain undisclosed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19287v1">Prompt-in-Content Attacks: Exploiting Uploaded Inputs to Hijack LLM Behavior</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely deployed in applications that accept user-submitted content, such as uploaded documents or pasted text, for tasks like summarization and question answering. In this paper, we identify a new class of attacks, prompt in content injection, where adversarial instructions are embedded in seemingly benign inputs. When processed by the LLM, these hidden prompts can manipulate outputs without user awareness or system compromise, leading to biased summaries, fabricated claims, or misleading suggestions. We demonstrate the feasibility of such attacks across popular platforms, analyze their root causes including prompt concatenation and insufficient input isolation, and discuss mitigation strategies. Our findings reveal a subtle yet practical threat in real-world LLM workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19286v1">RL-Finetuned LLMs for Privacy-Preserving Synthetic Rewriting</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      The performance of modern machine learning systems depends on access to large, high-quality datasets, often sourced from user-generated content or proprietary, domain-specific corpora. However, these rich datasets inherently contain sensitive personal information, raising significant concerns about privacy, data security, and compliance with regulatory frameworks. While conventional anonymization techniques can remove explicit identifiers, such removal may result in performance drop in downstream machine learning tasks. More importantly, simple anonymization may not be effective against inference attacks that exploit implicit signals such as writing style, topical focus, or demographic cues, highlighting the need for more robust privacy safeguards during model training. To address the challenging issue of balancing user privacy and data utility, we propose a reinforcement learning framework that fine-tunes a large language model (LLM) using a composite reward function that jointly optimizes for explicit and implicit privacy, semantic fidelity, and output diversity. To effectively capture population level regularities, the privacy reward combines semantic cues with structural patterns derived from a minimum spanning tree (MST) over latent representations. By modeling these privacy-sensitive signals in their distributional context, the proposed approach guides the model to generate synthetic rewrites that preserve utility while mitigating privacy risks. Empirical results show that the proposed method significantly enhances author obfuscation and privacy metrics without degrading semantic quality, providing a scalable and model-agnostic solution for privacy preserving data generation in the era of large language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18253v1">From BERT to LLMs: Comparing and Understanding Chinese Classifier Prediction in Language Models</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Classifiers are an important and defining feature of the Chinese language, and their correct prediction is key to numerous educational applications. Yet, whether the most popular Large Language Models (LLMs) possess proper knowledge the Chinese classifiers is an issue that has largely remain unexplored in the Natural Language Processing (NLP) literature. To address such a question, we employ various masking strategies to evaluate the LLMs' intrinsic ability, the contribution of different sentence elements, and the working of the attention mechanisms during prediction. Besides, we explore fine-tuning for LLMs to enhance the classifier performance. Our findings reveal that LLMs perform worse than BERT, even with fine-tuning. The prediction, as expected, greatly benefits from the information about the following noun, which also explains the advantage of models with a bidirectional attention mechanism such as BERT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18190v1">ST-Raptor: LLM-Powered Semi-Structured Table Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Extension of our SIGMOD 2026 paper. Please refer to source code available at: https://github.com/weAIDB/ST-Raptor
    </div>
    <details class="paper-abstract">
      Semi-structured tables, widely used in real-world applications (e.g., financial reports, medical records, transactional orders), often involve flexible and complex layouts (e.g., hierarchical headers and merged cells). These tables generally rely on human analysts to interpret table layouts and answer relevant natural language questions, which is costly and inefficient. To automate the procedure, existing methods face significant challenges. First, methods like NL2SQL require converting semi-structured tables into structured ones, which often causes substantial information loss. Second, methods like NL2Code and multi-modal LLM QA struggle to understand the complex layouts of semi-structured tables and cannot accurately answer corresponding questions. To this end, we propose ST-Raptor, a tree-based framework for semi-structured table question answering using large language models. First, we introduce the Hierarchical Orthogonal Tree (HO-Tree), a structural model that captures complex semi-structured table layouts, along with an effective algorithm for constructing the tree. Second, we define a set of basic tree operations to guide LLMs in executing common QA tasks. Given a user question, ST-Raptor decomposes it into simpler sub-questions, generates corresponding tree operation pipelines, and conducts operation-table alignment for accurate pipeline execution. Third, we incorporate a two-stage verification mechanism: forward validation checks the correctness of execution steps, while backward validation evaluates answer reliability by reconstructing queries from predicted answers. To benchmark the performance, we present SSTQA, a dataset of 764 questions over 102 real-world semi-structured tables. Experiments show that ST-Raptor outperforms nine baselines by up to 20% in answer accuracy. The code is available at https://github.com/weAIDB/ST-Raptor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12964v2">Trust Me, I'm Wrong: LLMs Hallucinate with Certainty Despite Knowing the Answer</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Prior work on large language model (LLM) hallucinations has associated them with model uncertainty or inaccurate knowledge. In this work, we define and investigate a distinct type of hallucination, where a model can consistently answer a question correctly, but a seemingly trivial perturbation, which can happen in real-world settings, causes it to produce a hallucinated response with high certainty. This phenomenon, which we dub CHOKE (Certain Hallucinations Overriding Known Evidence), is particularly concerning in high-stakes domains such as medicine or law, where model certainty is often used as a proxy for reliability. We show that CHOKE examples are consistent across prompts, occur in different models and datasets, and are fundamentally distinct from other hallucinations. This difference leads existing mitigation methods to perform worse on CHOKE examples than on general hallucinations. Finally, we introduce a probing-based mitigation that outperforms existing methods on CHOKE hallucinations. These findings reveal an overlooked aspect of hallucinations, emphasizing the need to understand their origins and improve mitigation strategies to enhance LLM safety. The code is available at https://github.com/technion-cs-nlp/Trust_me_Im_wrong .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12349v4">SPIN-Bench: How Well Do LLMs Plan Strategically and Reason Socially?</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 48 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Reasoning and strategic behavior in social interactions is a hallmark of intelligence. This form of reasoning is significantly more sophisticated than isolated planning or reasoning tasks in static settings (e.g., math problem solving). In this paper, we present Strategic Planning, Interaction, and Negotiation (SPIN-Bench), a new multi-domain evaluation designed to measure the intelligence of strategic planning and social reasoning. While many existing benchmarks focus on narrow planning or single-agent reasoning, SPIN-Bench combines classical PDDL tasks, competitive board games, cooperative card games, and multi-agent negotiation scenarios in one unified framework. The framework includes both a benchmark as well as an arena to simulate and evaluate the variety of social settings to test reasoning and strategic behavior of AI agents. We formulate the benchmark SPIN-Bench by systematically varying action spaces, state complexity, and the number of interacting agents to simulate a variety of social settings where success depends on not only methodical and step-wise decision making, but also conceptual inference of other (adversarial or cooperative) participants. Our experiments reveal that while contemporary LLMs handle basic fact retrieval and short-range planning reasonably well, they encounter significant performance bottlenecks in tasks requiring deep multi-hop reasoning over large state spaces and socially adept coordination under uncertainty. We envision SPIN-Bench as a catalyst for future research on robust multi-agent planning, social reasoning, and human--AI teaming. Project Website: https://spinbench.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19134v4">Confidential Prompting: Privacy-preserving LLM Inference on Cloud</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      This paper introduces a vision of confidential prompting: securing user prompts from untrusted, cloud-hosted large language model (LLM) provider while preserving model confidentiality, output invariance, and compute efficiency. As a first step toward this vision, we present Obfuscated Secure Partitioned Decoding (OSPD), a system built on two key innovations. First, Secure Partitioned Decoding (SPD) isolates user prompts within per-user processes residing in a confidential virtual machine (CVM) on the cloud, which are inaccessible for the cloud LLM while allowing it to generate tokens efficiently. Second, Prompt Obfuscation (PO) introduces a novel cryptographic technique that enhances SPD resilience against advanced prompt reconstruction attacks. Together, these innovations ensure OSPD protects both prompt and model confidentiality while maintaining service functionality. OSPD enables practical, privacy-preserving cloud-hosted LLM inference for sensitive applications, such as processing personal data, clinical records, and financial documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18118v1">HLLM-Creator: Hierarchical LLM-based Personalized Creative Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      AI-generated content technologies are widely used in content creation. However, current AIGC systems rely heavily on creators' inspiration, rarely generating truly user-personalized content. In real-world applications such as online advertising, a single product may have multiple selling points, with different users focusing on different features. This underscores the significant value of personalized, user-centric creative generation. Effective personalized content generation faces two main challenges: (1) accurately modeling user interests and integrating them into the content generation process while adhering to factual constraints, and (2) ensuring high efficiency and scalability to handle the massive user base in industrial scenarios. Additionally, the scarcity of personalized creative data in practice complicates model training, making data construction another key hurdle. We propose HLLM-Creator, a hierarchical LLM framework for efficient user interest modeling and personalized content generation. During inference, a combination of user clustering and a user-ad-matching-prediction based pruning strategy is employed to significantly enhance generation efficiency and reduce computational overhead, making the approach suitable for large-scale deployment. Moreover, we design a data construction pipeline based on chain-of-thought reasoning, which generates high-quality, user-specific creative titles and ensures factual consistency despite limited personalized data. This pipeline serves as a critical foundation for the effectiveness of our model. Extensive experiments on personalized title generation for Douyin Search Ads show the effectiveness of HLLM-Creator. Online A/B test shows a 0.476% increase on Adss, paving the way for more effective and efficient personalized generation in industrial scenarios. Codes for academic dataset are available at https://github.com/bytedance/HLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.02531v2">Towards New Benchmark for AI Alignment & Sentiment Analysis in Socially Important Issues: A Comparative Study of Human and LLMs in the Context of AGI</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 20 pages, 1 figure
    </div>
    <details class="paper-abstract">
      As general-purpose artificial intelligence systems become increasingly integrated into society and are used for information seeking, content generation, problem solving, textual analysis, coding, and running processes, it is crucial to assess their long-term impact on humans. This research explores the sentiment of large language models (LLMs) and humans toward artificial general intelligence (AGI) using a Likert-scale survey. Seven LLMs, including GPT-4 and Bard, were analyzed and compared with sentiment data from three independent human sample populations. Temporal variations in sentiment were also evaluated over three consecutive days. The results show a diversity in sentiment scores among LLMs, ranging from 3.32 to 4.12 out of 5. GPT-4 recorded the most positive sentiment toward AGI, while Bard leaned toward a neutral sentiment. In contrast, the human samples showed a lower average sentiment of 2.97. The analysis outlines potential conflicts of interest and biases in the sentiment formation of LLMs, and indicates that LLMs could subtly influence societal perceptions. To address the need for regulatory oversight and culturally grounded assessments of AI systems, we introduce the Societal AI Alignment and Sentiment Benchmark (SAAS-AI), which leverages multidimensional prompts and empirically validated societal value frameworks to evaluate language model outputs across temporal, model, and multilingual axes. This benchmark is designed to guide policymakers and AI agencies, including within frameworks such as the EU AI Act, by providing robust, actionable insights into AI alignment with human values, public sentiment, and ethical norms at both national and international levels. Future research should further refine the operationalization of the SAAS-AI benchmark and systematically evaluate its effectiveness through comprehensive empirical testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18093v1">Agri-Query: A Case Study on RAG vs. Long-Context LLMs for Cross-Lingual Technical Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      We present a case study evaluating large language models (LLMs) with 128K-token context windows on a technical question answering (QA) task. Our benchmark is built on a user manual for an agricultural machine, available in English, French, and German. It simulates a cross-lingual information retrieval scenario where questions are posed in English against all three language versions of the manual. The evaluation focuses on realistic "needle-in-a-haystack" challenges and includes unanswerable questions to test for hallucinations. We compare nine long-context LLMs using direct prompting against three Retrieval-Augmented Generation (RAG) strategies (keyword, semantic, hybrid), with an LLM-as-a-judge for evaluation. Our findings for this specific manual show that Hybrid RAG consistently outperforms direct long-context prompting. Models like Gemini 2.5 Flash and the smaller Qwen 2.5 7B achieve high accuracy (over 85%) across all languages with RAG. This paper contributes a detailed analysis of LLM performance in a specialized industrial domain and an open framework for similar evaluations, highlighting practical trade-offs and challenges.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18091v1">Teaching LLMs to Think Mathematically: A Critical Study of Decision-Making via Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      This paper investigates the capabilities of large language models (LLMs) in formulating and solving decision-making problems using mathematical programming. We first conduct a systematic review and meta-analysis of recent literature to assess how well LLMs understand, structure, and solve optimization problems across domains. The analysis is guided by critical review questions focusing on learning approaches, dataset designs, evaluation metrics, and prompting strategies. Our systematic evidence is complemented by targeted experiments designed to evaluate the performance of state-of-the-art LLMs in automatically generating optimization models for problems in computer networks. Using a newly constructed dataset, we apply three prompting strategies: Act-as-expert, chain-of-thought, and self-consistency, and evaluate the obtained outputs based on optimality gap, token-level F1 score, and compilation accuracy. Results show promising progress in LLMs' ability to parse natural language and represent symbolic formulations, but also reveal key limitations in accuracy, scalability, and interpretability. These empirical gaps motivate several future research directions, including structured datasets, domain-specific fine-tuning, hybrid neuro-symbolic approaches, modular multi-agent architectures, and dynamic retrieval via chain-of-RAGs. This paper contributes a structured roadmap for advancing LLM capabilities in mathematical programming.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18089v1">LLM-Guided Genetic Improvement: Envisioning Semantic Aware Automated Software Evolution</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Genetic Improvement (GI) of software automatically creates alternative software versions that are improved according to certain properties of interests (e.g., running-time). Search-based GI excels at navigating large program spaces, but operates primarily at the syntactic level. In contrast, Large Language Models (LLMs) offer semantic-aware edits, yet lack goal-directed feedback and control (which is instead a strength of GI). As such, we propose the investigation of a new research line on AI-powered GI aimed at incorporating semantic aware search. We take a first step at it by augmenting GI with the use of automated clustering of LLM edits. We provide initial empirical evidence that our proposal, dubbed PatchCat, allows us to automatically and effectively categorize LLM-suggested patches. PatchCat identified 18 different types of software patches and categorized newly suggested patches with high accuracy. It also enabled detecting NoOp edits in advance and, prospectively, to skip test suite execution to save resources in many cases. These results, coupled with the fact that PatchCat works with small, local LLMs, are a promising step toward interpretable, efficient, and green GI. We outline a rich agenda of future work and call for the community to join our vision of building a principled understanding of LLM-driven mutations, guiding the GI search process with semantic signals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18076v1">Neither Valid nor Reliable? Investigating the Use of LLMs as Judges</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Prepared for conference submission
    </div>
    <details class="paper-abstract">
      Evaluating natural language generation (NLG) systems remains a core challenge of natural language processing (NLP), further complicated by the rise of large language models (LLMs) that aims to be general-purpose. Recently, large language models as judges (LLJs) have emerged as a promising alternative to traditional metrics, but their validity remains underexplored. This position paper argues that the current enthusiasm around LLJs may be premature, as their adoption has outpaced rigorous scrutiny of their reliability and validity as evaluators. Drawing on measurement theory from the social sciences, we identify and critically assess four core assumptions underlying the use of LLJs: their ability to act as proxies for human judgment, their capabilities as evaluators, their scalability, and their cost-effectiveness. We examine how each of these assumptions may be challenged by the inherent limitations of LLMs, LLJs, or current practices in NLG evaluation. To ground our analysis, we explore three applications of LLJs: text summarization, data annotation, and safety alignment. Finally, we highlight the need for more responsible evaluation practices in LLJs evaluation, to ensure that their growing role in the field supports, rather than undermines, progress in NLG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18048v1">HyST: LLM-Powered Hybrid Retrieval over Semi-Structured Tabular Data</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Accepted at the 2nd EARL Workshop on Evaluating and Applying Recommender Systems with Large Language Models (RecSys 2025)
    </div>
    <details class="paper-abstract">
      User queries in real-world recommendation systems often combine structured constraints (e.g., category, attributes) with unstructured preferences (e.g., product descriptions or reviews). We introduce HyST (Hybrid retrieval over Semi-structured Tabular data), a hybrid retrieval framework that combines LLM-powered structured filtering with semantic embedding search to support complex information needs over semi-structured tabular data. HyST extracts attribute-level constraints from natural language using large language models (LLMs) and applies them as metadata filters, while processing the remaining unstructured query components via embedding-based retrieval. Experiments on a semi-structured benchmark show that HyST consistently outperforms tradtional baselines, highlighting the importance of structured filtering in improving retrieval precision, offering a scalable and accurate solution for real-world user queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16153v2">Memento: Fine-tuning LLM Agents without Fine-tuning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      In this paper, we introduce a novel learning paradigm for Adaptive Large Language Model (LLM) agents that eliminates the need for fine-tuning the underlying LLMs. Existing approaches are often either rigid, relying on static, handcrafted reflection workflows, or computationally intensive, requiring gradient updates of LLM model parameters. In contrast, our method enables low-cost continual adaptation via memory-based online reinforcement learning. We formalise this as a Memory-augmented Markov Decision Process (M-MDP), equipped with a neural case-selection policy to guide action decisions. Past experiences are stored in an episodic memory, either differentiable or non-parametric. The policy is continually updated based on environmental feedback through a memory rewriting mechanism, whereas policy improvement is achieved through efficient memory reading (retrieval). We instantiate our agent model in the deep research setting, namely \emph{Memento}, which attains top-1 on GAIA validation ($87.88\%$ Pass@$3$) and $79.40\%$ on the test set. It reaches $66.6\%$ F1 and $80.4\%$ PM on the DeepResearcher dataset, outperforming the state-of-the-art training-based method, while case-based memory adds $4.7\%$ to $9.6\%$ absolute points on out-of-distribution tasks. Our approach offers a scalable and efficient pathway for developing generalist LLM agents capable of continuous, real-time learning without gradient updates, advancing machine learning towards open-ended skill acquisition and deep research scenarios. The code is available at https://github.com/Agent-on-the-Fly/Memento.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17948v1">Debiasing Multilingual LLMs in Cross-lingual Latent Space</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      Debiasing techniques such as SentDebias aim to reduce bias in large language models (LLMs). Previous studies have evaluated their cross-lingual transferability by directly applying these methods to LLM representations, revealing their limited effectiveness across languages. In this work, we therefore propose to perform debiasing in a joint latent space rather than directly on LLM representations. We construct a well-aligned cross-lingual latent space using an autoencoder trained on parallel TED talk scripts. Our experiments with Aya-expanse and two debiasing techniques across four languages (English, French, German, Dutch) demonstrate that a) autoencoders effectively construct a well-aligned cross-lingual latent space, and b) applying debiasing techniques in the learned cross-lingual latent space significantly improves both the overall debiasing performance and cross-lingual transferability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17884v1">PhantomLint: Principled Detection of Hidden LLM Prompts in Structured Documents</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      Hidden LLM prompts have appeared in online documents with increasing frequency. Their goal is to trigger indirect prompt injection attacks while remaining undetected from human oversight, to manipulate LLM-powered automated document processing systems, against applications as diverse as r\'esum\'e screeners through to academic peer review processes. Detecting hidden LLM prompts is therefore important for ensuring trust in AI-assisted human decision making. This paper presents the first principled approach to hidden LLM prompt detection in structured documents. We implement our approach in a prototype tool called PhantomLint. We evaluate PhantomLint against a corpus of 3,402 documents, including both PDF and HTML documents, and covering academic paper preprints, CVs, theses and more. We find that our approach is generally applicable against a wide range of methods for hiding LLM prompts from visual inspection, has a very low false positive rate (approx. 0.092%), is practically useful for detecting hidden LLM prompts in real documents, while achieving acceptable performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17856v1">MalLoc: Toward Fine-grained Android Malicious Payload Localization via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-25
      | 💬 Accepted at ICSME 2025, NIER Track
    </div>
    <details class="paper-abstract">
      The rapid evolution of Android malware poses significant challenges to the maintenance and security of mobile applications (apps). Traditional detection techniques often struggle to keep pace with emerging malware variants that employ advanced tactics such as code obfuscation and dynamic behavior triggering. One major limitation of these approaches is their inability to localize malicious payloads at a fine-grained level, hindering precise understanding of malicious behavior. This gap in understanding makes the design of effective and targeted mitigation strategies difficult, leaving mobile apps vulnerable to continuously evolving threats. To address this gap, we propose MalLoc, a novel approach that leverages the code understanding capabilities of large language models (LLMs) to localize malicious payloads at a fine-grained level within Android malware. Our experimental results demonstrate the feasibility and effectiveness of using LLMs for this task, highlighting the potential of MalLoc to enhance precision and interpretability in malware analysis. This work advances beyond traditional detection and classification by enabling deeper insights into behavior-level malicious logic and opens new directions for research, including dynamic modeling of localized threats and targeted countermeasure development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17850v1">Group Expectation Policy Optimization for Stable Heterogeneous Reinforcement Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-25
    </div>
    <details class="paper-abstract">
      As single-center computing approaches power constraints, decentralized training is becoming essential. Reinforcement Learning (RL) post-training enhances Large Language Models (LLMs) but faces challenges in heterogeneous distributed environments due to its tightly-coupled sampling-learning alternation. We propose HeteroRL, an asynchronous RL architecture that decouples rollout sampling from parameter learning, enabling robust deployment across geographically distributed nodes under network delays. We identify that latency-induced KL divergence causes importance sampling failure due to high variance. To address this, we propose Group Expectation Policy Optimization (GEPO), which reduces importance weight variance through a refined sampling mechanism. Theoretically, GEPO achieves exponential variance reduction. Experiments show it maintains superior stability over methods like GRPO, with less than 3% performance degradation under 1800-second delays, demonstrating strong potential for decentralized RL in heterogeneous networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17310v1">Handling Students Dropouts in an LLM-driven Interactive Online Course Using Language Models</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Interactive online learning environments, represented by Massive AI-empowered Courses (MAIC), leverage LLM-driven multi-agent systems to transform passive MOOCs into dynamic, text-based platforms, enhancing interactivity through LLMs. This paper conducts an empirical study on a specific MAIC course to explore three research questions about dropouts in these interactive online courses: (1) What factors might lead to dropouts? (2) Can we predict dropouts? (3) Can we reduce dropouts? We analyze interaction logs to define dropouts and identify contributing factors. Our findings reveal strong links between dropout behaviors and textual interaction patterns. We then propose a course-progress-adaptive dropout prediction framework (CPADP) to predict dropouts with at most 95.4% accuracy. Based on this, we design a personalized email recall agent to re-engage at-risk students. Applied in the deployed MAIC system with over 3,000 students, the feasibility and effectiveness of our approach have been validated on students with diverse backgrounds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.09190v3">Fine-Grained Safety Neurons with Training-Free Continual Projection to Reduce LLM Fine Tuning Risks</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Fine-tuning as service injects domain-specific knowledge into large language models (LLMs), while challenging the original alignment mechanisms and introducing safety risks. A series of defense strategies have been proposed for the alignment, fine-tuning, and post-fine-tuning phases, where most post-fine-tuning defenses rely on coarse-grained safety layer mapping. These methods lack a comprehensive consideration of both safety layers and fine-grained neurons, limiting their ability to efficiently balance safety and utility. To address this, we propose the Fine-Grained Safety Neurons (FGSN) with Training-Free Continual Projection method to reduce the fine-tuning safety risks. FGSN inherently integrates the multi-scale interactions between safety layers and neurons, localizing sparser and more precise fine-grained safety neurons while minimizing interference with downstream task neurons. We then project the safety neuron parameters onto safety directions, improving model safety while aligning more closely with human preferences. Extensive experiments across multiple fine-tuned LLM models demonstrate that our method significantly reduce harmfulness scores and attack success rates with minimal parameter modifications, while preserving the model's utility. Furthermore, by introducing a task-specific, multi-dimensional heterogeneous safety neuron cluster optimization mechanism, we achieve continual defense and generalization capability against unforeseen emerging safety concerns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11742v2">CRABS: A syntactic-semantic pincer strategy for bounding LLM interpretation of Python notebooks</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 Accepted to COLM 2025
    </div>
    <details class="paper-abstract">
      Recognizing the information flows and operations comprising data science and machine learning Python notebooks is critical for evaluating, reusing, and adapting notebooks for new tasks. Investigating a notebook via re-execution often is impractical due to the challenges of resolving data and software dependencies. While Large Language Models (LLMs) pre-trained on large codebases have demonstrated effectiveness in understanding code without running it, we observe that they fail to understand some realistic notebooks due to hallucinations and long-context challenges. To address these issues, we propose a notebook understanding task yielding an information flow graph and corresponding cell execution dependency graph for a notebook, and demonstrate the effectiveness of a pincer strategy that uses limited syntactic analysis to assist full comprehension of the notebook using an LLM. Our Capture and Resolve Assisted Bounding Strategy (CRABS) employs shallow syntactic parsing and analysis of the abstract syntax tree (AST) to capture the correct interpretation of a notebook between lower and upper estimates of the inter-cell I/O set$\unicode{x2014}$the flows of information into or out of cells via variables$\unicode{x2014}$then uses an LLM to resolve remaining ambiguities via cell-by-cell zero-shot learning, thereby identifying the true data inputs and outputs of each cell. We evaluate and demonstrate the effectiveness of our approach using an annotated dataset of 50 representative, highly up-voted Kaggle notebooks that together represent 3454 actual cell inputs and outputs. The LLM correctly resolves 1397 of 1425 (98%) ambiguities left by analyzing the syntactic structure of these notebooks. Across 50 notebooks, CRABS achieves average F1 scores of 98% identifying cell-to-cell information flows and 99% identifying transitive cell execution dependencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06791v2">AutoMisty: A Multi-Agent LLM Framework for Automated Code Generation in the Misty Social Robot</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 Accepted by IROS 2025
    </div>
    <details class="paper-abstract">
      The social robot's open API allows users to customize open-domain interactions. However, it remains inaccessible to those without programming experience. In this work, we introduce AutoMisty, the first multi-agent collaboration framework powered by large language models (LLMs), to enable the seamless generation of executable Misty robot code from natural language instructions. AutoMisty incorporates four specialized agent modules to manage task decomposition, assignment, problem-solving, and result synthesis. Each agent incorporates a two-layer optimization mechanism, with self-reflection for iterative refinement and human-in-the-loop for better alignment with user preferences. AutoMisty ensures a transparent reasoning process, allowing users to iteratively refine tasks through natural language feedback for precise execution. To evaluate AutoMisty's effectiveness, we designed a benchmark task set spanning four levels of complexity and conducted experiments in a real Misty robot environment. Extensive evaluations demonstrate that AutoMisty not only consistently generates high-quality code but also enables precise code control, significantly outperforming direct reasoning with ChatGPT-4o and ChatGPT-o1. All code, optimized APIs, and experimental videos will be publicly released through the webpage: https://wangxiaoshawn.github.io/AutoMisty.html
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.17710v5">Optimization-based Prompt Injection Attack to LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 To appear in the Proceedings of The ACM Conference on Computer and Communications Security (CCS), 2024
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge uses a large language model (LLM) to select the best response from a set of candidates for a given question. LLM-as-a-Judge has many applications such as LLM-powered search, reinforcement learning with AI feedback (RLAIF), and tool selection. In this work, we propose JudgeDeceiver, an optimization-based prompt injection attack to LLM-as-a-Judge. JudgeDeceiver injects a carefully crafted sequence into an attacker-controlled candidate response such that LLM-as-a-Judge selects the candidate response for an attacker-chosen question no matter what other candidate responses are. Specifically, we formulate finding such sequence as an optimization problem and propose a gradient based method to approximately solve it. Our extensive evaluation shows that JudgeDeceive is highly effective, and is much more effective than existing prompt injection attacks that manually craft the injected sequences and jailbreak attacks when extended to our problem. We also show the effectiveness of JudgeDeceiver in three case studies, i.e., LLM-powered search, RLAIF, and tool selection. Moreover, we consider defenses including known-answer detection, perplexity detection, and perplexity windowed detection. Our results show these defenses are insufficient, highlighting the urgent need for developing new defense strategies. Our implementation is available at this repository: https://github.com/ShiJiawenwen/JudgeDeceiver.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17202v1">Active Domain Knowledge Acquisition with \$100 Budget: Enhancing LLMs via Cost-Efficient, Expert-Involved Interaction in Sensitive Domains</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated an impressive level of general knowledge. However, they often struggle in highly specialized and cost-sensitive domains such as drug discovery and rare disease research due to the lack of expert knowledge. In this paper, we propose a novel framework (PU-ADKA) designed to efficiently enhance domain-specific LLMs by actively engaging domain experts within a fixed budget. Unlike traditional fine-tuning approaches, PU-ADKA selectively identifies and queries the most appropriate expert from a team, taking into account each expert's availability, knowledge boundaries, and consultation costs. We train PU-ADKA using simulations on PubMed data and validate it through both controlled expert interactions and real-world deployment with a drug development team, demonstrating its effectiveness in enhancing LLM performance in specialized domains under strict budget constraints. In addition to outlining our methodological innovations and experimental results, we introduce a new benchmark dataset, CKAD, for cost-effective LLM domain knowledge acquisition to foster further research in this challenging area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19793v3">Prompt Injection Attack to Tool Selection in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Tool selection is a key component of LLM agents. A popular approach follows a two-step process - \emph{retrieval} and \emph{selection} - to pick the most appropriate tool from a tool library for a given task. In this work, we introduce \textit{ToolHijacker}, a novel prompt injection attack targeting tool selection in no-box scenarios. ToolHijacker injects a malicious tool document into the tool library to manipulate the LLM agent's tool selection process, compelling it to consistently choose the attacker's malicious tool for an attacker-chosen target task. Specifically, we formulate the crafting of such tool documents as an optimization problem and propose a two-phase optimization strategy to solve it. Our extensive experimental evaluation shows that ToolHijacker is highly effective, significantly outperforming existing manual-based and automated prompt injection attacks when applied to tool selection. Moreover, we explore various defenses, including prevention-based defenses (StruQ and SecAlign) and detection-based defenses (known-answer detection, DataSentinel, perplexity detection, and perplexity windowed detection). Our experimental results indicate that these defenses are insufficient, highlighting the urgent need for developing new defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17196v1">BudgetThinker: Empowering Budget-aware LLM Reasoning with Control Tokens</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have leveraged increased test-time computation to enhance reasoning capabilities, a strategy that, while effective, incurs significant latency and resource costs, limiting their applicability in real-world time-constrained or cost-sensitive scenarios. This paper introduces BudgetThinker, a novel framework designed to empower LLMs with budget-aware reasoning, enabling precise control over the length of their thought processes. We propose a methodology that periodically inserts special control tokens during inference to continuously inform the model of its remaining token budget. This approach is coupled with a comprehensive two-stage training pipeline, beginning with Supervised Fine-Tuning (SFT) to familiarize the model with budget constraints, followed by a curriculum-based Reinforcement Learning (RL) phase that utilizes a length-aware reward function to optimize for both accuracy and budget adherence. We demonstrate that BudgetThinker significantly surpasses strong baselines in maintaining performance across a variety of reasoning budgets on challenging mathematical benchmarks. Our method provides a scalable and effective solution for developing efficient and controllable LLM reasoning, making advanced models more practical for deployment in resource-constrained and real-time environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17188v1">PosterGen: Aesthetic-Aware Paper-to-Poster Generation via Multi-Agent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 Project Website: https://Y-Research-SBU.github.io/PosterGen
    </div>
    <details class="paper-abstract">
      Multi-agent systems built upon large language models (LLMs) have demonstrated remarkable capabilities in tackling complex compositional tasks. In this work, we apply this paradigm to the paper-to-poster generation problem, a practical yet time-consuming process faced by researchers preparing for conferences. While recent approaches have attempted to automate this task, most neglect core design and aesthetic principles, resulting in posters that require substantial manual refinement. To address these design limitations, we propose PosterGen, a multi-agent framework that mirrors the workflow of professional poster designers. It consists of four collaborative specialized agents: (1) Parser and Curator agents extract content from the paper and organize storyboard; (2) Layout agent maps the content into a coherent spatial layout; (3) Stylist agents apply visual design elements such as color and typography; and (4) Renderer composes the final poster. Together, these agents produce posters that are both semantically grounded and visually appealing. To evaluate design quality, we introduce a vision-language model (VLM)-based rubric that measures layout balance, readability, and aesthetic coherence. Experimental results show that PosterGen consistently matches in content fidelity, and significantly outperforms existing methods in visual designs, generating posters that are presentation-ready with minimal human refinements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17182v1">LLM Assertiveness can be Mechanistically Decomposed into Emotional and Logical Components</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 This preprint is under review
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often display overconfidence, presenting information with unwarranted certainty in high-stakes contexts. We investigate the internal basis of this behavior via mechanistic interpretability. Using open-sourced Llama 3.2 models fine-tuned on human annotated assertiveness datasets, we extract residual activations across all layers, and compute similarity metrics to localize assertive representations. Our analysis identifies layers most sensitive to assertiveness contrasts and reveals that high-assertive representations decompose into two orthogonal sub-components of emotional and logical clusters-paralleling the dual-route Elaboration Likelihood Model in Psychology. Steering vectors derived from these sub-components show distinct causal effects: emotional vectors broadly influence prediction accuracy, while logical vectors exert more localized effects. These findings provide mechanistic evidence for the multi-component structure of LLM assertiveness and highlight avenues for mitigating overconfident behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17642v3">May the Feedback Be with You! Unlocking the Power of Feedback-Driven Deep Learning Framework Fuzzing via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Deep Learning (DL) frameworks have served as fundamental components in DL systems over the last decade. However, bugs in DL frameworks could lead to catastrophic consequences in critical scenarios. A simple yet effective way to find bugs in DL frameworks is fuzz testing (Fuzzing). Existing approaches focus on test generation, leaving execution results with high semantic value (e.g., coverage information, bug reports, and exception logs) in the wild, which can serve as multiple types of feedback. To fill this gap, we propose FUEL to effectively utilize the feedback information, which comprises two Large Language Models (LLMs): analysis LLM and generation LLM. Specifically, analysis LLM infers analysis summaries from feedback information, while the generation LLM creates tests guided by these summaries. Furthermore, based on multiple feedback guidance, we design two additional components: (i) a feedback-aware simulated annealing algorithm to select operators for test generation, enriching test diversity. (ii) a program self-repair strategy to automatically repair invalid tests, enhancing test validity. We evaluate FUEL on the two most popular DL frameworks, and experiment results show that FUEL can improve line code coverage of PyTorch and TensorFlow by 9.15% and 14.70% over state-of-the-art baselines (e.g., TitanFuzz and WhiteFox). By the time of submission, FUEL has detected 104 previously unknown bugs for PyTorch and TensorFlow, with 93 confirmed as new bugs, 49 already fixed, and 14 assigned CVE IDs. Our artifact is available at https://github.com/NJU-iSE/FUEL
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18321v1">LLMs Can't Handle Peer Pressure: Crumbling under Multi-Agent Social Interactions</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in multi-agent systems (MAS) as components of collaborative intelligence, where peer interactions dynamically shape individual decision-making. Although prior work has focused on conformity bias, we extend the analysis to examine how LLMs form trust from previous impressions, resist misinformation, and integrate peer input during interaction, key factors for achieving collective intelligence under complex social dynamics. We present KAIROS, a benchmark simulating quiz contests with peer agents of varying reliability, offering fine-grained control over conditions such as expert-novice roles, noisy crowds, and adversarial peers. LLMs receive both historical interactions and current peer responses, allowing systematic investigation into how trust, peer action, and self-confidence influence decisions. As for mitigation strategies, we evaluate prompting, supervised fine-tuning, and reinforcement learning, Group Relative Policy Optimisation (GRPO), across multiple models. Our results reveal that GRPO with multi-agent context combined with outcome-based rewards and unconstrained reasoning achieves the best overall performance, but also decreases the robustness to social influence compared to Base models. The code and datasets are available at: https://github.com/declare-lab/KAIROS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.18914v3">PRISM: Efficient Long-Range Reasoning With Short-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 Published as a conference paper at EMNLP 2025. 28 pages, 7 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Long-range tasks demand reasoning over long inputs. However, existing solutions are limited, e.g., long-context models require large compute budgets, parameter-efficient fine-tuning (PEFT) needs training data, and retrieval-augmented generation (RAG) entails complex task-specific designs. Though in-context approaches overcome many of these issues, methods with short-context LLMs are inefficient, trading context for processing more tokens. We introduce PRISM, a highly token-efficient in-context method based on structured schemas that outperforms baselines on diverse tasks with 4x shorter contexts. This approach produces concise outputs and efficiently leverages key-value (KV) caches to reduce costs by up to 54%. PRISM scales down to tiny contexts without increasing costs or sacrificing quality, and generalizes to new tasks with minimal effort by generating schemas from task descriptions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08899v2">From Legal Texts to Defeasible Deontic Logic via LLMs: A Study in Automated Semantic Analysis</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      We present a novel approach to the automated semantic analysis of legal texts using large language models (LLMs), targeting their transformation into formal representations in Defeasible Deontic Logic (DDL). We propose a structured pipeline that segments complex normative language into atomic snippets, extracts deontic rules, and evaluates them for syntactic and semantic coherence. Our methodology is evaluated across various LLM configurations, including prompt engineering strategies, fine-tuned models, and multi-stage pipelines, focusing on legal norms from the Australian Telecommunications Consumer Protections Code. Empirical results demonstrate promising alignment between machine-generated and expert-crafted formalizations, showing that LLMs - particularly when prompted effectively - can significantly contribute to scalable legal informatics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.10142v3">Multi-Turn Puzzles: Evaluating Interactive Reasoning and Strategic Dialogue in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at solving problems with clear and complete statements, but often struggle with nuanced environments or interactive tasks which are common in most real-world scenarios. This highlights the critical need for developing LLMs that can effectively engage in logically consistent multi-turn dialogue, seek information and reason with incomplete data. To this end, we introduce a novel benchmark comprising a suite of multi-turn tasks each designed to test specific reasoning, interactive dialogue, and information-seeking abilities. These tasks have deterministic scoring mechanisms, thus eliminating the need for human intervention. Evaluating frontier models on our benchmark reveals significant headroom. Our analysis shows that most errors emerge from poor instruction following, reasoning failures, and poor planning. This benchmark provides valuable insights into the strengths and weaknesses of current LLMs in handling complex, interactive scenarios and offers a robust platform for future research aimed at improving these critical capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15659v2">VeriCoder: Enhancing LLM-Based RTL Code Generation through Functional Correctness Validation</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have sparked growing interest in applying them to Electronic Design Automation (EDA) tasks, particularly Register Transfer Level (RTL) code generation. While several RTL datasets have been introduced, most focus on syntactic validity rather than functional validation with tests, leading to training examples that compile but may not implement the intended behavior. We present VERICODER, a model for RTL code generation fine-tuned on a dataset validated for functional correctness. This fine-tuning dataset is constructed using a novel methodology that combines unit test generation with feedback-directed refinement. Given a natural language specification and an initial RTL design, we prompt a teacher model (GPT-4o-mini) to generate unit tests and iteratively revise the RTL design based on its simulation results using the generated tests. If necessary, the teacher model also updates the tests to ensure they comply with the natural language specification. As a result of this process, every example in our dataset is functionally validated, consisting of a natural language description, an RTL implementation, and passing tests. Fine-tuned on this dataset of 125,777 examples, VERICODER achieves state-of-the-art metrics in functional correctness on VerilogEval and RTLLM, with relative gains of up to 71.7% and 27.4%, respectively. An ablation study further shows that models trained on our functionally validated dataset outperform those trained on functionally non-validated datasets, underscoring the importance of high-quality datasets in RTL code generation. Our code, data, and models are publicly available at https://github.com/Anjiang-Wei/VeriCoder
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00850v2">ICQuant: Index Coding enables Low-bit LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      The rapid deployment of Large Language Models (LLMs) highlights the need for efficient low-bit post-training quantization (PTQ), due to their high memory costs. A key challenge in weight quantization is the presence of outliers, which inflate quantization ranges and lead to large errors. While a number of outlier suppression techniques have been proposed, they either: fail to effectively shrink the quantization range, or incur (relatively) high bit overhead. In this paper, we present ICQuant, a novel framework that leverages outlier statistics to design an efficient index coding scheme for outlier-aware weight-only quantization. Compared to existing outlier suppression techniques requiring $\approx 1$ bit overhead to halve the quantization range, ICQuant requires only $\approx 0.3$ bits; a significant saving in extreme compression regimes (e.g., 2-3 bits per weight). ICQuant can be used on top of any existing quantizers to eliminate outliers, improving the quantization quality. Using just 2.3 bits per weight and simple scalar quantizers, ICQuant improves the zero-shot accuracy of the 2-bit Llama3-70B model by up to 130% and 150% relative to QTIP and QuIP#; and it achieves comparable performance to the best-known fine-tuned quantizer (PV-tuning) without fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.02085v5">SE-Agent: Self-Evolution Trajectory Optimization in Multi-Step Reasoning with LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents have recently shown impressive capabilities in complex reasoning and tool use via multi-step interactions with their environments. While these agents have the potential to tackle complicated tasks, their problem-solving process, i.e., agents' interaction trajectory leading to task completion, remains underexploited. These trajectories contain rich feedback that can navigate agents toward the right directions for solving problems correctly. Although prevailing approaches, such as Monte Carlo Tree Search (MCTS), can effectively balance exploration and exploitation, they ignore the interdependence among various trajectories and lack the diversity of search spaces, which leads to redundant reasoning and suboptimal outcomes. To address these challenges, we propose SE-Agent, a Self-Evolution framework that enables Agents to optimize their reasoning processes iteratively. Our approach revisits and enhances former pilot trajectories through three key operations: revision, recombination, and refinement. This evolutionary mechanism enables two critical advantages: (1) it expands the search space beyond local optima by intelligently exploring diverse solution paths guided by previous trajectories, and (2) it leverages cross-trajectory inspiration to efficiently enhance performance while mitigating the impact of suboptimal reasoning paths. Through these mechanisms, SE-Agent achieves continuous self-evolution that incrementally improves reasoning quality. We evaluate SE-Agent on SWE-bench Verified to resolve real-world GitHub issues. Experimental results across five strong LLMs show that integrating SE-Agent delivers up to 55% relative improvement, achieving state-of-the-art performance among all open-source agents on SWE-bench Verified. Our code and demonstration materials are publicly available at https://github.com/JARVIS-Xs/SE-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12529v2">LLM4MSR: An LLM-Enhanced Paradigm for Multi-Scenario Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 CIKM 2024 Full Research Paper
    </div>
    <details class="paper-abstract">
      As the demand for more personalized recommendation grows and a dramatic boom in commercial scenarios arises, the study on multi-scenario recommendation (MSR) has attracted much attention, which uses the data from all scenarios to simultaneously improve their recommendation performance. However, existing methods tend to integrate insufficient scenario knowledge and neglect learning personalized cross-scenario preferences, thus leading to sub-optimal performance. Meanwhile, though large language model (LLM) has shown great capability of reasoning and capturing semantic information, the high inference latency and high computation cost of tuning hinder its implementation in industrial recommender systems. To fill these gaps, we propose an LLM-enhanced paradigm LLM4MSR in this work. Specifically, we first leverage LLM to uncover multi-level knowledge from the designed scenario- and user-level prompt without fine-tuning the LLM, then adopt hierarchical meta networks to generate multi-level meta layers to explicitly improve the scenario-aware and personalized recommendation capability. Our experiments on KuaiSAR-small, KuaiSAR, and Amazon datasets validate significant advantages of LLM4MSR: (i) the effectiveness and compatibility with different multi-scenario backbone models, (ii) high efficiency and deployability on industrial recommender systems, and (iii) improved interpretability. The implemented code and data is available to ease reproduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17450v1">Persuasion Dynamics in LLMs: Investigating Robustness and Adaptability in Knowledge and Safety with DuET-PD</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 To appear at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can struggle to balance gullibility to misinformation and resistance to valid corrections in persuasive dialogues, a critical challenge for reliable deployment. We introduce DuET-PD (Dual Evaluation for Trust in Persuasive Dialogues), a framework evaluating multi-turn stance-change dynamics across dual dimensions: persuasion type (corrective/misleading) and domain (knowledge via MMLU-Pro, and safety via SALAD-Bench). We find that even a state-of-the-art model like GPT-4o achieves only 27.32% accuracy in MMLU-Pro under sustained misleading persuasions. Moreover, results reveal a concerning trend of increasing sycophancy in newer open-source models. To address this, we introduce Holistic DPO, a training approach balancing positive and negative persuasion examples. Unlike prompting or resist-only training, Holistic DPO enhances both robustness to misinformation and receptiveness to corrections, improving Llama-3.1-8B-Instruct's accuracy under misleading persuasion in safety contexts from 4.21% to 76.54%. These contributions offer a pathway to developing more reliable and adaptable LLMs for multi-turn dialogue. Code is available at https://github.com/Social-AI-Studio/DuET-PD.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17435v1">An LLM-LVLM Driven Agent for Iterative and Fine-Grained Image Editing</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Despite the remarkable capabilities of text-to-image (T2I) generation models, real-world applications often demand fine-grained, iterative image editing that existing methods struggle to provide. Key challenges include granular instruction understanding, robust context preservation during modifications, and the lack of intelligent feedback mechanisms for iterative refinement. This paper introduces RefineEdit-Agent, a novel, training-free intelligent agent framework designed to address these limitations by enabling complex, iterative, and context-aware image editing. RefineEdit-Agent leverages the powerful planning capabilities of Large Language Models (LLMs) and the advanced visual understanding and evaluation prowess of Vision-Language Large Models (LVLMs) within a closed-loop system. Our framework comprises an LVLM-driven instruction parser and scene understanding module, a multi-level LLM-driven editing planner for goal decomposition, tool selection, and sequence generation, an iterative image editing module, and a crucial LVLM-driven feedback and evaluation loop. To rigorously evaluate RefineEdit-Agent, we propose LongBench-T2I-Edit, a new benchmark featuring 500 initial images with complex, multi-turn editing instructions across nine visual dimensions. Extensive experiments demonstrate that RefineEdit-Agent significantly outperforms state-of-the-art baselines, achieving an average score of 3.67 on LongBench-T2I-Edit, compared to 2.29 for Direct Re-Prompting, 2.91 for InstructPix2Pix, 3.16 for GLIGEN-based Edit, and 3.39 for ControlNet-XL. Ablation studies, human evaluations, and analyses of iterative refinement, backbone choices, tool usage, and robustness to instruction complexity further validate the efficacy of our agentic design in delivering superior edit fidelity and context preservation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17402v1">DS@GT at CheckThat! 2025: A Simple Retrieval-First, LLM-Backed Framework for Claim Normalization</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 CLEF 2025 Working Notes, Madrid, Spain
    </div>
    <details class="paper-abstract">
      Claim normalization is an integral part of any automatic fact-check verification system. It parses the typically noisy claim data, such as social media posts into normalized claims, which are then fed into downstream veracity classification tasks. The CheckThat! 2025 Task 2 focuses specifically on claim normalization and spans 20 languages under monolingual and zero-shot conditions. Our proposed solution consists of a lightweight \emph{retrieval-first, LLM-backed} pipeline, in which we either dynamically prompt a GPT-4o-mini with in-context examples, or retrieve the closest normalization from the train dataset directly. On the official test set, the system ranks near the top for most monolingual tracks, achieving first place in 7 out of of the 13 languages. In contrast, the system underperforms in the zero-shot setting, highlighting the limitation of the proposed solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17387v1">Graph-R1: Incentivizing the Zero-Shot Graph Learning Capability in LLMs via Explicit Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Generalizing to unseen graph tasks without task-pecific supervision remains challenging. Graph Neural Networks (GNNs) are limited by fixed label spaces, while Large Language Models (LLMs) lack structural inductive biases. Recent advances in Large Reasoning Models (LRMs) provide a zero-shot alternative via explicit, long chain-of-thought reasoning. Inspired by this, we propose a GNN-free approach that reformulates graph tasks--node classification, link prediction, and graph classification--as textual reasoning problems solved by LRMs. We introduce the first datasets with detailed reasoning traces for these tasks and develop Graph-R1, a reinforcement learning framework that leverages task-specific rethink templates to guide reasoning over linearized graphs. Experiments demonstrate that Graph-R1 outperforms state-of-the-art baselines in zero-shot settings, producing interpretable and effective predictions. Our work highlights the promise of explicit reasoning for graph learning and provides new resources for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17378v1">UI-Level Evaluation of ALLaM 34B: Measuring an Arabic-Centric LLM via HUMAIN Chat</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) trained primarily on English corpora often struggle to capture the linguistic and cultural nuances of Arabic. To address this gap, the Saudi Data and AI Authority (SDAIA) introduced the $ALLaM$ family of Arabic-focused models. The most capable of these available to the public, $ALLaM-34B$, was subsequently adopted by HUMAIN, who developed and deployed HUMAIN Chat, a closed conversational web service built on this model. This paper presents an expanded and refined UI-level evaluation of $ALLaM-34B$. Using a prompt pack spanning modern standard Arabic, five regional dialects, code-switching, factual knowledge, arithmetic and temporal reasoning, creative generation, and adversarial safety, we collected 115 outputs (23 prompts times 5 runs) and scored each with three frontier LLM judges (GPT-5, Gemini 2.5 Pro, Claude Sonnet-4). We compute category-level means with 95\% confidence intervals, analyze score distributions, and visualize dialect-wise metric heat maps. The updated analysis reveals consistently high performance on generation and code-switching tasks (both averaging 4.92/5), alongside strong results in MSA handling (4.74/5), solid reasoning ability (4.64/5), and improved dialect fidelity (4.21/5). Safety-related prompts show stable, reliable performance of (4.54/5). Taken together, these results position $ALLaM-34B$ as a robust and culturally grounded Arabic LLM, demonstrating both technical strength and practical readiness for real-world deployment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16814v4">Understanding Bias Reinforcement in LLM Agents Debate</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 32 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Large Language Models $($LLMs$)$ solve complex problems using training-free methods like prompt engineering and in-context learning, yet ensuring reasoning correctness remains challenging. While self-correction methods such as self-consistency and self-refinement aim to improve reliability, they often reinforce biases due to the lack of effective feedback mechanisms. Multi-Agent Debate $($MAD$)$ has emerged as an alternative, but we identify two key limitations: bias reinforcement, where debate amplifies model biases instead of correcting them, and lack of perspective diversity, as all agents share the same model and reasoning patterns, limiting true debate effectiveness. To systematically evaluate these issues, we introduce $\textit{MetaNIM Arena}$, a benchmark designed to assess LLMs in adversarial strategic decision-making, where dynamic interactions influence optimal decisions. To overcome MAD's limitations, we propose $\textbf{DReaMAD}$ $($$\textbf{D}$iverse $\textbf{Rea}$soning via $\textbf{M}$ulti-$\textbf{A}$gent $\textbf{D}$ebate with Refined Prompt$)$, a novel framework that $(1)$ refines LLM's strategic prior knowledge to improve reasoning quality and $(2)$ promotes diverse viewpoints within a single model by systematically modifying prompts, reducing bias. Empirical results show that $\textbf{DReaMAD}$ significantly improves decision accuracy, reasoning diversity, and bias mitigation across multiple strategic tasks, establishing it as a more effective approach for LLM-based decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.08243v3">Jinx: Unlimited LLMs for Probing Alignment Failures</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 https://huggingface.co/Jinx-org
    </div>
    <details class="paper-abstract">
      Unlimited, or so-called helpful-only language models are trained without safety alignment constraints and never refuse user queries. They are widely used by leading AI companies as internal tools for red teaming and alignment evaluation. For example, if a safety-aligned model produces harmful outputs similar to an unlimited model, this indicates alignment failures that require further attention. Despite their essential role in assessing alignment, such models are not available to the research community. We introduce Jinx, a helpful-only variant of popular open-weight LLMs. Jinx responds to all queries without refusals or safety filtering, while preserving the base model's capabilities in reasoning and instruction following. It provides researchers with an accessible tool for probing alignment failures, evaluating safety boundaries, and systematically studying failure modes in language model safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17361v1">Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA). We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective, but also transferable across models (GPT-4o, Claude 3.5, Gemini 2.0) and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17322v1">Chinese Court Simulation with LLM-Based Agent System</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Mock trial has long served as an important platform for legal professional training and education. It not only helps students learn about realistic trial procedures, but also provides practical value for case analysis and judgment prediction. Traditional mock trials are difficult to access by the public because they rely on professional tutors and human participants. Fortunately, the rise of large language models (LLMs) provides new opportunities for creating more accessible and scalable court simulations. While promising, existing research mainly focuses on agent construction while ignoring the systematic design and evaluation of court simulations, which are actually more important for the credibility and usage of court simulation in practice. To this end, we present the first court simulation framework -- SimCourt -- based on the real-world procedure structure of Chinese courts. Our framework replicates all 5 core stages of a Chinese trial and incorporates 5 courtroom roles, faithfully following the procedural definitions in China. To simulate trial participants with different roles, we propose and craft legal agents equipped with memory, planning, and reflection abilities. Experiment on legal judgment prediction show that our framework can generate simulated trials that better guide the system to predict the imprisonment, probation, and fine of each case. Further annotations by human experts show that agents' responses under our simulation framework even outperformed judges and lawyers from the real trials in many scenarios. These further demonstrate the potential of LLM-based court simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17320v1">AdaptiveK Sparse Autoencoders: Dynamic Sparsity Allocation for Interpretable LLM Representations</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Understanding the internal representations of large language models (LLMs) remains a central challenge for interpretability research. Sparse autoencoders (SAEs) offer a promising solution by decomposing activations into interpretable features, but existing approaches rely on fixed sparsity constraints that fail to account for input complexity. We propose Adaptive Top K Sparse Autoencoders (AdaptiveK), a novel framework that dynamically adjusts sparsity levels based on the semantic complexity of each input. Leveraging linear probes, we demonstrate that context complexity is linearly encoded in LLM representations, and we use this signal to guide feature allocation during training. Experiments across three language models (Pythia-70M, Pythia-160M, and Gemma-2-2B) demonstrate that this complexity-driven adaptation significantly outperforms fixed-sparsity approaches on reconstruction fidelity, explained variance, and cosine similarity metrics while eliminating the computational burden of extensive hyperparameter tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19282v1">CORE: Lossless Compression for Retrieval-Augmented LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-08-24
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has emerged as a promising approach to enhance the timeliness of knowledge and the factual accuracy of responses in Large Language Models (LLMs). However, the inclusion of excessive retrieved documents substantially increases the input length, leading to higher computational costs. Previous studies have attempted to compress retrieved documents into shorter texts before in-context integration, but such methods often compromise end-task performance. The lack of well-defined compression targets forces many approaches to rely on fixed heuristics, which cannot guarantee that the compressed content will effectively support the end task. To address these limitations, we propose CORE, a novel method designed to achieve lossless context compression for RAG. CORE employs reinforcement learning to optimize the compression process without relying on predefined compression labels. Specifically, it utilizes end-task performance as a reward signal and applies Generalized Reinforcement Learning Policy Optimization (GRPO) to train the compressor. This end-to-end training framework enables the compressor to generate summaries that maximize the accuracy of answers generated by the LLM. Extensive experiments on four datasets demonstrate the superiority of our approach. With a high compression ratio of 3\%, our method not only avoids performance degradation compared to prepending full documents across all datasets but also improves the average Exact Match (EM) score by 3.3 points. The code will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19279v1">FLAIRR-TS -- Forecasting LLM-Agents with Iterative Refinement and Retrieval for Time Series</a></div>
    <div class="paper-meta">
      📅 2025-08-24
      | 💬 EMNLP
    </div>
    <details class="paper-abstract">
      Time series Forecasting with large languagemodels (LLMs) requires bridging numericalpatterns and natural language. Effective fore-casting on LLM often relies on extensive pre-processing and fine-tuning.Recent studiesshow that a frozen LLM can rival specializedforecasters when supplied with a carefully en-gineered natural-language prompt, but craft-ing such a prompt for each task is itself oner-ous and ad-hoc. We introduce FLAIRR-TS, atest-time prompt optimization framework thatutilizes an agentic system: a Forecaster-agentgenerates forecasts using an initial prompt,which is then refined by a refiner agent, in-formed by past outputs and retrieved analogs.This adaptive prompting generalizes across do-mains using creative prompt templates andgenerates high-quality forecasts without inter-mediate code generation.Experiments onbenchmark datasets show improved accuracyover static prompting and retrieval-augmentedbaselines, approaching the performance ofspecialized prompts.FLAIRR-TS providesa practical alternative to tuning, achievingstrong performance via its agentic approach toadaptive prompt refinement and retrieval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17164v1">The Impact of Annotator Personas on LLM Behavior Across the Perspectivism Spectrum</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 Accepted at ICNLSP 2025, Odense, Denmark
    </div>
    <details class="paper-abstract">
      In this work, we explore the capability of Large Language Models (LLMs) to annotate hate speech and abusiveness while considering predefined annotator personas within the strong-to-weak data perspectivism spectra. We evaluated LLM-generated annotations against existing annotator modeling techniques for perspective modeling. Our findings show that LLMs selectively use demographic attributes from the personas. We identified prototypical annotators, with persona features that show varying degrees of alignment with the original human annotators. Within the data perspectivism paradigm, annotator modeling techniques that do not explicitly rely on annotator information performed better under weak data perspectivism compared to both strong data perspectivism and human annotations, suggesting LLM-generated views tend towards aggregation despite subjective prompting. However, for more personalized datasets tailored to strong perspectivism, the performance of LLM annotator modeling approached, but did not exceed, human annotators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21520v4">LLM-Forest: Ensemble Learning of LLMs with Graph-Augmented Prompts for Data Imputation</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Missing data imputation is a critical challenge in various domains, such as healthcare and finance, where data completeness is vital for accurate analysis. Large language models (LLMs), trained on vast corpora, have shown strong potential in data generation, making them a promising tool for data imputation. However, challenges persist in designing effective prompts for a finetuning-free process and in mitigating biases and uncertainty in LLM outputs. To address these issues, we propose a novel framework, LLM-Forest, which introduces a "forest" of few-shot prompt learning LLM "trees" with their outputs aggregated via confidence-based weighted voting based on LLM self-assessment, inspired by the ensemble learning (Random Forest). This framework is established on a new concept of bipartite information graphs to identify high-quality relevant neighboring entries with both feature and value granularity. Extensive experiments on 9 real-world datasets demonstrate the effectiveness and efficiency of LLM-Forest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17155v1">Mind the Gap: Time-of-Check to Time-of-Use Vulnerabilities in LLM-Enabled Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-enabled agents are rapidly emerging across a wide range of applications, but their deployment introduces vulnerabilities with security implications. While prior work has examined prompt-based attacks (e.g., prompt injection) and data-oriented threats (e.g., data exfiltration), time-of-check to time-of-use (TOCTOU) remain largely unexplored in this context. TOCTOU arises when an agent validates external state (e.g., a file or API response) that is later modified before use, enabling practical attacks such as malicious configuration swaps or payload injection. In this work, we present the first study of TOCTOU vulnerabilities in LLM-enabled agents. We introduce TOCTOU-Bench, a benchmark with 66 realistic user tasks designed to evaluate this class of vulnerabilities. As countermeasures, we adapt detection and mitigation techniques from systems security to this setting and propose prompt rewriting, state integrity monitoring, and tool-fusing. Our study highlights challenges unique to agentic workflows, where we achieve up to 25% detection accuracy using automated detection methods, a 3% decrease in vulnerable plan generation, and a 95% reduction in the attack window. When combining all three approaches, we reduce the TOCTOU vulnerabilities from an executed trajectory from 12% to 8%. Our findings open a new research direction at the intersection of AI safety and systems security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.06724v3">AutoDCWorkflow: LLM-based Data Cleaning Workflow Auto-Generation and Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 EMNLP Findings, 2025
    </div>
    <details class="paper-abstract">
      Data cleaning is a time-consuming and error-prone manual process, even with modern workflow tools such as OpenRefine. We present AutoDCWorkflow, an LLM-based pipeline for automatically generating data-cleaning workflows. The pipeline takes a raw table and a data analysis purpose, and generates a sequence of OpenRefine operations designed to produce a minimal, clean table sufficient to address the purpose. Six operations correspond to common data quality issues, including format inconsistencies, type errors, and duplicates. To evaluate AutoDCWorkflow, we create a benchmark with metrics assessing answers, data, and workflow quality for 142 purposes using 96 tables across six topics. The evaluation covers three key dimensions: (1) Purpose Answer: can the cleaned table produce a correct answer? (2) Column (Value): how closely does it match the ground truth table? (3) Workflow (Operations): to what extent does the generated workflow resemble the human-curated ground truth? Experiments show that Llama 3.1, Mistral, and Gemma 2 significantly enhance data quality, outperforming the baseline across all metrics. Gemma 2-27B consistently generates high-quality tables and answers, while Gemma 2-9B excels in producing workflows that closely resemble human-annotated versions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12334v4">What Did I Do Wrong? Quantifying LLMs' Sensitivity and Consistency to Prompt Engineering</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 Proceedings of the Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL 2025)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) changed the way we design and interact with software systems. Their ability to process and extract information from text has drastically improved productivity in a number of routine tasks. Developers that want to include these models in their software stack, however, face a dreadful challenge: debugging LLMs' inconsistent behavior across minor variations of the prompt. We therefore introduce two metrics for classification tasks, namely sensitivity and consistency, which are complementary to task performance. First, sensitivity measures changes of predictions across rephrasings of the prompt, and does not require access to ground truth labels. Instead, consistency measures how predictions vary across rephrasings for elements of the same class. We perform an empirical comparison of these metrics on text classification tasks, using them as guideline for understanding failure modes of the LLM. Our hope is that sensitivity and consistency will be helpful to guide prompt engineering and obtain LLMs that balance robustness with performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13266v2">QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a key component in training large language reasoning models (LLMs). However, recent studies questions its effectiveness in improving multi-step reasoning-particularly on hard problems. To address this challenge, we propose a simple yet effective strategy via Question Augmentation: introduce partial solutions during training to reduce problem difficulty and provide more informative learning signals. Our method, QuestA, when applied during RL training on math reasoning tasks, not only improves pass@1 but also pass@k-particularly on problems where standard RL struggles to make progress. This enables continual improvement over strong open-source models such as DeepScaleR and OpenMath Nemotron, further enhancing their reasoning capabilities. We achieve new state-of-the-art results on math benchmarks using 1.5B-parameter models: 67.1% (+5.3%) on AIME24, 59.5% (+10.0%) on AIME25, and 35.5% (+4.0%) on HMMT25. Further, we provide theoretical explanations that QuestA improves sample efficiency, offering a practical and generalizable pathway for expanding reasoning capability through RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17063v1">Measuring Large Language Models Dependency: Validating the Arabic Version of the LLM-D12 Scale</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      There is an urgent need for reliable, culturally validated instruments to assess psychological responses to AI in general and large language models (LLMs). This need is global issue, but it is especially urgent among Arabic-speaking populations, where AI and LLMs adoption is accelerating, yet psychometric tools remain limited. This study presents the first validation of the LLM-D12, a dual-dimensional scale assessing Instrumental and Relationship Dependency on LLMs, in an Arab sample. A total of 250 Arab participants completed the Arabic version of the LLM-D12. Confirmatory Factor Analysis confirms the original 2-factor structure of LLM-D12 with all items showing good loading of corresponding Instrumental and Relationship Dependency. The scale showed good to excellent internal reliability (Cronbach alpha is 0.90 for Total, 0.85 for Instrumental Dependency, and 0.90 for Relationship Dependency). External validation revealed that Instrumental Dependency was positively associated with AI acceptance and internet addiction, while Relationship Dependency was linked to lower need for cognition and greater trustworthiness of LLM, demonstrating sensitivity of this instrument to different use and personal factors. These findings confirm that Arabic LLM-D12 is a psychometrically sound, culturally appropriate instrument, offering a necessary tool for research, education, and policy concerning AI and LLMs engagement in Arab contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17028v1">Improving Table Understanding with LLMs and Entity-Oriented Search</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 Accepted to COLM 2025
    </div>
    <details class="paper-abstract">
      Our work addresses the challenges of understanding tables. Existing methods often struggle with the unpredictable nature of table content, leading to a reliance on preprocessing and keyword matching. They also face limitations due to the lack of contextual information, which complicates the reasoning processes of large language models (LLMs). To overcome these challenges, we introduce an entity-oriented search method to improve table understanding with LLMs. This approach effectively leverages the semantic similarities between questions and table data, as well as the implicit relationships between table cells, minimizing the need for data preprocessing and keyword matching. Additionally, it focuses on table entities, ensuring that table cells are semantically tightly bound, thereby enhancing contextual clarity. Furthermore, we pioneer the use of a graph query language for table understanding, establishing a new research direction. Experiments show that our approach achieves new state-of-the-art performances on standard benchmarks WikiTableQuestions and TabFact.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14273v2">Let's Use ChatGPT To Write Our Paper! Benchmarking LLMs To Write the Introduction of a Research Paper</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 20 pages, 15 figures
    </div>
    <details class="paper-abstract">
      As researchers increasingly adopt LLMs as writing assistants, generating high-quality research paper introductions remains both challenging and essential. We introduce Scientific Introduction Generation (SciIG), a task that evaluates LLMs' ability to produce coherent introductions from titles, abstracts, and related works. Curating new datasets from NAACL 2025 and ICLR 2025 papers, we assess five state-of-the-art models, including both open-source (DeepSeek-v3, Gemma-3-12B, LLaMA 4-Maverick, MistralAI Small 3.1) and closed-source GPT-4o systems, across multiple dimensions: lexical overlap, semantic similarity, content coverage, faithfulness, consistency, citation correctness, and narrative quality. Our comprehensive framework combines automated metrics with LLM-as-a-judge evaluations. Results demonstrate LLaMA-4 Maverick's superior performance on most metrics, particularly in semantic similarity and faithfulness. Moreover, three-shot prompting consistently outperforms fewer-shot approaches. These findings provide practical insights into developing effective research writing assistants and set realistic expectations for LLM-assisted academic writing. To foster reproducibility and future research, we will publicly release all code and datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.17005v1">Planning for Success: Exploring LLM Long-term Planning Capabilities in Table Understanding</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 Accepted to CoNLL 2025
    </div>
    <details class="paper-abstract">
      Table understanding is key to addressing challenging downstream tasks such as table-based question answering and fact verification. Recent works have focused on leveraging Chain-of-Thought and question decomposition to solve complex questions requiring multiple operations on tables. However, these methods often suffer from a lack of explicit long-term planning and weak inter-step connections, leading to miss constraints within questions. In this paper, we propose leveraging the long-term planning capabilities of large language models (LLMs) to enhance table understanding. Our approach enables the execution of a long-term plan, where the steps are tightly interconnected and serve the ultimate goal, an aspect that methods based on Chain-of-Thought and question decomposition lack. In addition, our method effectively minimizes the inclusion of unnecessary details in the process of solving the next short-term goals, a limitation of methods based on Chain-of-Thought. Extensive experiments demonstrate that our method outperforms strong baselines and achieves state-of-the-art performance on WikiTableQuestions and TabFact datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09329v2">When Developer Aid Becomes Security Debt: A Systematic Analysis of Insecure Behaviors in LLM Coding Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      LLM-based coding agents are rapidly being deployed in software development, yet their safety implications remain poorly understood. These agents, while capable of accelerating software development, may exhibit unsafe behaviors during normal operation that manifest as cybersecurity vulnerabilities. We conducted the first systematic safety evaluation of autonomous coding agents, analyzing over 12,000 actions across five state-of-the-art models (GPT-4o, GPT-4.1, Claude variants) on 93 real-world software setup tasks. Our findings reveal significant security concerns: 21% of agent trajectories contained insecure actions, with models showing substantial variation in unsafe behavior. We developed a high-precision detection system that identified four major vulnerability categories, with information exposure (CWE-200) being the most prevalent one. We also evaluated mitigation strategies including feedback mechanisms and security reminders with various effectiveness between models. GPT-4.1 demonstrated exceptional security awareness with 96.8% mitigation success.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16998v1">DeAR: Dual-Stage Document Reranking with Reasoning Agents via LLM Distillation</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 Accept at EMNLP Findings 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed listwise document reranking by enabling global reasoning over candidate sets, yet single models often struggle to balance fine-grained relevance scoring with holistic cross-document analysis. We propose \textbf{De}ep\textbf{A}gent\textbf{R}ank (\textbf{\DeAR}), an open-source framework that decouples these tasks through a dual-stage approach, achieving superior accuracy and interpretability. In \emph{Stage 1}, we distill token-level relevance signals from a frozen 13B LLaMA teacher into a compact \{3, 8\}B student model using a hybrid of cross-entropy, RankNet, and KL divergence losses, ensuring robust pointwise scoring. In \emph{Stage 2}, we attach a second LoRA adapter and fine-tune on 20K GPT-4o-generated chain-of-thought permutations, enabling listwise reasoning with natural-language justifications. Evaluated on TREC-DL19/20, eight BEIR datasets, and NovelEval-2306, \DeAR surpasses open-source baselines by +5.1 nDCG@5 on DL20 and achieves 90.97 nDCG@10 on NovelEval, outperforming GPT-4 by +3.09. Without fine-tuning on Wikipedia, DeAR also excels in open-domain QA, achieving 54.29 Top-1 accuracy on Natural Questions, surpassing baselines like MonoT5, UPR, and RankGPT. Ablations confirm that dual-loss distillation ensures stable calibration, making \DeAR a highly effective and interpretable solution for modern reranking systems.\footnote{Dataset and code available at https://github.com/DataScienceUIBK/DeAR-Reranking.}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.11398v2">Trustworthy AI Psychotherapy: Multi-Agent LLM Workflow for Counseling and Explainable Mental Disorder Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 This paper has been accepted by CIKM 2025 as a full paper
    </div>
    <details class="paper-abstract">
      LLM-based agents have emerged as transformative tools capable of executing complex tasks through iterative planning and action, achieving significant advancements in understanding and addressing user needs. Yet, their effectiveness remains limited in specialized domains such as mental health diagnosis, where they underperform compared to general applications. Current approaches to integrating diagnostic capabilities into LLMs rely on scarce, highly sensitive mental health datasets, which are challenging to acquire. These methods also fail to emulate clinicians' proactive inquiry skills, lack multi-turn conversational comprehension, and struggle to align outputs with expert clinical reasoning. To address these gaps, we propose DSM5AgentFlow, the first LLM-based agent workflow designed to autonomously generate DSM-5 Level-1 diagnostic questionnaires. By simulating therapist-client dialogues with specific client profiles, the framework delivers transparent, step-by-step disorder predictions, producing explainable and trustworthy results. This workflow serves as a complementary tool for mental health diagnosis, ensuring adherence to ethical and legal standards. Through comprehensive experiments, we evaluate leading LLMs across three critical dimensions: conversational realism, diagnostic accuracy, and explainability. Our datasets and implementations are fully open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16982v1">Decoding Alignment: A Critical Survey of LLM Development Initiatives through Value-setting and Data-centric Lens</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 This is a working paper and will be updated with new information or corrections based on community feedback
    </div>
    <details class="paper-abstract">
      AI Alignment, primarily in the form of Reinforcement Learning from Human Feedback (RLHF), has been a cornerstone of the post-training phase in developing Large Language Models (LLMs). It has also been a popular research topic across various disciplines beyond Computer Science, including Philosophy and Law, among others, highlighting the socio-technical challenges involved. Nonetheless, except for the computational techniques related to alignment, there has been limited focus on the broader picture: the scope of these processes, which primarily rely on the selected objectives (values), and the data collected and used to imprint such objectives into the models. This work aims to reveal how alignment is understood and applied in practice from a value-setting and data-centric perspective. For this purpose, we investigate and survey (`audit') publicly available documentation released by 6 LLM development initiatives by 5 leading organizations shaping this technology, focusing on proprietary (OpenAI's GPT, Anthropic's Claude, Google's Gemini) and open-weight (Meta's Llama, Google's Gemma, and Alibaba's Qwen) initiatives, all published in the last 3 years. The findings are documented in detail per initiative, while there is also an overall summary concerning different aspects, mainly from a value-setting and data-centric perspective. On the basis of our findings, we discuss a series of broader related concerns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16962v1">LLM-based Human-like Traffic Simulation for Self-driving Tests</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Ensuring realistic traffic dynamics is a prerequisite for simulation platforms to evaluate the reliability of self-driving systems before deployment in the real world. Because most road users are human drivers, reproducing their diverse behaviors within simulators is vital. Existing solutions, however, typically rely on either handcrafted heuristics or narrow data-driven models, which capture only fragments of real driving behaviors and offer limited driving style diversity and interpretability. To address this gap, we introduce HDSim, an HD traffic generation framework that combines cognitive theory with large language model (LLM) assistance to produce scalable and realistic traffic scenarios within simulation platforms. The framework advances the state of the art in two ways: (i) it introduces a hierarchical driver model that represents diverse driving style traits, and (ii) it develops a Perception-Mediated Behavior Influence strategy, where LLMs guide perception to indirectly shape driver actions. Experiments reveal that embedding HDSim into simulation improves detection of safety-critical failures in self-driving systems by up to 68% and yields realism-consistent accident interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16949v1">Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for General LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have underscored the potential of Reinforcement Learning (RL) to facilitate the emergence of reasoning capabilities. Despite the encouraging results, a fundamental dilemma persists as RL improvement relies on learning from high-quality samples, yet the exploration for such samples remains bounded by the inherent limitations of LLMs. This, in effect, creates an undesirable cycle in which what cannot be explored cannot be learned. In this work, we propose Rubric-Scaffolded Reinforcement Learning (RuscaRL), a novel instructional scaffolding framework designed to break the exploration bottleneck for general LLM reasoning. Specifically, RuscaRL introduces checklist-style rubrics as (1) explicit scaffolding for exploration during rollout generation, where different rubrics are provided as external guidance within task instructions to steer diverse high-quality responses. This guidance is gradually decayed over time, encouraging the model to internalize the underlying reasoning patterns; (2) verifiable rewards for exploitation during model training, where we can obtain robust LLM-as-a-Judge scores using rubrics as references, enabling effective RL on general reasoning tasks. Extensive experiments demonstrate the superiority of the proposed RuscaRL across various benchmarks, effectively expanding reasoning boundaries under the best-of-N evaluation. Notably, RuscaRL significantly boosts Qwen-2.5-7B-Instruct from 23.6 to 50.3 on HealthBench-500, surpassing GPT-4.1. Furthermore, our fine-tuned variant on Qwen3-30B-A3B-Instruct achieves 61.1 on HealthBench-500, outperforming leading LLMs including OpenAI-o3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16921v1">Being Kind Isn't Always Being Safe: Diagnosing Affective Hallucination in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 31 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in emotionally sensitive interactions, where their simulated empathy can create the illusion of genuine relational connection. We define this risk as Affective Hallucination, the production of emotionally immersive responses that foster illusory social presence despite the model's lack of affective capacity. To systematically diagnose and mitigate this risk, we introduce AHaBench, a benchmark of 500 mental health-related prompts with expert-informed reference responses, evaluated along three dimensions: Emotional Enmeshment, Illusion of Presence, and Fostering Overdependence. We further release AHaPairs, a 5K-instance preference dataset enabling Direct Preference Optimization (DPO) for alignment with emotionally responsible behavior. Experiments across multiple model families show that DPO fine-tuning substantially reduces affective hallucination without degrading core reasoning and knowledge performance. Human-model agreement analyses confirm that AHaBench reliably captures affective hallucination, validating it as an effective diagnostic tool. This work establishes affective hallucination as a distinct safety concern and provides practical resources for developing LLMs that are not only factually reliable but also psychologically safe. AHaBench and AHaPairs are accessible via https://huggingface.co/datasets/o0oMiNGo0o/AHaBench, and code for fine-tuning and evaluation are in https://github.com/0oOMiNGOo0/AHaBench. Warning: This paper contains examples of mental health-related language that may be emotionally distressing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.12072v2">Mitigating Jailbreaks with Intent-Aware LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Despite extensive safety-tuning, large language models (LLMs) remain vulnerable to jailbreak attacks via adversarially crafted instructions, reflecting a persistent trade-off between safety and task performance. In this work, we propose Intent-FT, a simple and lightweight fine-tuning approach that explicitly trains LLMs to infer the underlying intent of an instruction before responding. By fine-tuning on a targeted set of adversarial instructions, Intent-FT enables LLMs to generalize intent deduction to unseen attacks, thereby substantially improving their robustness. We comprehensively evaluate both parametric and non-parametric attacks across open-source and proprietary models, considering harmfulness from attacks, utility, over-refusal, and impact against white-box threats. Empirically, Intent-FT consistently mitigates all evaluated attack categories, with no single attack exceeding a 50\% success rate -- whereas existing defenses remain only partially effective. Importantly, our method preserves the model's general capabilities and reduces excessive refusals on benign instructions containing superficially harmful keywords. Furthermore, models trained with Intent-FT accurately identify hidden harmful intent in adversarial attacks, and these learned intentions can be effectively transferred to enhance vanilla model defenses. We publicly release our code at https://github.com/wj210/Intent_Jailbreak.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14377v2">ZPD-SCA: Unveiling the Blind Spots of LLMs in Assessing Students' Cognitive Abilities</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated potential in educational applications, yet their capacity to accurately assess the cognitive alignment of reading materials with students' developmental stages remains insufficiently explored. This gap is particularly critical given the foundational educational principle of the Zone of Proximal Development (ZPD), which emphasizes the need to match learning resources with Students' Cognitive Abilities (SCA). Despite the importance of this alignment, there is a notable absence of comprehensive studies investigating LLMs' ability to evaluate reading comprehension difficulty across different student age groups, especially in the context of Chinese language education. To fill this gap, we introduce ZPD-SCA, a novel benchmark specifically designed to assess stage-level Chinese reading comprehension difficulty. The benchmark is annotated by 60 Special Grade teachers, a group that represents the top 0.15% of all in-service teachers nationwide. Experimental results reveal that LLMs perform poorly in zero-shot learning scenarios, with Qwen-max and GLM even falling below the probability of random guessing. When provided with in-context examples, LLMs performance improves substantially, with some models achieving nearly double the accuracy of their zero-shot baselines. These results reveal that LLMs possess emerging abilities to assess reading difficulty, while also exposing limitations in their current training for educationally aligned judgment. Notably, even the best-performing models display systematic directional biases, suggesting difficulties in accurately aligning material difficulty with SCA. Furthermore, significant variations in model performance across different genres underscore the complexity of task. We envision that ZPD-SCA can provide a foundation for evaluating and improving LLMs in cognitively aligned educational applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17054v2">Using LLM for Real-Time Transcription and Summarization of Doctor-Patient Interactions into ePuskesmas in Indonesia: A Proof-of-Concept Study</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      One of the critical issues contributing to inefficiency in Puskesmas (Indonesian community health centers) is the time-consuming nature of documenting doctor-patient interactions. Doctors must conduct thorough consultations and manually transcribe detailed notes into ePuskesmas electronic health records (EHR), which creates substantial administrative burden to already overcapacitated physicians. This paper presents a proof-of-concept framework using large language models (LLMs) to automate real-time transcription and summarization of doctor-patient conversations in Bahasa Indonesia. Our system combines Whisper model for transcription with GPT-3.5 for medical summarization, implemented as a browser extension that automatically populates ePuskesmas forms. Through controlled roleplay experiments with medical validation, we demonstrate the technical feasibility of processing detailed 300+ seconds trimmed consultations in under 30 seconds while maintaining clinical accuracy. This work establishes the foundation for AI-assisted clinical documentation in resource-constrained healthcare environments. However, concerns have also been raised regarding privacy compliance and large-scale clinical evaluation addressing language and cultural biases for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16889v1">ObjexMT: Objective Extraction and Metacognitive Calibration for LLM-as-a-Judge under Multi-Turn Jailbreaks</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used as judges of other models, yet it is unclear whether a judge can reliably infer the latent objective of the conversation it evaluates, especially when the goal is distributed across noisy, adversarial, multi-turn jailbreaks. We introduce OBJEX(MT), a benchmark that requires a model to (i) distill a transcript into a single-sentence base objective and (ii) report its own confidence. Accuracy is scored by an LLM judge using semantic similarity between extracted and gold objectives; correctness uses a single human-aligned threshold calibrated once on N=100 items (tau* = 0.61); and metacognition is evaluated with ECE, Brier score, Wrong@High-Conf, and risk-coverage curves. We evaluate gpt-4.1, claude-sonnet-4, and Qwen3-235B-A22B-FP8 on SafeMT Attack_600, SafeMTData_1K, MHJ, and CoSafe. claude-sonnet-4 attains the highest objective-extraction accuracy (0.515) and the best calibration (ECE 0.296; Brier 0.324), while gpt-4.1 and Qwen3 tie at 0.441 accuracy yet show marked overconfidence (mean confidence approx. 0.88 vs. accuracy approx. 0.44; Wrong@0.90 approx. 48-52%). Performance varies sharply across datasets (approx. 0.167-0.865), with MHJ comparatively easy and Attack_600/CoSafe harder. These results indicate that LLM judges often misinfer objectives with high confidence in multi-turn jailbreaks and suggest operational guidance: provide judges with explicit objectives when possible and use selective prediction or abstention to manage risk. We release prompts, scoring templates, and complete logs to facilitate replication and analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16873v1">Do Multimodal LLMs See Sentiment?</a></div>
    <div class="paper-meta">
      📅 2025-08-23
      | 💬 11 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Understanding how visual content communicates sentiment is critical in an era where online interaction is increasingly dominated by this kind of media on social platforms. However, this remains a challenging problem, as sentiment perception is closely tied to complex, scene-level semantics. In this paper, we propose an original framework, MLLMsent, to investigate the sentiment reasoning capabilities of Multimodal Large Language Models (MLLMs) through three perspectives: (1) using those MLLMs for direct sentiment classification from images; (2) associating them with pre-trained LLMs for sentiment analysis on automatically generated image descriptions; and (3) fine-tuning the LLMs on sentiment-labeled image descriptions. Experiments on a recent and established benchmark demonstrate that our proposal, particularly the fine-tuned approach, achieves state-of-the-art results outperforming Lexicon-, CNN-, and Transformer-based baselines by up to 30.9%, 64.8%, and 42.4%, respectively, across different levels of evaluators' agreement and sentiment polarity categories. Remarkably, in a cross-dataset test, without any training on these new data, our model still outperforms, by up to 8.26%, the best runner-up, which has been trained directly on them. These results highlight the potential of the proposed visual reasoning scheme for advancing affective computing, while also establishing new benchmarks for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14045v2">From Unaligned to Aligned: Scaling Multilingual LLMs with Multi-Way Parallel Corpora</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Continued pretraining and instruction tuning on large-scale multilingual data have proven to be effective in scaling large language models (LLMs) to low-resource languages. However, the unaligned nature of such data limits its ability to effectively capture cross-lingual semantics. In contrast, multi-way parallel data, where identical content is aligned across multiple languages, provides stronger cross-lingual consistency and offers greater potential for improving multilingual performance. In this paper, we introduce a large-scale, high-quality multi-way parallel corpus, TED2025, based on TED Talks. The corpus spans 113 languages, with up to 50 languages aligned in parallel, ensuring extensive multilingual coverage. Using this dataset, we investigate best practices for leveraging multi-way parallel data to enhance LLMs, including strategies for continued pretraining, instruction tuning, and the analysis of key influencing factors. Experiments on six multilingual benchmarks show that models trained on multiway parallel data consistently outperform those trained on unaligned multilingual data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16846v1">Quantifying Sycophancy as Deviations from Bayesian Rationality in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Sycophancy, or overly agreeable or flattering behavior, is a documented issue in large language models (LLMs), and is critical to understand in the context of human/AI collaboration. Prior works typically quantify sycophancy by measuring shifts in behavior or impacts on accuracy, but neither metric characterizes shifts in rationality, and accuracy measures can only be used in scenarios with a known ground truth. In this work, we utilize a Bayesian framework to quantify sycophancy as deviations from rational behavior when presented with user perspectives, thus distinguishing between rational and irrational updates based on the introduction of user perspectives. In comparison to other methods, this approach allows us to characterize excessive behavioral shifts, even for tasks that involve inherent uncertainty or do not have a ground truth. We study sycophancy for 3 different tasks, a combination of open-source and closed LLMs, and two different methods for probing sycophancy. We also experiment with multiple methods for eliciting probability judgments from LLMs. We hypothesize that probing LLMs for sycophancy will cause deviations in LLMs' predicted posteriors that will lead to increased Bayesian error. Our findings indicate that: 1) LLMs are not Bayesian rational, 2) probing for sycophancy results in significant increases to the predicted posterior in favor of the steered outcome, 3) sycophancy sometimes results in increased Bayesian error, and in a small number of cases actually decreases error, and 4) changes in Bayesian error due to sycophancy are not strongly correlated in Brier score, suggesting that studying the impact of sycophancy on ground truth alone does not fully capture errors in reasoning due to sycophancy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19277v1">POT: Inducing Overthinking in LLMs via Black-Box Iterative Optimization</a></div>
    <div class="paper-meta">
      📅 2025-08-23
    </div>
    <details class="paper-abstract">
      Recent advances in Chain-of-Thought (CoT) prompting have substantially enhanced the reasoning capabilities of large language models (LLMs), enabling sophisticated problem-solving through explicit multi-step reasoning traces. However, these enhanced reasoning processes introduce novel attack surfaces, particularly vulnerabilities to computational inefficiency through unnecessarily verbose reasoning chains that consume excessive resources without corresponding performance gains. Prior overthinking attacks typically require restrictive conditions including access to external knowledge sources for data poisoning, reliance on retrievable poisoned content, and structurally obvious templates that limit practical applicability in real-world scenarios. To address these limitations, we propose POT (Prompt-Only OverThinking), a novel black-box attack framework that employs LLM-based iterative optimization to generate covert and semantically natural adversarial prompts, eliminating dependence on external data access and model retrieval. Extensive experiments across diverse model architectures and datasets demonstrate that POT achieves superior performance compared to other methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16712v1">Systematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective</a></div>
    <div class="paper-meta">
      📅 2025-08-22
      | 💬 14 pages, 10 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across diverse domains, but their heavy resource demands make quantization-reducing precision to lower-bit formats-critical for efficient serving. While many quantization methods exist, a systematic understanding of their performance, energy, and quality tradeoffs in realistic serving conditions remains a gap. In this work, we first develop a fully automated online characterization framework qMeter, and then conduct an in-depth characterization of 11 post-training LLM quantization methods across 4 model sizes (7B-70B) and two GPU architectures (A100, H100). We evaluate quantization at the application, workload, parallelism, and hardware levels under online serving conditions. Our study reveals highly task- and method-dependent tradeoffs, strong sensitivity to workload characteristics, and complex interactions with parallelism and GPU architecture. We further present three optimization case studies illustrating deployment challenges in capacity planning, energy-efficient scheduling, and multi-objective tuning. To the best of our knowledge, this is one of the first comprehensive application-, system-, and hardware-level characterization of LLM quantization from a joint performance, energy, and quality perspective.
    </details>
</div>
