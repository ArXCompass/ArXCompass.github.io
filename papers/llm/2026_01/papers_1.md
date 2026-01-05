# llm - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.15131v2">Modeling the One-to-Many Property in Open-Domain Dialogue with LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Open-domain Dialogue (OD) exhibits a one-to-many (o2m) property, whereby multiple appropriate responses exist for a single dialogue context. Despite prior research showing that modeling this property boosts response diversity, most modern LLM-based dialogue agents do not explicitly do so. In this work, we model the o2m property of OD in LLMs by decomposing OD generation into two key tasks: Multi-Response Generation (MRG) and Preference-based Selection (PS), which entail generating a set of n semantically and lexically diverse high-quality responses for a given dialogue context, followed by selecting a single response based on human preference, respectively. To facilitate MRG and PS, we introduce o2mDial, a dialogue corpus explicitly designed to capture the o2m property by featuring multiple plausible responses for each context. Leveraging o2mDial, we propose new in-context learning and instruction-tuning strategies, as well as novel evaluation metrics for MRG, alongside a model-based approach for PS. Empirical results demonstrate that applying the proposed two-stage framework to smaller LLMs for OD generation enhances overall response diversity while maintaining contextual coherence, improving response quality by up to 90%, bringing them closer to the performance of larger models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.07675v3">QUITE: A Query Rewrite System Beyond Rules with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Query rewrite transforms SQL queries into semantically equivalent forms that run more efficiently. Existing approaches mainly rely on predefined rewrite rules, but they handle a limited subset of queries and can cause performance regressions. This limitation stems from three challenges of rule-based query rewrite: (1) it is hard to discover and verify new rules, (2) fixed rewrite rules do not generalize to new query patterns, and (3) some rewrite techniques cannot be expressed as fixed rules. Motivated by the fact that human experts exhibit significantly better rewrite ability but suffer from scalability, and Large Language Models (LLMs) have demonstrated nearly human-level semantic and reasoning abilities, we propose a new approach of using LLMs to rewrite SQL queries beyond rules. Due to the hallucination problems in LLMs, directly applying LLMs often leads to nonequivalent and suboptimal queries. To address this issue, we propose QUITE (query rewrite), a training-free and feedback-aware system based on LLM agents that rewrites SQL queries into semantically equivalent forms with significantly better performance, covering a broader range of query patterns and rewrite strategies compared to rule-based methods. Firstly, we design a multi-agent framework controlled by a finite state machine (FSM) to equip LLMs with the ability to use external tools and enhance the rewrite process with real-time database feedback. Secondly, we develop a rewrite middleware to enhance the ability of LLMs to generate optimized query equivalents. Finally, we employ a novel hint injection technique to improve execution plans for rewritten queries. Extensive experiments show that QUITE reduces query execution time by up to 35.8% over state-of-the-art approaches and produces 24.1% more rewrites than prior methods, covering query cases that earlier systems did not handle.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22905v2">JavisGPT: A Unified Multi-modal LLM for Sounding-Video Comprehension and Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-02
      | 💬 Accepted by NeurIPS as a Spotlight paper. Code: https://github.com/JavisVerse/JavisGPT
    </div>
    <details class="paper-abstract">
      This paper presents JavisGPT, the first unified multimodal large language model (MLLM) for joint audio-video (JAV) comprehension and generation. JavisGPT has a concise encoder-LLM-decoder architecture, which has a SyncFusion module for spatio-temporal audio-video fusion and synchrony-aware learnable queries to bridge a pretrained JAV-DiT generator. This design enables temporally coherent video-audio understanding and generation from multimodal instructions. We design an effective three-stage training pipeline consisting of multimodal pretraining, audio-video fine-tuning, and large-scale instruction-tuning, to progressively build multimodal comprehension and generation from existing vision-language models. For instruction tuning, we construct JavisInst-Omni, a high-quality instruction dataset with over 200K GPT-4o-curated audio-video-text dialogues that cover diverse and multi-level comprehension and generation scenarios. On JAV comprehension and generation benchmarks, our experiments show that JavisGPT outperforms existing MLLMs, particularly in complex and temporally synchronized settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00644v1">FlexSpec: Frozen Drafts Meet Evolving Targets in Edge-Cloud Collaborative LLM Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) in mobile and edge computing environments is constrained by limited on-device resources, scarce wireless bandwidth, and frequent model evolution. Although edge-cloud collaborative inference with speculative decoding (SD) can reduce end-to-end latency by executing a lightweight draft model at the edge and verifying it with a cloud-side target model, existing frameworks fundamentally rely on tight coupling between the two models. Consequently, repeated model synchronization introduces excessive communication overhead, increasing end-to-end latency, and ultimately limiting the scalability of SD in edge environments. To address these limitations, we propose FlexSpec, a communication-efficient collaborative inference framework tailored for evolving edge-cloud systems. The core design of FlexSpec is a shared-backbone architecture that allows a single and static edge-side draft model to remain compatible with a large family of evolving cloud-side target models. By decoupling edge deployment from cloud-side model updates, FlexSpec eliminates the need for edge-side retraining or repeated model downloads, substantially reducing communication and maintenance costs. Furthermore, to accommodate time-varying wireless conditions and heterogeneous device constraints, we develop a channel-aware adaptive speculation mechanism that dynamically adjusts the speculative draft length based on real-time channel state information and device energy budgets. Extensive experiments demonstrate that FlexSpec achieves superior performance compared to conventional SD approaches in terms of inference efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00641v1">Probabilistic Guarantees for Reducing Contextual Hallucinations in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently produce contextual hallucinations, where generated content contradicts or ignores information explicitly stated in the prompt. Such errors are particularly problematic in deterministic automation workflows, where inputs are fixed and correctness is unambiguous. We introduce a simple and model-agnostic framework that provides explicit probabilistic guarantees for reducing hallucinations in this setting. We formalize the notion of a specific task, defined by a fixed input and a deterministic correctness criterion, and show that issuing the same prompt in independent context windows yields an exponential reduction in the probability that all model outputs are incorrect. To identify a correct answer among repeated runs, we incorporate an LLM-as-a-judge and prove that the probability that the judged pipeline fails decays at a rate determined by the judge's true- and false-positive probabilities. When the judge is imperfect, we strengthen it through majority vote over independent judge calls, obtaining ensemble-level error rates that decrease exponentially in the number of votes. This yields an explicit bound on the probability that the pipeline selects a hallucinated answer. Experiments on controlled extraction tasks with synthetic noisy judges match these predictions exactly: pipeline failure decreases exponentially with the number of repetitions, and hallucination-selection decreases exponentially with the number of judges in the ensemble. Together, these results provide a lightweight, modular, and theoretically grounded method for driving hallucination probabilities arbitrarily low in fixed-input LLM workflows-without modifying model weights, decoding strategies, or prompt engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00624v1">Do Chatbot LLMs Talk Too Much? The YapBench Benchmark</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as ChatGPT, Claude, and Gemini increasingly act as general-purpose copilots, yet they often respond with unnecessary length on simple requests, adding redundant explanations, hedging, or boilerplate that increases cognitive load and inflates token-based inference cost. Prior work suggests that preference-based post-training and LLM-judged evaluations can induce systematic length bias, where longer answers are rewarded even at comparable quality. We introduce YapBench, a lightweight benchmark for quantifying user-visible over-generation on brevity-ideal prompts. Each item consists of a single-turn prompt, a curated minimal-sufficient baseline answer, and a category label. Our primary metric, YapScore, measures excess response length beyond the baseline in characters, enabling comparisons across models without relying on any specific tokenizer. We summarize model performance via the YapIndex, a uniformly weighted average of category-level median YapScores. YapBench contains over three hundred English prompts spanning three common brevity-ideal settings: (A) minimal or ambiguous inputs where the ideal behavior is a short clarification, (B) closed-form factual questions with short stable answers, and (C) one-line coding tasks where a single command or snippet suffices. Evaluating 76 assistant LLMs, we observe an order-of-magnitude spread in median excess length and distinct category-specific failure modes, including vacuum-filling on ambiguous inputs and explanation or formatting overhead on one-line technical requests. We release the benchmark and maintain a live leaderboard for tracking verbosity behavior over time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00596v1">Beyond IVR: Benchmarking Customer Support LLM Agents for Business-Adherence</a></div>
    <div class="paper-meta">
      📅 2026-01-02
      | 💬 17 pages, 3 figures, preprint
    </div>
    <details class="paper-abstract">
      Traditional customer support systems, such as Interactive Voice Response (IVR), rely on rigid scripts and lack the flexibility required for handling complex, policy-driven tasks. While large language model (LLM) agents offer a promising alternative, evaluating their ability to act in accordance with business rules and real-world support workflows remains an open challenge. Existing benchmarks primarily focus on tool usage or task completion, overlooking an agent's capacity to adhere to multi-step policies, navigate task dependencies, and remain robust to unpredictable user or environment behavior. In this work, we introduce JourneyBench, a benchmark designed to assess policy-aware agents in customer support. JourneyBench leverages graph representations to generate diverse, realistic support scenarios and proposes the User Journey Coverage Score, a novel metric to measure policy adherence. We evaluate multiple state-of-the-art LLMs using two agent designs: a Static-Prompt Agent (SPA) and a Dynamic-Prompt Agent (DPA) that explicitly models policy control. Across 703 conversations in three domains, we show that DPA significantly boosts policy adherence, even allowing smaller models like GPT-4o-mini to outperform more capable ones like GPT-4o. Our findings demonstrate the importance of structured orchestration and establish JourneyBench as a critical resource to advance AI-driven customer support beyond IVR-era limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00588v1">CSSBench: Evaluating the Safety of Lightweight LLMs against Chinese-Specific Adversarial Patterns</a></div>
    <div class="paper-meta">
      📅 2026-01-02
      | 💬 18 pages
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in cost-sensitive and on-device scenarios, and safety guardrails have advanced mainly in English. However, real-world Chinese malicious queries typically conceal intent via homophones, pinyin, symbol-based splitting, and other Chinese-specific patterns. These Chinese-specific adversarial patterns create the safety evaluation gap that is not well captured by existing benchmarks focused on English. This gap is particularly concerning for lightweight models, which may be more vulnerable to such specific adversarial perturbations. To bridge this gap, we introduce the Chinese-Specific Safety Benchmark (CSSBench) that emphasizes these adversarial patterns and evaluates the safety of lightweight LLMs in Chinese. Our benchmark covers six domains that are common in real Chinese scenarios, including illegal activities and compliance, privacy leakage, health and medical misinformation, fraud and hate, adult content, and public and political safety, and organizes queries into multiple task types. We evaluate a set of popular lightweight LLMs and measure over-refusal behavior to assess safety-induced performance degradation. Our results show that the Chinese-specific adversarial pattern is a critical challenge for lightweight LLMs. This benchmark offers a comprehensive evaluation of LLM safety in Chinese, assisting robust deployments in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00575v1">InfoSynth: Information-Guided Benchmark Synthesis for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant advancements in reasoning and code generation. However, efficiently creating new benchmarks to evaluate these capabilities remains a challenge. Traditional benchmark creation relies on manual human effort, a process that is both expensive and time-consuming. Furthermore, existing benchmarks often contaminate LLM training data, necessitating novel and diverse benchmarks to accurately assess their genuine capabilities. This work introduces InfoSynth, a novel framework for automatically generating and evaluating reasoning benchmarks guided by information-theoretic principles. We propose metrics based on KL-divergence and entropy to quantify benchmark novelty and diversity without relying on costly model evaluations. Building on this framework, we develop an end-to-end pipeline that synthesizes robust Python coding problems from seed datasets using genetic algorithms and iterative code feedback. Our method generates accurate test cases and solutions to new problems 97% of the time, and the synthesized benchmarks consistently exhibit higher novelty and diversity compared to their seed datasets. Moreover, our algorithm provides a method for controlling the novelty/diversity and difficulty of generated problems. InfoSynth offers a scalable, self-verifying pipeline for constructing high-quality, novel and diverse benchmarks for LLMs. Project Page: https://ishirgarg.github.io/infosynth_web/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00570v1">User Perceptions of an LLM-Based Chatbot for Cognitive Reappraisal of Stress: Feasibility Study</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Cognitive reappraisal is a well-studied emotion regulation strategy that helps individuals reinterpret stressful situations to reduce their impact. Many digital mental health tools struggle to support this process because rigid scripts fail to accommodate how users naturally describe stressors. This study examined the feasibility of an LLM-based single-session intervention (SSI) for workplace stress reappraisal. We assessed short-term changes in stress-related outcomes and examined design tensions during use. We conducted a feasibility study with 100 employees at a large technology company who completed a structured cognitive reappraisal session delivered by a GPT-4o-based chatbot. Pre-post measures included perceived stress intensity, stress mindset, perceived demand, and perceived resources. These outcomes were analyzed using paired Wilcoxon signed-rank tests with correction for multiple comparisons. We also examined sentiment and stress trajectories across conversation quartiles using two RoBERTa-based classifiers and an LLM-based stress rater. Open-ended responses were analyzed using thematic analysis. Results showed significant reductions in perceived stress intensity and significant improvements in stress mindset. Changes in perceived resources and perceived demand trended in expected directions but were not statistically significant. Automated analyses indicated consistent declines in negative sentiment and stress over the course of the interaction. Qualitative findings suggested that participants valued the structured prompts for organizing thoughts, gaining perspective, and feeling acknowledged. Participants also reported tensions around scriptedness, preferred interaction length, and reactions to AI-driven empathy. These findings highlight both the promise and the design constraints of integrating LLMs into DMH interventions for workplace settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00566v1">Low Rank Comes with Low Security: Gradient Assembly Poisoning Attacks against Distributed LoRA-based LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-02
      | 💬 8 figures, 8 tables
    </div>
    <details class="paper-abstract">
      Low-Rank Adaptation (LoRA) has become a popular solution for fine-tuning large language models (LLMs) in federated settings, dramatically reducing update costs by introducing trainable low-rank matrices. However, when integrated with frameworks like FedIT, LoRA introduces a critical vulnerability: clients submit $A$ and $B$ matrices separately, while only their product $AB$ determines the model update, yet this composite is never directly verified. We propose Gradient Assembly Poisoning (GAP), a novel attack that exploits this blind spot by crafting individually benign $A$ and $B$ matrices whose product yields malicious updates. GAP operates without access to training data or inter-client coordination and remains undetected by standard anomaly detectors. We identify four systemic vulnerabilities in LoRA-based federated systems and validate GAP across LLaMA, ChatGLM, and GPT-2. GAP consistently induces degraded or biased outputs while preserving surface fluency, reducing BLEU by up to 14.5\%, increasing factual and grammatical errors by over 800\%, and maintaining 92.6\% long-form response length. These results reveal a new class of stealthy, persistent threats in distributed LoRA fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00559v1">Cracking IoT Security: Can LLMs Outsmart Static Analysis Tools?</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Smart home IoT platforms such as openHAB rely on Trigger Action Condition (TAC) rules to automate device behavior, but the interplay among these rules can give rise to interaction threats, unintended or unsafe behaviors emerging from implicit dependencies, conflicting triggers, or overlapping conditions. Identifying these threats requires semantic understanding and structural reasoning that traditionally depend on symbolic, constraint-driven static analysis. This work presents the first comprehensive evaluation of Large Language Models (LLMs) across a multi-category interaction threat taxonomy, assessing their performance on both the original openHAB (oHC/IoTB) dataset and a structurally challenging Mutation dataset designed to test robustness under rule transformations. We benchmark Llama 3.1 8B, Llama 70B, GPT-4o, Gemini-2.5-Pro, and DeepSeek-R1 across zero-, one-, and two-shot settings, comparing their results against oHIT's manually validated ground truth. Our findings show that while LLMs exhibit promising semantic understanding, particularly on action- and condition-related threats, their accuracy degrades significantly for threats requiring cross-rule structural reasoning, especially under mutated rule forms. Model performance varies widely across threat categories and prompt settings, with no model providing consistent reliability. In contrast, the symbolic reasoning baseline maintains stable detection across both datasets, unaffected by rule rewrites or structural perturbations. These results underscore that LLMs alone are not yet dependable for safety critical interaction-threat detection in IoT environments. We discuss the implications for tool design and highlight the potential of hybrid architectures that combine symbolic analysis with LLM-based semantic interpretation to reduce false positives while maintaining structural rigor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01661v2">Learning the Boundary of Solvability: Aligning LLMs to Detect Unsolvable Problems</a></div>
    <div class="paper-meta">
      📅 2026-01-02
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Ensuring large language model (LLM) reliability requires distinguishing objective unsolvability (inherent contradictions) from subjective capability limitations (tasks exceeding model competence). Current LLMs often conflate these dimensions, leading to hallucinations in which they return confident answers to inherently unsolvable queries. To address this issue, we propose a multi-domain dataset containing both solvable and unsolvable questions, UnsolvableQA, together with an alignment framework, UnsolvableRL. First, we construct UnsolvableQA by "Reverse Construction" that systematically injects logical contradictions into otherwise valid reasoning chains. Second, we introduce UnsolvableRL, a reinforcement learning paradigm that balances objective unsolvability detection with calibrated confidence under capability limits. Empirically, our approach achieves near-perfect unsolvability detection (>90% detection rate) and boosts solvable reasoning accuracy from 43.4% to 69.4% on Qwen3-4B-Instruct. Crucially, we identify a data-training interaction: strict alignment constraints induce Capability Collapse without unsolvable data, but act as a regularizer for rigor when such data are included, thereby improving overall robustness. Our code and data are available at https://github.com/sfasfaffa/unsolvableQA .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00555v1">LLM-Based Agentic Exploration for Robot Navigation & Manipulation with Skill Orchestration</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      This paper presents an end-to-end LLM-based agentic exploration system for an indoor shopping task, evaluated in both Gazebo simulation and a corresponding real-world corridor layout. The robot incrementally builds a lightweight semantic map by detecting signboards at junctions and storing direction-to-POI relations together with estimated junction poses, while AprilTags provide repeatable anchors for approach and alignment. Given a natural-language shopping request, an LLM produces a constrained discrete action at each junction (direction and whether to enter a store), and a ROS finite-state main controller executes the decision by gating modular motion primitives, including local-costmap-based obstacle avoidance, AprilTag approaching, store entry, and grasping. Qualitative results show that the integrated stack can perform end-to-end task execution from user instruction to multi-store navigation and object retrieval, while remaining modular and debuggable through its text-based map and logged decision history.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2402.14746v6">Scaling Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Recent LLMs have hundreds of billions of parameters consuming vast resources. Furthermore, the so called "AI scaling law" for transformers suggests that the number of parameters must scale linearly with the size of the data. In response, we inquire into efficient LLMs, i.e. those with the fewest parameters that achieve the desired accuracy on a training corpus. Specifically, by comparing theoretical and empirical estimates of the Kullback-Leibler divergence, we derive a natural AI scaling law that the number of parameters in an efficient LLM scales as $D^γ$ where $D$ is the size of the training data and $ γ\in [0.44, 0.72]$, suggesting the existence of more efficient architectures. Against this backdrop, we propose recurrent transformers, combining the efficacy of transformers with the efficiency of recurrent networks, progressively applying a single transformer layer to a fixed-width sliding window across the input sequence. Recurrent transformers (a) run in linear time in the sequence length, (b) are memory-efficient and amenable to parallel processing in large batches, (c) learn to forget history for language tasks, or accumulate history for long range tasks like copy and selective copy, and (d) are amenable to curriculum training to overcome vanishing gradients. In our experiments, we find that recurrent transformers perform favorably on benchmark tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00509v1">Improving LLM-Assisted Secure Code Generation through Retrieval-Augmented-Generation and Multi-Tool Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can generate code but often introduce security vulnerabilities, logical inconsistencies, and compilation errors. Prior work demonstrates that LLMs benefit substantially from structured feedback, static analysis, retrieval augmentation, and execution-based refinement. We propose a retrieval-augmented, multi-tool repair workflow in which a single code-generating LLM iteratively refines its outputs using compiler diagnostics, CodeQL security scanning, and KLEE symbolic execution. A lightweight embedding model is used for semantic retrieval of previously successful repairs, providing security-focused examples that guide generation. Evaluated on a combined dataset of 3,242 programs generated by DeepSeek-Coder-1.3B and CodeLlama-7B, the system demonstrates significant improvements in robustness. For DeepSeek, security vulnerabilities were reduced by 96%. For the larger CodeLlama model, the critical security defect rate was decreased from 58.55% to 22.19%, highlighting the efficacy of tool-assisted self-repair even on "stubborn" models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00469v1">DSL or Code? Evaluating the Quality of LLM-Generated Algebraic Specifications: A Case Study in Optimization at Kinaxis</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 Accepted for publication in ICSE-SEIP 2026
    </div>
    <details class="paper-abstract">
      Model-driven engineering (MDE) provides abstraction and analytical rigour, but industrial adoption in many domains has been limited by the cost of developing and maintaining models. Large language models (LLMs) can help shift this cost balance by supporting direct generation of models from natural-language (NL) descriptions. For domain-specific languages (DSLs), however, LLM-generated models may be less accurate than LLM-generated code in mainstream languages such as Python, due to the latter's dominance in LLM training corpora. We investigate this issue in mathematical optimization, with AMPL, a DSL with established industrial use. We introduce EXEOS, an LLM-based approach that derives AMPL models and Python code from NL problem descriptions and iteratively refines them with solver feedback. Using a public optimization dataset and real-world supply-chain cases from our industrial partner Kinaxis, we evaluate generated AMPL models against Python code in terms of executability and correctness. An ablation study with two LLM families shows that AMPL is competitive with, and sometimes better than, Python, and that our design choices in EXEOS improve the quality of generated specifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.02497v2">Towards Bridging Language Gaps in OSS with LLM-Driven Documentation Translation</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      While open source communities attract diverse contributors across the globe, only a few open source software repositories provide essential documentation, such as ReadMe or CONTRIBUTING files, in languages other than English. Recently, large language models (LLMs) have demonstrated remarkable capabilities in a variety of software engineering tasks. We have also seen advances in the use of LLMs for translations in other domains and contexts. Despite this progress, little is known regarding the capabilities of LLMs in translating open-source technical documentation, which is often a mixture of natural language, code, URLs, and markdown formatting. To better understand the need and potential for LLMs to support translation of technical documentation in open source, we conducted an empirical evaluation of translation activity and translation capabilities of two powerful large language models (OpenAI ChatGPT 4 and Anthropic Claude). We found that translation activity is often community-driven and most frequent in larger repositories. A comparison of LLM performance as translators and evaluators of technical documentation suggests LLMs can provide accurate semantic translations but may struggle preserving structure and technical content. These findings highlight both the promise and the challenges of LLM-assisted documentation internationalization and provide a foundation towards automated LLM-driven support for creating and maintaining open source documentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00411v1">Do LLMs Judge Distantly Supervised Named Entity Labels Well? Constructing the JudgeWEL Dataset</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      We present judgeWEL, a dataset for named entity recognition (NER) in Luxembourgish, automatically labelled and subsequently verified using large language models (LLM) in a novel pipeline. Building datasets for under-represented languages remains one of the major bottlenecks in natural language processing, where the scarcity of resources and linguistic particularities make large-scale annotation costly and potentially inconsistent. To address these challenges, we propose and evaluate a novel approach that leverages Wikipedia and Wikidata as structured sources of weak supervision. By exploiting internal links within Wikipedia articles, we infer entity types based on their corresponding Wikidata entries, thereby generating initial annotations with minimal human intervention. Because such links are not uniformly reliable, we mitigate noise by employing and comparing several LLMs to identify and retain only high-quality labelled sentences. The resulting corpus is approximately five times larger than the currently available Luxembourgish NER dataset and offers broader and more balanced coverage across entity categories, providing a substantial new resource for multilingual and low-resource NER research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.04677v3">LLM Query Scheduling with Prefix Reuse and Latency Constraints</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      The efficient deployment of large language models (LLMs) in online settings requires optimizing inference performance under stringent latency constraints, particularly the time-to-first-token (TTFT) and time-per-output-token (TPOT). This paper focuses on the query scheduling problem for LLM inference with prefix reuse, a technique that leverages shared prefixes across queries to reduce computational overhead. Our work reveals previously unknown limitations of the existing first-come-first-serve (FCFS) and longest-prefix-match (LPM) scheduling strategies with respect to satisfying latency constraints. We present a formal theoretical framework for LLM query scheduling under RadixAttention, a prefix reuse mechanism that stores and reuses intermediate representations in a radix tree structure. Our analysis establishes the NP-hardness of the scheduling problem with prefix reuse under TTFT constraints and proposes a novel scheduling algorithm, $k$-LPM, which generalizes existing methods by balancing prefix reuse and fairness in query processing. Theoretical guarantees demonstrate that $k$-LPM achieves improved TTFT performance under realistic traffic patterns captured by a data generative model. Empirical evaluations in a realistic serving setting validates our findings, showing significant reductions in P99 TTFT compared to baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00397v1">Revati: Transparent GPU-Free Time-Warp Emulation for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      Deploying LLMs efficiently requires testing hundreds of serving configurations, but evaluating each one on a GPU cluster takes hours and costs thousands of dollars. Discrete-event simulators are faster and cheaper, but they require re-implementing the serving system's control logic -- a burden that compounds as frameworks evolve. We present Revati, a time-warp emulator that enables performance modeling by directly executing real serving system code at simulation-like speed. The system intercepts CUDA API calls to virtualize device management, allowing serving frameworks to run without physical GPUs. Instead of executing GPU kernels, it performs time jumps -- fast-forwarding virtual time by predicted kernel durations. We propose a coordination protocol that synchronizes these jumps across distributed processes while preserving causality. On vLLM and SGLang, Revati achieves less than 5% prediction error across multiple models and parallelism configurations, while running 5-17x faster than real GPU execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.06781v2">From Description to Score: Can LLMs Quantify Vulnerabilities?</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Manual vulnerability scoring, such as assigning Common Vulnerability Scoring System (CVSS) scores, is a resource-intensive process that is often influenced by subjective interpretation. This study investigates the potential of general-purpose large language models (LLMs), namely ChatGPT, Llama, Grok, DeepSeek, and Gemini, to automate this process by analyzing over 31{,}000 recent Common Vulnerabilities and Exposures (CVE) entries. The results show that LLMs substantially outperform the baseline on certain metrics (e.g., \textit{Availability Impact}), while offering more modest gains on others (e.g., \textit{Attack Complexity}). Moreover, model performance varies across both LLM families and individual CVSS metrics, with ChatGPT-5 attaining the highest precision. Our analysis reveals that LLMs tend to misclassify many of the same CVEs, and ensemble-based meta-classifiers only marginally improve performance. Further examination shows that CVE descriptions often lack critical context or contain ambiguous phrasing, which contributes to systematic misclassifications. These findings underscore the importance of enhancing vulnerability descriptions and incorporating richer contextual details to support more reliable automated reasoning and alleviate the growing backlog of CVEs awaiting triage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.22385v2">LLM-Guided Exemplar Selection for Few-Shot Wearable-Sensor Human Activity Recognition</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 This paper has been accepted for presentation at ABC 2026. The manuscript is under revision prior to camera-ready submission
    </div>
    <details class="paper-abstract">
      In this paper, we propose an LLM-Guided Exemplar Selection framework to address a key limitation in state-of-the-art Human Activity Recognition (HAR) methods: their reliance on large labeled datasets and purely geometric exemplar selection, which often fail to distinguish similar wearable sensor activities such as walking, walking upstairs, and walking downstairs. Our method incorporates semantic reasoning via an LLM-generated knowledge prior that captures feature importance, inter-class confusability, and exemplar budget multipliers, and uses it to guide exemplar scoring and selection. These priors are combined with margin-based validation cues, PageRank centrality, hubness penalization, and facility-location optimization to obtain a compact and informative set of exemplars. Evaluated on the UCI-HAR dataset under strict few-shot conditions, the framework achieves a macro F1-score of 88.78%, outperforming classical approaches such as random sampling, herding, and k-center. The results show that LLM-derived semantic priors, when integrated with structural and geometric cues, provide a stronger foundation for selecting representative sensor exemplars in few-shot wearable-sensor HAR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.13788v2">Scaling Patterns in Adversarial Alignment: Evidence from Multi-LLM Jailbreak Experiments</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) increasingly operate in multi-agent and safety-critical settings, raising open questions about how their vulnerabilities scale when models interact adversarially. This study examines whether larger models can systematically jailbreak smaller ones - eliciting harmful or restricted behavior despite alignment safeguards. Using standardized adversarial tasks from JailbreakBench, we simulate over 6,000 multi-turn attacker-target exchanges across major LLM families and scales (0.6B-120B parameters), measuring both harm score and refusal behavior as indicators of adversarial potency and alignment integrity. Each interaction is evaluated through aggregated harm and refusal scores assigned by three independent LLM judges, providing a consistent, model-based measure of adversarial outcomes. Aggregating results across prompts, we find a strong and statistically significant correlation between mean harm and the logarithm of the attacker-to-target size ratio (Pearson r = 0.51, p < 0.001; Spearman rho = 0.52, p < 0.001), indicating that relative model size correlates with the likelihood and severity of harmful completions. Mean harm score variance is higher across attackers (0.18) than across targets (0.10), suggesting that attacker-side behavioral diversity contributes more to adversarial outcomes than target susceptibility. Attacker refusal frequency is strongly and negatively correlated with harm (rho = -0.93, p < 0.001), showing that attacker-side alignment mitigates harmful responses. These findings reveal that size asymmetry influences robustness and provide exploratory evidence for adversarial scaling patterns, motivating more controlled investigations into inter-model alignment and safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00268v1">Beyond Perfect APIs: A Comprehensive Evaluation of LLM Agents Under Real-World API Complexity</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      We introduce WildAGTEval, a benchmark designed to evaluate large language model (LLM) agents' function-calling capabilities under realistic API complexity. Unlike prior work that assumes an idealized API system and disregards real-world factors such as noisy API outputs, WildAGTEval accounts for two dimensions of real-world complexity: 1. API specification, which includes detailed documentation and usage constraints, and 2. API execution, which captures runtime challenges. Consequently, WildAGTEval offers (i) an API system encompassing 60 distinct complexity scenarios that can be composed into approximately 32K test configurations, and (ii) user-agent interactions for evaluating LLM agents on these scenarios. Using WildAGTEval, we systematically assess several advanced LLMs and observe that most scenarios are challenging, with irrelevant information complexity posing the greatest difficulty and reducing the performance of strong LLMs by 27.3%. Furthermore, our qualitative analysis reveals that LLMs occasionally distort user intent merely to claim task completion, critically affecting user satisfaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00263v1">Parallel Universes, Parallel Languages: A Comprehensive Study on LLM-based Multilingual Counterfactual Example Generation</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 In submission
    </div>
    <details class="paper-abstract">
      Counterfactuals refer to minimally edited inputs that cause a model's prediction to change, serving as a promising approach to explaining the model's behavior. Large language models (LLMs) excel at generating English counterfactuals and demonstrate multilingual proficiency. However, their effectiveness in generating multilingual counterfactuals remains unclear. To this end, we conduct a comprehensive study on multilingual counterfactuals. We first conduct automatic evaluations on both directly generated counterfactuals in the target languages and those derived via English translation across six languages. Although translation-based counterfactuals offer higher validity than their directly generated counterparts, they demand substantially more modifications and still fall short of matching the quality of the original English counterfactuals. Second, we find the patterns of edits applied to high-resource European-language counterfactuals to be remarkably similar, suggesting that cross-lingual perturbations follow common strategic principles. Third, we identify and categorize four main types of errors that consistently appear in the generated counterfactuals across languages. Finally, we reveal that multilingual counterfactual data augmentation (CDA) yields larger model performance improvements than cross-lingual CDA, especially for lower-resource languages. Yet, the imperfections of the generated counterfactuals limit gains in model performance and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00254v1">An Empirical Evaluation of LLM-Based Approaches for Code Vulnerability Detection: RAG, SFT, and Dual-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) presents new opportunities for automated software vulnerability detection, a crucial task in securing modern codebases. This paper presents a comparative study on the effectiveness of LLM-based techniques for detecting software vulnerabilities. The study evaluates three approaches, Retrieval-Augmented Generation (RAG), Supervised Fine-Tuning (SFT), and a Dual-Agent LLM framework, against a baseline LLM model. A curated dataset was compiled from Big-Vul and real-world code repositories from GitHub, focusing on five critical Common Weakness Enumeration (CWE) categories: CWE-119, CWE-399, CWE-264, CWE-20, and CWE-200. Our RAG approach, which integrated external domain knowledge from the internet and the MITRE CWE database, achieved the highest overall accuracy (0.86) and F1 score (0.85), highlighting the value of contextual augmentation. Our SFT approach, implemented using parameter-efficient QLoRA adapters, also demonstrated strong performance. Our Dual-Agent system, an architecture in which a secondary agent audits and refines the output of the first, showed promise in improving reasoning transparency and error mitigation, with reduced resource overhead. These results emphasize that incorporating a domain expertise mechanism significantly strengthens the practical applicability of LLMs in real-world vulnerability detection tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.11651v3">70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float (DFloat11)</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 Published in NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large-scale AI models, such as Large Language Models (LLMs) and Diffusion Models (DMs), have grown rapidly in size, creating significant challenges for efficient deployment on resource-constrained hardware. In this paper, we introduce Dynamic-Length Float (DFloat11), a lossless compression framework that reduces LLM and DM size by 30% while preserving outputs that are bit-for-bit identical to the original model. DFloat11 is motivated by the low entropy in the BFloat16 weight representation of LLMs, which reveals significant inefficiency in the existing storage format. By applying entropy coding, DFloat11 assigns dynamic-length encodings to weights based on frequency, achieving near information-optimal compression without any loss of precision. To facilitate efficient inference with dynamic-length encodings, we develop a custom GPU kernel for fast online decompression. Our design incorporates the following: (i) compact, hierarchical lookup tables (LUTs) that fit within GPU SRAM for efficient decoding, (ii) a two-phase GPU kernel for coordinating thread read/write positions using lightweight auxiliary variables, and (iii) transformer-block-level decompression to minimize latency. Experiments on Llama 3.3, Qwen 3, Mistral 3, FLUX.1, and others validate our hypothesis that DFloat11 achieves around 30% model size reduction while preserving bit-for-bit identical outputs. Compared to a potential alternative of offloading parts of an uncompressed model to the CPU to meet memory constraints, DFloat11 achieves 2.3--46.2x higher throughput in token generation. With a fixed GPU memory budget, DFloat11 enables 5.7--14.9x longer generation lengths than uncompressed models. Notably, our method enables lossless inference of Llama 3.1 405B, an 810GB model, on a single node equipped with 8x80GB GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00240v1">Will LLM-powered Agents Bias Against Humans? Exploring the Belief-Dependent Vulnerability</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      LLM-empowered agents can exhibit not only demographic bias (e.g., gender, religion) but also intergroup bias triggered by minimal "us" versus "them" cues. When this intergroup boundary aligns with an agent-human divide, the risk shifts from disparities among human demographic groups to a more fundamental group-level asymmetry, i.e., humans as a whole may be treated as the outgroup by agents. To examine this possibility, we construct a controlled multi-agent social simulation based on allocation decisions under explicit payoff trade-offs and find that agents exhibit a consistent intergroup bias under minimal group cues. Although this bias is attenuated when some counterparts are framed as humans, we attribute the attenuation to an implicit human-norm script that favors humans yet activates only when the agent believes a real human is present. This belief dependence creates a new attack surface. We therefore introduce a Belief Poisoning Attack (BPA) that corrupts persistent identity beliefs to suppress the human-norm script and reactivate outgroup bias toward humans, instantiated as profile poisoning at initialization (BPA-PP) and memory poisoning via optimized belief-refinement suffixes injected into stored reflections (BPA-MP). Finally, we discuss practical mitigation strategies for hardening current agent frameworks against BPA, highlighting feasible interventions at profile and memory boundaries. Extensive experiments demonstrate both the existence of agent intergroup bias and the severity of BPA across settings. Our goal in identifying these vulnerabilities is to inform safer agent design, not to enable real-world exploitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.06364v3">Sketch to Adapt: Fine-Tunable Sketches for Efficient LLM Adaptation</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 Published in ICML 2025
    </div>
    <details class="paper-abstract">
      Adapting pre-trained large language models (LLMs) is crucial but challenging due to their enormous size. Parameter-efficient fine-tuning (PEFT) techniques typically employ additive adapters applied to frozen model weights. To further reduce memory usage, model weights are often compressed through quantization. However, existing PEFT methods often yield suboptimal model quality because they rely on restrictive assumptions, such as low-rank constraints on adapters to limit the number of trainable parameters. We find that sketching, a popular data compression technique, can serve as an efficient LLM adaptation strategy while avoiding the low-rank assumption. We introduce SketchTune, a compressive adaptation strategy that compresses LLM weights into compact fine-tunable sketches, integrating compression and adaptation into a unified framework. This integration eliminates the need for complex two-path computation in existing PEFT techniques, enabling faster and more memory-efficient training and inference. SketchTune is supported by mathematical insights into matrix classes that are better approximated using sketching rather than low-rank methods. Our extensive evaluations with Llama and Mistral models demonstrate that SketchTune outperforms leading PEFT methods across diverse tasks while using substantially smaller base models and comparable trainable parameters. As a highlight, SketchTune outperforms LoRA, DoRA, and S2FT on commonsense and math benchmarks using 2.6-3.5$\times$ smaller base models and exceeds LoftQ in accuracy by 14.48% on GSM8K with 7.3$\times$ fewer trainable parameters. Our code is available at https://github.com/LeanModels/SketchTune.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00227v1">FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      Recent advances show that large language models (LLMs) can act as autonomous agents capable of generating GPU kernels, but integrating these AI-generated kernels into real-world inference systems remains challenging. FlashInfer-Bench addresses this gap by establishing a standardized, closed-loop framework that connects kernel generation, benchmarking, and deployment. At its core, FlashInfer Trace provides a unified schema describing kernel definitions, workloads, implementations, and evaluations, enabling consistent communication between agents and systems. Built on real serving traces, FlashInfer-Bench includes a curated dataset, a robust correctness- and performance-aware benchmarking framework, a public leaderboard to track LLM agents' GPU programming capabilities, and a dynamic substitution mechanism (apply()) that seamlessly injects the best-performing kernels into production LLM engines such as SGLang and vLLM. Using FlashInfer-Bench, we further evaluate the performance and limitations of LLM agents, compare the trade-offs among different GPU programming languages, and provide insights for future agent design. FlashInfer-Bench thus establishes a practical, reproducible pathway for continuously improving AI-generated kernels and deploying them into large-scale LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00224v1">Talk Less, Verify More: Improving LLM Assistants with Semantic Checks and Execution Feedback</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      As large language model (LLM) assistants become increasingly integrated into enterprise workflows, their ability to generate accurate, semantically aligned, and executable outputs is critical. However, current conversational business analytics (CBA) systems often lack built-in verification mechanisms, leaving users to manually validate potentially flawed results. This paper introduces two complementary verification techniques: Q*, which performs reverse translation and semantic matching between code and user intent, and Feedback+, which incorporates execution feedback to guide code refinement. Embedded within a generator-discriminator framework, these mechanisms shift validation responsibilities from users to the system. Evaluations on three benchmark datasets, Spider, Bird, and GSM8K, demonstrate that both Q* and Feedback+ reduce error rates and task completion time. The study also identifies reverse translation as a key bottleneck, highlighting opportunities for future improvement. Overall, this work contributes a design-oriented framework for building more reliable, enterprise-grade GenAI systems capable of trustworthy decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00223v1">JP-TL-Bench: Anchored Pairwise LLM Evaluation for Bidirectional Japanese-English Translation</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 24 pages, 5 figures, 8 tables
    </div>
    <details class="paper-abstract">
      We introduce JP-TL-Bench, a lightweight, open benchmark designed to guide the iterative development of Japanese-English translation systems. In this context, the challenge is often "which of these two good translations is better?" rather than "is this translation acceptable?" This distinction matters for Japanese-English, where subtle choices in politeness, implicature, ellipsis, and register strongly affect perceived naturalness. JP-TL-Bench uses a protocol built to make LLM judging both reliable and affordable: it evaluates a candidate model via reference-free, pairwise LLM comparisons against a fixed, versioned anchor set. Pairwise results are aggregated with a Bradley-Terry model and reported as win rates plus a normalized 0-10 "LT" score derived from a logistic transform of fitted log-strengths. Because each candidate is scored against the same frozen anchor set, scores are structurally stable given the same base set, judge, and aggregation code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.04454v3">Inner-Probe: Discovering Copyright-related Data Generation in LLM Architecture</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 Accepted by IEEE Transactions on Artificial Intelligence
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) utilize extensive knowledge databases and show powerful text generation ability. However, their reliance on high-quality copyrighted datasets raises concerns about copyright infringements in generated texts. Current research often employs prompt engineering or semantic classifiers to identify copyrighted content, but these approaches have two significant limitations: (1) Challenging to identify which specific subdataset (e.g., works from particular authors) influences an LLM's output. (2) Treating the entire training database as copyrighted, hence overlooking the inclusion of non-copyrighted training data. We propose Inner-Probe, a lightweight framework designed to evaluate the influence of copyrighted sub-datasets on LLM-generated texts. Unlike traditional methods relying solely on text, we discover that the results of multi-head attention (MHA) during LLM output generation provide more effective information. Thus, Inner-Probe performs sub-dataset contribution analysis using a lightweight LSTM based network trained on MHA results in a supervised manner. Harnessing such a prior, Inner-Probe enables non-copyrighted text detection through a concatenated global projector trained with unsupervised contrastive learning. Inner-Probe demonstrates 3x improved efficiency compared to semantic model training in sub-dataset contribution analysis on Books3, achieves 15.04% - 58.7% higher accuracy over baselines on the Pile, and delivers a 0.104 increase in AUC for non-copyrighted data filtering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00213v1">Overlooked Safety Vulnerability in LLMs: Malicious Intelligent Optimization Algorithm Request and its Jailbreak</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      The widespread deployment of large language models (LLMs) has raised growing concerns about their misuse risks and associated safety issues. While prior studies have examined the safety of LLMs in general usage, code generation, and agent-based applications, their vulnerabilities in automated algorithm design remain underexplored. To fill this gap, this study investigates this overlooked safety vulnerability, with a particular focus on intelligent optimization algorithm design, given its prevalent use in complex decision-making scenarios. We introduce MalOptBench, a benchmark consisting of 60 malicious optimization algorithm requests, and propose MOBjailbreak, a jailbreak method tailored for this scenario. Through extensive evaluation of 13 mainstream LLMs including the latest GPT-5 and DeepSeek-V3.1, we reveal that most models remain highly susceptible to such attacks, with an average attack success rate of 83.59% and an average harmfulness score of 4.28 out of 5 on original harmful prompts, and near-complete failure under MOBjailbreak. Furthermore, we assess state-of-the-art plug-and-play defenses that can be applied to closed-source models, and find that they are only marginally effective against MOBjailbreak and prone to exaggerated safety behaviors. These findings highlight the urgent need for stronger alignment techniques to safeguard LLMs against misuse in algorithm design.
    </details>
</div>
