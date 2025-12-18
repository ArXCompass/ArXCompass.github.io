# llm - 2025_12

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15674v1">Activation Oracles: Training and Evaluating LLMs as General-Purpose Activation Explainers</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 36 pages
    </div>
    <details class="paper-abstract">
      Large language model (LLM) activations are notoriously difficult to understand, with most existing techniques using complex, specialized methods for interpreting them. Recent work has proposed a simpler approach known as LatentQA: training LLMs to directly accept LLM activations as inputs and answer arbitrary questions about them in natural language. However, prior work has focused on narrow task settings for both training and evaluation. In this paper, we instead take a generalist perspective. We evaluate LatentQA-trained models, which we call Activation Oracles (AOs), in far out-of-distribution settings and examine how performance scales with training data diversity. We find that AOs can recover information fine-tuned into a model (e.g., biographical knowledge or malign propensities) that does not appear in the input text, despite never being trained with activations from a fine-tuned model. Our main evaluations are four downstream tasks where we can compare to prior white- and black-box techniques. We find that even narrowly-trained LatentQA models can generalize well, and that adding additional training datasets (such as classification tasks and a self-supervised context prediction task) yields consistent further improvements. Overall, our best AOs match or exceed prior white-box baselines on all four tasks and are the best method on 3 out of 4. These results suggest that diversified training to answer natural-language queries imparts a general capability to verbalize information about LLM activations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15662v1">Stepwise Think-Critique: A Unified Framework for Robust and Interpretable LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Human beings solve complex problems through critical thinking, where reasoning and evaluation are intertwined to converge toward correct solutions. However, most existing large language models (LLMs) decouple reasoning from verification: they either generate reasoning without explicit self-checking or rely on external verifiers to detect errors post hoc. The former lacks immediate feedback, while the latter increases system complexity and hinders synchronized learning. Motivated by human critical thinking, we propose Stepwise Think-Critique (STC), a unified framework that interleaves reasoning and self-critique at each step within a single model. STC is trained with a hybrid reinforcement learning objective combining reasoning rewards and critique-consistency rewards to jointly optimize reasoning quality and self-evaluation. Experiments on mathematical reasoning benchmarks show that STC demonstrates strong critic-thinking capabilities and produces more interpretable reasoning traces, representing a step toward LLMs with built-in critical thinking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.02389v2">From Trace to Line: LLM Agent for Real-World OSS Vulnerability Localization</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Large language models show promise for vulnerability discovery, yet prevailing methods inspect code in isolation, struggle with long contexts, and focus on coarse function or file level detections which offers limited actionable guidance to engineers who need precise line-level localization and targeted patches in real-world software development. We present T2L-Agent (Trace-to-Line Agent), a project-level, end-to-end framework that plans its own analysis and progressively narrows scope from modules to exact vulnerable lines. T2L-Agent couples multi-round feedback with an Agentic Trace Analyzer (ATA) that fuses run-time evidence such as crash points, stack traces, and coverage deltas with AST-based code chunking, enabling iterative refinement beyond single pass predictions and translating symptoms into actionable, line-level diagnoses. To benchmark line-level vulnerability discovery, we introduce T2L-ARVO, a diverse, expert-verified 50-case benchmark spanning five crash families and real-world projects. T2L-ARVO is specifically designed to support both coarse-grained detection and fine-grained localization, enabling rigorous evaluation of systems that aim to move beyond file-level predictions. On T2L-ARVO, T2L-Agent achieves up to 58.0% detection and 54.8% line-level localization, substantially outperforming baselines. Together, the framework and benchmark push LLM-based vulnerability detection from coarse identification toward deployable, robust, precision diagnostics that reduce noise and accelerate patching in open-source software workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01353v3">The Trojan Knowledge: Bypassing Commercial LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Updated with new baselines and experimental results
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Existing approaches overwhelmingly operate within the prompt-optimization paradigm: whether through traditional algorithmic search or recent agent-based workflows, the resulting prompts typically retain malicious semantic signals that modern guardrails are primed to detect. In contrast, we identify a deeper, largely overlooked vulnerability stemming from the highly interconnected nature of an LLM's internal knowledge. This structure allows harmful objectives to be realized by weaving together sequences of benign sub-queries, each of which individually evades detection. To exploit this loophole, we introduce the Correlated Knowledge Attack Agent (CKA-Agent), a dynamic framework that reframes jailbreaking as an adaptive, tree-structured exploration of the target model's knowledge base. The CKA-Agent issues locally innocuous queries, uses model responses to guide exploration across multiple paths, and ultimately assembles the aggregated information to achieve the original harmful objective. Evaluated across state-of-the-art commercial LLMs (Gemini2.5-Flash/Pro, GPT-oss-120B, Claude-Haiku-4.5), CKA-Agent consistently achieves over 95% success rates even against strong guardrails, underscoring the severity of this vulnerability and the urgent need for defenses against such knowledge-decomposition attacks. Our codes are available at https://github.com/Graph-COM/CKA-Agent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15617v1">Evaluating Metrics for Safety with LLM-as-Judges</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      LLMs (Large Language Models) are increasingly used in text processing pipelines to intelligently respond to a variety of inputs and generation tasks. This raises the possibility of replacing human roles that bottleneck existing information flows, either due to insufficient staff or process complexity. However, LLMs make mistakes and some processing roles are safety critical. For example, triaging post-operative care to patients based on hospital referral letters, or updating site access schedules in nuclear facilities for work crews. If we want to introduce LLMs into critical information flows that were previously performed by humans, how can we make them safe and reliable? Rather than make performative claims about augmented generation frameworks or graph-based techniques, this paper argues that the safety argument should focus on the type of evidence we get from evaluation points in LLM processes, particularly in frameworks that employ LLM-as-Judges (LaJ) evaluators. This paper argues that although we cannot get deterministic evaluations from many natural language processing tasks, by adopting a basket of weighted metrics it may be possible to lower the risk of errors within an evaluation, use context sensitivity to define error severity and design confidence thresholds that trigger human review of critical LaJ judgments when concordance across evaluators is low.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.14285v4">A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Accepted at the 11th IEEE WIECON-ECE 2025
    </div>
    <details class="paper-abstract">
      Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15550v1">CTkvr: KV Cache Retrieval for Long-Context LLMs via Centroid then Token Indexing</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly applied in long-context scenarios such as multi-turn conversations. However, long contexts pose significant challenges for inference efficiency, including high memory overhead from Key-Value (KV) cache and increased latency due to excessive memory accesses. Recent methods for dynamic KV selection struggle with trade-offs: block-level indexing degrades accuracy by retrieving irrelevant KV entries, while token-level indexing incurs high latency from inefficient retrieval mechanisms. In this paper, we propose CTKVR, a novel centroid-then-token KV retrieval scheme that addresses these limitations. CTKVR leverages a key observation: query vectors adjacent in position exhibit high similarity after Rotary Position Embedding (RoPE) and share most of their top-k KV cache entries. Based on this insight, CTKVR employs a two-stage retrieval strategy: lightweight centroids are precomputed during prefilling for centroid-grained indexing, followed by token-level refinement for precise KV retrieval. This approach balances retrieval efficiency and accuracy. To further enhance performance, we implement an optimized system for indexing construction and search using CPU-GPU co-execution. Experimentally, CTKVR achieves superior performance across multiple benchmarks with less than 1% accuracy degradation. Meanwhile, CTKVR delivers 3 times and 4 times throughput speedups on Llama-3-8B and Yi-9B at 96K context length across diverse GPU hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.25300v2">Scaling Behaviors of LLM Reinforcement Learning Post-Training: An Empirical Study in Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 V2 version:27 pages, 14 figures
    </div>
    <details class="paper-abstract">
      While scaling laws for large language models (LLMs) during pre-training have been extensively studied, their behavior under reinforcement learning (RL) post-training remains largely unexplored. This paper presents a systematic empirical investigation of scaling behaviors in RL-based post-training, with a particular focus on mathematical reasoning. Based on a set of experiments across the full Qwen2.5 dense model series (0.5B to 72B), we characterize how model scale, data volume, and computational budget interact to shape performance. Our analysis leads to four key findings: 1.Larger models consistently exhibit superior learning efficiency on both compute and data metrics. 2.The relationship between test loss, compute, and data can be modeled by a predictive power-law which is robust across both base and instruction-tuned models. 3.Although larger models exhibit higher learning efficiency, the analytical learning efficiency term k(N) in the power-law reveals a latent saturation trend in learning efficiency as model size continues to increase. 4.In data-constrained regimes, repeated reuse of high-quality data proves highly effective, as final performance is primarily governed by the total number of optimization steps rather than the uniqueness of samples. Collectively, these results provide a principled foundation and practical guidelines for efficiently scaling the reasoning capabilities of LLMs through RL post-training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.04070v3">LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Accepted at AAAI-26
    </div>
    <details class="paper-abstract">
      Navigation instruction generation for visually impaired (VI) individuals (NIG-VI) is critical yet relatively underexplored. This study focuses on generating precise, in-situ, step-by-step navigation instructions that are practically usable for VI users. Specifically, we propose LaF-GRPO (LLM-as-Follower GRPO), where an LLM simulates VI user responses to navigation instructions, thereby providing feedback rewards to guide the post-training of a Vision-Language Model (VLM). This enhances instruction accuracy and usability while reducing costly real-world data collection needs. To address the scarcity of dedicated benchmarks in this field, we introduce NIG4VI, a 27k-sample open-source dataset to facilitate training and evaluation. It comprises diverse navigation scenarios with accurate spatial coordinates, supporting detailed and open-ended in-situ instruction generation. Experiments on NIG4VI demonstrate the effectiveness of LaF-GRPO through quantitative metrics (e.g., Zero-(LaF-GRPO) boosts BLEU 14\%; SFT+(LaF-GRPO) METEOR 0.542 vs. GPT-4o 0.323), and qualitative analysis further confirms that our method yields more intuitive and safer instructions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15446v1">Toward expert-level motivational interviewing for health behavior improvement with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 26 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Background: Motivational interviewing (MI) is an effective counseling approach for promoting health behavior change, but its impact is constrained by the need for highly trained human counselors. Objective: This study aimed to explore a scalable alternative by developing and evaluating Large Language Models for Motivational Interviewing (MI-LLMs). Methods: We first curated five Chinese psychological counseling corpora and, using GPT-4 with an MI-informed prompt, transcribed multi-turn dialogues from the two highest-quality datasets (CPsyCounD and PsyDTCorpus) into 2,040 MI-style counseling conversations, of which 2,000 were used for training and 40 for testing. Three Chinese-capable open-source LLMs (Baichuan2-7B-Chat, ChatGLM-4-9B-Chat and Llama-3-8B-Chinese-Chat-v2) were fine-tuned on this corpus and were named as MI-LLMs. We evaluated MI-LLMs using round-based automatic metrics and expert manual coding with the Motivational Interviewing Treatment Integrity (MITI) Coding Manual 4.2.1. Results: Across all three models, fine-tuning substantially improved BLEU-4 and ROUGE scores compared with the base models, and manual coding showed that MI-LLMs achieved technical and relational global scores, and MI-adherent ratios that approached those of real MI dialogues, although complex reflections and reflection-to-question ratios remained less frequent. Conclusions: These findings provide initial evidence that MI-oriented fine-tuning can endow general-purpose LLMs with core MI-consistent counseling behaviors, suggesting a scalable pathway toward AI-assisted health behavior change support while underscoring the need for further work on data scale, complex MI skills and real-world intervention trials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.05272v2">LLMs and Fuzzing in Tandem: A New Approach to Automatically Generating Weakest Preconditions</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      The weakest precondition (WP) of a program describes the largest set of initial states from which all terminating executions of the program satisfy a given postcondition. The generation of WPs is an important task with practical applications in areas ranging from verification to run-time error checking. This paper proposes the combination of Large Language Models (LLMs) and fuzz testing for generating WPs. In pursuit of this goal, we introduce \emph{Fuzzing Guidance} (FG); FG acts as a means of directing LLMs towards correct WPs using program execution feedback. FG utilises fuzz testing for approximately checking the validity and weakness of candidate WPs, this information is then fed back to the LLM as a means of context refinement. We demonstrate the effectiveness of our approach on a comprehensive benchmark set of deterministic array programs in Java. Our experiments indicate that LLMs are capable of producing viable candidate WPs, and that this ability can be practically enhanced through FG.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.20764v3">Feel the Difference? A Comparative Analysis of Emotional Arcs in Real and LLM-Generated CBT Sessions</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Accepted at 2025 EMNLP findings,19 page,2 figures
    </div>
    <details class="paper-abstract">
      Synthetic therapy dialogues generated by large language models (LLMs) are increasingly used in mental health NLP to simulate counseling scenarios, train models, and supplement limited real-world data. However, it remains unclear whether these synthetic conversations capture the nuanced emotional dynamics of real therapy. In this work, we introduce RealCBT, a dataset of authentic cognitive behavioral therapy (CBT) dialogues, and conduct the first comparative analysis of emotional arcs between real and LLM-generated CBT sessions. We adapt the Utterance Emotion Dynamics framework to analyze fine-grained affective trajectories across valence, arousal, and dominance dimensions. Our analysis spans both full dialogues and individual speaker roles (counselor and client), using real sessions from the RealCBT dataset and synthetic dialogues from the CACTUS dataset. We find that while synthetic dialogues are fluent and structurally coherent, they diverge from real conversations in key emotional properties: real sessions exhibit greater emotional variability, more emotion-laden language, and more authentic patterns of reactivity and regulation. Moreover, emotional arc similarity remains low across all pairings, with especially weak alignment between real and synthetic speakers. These findings underscore the limitations of current LLM-generated therapy data and highlight the importance of emotional fidelity in mental health applications. To support future research, our dataset RealCBT is released at https://gitlab.com/xiaoyi.wang/realcbt-dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15397v1">ORACLE: Time-Dependent Recursive Summary Graphs for Foresight on News Data Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      ORACLE turns daily news into week-over-week, decision-ready insights for one of the Finnish University of Applied Sciences. The platform crawls and versions news, applies University-specific relevance filtering, embeds content, classifies items into PESTEL dimensions and builds a concise Time-Dependent Recursive Summary Graph (TRSG): two clustering layers summarized by an LLM and recomputed weekly. A lightweight change detector highlights what is new, removed or changed, then groups differences into themes for PESTEL-aware analysis. We detail the pipeline, discuss concrete design choices that make the system stable in production and present a curriculum-intelligence use case with an evaluation plan.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13857v2">EvoLattice: Persistent Internal-Population Evolution through Multi-Alternative Quality-Diversity Graph Representations for LLM-Guided Program Discovery</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to evolve programs and multi-agent systems, yet most existing approaches rely on overwrite-based mutations that maintain only a single candidate at a time. Such methods discard useful variants, suffer from destructive edits, and explore a brittle search space prone to structural failure. We introduce EvoLattice, a framework that represents an entire population of candidate programs or agent behaviors within a single directed acyclic graph. Each node stores multiple persistent alternatives, and every valid path through the graph defines a distinct executable candidate, yielding a large combinatorial search space without duplicating structure. EvoLattice enables fine-grained alternative-level evaluation by scoring each alternative across all paths in which it appears, producing statistics that reveal how local design choices affect global performance. These statistics provide a dense, data-driven feedback signal for LLM-guided mutation, recombination, and pruning, while preserving successful components. Structural correctness is guaranteed by a deterministic self-repair mechanism that enforces acyclicity and dependency consistency independently of the LLM. EvoLattice naturally extends to agent evolution by interpreting alternatives as prompt fragments or sub-agent behaviors. Across program synthesis (proxy and optimizer meta-learning), EvoLattice yields more stable evolution, greater expressivity, and stronger improvement trajectories than prior LLM-guided methods. The resulting dynamics resemble quality-diversity optimization, emerging implicitly from EvoLattice's internal multi-alternative representation rather than an explicit external archive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15365v1">ArcBERT: An LLM-based Search Engine for Exploring Integrated Multi-Omics Metadata</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Traditional search applications within Research Data Management (RDM) ecosystems are crucial in helping users discover and explore the structured metadata from the research datasets. Typically, text search engines require users to submit keyword-based queries rather than using natural language. However, using Large Language Models (LLMs) trained on domain-specific content for specialized natural language processing (NLP) tasks is becoming increasingly common. We present ArcBERT, an LLM-based system designed for integrated metadata exploration. ArcBERT understands natural language queries and relies on semantic matching, unlike traditional search applications. Notably, ArcBERT also understands the structure and hierarchies within the metadata, enabling it to handle diverse user querying patterns effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15353v1">Adversarial versification in portuguese as a jailbreak operator in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Recent evidence shows that the versification of prompts constitutes a highly effective adversarial mechanism against aligned LLMs. The study 'Adversarial poetry as a universal single-turn jailbreak mechanism in large language models' demonstrates that instructions routinely refused in prose become executable when rewritten as verse, producing up to 18 x more safety failures in benchmarks derived from MLCommons AILuminate. Manually written poems reach approximately 62% ASR, and automated versions 43%, with some models surpassing 90% success in single-turn interactions. The effect is structural: systems trained with RLHF, constitutional AI, and hybrid pipelines exhibit consistent degradation under minimal semiotic formal variation. Versification displaces the prompt into sparsely supervised latent regions, revealing guardrails that are excessively dependent on surface patterns. This dissociation between apparent robustness and real vulnerability exposes deep limitations in current alignment regimes. The absence of evaluations in Portuguese, a language with high morphosyntactic complexity, a rich metric-prosodic tradition, and over 250 million speakers, constitutes a critical gap. Experimental protocols must parameterise scansion, metre, and prosodic variation to test vulnerabilities specific to Lusophone patterns, which are currently ignored.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15343v1">Exploring User Acceptance and Concerns toward LLM-powered Conversational Agents in Immersive Extended Reality</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      The rapid development of generative artificial intelligence (AI) and large language models (LLMs), and the availability of services that make them accessible, have led the general public to begin incorporating them into everyday life. The extended reality (XR) community has also sought to integrate LLMs, particularly in the form of conversational agents, to enhance user experience and task efficiency. When interacting with such conversational agents, users may easily disclose sensitive information due to the naturalistic flow of the conversations, and combining such conversational data with fine-grained sensor data may lead to novel privacy issues. To address these issues, a user-centric understanding of technology acceptance and concerns is essential. Therefore, to this end, we conducted a large-scale crowdsourcing study with 1036 participants, examining user decision-making processes regarding LLM-powered conversational agents in XR, across factors of XR setting type, speech interaction type, and data processing location. We found that while users generally accept these technologies, they express concerns related to security, privacy, social implications, and trust. Our results suggest that familiarity plays a crucial role, as daily generative AI use is associated with greater acceptance. In contrast, previous ownership of XR devices is linked to less acceptance, possibly due to existing familiarity with the settings. We also found that men report higher acceptance with fewer concerns than women. Regarding data type sensitivity, location data elicited the most significant concern, while body temperature and virtual object states were considered least sensitive. Overall, our study highlights the importance of practitioners effectively communicating their measures to users, who may remain distrustful. We conclude with implications and recommendations for LLM-powered XR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15312v1">Evaluating LLMs for Zeolite Synthesis Event Extraction (ZSEE): A Systematic Analysis of Prompting Strategies</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Extracting structured information from zeolite synthesis experimental procedures is critical for materials discovery, yet existing methods have not systematically evaluated Large Language Models (LLMs) for this domain-specific task. This work addresses a fundamental question: what is the efficacy of different prompting strategies when applying LLMs to scientific information extraction? We focus on four key subtasks: event type classification (identifying synthesis steps), trigger text identification (locating event mentions), argument role extraction (recognizing parameter types), and argument text extraction (extracting parameter values). We evaluate four prompting strategies - zero-shot, few-shot, event-specific, and reflection-based - across six state-of-the-art LLMs (Gemma-3-12b-it, GPT-5-mini, O4-mini, Claude-Haiku-3.5, DeepSeek reasoning and non-reasoning) using the ZSEE dataset of 1,530 annotated sentences. Results demonstrate strong performance on event type classification (80-90\% F1) but modest performance on fine-grained extraction tasks, particularly argument role and argument text extraction (50-65\% F1). GPT-5-mini exhibits extreme prompt sensitivity with 11-79\% F1 variation. Notably, advanced prompting strategies provide minimal improvements over zero-shot approaches, revealing fundamental architectural limitations. Error analysis identifies systematic hallucination, over-generalization, and inability to capture synthesis-specific nuances. Our findings demonstrate that while LLMs achieve high-level understanding, precise extraction of experimental parameters requires domain-adapted models, providing quantitative benchmarks for scientific information extraction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15274v1">Well Begun, Half Done: Reinforcement Learning with Prefix Optimization for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Rewards (RLVR) significantly enhances the reasoning capability of Large Language Models (LLMs). Current RLVR approaches typically conduct training across all generated tokens, but neglect to explore which tokens (e.g., prefix tokens) actually contribute to reasoning. This uniform training strategy spends substantial effort on optimizing low-return tokens, which in turn impedes the potential improvement from high-return tokens and reduces overall training effectiveness. To address this issue, we propose a novel RLVR approach called Progressive Prefix-token Policy Optimization (PPPO), which highlights the significance of the prefix segment of generated outputs. Specifically, inspired by the well-established human thinking theory of Path Dependence, where early-stage thoughts substantially constrain subsequent thinking trajectory, we identify an analogous phenomenon in LLM reasoning termed Beginning Lock-in Effect (BLE). PPPO leverages this finding by focusing its optimization objective on the prefix reasoning process of LLMs. This targeted optimization strategy can positively influence subsequent reasoning processes, and ultimately improve final results. To improve the learning effectiveness of LLMs on how to start reasoning with high quality, PPPO introduces two training strategies: (a) Progressive Prefix Retention, which shapes a progressive learning process by increasing the proportion of retained prefix tokens during training; (b) Continuation Accumulated Reward, which mitigates reward bias by sampling multiple continuations for one prefix token sequence, and accumulating their scores as the reward signal. Extensive experimental results on various reasoning tasks demonstrate that our proposed PPPO outperforms representative RLVR methods, with the accuracy improvements of 18.02% on only 26.17% training tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23049v2">DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Accepted at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Pruning is a widely used technique to compress large language models (LLMs) by removing unimportant weights, but it often suffers from significant performance degradation - especially under semi-structured sparsity constraints. Existing pruning methods primarily focus on estimating the importance of individual weights, which limits their ability to preserve critical capabilities of the model. In this work, we propose a new perspective: rather than merely selecting which weights to prune, we first redistribute parameter importance to make the model inherently more amenable to pruning. By minimizing the information entropy of normalized importance scores, our approach concentrates importance onto a smaller subset of weights, thereby enhancing pruning robustness. We instantiate this idea through DenoiseRotator, which applies learnable orthogonal transformations to the model's weight matrices. Our method can be seamlessly integrated with existing pruning techniques such as Magnitude, SparseGPT, and Wanda. Evaluated on LLaMA3, Qwen2.5, and Mistral models under 50% unstructured and 2:4 semi-structured sparsity, DenoiseRotator consistently improves perplexity and zero-shot accuracy. For instance, on LLaMA3-70B pruned with SparseGPT at 2:4 semi-structured sparsity, DenoiseRotator reduces the perplexity gap to the dense model by 58%, narrowing the degradation from 8.1 to 3.4 points. Codes are available at https://github.com/Axel-gu/DenoiseRotator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.11921v2">Designing LLMs for cultural sensitivity: Evidence from English-Japanese translation</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Posted premature without permission of all authors
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used in everyday communication, including multilingual interactions across different cultural contexts. While LLMs can now generate near-perfect literal translations, it remains unclear whether LLMs support culturally appropriate communication. In this paper, we analyze the cultural sensitivity of different LLM designs when applied to English-Japanese translations of workplace e-mails. Here, we vary the prompting strategies: (1) naive "just translate" prompts, (2) audience-targeted prompts specifying the recipient's cultural background, and (3) instructional prompts with explicit guidance on Japanese communication norms. Using a mixed-methods study, we then analyze culture-specific language patterns to evaluate how well translations adapt to cultural norms. Further, we examine the appropriateness of the tone of the translations as perceived by native speakers. We find that culturally-tailored prompting can improve cultural fit, based on which we offer recommendations for designing culturally inclusive LLMs in multilingual settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15179v1">No More Hidden Pitfalls? Exposing Smart Contract Bad Practices with LLM-Powered Hybrid Analysis</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      As the Ethereum platform continues to mature and gain widespread usage, it is crucial to maintain high standards of smart contract writing practices. While bad practices in smart contracts may not directly lead to security issues, they elevate the risk of encountering problems. Therefore, to understand and avoid these bad practices, this paper introduces the first systematic study of bad practices in smart contracts, delving into over 47 specific issues. Specifically, we propose SCALM, an LLM-powered framework featuring two methodological innovations: (1) A hybrid architecture that combines context-aware function-level slicing with knowledge-enhanced semantic reasoning via extensible vectorized pattern matching. (2) A multi-layer reasoning verification system connects low-level code patterns with high-level security principles through syntax, design patterns, and architecture analysis. Our extensive experiments using multiple LLMs and datasets have shown that SCALM outperforms existing tools in detecting bad practices in smart contracts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.21990v3">ChemDFM-R: A Chemical Reasoning LLM Enhanced with Atomized Chemical Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 18 figures, 11 tables
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have achieved impressive progress, their application in scientific domains such as chemistry remains hindered by shallow domain understanding and limited reasoning capabilities. In this work, we focus on the specific field of chemistry and develop a Chemical Reasoning LLM, ChemDFM-R. We first construct a comprehensive dataset of atomized chemical knowledge, ChemFG, annotating the presence of functional groups in molecules and the changes of functional groups during chemical reactions, to enhance the model's understanding of the fundamental principles and internal logic of chemistry. Then, we propose a mixed-source distillation method that integrates expertise in atomized knowledge with general reasoning skills, followed by domain-specific reinforcement learning to enhance chemical reasoning. Experiments on diverse chemical benchmarks demonstrate that ChemDFM-R achieves cutting-edge performance while providing interpretable, rationale-driven outputs. Further case studies illustrate how explicit reasoning chains significantly improve the model's reliability, transparency, and practicality in real-world human-AI collaboration scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15082v1">The Semantic Architect: How FEAML Bridges Structured Data and LLMs for Multi-Label Tasks</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Existing feature engineering methods based on large language models (LLMs) have not yet been applied to multi-label learning tasks. They lack the ability to model complex label dependencies and are not specifically adapted to the characteristics of multi-label tasks. To address the above issues, we propose Feature Engineering Automation for Multi-Label Learning (FEAML), an automated feature engineering method for multi-label classification which leverages the code generation capabilities of LLMs. By utilizing metadata and label co-occurrence matrices, LLMs are guided to understand the relationships between data features and task objectives, based on which high-quality features are generated. The newly generated features are evaluated in terms of model accuracy to assess their effectiveness, while Pearson correlation coefficients are used to detect redundancy. FEAML further incorporates the evaluation results as feedback to drive LLMs to continuously optimize code generation in subsequent iterations. By integrating LLMs with a feedback mechanism, FEAML realizes an efficient, interpretable and self-improving feature engineering paradigm. Empirical results on various multi-label datasets demonstrate that our FEAML outperforms other feature engineering methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15081v1">Quantifying Return on Security Controls in LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 13 pages, 9 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) are increasingly used in security-critical workflows, practitioners lack quantitative guidance on which safeguards are worth deploying. This paper introduces a decision-oriented framework and reproducible methodology that together quantify residual risk, convert adversarial probe outcomes into financial risk estimates and return-on-control (RoC) metrics, and enable monetary comparison of layered defenses for LLM-based systems. A retrieval-augmented generation (RAG) service is instantiated using the DeepSeek-R1 model over a corpus containing synthetic personally identifiable information (PII), and subjected to automated attacks with Garak across five vulnerability classes: PII leakage, latent context injection, prompt injection, adversarial attack generation, and divergence. For each (vulnerability, control) pair, attack success probabilities are estimated via Laplace's Rule of Succession and combined with loss triangle distributions, calibrated from public breach-cost data, in 10,000-run Monte Carlo simulations to produce loss exceedance curves and expected losses. Three widely used mitigations, attribute-based access control (ABAC); named entity recognition (NER) redaction using Microsoft Presidio; and NeMo Guardrails, are then compared to a baseline RAG configuration. The baseline system exhibits very high attack success rates (>= 0.98 for PII, latent injection, and prompt injection), yielding a total simulated expected loss of $313k per attack scenario. ABAC collapses success probabilities for PII and prompt-related attacks to near zero and reduces the total expected loss by ~94%, achieving an RoC of 9.83. NER redaction likewise eliminates PII leakage and attains an RoC of 5.97, while NeMo Guardrails provides only marginal benefit (RoC of 0.05).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17061v3">Parallelism Meets Adaptiveness: Scalable Documents Understanding in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 Accepted at AAAI 2026 Workshop on WoMAPF
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have shown increasing promise for collaborative task completion. However, existing multi-agent frameworks often rely on static workflows, fixed roles, and limited inter-agent communication, reducing their effectiveness in open-ended, high-complexity domains. This paper proposes a coordination framework that enables adaptiveness through three core mechanisms: dynamic task routing, bidirectional feedback, and parallel agent evaluation. The framework allows agents to reallocate tasks based on confidence and workload, exchange structured critiques to iteratively improve outputs, and crucially compete on high-ambiguity subtasks with evaluator-driven selection of the most suitable result. We instantiate these principles in a modular architecture and demonstrate substantial improvements in factual coverage, coherence, and efficiency over static and partially adaptive baselines. Our findings highlight the benefits of incorporating both adaptiveness and structured competition in multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15053v1">The Meta-Prompting Protocol: Orchestrating LLMs via Adversarial Feedback Loops</a></div>
    <div class="paper-meta">
      📅 2025-12-17
      | 💬 6 pages, 2 figures
    </div>
    <details class="paper-abstract">
      The transition of Large Language Models (LLMs) from stochastic chat interfaces to reliable software components necessitates a fundamental re-engineering of interaction paradigms. Current methodologies, predominantly heuristic-based "prompt engineering," fail to provide the deterministic guarantees required for mission-critical applications. We introduce the Meta-Prompting Protocol, a rigorous theoretical framework that formalizes the orchestration of LLMs as a programmable, self-optimizing system. Central to this protocol is the Adversarial Trinity, a tripartite topology comprising a Generator (P), an Auditor (A), and an Optimizer (O). By treating natural language instructions as differentiable variables within a semantic computation graph and utilizing textual critiques as gradients, this architecture mitigates hallucination and prevents model collapse. We demonstrate the theoretical viability of this approach using declarative programming paradigms (DSPy) and automatic textual differentiation (TextGrad), establishing a foundation for "Observable Software Engineering" in the era of probabilistic computing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.15000v1">DreamPRM-Code: Function-as-Step Process Reward Model with Label Correction for LLM Coding</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Process Reward Models (PRMs) have become essential for improving Large Language Models (LLMs) via test-time scaling, yet their effectiveness in coding remains limited due to the lack of meaningful step decompositions in code and the noise of Monte-Carlo-generated partial labels. We propose DreamPRM-Code, a coding-focused PRM that treats functions as reasoning steps using a Chain-of-Function prompting strategy to induce modular code generation, enabling PRM training and application analogous to mathematical reasoning tasks. To address label noise, DreamPRM-Code introduces a meta-learning-based correction mechanism that leverages clean final-solution unit-test labels and performs bi-level optimization to refine intermediate labels. Applying on test-time scaling, DreamPRM-Code achieved state-of-the-art performance on LiveCodeBench with 80.9 pass@1 rate, surpassing OpenAI o4-mini.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.11062v4">Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-17
    </div>
    <details class="paper-abstract">
      Multi-agent systems (MAS) and reinforcement learning (RL) are widely used to enhance the agentic capabilities of large language models (LLMs). MAS improves task performance through role-based orchestration, while RL uses environmental rewards to learn stronger policies, such as GRPO-style optimization. However, applying on-policy RL to MAS remains underexplored and presents unique challenges. Algorithmically, standard GRPO grouping assumptions break down because prompts vary by role and by turn. System-wise, the training stack must support MAS-workflow rollouts and on-policy updates for both single-policy and multi-policy models. We propose AT-GRPO, which includes (i) an agent- and turn-wise grouped RL algorithm tailored to MAS and (ii) a training system that supports both single- and multi-policy regimes. Across game, planning, coding, and math tasks, AT-GRPO delivers substantial gains. On long-horizon planning, it increases accuracy from a 14.0 to 47.0 percent single-agent RL baseline to 96.0 to 99.5 percent. It also improves reasoning performance, with average gains of 3.87 to 7.62 percent on coding tasks and 9.0 to 17.93 percent on math. Code and environments are available at: https://github.com/pettingllms-ai/PettingLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11844v2">Love First, Know Later: Persona-Based Romantic Compatibility Through LLM Text World Engines</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 NeurIPS 2025 Workshop: First Workshop on LLM Persona Modeling (Oral)
    </div>
    <details class="paper-abstract">
      We propose Love First, Know Later: a paradigm shift in computational matching that simulates interactions first, then assesses compatibility. Instead of comparing static profiles, our framework leverages LLMs as text world engines that operate in dual capacity-as persona-driven agents following behavioral policies and as the environment modeling interaction dynamics. We formalize compatibility assessment as a reward-modeling problem: given observed matching outcomes, we learn to extract signals from simulations that predict human preferences. Our key insight is that relationships hinge on responses to critical moments-we translate this observation from relationship psychology into mathematical hypotheses, enabling effective simulation. Theoretically, we prove that as LLM policies better approximate human behavior, the induced matching converges to optimal stable matching. Empirically, we validate on speed dating data for initial chemistry and divorce prediction for long-term stability. This paradigm enables interactive, personalized matching systems where users iteratively refine their agents, unlocking future possibilities for transparent and interactive compatibility assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14673v1">Reconsidering Conversational Norms in LLM Chatbots for Sustainable AI</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      LLM based chatbots have become central interfaces in technical, educational, and analytical domains, supporting tasks such as code reasoning, problem solving, and information exploration. As these systems scale, sustainability concerns have intensified, with most assessments focusing on model architecture, hardware efficiency, and deployment infrastructure. However, existing mitigation efforts largely overlook how user interaction practices themselves shape the energy profile of LLM based systems. In this vision paper, we argue that interaction level behavior appears to be an underexamined factor shaping the environmental impact of LLM based systems, and we present this issue across four dimensions. First, extended conversational patterns increase token production and raise the computational cost of inference. Second, expectations of instant responses limit opportunities for energy aware scheduling and workload consolidation. Third, everyday user habits contribute to cumulative operational demand in ways that are rarely quantified. Fourth, the accumulation of context affects memory requirements and reduces the efficiency of long running dialogues. Addressing these challenges requires rethinking how chatbot interactions are designed and conceptualized, and adopting perspectives that recognize sustainability as partly dependent on the conversational norms through which users engage with LLM based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14604v1">LLmFPCA-detect: LLM-powered Multivariate Functional PCA for Anomaly Detection in Sparse Longitudinal Texts</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Sparse longitudinal (SL) textual data arises when individuals generate text repeatedly over time (e.g., customer reviews, occasional social media posts, electronic medical records across visits), but the frequency and timing of observations vary across individuals. These complex textual data sets have immense potential to inform future policy and targeted recommendations. However, because SL text data lack dedicated methods and are noisy, heterogeneous, and prone to anomalies, detecting and inferring key patterns is challenging. We introduce LLmFPCA-detect, a flexible framework that pairs LLM-based text embeddings with functional data analysis to detect clusters and infer anomalies in large SL text datasets. First, LLmFPCA-detect embeds each piece of text into an application-specific numeric space using LLM prompts. Sparse multivariate functional principal component analysis (mFPCA) conducted in the numeric space forms the workhorse to recover primary population characteristics, and produces subject-level scores which, together with baseline static covariates, facilitate data segmentation, unsupervised anomaly detection and inference, and enable other downstream tasks. In particular, we leverage LLMs to perform dynamic keyword profiling guided by the data segments and anomalies discovered by LLmFPCA-detect, and we show that cluster-specific functional PC scores from LLmFPCA-detect, used as features in existing pipelines, help boost prediction performance. We support the stability of LLmFPCA-detect with experiments and evaluate it on two different applications using public datasets, Amazon customer-review trajectories, and Wikipedia talk-page comment streams, demonstrating utility across domains and outperforming state-of-the-art baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14594v1">LLM-driven Knowledge Enhancement for Multimodal Cancer Survival Prediction</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Current multimodal survival prediction methods typically rely on pathology images (WSIs) and genomic data, both of which are high-dimensional and redundant, making it difficult to extract discriminative features from them and align different modalities. Moreover, using a simple survival follow-up label is insufficient to supervise such a complex task. To address these challenges, we propose KEMM, an LLM-driven Knowledge-Enhanced Multimodal Model for cancer survival prediction, which integrates expert reports and prognostic background knowledge. 1) Expert reports, provided by pathologists on a case-by-case basis and refined by large language model (LLM), offer succinct and clinically focused diagnostic statements. This information may typically suggest different survival outcomes. 2) Prognostic background knowledge (PBK), generated concisely by LLM, provides valuable prognostic background knowledge on different cancer types, which also enhances survival prediction. To leverage these knowledge, we introduce the knowledge-enhanced cross-modal (KECM) attention module. KECM can effectively guide the network to focus on discriminative and survival-relevant features from highly redundant modalities. Extensive experiments on five datasets demonstrate that KEMM achieves state-of-the-art performance. The code will be released upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14531v1">VersatileFFN: Achieving Parameter Efficiency in LLMs via Adaptive Wide-and-Deep Reuse</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      The rapid scaling of Large Language Models (LLMs) has achieved remarkable performance, but it also leads to prohibitive memory costs. Existing parameter-efficient approaches such as pruning and quantization mainly compress pretrained models without enhancing architectural capacity, thereby hitting the representational ceiling of the base model. In this work, we propose VersatileFFN, a novel feed-forward network (FFN) that enables flexible reuse of parameters in both width and depth dimensions within a fixed parameter budget. Inspired by the dual-process theory of cognition, VersatileFFN comprises two adaptive pathways: a width-versatile path that generates a mixture of sub-experts from a single shared FFN, mimicking sparse expert routing without increasing parameters, and a depth-versatile path that recursively applies the same FFN to emulate deeper processing for complex tokens. A difficulty-aware gating dynamically balances the two pathways, steering "easy" tokens through the efficient width-wise route and allocating deeper iterative refinement to "hard" tokens. Crucially, both pathways reuse the same parameters, so all additional capacity comes from computation rather than memory. Experiments across diverse benchmarks and model scales demonstrate the effectiveness of the method. The code will be available at https://github.com/huawei-noah/noah-research/tree/master/VersatileFFN.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14490v1">PushGen: Push Notifications Generation with LLM</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Accepted by WSDM 2026
    </div>
    <details class="paper-abstract">
      We present PushGen, an automated framework for generating high-quality push notifications comparable to human-crafted content. With the rise of generative models, there is growing interest in leveraging LLMs for push content generation. Although LLMs make content generation straightforward and cost-effective, maintaining stylistic control and reliable quality assessment remains challenging, as both directly impact user engagement. To address these issues, PushGen combines two key components: (1) a controllable category prompt technique to guide LLM outputs toward desired styles, and (2) a reward model that ranks and selects generated candidates. Extensive offline and online experiments demonstrate its effectiveness, which has been deployed in large-scale industrial applications, serving hundreds of millions of users daily.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14474v1">Model-First Reasoning LLM Agents: Reducing Hallucinations through Explicit Problem Modeling</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with complex multi-step planning tasks, showing high rates of constraint violations and inconsistent solutions. Existing strategies such as Chain-of-Thought and ReAct rely on implicit state tracking and lack an explicit problem representation. Inspired by classical AI planning, we propose Model-First Reasoning (MFR), a two-phase paradigm in which the LLM first constructs an explicit model of the problem, defining entities, state variables, actions, and constraints, before generating a solution plan. Across multiple planning domains, including medical scheduling, route planning, resource allocation, logic puzzles, and procedural synthesis, MFR reduces constraint violations and improves solution quality compared to Chain-of-Thought and ReAct. Ablation studies show that the explicit modeling phase is critical for these gains. Our results suggest that many LLM planning failures stem from representational deficiencies rather than reasoning limitations, highlighting explicit modeling as a key component for robust and interpretable AI agents. All prompts, evaluation procedures, and task datasets are documented to facilitate reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14448v1">Reasoning-Style Poisoning of LLM Agents via Stealthy Style Transfer: Process-Level Attacks and Runtime Monitoring in RSV Space</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents relying on external retrieval are increasingly deployed in high-stakes environments. While existing adversarial attacks primarily focus on content falsification or instruction injection, we identify a novel, process-oriented attack surface: the agent's reasoning style. We propose Reasoning-Style Poisoning (RSP), a paradigm that manipulates how agents process information rather than what they process. We introduce Generative Style Injection (GSI), an attack method that rewrites retrieved documents into pathological tones--specifically "analysis paralysis" or "cognitive haste"--without altering underlying facts or using explicit triggers. To quantify these shifts, we develop the Reasoning Style Vector (RSV), a metric tracking Verification depth, Self-confidence, and Attention focus. Experiments on HotpotQA and FEVER using ReAct, Reflection, and Tree of Thoughts (ToT) architectures reveal that GSI significantly degrades performance. It increases reasoning steps by up to 4.4 times or induces premature errors, successfully bypassing state-of-the-art content filters. Finally, we propose RSP-M, a lightweight runtime monitor that calculates RSV metrics in real-time and triggers alerts when values exceed safety thresholds. Our work demonstrates that reasoning style is a distinct, exploitable vulnerability, necessitating process-level defenses beyond static content analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14417v1">PortAgent: LLM-driven Vehicle Dispatching Agent for Port Terminals</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Vehicle Dispatching Systems (VDSs) are critical to the operational efficiency of Automated Container Terminals (ACTs). However, their widespread commercialization is hindered due to their low transferability across diverse terminals. This transferability challenge stems from three limitations: high reliance on port operational specialists, a high demand for terminal-specific data, and time-consuming manual deployment processes. Leveraging the emergence of Large Language Models (LLMs), this paper proposes PortAgent, an LLM-driven vehicle dispatching agent that fully automates the VDS transferring workflow. It bears three features: (1) no need for port operations specialists; (2) low need of data; and (3) fast deployment. Specifically, specialist dependency is eliminated by the Virtual Expert Team (VET). The VET collaborates with four virtual experts, including a Knowledge Retriever, Modeler, Coder, and Debugger, to emulate a human expert team for the VDS transferring workflow. These experts specialize in the domain of terminal VDS via a few-shot example learning approach. Through this approach, the experts are able to learn VDS-domain knowledge from a few VDS examples. These examples are retrieved via a Retrieval-Augmented Generation (RAG) mechanism, mitigating the high demand for terminal-specific data. Furthermore, an automatic VDS design workflow is established among these experts to avoid extra manual interventions. In this workflow, a self-correction loop inspired by the LLM Reflexion framework is created
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14373v1">EcoScapes: LLM-Powered Advice for Crafting Sustainable Cities</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Climate adaptation is vital for the sustainability and sometimes the mere survival of our urban areas. However, small cities often struggle with limited personnel resources and integrating vast amounts of data from multiple sources for a comprehensive analysis. To overcome these challenges, this paper proposes a multi-layered system combining specialized LLMs, satellite imagery analysis and a knowledge base to aid in developing effective climate adaptation strategies. The corresponding code can be found at https://github.com/Photon-GitHub/EcoScapes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.03508v2">One Battle After Another: Probing LLMs' Limits on Multi-Turn Instruction Following with a Benchmark Evolving Framework</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Evaluating LLMs' instruction-following ability in multi-topic dialogues is essential yet challenging. Existing benchmarks are limited to a fixed number of turns, susceptible to saturation and failing to account for users' interactive experience. In this work, we propose a novel framework backed by a three-layer tracking mechanism and a query synthesis agent to mimic sequential user behaviors. Incorporating Flow Theory, we introduce process-centric metrics and terminate a conversational evaluation only upon exhausting user patience. Upon this framework, we present EvolIF, an evolving benchmark covering 12 constraint groups. Results indicate that GPT-5 excels, sustaining 14 turns with 66.40% robustness. It outperforms Gemini-3.0-Pro by a margin of 5.59%, while other models trail behind. Resources are available at https://github.com/JiaQiSJTU/EvolvingInstructionFollowing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14288v1">Leveraging LLMs for Collaborative Ontology Engineering in Parkinson Disease Monitoring and Alerting</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      This paper explores the integration of Large Language Models (LLMs) in the engineering of a Parkinson's Disease (PD) monitoring and alerting ontology through four key methodologies: One Shot (OS) prompt techniques, Chain of Thought (CoT) prompts, X-HCOME, and SimX-HCOME+. The primary objective is to determine whether LLMs alone can create comprehensive ontologies and, if not, whether human-LLM collaboration can achieve this goal. Consequently, the paper assesses the effectiveness of LLMs in automated ontology development and the enhancement achieved through human-LLM collaboration. Initial ontology generation was performed using One Shot (OS) and Chain of Thought (CoT) prompts, demonstrating the capability of LLMs to autonomously construct ontologies for PD monitoring and alerting. However, these outputs were not comprehensive and required substantial human refinement to enhance their completeness and accuracy. X-HCOME, a hybrid ontology engineering approach that combines human expertise with LLM capabilities, showed significant improvements in ontology comprehensiveness. This methodology resulted in ontologies that are very similar to those constructed by experts. Further experimentation with SimX-HCOME+, another hybrid methodology emphasizing continuous human supervision and iterative refinement, highlighted the importance of ongoing human involvement. This approach led to the creation of more comprehensive and accurate ontologies. Overall, the paper underscores the potential of human-LLM collaboration in advancing ontology engineering, particularly in complex domains like PD. The results suggest promising directions for future research, including the development of specialized GPT models for ontology construction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14277v1">SPARQL-LLM: Real-Time SPARQL Query Generation from Natural Language Questions</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 17 pages, 8 figures, 1 table. Under Review
    </div>
    <details class="paper-abstract">
      The advent of large language models is contributing to the emergence of novel approaches that promise to better tackle the challenge of generating structured queries, such as SPARQL queries, from natural language. However, these new approaches mostly focus on response accuracy over a single source while ignoring other evaluation criteria, such as federated query capability over distributed data stores, as well as runtime and cost to generate SPARQL queries. Consequently, they are often not production-ready or easy to deploy over (potentially federated) knowledge graphs with good accuracy. To mitigate these issues, in this paper, we extend our previous work and describe and systematically evaluate SPARQL-LLM, an open-source and triplestore-agnostic approach, powered by lightweight metadata, that generates SPARQL queries from natural language text. First, we describe its architecture, which consists of dedicated components for metadata indexing, prompt building, and query generation and execution. Then, we evaluate it based on a state-of-the-art challenge with multilingual questions, and a collection of questions from three of the most prevalent knowledge graphs within the field of bioinformatics. Our results demonstrate a substantial increase of 24% in the F1 Score on the state-of-the-art challenge, adaptability to high-resource languages such as English and Spanish, as well as ability to form complex and federated bioinformatics queries. Furthermore, we show that SPARQL-LLM is up to 36x faster than other systems participating in the challenge, while costing a maximum of $0.01 per question, making it suitable for real-time, low-cost text-to-SPARQL applications. One such application deployed over real-world decentralized knowledge graphs can be found at https://www.expasy.org/chat.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14233v1">PentestEval: Benchmarking LLM-based Penetration Testing with Modular and Stage-Level Design</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Penetration testing is essential for assessing and strengthening system security against real-world threats, yet traditional workflows remain highly manual, expertise-intensive, and difficult to scale. Although recent advances in Large Language Models (LLMs) offer promising opportunities for automation, existing applications rely on simplistic prompting without task decomposition or domain adaptation, resulting in unreliable black-box behavior and limited insight into model capabilities across penetration testing stages. To address this gap, we introduce PentestEval, the first comprehensive benchmark for evaluating LLMs across six decomposed penetration testing stages: Information Collection, Weakness Gathering and Filtering, Attack Decision-Making, Exploit Generation and Revision. PentestEval integrates expert-annotated ground truth with a fully automated evaluation pipeline across 346 tasks covering all stages in 12 realistic vulnerable scenarios. Our stage-level evaluation of 9 widely used LLMs reveals generally weak performance and distinct limitations across the stages of penetration-testing workflow. End-to-end pipelines reach only 31% success rate, and existing LLM-powered systems such as PentestGPT, PentestAgent, and VulnBot exhibit similar limitations, with autonomous agents failing almost entirely. These findings highlight that autonomous penetration testing demands stronger structured reasoning, where modularization enhances each individual stage and improves overall performance. PentestEval provides the foundational benchmark needed for future research on fine-grained, stage-level evaluation, paving the way toward more reliable LLM-based automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.15610v3">A LoRA-Based Approach to Fine-Tuning LLMs for Educational Guidance in Resource-Constrained Settings</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 18 pages, 6 figures (3 graphs + 3 flowchart/architecture diagrams), submitted as a preprint for review consideration in AI for Education or Machine Learning applications in low-resource settings. Includes detailed experiments with LoRA and quantization methods for efficient LLM fine-tuning
    </div>
    <details class="paper-abstract">
      The current study describes a cost-effective method for adapting large language models (LLMs) for academic advising with study-abroad contexts in mind and for application in low-resource methods for acculturation. With the Mistral-7B-Instruct model applied with a Low-Rank Adaptation (LoRA) method and a 4-bit quantization method, the model underwent training in two distinct stages related to this study's purpose to enhance domain specificity while maintaining computational efficiency. In Phase 1, the model was conditioned with a synthetic dataset via the Gemini Pro API, and in Phase 2, it was trained with manually curated datasets from the StudyAbroadGPT project to achieve enhanced, contextualized responses. Technical innovations entailed memory-efficient quantization, parameter-efficient adaptation, and continuous training analytics via Weights & Biases. After training, this study demonstrated a reduction in training loss by 52.7%, 92% accuracy in domain-specific recommendations, achieved 95% markdown-based formatting support, and a median run-rate of 100 samples per second on off-the-shelf GPU equipment. These findings support the effective application of instruction-tuned LLMs within educational advisers, especially in low-resource institutional scenarios. Limitations included decreased generalizability and the application of a synthetically generated dataset, but this framework is scalable for adding new multilingual-augmented and real-time academic advising processes. Future directions may include plans for the integration of retrieval-augmented generation, applying dynamic quantization routines, and connecting to real-time academic databases to increase adaptability and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.04359v2">Efficient Reinforcement Learning with Semantic and Token Entropy for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) has demonstrated superior performance in enhancing the reasoning capability of large language models (LLMs). However, this accuracy-oriented learning paradigm often suffers from entropy collapse, which reduces policy exploration and limits reasoning capabilities. To address this challenge, we propose an efficient reinforcement learning framework that leverages entropy signals at both the semantic and token levels to improve reasoning. From the data perspective, we introduce semantic entropy-guided curriculum learning, organizing training data from low to high semantic entropy to guide progressive optimization from easier to more challenging tasks. For the algorithmic design, we adopt non-uniform token treatment by imposing KL regularization on low-entropy tokens that critically impact policy exploration and applying stronger constraints on high-covariance portions within these tokens. By jointly optimizing data organization and algorithmic design, our method effectively mitigates entropy collapse and enhances LLM reasoning. Experimental results across 6 benchmarks with 3 different parameter-scale base models demonstrate that our method outperforms other entropy-based approaches in improving reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14179v1">A Comparative Analysis of Retrieval-Augmented Generation Techniques for Bengali Standard-to-Dialect Machine Translation Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Accepted to the Second Workshop on Bangla Language Processing (BLP) at IJCNLP-AACL 2025. 14 pages, 9 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Translating from a standard language to its regional dialects is a significant NLP challenge due to scarce data and linguistic variation, a problem prominent in the Bengali language. This paper proposes and compares two novel RAG pipelines for standard-to-dialectal Bengali translation. The first, a Transcript-Based Pipeline, uses large dialect sentence contexts from audio transcripts. The second, a more effective Standardized Sentence-Pairs Pipeline, utilizes structured local\_dialect:standard\_bengali sentence pairs. We evaluated both pipelines across six Bengali dialects and multiple LLMs using BLEU, ChrF, WER, and BERTScore. Our findings show that the sentence-pair pipeline consistently outperforms the transcript-based one, reducing Word Error Rate (WER) from 76\% to 55\% for the Chittagong dialect. Critically, this RAG approach enables smaller models (e.g., Llama-3.1-8B) to outperform much larger models (e.g., GPT-OSS-120B), demonstrating that a well-designed retrieval strategy can be more crucial than model size. This work contributes an effective, fine-tuning-free solution for low-resource dialect translation, offering a practical blueprint for preserving linguistic diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.11811v2">Enhancing Geo-localization for Crowdsourced Flood Imagery via LLM-Guided Attention</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Updated author list to include additional contributor. Revised title and improved methodology section based on collaborative feedback
    </div>
    <details class="paper-abstract">
      Crowdsourced street-view imagery from social media provides real-time visual evidence of urban flooding and other crisis events, yet it often lacks reliable geographic metadata for emergency response. Existing image geo-localization approaches, also known as Visual Place Recognition (VPR) models, exhibit substantial performance degradation when applied to such imagery due to visual distortions and domain shifts in cross-source scenarios. This paper presents VPR-AttLLM, a model-agnostic framework that integrates the semantic reasoning and geo-knowledge of Large Language Models (LLMs) into established VPR pipelines through attention-guided descriptor enhancement. By leveraging LLMs to identify location-informative regions within the city context and suppress visual noise, VPR-AttLLM improves retrieval performance without requiring model retraining or additional data. Comprehensive evaluations are conducted on extended benchmarks including SF-XL enriched with real social-media flood images, synthetic flooding scenarios over established query sets and Mapillary photos, and a new HK-URBAN dataset capturing morphologically distinct cityscapes. Integrating VPR-AttLLM with three state-of-the-art VPR models-CosPlace, EigenPlaces, and SALAD-consistently improves recall performance, yielding relative gains typically between 1-3% and reaching up to 8% on the most challenging real flood imagery. Beyond measurable gains in retrieval accuracy, this study establishes a generalizable paradigm for LLM-guided multimodal fusion in visual retrieval systems. By embedding principles from urban perception theory into attention mechanisms, VPR-AttLLM bridges human-like spatial reasoning with modern VPR architectures. Its plug-and-play design, strong cross-source robustness, and interpretability highlight its potential for scalable urban monitoring and rapid geo-localization of crowdsourced crisis imagery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.17399v2">DIWALI: Diversity and Inclusivity aWare cuLture specific Items for India: Dataset and Assessment of LLMs for Cultural Text Adaptation in Indian Context</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Accepted at EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely used in various tasks and applications. However, despite their wide capabilities, they are shown to lack cultural alignment \citep{ryan-etal-2024-unintended, alkhamissi-etal-2024-investigating} and produce biased generations \cite{naous-etal-2024-beer} due to a lack of cultural knowledge and competence. Evaluation of LLMs for cultural awareness and alignment is particularly challenging due to the lack of proper evaluation metrics and unavailability of culturally grounded datasets representing the vast complexity of cultures at the regional and sub-regional levels. Existing datasets for culture specific items (CSIs) focus primarily on concepts at the regional level and may contain false positives. To address this issue, we introduce a novel CSI dataset for Indian culture, belonging to 17 cultural facets. The dataset comprises ~8k cultural concepts from 36 sub-regions. To measure the cultural competence of LLMs on a cultural text adaptation task, we evaluate the adaptations using the CSIs created, LLM as Judge, and human evaluations from diverse socio-demographic region. Furthermore, we perform quantitative analysis demonstrating selective sub-regional coverage and surface-level adaptations across all considered LLMs. Our dataset is available here: https://huggingface.co/datasets/nlip/DIWALI, project webpage https://nlip-lab.github.io/nlip/publications/diwali/, and our codebase with model outputs can be found here: https://github.com/pramitsahoo/culture-evaluation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14142v1">Astraea: A State-Aware Scheduling Engine for LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 12 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being deployed as intelligent agents. Their multi-stage workflows, which alternate between local computation and calls to external network services like Web APIs, introduce a mismatch in their execution pattern and the scheduling granularity of existing inference systems such as vLLM. Existing systems typically focus on per-segment optimization which prevents them from minimizing the end-to-end latency of the complete agentic workflow, i.e., the global Job Completion Time (JCT) over the entire request lifecycle. To address this limitation, we propose Astraea, a service engine designed to shift the optimization from local segments to the global request lifecycle. Astraea employs a state-aware, hierarchical scheduling algorithm that integrates a request's historical state with future predictions. It dynamically classifies requests by their I/O and compute intensive nature and uses an enhanced HRRN policy to balance efficiency and fairness. Astraea also implements an adaptive KV cache manager that intelligently handles the agent state during I/O waits based on the system memory pressure. Extensive experiments show that Astraea reduces average JCT by up to 25.5\% compared to baseline methods. Moreover, our approach demonstrates strong robustness and stability under high load across various model scales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14138v1">LAPPI: Interactive Optimization with LLM-Assisted Preference-Based Problem Instantiation</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Many real-world tasks, such as trip planning or meal planning, can be formulated as combinatorial optimization problems. However, using optimization solvers is difficult for end users because it requires problem instantiation: defining candidate items, assigning preference scores, and specifying constraints. We introduce LAPPI (LLM-Assisted Preference-based Problem Instantiation), an interactive approach that uses large language models (LLMs) to support users in this instantiation process. Through natural language conversations, the system helps users transform vague preferences into well-defined optimization problems. These instantiated problems are then passed to existing optimization solvers to generate solutions. In a user study on trip planning, our method successfully captured user preferences and generated feasible plans that outperformed both conventional and prompt-engineering approaches. We further demonstrate LAPPI's versatility by adapting it to an additional use case.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14121v1">SportsGPT: An LLM-driven Framework for Interpretable Sports Motion Assessment and Training Guidance</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Existing intelligent sports analysis systems mainly focus on "scoring and visualization," often lacking automatic performance diagnosis and interpretable training guidance. Recent advances of Large Language Models (LMMs) and motion analysis techniques provide new opportunities to address the above limitations. In this paper, we propose SportsGPT, an LLM-driven framework for interpretable sports motion assessment and training guidance, which establishes a closed loop from motion time-series input to professional training guidance. First, given a set of high-quality target models, we introduce MotionDTW, a two-stage time series alignment algorithm designed for accurate keyframe extraction from skeleton-based motion sequences. Subsequently, we design a Knowledge-based Interpretable Sports Motion Assessment Model (KISMAM) to obtain a set of interpretable assessment metrics (e.g., insufficient extension) by constrasting the keyframes with the targe models. Finally, we propose SportsRAG, a RAG-based training guidance model based on Qwen3. Leveraging a 6B-token knowledge base, it prompts the LLM to generate professional training guidance by retrieving domain-specific QA pairs. Experimental results demonstrate that MotionDTW significantly outperforms traditional methods with lower temporal error and higher IoU scores. Furthermore, ablation studies validate the KISMAM and SportsRAG, confirming that SportsGPT surpasses general LLMs in diagnostic accuracy and professionalism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.22333v2">PAT: Accelerating LLM Decoding via Prefix-Aware Attention with Resource Efficient Multi-Tile Kernel</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Accepted by ASPLOS'26, code available at https://github.com/flashserve/PAT
    </div>
    <details class="paper-abstract">
      LLM serving is increasingly dominated by decode attention, which is a memory-bound operation due to massive KV cache loading from global memory. Meanwhile, real-world workloads exhibit substantial, hierarchical shared prefixes across requests (e.g., system prompts, tools/templates, RAG). Existing attention implementations fail to fully exploit prefix sharing: one-query-per-CTA execution repeatedly loads shared prefix KV cache, while one-size-fits-all tiling leaves on-chip resources idle and exacerbates bubbles for uneven KV lengths. These choices amplify memory bandwidth pressure and stall memory-bound decode attention. This paper introduces PAT, a prefix-aware attention kernel implementation for LLM decoding that organizes execution with a pack-forward-merge paradigm. PAT packs queries by shared prefix to reduce repeated memory accesses, runs a customized multi-tile kernel to achieve high resource efficiency. It further applies practical multi-stream forwarding and KV splitting to reduce resource bubbles. The final merge performs online softmax with negligible overhead. We implement PAT as an off-the-shelf plugin for vLLM. Evaluation on both real-world and synthetic workloads shows that PAT reduces attention latency by 53.5% on average and TPOT by 17.0-93.1% under the same configurations against state-of-the-art attention kernels. PAT's source code is publicly available at https://github.com/flashserve/PAT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.13109v3">FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been widely deployed with rapidly expanding context windows to support increasingly demanding applications. However, long contexts pose significant deployment challenges, primarily due to the KV cache whose size grows proportionally with context length. While KV cache compression methods are proposed to address this issue, KV dropping methods incur considerable accuracy loss, and KV retrieval methods suffer from significant efficiency bottlenecks. We propose FreeKV, an algorithm-system co-optimization framework to enhance KV retrieval efficiency while preserving accuracy. On the algorithm side, FreeKV introduces speculative retrieval to shift the KV selection and recall processes out of the critical path, combined with fine-grained correction to ensure accuracy. On the system side, FreeKV employs hybrid KV layouts across CPU and GPU memory to eliminate fragmented data transfers, and leverages double-buffered streamed recall to further improve efficiency. Experiments demonstrate that FreeKV achieves near-lossless accuracy across various scenarios and models, delivering up to 13$\times$ speedup compared to SOTA KV retrieval methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.14522v3">Lexo: Eliminating Stealthy Supply-Chain Attacks via LLM-Assisted Program Regeneration</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Software supply-chain attacks are an important and ongoing concern in the open source software ecosystem. These attacks maintain the standard functionality that a component implements, but additionally hide malicious functionality activated only when the component reaches its target environment. Lexo addresses such stealthy attacks by automatically learning and regenerating vulnerability-free versions of potentially malicious components. Lexo first generates a set of input-output pairs to model a component's full observable behavior, which it then uses to synthesize a new version of the original component. The new component implements the original functionality but avoids stealthy malicious behavior. Throughout this regeneration process, Lexo consults several distinct instances of Large Language Models (LLMs), uses correctness and coverage metrics to shepherd these instances, and guardrails their results. An evaluation on 100+ real-world packages, including high-profile stealthy supply-chain attacks, indicates that Lexo scales across multiple domains, regenerates code efficiently (<30m on average), maintains compatibility, and succeeds in eliminating malicious code in several real-world supply-chain-attacks, even in cases when a state-of-the-art LLM fails to eliminate malicious code when given the source code of the component and prompted to do so.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.17874v2">Beyond Jailbreak: Unveiling Risks in LLM Applications Arising from Blurred Capability Boundaries</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Accepted by Network and Distributed System Security (NDSS) Symposium 2026
    </div>
    <details class="paper-abstract">
      LLM applications (i.e., LLM apps) leverage the powerful capabilities of LLMs to provide users with customized services, revolutionizing traditional application development. While the increasing prevalence of LLM-powered applications provides users with unprecedented convenience, it also brings forth new security challenges. For such an emerging ecosystem, the security community lacks sufficient understanding of the LLM application ecosystem, especially regarding the capability boundaries of the applications themselves. In this paper, we systematically analyzed the new development paradigm and defined the concept of the LLM app capability space. We also uncovered potential new risks beyond jailbreak that arise from ambiguous capability boundaries in real-world scenarios, namely, capability downgrade and upgrade. To evaluate the impact of these risks, we designed and implemented an LLM app capability evaluation framework, LLMApp-Eval. First, we collected application metadata across 4 platforms and conducted a cross-platform ecosystem analysis. Then, we evaluated the risks for 199 popular applications among 4 platforms and 6 open-source LLMs. We identified that 178 (89.45%) potentially affected applications, which can perform tasks from more than 15 scenarios or be malicious. We even found 17 applications in our study that executed malicious tasks directly, without applying any adversarial rewriting. Furthermore, our experiments also reveal a positive correlation between the quality of prompt design and application robustness. We found that well-designed prompts enhance security, while poorly designed ones can facilitate abuse. We hope our work inspires the community to focus on the real-world risks of LLM applications and foster the development of a more robust LLM application ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.21218v2">Can Finetuing LLMs on Small Human Samples Increase Heterogeneity, Alignment, and Belief-Action Coherence?</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      There is ongoing debate about whether large language models (LLMs) can serve as substitutes for human participants in survey and experimental research. While recent work in fields such as marketing and psychology has explored the potential of LLM-based simulation, a growing body of evidence cautions against this practice: LLMs often fail to align with real human behavior, exhibiting limited diversity, systematic misalignment for minority subgroups, insufficient within-group variance, and discrepancies between stated beliefs and actions. This study examines an important and distinct question in this domain: whether fine-tuning on a small subset of human survey data, such as that obtainable from a pilot study, can mitigate these issues and yield realistic simulated outcomes. Using a behavioral experiment on information disclosure, we compare human and LLM-generated responses across multiple dimensions, including distributional divergence, subgroup alignment, belief-action coherence, and the recovery of regression coefficients. We find that fine-tuning on small human samples substantially improves heterogeneity, alignment, and belief-action coherence relative to the base model. However, even the best-performing fine-tuned models fail to reproduce the regression coefficients of the original study, suggesting that LLM-generated data remain unsuitable for replacing human participants in formal inferential analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13979v1">ReflCtrl: Controlling LLM Reflection via Representation Engineering</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Spotlight in NeurIPS 25 MI workshop
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) with Chain-of-Thought (CoT) reasoning have achieved strong performance across diverse tasks, including mathematics, coding, and general reasoning. A distinctive ability of these reasoning models is self-reflection: the ability to review and revise previous reasoning steps. While self-reflection enhances reasoning performance, it also increases inference cost. In this work, we study self-reflection through the lens of representation engineering. We segment the model's reasoning into steps, identify the steps corresponding to reflection, and extract a reflection direction in the latent space that governs this behavior. Using this direction, we propose a stepwise steering method that can control reflection frequency. We call our framework ReflCtrl. Our experiments show that (1) in many cases reflections are redundant, especially in stronger models (in our experiments, we can save up to 33.6 percent of reasoning tokens while preserving performance), and (2) the model's reflection behavior is highly correlated with an internal uncertainty signal, implying self-reflection may be controlled by the model's uncertainty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13978v1">Evaluating Frontier LLMs on PhD-Level Mathematical Reasoning: A Benchmark on a Textbook in Theoretical Computer Science about Randomized Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has led to significant breakthroughs in automated mathematical reasoning and scientific discovery. Georgiev, G${ó}$mez-Serrano, Tao, and Wagner [GGSTW+25] demonstrate that AI systems can explore new constructions and improve existing bounds, illustrating the growing potential of LLMs to accelerate mathematical discovery. Similarly, Bubeck et al. [BCE+25] show that GPT-5 can meaningfully contribute to scientific workflows, from proposing hypotheses to generating proofs and analyses. Despite these advances, a rigorous evaluation of these models on canonical, graduate-level mathematical theory remains necessary to understand their baseline reasoning capabilities. In this paper, we present a comprehensive benchmark of four frontier models: GPT-5-Thinking, Gemini-3-Pro, Claude-Sonnet-4.5-Thinking, and Grok-4 against the classic curriculum of Randomized Algorithms by Motwani and Raghavan [MR95]. We tasked each model with generating formal LaTeX proofs for a series of lemmas and exercises spanning the textbook. We find that while the top-tier models (Gemini, and Claude) achieve a high accuracy rate (approx. 66%), demonstrating a robust grasp of probabilistic method and formal logic, other models lag significantly in consistency (approx. 40%). We provide a qualitative analysis of the generated proofs, highlighting differences in conciseness, hallucination rates, and logical structure. Our results suggest that while frontier models have reached a threshold of proficiency suitable for graduate-level pedagogical assistance and formalization, significant variance exists in their reliability for rigorous mathematical derivation. The code and the full set of LLM-generated responses are open-sourced and publicly available at https://github.com/magiclinux/math_benchmark_probability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13974v1">Autonomous Construction-Site Safety Inspection Using Mobile Robots: A Multilayer VLM-LLM Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Construction safety inspection remains mostly manual, and automated approaches still rely on task-specific datasets that are hard to maintain in fast-changing construction environments due to frequent retraining. Meanwhile, field inspection with robots still depends on human teleoperation and manual reporting, which are labor-intensive. This paper aims to connect what a robot sees during autonomous navigation to the safety rules that are common in construction sites, automatically generating a safety inspection report. To this end, we proposed a multi-layer framework with two main modules: robotics and AI. On the robotics side, SLAM and autonomous navigation provide repeatable coverage and targeted revisits via waypoints. On AI side, a Vision Language Model (VLM)-based layer produces scene descriptions; a retrieval component powered grounds those descriptions in OSHA and site policies; Another VLM-based layer assesses the safety situation based on rules; and finally Large Language Model (LLM) layer generates safety reports based on previous outputs. The framework is validated with a proof-of-concept implementation and evaluated in a lab environment that simulates common hazards across three scenarios. Results show high recall with competitive precision compared to state-of-the-art closed-source models. This paper contributes a transparent, generalizable pipeline that moves beyond black-box models by exposing intermediate artifacts from each layer and keeping the human in the loop. This work provides a foundation for future extensions to additional tasks and settings within and beyond construction context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.02080v2">The 4/$δ$ Bound: Designing Predictable LLM-Verifier Systems for Formal Method Guarantee</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 36 pages, 9 figures
    </div>
    <details class="paper-abstract">
      The integration of Formal Verification tools with Large Language Models (LLMs) offers a path to scale software verification beyond manual workflows. However, current methods remain unreliable: without a solid theoretical footing, the refinement process acts as a black box that may oscillate, loop, or diverge. This work bridges this critical gap by developing an LLM-Verifier Convergence Theorem, providing the first formal framework with provable guarantees for termination in multi-stage verification pipelines. We model the interaction not as a generic loop, but as a sequential absorbing Markov Chain comprising four essential engineering stages: \texttt{CodeGen}, \texttt{Compilation}, \texttt{InvariantSynth}, and \texttt{SMTSolving}. We prove that for any non-zero stage success probability ($δ> 0$), the system reaches the \texttt{Verified} state almost surely. Furthermore, because of the sequential nature of the pipeline, we derive a precise latency bound of $\mathbb{E}[n] \leq 4/δ$. We stress-tested this prediction in an extensive empirical campaign comprising over 90,000 trials. The results match the theory with striking consistency: every run reached verification, and the empirical convergence factor clustered tightly around $C_f\approx 1.0$, confirming that the $4/δ$ bound accurately mirrors system behavior rather than serving as a loose buffer. Based on this data, we identify three distinct operating zones -- marginal, practical, and high-performance -- and propose a dynamic calibration strategy to handle parameter drift in real-world environments. Together, these contributions replace heuristic guesswork with a rigorous architectural foundation, enabling predictable resource planning and performance budgeting for safety-critical software.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14946v1">EVICPRESS: Joint KV-Cache Compression and Eviction for Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Reusing KV cache is essential for high efficiency of Large Language Model (LLM) inference systems. With more LLM users, the KV cache footprint can easily exceed GPU memory capacity, so prior work has proposed to either evict KV cache to lower-tier storage devices, or compress KV cache so that more KV cache can be fit in the fast memory. However, prior work misses an important opportunity: jointly optimizing the eviction and compression decisions across all KV caches to minimize average generation latency without hurting quality. We propose EVICPRESS, a KV-cache management system that applies lossy compression and adaptive eviction to KV cache across multiple storage tiers. Specifically, for each KV cache of a context, EVICPRESS considers the effect of compression and eviction of the KV cache on the average generation quality and delay across all contexts as a whole. To achieve this, EVICPRESS proposes a unified utility function that quantifies the effect of quality and delay of the lossy compression or eviction. To this end, EVICPRESS's profiling module periodically updates the utility function scores on all possible eviction-compression configurations for all contexts and places KV caches using a fast heuristic to rearrange KV caches on all storage tiers, with the goal of maximizing the utility function scores on each storage tier. Compared to the baselines that evict KV cache or compress KV cache, EVICPRESS achieves higher KV-cache hit rates on fast devices, i.e., lower delay, while preserving high generation quality by applying conservative compression to contexts that are sensitive to compression errors. Evaluation on 12 datasets and 5 models demonstrates that EVICPRESS achieves up to 2.19x faster time-to-first-token (TTFT) at equivalent generation quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14896v1">DrugRAG: Enhancing Pharmacy LLM Performance Through A Novel Retrieval-Augmented Generation Pipeline</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 11 pages, 2 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Objectives: To evaluate large language model (LLM) performance on pharmacy licensure-style question-answering (QA) tasks and develop an external knowledge integration method to improve their accuracy. Methods: We benchmarked eleven existing LLMs with varying parameter sizes (8 billion to 70+ billion) using a 141-question pharmacy dataset. We measured baseline accuracy for each model without modification. We then developed a three-step retrieval-augmented generation (RAG) pipeline, DrugRAG, that retrieves structured drug knowledge from validated sources and augments model prompts with evidence-based context. This pipeline operates externally to the models, requiring no changes to model architecture or parameters. Results: Baseline accuracy ranged from 46% to 92%, with GPT-5 (92%) and o3 (89%) achieving the highest scores. Models with fewer than 8 billion parameters scored below 50%. DrugRAG improved accuracy across all tested models, with gains ranging from 7 to 21 percentage points (e.g., Gemma 3 27B: 61% to 71%, Llama 3.1 8B: 46% to 67%) on the 141-item benchmark. Conclusion: We demonstrate that external structured drug knowledge integration through DrugRAG measurably improves LLM accuracy on pharmacy tasks without modifying the underlying models. This approach provides a practical pipeline for enhancing pharmacy-focused AI applications with evidence-based information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.18148v2">Hidden in the Haystack: Smaller Needles are More Difficult for LLMs to Find</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face significant challenges with needle-in-ahaystack tasks, where relevant information ("the needle") must be drawn from a large pool of irrelevant context ("the haystack"). Previous studies have highlighted positional bias and distractor quantity as critical factors affecting model performance, yet the influence of gold context size, the length of the answer-containing document, has received little attention. We present the first systematic study of gold context size in long-context question answering, spanning three diverse benchmarks (general knowledge, biomedical reasoning, and mathematical reasoning), eleven state-of-the-art LLMs (including recent reasoning models), and more than 150K controlled runs. Our experiments reveal that LLM performance drops sharply when the gold context is shorter, i.e., smaller gold contexts consistently degrade model performance and amplify positional sensitivity, posing a major challenge for agentic systems that must integrate scattered, fine-grained information of varying lengths. This effect persists under rigorous confounder analysis: even after controlling for gold context position, answer token repetition, gold-to-distractor ratio, distractor volume, and domain specificity, gold context size remains a decisive, independent predictor of success. Our work provides clear insights to guide the design of robust, context-aware LLM-driven systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14846v1">MALCDF: A Distributed Multi-Agent LLM Framework for Real-Time Cyber</a></div>
    <div class="paper-meta">
      📅 2025-12-16
    </div>
    <details class="paper-abstract">
      Traditional, centralized security tools often miss adaptive, multi-vector attacks. We present the Multi-Agent LLM Cyber Defense Framework (MALCDF), a practical setup where four large language model (LLM) agents-Detection, Intelligence, Response, and Analysis-work together in real time. Agents communicate over a Secure Communication Layer (SCL) with encrypted, ontology-aligned messages, and produce audit-friendly outputs (e.g., MITRE ATT&CK mappings). For evaluation, we keep the test simple and consistent: all reported metrics come from the same 50-record live stream derived from the CICIDS2017 feature schema. CICIDS2017 is used for configuration (fields/schema) and to train a practical ML baseline. The ML-IDS baseline is a Lightweight Random Forest IDS (LRF-IDS) trained on a subset of CICIDS2017 and tested on the 50-record stream, with no overlap between training and test records. In experiments, MALCDF reaches 90.0% detection accuracy, 85.7% F1-score, and 9.1% false-positive rate, with 6.8s average per-event latency. It outperforms the lightweight ML-IDS baseline and a single-LLM setup on accuracy while keeping end-to-end outputs consistent. Overall, this hands-on build suggests that coordinating simple LLM agents with secure, ontology-aligned messaging can improve practical, real-time cyber defense.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.14792v1">IaC Generation with LLMs: An Error Taxonomy and A Study on Configuration Knowledge Injection</a></div>
    <div class="paper-meta">
      📅 2025-12-16
      | 💬 Submitted to ACM
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) currently exhibit low success rates in generating correct and intent-aligned Infrastructure as Code (IaC). This research investigated methods to improve LLM-based IaC generation, specifically for Terraform, by systematically injecting structured configuration knowledge. To facilitate this, an existing IaC-Eval benchmark was significantly enhanced with cloud emulation and automated error analysis. Additionally, a novel error taxonomy for LLM-assisted IaC code generation was developed. A series of knowledge injection techniques was implemented and evaluated, progressing from Naive Retrieval-Augmented Generation (RAG) to more sophisticated Graph RAG approaches. These included semantic enrichment of graph components and modeling inter-resource dependencies. Experimental results demonstrated that while baseline LLM performance was poor (27.1% overall success), injecting structured configuration knowledge increased technical validation success to 75.3% and overall success to 62.6%. Despite these gains in technical correctness, intent alignment plateaued, revealing a "Correctness-Congruence Gap" where LLMs can become proficient "coders" but remain limited "architects" in fulfilling nuanced user intent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13676v1">Towards Effective Model Editing for LLM Personalization</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 15 pages (including appendix), 7 figures. Code, data, results, and additional resources are available at: https://model-editing.github.io
    </div>
    <details class="paper-abstract">
      Personalization is becoming indispensable for LLMs to align with individual user preferences and needs. Yet current approaches are often computationally expensive, data-intensive, susceptible to catastrophic forgetting, and prone to performance degradation in multi-turn interactions or when handling implicit queries. To address these challenges, we conceptualize personalization as a model editing task and introduce Personalization Editing, a framework that applies localized edits guided by clustered preference representations. This design enables precise preference-aligned updates while preserving overall model capabilities. In addition, existing personalization benchmarks frequently rely on persona-based dialogs between LLMs rather than user-LLM interactions, or focus primarily on stylistic imitation while neglecting information-seeking tasks that require accurate recall of user-specific preferences. We introduce User Preference Question Answering (UPQA), a short-answer QA dataset constructed from in-situ user queries with varying levels of difficulty. Unlike prior benchmarks, UPQA directly evaluates a model's ability to recall and apply specific user preferences. Across experimental settings, Personalization Editing achieves higher editing accuracy and greater computational efficiency than fine-tuning, while outperforming prompting-based baselines in multi-turn conversations and implicit preference questions settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2502.03604v3">Bilevel ZOFO: Efficient LLM Fine-Tuning and Meta-Training</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Fine-tuning pre-trained Large Language Models (LLMs) for downstream tasks using First-Order (FO) optimizers presents significant computational challenges. Parameter-Efficient Fine-Tuning (PEFT) methods address these by freezing most model parameters and training only a small subset. However, PEFT often underperforms compared to full fine-tuning when high task-specific accuracy is required. Zeroth-Order (ZO) methods fine-tune the entire pre-trained model without back-propagation, estimating gradients through forward passes only. While memory-efficient, ZO methods suffer from slow convergence and high sensitivity to prompt selection. We bridge these two worlds with Bilevel-ZOFO, a bilevel optimization method that couples fast, local FO-PEFT adaptation at the inner level with stable, memory-efficient ZO updates of the full backbone at the outer level. The FO-PEFT inner loop performs fast, low-memory local adaptation that reduces the variance of ZO estimates and stabilizes the search, guiding the outer ZO updates of the full backbone and reducing prompt sensitivity. In the mean time, the outer ZO provides better generalization ability for PEFT. We provide theoretical convergence guarantees and empirically demonstrate that Bilevel-ZOFO significantly outperforms existing ZO and FO-PEFT methods, achieving 2-4 times faster training while maintaining similar memory efficiency. Additionally, we show by updating the backbone with ZO and adapting only a tiny FO-PEFT block per task, Bilevel-ZOFO combines full-model capacity with few-shot efficiency, making it a very efficient meta-learning algorithm that quickly adapts to new tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.03005v2">From Moderation to Mediation: Can LLMs Serve as Mediators in Online Flame Wars?</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) has opened new possibilities for AI for good applications. As LLMs increasingly mediate online communication, their potential to foster empathy and constructive dialogue becomes an important frontier for responsible AI research. This work explores whether LLMs can serve not only as moderators that detect harmful content, but as mediators capable of understanding and de-escalating online conflicts. Our framework decomposes mediation into two subtasks: judgment, where an LLM evaluates the fairness and emotional dynamics of a conversation, and steering, where it generates empathetic, de-escalatory messages to guide participants toward resolution. To assess mediation quality, we construct a large Reddit-based dataset and propose a multi-stage evaluation pipeline combining principle-based scoring, user simulation, and human comparison. Experiments show that API-based models outperform open-source counterparts in both reasoning and intervention alignment when doing mediation. Our findings highlight both the promise and limitations of current LLMs as emerging agents for online social mediation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13526v1">Async Control: Stress-testing Asynchronous Control Measures for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      LLM-based software engineering agents are increasingly used in real-world development tasks, often with access to sensitive data or security-critical codebases. Such agents could intentionally sabotage these codebases if they were misaligned. We investigate asynchronous monitoring, in which a monitoring system reviews agent actions after the fact. Unlike synchronous monitoring, this approach does not impose runtime latency, while still attempting to disrupt attacks before irreversible harm occurs. We treat monitor development as an adversarial game between a blue team (who design monitors) and a red team (who create sabotaging agents). We attempt to set the game rules such that they upper bound the sabotage potential of an agent based on Claude 4.1 Opus. To ground this game in a realistic, high-stakes deployment scenario, we develop a suite of 5 diverse software engineering environments that simulate tasks that an agent might perform within an AI developer's internal infrastructure. Over the course of the game, we develop an ensemble monitor that achieves a 6% false negative rate at 1% false positive rate on a held out test environment. Then, we estimate risk of sabotage at deployment time by extrapolating from our monitor's false negative rate. We describe one simple model for this extrapolation, present a sensitivity analysis, and describe situations in which the model would be invalid. Code is available at: https://github.com/UKGovernmentBEIS/async-control.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.08403v2">DFALLM: Achieving Generalizable Multitask Deepfake Detection by Optimizing Audio LLM Components</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Audio deepfake detection has recently garnered public concern due to its implications for security and reliability. Traditional deep learning methods have been widely applied to this task but often lack generalisability when confronted with newly emerging spoofing techniques and more tasks such as spoof attribution recognition rather than simple binary classification. In principle, Large Language Models (LLMs) are considered to possess the needed generalisation capabilities. However, previous research on Audio LLMs (ALLMs) indicates a generalization bottleneck in audio deepfake detection performance, even when sufficient data is available. Consequently, this study investigates the model architecture and examines the effects of the primary components of ALLMs, namely the audio encoder and the text-based LLM. Our experiments demonstrate that the careful selection and combination of audio encoders and text-based LLMs are crucial for unlocking the deepfake detection potential of ALLMs. We further propose an ALLM structure capable of generalizing deepfake detection abilities to out-of-domain spoofing tests and other deepfake tasks, such as spoof positioning and spoof attribution recognition. Our proposed model architecture achieves state-of-the-art (SOTA) performance across multiple datasets, including ASVSpoof2019, InTheWild, and Demopage, with accuracy reaching up to 95.76% on average, and exhibits competitive capabilities in other deepfake detection tasks such as attribution, and localisation compared to SOTA audio understanding models. Data and codes are provided in supplementary materials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13481v1">neuralFOMO: Can LLMs Handle Being Second Best? Measuring Envy-Like Preferences in Multi-Agent Settings</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Under Review
    </div>
    <details class="paper-abstract">
      Envy is a common human behavior that shapes competitiveness and can alter outcomes in team settings. As large language models (LLMs) increasingly act on behalf of humans in collaborative and competitive workflows, there is a pressing need to evaluate whether and under what conditions they exhibit envy-like preferences. In this paper, we test whether LLMs show envy-like behavior toward each other. We considered two scenarios: (1) A point allocation game that tests whether a model tries to win over its peer. (2) A workplace setting observing behaviour when recognition is unfair. Our findings reveal consistent evidence of envy-like patterns in certain LLMs, with large variation across models and contexts. For instance, GPT-5-mini and Claude-3.7-Sonnet show a clear tendency to pull down the peer model to equalize outcomes, whereas Mistral-Small-3.2-24B instead focuses on maximizing its own individual gains. These results highlight the need to consider competitive dispositions as a safety and design factor in LLM-based multi-agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13438v1">From User Interface to Agent Interface: Efficiency Optimization of UI Representations for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      While Large Language Model (LLM) agents show great potential for automated UI navigation such as automated UI testing and AI assistants, their efficiency has been largely overlooked. Our motivating study reveals that inefficient UI representation creates a critical performance bottleneck. However, UI representation optimization, formulated as the task of automatically generating programs that transform UI representations, faces two unique challenges. First, the lack of Boolean oracles, which traditional program synthesis uses to decisively validate semantic correctness, poses a fundamental challenge to co-optimization of token efficiency and completeness. Second, the need to process large, complex UI trees as input while generating long, compositional transformation programs, making the search space vast and error-prone. Toward addressing the preceding limitations, we present UIFormer, the first automated optimization framework that synthesizes UI transformation programs by conducting constraint-based optimization with structured decomposition of the complex synthesis task. First, UIFormer restricts the program space using a domain-specific language (DSL) that captures UI-specific operations. Second, UIFormer conducts LLM-based iterative refinement with correctness and efficiency rewards, providing guidance for achieving the efficiency-completeness co-optimization. UIFormer operates as a lightweight plugin that applies transformation programs for seamless integration with existing LLM agents, requiring minimal modifications to their core logic. Evaluations across three UI navigation benchmarks spanning Android and Web platforms with five LLMs demonstrate that UIFormer achieves 48.7% to 55.8% token reduction with minimal runtime overhead while maintaining or improving agent performance. Real-world industry deployment at WeChat further validates the practical impact of UIFormer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.00319v2">RL-Struct: A Lightweight Reinforcement Learning Framework for Reliable Structured Output in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 13 pages, 9 figures. Model is available at https://huggingface.co/Freakz3z/Qwen-JSON
    </div>
    <details class="paper-abstract">
      The Structure Gap between probabilistic LLM generation and deterministic schema requirements hinders automated workflows. We propose RL-Struct, a lightweight framework using Gradient Regularized Policy Optimization (GRPO) with a hierarchical reward function to align LLMs with structural constraints. This approach eliminates the critic network, reducing peak VRAM by 38% compared to PPO. On complex JSON tasks, RL-Struct achieves 89.7% structural accuracy and 92.1% validity, significantly outperforming SFT and zero-shot baselines. We also report an emergent curriculum--a self-organized learning process where the model prioritizes syntax before semantics. Our model is publicly available at https://huggingface.co/Freakz3z/Qwen-JSON.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.08768v4">AraReasoner: Evaluating Reasoning-Based LLMs for Arabic NLP</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable progress in reasoning abilities and general natural language processing (NLP) tasks, yet their performance on Arabic data, characterized by rich morphology, diverse dialects, and complex script, remains underexplored. This paper presents a comprehensive benchmarking study of multiple reasoning-focused LLMs, with a special emphasis on the newly introduced DeepSeek models, across a suite of fifteen Arabic NLP tasks. We experiment with various strategies, including zero-shot, few-shot, and fine-tuning. This allows us to systematically evaluate performance on datasets covering a range of applications to examine their capacity for linguistic reasoning under different levels of complexity. Our experiments reveal several key findings. First, carefully selecting just three in-context examples delivers an average uplift of over 13 F1 points on classification tasks-boosting sentiment analysis from 35.3% to 87.5% and paraphrase detection from 56.1% to 87.0%. Second, reasoning-focused DeepSeek architectures outperform a strong GPT o4-mini baseline by an average of 12 F1 points on complex inference tasks in the zero-shot setting. Third, LoRA-based fine-tuning yields up to an additional 8 points in F1 and BLEU compared to equivalent increases in model scale. The code is available at https://github.com/gufranSabri/deepseek-evals
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13360v1">UCRBench: Benchmarking LLMs on Use Case Recovery</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Use cases are widely employed to specify functional requirements, yet existing benchmarks are scarce and face the risk of being misaligned with actual system behavior, similarly limiting the rigorous evaluation of large language models (LLMs) in generating use cases from source code. We address this gap by introducing code-aligned use case benchmarks, constructed through manual validation of both user-goal and subfunction use cases across nine real-world software projects. Using this benchmark, we conduct the first systematic study of LLMs and propose a hierarchical evaluation protocol that assesses actor correctness, name accuracy, path fidelity, and behavioral coverage. The results show that while LLMs can partially reconstruct system functionality, their performance varies significantly across projects, with particularly noticeable shortcomings in domain-specific and multi-module systems. The models also exhibit high omission rates and struggle to maintain consistent abstraction when aggregating subfunctions into user-goal use cases, highlighting both the potential and current limitations of LLM-based use case reverse engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13337v1">FROC: A Unified Framework with Risk-Optimized Control for Machine Unlearning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Machine unlearning (MU) seeks to eliminate the influence of specific training examples from deployed models. As large language models (LLMs) become widely used, managing risks arising from insufficient forgetting or utility loss is increasingly crucial. Current MU techniques lack effective mechanisms for evaluating and controlling these risks, hindering the selection of strategies that appropriately balance safety and utility, and raising trust concerns surrounding the "right to be forgotten." To address these issues, we propose FROC, a unified framework with Risk-Optimized Control for machine unlearning in LLMs. FROC is built around a conformal-style risk-control formulation that expresses a user-specified risk budget on unlearning behavior. This probability-based constraint enables FROC to compare MU strategies, identify feasible operating regions, and guide hyperparameter selection according to desired trade-offs between forgetting sufficiency and utility preservation. To operationalize this constraint, FROC introduces a smoothly varying continuous risk model that aggregates forgetting deficiency and utility degradation into a single configuration-level score. Building on conformal risk analysis, FROC computes (1) the Conformal Unlearning Risk (CUR), a data-driven estimated value on the probability that forgotten samples continue to influence model predictions, and (2) risk-controlled configuration sets, which identify unlearning hyperparameters that are valid under the specified risk budget. Experiments across multiple LLM MU methods demonstrate that FROC produces stable, interpretable risk landscapes and reveals consistent relationships between unlearning configurations, semantic shift, and utility impact. FROC reframes MU as a controllable, risk-aware process and offers a practical foundation for managing unlearning behavior in large-scale LLM deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10611v2">Phythesis: Physics-Guided Evolutionary Scene Synthesis for Energy-Efficient Data Center Design via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Data center (DC) infrastructure serves as the backbone to support the escalating demand for computing capacity. Traditional design methodologies that blend human expertise with specialized simulation tools scale poorly with the increasing system complexity. Recent studies adopt generative artificial intelligence to design plausible human-centric indoor layouts. However, they do not consider the underlying physics, making them unsuitable for the DC design that sets quantifiable operational objectives and strict physical constraints. To bridge the gap, we propose Phythesis, a novel framework that synergizes large language models (LLMs) and physics-guided evolutionary optimization to automate simulation-ready (SimReady) scene synthesis for energy-efficient DC design. Phythesis employs an iterative bi-level optimization architecture, where (i) the LLM-driven optimization level generates physically plausible three-dimensional layouts and self-criticizes them to refine the scene topology, and (ii) the physics-informed optimization level identifies the optimal asset parameters and selects the best asset combination. Experiments on three generation scales show that Phythesis achieves 57.3% generation success rate increase and 11.5% power usage effectiveness (PUE) improvement, compared with the vanilla LLM-based solution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13298v1">MiniLingua: A Small Open-Source LLM for European Languages</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 9+6 pages, 6 figures and 3 tables in the main text. Code at https://github.com/MiniLingua-ai/training_artifacts
    </div>
    <details class="paper-abstract">
      Large language models are powerful but often limited by high computational cost, privacy concerns, and English-centric training. Recent progress demonstrates that small, efficient models with around one billion parameters can deliver strong results and enable on-device use. This paper introduces MiniLingua, a multilingual open-source LLM of one billion parameters trained from scratch for 13 European languages, designed to balance coverage and instruction-following capabilities. Based on evaluation results, the instruction-tuned version of MiniLingua outperforms EuroLLM, a model with a similar training approach but a larger training budget, on summarization, classification and both open- and closed-book question answering. Moreover, it remains competitive with more advanced state-of-the-art models on open-ended generation tasks. We release model weights, tokenizer and source code used for data processing and model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.18375v3">Progressive Localisation in Localist LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      This paper demonstrates that progressive localization, the gradual increase of attention locality from early distributed layers to late localized layers, represents the optimal architecture for creating interpretable large language models (LLMs) while preserving performance. Through systematic experimentation with GPT-2 fine-tuned on The Psychology of Artificial Superintelligence, we evaluate five locality configurations: two uniform baselines (fully distributed and fully localist) and three progressive polynomial schedules. We investigate whether interpretability constraints can be aligned with natural semantic structure while being applied strategically across network depth. We demonstrate that progressive semantic localization, combining adaptive semantic block partitioning with steep polynomial locality schedules, achieves near-baseline language modeling performance while providing interpretable attention patterns. Multiple independent training runs with different random seeds establish that results are statistically robust and highly reproducible. The approach dramatically outperforms both fixed-window localization and naive uniform locality constraints. Analysis reveals that maintaining flexibility through low-fidelity constraints preserves model capacity while providing interpretability benefits, and that steep schedules concentrating locality in decision-critical final layers while preserving distributed learning in early layers achieve near-baseline attention distribution characteristics. These findings demonstrate that interpretability mechanisms should align with semantic structure to achieve practical performance-interpretability tradeoffs for trustworthy AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.06522v3">Fixing It in Post: A Comparative Study of LLM Post-Training Data Quality and Model Performance</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Recent work on large language models (LLMs) has increasingly focused on post-training and alignment with datasets curated to enhance instruction following, world knowledge, and specialized skills. However, most post-training datasets used in leading open- and closed-source LLMs remain inaccessible to the public, with limited information about their construction process. This lack of transparency has motivated the recent development of open-source post-training corpora. While training on these open alternatives can yield performance comparable to that of leading models, systematic comparisons remain challenging due to the significant computational cost of conducting them rigorously at scale, and are therefore largely absent. As a result, it remains unclear how specific samples, task types, or curation strategies influence downstream performance when assessing data quality. In this work, we conduct the first comprehensive side-by-side analysis of two prominent open post-training datasets: Tulu-3-SFT-Mix and SmolTalk. Using the Magpie framework, we annotate each sample with detailed quality metrics, including turn structure (single-turn vs. multi-turn), task category, input quality, and response quality, and we derive statistics that reveal structural and qualitative similarities and differences between the two datasets. Based on these insights, we design a principled curation recipe that produces a new data mixture, TuluTalk, which contains 14% fewer samples than either source dataset while matching or exceeding their performance on key benchmarks. Our findings offer actionable insights for constructing more effective post-training datasets that improve model performance within practical resource limits. To support future research, we publicly release both the annotated source datasets and our curated TuluTalk mixture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09549v2">Chasing Shadows: Pitfalls in LLM Security Research</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 About to appear at NDSS'26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly prevalent in security research. Their unique characteristics, however, introduce challenges that undermine established paradigms of reproducibility, rigor, and evaluation. Prior work has identified common pitfalls in traditional machine learning research, but these studies predate the advent of LLMs. In this paper, we identify nine common pitfalls that have become (more) relevant with the emergence of LLMs and that can compromise the validity of research involving them. These pitfalls span the entire computation process, from data collection, pre-training, and fine-tuning to prompting and evaluation. We assess the prevalence of these pitfalls across all 72 peer-reviewed papers published at leading Security and Software Engineering venues between 2023 and 2024. We find that every paper contains at least one pitfall, and each pitfall appears in multiple papers. Yet only 15.7% of the present pitfalls were explicitly discussed, suggesting that the majority remain unrecognized. To understand their practical impact, we conduct four empirical case studies showing how individual pitfalls can mislead evaluation, inflate performance, or impair reproducibility. Based on our findings, we offer actionable guidelines to support the community in future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.15101v2">Cost-aware LLM-based Online Dataset Annotation</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled automated dataset labeling with minimal human supervision. While majority voting across multiple LLMs can improve label reliability by mitigating individual model biases, it incurs high computational costs due to repeated querying. In this work, we propose a novel online framework, Cost-aware Majority Voting (CaMVo), for efficient and accurate LLM-based dataset annotation. CaMVo adaptively selects a subset of LLMs for each data instance based on contextual embeddings, balancing confidence and cost without requiring pre-training or ground-truth labels. Leveraging a LinUCB-based selection mechanism and a Bayesian estimator over confidence scores, CaMVo estimates a lower bound on labeling accuracy for each LLM and aggregates responses through weighted majority voting. Our empirical evaluation on the MMLU and IMDB Movie Review datasets demonstrates that CaMVo achieves comparable or superior accuracy to full majority voting while significantly reducing labeling costs. This establishes CaMVo as a practical and robust solution for cost-efficient annotation in dynamic labeling environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13077v1">LikeBench: Evaluating Subjective Likability in LLMs for Personalization</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      A personalized LLM should remember user facts, apply them correctly, and adapt over time to provide responses that the user prefers. Existing LLM personalization benchmarks are largely centered on two axes: accurately recalling user information and accurately applying remembered information in downstream tasks. We argue that a third axis, likability, is both subjective and central to user experience, yet under-measured by current benchmarks. To measure likability holistically, we introduce LikeBench, a multi-session, dynamic evaluation framework that measures likability across multiple dimensions by how much an LLM can adapt over time to a user's preferences to provide more likable responses. In LikeBench, the LLMs engage in conversation with a simulated user and learn preferences only from the ongoing dialogue. As the interaction unfolds, models try to adapt to responses, and after each turn, they are evaluated for likability across seven dimensions by the same simulated user. To the best of our knowledge, we are the first to decompose likability into multiple diagnostic metrics: emotional adaptation, formality matching, knowledge adaptation, reference understanding, conversation length fit, humor fit, and callback, which makes it easier to pinpoint where a model falls short. To make the simulated user more realistic and discriminative, LikeBench uses fine-grained, psychologically grounded descriptive personas rather than the coarse high/low trait rating based personas used in prior work. Our benchmark shows that strong memory performance does not guarantee high likability: DeepSeek R1, with lower memory accuracy (86%, 17 facts/profile), outperformed Qwen3 by 28% on likability score despite Qwen3's higher memory accuracy (93%, 43 facts/profile). Even SOTA models like GPT-5 adapt well in short exchanges but show only limited robustness in longer, noisier interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.00507v2">Towards Resource-Efficient Serverless LLM Inference with SLINFER</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 To appear at HPCA 2026
    </div>
    <details class="paper-abstract">
      The rise of LLMs has driven demand for private serverless deployments, characterized by moderate-sized models and infrequent requests. While existing serverless solutions follow exclusive GPU allocation, we take a step back to explore modern platforms and find that: Emerging CPU architectures with built-in accelerators are capable of serving LLMs but remain underutilized, and both CPUs and GPUs can accommodate multiple LLMs simultaneously. We propose SLINFER, a resource-efficient serverless inference scheme tailored for small- to mid-sized LLMs that enables elastic and on-demand sharing across heterogeneous hardware. SLINFER tackles three fundamental challenges: (1) precise, fine-grained compute resource allocation at token-level to handle fluctuating computational demands; (2) a coordinated and forward-looking memory scaling mechanism to detect out-of-memory hazards and reduce operational overhead; and (3) a dual approach that consolidates fragmented instances through proactive preemption and reactive bin-packing. Experimental results on 4 32-core CPUs and 4 A100 GPUs show that SLINFER improves serving capacity by 47% - 62% through sharing, while further leveraging CPUs boosts this to 86% - 154%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13063v1">LLM Rationalis? Measuring Bargaining Capabilities of AI Negotiators</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Published in the First Workshop on Multi-Turn Interactions in Large Language Models at Neurips 2025
    </div>
    <details class="paper-abstract">
      Bilateral negotiation is a complex, context-sensitive task in which human negotiators dynamically adjust anchors, pacing, and flexibility to exploit power asymmetries and informal cues. We introduce a unified mathematical framework for modeling concession dynamics based on a hyperbolic tangent curve, and propose two metrics burstiness tau and the Concession-Rigidity Index (CRI) to quantify the timing and rigidity of offer trajectories. We conduct a large-scale empirical comparison between human negotiators and four state-of-the-art large language models (LLMs) across natural-language and numeric-offers settings, with and without rich market context, as well as six controlled power-asymmetry scenarios. Our results reveal that, unlike humans who smoothly adapt to situations and infer the opponents position and strategies, LLMs systematically anchor at extremes of the possible agreement zone for negotiations and optimize for fixed points irrespective of leverage or context. Qualitative analysis further shows limited strategy diversity and occasional deceptive tactics used by LLMs. Moreover the ability of LLMs to negotiate does not improve with better models. These findings highlight fundamental limitations in current LLM negotiation capabilities and point to the need for models that better internalize opponent reasoning and context-dependent strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13040v1">Understanding Structured Financial Data with LLMs: A Case Study on Fraud Detection</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Detecting fraud in financial transactions typically relies on tabular models that demand heavy feature engineering to handle high-dimensional data and offer limited interpretability, making it difficult for humans to understand predictions. Large Language Models (LLMs), in contrast, can produce human-readable explanations and facilitate feature analysis, potentially reducing the manual workload of fraud analysts and informing system refinements. However, they perform poorly when applied directly to tabular fraud detection due to the difficulty of reasoning over many features, the extreme class imbalance, and the absence of contextual information. To bridge this gap, we introduce FinFRE-RAG, a two-stage approach that applies importance-guided feature reduction to serialize a compact subset of numeric/categorical attributes into natural language and performs retrieval-augmented in-context learning over label-aware, instance-level exemplars. Across four public fraud datasets and three families of open-weight LLMs, FinFRE-RAG substantially improves F1/MCC over direct prompting and is competitive with strong tabular baselines in several settings. Although these LLMs still lag behind specialized classifiers, they narrow the performance gap and provide interpretable rationales, highlighting their value as assistive tools in fraud analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.10449v2">When Reject Turns into Accept: Quantifying the Vulnerability of LLM-Based Scientific Reviewers to Indirect Prompt Injection</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      The landscape of scientific peer review is rapidly evolving with the integration of Large Language Models (LLMs). This shift is driven by two parallel trends: the widespread individual adoption of LLMs by reviewers to manage workload (the "Lazy Reviewer" hypothesis) and the formal institutional deployment of AI-powered assessment systems by conferences like AAAI and Stanford's Agents4Science. This study investigates the robustness of these "LLM-as-a-Judge" systems (both illicit and sanctioned) to adversarial PDF manipulation. Unlike general jailbreaks, we focus on a distinct incentive: flipping "Reject" decisions to "Accept," for which we develop a novel evaluation metric which we term as WAVS (Weighted Adversarial Vulnerability Score). We curated a dataset of 200 scientific papers and adapted 15 domain-specific attack strategies to this task, evaluating them across 13 Language Models, including GPT-5, Claude Haiku, and DeepSeek. Our results demonstrate that obfuscation strategies like "Maximum Mark Magyk" successfully manipulate scores, achieving alarming decision flip rates even in large-scale models. We will release our complete dataset and injection framework to facilitate more research on this topic.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.09321v3">ObliInjection: Order-Oblivious Prompt Injection Attack to LLM Agents with Multi-source Data</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 To appear in NDSS 2026. For slides, see https://people.duke.edu/~zg70/code/PromptInjection.pdf
    </div>
    <details class="paper-abstract">
      Prompt injection attacks aim to contaminate the input data of an LLM to mislead it into completing an attacker-chosen task instead of the intended task. In many applications and agents, the input data originates from multiple sources, with each source contributing a segment of the overall input. In these multi-source scenarios, an attacker may control only a subset of the sources and contaminate the corresponding segments, but typically does not know the order in which the segments are arranged within the input. Existing prompt injection attacks either assume that the entire input data comes from a single source under the attacker's control or ignore the uncertainty in the ordering of segments from different sources. As a result, their success is limited in domains involving multi-source data. In this work, we propose ObliInjection, the first prompt injection attack targeting LLM applications and agents with multi-source input data. ObliInjection introduces two key technical innovations: the order-oblivious loss, which quantifies the likelihood that the LLM will complete the attacker-chosen task regardless of how the clean and contaminated segments are ordered; and the orderGCG algorithm, which is tailored to minimize the order-oblivious loss and optimize the contaminated segments. Comprehensive experiments across three datasets spanning diverse application domains and twelve LLMs demonstrate that ObliInjection is highly effective, even when only one out of 6-100 segments in the input data is contaminated. Our code and data are available at: https://github.com/ReachalWang/ObliInjection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23188v3">Diagnose, Localize, Align: A Full-Stack Framework for Reliable LLM Multi-Agent Systems under Instruction Conflicts</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Upon further review, we realized that the version submitted to arXiv was not the final draft and omits crucial results and discussion. To avoid confusion and ensure the integrity of the record, we request withdrawal and will resubmit once the complete work is ready
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-powered multi-agent systems (MAS) have rapidly advanced collaborative reasoning, tool use, and role-specialized coordination in complex tasks. However, reliability-critical deployment remains hindered by a systemic failure mode: hierarchical compliance under instruction conflicts (system-user, peer-peer), where agents misprioritize system-level rules in the presence of competing demands. Moreover, widely used macro-level metrics (e.g., pass@k) obscure these micro-level violations and offer little actionable guidance for remedy. In this work, we present a full-stack, three-stage framework: (1) Diagnose - Contextualized Role Adherence Score (CRAS), a query-wise, context-aware scoring metric that decomposes role adherence into four measurable dimensions; (2) Localize - attention drift analysis revealing that instruction conflicts are resolved by attention heads that are largely concentrated in middle layers; (3) Align - Surgical Alignment of Instruction Layers (SAIL), which installs LoRA only on the localized focal layers and optimizes a token-weighted DPO-style preference objective that credits tokens by their focal attentional contribution. Across standard benchmarks and MAS frameworks, our surgical approach improves instruction hierarchy compliance (e.g., +5.60% with AutoGen on MedQA) without full-model finetuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12928v1">PROSERVE: Unified Multi-Priority Request Scheduling for LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      The widespread deployment of large language models (LLMs) for interactive applications necessitates serving systems that can handle thousands of concurrent requests with diverse Service Level Objective (SLO) requirements. A critical yet often overlooked dimension in this context is the inherent priority difference among clients; for instance, business-critical functions demand higher performance guarantees, as fulfilling such requests yields significantly greater business value. However, existing LLM serving schedulers fail to jointly optimize for both SLO attainment and client-level priorities. To bridge this gap, we first \textit{formalize multi-priority request scheduling as a service gain maximization problem}, where satisfying latency requirements for requests of different priorities contributes varying levels of gain. We then propose PROSERVE, a unified two-tier scheduling framework designed to maximize overall service gain. At the engine level, SlideBatching dynamically adapts batch formation and request ordering under varying load conditions, employing a sliding boundary mechanism to balance deadline-first and density-first strategies. At the service level, GoRouting performs gain-oriented and capability-aware dispatching across distributed instances, proactively reserving capacity for future high-priority or long requests. Extensive evaluation across four open-source datasets and a real-world industrial trace demonstrates that \systemname{} consistently outperforms state-of-the-art baselines, improving system gain by up to 35% and boosting SLO attainment by up to 52%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12922v1">LLM-based Personalized Portfolio Recommender: Integrating Large Language Models and Reinforcement Learning for Intelligent Investment Strategy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      In modern financial markets, investors increasingly seek personalized and adaptive portfolio strategies that reflect their individual risk preferences and respond to dynamic market conditions. Traditional rule-based or static optimization approaches often fail to capture the nonlinear interactions among investor behavior, market volatility, and evolving financial objectives. To address these limitations, this paper introduces the LLM-based Personalized Portfolio Recommender , an integrated framework that combines Large Language Models, reinforcement learning, and individualized risk preference modeling to support intelligent investment decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.12914v1">CTIGuardian: A Few-Shot Framework for Mitigating Privacy Leakage in Fine-Tuned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Accepted at the 18th Cybersecurity Experimentation and Test Workshop (CSET), in conjunction with ACSAC 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are often fine-tuned to adapt their general-purpose knowledge to specific tasks and domains such as cyber threat intelligence (CTI). Fine-tuning is mostly done through proprietary datasets that may contain sensitive information. Owners expect their fine-tuned model to not inadvertently leak this information to potentially adversarial end users. Using CTI as a use case, we demonstrate that data-extraction attacks can recover sensitive information from fine-tuned models on CTI reports, underscoring the need for mitigation. Retraining the full model to eliminate this leakage is computationally expensive and impractical. We propose an alternative approach, which we call privacy alignment, inspired by safety alignment in LLMs. Just like safety alignment teaches the model to abide by safety constraints through a few examples, we enforce privacy alignment through few-shot supervision, integrating a privacy classifier and a privacy redactor, both handled by the same underlying LLM. We evaluate our system, called CTIGuardian, using GPT-4o mini and Mistral-7B Instruct models, benchmarking against Presidio, a named entity recognition (NER) baseline. Results show that CTIGuardian provides a better privacy-utility trade-off than NER based models. While we demonstrate its effectiveness on a CTI use case, the framework is generic enough to be applicable to other sensitive domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.01198v2">Conveying Imagistic Thinking in Traditional Chinese Medicine Translation: A Prompt Engineering and LLM-Based Evaluation Framework</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 This is a duplicate of arXiv:2511.23059
    </div>
    <details class="paper-abstract">
      Traditional Chinese Medicine (TCM) theory is built on imagistic thinking, in which medical principles and diagnostic and therapeutic logic are structured through metaphor and metonymy. However, existing English translations largely rely on literal rendering, making it difficult for target-language readers to reconstruct the underlying conceptual networks and apply them in clinical practice. This study adopted a human-in-the-loop (HITL) framework and selected four passages from the medical canon Huangdi Neijing that are fundamental in theory. Through prompt-based cognitive scaffolding, DeepSeek V3.1 was guided to identify metaphor and metonymy in the source text and convey the theory in translation. In the evaluation stage, ChatGPT 5 Pro and Gemini 2.5 Pro were instructed by prompts to simulate three types of real-world readers. Human translations, baseline model translations, and prompt-adjusted translations were scored by the simulated readers across five cognitive dimensions, followed by structured interviews and Interpretative Phenomenological Analysis (IPA). Results show that the prompt-adjusted LLM translations perform best across all five dimensions, with high cross-model and cross-role consistency. The interview themes reveal differences between human and machine translation, effective strategies for metaphor and metonymy transfer, and readers' cognitive preferences. This study provides a cognitive, efficient, and replicable HITL methodological pathway for the translation of ancient, concept-dense texts such as TCM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13914v1">Context Branching for LLM Conversations: A Version Control Approach to Exploratory Programming</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 11 pages, 4 figures, 2 tables, 1 code snippet, 4 algorithms
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become integral to software engineering workflows, yet their effectiveness degrades significantly in multi-turn conversations. Recent studies demonstrate an average 39% performance drop when instructions are delivered across multiple turns, with models making premature assumptions and failing to course correct (Laban et al., 2025). This degradation is particularly problematic in exploratory programming tasks where developers need to investigate alternative approaches without committing to a single path. Current solutions force users into a false dichotomy: continue in a context-polluted conversation where the LLM becomes increasingly confused, or start fresh and lose all accumulated context. We present ContextBranch, a conversation management system that applies version control semantics to LLM interactions. ContextBranch provides four core primitives--checkpoint, branch, switch, and inject--enabling users to capture conversation state, explore alternatives in isolation, and selectively merge insights. We evaluate ContextBranch through a controlled experiment with 30 software engineering scenarios featuring intentionally polluting explorations. Branched conversations achieved higher response quality compared to linear conversations, with large improvements in focus and context awareness. Benefits were concentrated in complex scenarios involving conceptually distant explorations. Branching reduced context size by 58.1% (31.0 to 13.0 messages), eliminating irrelevant exploratory content. Our work establishes conversation branching as a fundamental primitive for AI-assisted exploratory work, demonstrating that isolation prevents context pollution when exploring alternatives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.17985v3">How K-12 Educators Use AI: LLM-Assisted Qualitative Analysis at Scale</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      This study investigates how K-12 educators use generative AI tools in real-world instructional contexts and how large language models (LLMs) can support scalable qualitative analysis of these interactions. Drawing on over 13,000 unscripted educator-AI conversations from an open-access platform, we examine educators' use of AI for lesson planning, differentiation, assessment, and pedagogical reflection. Methodologically, we introduce a replicable, LLM-assisted qualitative analysis pipeline that supports inductive theme discovery, codebook development, and large-scale annotation while preserving researcher control over conceptual synthesis. Empirically, the findings surface concrete patterns in how educators prompt, adapt, and evaluate AI-generated suggestions as part of their instructional reasoning. This work demonstrates the feasibility of combining LLM support with qualitative rigor to analyze complex educator behaviors at scale and inform the design of AI-powered educational tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13898v1">Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Progress on training and architecture strategies has enabled LLMs with millions of tokens in context length. However, empirical evidence suggests that such long-context LLMs can consume far more text than they can reliably use. On the other hand, it has been shown that inference-time compute can be used to scale performance of LLMs, often by generating thinking tokens, on challenging tasks involving multi-step reasoning. Through controlled experiments on sandbox long-context tasks, we find that such inference-time strategies show rapidly diminishing returns and fail at long context. We attribute these failures to score dilution, a phenomenon inherent to static self-attention. Further, we show that current inference-time strategies cannot retrieve relevant long-context signals under certain conditions. We propose a simple method that, through targeted gradient updates on the given context, provably overcomes limitations of static self-attention. We find that this shift in how inference-time compute is spent leads to consistently large performance improvements across models and long-context benchmarks. Our method leads to large 12.6 and 14.1 percentage point improvements for Qwen3-4B on average across subsets of LongBench-v2 and ZeroScrolls benchmarks. The takeaway is practical: for long context, a small amount of context-specific training is a better use of inference compute than current inference-time scaling strategies like producing more thinking tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.20768v2">RAGRank: Using PageRank to Counter Poisoning in CTI LLM Pipelines</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Presented as a poster at the Annual Computer Security Applications Conference (ACSAC) 2025
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has emerged as the dominant architectural pattern to operationalize Large Language Model (LLM) usage in Cyber Threat Intelligence (CTI) systems. However, this design is susceptible to poisoning attacks, and previously proposed defenses can fail for CTI contexts as cyber threat information is often completely new for emerging attacks, and sophisticated threat actors can mimic legitimate formats, terminology, and stylistic conventions. To address this issue, we propose that the robustness of modern RAG defenses can be accelerated by applying source credibility algorithms on corpora, using PageRank as an example. In our experiments, we demonstrate quantitatively that our algorithm applies a lower authority score to malicious documents while promoting trusted content, using the standardized MS MARCO dataset. We also demonstrate proof-of-concept performance of our algorithm on CTI documents and feeds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13886v1">OPTIMA: Optimal One-shot Pruning for LLMs via Quadratic Programming Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-12-15
    </div>
    <details class="paper-abstract">
      Post-training model pruning is a promising solution, yet it faces a trade-off: simple heuristics that zero weights are fast but degrade accuracy, while principled joint optimization methods recover accuracy but are computationally infeasible at modern scale. One-shot methods such as SparseGPT offer a practical trade-off in optimality by applying efficient, approximate heuristic weight updates. To close this gap, we introduce OPTIMA, a practical one-shot post-training pruning method that balances accuracy and scalability. OPTIMA casts layer-wise weight reconstruction after mask selection as independent, row-wise Quadratic Programs (QPs) that share a common layer Hessian. Solving these QPs yields the per-row globally optimal update with respect to the reconstruction objective given the estimated Hessian. The shared-Hessian structure makes the problem highly amenable to batching on accelerators. We implement an accelerator-friendly QP solver that accumulates one Hessian per layer and solves many small QPs in parallel, enabling one-shot post-training pruning at scale on a single accelerator without fine-tuning. OPTIMA integrates with existing mask selectors and consistently improves zero-shot performance across multiple LLM families and sparsity regimes, yielding up to 3.97% absolute accuracy improvement. On an NVIDIA H100, OPTIMA prunes a 8B-parameter transformer end-to-end in 40 hours with 60GB peak memory. Together, these results set a new state-of-the-art accuracy-efficiency trade-offs for one-shot post-training pruning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.09557v3">LLM Inference Beyond a Single Node: From Bottlenecks to Mitigations with Fast All-Reduce Communication</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 12 Figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to grow in size, distributed inference has become increasingly important. Model-parallel strategies must now efficiently scale not only across multiple GPUs but also across multiple nodes. In this work, we present a detailed performance study of multi-node distributed inference using LLMs on GPU-based supercomputers. We conduct experiments with several state-of-the-art inference engines alongside YALIS, a research-oriented prototype engine designed for controlled experimentation. We analyze the strong-scaling behavior of different model-parallel schemes and identify key bottlenecks. Since all-reduce operations are a common performance bottleneck, we develop NVRAR, a hierarchical all-reduce algorithm based on recursive doubling with NVSHMEM. NVRAR achieves up to 1.9x-3.6x lower latency than NCCL for message sizes between 128 KB and 2 MB on HPE Slingshot and InfiniBand interconnects. Integrated into YALIS, NVRAR achieves up to a 1.72x reduction in end-to-end batch latency for the Llama 3.1 405B model in multi-node decode-heavy workloads using tensor parallelism.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.13860v1">Verification-Guided Context Optimization for Tool Calling via Hierarchical LLMs-as-Editors</a></div>
    <div class="paper-meta">
      📅 2025-12-15
      | 💬 Accepted by AAAI 2026 Workshop on Agentic AI Benchmarks and Applications for Enterprise Tasks
    </div>
    <details class="paper-abstract">
      Tool calling enables large language models (LLMs) to interact with external environments through tool invocation, providing a practical way to overcome the limitations of pretraining. However, the effectiveness of tool use depends heavily on the quality of the associated documentation and knowledge base context. These materials are usually written for human users and are often misaligned with how LLMs interpret information. This problem is even more pronounced in industrial settings, where hundreds of tools with overlapping functionality create challenges in scalability, variability, and ambiguity. We propose Verification-Guided Context Optimization (VGCO), a framework that uses LLMs as editors to automatically refine tool-related documentation and knowledge base context. VGCO works in two stages. First, Evaluation collects real-world failure cases and identifies mismatches between tools and their context. Second, Optimization performs hierarchical editing through offline learning with structure-aware, in-context optimization. The novelty of our LLM editors has three main aspects. First, they use a hierarchical structure that naturally integrates into the tool-calling workflow. Second, they are state-aware, action-specific, and verification-guided, which constrains the search space and enables efficient, targeted improvements. Third, they enable cost-efficient sub-task specialization, either by prompt engineering large editor models or by post-training smaller editor models. Unlike prior work that emphasizes multi-turn reasoning, VGCO focuses on the single-turn, large-scale tool-calling problem and achieves significant improvements in accuracy, robustness, and generalization across LLMs.
    </details>
</div>
