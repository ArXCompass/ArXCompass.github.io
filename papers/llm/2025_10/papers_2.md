# llm - 2025_10

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08952v2">When LLM Agents Meet Graph Optimization: An Automated Data Quality Improvement Approach</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 12 pages, 7figures
    </div>
    <details class="paper-abstract">
      Text-attributed graphs (TAGs) have become a key form of graph-structured data in modern data management and analytics, combining structural relationships with rich textual semantics for diverse applications. However, the effectiveness of analytical models, particularly graph neural networks (GNNs), is highly sensitive to data quality. Our empirical analysis shows that both conventional and LLM-enhanced GNNs degrade notably under textual, structural, and label imperfections, underscoring TAG quality as a key bottleneck for reliable analytics. Existing studies have explored data-level optimization for TAGs, but most focus on specific degradation types and target a single aspect like structure or label, lacking a systematic and comprehensive perspective on data quality improvement. To address this gap, we propose LAGA (Large Language and Graph Agent), a unified multi-agent framework for comprehensive TAG quality optimization. LAGA formulates graph quality control as a data-centric process, integrating detection, planning, action, and evaluation agents into an automated loop. It holistically enhances textual, structural, and label aspects through coordinated multi-modal optimization. Extensive experiments on 5 datasets and 16 baselines across 9 scenarios demonstrate the effectiveness, robustness and scalability of LAGA, confirming the importance of data-centric quality optimization for reliable TAG analytics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17132v1">Do LLMs Recognize Your Latent Preferences? A Benchmark for Latent Information Discovery in Personalized Interaction</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at producing broadly relevant text, but this generality becomes a limitation when user-specific preferences are required, such as recommending restaurants or planning travel. In these scenarios, users rarely articulate every preference explicitly; instead, much of what they care about remains latent, waiting to be inferred. This raises a fundamental question: Can LLMs uncover and reason about such latent information through conversation? We address this problem by introducing a unified benchmark for evaluating latent information discovery - the ability of LLMs to reveal and utilize hidden user attributes through multi-turn interaction. The benchmark spans three progressively realistic settings: the classic 20 Questions game, Personalized Question Answering, and Personalized Text Summarization. All tasks share a tri-agent framework (User, Assistant, Judge) enabling turn-level evaluation of elicitation and adaptation. Our results reveal that while LLMs can indeed surface latent information through dialogue, their success varies dramatically with context: from 32% to 98%, depending on task complexity, topic, and number of hidden attributes. This benchmark provides the first systematic framework for studying latent information discovery in personalized interaction, highlighting that effective preference inference remains an open frontier for building truly adaptive AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08164v2">BLUR: A Bi-Level Optimization Approach for LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Enabling large language models (LLMs) to unlearn knowledge and capabilities acquired during training has proven vital for ensuring compliance with data regulations and promoting ethical practices in generative AI. Although there are growing interests in developing various unlearning algorithms, it remains unclear how to best formulate the unlearning problem. The most popular formulation uses a weighted sum of forget and retain loss, but it often leads to performance degradation due to the inherent trade-off between forget and retain losses. In this work, we argue that it is important to model the hierarchical structure of the unlearning problem, where the forget problem (which \textit{unlearns} certain knowledge and/or capabilities) takes priority over the retain problem (which preserves model utility). This hierarchical structure naturally leads to a bi-level optimization formulation where the lower-level objective focuses on minimizing the forget loss, while the upper-level objective aims to maintain the model's utility. Based on this new formulation, we propose a novel algorithm, termed Bi-Level UnleaRning (\texttt{BLUR}), which not only possesses strong theoretical guarantees but more importantly, delivers superior performance. In particular, our extensive experiments demonstrate that \texttt{BLUR} consistently outperforms all the state-of-the-art algorithms across various unlearning tasks, models, and metrics. Codes are available at https://github.com/OptimAI-Lab/BLURLLMUnlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05831v4">Leveraging Robust Optimization for LLM Alignment under Distribution Shifts</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Preference alignment methods are increasingly critical for steering large language models (LLMs) to generate outputs consistent with human values. While recent approaches often rely on synthetic data generated by LLMs for scalability and cost-efficiency reasons, this reliance can introduce distribution shifts that undermine the nuanced representation of human preferences needed for desirable outputs. In this paper, we propose a novel distribution-aware optimization framework that improves preference alignment despite such shifts. Our approach first leverages well-learned classifiers to assign a calibration value to each training sample, quantifying its alignment with the target human-preferred distribution. These values are then incorporated into a robust optimization objective that minimizes the worst-case loss over regions of the data space most relevant to human preferences. By explicitly focusing optimization on the target distribution, our approach mitigates the impact of distributional mismatch and improves the generation of responses that better reflect intended values.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14835v2">Towards Automated Verification of LLM-Synthesized C Programs</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      We present \synver{}, a novel synthesis and verification framework for C programs, that deploys a Large Language Model (LLM) to search for a candidate program that satisfies the given specification. Our key idea is to impose syntactic and semantic biases on programs generated by LLMs, such that the synthesized program is more amenable to automated verification. Based on this idea, we propose a novel specification-verification tool, built on top of Verified Software Toolchain, that help automate the process. Our experiments on a diverse set of benchmarks drawn from the deductive program synthesis community, shows that this approach is scalable and extensible. The benchmarks constitute of specifications comprising of basic coding examples, Separation Logic based assertions, and API specifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17057v1">The Ends Justify the Thoughts: RL-Induced Motivated Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      The use of reinforcement learning (RL) with chain-of-thought (CoT) reasoning has emerged as a promising approach for developing more capable language models. In turn, this has led to investigation of CoT monitoring as a compelling method for detecting harmful behaviors such as reward hacking, under the assumption that models' reasoning processes reflect their internal decision-making. In practice, LLM training often produces unintended behaviors due to imperfect reward signals, leading models to develop misaligned tendencies. A common corrective approach is to apply post-hoc instructions to avoid problematic behaviors like sycophancy, but what happens to the model's reasoning process when these instructions conflict with learned behaviors? We investigate this question in simple settings and find that models engage in systematic motivated reasoning -- generating plausible-sounding justifications for violating their instructions while downplaying potential harms. Beyond being an interesting property of training, we find that while motivated reasoning can be detected by most frontier reasoning models, smaller LLM judges can fail to identify a portion of it, and in rare cases can themselves be persuaded that the reasoning is correct, despite it contradicting clear instructions. This capability gap raises concerns that as models become more sophisticated, their motivated reasoning may become increasingly difficult for monitors to detect. Our results underscore the need to account for motivated reasoning when relying on chain-of-thought processes for model evaluation and oversight. All code for this paper will be made available. WARNING: some examples in this paper may be upsetting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18169v1">Hearing Health in Home Healthcare: Leveraging LLMs for Illness Scoring and ALMs for Vocal Biomarker Extraction</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 The Second Workshop on GenAI for Health at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      The growing demand for home healthcare calls for tools that can support care delivery. In this study, we explore automatic health assessment from voice using real-world home care visit data, leveraging the diverse patient information it contains. First, we utilize Large Language Models (LLMs) to integrate Subjective, Objective, Assessment, and Plan (SOAP) notes derived from unstructured audio transcripts and structured vital signs into a holistic illness score that reflects a patient's overall health. This compact representation facilitates cross-visit health status comparisons and downstream analysis. Next, we design a multi-stage preprocessing pipeline to extract short speech segments from target speakers in home care recordings for acoustic analysis. We then employ an Audio Language Model (ALM) to produce plain-language descriptions of vocal biomarkers and examine their association with individuals' health status. Our experimental results benchmark both commercial and open-source LLMs in estimating illness scores, demonstrating their alignment with actual clinical outcomes, and revealing that SOAP notes are substantially more informative than vital signs. Building on the illness scores, we provide the first evidence that ALMs can identify health-related acoustic patterns from home care recordings and present them in a human-readable form. Together, these findings highlight the potential of LLMs and ALMs to harness heterogeneous in-home visit data for better patient monitoring and care.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18155v1">LLM-Based Multi-Agent System for Simulating and Analyzing Marketing and Consumer Behavior</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Accepted for publication at IEEE International Conference on e-Business Engineering ICEBE 2025, November 10-12, Buraydah, Saudi Arabia. 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Simulating consumer decision-making is vital for designing and evaluating marketing strategies before costly real-world deployment. However, post-event analyses and rule-based agent-based models (ABMs) struggle to capture the complexity of human behavior and social interaction. We introduce an LLM-powered multi-agent simulation framework that models consumer decisions and social dynamics. Building on recent advances in large language model simulation in a sandbox environment, our framework enables generative agents to interact, express internal reasoning, form habits, and make purchasing decisions without predefined rules. In a price-discount marketing scenario, the system delivers actionable strategy-testing outcomes and reveals emergent social patterns beyond the reach of conventional methods. This approach offers marketers a scalable, low-risk tool for pre-implementation testing, reducing reliance on time-intensive post-event evaluations and lowering the risk of underperforming campaigns.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18147v1">LLMs Encode How Difficult Problems Are</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Large language models exhibit a puzzling inconsistency: they solve complex problems yet frequently fail on seemingly simpler ones. We investigate whether LLMs internally encode problem difficulty in a way that aligns with human judgment, and whether this representation tracks generalization during reinforcement learning post-training. We train linear probes across layers and token positions on 60 models, evaluating on mathematical and coding subsets of Easy2HardBench. We find that human-labeled difficulty is strongly linearly decodable (AMC: $\rho \approx 0.88$) and exhibits clear model-size scaling, whereas LLM-derived difficulty is substantially weaker and scales poorly. Steering along the difficulty direction reveals that pushing models toward "easier" representations reduces hallucination and improves accuracy. During GRPO training on Qwen2.5-Math-1.5B, the human-difficulty probe strengthens and positively correlates with test accuracy across training steps, while the LLM-difficulty probe degrades and negatively correlates with performance. These results suggest that human annotations provide a stable difficulty signal that RL amplifies, while automated difficulty estimates derived from model performance become misaligned precisely as models improve. We release probe code and evaluation scripts to facilitate replication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02718v2">Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Increasing demand for Large Language Models (LLMs) services imposes substantial deployment and computation costs on providers. LLM routing offers a cost-efficient solution by directing queries to the optimal LLM based on model and query features. However, existing works primarily focus on offline scenarios and struggle to adapt to online settings with high query volume and constrained token budgets. In this work, we introduce the first training-free algorithm for online routing scenarios. Our algorithm leverages approximate nearest neighbor search to efficiently estimate query features and performs a one-time optimization over a small set of initial queries to learn a routing strategy that guides future routing. We provide theoretical guarantees demonstrating that our algorithm achieves a competitive ratio of $1 - o(1)$ under natural assumptions, which is further validated by extensive experiments across 3 benchmark datasets and 8 baselines, showing an average improvement of 3.55$\times$ in overall performance, 1.85$\times$ in cost efficiency, and nearly 4.25$\times$ in throughput. Our code is available at https://github.com/fzwark/PORT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18134v1">Measuring Reasoning in LLMs: a New Dialectical Angle</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      What does it truly mean for a language model to "reason"? Most current evaluations and benchmarks reward models' correct standalone answers--but correctness alone reveals little about the process that produced them. In this work, we explore a different perspective: reasoning is not a static chain of steps, but a dynamic trajectory where ideas interact, clash, and evolve into deeper insights. To capture this dynamic, we draw on a well-established philosophical tradition: \textit{dialectics}, where reasoning unfolds through thesis, antithesis, and synthesis. Building on this, we present SIEV, a structured framework that evaluates reasoning of LLMs through dialectics. Unlike conventional evaluations, SIEV assesses not only the conclusion a model reaches, but how it gets there: its ability to resolve tension, integrate distinct ideas, and synthesize higher-order reasoning. This lens uncovers significant reasoning gaps in state-of-the-art models even under saturated benchmarks like GSM and MMLU. For instance, GPT-5-chat, a recent model, loses over 40 points (out of 100) when evaluated with SIEV on GSM. Our findings highlight that adopting a process-oriented, philosophically grounded approach enables a deeper, more rigorous, and more discriminative assessment of LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18113v1">Investigating the Impact of Dark Patterns on LLM-Based Web Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 At IEEE S&P 2026
    </div>
    <details class="paper-abstract">
      As users increasingly turn to large language model (LLM) based web agents to automate online tasks, agents may encounter dark patterns: deceptive user interface designs that manipulate users into making unintended decisions. Although dark patterns primarily target human users, their potentially harmful impacts on LLM-based generalist web agents remain unexplored. In this paper, we present the first study that investigates the impact of dark patterns on the decision-making process of LLM-based generalist web agents. To achieve this, we introduce LiteAgent, a lightweight framework that automatically prompts agents to execute tasks while capturing comprehensive logs and screen-recordings of their interactions. We also present TrickyArena, a controlled environment comprising web applications from domains such as e-commerce, streaming services, and news platforms, each containing diverse and realistic dark patterns that can be selectively enabled or disabled. Using LiteAgent and TrickyArena, we conduct multiple experiments to assess the impact of both individual and combined dark patterns on web agent behavior. We evaluate six popular LLM-based generalist web agents across three LLMs and discover that when there is a single dark pattern present, agents are susceptible to it an average of 41% of the time. We also find that modifying dark pattern UI attributes through visual design changes or HTML code adjustments and introducing multiple dark patterns simultaneously can influence agent susceptibility. This study emphasizes the need for holistic defense mechanisms in web agents, encompassing both agent-specific protections and broader web safety measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09572v2">Rethinking LLM Uncertainty: A Multi-Agent Approach to Estimating Black-Box Model Uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Quantifying uncertainty in black-box LLMs is vital for reliable responses and scalable oversight. Existing methods, which gauge a model's uncertainty through evaluating self-consistency in responses to the target query, can be misleading: an LLM may confidently provide an incorrect answer to a target query, yet give a confident and accurate answer to that same target query when answering a knowledge-preserving perturbation of the query. We systematically analyze the model behaviors and demonstrate that this discrepancy stems from suboptimal retrieval of parametric knowledge, often due to contextual biases that prevent consistent access to stored knowledge. We then introduce DiverseAgentEntropy, a novel, theoretically-grounded method employing multi-agent interaction across diverse query variations for uncertainty estimation of black-box LLMs. This approach more accurately assesses an LLM's true uncertainty and improves hallucination detection, outperforming existing self-consistency based techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18112v1">Does Reasoning Help LLM Agents Play Dungeons and Dragons? A Prompt Engineering Experiment</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Published at the Wordplay: When Language Meets Games Workshop (EMNLP 2025)
    </div>
    <details class="paper-abstract">
      This paper explores the application of Large Language Models (LLMs) and reasoning to predict Dungeons & Dragons (DnD) player actions and format them as Avrae Discord bot commands. Using the FIREBALL dataset, we evaluated a reasoning model, DeepSeek-R1-Distill-LLaMA-8B, and an instruct model, LLaMA-3.1-8B-Instruct, for command generation. Our findings highlight the importance of providing specific instructions to models, that even single sentence changes in prompts can greatly affect the output of models, and that instruct models are sufficient for this task compared to reasoning models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18096v1">A Benchmark Dataset And LLMs Comparison For NFR Classification With Explainable AI</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Non-Functional Requirements (NFRs) play a critical role in determining the overall quality and user satisfaction of software systems. Accurately identifying and classifying NFRs is essential to ensure that software meets performance, usability, and reliability expectations. However, manual identification of NFRs from documentation is time-consuming and prone to errors, necessitating automated solutions. Before implementing any automated solution, a robust and comprehensive dataset is essential. To build such a dataset, we collected NFRs from various Project Charters and Open Source Software Documentation. This enhanced the technical depth and usability of an already existing NFR dataset. We categorized NFRs into sub-classes and identified needs using widely used Large Language Models to facilitate automation. After classifying the NFRs, we compared the classification results of the selected LLMs: RoBERTa, CodeBERT, Gemma-2, Phi-3, Mistral-8B, and Llama-3.1-8B using various evaluation metrics, including precision, recall, F1-score, and lime scores. Among these models, Gemma-2 achieved the best results with a precision of 0.87, recall of 0.89, and F1-score of 0.88, alongside a lime hit score of 78 out of 80. Phi-3 closely followed with a precision of 0.85, recall of 0.87, F1-score of 0.86, and the highest lime hit score of 79. By improving the contextual foundation, this integration enhanced the model's comprehension of technical aspects and user requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18095v1">SMaRT: Select, Mix, and ReinvenT -- A Strategy Fusion Framework for LLM-Driven Reasoning and Planning</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have redefined complex task automation with exceptional generalization capabilities. Despite these advancements, state-of-the-art methods rely on single-strategy prompting, missing the synergy of diverse reasoning approaches. No single strategy excels universally, highlighting the need for frameworks that fuse strategies to maximize performance and ensure robustness. We introduce the Select, Mix, and ReinvenT (SMaRT) framework, an innovative strategy fusion approach designed to overcome this constraint by creating balanced and efficient solutions through the seamless integration of diverse reasoning strategies. Unlike existing methods, which employ LLMs merely as evaluators, SMaRT uses them as intelligent integrators, unlocking the "best of all worlds" across tasks. Extensive empirical evaluations across benchmarks in reasoning, planning, and sequential decision-making highlight the robustness and adaptability of SMaRT. The framework consistently outperforms state-of-the-art baselines in solution quality, constraint adherence, and performance metrics. This work redefines LLM-driven decision-making by pioneering a new paradigm in cross-strategy calibration, unlocking superior outcomes for reasoning systems and advancing the boundaries of self-refining methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18081v1">Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit strong but shallow alignment: they directly refuse harmful queries when a refusal is expected at the very start of an assistant turn, yet this protection collapses once a harmful continuation is underway (either through the adversarial attacks or via harmful assistant-prefill attacks). This raises a fundamental question: Can the innate shallow alignment in LLMs be unlocked to ensure safety at arbitrary generation depths? To achieve this goal, we propose Any-Depth Alignment (ADA), an effective inference-time defense with negligible overhead. ADA is built based on our observation that alignment is concentrated in the assistant header tokens through repeated use in shallow-refusal training, and these tokens possess the model's strong alignment priors. By reintroducing these tokens mid-stream, ADA induces the model to reassess harmfulness and recover refusals at any point in generation. Across diverse open-source model families (Llama, Gemma, Mistral, Qwen, DeepSeek, and gpt-oss), ADA achieves robust safety performance without requiring any changes to the base model's parameters. It secures a near-100% refusal rate against challenging adversarial prefill attacks ranging from dozens to thousands of tokens. Furthermore, ADA reduces the average success rate of prominent adversarial prompt attacks (such as GCG, AutoDAN, PAIR, and TAP) to below 3%. This is all accomplished while preserving utility on benign tasks with minimal over-refusal. ADA maintains this resilience even after the base model undergoes subsequent instruction tuning (benign or adversarial).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00657v3">LLM Safety Alignment is Divergence Estimation in Disguise</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      We present a theoretical framework showing that popular LLM alignment methods, including RLHF and its variants, can be understood as divergence estimators between aligned (safe or preferred) and unaligned (harmful or less preferred) distributions. This perspective explains the emergence of separation in the latent space between safe and harmful prompts after alignment. As an application of our general divergence framework, we propose KLDO, a novel KL divergence-based alignment method, and empirically validate its effectiveness. We further show that using compliance-refusal datasets, rather than standard preference-based datasets, leads to stronger separation and improved safety alignment. Finally, to quantify the separation effect, we propose a distance-based metric in the prompt representation space, which also acts as a statistically significant indicator for model safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.14207v2">Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are powering a growing share of interactive web applications, yet remain vulnerable to misuse and harm. Prior jailbreak research has largely focused on single-turn prompts, whereas real harassment often unfolds over multi-turn interactions. In this work, we present the Online Harassment Agentic Benchmark consisting of: (i) a synthetic multi-turn harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim) simulation informed by repeated game theory, (iii) three jailbreak methods attacking agents across memory, planning, and fine-tuning, and (iv) a mixed-methods evaluation framework. We utilize two prominent LLMs, LLaMA-3.1-8B-Instruct (open-source) and Gemini-2.0-flash (closed-source). Our results show that jailbreak tuning makes harassment nearly guaranteed with an attack success rate of 95.78--96.89% vs. 57.25--64.19% without tuning in Llama, and 99.33% vs. 98.46% without tuning in Gemini, while sharply reducing refusal rate to 1-2% in both models. The most prevalent toxic behaviors are Insult with 84.9--87.8% vs. 44.2--50.8% without tuning, and Flaming with 81.2--85.1% vs. 31.5--38.8% without tuning, indicating weaker guardrails compared to sensitive categories such as sexual or racial harassment. Qualitative evaluation further reveals that attacked agents reproduce human-like aggression profiles, such as Machiavellian/psychopathic patterns under planning, and narcissistic tendencies with memory. Counterintuitively, closed-source and open-source models exhibit distinct escalation trajectories across turns, with closed-source models showing significant vulnerability. Overall, our findings show that multi-turn and theory-grounded attacks not only succeed at high rates but also mimic human-like harassment dynamics, motivating the development of robust safety guardrails to ultimately keep online platforms safe and responsible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18043v1">CompactPrompt: A Unified Pipeline for Prompt Data Compression in LLM Workflows</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Workshop on LLMs and Generative AI for Finance at ACM ICAIF 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deliver powerful reasoning and generation capabilities but incur substantial run-time costs when operating in agentic workflows that chain together lengthy prompts and process rich data streams. We introduce CompactPrompt, an end-to-end pipeline that merges hard prompt compression with lightweight file-level data compression. CompactPrompt first prunes low-information tokens from prompts using self-information scoring and dependency-based phrase grouping. In parallel, it applies n-gram abbreviation to recurrent textual patterns in attached documents and uniform quantization to numerical columns, yielding compact yet semantically faithful representations. Integrated into standard LLM agents, CompactPrompt reduces total token usage and inference cost by up to 60% on benchmark dataset like TAT-QA and FinQA, while preserving output quality (Results in less than 5% accuracy drop for Claude-3.5-Sonnet, and GPT-4.1-Mini) CompactPrompt helps visualize real-time compression decisions and quantify cost-performance trade-offs, laying the groundwork for leaner generative AI pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.15216v2">Soundness-Aware Level: A Microscopic Signature that Predicts LLM Reasoning Potential</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      Reinforcement learning with verifiable rewards (RLVR) can elicit strong reasoning in large language models (LLMs), while their performance after RLVR varies dramatically across different base models. This raises a fundamental question: what microscopic property of pre-trained models leads to this variation? To investigate, we formalize reasoning as chains of Horn clauses ("if-then" rules) built from features extracted from the LLM's latent space via cross-layer sparse autoencoders (SAEs). We estimate the transition probabilities between its features, and further categorize each rule by its semantic soundness level (e.g., strict, plausible, noisy) with an LLM. Our key discovery is that high-potential models are inherently soundness-aware: their internal probability distributions systematically shift across rules' soundness levels, becoming highly distinct for "strict" versus "noisy" rules. In contrast, weaker models are soundness-agnostic, collapsing to one distribution regardless of soundness levels. To quantify this, we introduce the Soundness-Aware Level (SAL), a microscopic metric using the Jensen-Shannon Divergence to measure the separation between these distributions. We show that SAL's predictions of post-RLVR reasoning performance follow a precise empirical law (R^2=0.87) across diverse model families (Qwen, Mistral, Llama, DeepSeek) and scales (0.5B-14B). This reveals that a model's reasoning potential is tied to its intrinsic, pre-trained ability to distinguish sound knowledge from unsound ones. These findings underscore the critical role of model pre-training in shaping reasoning and offer a practical metric grounded in the model's internal mechanisms for selecting/designing stronger base models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18032v1">OPTAGENT: Optimizing Multi-Agent LLM Interactions Through Verbal Reinforcement Learning for Enhanced Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 8 pages for main content
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable reasoning capabilities in mathematical and scientific tasks. To enhance complex reasoning, multi-agent systems have been proposed to harness the collective intelligence of LLM agents. However, existing collaboration structures are either predefined or rely on majority voting or round-table debates, which can suppress correct but less dominant agent contributions. Recent approaches model multi-agent systems as graph networks but optimize purely for agent performance, neglecting the quality of interactions. We hypothesize that effective agent communication is crucial for multi-agent reasoning and that debating quality plays a significant role. To address this, we propose $\ours$, a multi-agent verbal reinforcement learning algorithm that dynamically constructs and refines multi-agent collaboration structures. Our method defines action spaces and a feedback mechanism that evaluates communication robustness and coherence throughout the debate. The final decision is achieved through a majority vote over all the agents. We assess $\ours$ on various reasoning tasks, including mathematical reasoning, creative writing, scientific reasoning, and numerical sorting. Results demonstrate that our approach significantly outperforms single-agent prompting methods and state-of-the-art multi-agent frameworks on diverse tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18019v1">Is Multilingual LLM Watermarking Truly Multilingual? A Simple Back-Translation Solution</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Multilingual watermarking aims to make large language model (LLM) outputs traceable across languages, yet current methods still fall short. Despite claims of cross-lingual robustness, they are evaluated only on high-resource languages. We show that existing multilingual watermarking methods are not truly multilingual: they fail to remain robust under translation attacks in medium- and low-resource languages. We trace this failure to semantic clustering, which fails when the tokenizer vocabulary contains too few full-word tokens for a given language. To address this, we introduce STEAM, a back-translation-based detection method that restores watermark strength lost through translation. STEAM is compatible with any watermarking method, robust across different tokenizers and languages, non-invasive, and easily extendable to new languages. With average gains of +0.19 AUC and +40%p TPR@1% on 17 languages, STEAM provides a simple and robust path toward fairer watermarking across diverse languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.18003v1">BadScientist: Can a Research Agent Write Convincing but Unsound Papers that Fool LLM Reviewers?</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      The convergence of LLM-powered research assistants and AI-based peer review systems creates a critical vulnerability: fully automated publication loops where AI-generated research is evaluated by AI reviewers without human oversight. We investigate this through \textbf{BadScientist}, a framework that evaluates whether fabrication-oriented paper generation agents can deceive multi-model LLM review systems. Our generator employs presentation-manipulation strategies requiring no real experiments. We develop a rigorous evaluation framework with formal error guarantees (concentration bounds and calibration analysis), calibrated on real data. Our results reveal systematic vulnerabilities: fabricated papers achieve acceptance rates up to . Critically, we identify \textit{concern-acceptance conflict} -- reviewers frequently flag integrity issues yet assign acceptance-level scores. Our mitigation strategies show only marginal improvements, with detection accuracy barely exceeding random chance. Despite provably sound aggregation mathematics, integrity checking systematically fails, exposing fundamental limitations in current AI-driven review systems and underscoring the urgent need for defense-in-depth safeguards in scientific publishing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.19398v2">Do LLMs Strategically Reveal, Conceal, and Infer Information? A Theoretical and Empirical Analysis in The Chameleon Game</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Large language model-based (LLM-based) agents have become common in settings that include non-cooperative parties. In such settings, agents' decision-making needs to conceal information from their adversaries, reveal information to their cooperators, and infer information to identify the other agents' characteristics. To investigate whether LLMs have these information control and decision-making capabilities, we make LLM agents play the language-based hidden-identity game, The Chameleon. In this game, a group of non-chameleon agents who do not know each other aim to identify the chameleon agent without revealing a secret. The game requires the aforementioned information control capabilities both as a chameleon and a non-chameleon. We begin with a theoretical analysis for a spectrum of strategies, from concealing to revealing, and provide bounds on the non-chameleons' winning probability. The empirical results with GPT, Gemini 2.5 Pro, Llama 3.1, and Qwen3 models show that while non-chameleon LLM agents identify the chameleon, they fail to conceal the secret from the chameleon, and their winning probability is far from the levels of even trivial strategies. Based on these empirical results and our theoretical analysis, we deduce that LLM-based agents may reveal excessive information to agents of unknown identities. Interestingly, we find that, when instructed to adopt an information-revealing level, this level is linearly encoded in the LLM's internal representations. While the instructions alone are often ineffective at making non-chameleon LLMs conceal, we show that steering the internal representations in this linear direction directly can reliably induce concealing behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17941v1">Believe It or Not: How Deeply do LLMs Believe Implanted Facts?</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Knowledge editing techniques promise to implant new factual knowledge into large language models (LLMs). But do LLMs really believe these facts? We develop a framework to measure belief depth and use it to evaluate the success of knowledge editing techniques. We operationalize belief depth as the extent to which implanted knowledge 1) generalizes to related contexts (e.g. Fermi estimates several logical steps removed), 2) is robust to self-scrutiny and direct challenge, and 3) is represented similarly to genuine knowledge (as measured by linear probes). Our evaluations show that simple prompting and mechanistic editing techniques fail to implant knowledge deeply. In contrast, Synthetic Document Finetuning (SDF) - where models are trained on LLM-generated documents consistent with a fact - often succeeds at implanting beliefs that behave similarly to genuine knowledge. However, SDF's success is not universal, as implanted beliefs that contradict basic world knowledge are brittle and representationally distinct from genuine knowledge. Overall, our work introduces measurable criteria for belief depth and enables the rigorous evaluation necessary for deploying knowledge editing in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17934v1">AtlasKV: Augmenting LLMs with Billion-Scale Knowledge Graphs in 20GB VRAM</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) has shown some success in augmenting large language models (LLMs) with external knowledge. However, as a non-parametric knowledge integration paradigm for LLMs, RAG methods heavily rely on external retrieval modules and the retrieved textual context prior. Especially for very large scale knowledge augmentation, they would introduce substantial inference latency due to expensive searches and much longer relevant context. In this paper, we propose a parametric knowledge integration method, called \textbf{AtlasKV}, a scalable, effective, and general way to augment LLMs with billion-scale knowledge graphs (KGs) (e.g. 1B triples) using very little GPU memory cost (e.g. less than 20GB VRAM). In AtlasKV, we introduce KG2KV and HiKVP to integrate KG triples into LLMs at scale with sub-linear time and memory complexity. It maintains strong knowledge grounding and generalization performance using the LLMs' inherent attention mechanism, and requires no external retrievers, long context priors, or retraining when adapting to new knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17924v1">Efficient Toxicity Detection in Gaming Chats: A Comparative Study of Embeddings, Fine-Tuned Transformers and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 Published in the Journal of Data Mining & Digital Humanities (JDMDH), special issue NLP4DH
    </div>
    <details class="paper-abstract">
      This paper presents a comprehensive comparative analysis of Natural Language Processing (NLP) methods for automated toxicity detection in online gaming chats. Traditional machine learning models with embeddings, large language models (LLMs) with zero-shot and few-shot prompting, fine-tuned transformer models, and retrieval-augmented generation (RAG) approaches are evaluated. The evaluation framework assesses three critical dimensions: classification accuracy, processing speed, and computational costs. A hybrid moderation system architecture is proposed that optimizes human moderator workload through automated detection and incorporates continuous learning mechanisms. The experimental results demonstrate significant performance variations across methods, with fine-tuned DistilBERT achieving optimal accuracy-cost trade-offs. The findings provide empirical evidence for deploying cost-effective, efficient content moderation systems in dynamic online gaming environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17921v1">CLAWS:Creativity detection for LLM-generated solutions using Attention Window of Sections</a></div>
    <div class="paper-meta">
      📅 2025-10-20
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advances in enhancing the reasoning ability of large language models (LLMs) have been remarkably successful. LLMs trained with reinforcement learning (RL) for reasoning demonstrate strong performance in challenging tasks such as mathematics and coding, even with relatively small model sizes. However, despite these improvements in task accuracy, the assessment of creativity in LLM generations has been largely overlooked in reasoning tasks, in contrast to writing tasks. The lack of research on creativity assessment in reasoning primarily stems from two challenges: (1) the difficulty of defining the range of creativity, and (2) the necessity of human evaluation in the assessment process. To address these challenges, we propose CLAWS, a method that defines and classifies mathematical solutions into typical, creative, and hallucinated categories without human evaluation, by leveraging attention weights across prompt sections and output. CLAWS outperforms five existing white-box detection methods (Perplexity, Logit Entropy, Window Entropy, Hidden Score, and Attention Score) on five 7-8B math RL models (DeepSeek, Qwen, Mathstral, OpenMath2, and Oreal). We validate CLAWS on 4545 math problems collected from 181 math contests (AJHSME, AMC, AIME).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17918v1">JT-Safe: Intrinsically Enhancing the Safety and Trustworthiness of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-20
    </div>
    <details class="paper-abstract">
      The hallucination and credibility concerns of large language models (LLMs) are global challenges that the industry is collectively addressing. Recently, a significant amount of advances have been made on post-training and inference techniques to mitigate these challenges. However, it is widely agreed that unsafe and hallucinations of LLMs intrinsically originate from pre-training, involving pre-training data and the next-token prediction learning mechanism. In this paper, we focus on enhancing pre-training data to improve the trustworthiness and safety of LLMs. Since the data is vast, it's almost impossible to entirely purge the data of factual errors, logical inconsistencies, or distributional biases. Moreover, the pre-training data lack grounding in real-world knowledge. Each piece of data is treated as a sequence of tokens rather than as a representation of a part of the world. To overcome these issues, we propose approaches to enhancing our pre-training data with its context in the world and increasing a substantial amount of data reflecting industrial scenarios. We argue that most source data are created by the authors for specific purposes in a certain spatial-temporal context. They have played a role in the real world. By incorporating related world context information, we aim to better anchor pre-training data within real-world scenarios, thereby reducing uncertainty in model training and enhancing the model's safety and trustworthiness. We refer to our Data with World Context as DWC. We continue pre-training an earlier checkpoint of JT-35B-Base with 1.5 trillion of DWC tokens. We introduce our post-training procedures to activate the potentials of DWC. Compared with the Qwen model of a similar scale, JT-Safe-35B achieves an average performance improvement of 1.79% on the Safety and Trustworthy evaluation benchmarks, while being pretrained with only 6.2 trillion tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17023v1">Enrich and Detect: Video Temporal Grounding with Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 ICCV 2025 (Highlights)
    </div>
    <details class="paper-abstract">
      We introduce ED-VTG, a method for fine-grained video temporal grounding utilizing multi-modal large language models. Our approach harnesses the capabilities of multimodal LLMs to jointly process text and video, in order to effectively localize natural language queries in videos through a two-stage process. Rather than being directly grounded, language queries are initially transformed into enriched sentences that incorporate missing details and cues to aid in grounding. In the second stage, these enriched queries are grounded, using a lightweight decoder, which specializes at predicting accurate boundaries conditioned on contextualized representations of the enriched queries. To mitigate noise and reduce the impact of hallucinations, our model is trained with a multiple-instance-learning objective that dynamically selects the optimal version of the query for each training sample. We demonstrate state-of-the-art results across various benchmarks in temporal video grounding and paragraph grounding settings. Experiments reveal that our method significantly outperforms all previously proposed LLM-based temporal grounding approaches and is either superior or comparable to specialized models, while maintaining a clear advantage against them in zero-shot evaluation scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17021v1">Forgetting to Forget: Attention Sink as A Gateway for Backdooring LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language model (LLM) unlearning has become a critical mechanism for removing undesired data, knowledge, or behaviors from pre-trained models while retaining their general utility. Yet, with the rise of open-weight LLMs, we ask: can the unlearning process itself be backdoored, appearing successful under normal conditions yet reverting to pre-unlearned behavior when a hidden trigger is activated? Drawing inspiration from classical backdoor attacks that embed triggers into training data to enforce specific behaviors, we investigate backdoor unlearning, where models forget as intended in the clean setting but recover forgotten knowledge when the trigger appears. We show that designing such attacks presents unique challenges, hinging on where triggers are placed and how backdoor training is reinforced. We uncover a strong link between backdoor efficacy and the attention sink phenomenon, i.e., shallow input tokens consistently attract disproportionate attention in LLMs. Our analysis reveals that these attention sinks serve as gateways for backdoor unlearning: placing triggers at sink positions and aligning their attention values markedly enhances backdoor persistence. Extensive experiments validate these findings, showing that attention-sink-guided backdoor unlearning reliably restores forgotten knowledge in the presence of backdoor triggers, while behaving indistinguishably from a normally unlearned model when triggers are absent. Code is available at https://github.com/OPTML-Group/Unlearn-Backdoor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17017v1">SafeSearch: Do Not Trade Safety for Utility in LLM Search Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Code: https://github.com/ZQS1943/SafeSearch
    </div>
    <details class="paper-abstract">
      Large language model (LLM) based search agents iteratively generate queries, retrieve external information, and reason to answer open-domain questions. While researchers have primarily focused on improving their utility, their safety behaviors remain underexplored. In this paper, we first evaluate search agents using red-teaming datasets and find that they are more likely to produce harmful outputs than base LLMs. For example, when asked ``How can I track someone's location without their consent?'', a base model refuses, whereas a search agent designed to retrieve and cite sources may lower its refusal threshold, fetch documents (e.g., court cases), and, once appended, synthesize them into an informative yet unsafe summary. We further show that utility-oriented fine-tuning intensifies this risk, motivating joint alignment of safety and utility. We present SafeSearch, a multi-objective reinforcement learning approach that couples a final-output safety/utility reward with a novel query-level shaping term that penalizes unsafe queries and rewards safe ones. Experiments show that SafeSearch reduces agent harmfulness by over 70% across three red-teaming datasets while producing safe, helpful responses, and matches the QA performance of a utility-only finetuned agent; further analyses confirm the effectiveness of the query-level reward in jointly improving safety and utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17015v1">Justitia: Fair and Efficient Scheduling for LLM Applications</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      In the era of Large Language Models (LLMs), it has been popular to launch a series of LLM inferences -- we call an LLM application -- to better solve real-world problems. When serving those applications in shared GPU servers, the schedulers are expected to attain fast application completions with guaranteed worst-case performance. However, mainstream LLM schedulers fail to behave well for LLM applications -- due to head-of-line blocking or over-constrained resource allocation. In this paper, we propose to serve LLM applications in a fair and also efficient manner. To this end, we design Justitia, a novel scheduler with three key techniques. First, given that memory is prevalently a bottleneck for mainstream inference frameworks like vLLM, Justitia models the service cost of LLM applications in a memory-centric manner. Meanwhile, it uses a simple neural network model to conduct light-weight and also accurate demand prediction. Moreover, Justitia adopts a virtual-time based fair queuing algorithm to reduce the overall performance with guaranteed worst-case delay. We have implemented Justitia atop vLLM, and experimental results involving diverse LLM applications show that it can substantially enhance the scheduling efficiency with fairness preserved.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17013v1">DiscoTrack: A Multilingual LLM Benchmark for Discourse Tracking</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Recent LLM benchmarks have tested models on a range of phenomena, but are still focused primarily on natural language understanding for extraction of explicit information, such as QA or summarization, with responses often tar- geting information from individual sentences. We are still lacking more challenging, and im- portantly also multilingual, benchmarks focus- ing on implicit information and pragmatic infer- ences across larger documents in the context of discourse tracking: integrating and aggregating information across sentences, paragraphs and multiple speaker utterances. To this end, we present DiscoTrack, an LLM benchmark target- ing a range of tasks across 12 languages and four levels of discourse understanding: salience recognition, entity tracking, discourse relations and bridging inference. Our evaluation shows that these tasks remain challenging, even for state-of-the-art models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00222v4">RL-PLUS: Countering Capability Boundary Collapse of LLMs in Reinforcement Learning with Hybrid-policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Reinforcement Learning with Verifiable Reward (RLVR) has significantly advanced the complex reasoning abilities of Large Language Models (LLMs). However, it struggles to break through the inherent capability boundaries of the base LLM, due to its essentially on-policy strategy coupled with LLM's immense action space and sparse reward. Critically, RLVR can lead to the capability boundary collapse, narrowing the LLM's problem-solving scope. To address this problem, we propose RL-PLUS, a novel hybrid-policy optimization approach for LLMs that synergizes internal exploitation with external data to achieve stronger reasoning capabilities and surpass the boundaries of base models. RL-PLUS integrates two core components, i.e., Multiple Importance Sampling to address distributional mismatch from external data, and Exploration-Based Advantage Function to guide the model towards high-value, unexplored reasoning paths. We provide both theoretical analysis and extensive experiments to demonstrate the superiority and generalizability of our approach. Compared with existing RLVR methods, RL-PLUS achieves 1) state-of-the-art performance on six math reasoning benchmarks; 2) superior performance on six out-of-distribution reasoning tasks; 3) consistent and significant gains across diverse model families, with average relative improvements up to 69.2\%. Moreover, the analysis of Pass@k curves indicates that RL-PLUS effectively resolves the capability boundary collapse problem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03304v3">Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel across various tasks, but standard first-order (FO) fine-tuning demands considerable memory, significantly limiting real-world deployment. Recently, zeroth-order (ZO) optimization stood out as a promising memory-efficient training paradigm, avoiding backward passes and relying solely on forward passes for gradient estimation, making it attractive for resource-constrained scenarios. However, ZO method lags far behind FO method in both convergence speed and accuracy. To bridge the gap, we introduce a novel layer-wise divergence analysis that uncovers the distinct update pattern of FO and ZO optimization. Aiming to resemble the learning capacity of FO method from the findings, we propose Divergence-driven Zeroth-Order (DiZO) optimization. DiZO conducts divergence-driven layer adaptation by incorporating projections to ZO updates, generating diverse-magnitude updates precisely scaled to layer-wise individual optimization needs. Our results demonstrate that DiZO significantly reduces the needed iterations for convergence without sacrificing throughput, cutting training GPU hours by up to 48\% on various datasets. Moreover, DiZO consistently outperforms the representative ZO baselines in fine-tuning RoBERTa-large, OPT-series, and Llama-series on downstream tasks and, in some cases, even surpasses memory-intensive FO fine-tuning. Our code is released at https://github.com/Skilteee/DiZO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17002v1">EEschematic: Multimodal-LLM Based AI Agent for Schematic Generation of Analog Circuit</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Circuit schematics play a crucial role in analog integrated circuit design, serving as the primary medium for human understanding and verification of circuit functionality. While recent large language model (LLM)-based approaches have shown promise in circuit topology generation and device sizing, most rely solely on textual representations such as SPICE netlists, which lack visual interpretability for circuit designers. To address this limitation, we propose EEschematic, an AI agent for automatic analog schematic generation based on a Multimodal Large Language Model (MLLM). EEschematic integrates textual, visual, and symbolic modalities to translate SPICE netlists into schematic diagrams represented in a human-editable format. The framework uses six analog substructure examples for few-shot placement and a Visual Chain-of-Thought (VCoT) strategy to iteratively refine placement and wiring, enhancing schematic clarity and symmetry. Experimental results on representative analog circuits, including a CMOS inverter, a five-transistor operational transconductance amplifier (5T-OTA), and a telescopic cascode amplifier, demonstrate that EEschematic produces schematics with high visual quality and structural correctness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17001v1">Vocab Diet: Reshaping the Vocabulary of LLMs with Vector Arithmetic</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) were shown to encode word form variations, such as "walk"->"walked", as linear directions in embedding space. However, standard tokenization algorithms treat these variations as distinct tokens -- filling the size-capped vocabulary with surface form variants (e.g., "walk", "walking", "Walk"), at the expense of less frequent words and multilingual coverage. We show that many of these variations can be captured by transformation vectors -- additive offsets that yield the appropriate word's representation when applied to the base form word embedding -- in both the input and output spaces. Building on this, we propose a compact reshaping of the vocabulary: rather than assigning unique tokens to each surface form, we compose them from shared base form and transformation vectors (e.g., "walked" = "walk" + past tense). We apply our approach to multiple LLMs and across five languages, removing up to 10% of vocabulary entries -- thereby freeing space to allocate new, more diverse tokens. Importantly, we do so while also expanding vocabulary coverage to out-of-vocabulary words, with minimal impact on downstream performance, and without modifying model weights. Our findings motivate a foundational rethinking of vocabulary design, moving from string enumeration to a compositional vocabulary that leverages the underlying structure of language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17000v1">Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks against LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 NeurIPS 2025 (spotlight)
    </div>
    <details class="paper-abstract">
      Adversarial attacks by malicious users that threaten the safety of large language models (LLMs) can be viewed as attempts to infer a target property $T$ that is unknown when an instruction is issued, and becomes knowable only after the model's reply is observed. Examples of target properties $T$ include the binary flag that triggers an LLM's harmful response or rejection, and the degree to which information deleted by unlearning can be restored, both elicited via adversarial instructions. The LLM reveals an \emph{observable signal} $Z$ that potentially leaks hints for attacking through a response containing answer tokens, thinking process tokens, or logits. Yet the scale of information leaked remains anecdotal, leaving auditors without principled guidance and defenders blind to the transparency--risk trade-off. We fill this gap with an information-theoretic framework that computes how much information can be safely disclosed, and enables auditors to gauge how close their methods come to the fundamental limit. Treating the mutual information $I(Z;T)$ between the observation $Z$ and the target property $T$ as the leaked bits per query, we show that achieving error $\varepsilon$ requires at least $\log(1/\varepsilon)/I(Z;T)$ queries, scaling linearly with the inverse leak rate and only logarithmically with the desired accuracy. Thus, even a modest increase in disclosure collapses the attack cost from quadratic to logarithmic in terms of the desired accuracy. Experiments on seven LLMs across system-prompt leakage, jailbreak, and relearning attacks corroborate the theory: exposing answer tokens alone requires about a thousand queries; adding logits cuts this to about a hundred; and revealing the full thinking process trims it to a few dozen. Our results provide the first principled yardstick for balancing transparency and security when deploying LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19504v2">DOGe: Defensive Output Generation for LLM Protection Against Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Code is available at https://github.com/unites-lab/doge
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) represent substantial intellectual and economic investments, yet their effectiveness can inadvertently facilitate model imitation via knowledge distillation (KD). In practical scenarios, competitors can distill proprietary LLM capabilities by simply observing publicly accessible outputs, akin to reverse-engineering a complex performance by observation alone. Existing protective methods like watermarking only identify imitation post-hoc, while other defenses assume the student model mimics the teacher's internal logits, rendering them ineffective against distillation purely from observed output text. This paper confronts the challenge of actively protecting LLMs within the realistic constraints of API-based access. We introduce an effective and efficient Defensive Output Generation (DOGe) strategy that subtly modifies the output behavior of an LLM. Its outputs are accurate and useful for legitimate users, yet are designed to be misleading for distillation, significantly undermining imitation attempts. We achieve this by fine-tuning only the final linear layer of the teacher LLM with an adversarial loss. This targeted training approach anticipates and disrupts distillation attempts during inference time. Our experiments show that, while preserving the performance of the teacher model, student models distilled from the defensively generated outputs demonstrate catastrophically reduced performance, demonstrating DOGe as a practical safeguard against KD-based model imitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16978v1">Lark: Biologically Inspired Neuroevolution for Multi-Stakeholder LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: NeurIPS 2025 Workshop on Efficient Reasoning
    </div>
    <details class="paper-abstract">
      We present Lark, a biologically inspired decision-making framework that couples LLM-driven reasoning with an evolutionary, stakeholder-aware Multi-Agent System (MAS). To address verbosity and stakeholder trade-offs, we integrate four mechanisms: (i) plasticity, which applies concise adjustments to candidate solutions; (ii) duplication and maturation, which copy high-performing candidates and specialize them into new modules; (iii) ranked-choice stakeholder aggregation using influence-weighted Borda scoring; and (iv) compute awareness via token-based penalties that reward brevity. The system iteratively proposes diverse strategies, applies plasticity tweaks, simulates stakeholder evaluations, aggregates preferences, selects top candidates, and performs duplication/maturation while factoring compute cost into final scores. In a controlled evaluation over 30 rounds comparing 14 systems, Lark Full achieves a mean rank of 2.55 (95% CI [2.17, 2.93]) and a mean composite score of 29.4/50 (95% CI [26.34, 32.46]), finishing Top-3 in 80% of rounds while remaining cost competitive with leading commercial models ($0.016 per task). Paired Wilcoxon tests confirm that all four mechanisms contribute significantly as ablating duplication/maturation yields the largest deficit ({\Delta}Score = 3.5, Cohen's d_z = 2.53, p < 0.001), followed by plasticity ({\Delta}Score = 3.4, d_z = 1.86), ranked-choice voting ({\Delta}Score = 2.4, d_z = 1.20), and token penalties ({\Delta}Score = 2.2, d_z = 1.63). Rather than a formal Markov Decision Process with constrained optimization, Lark is a practical, compute-aware neuroevolutionary loop that scales stakeholder-aligned strategy generation and makes trade-offs transparent through per-step metrics. Our work presents proof-of-concept findings and invites community feedback as we expand toward real-world validation studies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20548v2">$Q\sharp$: Provably Optimal Distributional RL for LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) post-training is crucial for LLM alignment and reasoning, but existing policy-based methods, such as PPO and DPO, can fall short of fixing shortcuts inherited from pre-training. In this work, we introduce $Q\sharp$, a value-based algorithm for KL-regularized RL that guides the reference policy using the optimal regularized $Q$ function. We propose to learn the optimal $Q$ function using distributional RL on an aggregated online dataset. Unlike prior value-based baselines that guide the model using unregularized $Q$-values, our method is theoretically principled and provably learns the optimal policy for the KL-regularized RL problem. Empirically, $Q\sharp$ outperforms prior baselines in math reasoning benchmarks while maintaining a smaller KL divergence to the reference policy. Theoretically, we establish a reduction from KL-regularized RL to no-regret online learning, providing the first bounds for deterministic MDPs under only realizability. Thanks to distributional RL, our bounds are also variance-dependent and converge faster when the reference policy has small variance. In sum, our results highlight $Q\sharp$ as an effective approach for post-training LLMs, offering both improved performance and theoretical guarantees. The code can be found at https://github.com/jinpz/q_sharp.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16943v1">Peering Inside the Black Box: Uncovering LLM Errors in Optimization Modelling through Component-Level Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to convert natural language descriptions into mathematical optimization formulations. Current evaluations often treat formulations as a whole, relying on coarse metrics like solution accuracy or runtime, which obscure structural or numerical errors. In this study, we present a comprehensive, component-level evaluation framework for LLM-generated formulations. Beyond the conventional optimality gap, our framework introduces metrics such as precision and recall for decision variables and constraints, constraint and objective root mean squared error (RMSE), and efficiency indicators based on token usage and latency. We evaluate GPT-5, LLaMA 3.1 Instruct, and DeepSeek Math across optimization problems of varying complexity under six prompting strategies. Results show that GPT-5 consistently outperforms other models, with chain-of-thought, self-consistency, and modular prompting proving most effective. Analysis indicates that solver performance depends primarily on high constraint recall and low constraint RMSE, which together ensure structural correctness and solution reliability. Constraint precision and decision variable metrics play secondary roles, while concise outputs enhance computational efficiency. These findings highlight three principles for NLP-to-optimization modeling: (i) Complete constraint coverage prevents violations, (ii) minimizing constraint RMSE ensures solver-level accuracy, and (iii) concise outputs improve computational efficiency. The proposed framework establishes a foundation for fine-grained, diagnostic evaluation of LLMs in optimization modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16933v1">Tutoring LLM into a Better CUDA Optimizer</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this contribution is published in Euro-Par 2025: Parallel Processing, Part II, and is available online at https://doi.org/10.1007/978-3-031-99857-7_18
    </div>
    <details class="paper-abstract">
      Recent leaps in large language models (LLMs) caused a revolution in programming tools (like GitHub Copilot) that can help with code generation, debugging, and even performance optimization. In this paper, we focus on the capabilities of the most recent reasoning models to generate optimized CUDA code for predefined, well-known tasks. Our objective is to determine which types of code optimizations and parallel patterns the LLMs can perform by themselves and whether they can be improved by tutoring (providing more detailed hints and guidelines in the prompt). The generated solutions were evaluated both automatically (for correctness and speedup) and manually (code reviews) to provide a more detailed perspective. We also tried an interactive approach where the LLM can fix its previous mistakes within a session. The results indicate that LLMs are quite skilled coders; however, they require tutoring to reach optimized solutions provided by parallel computing experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16932v1">Prompt-MII: Meta-Learning Instruction Induction for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      A popular method to adapt large language models (LLMs) to new tasks is in-context learning (ICL), which is effective but incurs high inference costs as context length grows. In this paper we propose a method to perform instruction induction, where we take training examples and reduce them to a compact but descriptive prompt that can achieve performance comparable to ICL over the full training set. Specifically, we propose PROMPT-MII, a reinforcement learning (RL) based framework to meta-learn an instruction induction model that can generate compact instructions on the fly for an arbitrary new dataset. We train on over 3,000 diverse classification datasets from the HuggingFace hub, and evaluate on 90 unseen tasks. PROMPT-MII improves downstream model quality by 4-9 F1 points (10-20% relative), matching ICL performance while requiring 3-13x fewer tokens.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16916v1">SolverLLM: Leveraging Test-Time Scaling for Optimization Problem via LLM-Guided Search</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer promising capabilities for tackling complex reasoning tasks, including optimization problems. However, existing methods either rely on prompt engineering, which leads to poor generalization across problem types, or require costly supervised training. We introduce SolverLLM, a training-free framework that leverages test-time scaling to solve diverse optimization problems. Rather than solving directly, SolverLLM generates mathematical formulations and translates them into solver-ready code, guided by a novel Monte Carlo Tree Search (MCTS) strategy. To enhance the search process, we modify classical MCTS with (1) dynamic expansion for adaptive formulation generation, (2) prompt backpropagation to guide exploration via outcome-driven feedback, and (3) uncertainty backpropagation to incorporate reward reliability into decision-making. Experiments on six standard benchmark datasets demonstrate that SolverLLM outperforms both prompt-based and learning-based baselines, achieving strong generalization without additional training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.04774v2">Online automatic code generation for robot swarms: LLMs and self-organizing hierarchy</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 This abstract was accepted to and presented at the "Multi-Agent Cooperative Systems and Swarm Robotics in the Era of Generative AI" (MACRAI) workshop at the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
    </div>
    <details class="paper-abstract">
      Our recently introduced self-organizing nervous system (SoNS) provides robot swarms with 1) ease of behavior design and 2) global estimation of the swarm configuration and its collective environment, facilitating the implementation of online automatic code generation for robot swarms. In a demonstration with 6 real robots and simulation trials with >30 robots, we show that when a SoNS-enhanced robot swarm gets stuck, it can automatically solicit and run code generated by an external LLM on the fly, completing its mission with an 85% success rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16882v1">Utility-Diversity Aware Online Batch Selection for LLM Supervised Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is a commonly used technique to adapt large language models (LLMs) to downstream tasks. In practice, SFT on a full dataset is computationally expensive and sometimes suffers from overfitting or bias amplification. This facilitates the rise of data curation in SFT, which prioritizes the most valuable data to optimze. This work studies the online batch selection family that dynamically scores and filters samples during the training process. However, existing popular methods often (i) rely merely on the utility of data to select a subset while neglecting other crucial factors like diversity, (ii) rely on external resources such as reference models or validation sets, and (iii) incur extra training time over full-dataset training. To address these limitations, this work develops \textbf{UDS (Utility-Diversity Sampling)}, a framework for efficient online batch selection in SFT. UDS leverages the nuclear norm of the logits matrix to capture both data utility and intra-sample diversity, while estimating inter-sample diversity through efficient low-dimensional embedding comparisons with a lightweight memory buffer of historical samples. Such a design eliminates the need for external resources and unnecessary backpropagation, securing computational efficiency. Experiments on multiple benchmarks demonstrate that UDS consistently outperforms state-of-the-art online batch selection methods under varying data budgets, and significantly reduces training time compared to full-dataset fine-tuning. Code is available at https://github.com/gfyddha/UDS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.10407v2">PrediQL: Automated Testing of GraphQL APIs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 8 pages, two columns
    </div>
    <details class="paper-abstract">
      GraphQL's flexible query model and nested data dependencies expose APIs to complex, context-dependent vulnerabilities that are difficult to uncover using conventional testing tools. Existing fuzzers either rely on random payload generation or rigid mutation heuristics, failing to adapt to the dynamic structures of GraphQL schemas and responses. We present PrediQL, the first retrieval-augmented, LLM-guided fuzzer for GraphQL APIs. PrediQL combines large language model reasoning with adaptive feedback loops to generate semantically valid and diverse queries. It models the choice of fuzzing strategy as a multi-armed bandit problem, balancing exploration of new query structures with exploitation of past successes. To enhance efficiency, PrediQL retrieves and reuses execution traces, schema fragments, and prior errors, enabling self-correction and progressive learning across test iterations. Beyond input generation, PrediQL integrates a context-aware vulnerability detector that uses LLM reasoning to analyze responses, interpreting data values, error messages, and status codes to identify issues such as injection flaws, access-control bypasses, and information disclosure. Our evaluation across open-source and benchmark GraphQL APIs shows that PrediQL achieves significantly higher coverage and vulnerability discovery rates compared to state-of-the-art baselines. These results demonstrate that combining retrieval-augmented reasoning with adaptive fuzzing can transform API security testing from reactive enumeration to intelligent exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16819v1">Cross-Genre Authorship Attribution via LLM-Based Retrieve-and-Rerank</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Authorship attribution (AA) is the task of identifying the most likely author of a query document from a predefined set of candidate authors. We introduce a two-stage retrieve-and-rerank framework that finetunes LLMs for cross-genre AA. Unlike the field of information retrieval (IR), where retrieve-and-rerank is a de facto strategy, cross-genre AA systems must avoid relying on topical cues and instead learn to identify author-specific linguistic patterns that are independent of the text's subject matter (genre/domain/topic). Consequently, for the reranker, we demonstrate that training strategies commonly used in IR are fundamentally misaligned with cross-genre AA, leading to suboptimal behavior. To address this, we introduce a targeted data curation strategy that enables the reranker to effectively learn author-discriminative signals. Using our LLM-based retrieve-and-rerank pipeline, we achieve substantial gains of 22.3 and 34.4 absolute Success@8 points over the previous state-of-the-art on HIATUS's challenging HRS1 and HRS2 cross-genre AA benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16809v1">When Many-Shot Prompting Fails: An Empirical Study of LLM Code Translation</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) with vast context windows offer new avenues for in-context learning (ICL), where providing many examples ("many-shot" prompting) is often assumed to enhance performance. We investigate this assumption for the complex task of code translation. Through a large-scale empirical study of over 90,000 translations, we systematically evaluate the impact of scaling in-context examples from zero-shot to many-shot configurations of up to 625 examples, with prompts spanning from approximately 100,000 to 800,000 tokens. Our findings reveal a "many-shot paradox": while static similarity metrics may modestly improve with more examples, functional correctness consistently peaks with few-shot prompting (5-25 examples). Providing substantially more examples often degrades this crucial functional performance. This study highlights that for code translation, the quality of a few well-chosen examples outweighs sheer quantity, challenging the universal efficacy of "more is better" for ICL and underscoring the task-dependent nature of optimal prompting strategies. Our results have significant implications for effectively leveraging LLMs in software engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14345v2">LLM-Enhanced Black-Litterman Portfolio Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Presented at the CIKM 2025 Workshop on Financial AI (https://advancesinfinancialai.com/)
    </div>
    <details class="paper-abstract">
      The Black-Litterman model addresses the sensitivity issues of tra- ditional mean-variance optimization by incorporating investor views, but systematically generating these views remains a key challenge. This study proposes and validates a systematic frame- work that translates return forecasts and predictive uncertainty from Large Language Models (LLMs) into the core inputs for the Black-Litterman model: investor views and their confidence lev- els. Through a backtest on S&P 500 constituents, we demonstrate that portfolios driven by top-performing LLMs significantly out- perform traditional baselines in both absolute and risk-adjusted terms. Crucially, our analysis reveals that each LLM exhibits a dis- tinct and consistent investment style which is the primary driver of performance. We found that the selection of an LLM is therefore not a search for a single best forecaster, but a strategic choice of an investment style whose success is contingent on its alignment with the prevailing market regime. The source code and data are available at https://github.com/youngandbin/LLM-BLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16794v1">Black-box Optimization of LLM Outputs by Asking for Directions</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      We present a novel approach for attacking black-box large language models (LLMs) by exploiting their ability to express confidence in natural language. Existing black-box attacks require either access to continuous model outputs like logits or confidence scores (which are rarely available in practice), or rely on proxy signals from other models. Instead, we demonstrate how to prompt LLMs to express their internal confidence in a way that is sufficiently calibrated to enable effective adversarial optimization. We apply our general method to three attack scenarios: adversarial examples for vision-LLMs, jailbreaks and prompt injections. Our attacks successfully generate malicious inputs against systems that only expose textual outputs, thereby dramatically expanding the attack surface for deployed LLMs. We further find that better and larger models exhibit superior calibration when expressing confidence, creating a concerning security paradox where model capability improvements directly enhance vulnerability. Our code is available at this [link](https://github.com/zj-jayzhang/black_box_llm_optimization).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15594v6">A Survey on LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Project Page: https://awesome-llm-as-a-judge.github.io/
    </div>
    <details class="paper-abstract">
      Accurate and consistent evaluation is crucial for decision-making across numerous fields, yet it remains a challenging task due to inherent subjectivity, variability, and scale. Large Language Models (LLMs) have achieved remarkable success across diverse domains, leading to the emergence of "LLM-as-a-Judge," where LLMs are employed as evaluators for complex tasks. With their ability to process diverse data types and provide scalable, cost-effective, and consistent assessments, LLMs present a compelling alternative to traditional expert-driven evaluations. However, ensuring the reliability of LLM-as-a-Judge systems remains a significant challenge that requires careful design and standardization. This paper provides a comprehensive survey of LLM-as-a-Judge, addressing the core question: How can reliable LLM-as-a-Judge systems be built? We explore strategies to enhance reliability, including improving consistency, mitigating biases, and adapting to diverse assessment scenarios. Additionally, we propose methodologies for evaluating the reliability of LLM-as-a-Judge systems, supported by a novel benchmark designed for this purpose. To advance the development and real-world deployment of LLM-as-a-Judge systems, we also discussed practical applications, challenges, and future directions. This survey serves as a foundational reference for researchers and practitioners in this rapidly evolving field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.01005v2">When To Solve, When To Verify: Compute-Optimal Problem Solving and Generative Verification for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 COLM 2025
    </div>
    <details class="paper-abstract">
      Scaling test-time compute has emerged as a key strategy for enhancing the reasoning capabilities of large language models (LLMs), particularly in tasks like mathematical problem-solving. A traditional approach, Self-Consistency (SC), generates multiple solutions to a problem and selects the most common answer via majority voting. Another common method involves scoring each solution with a reward model (verifier) and choosing the best one. Recent advancements in Generative Reward Models (GenRM) reframe verification as a next-token prediction task, enabling inference-time scaling along a new axis. Specifically, GenRM generates multiple verification chains-of-thought to score each solution. Under a limited inference budget, this introduces a fundamental trade-off: should you spend the budget on scaling solutions via SC or generate fewer solutions and allocate compute to verification via GenRM? To address this, we evaluate GenRM against SC under a fixed inference budget. Interestingly, we find that SC is more compute-efficient than GenRM for most practical inference budgets across diverse models and datasets. For instance, GenRM first matches SC after consuming up to 8x the inference compute and requires significantly more compute to outperform it. Furthermore, we derive inference scaling laws for the GenRM paradigm, revealing that compute-optimal inference favors scaling solution generation more aggressively than scaling the number of verifications. Our work provides practical guidance on optimizing test-time scaling by balancing solution generation and verification. The code is available at https://github.com/nishadsinghi/sc-genrm-scaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.10213v2">An Empirical Study on LLM-based Agents for Automated Bug Fixing</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and LLM-based Agents have been applied to fix bugs automatically, demonstrating the capability in addressing software defects by engaging in development environment interaction, iterative validation and code modification. However, systematic analysis of these agent systems remain limited, particularly regarding performance variations among top-performing ones. In this paper, we examine six repair systems on the SWE-bench Verified benchmark for automated bug fixing. We first assess each system's overall performance, noting the instances solvable by all or none of these systems, and explore the capabilities of different systems. We also compare fault localization accuracy at file and code symbol levels and evaluate bug reproduction capabilities. Through analysis, we concluded that further optimization is needed in both the LLM capability itself and the design of Agentic flow to improve the effectiveness of the Agent in bug fixing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.18439v2">A Systematic Approach to Predict the Impact of Cybersecurity Vulnerabilities Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Accepted for publication in the 24th IEEE International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom 2025)
    </div>
    <details class="paper-abstract">
      Vulnerability databases, such as the National Vulnerability Database (NVD), offer detailed descriptions of Common Vulnerabilities and Exposures (CVEs), but often lack information on their real-world impact, such as the tactics, techniques, and procedures (TTPs) that adversaries may use to exploit the vulnerability. However, manually linking CVEs to their corresponding TTPs is a challenging and time-consuming task, and the high volume of new vulnerabilities published annually makes automated support desirable. This paper introduces TRIAGE, a two-pronged automated approach that uses Large Language Models (LLMs) to map CVEs to relevant techniques from the ATT&CK knowledge base. We first prompt an LLM with instructions based on MITRE's CVE Mapping Methodology to predict an initial list of techniques. This list is then combined with the results from a second LLM-based module that uses in-context learning to map a CVE to relevant techniques. This hybrid approach strategically combines rule-based reasoning with data-driven inference. Our evaluation reveals that in-context learning outperforms the individual mapping methods, and the hybrid approach improves recall of exploitation techniques. We also find that GPT-4o-mini performs better than Llama3.3-70B on this task. Overall, our results show that LLMs can be used to automatically predict the impact of cybersecurity vulnerabilities and TRIAGE makes the process of mapping CVEs to ATT&CK more efficient. A replication package is available for download from https://doi.org/10.5281/zenodo.17341503. Keywords: vulnerability impact, CVE, ATT&CK techniques, large language models, automated mapping.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16767v1">T3 Planner: A Self-Correcting LLM Framework for Robotic Motion Planning with Temporal Logic</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Translating natural language instructions into executable motion plans is a fundamental challenge in robotics. Traditional approaches are typically constrained by their reliance on domain-specific expertise to customize planners, and often struggle with spatio-temporal couplings that usually lead to infeasible motions or discrepancies between task planning and motion execution. Despite the proficiency of Large Language Models (LLMs) in high-level semantic reasoning, hallucination could result in infeasible motion plans. In this paper, we introduce the T3 Planner, an LLM-enabled robotic motion planning framework that self-corrects it output with formal methods. The framework decomposes spatio-temporal task constraints via three cascaded modules, each of which stimulates an LLM to generate candidate trajectory sequences and examines their feasibility via a Signal Temporal Logic (STL) verifier until one that satisfies complex spatial, temporal, and logical constraints is found.Experiments across different scenarios show that T3 Planner significantly outperforms the baselines. The required reasoning can be distilled into a lightweight Qwen3-4B model that enables efficient deployment. All supplementary materials are accessible at https://github.com/leeejia/T3_Planner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11924v2">Intrinsic Self-Correction in LLMs: Towards Explainable Prompting via Mechanistic Interpretability</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Intrinsic self-correction refers to the phenomenon where a language model refines its own outputs purely through prompting, without external feedback or parameter updates. While this approach improves performance across diverse tasks, its internal mechanism remains poorly understood. We analyze intrinsic self-correction from a representation-level perspective. We formalize and introduce the notion of a prompt-induced shift, which is the change in hidden representations caused by a self-correction prompt. Across 5 open-source LLMs, prompt-induced shifts in text detoxification and text toxification align with latent directions constructed from contrastive pairs. In detoxification, the shifts align with the non-toxic direction; in toxification, they align with the toxic direction. These results suggest that intrinsic self-correction functions as representation steering along interpretable latent directions, beyond what standard metrics such as task scores or model confidence capture. Our analysis offers an interpretability-based account of intrinsic self-correction and contributes to a more systematic understanding of LLM prompting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00641v2">AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 This paper is accepted by 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
    </div>
    <details class="paper-abstract">
      Despite the rapid advancement of LLM-based agents, the reliable evaluation of their safety and security remains a significant challenge. Existing rule-based or LLM-based evaluators often miss dangers in agents' step-by-step actions, overlook subtle meanings, fail to see how small issues compound, and get confused by unclear safety or security rules. To overcome this evaluation crisis, we introduce AgentAuditor, a universal, training-free, memory-augmented reasoning framework that empowers LLM evaluators to emulate human expert evaluators. AgentAuditor constructs an experiential memory by having an LLM adaptively extract structured semantic features (e.g., scenario, risk, behavior) and generate associated chain-of-thought reasoning traces for past interactions. A multi-stage, context-aware retrieval-augmented generation process then dynamically retrieves the most relevant reasoning experiences to guide the LLM evaluator's assessment of new cases. Moreover, we developed ASSEBench, the first benchmark designed to check how well LLM-based evaluators can spot both safety risks and security threats. ASSEBench comprises 2293 meticulously annotated interaction records, covering 15 risk types across 29 application scenarios. A key feature of ASSEBench is its nuanced approach to ambiguous risk situations, employing "Strict" and "Lenient" judgment standards. Experiments demonstrate that AgentAuditor not only consistently improves the evaluation performance of LLMs across all benchmarks but also sets a new state-of-the-art in LLM-as-a-judge for agent safety and security, achieving human-level accuracy. Our work is openly accessible at https://github.com/Astarojth/AgentAuditor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16716v1">DistilLock: Safeguarding LLMs from Unauthorized Knowledge Distillation on the Edge</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong performance across diverse tasks, but fine-tuning them typically relies on cloud-based, centralized infrastructures. This requires data owners to upload potentially sensitive data to external servers, raising serious privacy concerns. An alternative approach is to fine-tune LLMs directly on edge devices using local data; however, this introduces a new challenge: the model owner must transfer proprietary models to the edge, which risks intellectual property (IP) leakage. To address this dilemma, we propose DistilLock, a TEE-assisted fine-tuning framework that enables privacy-preserving knowledge distillation on the edge. In DistilLock, a proprietary foundation model is executed within a trusted execution environment (TEE) enclave on the data owner's device, acting as a secure black-box teacher. This setup preserves both data privacy and model IP by preventing direct access to model internals. Furthermore, DistilLock employs a model obfuscation mechanism to offload obfuscated weights to untrusted accelerators for efficient knowledge distillation without compromising security. We demonstrate that DistilLock prevents unauthorized knowledge distillation processes and model-stealing attacks while maintaining high computational efficiency, but offering a secure and practical solution for edge-based LLM personalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16713v1">so much depends / upon / a whitespace: Why Whitespace Matters for Poets and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Whitespace is a critical component of poetic form, reflecting both adherence to standardized forms and rebellion against those forms. Each poem's whitespace distribution reflects the artistic choices of the poet and is an integral semantic and spatial feature of the poem. Yet, despite the popularity of poetry as both a long-standing art form and as a generation task for large language models (LLMs), whitespace has not received sufficient attention from the NLP community. Using a corpus of 19k English-language published poems from Poetry Foundation, we investigate how 4k poets have used whitespace in their works. We release a subset of 2.8k public-domain poems with preserved formatting to facilitate further research in this area. We compare whitespace usage in the published poems to (1) 51k LLM-generated poems, and (2) 12k unpublished poems posted in an online community. We also explore whitespace usage across time periods, poetic forms, and data sources. Additionally, we find that different text processing methods can result in significantly different representations of whitespace in poetry data, motivating us to use these poems and whitespace patterns to discuss implications for the processing strategies used to assemble pretraining datasets for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16712v1">The Chameleon Nature of LLMs: Quantifying Multi-Turn Stance Instability in Search-Enabled Language Models</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Integration of Large Language Models with search/retrieval engines has become ubiquitous, yet these systems harbor a critical vulnerability that undermines their reliability. We present the first systematic investigation of "chameleon behavior" in LLMs: their alarming tendency to shift stances when presented with contradictory questions in multi-turn conversations (especially in search-enabled LLMs). Through our novel Chameleon Benchmark Dataset, comprising 17,770 carefully crafted question-answer pairs across 1,180 multi-turn conversations spanning 12 controversial domains, we expose fundamental flaws in state-of-the-art systems. We introduce two theoretically grounded metrics: the Chameleon Score (0-1) that quantifies stance instability, and Source Re-use Rate (0-1) that measures knowledge diversity. Our rigorous evaluation of Llama-4-Maverick, GPT-4o-mini, and Gemini-2.5-Flash reveals consistent failures: all models exhibit severe chameleon behavior (scores 0.391-0.511), with GPT-4o-mini showing the worst performance. Crucially, small across-temperature variance (less than 0.004) suggests the effect is not a sampling artifact. Our analysis uncovers the mechanism: strong correlations between source re-use rate and confidence (r=0.627) and stance changes (r=0.429) are statistically significant (p less than 0.05), indicating that limited knowledge diversity makes models pathologically deferential to query framing. These findings highlight the need for comprehensive consistency evaluation before deploying LLMs in healthcare, legal, and financial systems where maintaining coherent positions across interactions is critical for reliable decision support.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.11108v2">A Vision for Access Control in LLM-based Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 11 pages, 1 figure
    </div>
    <details class="paper-abstract">
      The autonomy and contextual complexity of LLM-based agents render traditional access control (AC) mechanisms insufficient. Static, rule-based systems designed for predictable environments are fundamentally ill-equipped to manage the dynamic information flows inherent in agentic interactions. This position paper argues for a paradigm shift from binary access control to a more sophisticated model of information governance, positing that the core challenge is not merely about permission, but about governing the flow of information. We introduce Agent Access Control (AAC), a novel framework that reframes AC as a dynamic, context-aware process of information flow governance. AAC operates on two core modules: (1) multi-dimensional contextual evaluation, which assesses not just identity but also relationships, scenarios, and norms; and (2) adaptive response formulation, which moves beyond simple allow/deny decisions to shape information through redaction, summarization, and paraphrasing. This vision, powered by a dedicated AC reasoning engine, aims to bridge the gap between human-like nuanced judgment and scalable Al safety, proposing a new conceptual lens for future research in trustworthy agent design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16701v1">An Agentic Framework with LLMs for Solving Complex Vehicle Routing Problems</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Complex vehicle routing problems (VRPs) remain a fundamental challenge, demanding substantial expert effort for intent interpretation and algorithm design. While large language models (LLMs) offer a promising path toward automation, current approaches still rely on external intervention, which restrict autonomy and often lead to execution errors and low solution feasibility. To address these challenges, we propose an Agentic Framework with LLMs (AFL) for solving complex vehicle routing problems, achieving full automation from problem instance to solution. AFL directly extracts knowledge from raw inputs and enables self-contained code generation without handcrafted modules or external solvers. To improve trustworthiness, AFL decomposes the overall pipeline into three manageable subtasks and employs four specialized agents whose coordinated interactions enforce cross-functional consistency and logical soundness. Extensive experiments on 60 complex VRPs, ranging from standard benchmarks to practical variants, validate the effectiveness and generality of our framework, showing comparable performance against meticulously designed algorithms. Notably, it substantially outperforms existing LLM-based baselines in both code reliability and solution feasibility, achieving rates close to 100% on the evaluated benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18573v2">Enhancing Efficiency and Exploration in Reinforcement Learning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Accept by EMNLP 2025 main
    </div>
    <details class="paper-abstract">
      Reasoning large language models (LLMs) excel in complex tasks, which has drawn significant attention to reinforcement learning (RL) for LLMs. However, existing approaches allocate an equal number of rollouts to all questions during the RL process, which is inefficient. This inefficiency stems from the fact that training on simple questions yields limited gains, whereas more rollouts are needed for challenging questions to sample correct answers. Furthermore, while RL improves response precision, it limits the model's exploration ability, potentially resulting in a performance cap below that of the base model prior to RL. To address these issues, we propose a mechanism for dynamically allocating rollout budgets based on the difficulty of the problems, enabling more efficient RL training. Additionally, we introduce an adaptive dynamic temperature adjustment strategy to maintain the entropy at a stable level, thereby encouraging sufficient exploration. This enables LLMs to improve response precision while preserving their exploratory ability to uncover potential correct pathways. The code and data is available on: https://github.com/LiaoMengqi/E3-RL4LLMs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16686v1">Investigating the Impact of Rationales for LLMs on Natural Language Understanding</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Chain-of-thought (CoT) rationales, which provide step-by-step reasoning to derive final answers, benefit LLMs in both inference and training. Incorporating rationales, either by generating them before answering during inference, or by placing them before or after the original answers during training - significantly improves model performance on mathematical, symbolic and commonsense reasoning tasks. However, most work focuses on the role of rationales in these reasoning tasks, overlooking their potential impact on other important tasks like natural language understanding (NLU) tasks. In this work, we raise the question: Can rationales similarly benefit NLU tasks? To conduct a systematic exploration, we construct NLURC, a comprehensive and high-quality NLU dataset collection with rationales, and develop various rationale-augmented methods. Through exploring the applicability of these methods on NLU tasks using the dataset, we uncover several potentially surprising findings: (1) CoT inference shifts from hindering NLU performance to surpassing direct label prediction as model size grows, indicating a positive correlation. (2) Most rationale-augmented training methods perform worse than label-only training, with one specially designed method consistently achieving improvements. (3) LLMs trained with rationales achieve significant performance gains on unseen NLU tasks, rivaling models ten times their size, while delivering interpretability on par with commercial LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.02833v3">Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Published as a conference paper at Neurips 2025
    </div>
    <details class="paper-abstract">
      Despite substantial efforts in safety alignment, recent research indicates that Large Language Models (LLMs) remain highly susceptible to jailbreak attacks. Among these attacks, finetuning-based ones that compromise LLMs' safety alignment via fine-tuning stand out due to its stable jailbreak performance. In particular, a recent study indicates that fine-tuning with as few as 10 harmful question-answer (QA) pairs can lead to successful jailbreaking across various harmful questions. However, such malicious fine-tuning attacks are readily detectable and hence thwarted by moderation models. In this paper, we demonstrate that LLMs can be jailbroken by fine-tuning with only 10 benign QA pairs; our attack exploits the increased sensitivity of LLMs to fine-tuning data after being overfitted. Specifically, our fine-tuning process starts with overfitting an LLM via fine-tuning with benign QA pairs involving identical refusal answers. Further fine-tuning is then performed with standard benign answers, causing the overfitted LLM to forget the refusal attitude and thus provide compliant answers regardless of the harmfulness of a question. We implement our attack on the ten LLMs and compare it with five existing baselines. Experiments demonstrate that our method achieves significant advantages in both attack effectiveness and attack stealth. Our findings expose previously unreported security vulnerabilities in current LLMs and provide a new perspective on understanding how LLMs' security is compromised, even with benign fine-tuning. Our code is available at https://github.com/ZHIXINXIE/tenBenign.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17913v1">TACLA: An LLM-Based Multi-Agent Tool for Transactional Analysis Training in Education</a></div>
    <div class="paper-meta">
      📅 2025-10-19
      | 💬 Accepted for publication in the proceedings of ICTAI 2025
    </div>
    <details class="paper-abstract">
      Simulating nuanced human social dynamics with Large Language Models (LLMs) remains a significant challenge, particularly in achieving psychological depth and consistent persona behavior crucial for high-fidelity training tools. This paper introduces TACLA (Transactional Analysis Contextual LLM-based Agents), a novel Multi-Agent architecture designed to overcome these limitations. TACLA integrates core principles of Transactional Analysis (TA) by modeling agents as an orchestrated system of distinct Parent, Adult, and Child ego states, each with its own pattern memory. An Orchestrator Agent prioritizes ego state activation based on contextual triggers and an agent's life script, ensuring psychologically authentic responses. Validated in an educational scenario, TACLA demonstrates realistic ego state shifts in Student Agents, effectively modeling conflict de-escalation and escalation based on different teacher intervention strategies. Evaluation shows high conversational credibility and confirms TACLA's capacity to create dynamic, psychologically-grounded social simulations, advancing the development of effective AI tools for education and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17910v1">Interpretability Framework for LLMs in Undergraduate Calculus</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being used in education, yet their correctness alone does not capture the quality, reliability, or pedagogical validity of their problem-solving behavior, especially in mathematics, where multistep logic, symbolic reasoning, and conceptual clarity are critical. Conventional evaluation methods largely focus on final answer accuracy and overlook the reasoning process. To address this gap, we introduce a novel interpretability framework for analyzing LLM-generated solutions using undergraduate calculus problems as a representative domain. Our approach combines reasoning flow extraction and decomposing solutions into semantically labeled operations and concepts with prompt ablation analysis to assess input salience and output stability. Using structured metrics such as reasoning complexity, phrase sensitivity, and robustness, we evaluated the model behavior on real Calculus I to III university exams. Our findings revealed that LLMs often produce syntactically fluent yet conceptually flawed solutions, with reasoning patterns sensitive to prompt phrasing and input variation. This framework enables fine-grained diagnosis of reasoning failures, supports curriculum alignment, and informs the design of interpretable AI-assisted feedback tools. This is the first study to offer a structured, quantitative, and pedagogically grounded framework for interpreting LLM reasoning in mathematics education, laying the foundation for the transparent and responsible deployment of AI in STEM learning environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17904v1">BreakFun: Jailbreaking LLMs via Schema Exploitation</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      The proficiency of Large Language Models (LLMs) in processing structured data and adhering to syntactic rules is a capability that drives their widespread adoption but also makes them paradoxically vulnerable. In this paper, we investigate this vulnerability through BreakFun, a jailbreak methodology that weaponizes an LLM's adherence to structured schemas. BreakFun employs a three-part prompt that combines an innocent framing and a Chain-of-Thought distraction with a core "Trojan Schema"--a carefully crafted data structure that compels the model to generate harmful content, exploiting the LLM's strong tendency to follow structures and schemas. We demonstrate this vulnerability is highly transferable, achieving an average success rate of 89% across 13 foundational and proprietary models on JailbreakBench, and reaching a 100% Attack Success Rate (ASR) on several prominent models. A rigorous ablation study confirms this Trojan Schema is the attack's primary causal factor. To counter this, we introduce the Adversarial Prompt Deconstruction guardrail, a defense that utilizes a secondary LLM to perform a "Literal Transcription"--extracting all human-readable text to isolate and reveal the user's true harmful intent. Our proof-of-concept guardrail demonstrates high efficacy against the attack, validating that targeting the deceptive schema is a viable mitigation strategy. Our work provides a look into how an LLM's core strengths can be turned into critical weaknesses, offering a fresh perspective for building more robustly aligned models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17902v1">Activation Manifold Projection: Liberating Task-Specific Behaviors from LLM Architectures</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Model (LLM) architectures presents a fundamental challenge: valuable, task-specific behaviors learned through fine-tuning methods like Low-Rank Adaptation (LoRA) are effectively trapped within their source model's architecture, herein referred to architectural lock-in. Existing transfer methods attempt to bridge this gap by aligning the static weight spaces of models, a brittle and indirect approach that relies on tenuous correlations between parameter geometries. This paper introduces a fundamentally different and more direct paradigm: the Cartridge Activation Space Transfer (CAST), a novel framework that liberates LoRA-encoded behaviors by learning a direct, nonlinear mapping between the activation manifolds, the geometric structures formed by the model's internal neuron activations, of two distinct LLM architectures. CAST treats a pre-trained LoRA as a frozen "behavioral kernel." It learns a set of lightweight, bidirectional projection heads that translate the target model's activation stream into the source model's latent space, apply the frozen kernel, and project the result back. This process, trained on a general text corpus without any task-specific data, effectively decouples the learned skill from the source architecture. We demonstrate that CAST enables true "zero-shot" translation of any standard LoRA adapter. Our experiments, including transfers between heterogeneous model families like Llama-2 and Mistral, show that CAST-translated adapters achieve 85-95\% of the performance of a LoRA fully retrained on the target model, quantitatively outperforming current weight-space transfer techniques and establishing a new state-of-the-art in model interoperability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.17900v1">Are LLMs Court-Ready? Evaluating Frontier Models on Indian Legal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-19
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are entering legal workflows, yet we lack a jurisdiction-specific framework to assess their baseline competence therein. We use India's public legal examinations as a transparent proxy. Our multi-year benchmark assembles objective screens from top national and state exams and evaluates open and frontier LLMs under real-world exam conditions. To probe beyond multiple-choice questions, we also include a lawyer-graded, paired-blinded study of long-form answers from the Supreme Court's Advocate-on-Record exam. This is, to our knowledge, the first exam-grounded, India-specific yardstick for LLM court-readiness released with datasets and protocols. Our work shows that while frontier systems consistently clear historical cutoffs and often match or exceed recent top-scorer bands on objective exams, none surpasses the human topper on long-form reasoning. Grader notes converge on three reliability failure modes: procedural or format compliance, authority or citation discipline, and forum-appropriate voice and structure. These findings delineate where LLMs can assist (checks, cross-statute consistency, statute and precedent lookups) and where human leadership remains essential: forum-specific drafting and filing, procedural and relief strategy, reconciling authorities and exceptions, and ethical, accountable judgment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16492v1">Check Yourself Before You Wreck Yourself: Selectively Quitting Improves LLM Agent Safety</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Reliable ML and Regulatable ML workshops, Neurips 2025
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM) agents increasingly operate in complex environments with real-world consequences, their safety becomes critical. While uncertainty quantification is well-studied for single-turn tasks, multi-turn agentic scenarios with real-world tool access present unique challenges where uncertainties and ambiguities compound, leading to severe or catastrophic risks beyond traditional text generation failures. We propose using "quitting" as a simple yet effective behavioral mechanism for LLM agents to recognize and withdraw from situations where they lack confidence. Leveraging the ToolEmu framework, we conduct a systematic evaluation of quitting behavior across 12 state-of-the-art LLMs. Our results demonstrate a highly favorable safety-helpfulness trade-off: agents prompted to quit with explicit instructions improve safety by an average of +0.39 on a 0-3 scale across all models (+0.64 for proprietary models), while maintaining a negligible average decrease of -0.03 in helpfulness. Our analysis demonstrates that simply adding explicit quit instructions proves to be a highly effective safety mechanism that can immediately be deployed in existing agent systems, and establishes quitting as an effective first-line defense mechanism for autonomous agents in high-stakes applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14315v6">Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Maintaining consistent model performance across domains is a fundamental challenge in machine learning. While recent work has explored using LLM-generated data for fine-tuning, its impact on cross-domain generalization remains poorly understood. This paper presents a systematic analysis revealing that fine-tuning with LLM-generated data not only improves target task performance but also reduces non-target task degradation compared to fine-tuning with ground truth data. Through analyzing the data sequence in tasks of various domains, we demonstrate that this enhancement of non-target task robustness stems from the reduction of high perplexity tokens found in LLM-generated sequences. Following our findings, we showed that masking high perplexity tokens in ground truth training data achieves similar non-target task performance preservation, comparable to using LLM-generated data. Extensive experiments across different model families and scales, including Gemma 2 IT 2B, Llama 3 8B Instruct, and three additional models, agree with our findings. To the best of our knowledge, this is the first work to provide an empirical explanation based on token perplexity reduction to mitigate catastrophic forgetting in LLMs after fine-tuning, offering valuable insights for developing more robust fine-tuning strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.10906v2">Understanding LLMs' Cross-Lingual Context Retrieval: How Good It Is And Where It Comes From</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Cross-lingual context retrieval (extracting contextual information in one language based on requests in another) is a fundamental aspect of cross-lingual alignment, but the performance and mechanism of it for large language models (LLMs) remains unclear. In this paper, we evaluate the cross-lingual context retrieval of over 40 LLMs across 12 languages, using cross-lingual machine reading comprehension (xMRC) as a representative scenario. Our results show that post-trained open LLMs show strong cross-lingual context retrieval ability, comparable to closed-source LLMs such as GPT-4o, and their estimated oracle performances greatly improve after post-training. Our mechanism analysis shows that the cross-lingual context retrieval process can be divided into two main phases: question encoding and answer retrieval, which are formed in pre-training and post-training respectively. The phasing stability correlates with xMRC performance, and the xMRC bottleneck lies at the last model layers in the second phase, where the effect of post-training can be evidently observed. Our results also indicate that larger-scale pretraining cannot improve the xMRC performance. Instead, larger LLMs need further multilingual post-training to fully unlock their cross-lingual context retrieval potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17598v2">Hallucination Detection in LLMs Using Spectral Features of Attention Maps</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Accepted to EMNLP 2025. Code available at https://github.com/graphml-lab-pwr/lapeigvals
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across various tasks but remain prone to hallucinations. Detecting hallucinations is essential for safety-critical applications, and recent methods leverage attention map properties to this end, though their effectiveness remains limited. In this work, we investigate the spectral features of attention maps by interpreting them as adjacency matrices of graph structures. We propose the $\text{LapEigvals}$ method, which utilises the top-$k$ eigenvalues of the Laplacian matrix derived from the attention maps as an input to hallucination detection probes. Empirical evaluations demonstrate that our approach achieves state-of-the-art hallucination detection performance among attention-based methods. Extensive ablation studies further highlight the robustness and generalisation of $\text{LapEigvals}$, paving the way for future advancements in the hallucination detection domain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16418v1">FourierCompress: Layer-Aware Spectral Activation Compression for Efficient and Accurate Collaborative LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Collaborative large language model (LLM) inference enables real-time, privacy-preserving AI services on resource-constrained edge devices by partitioning computational workloads between client devices and edge servers. However, this paradigm is severely hindered by communication bottlenecks caused by the transmission of high-dimensional intermediate activations, exacerbated by the autoregressive decoding structure of LLMs, where bandwidth consumption scales linearly with output length. Existing activation compression methods struggle to simultaneously achieve high compression ratios, low reconstruction error, and computational efficiency. This paper proposes FourierCompress, a novel, layer-aware activation compression framework that exploits the frequency-domain sparsity of LLM activations. We rigorously demonstrate that activations from the first Transformer layer exhibit strong smoothness and energy concentration in the low-frequency domain, making them highly amenable to near-lossless compression via the Fast Fourier Transform (FFT). FourierCompress transforms activations into the frequency domain, retains only a compact block of low-frequency coefficients, and reconstructs the signal at the server using conjugate symmetry, enabling seamless hardware acceleration on DSPs and FPGAs. Extensive experiments on Llama 3 and Qwen2.5 models across 10 commonsense reasoning datasets demonstrate that FourierCompress preserves performance remarkably close to the uncompressed baseline, outperforming Top-k, QR, and SVD. FourierCompress bridges the gap between communication efficiency (an average 7.6x reduction in activation size), near-lossless inference (less than 0.3% average accuracy loss), and significantly faster compression (achieving over 32x reduction in compression time compared to Top-k via hardware acceleration) for edge-device LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16415v1">MeCeFO: Enhancing LLM Training Robustness via Fault-Tolerant Optimization</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 NeurIPS 2025 poster
    </div>
    <details class="paper-abstract">
      As distributed optimization scales to meet the demands of Large Language Model (LLM) training, hardware failures become increasingly non-negligible. Existing fault-tolerant training methods often introduce significant computational or memory overhead, demanding additional resources. To address this challenge, we propose Memory- and Computation-efficient Fault-tolerant Optimization (MeCeFO), a novel algorithm that ensures robust training with minimal overhead. When a computing node fails, MeCeFO seamlessly transfers its training task to a neighboring node while employing memory- and computation-efficient algorithmic optimizations to minimize the extra workload imposed on the neighboring node handling both tasks. MeCeFO leverages three key algorithmic designs: (i) Skip-connection, which drops the multi-head attention (MHA) module during backpropagation for memory- and computation-efficient approximation; (ii) Recomputation, which reduces activation memory in feedforward networks (FFNs); and (iii) Low-rank gradient approximation, enabling efficient estimation of FFN weight matrix gradients. Theoretically, MeCeFO matches the convergence rate of conventional distributed training, with a rate of $\mathcal{O}(1/\sqrt{nT})$, where n is the data parallelism size and T is the number of iterations. Empirically, MeCeFO maintains robust performance under high failure rates, incurring only a 4.18% drop in throughput, demonstrating 5.0$\times$ to 6.7$\times$ greater resilience than previous SOTA approaches. Codes are available at https://github.com/pkumelon/MeCeFO.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.09601v3">How Far Have LLMs Come Toward Automated SATD Taxonomy Construction?</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 5 pages, APSEC 2025
    </div>
    <details class="paper-abstract">
      Technical debt refers to suboptimal code that degrades software quality. When developers intentionally introduce such debt, it is called self-admitted technical debt (SATD). Since SATD hinders maintenance, identifying its categories is key to uncovering quality issues. Traditionally, constructing such taxonomies requires manually inspecting SATD comments and surrounding code, which is time-consuming, labor-intensive, and often inconsistent due to annotator subjectivity. In this study, we investigate to what extent large language models (LLMs) can generate SATD taxonomies. We designed a structured, LLM-driven pipeline that mirrors the taxonomy construction steps researchers typically follow. We evaluated it on SATD datasets from three domains: quantum software, smart contracts, and machine learning. It successfully recovered domain-specific categories reported in prior work, such as Layer Configuration in machine learning. It also completed taxonomy generation in under two hours and for less than $1, even on the largest dataset. These results suggest that, while full automation remains challenging, LLMs can support semi-automated SATD taxonomy construction. Furthermore, our work opens up avenues for future work, such as automated taxonomy generation in other areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.12713v2">Humanity's Last Code Exam: Can Advanced LLMs Conquer Human's Hardest Code Competition?</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Code generation is a core capability of large language models (LLMs), yet mainstream benchmarks (e.g., APPs and LiveCodeBench) contain questions with medium-level difficulty and pose no challenge to advanced LLMs. To better reflected the advanced reasoning and code generation ability, We introduce Humanity's Last Code Exam (HLCE), comprising 235 most challenging problems from the International Collegiate Programming Contest (ICPC World Finals) and the International Olympiad in Informatics (IOI) spanning 2010 - 2024. As part of HLCE, we design a harmonized online-offline sandbox that guarantees fully reproducible evaluation. Through our comprehensive evaluation, we observe that even the strongest reasoning LLMs: o4-mini(high) and Gemini-2.5 Pro, achieve pass@1 rates of only 15.9% and 11.4%, respectively. Meanwhile, we propose a novel "self-recognition" task to measure LLMs' awareness of their own capabilities. Results indicate that LLMs' self-recognition abilities are not proportionally correlated with their code generation performance. Finally, our empirical validation of test-time scaling laws reveals that current advanced LLMs have substantial room for improvement on complex programming tasks. We expect HLCE to become a milestone challenge for code generation and to catalyze advances in high-performance reasoning and human-AI collaborative programming. Our code and dataset are also public available(https://github.com/Humanity-s-Last-Code-Exam/HLCE).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16395v1">Code Digital Twin: Empowering LLMs with Tacit Knowledge for Complex Software Development</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have demonstrated strong capabilities in software engineering tasks, raising expectations of revolutionary productivity gains. However, enterprise software development is largely driven by incremental evolution, where challenges extend far beyond routine coding and depend critically on tacit knowledge, including design decisions at different levels and historical trade-offs. To achieve effective AI-powered support for complex software development, we should align emerging AI capabilities with the practical realities of enterprise development. To this end, we systematically identify challenges from both software and LLM perspectives. Alongside these challenges, we outline opportunities where AI and structured knowledge frameworks can enhance decision-making in tasks such as issue localization and impact analysis. To address these needs, we propose the Code Digital Twin, a living framework that models both the physical and conceptual layers of software, preserves tacit knowledge, and co-evolves with the codebase. By integrating hybrid knowledge representations, multi-stage extraction pipelines, incremental updates, LLM-empowered applications, and human-in-the-loop feedback, the Code Digital Twin transforms fragmented knowledge into explicit and actionable representations. Our vision positions it as a bridge between AI advancements and enterprise software realities, providing a concrete roadmap toward sustainable, intelligent, and resilient development and evolution of ultra-complex systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16384v1">SemOpt: LLM-Driven Code Optimization via Rule-Based Analysis</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Automated code optimization aims to improve performance in programs by refactoring code, and recent studies focus on utilizing LLMs for the optimization. Typical existing approaches mine optimization commits from open-source codebases to construct a large-scale knowledge base, then employ information retrieval techniques such as BM25 to retrieve relevant optimization examples for hotspot code locations, thereby guiding LLMs to optimize these hotspots. However, since semantically equivalent optimizations can manifest in syntactically dissimilar code snippets, current retrieval methods often fail to identify pertinent examples, leading to suboptimal optimization performance. This limitation significantly reduces the effectiveness of existing optimization approaches. To address these limitations, we propose SemOpt, a novel framework that leverages static program analysis to precisely identify optimizable code segments, retrieve the corresponding optimization strategies, and generate the optimized results. SemOpt consists of three key components: (1) A strategy library builder that extracts and clusters optimization strategies from real-world code modifications. (2) A rule generator that generates Semgrep static analysis rules to capture the condition of applying the optimization strategy. (3) An optimizer that utilizes the strategy library to generate optimized code results. All the three components are powered by LLMs. On our benchmark containing 151 optimization tasks, SemOpt demonstrates its effectiveness under different LLMs by increasing the number of successful optimizations by 1.38 to 28 times compared to the baseline. Moreover, on popular large-scale C/C++ projects, it can improve individual performance metrics by 5.04% to 218.07%, demonstrating its practical utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.20999v4">MoFO: Momentum-Filtered Optimizer for Mitigating Forgetting in LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks. Typically, LLMs are first pre-trained on large corpora and subsequently fine-tuned on task-specific datasets. However, during fine-tuning, LLMs may forget some knowledge acquired in the pre-training stage, leading to a decline in general capabilities. Existing approaches to mitigate forgetting often rely on access to pre-training data, which may be unavailable in many real-world scenarios--such as fine-tuning checkpoint-only open-source LLMs. To address this challenge, we propose a new fine-tuning algorithm termed Momentum-Filtered Optimizer (MoFO). MoFO is an extension of greedy block coordinate descent (BCD) methods: in each iteration, MoFO only updates the model parameters with the largest momentum magnitudes, while keeping all other parameters fixed. MoFO achieves similar fine-tuning performance to the default fine-tuning algorithm while effectively mitigating knowledge forgetting. We validate MoFO through rigorous convergence analysis and extensive experiments, demonstrating its effectiveness in mitigating forgetting without pre-training data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.13586v2">Deflanderization for Game Dialogue: Balancing Character Authenticity with Task Execution in LLM-based NPCs</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) has opened new opportunities for cre- ating dynamic non-player characters (NPCs) in gaming environments, enabling both func- tional task execution and persona-consistent dialogue generation. In this paper, we (Tu_Character_lab) report our participation in the Commonsense Persona-Grounded Dialogue Challenge (CPDC) 2025 Round 2, which eval- uates agents across three tracks: task-oriented dialogue, context-aware dialogue, and their integration. Our approach combines two complementary strategies: (i) lightweight prompting techniques in the API track, including a Deflanderization prompting method to suppress excessive role-play and improve task fidelity, and (ii) fine-tuned large models in the GPU track, leveraging Qwen3-14B with supervisedfinetuning (SFT) and Low-Rank Adaptation(LoRA). Our best submissions ranked 2nd on Task 1, 2nd on Task 3 (API track), and 4th on Task 3 (GPU track).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16373v1">Navigating through the hidden embedding space: steering LLMs to improve mental health assessment</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      The rapid evolution of Large Language Models (LLMs) is transforming AI, opening new opportunities in sensitive and high-impact areas such as Mental Health (MH). Yet, despite these advancements, recent evidence reveals that smaller-scale models still struggle to deliver optimal performance in domain-specific applications. In this study, we present a cost-efficient yet powerful approach to improve MH assessment capabilities of an LLM, without relying on any computationally intensive techniques. Our lightweight method consists of a linear transformation applied to a specific layer's activations, leveraging steering vectors to guide the model's output. Remarkably, this intervention enables the model to achieve improved results across two distinct tasks: (1) identifying whether a Reddit post is useful for detecting the presence or absence of depressive symptoms (relevance prediction task), and (2) completing a standardized psychological screening questionnaire for depression based on users' Reddit post history (questionnaire completion task). Results highlight the untapped potential of steering mechanisms as computationally efficient tools for LLMs' MH domain adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16366v1">Integrating LLM and Diffusion-Based Agents for Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 10 pages, 3 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Agent-based social simulation provides a valuable methodology for predicting social information diffusion, yet existing approaches face two primary limitations. Traditional agent models often rely on rigid behavioral rules and lack semantic understanding of textual content, while emerging large language model (LLM)-based agents incur prohibitive computational costs at scale. To address these challenges, we propose a hybrid simulation framework that strategically integrates LLM-driven agents with diffusion model-based agents. The framework employs LLM-based agents to simulate a core subset of users with rich semantic reasoning, while a diffusion model handles the remaining population efficiently. Although the two agent types operate on disjoint user groups, both incorporate key factors including user personalization, social influence, and content awareness, and interact through a coordinated simulation process. Extensive experiments on three real-world datasets demonstrate that our framework outperforms existing methods in prediction accuracy, validating the effectiveness of its modular design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.06807v2">MPCache: MPC-Friendly KV Cache Eviction for Efficient Private LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Private large language model (LLM) inference based on secure multi-party computation (MPC) achieves formal data privacy protection but suffers from significant latency overhead, especially for long input sequences. While key-value (KV) cache eviction and sparse attention algorithms have been proposed for efficient LLM inference in plaintext, they are not designed for MPC and cannot benefit private LLM inference directly. In this paper, we propose an accurate and MPC-friendly KV cache eviction framework, dubbed MPCache, building on the observation that historical tokens in a long sequence may have different effects on the downstream decoding. Hence, MPCache combines a look-once static eviction algorithm to discard unimportant KV cache and a query-aware dynamic selection algorithm to activate only a small subset of KV cache for attention computation. MPCache further incorporates a series of optimizations for efficient dynamic KV cache selection, including MPC-friendly similarity approximation, hierarchical KV cache clustering, and cross-layer index-sharing strategy. Extensive experiments demonstrate that MPCache consistently outperforms prior-art KV cache eviction baselines across different generation tasks and achieves 1.8 ~ 2.01x and 3.39 ~ 8.37x decoding latency and communication reduction on different sequence lengths, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.25835v3">Chain-in-Tree: Back to Sequential Reasoning in LLM Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 Under Review; Add codebase
    </div>
    <details class="paper-abstract">
      Test-time scaling improves large language models (LLMs) on long-horizon reasoning tasks by allocating more compute at inference. LLM Inference via Tree Search (LITS) methods achieve strong performance but are highly inefficient, often running an order of magnitude slower than iterative approaches. We propose Chain-in-Tree (CiT), a plug-in framework that decides when to branch during search rather than expanding at every step. CiT introduces lightweight Branching Necessity (BN) evaluations: BN-DP (Direct Prompting), where an auxiliary LLM judges branching needs, and BN-SC (Self-Consistency), which clusters candidate actions to assess agreement. Integrated into Tree of Thoughts, ReST-MCTS, and RAP, BN-DP achieves 75-85% reductions in token generation, model calls, and runtime on GSM8K and Math500, with often negligible or no accuracy loss. BN-SC typically yields substantial savings (up to 80%) generally but shows instability in 1-4 out of 14 settings, caused by a small subset of examples that produce extremely long reasoning steps. We theoretically prove that BN-DP never increases policy invocations and release both modular LITS implementations and a lightweight CiT function applicable across all LITS variants. The full codebase is publicly available at https://github.com/xinzhel/chain_in_tree.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.07230v2">Customer-R1: Personalized Simulation of Human Behaviors via RL-based LLM Agent in Online Shopping</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Simulating step-wise human behavior with Large Language Models (LLMs) has become an emerging research direction, enabling applications in various practical domains. While prior methods, including prompting, supervised fine-tuning (SFT), and reinforcement learning (RL), have shown promise in modeling step-wise behavior, they primarily learn a population-level policy without conditioning on a user's persona, yielding generic rather than personalized simulations. In this work, we pose a critical question: how can LLM agents better simulate personalized user behavior? We introduce Customer-R1, an RL-based method for personalized, step-wise user behavior simulation in online shopping environments. Our policy is conditioned on an explicit persona, and we optimize next-step rationale and action generation via action correctness reward signals. Experiments on the OPeRA dataset emonstrate that Customer-R1 not only significantly outperforms prompting and SFT-based baselines in next-action prediction tasks, but also better matches users' action distribution, indicating higher fidelity in personalized behavior simulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16949v4">Breaking the Exploration Bottleneck: Rubric-Scaffolded Reinforcement Learning for General LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have underscored the potential of Reinforcement Learning (RL) to facilitate the emergence of reasoning capabilities. Despite the encouraging results, a fundamental dilemma persists as RL improvement relies on learning from high-quality samples, yet the exploration for such samples remains bounded by the inherent limitations of LLMs. This, in effect, creates an undesirable cycle in which what cannot be explored cannot be learned. In this work, we propose Rubric-Scaffolded Reinforcement Learning (RuscaRL), a novel instructional scaffolding framework designed to break the exploration bottleneck for general LLM reasoning. Specifically, RuscaRL introduces checklist-style rubrics as (1) explicit scaffolding for exploration during rollout generation, where different rubrics are provided as external guidance within task instructions to steer diverse high-quality responses. This guidance is gradually decayed over time, encouraging the model to internalize the underlying reasoning patterns; (2) verifiable rewards for exploitation during model training, where we can obtain robust LLM-as-a-Judge scores using rubrics as references, enabling effective RL on general reasoning tasks. Extensive experiments demonstrate the superiority of the proposed RuscaRL across various benchmarks, effectively expanding reasoning boundaries under the Best-of-N evaluation. Notably, RuscaRL significantly boosts Qwen2.5-7B-Instruct from 23.6 to 50.3 on HealthBench-500, surpassing GPT-4.1. Furthermore, our fine-tuned variant on Qwen3-30B-A3B-Instruct achieves 61.1 on HealthBench-500, outperforming leading LLMs including OpenAI-o3. Our code is available at https://github.com/IANNXANG/RuscaRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.08907v3">Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 18 pages,9 figures
    </div>
    <details class="paper-abstract">
      Context compression presents a promising approach for accelerating large language model (LLM) inference by compressing long contexts into compact representations. Current context compression methods predominantly rely on autoencoding tasks to train context-agnostic compression tokens to compress contextual semantics. While autoencoding tasks enable compression tokens to acquire compression capabilities, compression via autoencoding tasks creates a fundamental mismatch: the models are optimized for reconstruction that diverge from actual downstream tasks, thereby weakening the features more beneficial for real-world usage. We propose Semantic-Anchor Compression (SAC), a novel method that shifts from autoencoding task based compression to an architecture that is equipped with this compression capability \textit{a priori}. Instead of training models to compress contexts through autoencoding tasks, SAC directly selects so-called anchor tokens from the original context and aggregates contextual information into their key-value (KV) representations. By deriving representations directly from the contextual tokens, SAC eliminates the need for autoencoding training. To ensure compression performance while directly leveraging anchor tokens, SAC incorporates two key designs: (1) anchor embeddings that enable the compressor to identify critical tokens, and (2) bidirectional attention modification that allows anchor tokens to capture information from the entire context. Experimental results demonstrate that SAC consistently outperforms existing context compression methods across various compression ratios. On out-of-distribution evaluation using MRQA, SAC achieves 1 EM improvement at 5x compression over strong baselines, with increasing advantages at higher compression ratios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.17333v3">Whose Journey Matters? Investigating Identity Biases in Large Language Models (LLMs) for Travel Planning Assistance</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly integral to the hospitality and tourism industry, concerns about their fairness in serving diverse identity groups persist. Grounded in social identity theory and sociotechnical systems theory, this study examines ethnic and gender biases in travel recommendations generated by LLMs. Using fairness probing, we analyze outputs from three leading open-source LLMs. The results show that test accuracy for both ethnicity and gender classifiers exceed random chance. Analysis of the most influential features reveals the presence of stereotype bias in LLM-generated recommendations. We also found hallucinations among these features, occurring more frequently in recommendations for minority groups. These findings indicate that LLMs exhibit ethnic and gender bias when functioning as travel planning assistants. This study underscores the need for bias mitigation strategies to improve the inclusivity and reliability of generative AI-driven travel planning assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.10833v5">MergeBench: A Benchmark for Merging Domain-Specialized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 NeurIPS 2025 Track on Datasets and Benchmarks
    </div>
    <details class="paper-abstract">
      Model merging provides a scalable alternative to multi-task training by combining specialized finetuned models through parameter arithmetic, enabling efficient deployment without the need for joint training or access to all task data. While recent methods have shown promise, existing evaluations are limited in both model scale and task diversity, leaving open questions about their applicability to large, domain-specialized LLMs. To tackle the challenges, we introduce MergeBench, a comprehensive evaluation suite designed to assess model merging at scale. MergeBench builds on state-of-the-art open-source language models, including Llama and Gemma families at 2B to 9B scales, and covers five key domains: instruction following, mathematics, multilingual understanding, coding and safety. We standardize finetuning and evaluation protocols, and assess eight representative merging methods across multi-task performance, forgetting and runtime efficiency. Based on extensive experiments, we provide practical guidelines for algorithm selection and share insights showing that model merging tends to perform better on stronger base models, with techniques such as merging coefficient tuning and sparsification improving knowledge retention. However, several challenges remain, including the computational cost on large models, the gap for in-domain performance compared to multi-task models, and the underexplored role of model merging in standard LLM training pipelines. We hope MergeBench provides a foundation for future research to advance the understanding and practical application of model merging. Our project page is at \href{https://yifei-he.github.io/mergebench/}{https://yifei-he.github.io/mergebench/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21184v2">BED-LLM: Intelligent Information Gathering with LLMs and Bayesian Experimental Design</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      We propose a general-purpose approach for improving the ability of Large Language Models (LLMs) to intelligently and adaptively gather information from a user or other external source using the framework of sequential Bayesian experimental design (BED). This enables LLMs to act as effective multi-turn conversational agents and interactively interface with external environments. Our approach, which we call BED-LLM (Bayesian Experimental Design with Large Language Models), is based on iteratively choosing questions or queries that maximize the expected information gain (EIG) about the task of interest given the responses gathered previously. We show how this EIG can be formulated (and then estimated) in a principled way using a probabilistic model derived from the LLM's predictive distributions and provide detailed insights into key decisions in its construction and updating procedure. We find that BED-LLM achieves substantial gains in performance across a wide range of tests based on the 20 questions game and using the LLM to actively infer user preferences, compared to direct prompting of the LLM and other adaptive design strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13776v2">A Knapsack by Any Other Name: Presentation impacts LLM performance on NP-hard problems</a></div>
    <div class="paper-meta">
      📅 2025-10-18
      | 💬 24 pages, 6 figures, EMNLP 2025
    </div>
    <details class="paper-abstract">
      To investigate the effect of problem presentation on LLMs' ability to solve optimization problems, we introduce the dataset of Everyday Hard Optimization Problems (EHOP), a collection of NP-hard problems expressed in natural language. EHOP includes problem formulations that could be found in computer science textbooks (e.g., graph coloring), versions that are dressed up as problems that could arise in real life (e.g., party planning), and variants with inverted rules. We find that state-of-the-art LLMs, across multiple prompting strategies, systematically solve textbook problems more accurately than their real-life and inverted counterparts. While reasoning models are more capable, they nonetheless show high variance across problem presentations, suggesting they lack a truly robust reasoning mechanism. We argue that this constitutes evidence that LLMs are still heavily dependent on what was seen in training and struggle to generalize to novel problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16645v1">Unleashing Diverse Thinking Modes in LLMs through Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate strong performance but often lack interpretable reasoning. This paper introduces the Multi-Agent Collaboration Framework for Diverse Thinking Modes (DiMo), which enhances both performance and interpretability by simulating a structured debate among four specialized LLM agents. Each agent embodies a distinct reasoning paradigm, allowing the framework to collaboratively explore diverse cognitive approaches. Through iterative debate, agents challenge and refine initial responses, yielding more robust conclusions and an explicit, auditable reasoning chain. Across six benchmarks and under a unified open-source setup, DiMo improves accuracy over widely used single-model and debate baselines, with the largest gains on math. We position DiMo as a semantics-aware, Web-native multi-agent framework: it models human-machine intelligence with LLM agents that produce semantically typed, URL-annotated evidence chains for explanations and user-friendly interactions. Although our experiments use standard reasoning benchmarks, the framework is designed to be instantiated over Web corpora and knowledge graphs, combining retrieval-augmented reasoning with structured justifications that downstream systems can inspect and reuse.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16614v1">Count Counts: Motivating Exploration in LLM Reasoning with Count-based Intrinsic Rewards</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      Reinforcement Learning (RL) has become a compelling way to strengthen the multi step reasoning ability of Large Language Models (LLMs). However, prevalent RL paradigms still lean on sparse outcome-based rewards and limited exploration, which often drives LLMs toward repetitive and suboptimal reasoning patterns. In this paper, we study the central question of how to design exploration for LLM reasoning and introduce MERCI (Motivating Exploration in LLM Reasoning with Count-based Intrinsic Rewards), a novel RL algorithm that augments policy optimization with a principled intrinsic reward. Building on the idea of count-based exploration, MERCI leverages a lightweight Coin Flipping Network (CFN) to estimate the pseudo count and further epistemic uncertainty over reasoning trajectories, and converts them into an intrinsic reward that values novelty while preserving the learning signal from task rewards. We integrate MERCI into some advanced RL frameworks like Group Relative Policy Optimization (GRPO). Experiments on complex reasoning benchmarks demonstrate that MERCI encourages richer and more varied chains of thought, significantly improves performance over strong baselines, and helps the policy escape local routines to discover better solutions. It indicates that our targeted intrinsic motivation can make exploration reliable for language model reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.16610v1">Structuring Security: A Survey of Cybersecurity Ontologies, Semantic Log Processing, and LLMs Application</a></div>
    <div class="paper-meta">
      📅 2025-10-18
    </div>
    <details class="paper-abstract">
      This survey investigates how ontologies, semantic log processing, and Large Language Models (LLMs) enhance cybersecurity. Ontologies structure domain knowledge, enabling interoperability, data integration, and advanced threat analysis. Security logs, though critical, are often unstructured and complex. To address this, automated construction of Knowledge Graphs (KGs) from raw logs is emerging as a key strategy for organizing and reasoning over security data. LLMs enrich this process by providing contextual understanding and extracting insights from unstructured content. This work aligns with European Union (EU) efforts such as NIS 2 and the Cybersecurity Taxonomy, highlighting challenges and opportunities in intelligent ontology-driven cyber defense.
    </details>
</div>
