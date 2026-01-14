# llm - 2026_01

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01094v1">SoulSeek: Exploring the Use of Social Cues in LLM-based Information Seeking</a></div>
    <div class="paper-meta">
      📅 2026-01-03
    </div>
    <details class="paper-abstract">
      Social cues, which convey others' presence, behaviors, or identities, play a crucial role in human information seeking by helping individuals judge relevance and trustworthiness. However, existing LLM-based search systems primarily rely on semantic features, creating a misalignment with the socialized cognition underlying natural information seeking. To address this gap, we explore how the integration of social cues into LLM-based search influences users' perceptions, experiences, and behaviors. Focusing on social media platforms that are beginning to adopt LLM-based search, we integrate design workshops, the implementation of the prototype system (SoulSeek), a between-subjects study, and mixed-method analyses to examine both outcome- and process-level findings. The workshop informs the prototype's cue-integrated design. The study shows that social cues improve perceived outcomes and experiences, promote reflective information behaviors, and reveal limits of current LLM-based search. We propose design implications emphasizing better social-knowledge understanding, personalized cue settings, and controllable interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01046v1">KV-Embedding: Training-free Text Embedding via Internal KV Re-routing in Decoder-only LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-03
    </div>
    <details class="paper-abstract">
      While LLMs are powerful embedding backbones, their application in training-free settings faces two structural challenges: causal attention restricts early tokens from accessing subsequent context, and the next-token prediction objective biases representations toward generation rather than semantic compression. To address these limitations, we propose KV-Embedding, a framework that activates the latent representation power of frozen LLMs. Our method leverages the observation that the key-value (KV) states of the final token at each layer encode a compressed view of the sequence. By re-routing these states as a prepended prefix, we enable all tokens to access sequence-level context within a single forward pass. To ensure model-agnostic applicability, we introduce an automated layer selection strategy based on intrinsic dimensionality. Evaluations on MTEB across Qwen, Mistral, and Llama backbones show that KV-Embedding outperforms existing training-free baselines by up to 10%, while maintaining robust performance on sequences up to 4,096 tokens. These results demonstrate that internal state manipulation offers an efficient alternative to input modification, and we hope this work encourages further exploration of LLM internals for representation learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01015v1">HyperJoin: LLM-augmented Hypergraph Link Prediction for Joinable Table Discovery</a></div>
    <div class="paper-meta">
      📅 2026-01-03
    </div>
    <details class="paper-abstract">
      As a pivotal task in data lake management, joinable table discovery has attracted widespread interest. While existing language model-based methods achieve remarkable performance by combining offline column representation learning with online ranking, their design insufficiently accounts for the underlying structural interactions: (1) offline, they directly model tables into isolated or pairwise columns, thereby struggling to capture the rich inter-table and intra-table structural information; and (2) online, they rank candidate columns based solely on query-candidate similarity, ignoring the mutual interactions among the candidates, leading to incoherent result sets. To address these limitations, we propose HyperJoin, a large language model (LLM)-augmented Hypergraph framework for Joinable table discovery. Specifically, we first construct a hypergraph to model tables using both the intra-table hyperedges and the LLM-augmented inter-table hyperedges. Consequently, the task of joinable table discovery is formulated as link prediction on this constructed hypergraph. We then design HIN, a Hierarchical Interaction Network that learns expressive column representations through bidirectional message passing over columns and hyperedges. To strengthen coherence and internal consistency in the result columns, we cast online ranking as a coherence-aware top-k column selection problem. We then introduce a reranking module that leverages a maximum spanning tree algorithm to prune noisy connections and maximize coherence. Experiments demonstrate the superiority of HyperJoin, achieving average improvements of 21.4% (Precision@15) and 17.2% (Recall@15) over the best baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01296v1">Aggressive Compression Enables LLM Weight Theft</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 An early version of this work was presented at the SoLAR Workshop at NeurIPS 2024
    </div>
    <details class="paper-abstract">
      As frontier AIs become more powerful and costly to develop, adversaries have increasing incentives to steal model weights by mounting exfiltration attacks. In this work, we consider exfiltration attacks where an adversary attempts to sneak model weights out of a datacenter over a network. While exfiltration attacks are multi-step cyber attacks, we demonstrate that a single factor, the compressibility of model weights, significantly heightens exfiltration risk for large language models (LLMs). We tailor compression specifically for exfiltration by relaxing decompression constraints and demonstrate that attackers could achieve 16x to 100x compression with minimal trade-offs, reducing the time it would take for an attacker to illicitly transmit model weights from the defender's server from months to days. Finally, we study defenses designed to reduce exfiltration risk in three distinct ways: making models harder to compress, making them harder to 'find,' and tracking provenance for post-attack analysis using forensic watermarks. While all defenses are promising, the forensic watermark defense is both effective and cheap, and therefore is a particularly attractive lever for mitigating weight-exfiltration risk.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.18665v4">Membership Inference Attacks on LLM-based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 This is paper is under review WWW 2026
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) based recommender systems (RecSys) can adapt to different domains flexibly. It utilizes in-context learning (ICL), i.e., prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, encompassing implicit feedback such as clicked items and explicit product reviews. Such private information may be exposed by novel privacy attacks. However, no study has been conducted on this important issue. We design several membership inference attacks (MIAs) aimed to revealing whether system prompts include victims' historical interactions. The attacks are \emph{Similarity, Memorization, Inquiry, and Poisoning attacks}, each utilizing unique features of LLMs or RecSys. We have carefully evaluated them on five of the latest open-source LLMs and three well-known RecSys benchmark datasets. The results confirm that the MIA threat to LLM RecSys is realistic: inquiry and poisoning attacks show significantly high attack advantages. We also discussed possible methods to mitigate such MIA threats. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts, the position of the victim in the shots, the number of poisoning items in the prompt,etc.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01279v1">LLM Collusion</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 44 pages
    </div>
    <details class="paper-abstract">
      We study how delegating pricing to large language models (LLMs) can facilitate collusion in a duopoly when both sellers rely on the same pre-trained model. The LLM is characterized by (i) a propensity parameter capturing its internal bias toward high-price recommendations and (ii) an output-fidelity parameter measuring how tightly outputs track that bias; the propensity evolves through retraining. We show that configuring LLMs for robustness and reproducibility can induce collusion via a phase transition: there exists a critical output-fidelity threshold that pins down long-run behavior. Below it, competitive pricing is the unique long-run outcome. Above it, the system is bistable, with competitive and collusive pricing both locally stable and the realized outcome determined by the model's initial preference. The collusive regime resembles tacit collusion: prices are elevated on average, yet occasional low-price recommendations provide plausible deniability. With perfect fidelity, full collusion emerges from any interior initial condition. For finite training batches of size $b$, infrequent retraining (driven by computational costs) further amplifies collusion: conditional on starting in the collusive basin, the probability of collusion approaches one as $b$ grows, since larger batches dampen stochastic fluctuations that might otherwise tip the system toward competition. The indeterminacy region shrinks at rate $O(1/\sqrt{b})$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01271v1">CatchAll: Repository-Aware Exception Handling with Knowledge-Guided LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-03
    </div>
    <details class="paper-abstract">
      Exception handling is a vital forward error-recovery mechanism in many programming languages, enabling developers to manage runtime anomalies through structured constructs (e.g., try-catch blocks). Improper or missing exception handling often leads to severe consequences, including system crashes and resource leaks. While large language models (LLMs) have demonstrated strong capabilities in code generation, they struggle with exception handling at the repository level, due to complex dependencies and contextual constraints. In this work, we propose CatchAll, a novel LLM-based approach for repository-aware exception handling. CatchAll equips LLMs with three complementary layers of exception-handling knowledge: (1) API-level exception knowledge, obtained from an empirically constructed API-exception mapping that characterizes the exception-throwing behaviors of APIs in real-world codebases; (2) repository-level execution context, which captures exception propagation by modeling contextual call traces around the target code; and (3) cross-repository handling knowledge, distilled from reusable exception-handling patterns mined from historical code across projects. The knowledge is encoded into structured prompts to guide the LLM in generating accurate and context-aware exception-handling code. To evaluate CatchAll, we construct two new benchmarks for repository-aware exception handling: a large-scale dataset RepoExEval and an executable subset RepoExEval-Exec. Experiments demonstrate that RepoExEval consistently outperforms state-of-the-art baselines, achieving a CodeBLEU score of 0.31 (vs. 0.27% for the best baseline), intent prediction accuracy of 60.1% (vs. 48.0%), and Pass@1 of 29% (vs. 25%). These results affirm RepoExEval's effectiveness in real-world repository-level exception handling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.15815v2">User-Assistant Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-03
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) are typically trained and deployed using structured role tags (e.g. system, user, assistant, tool) that explicitly mark the source of each piece of context. While these tags are essential for instruction following and controllability, asymmetries in the training data associated with different role tags can introduce inductive biases. In this paper, we study this phenomenon by formalizing user-assistant bias, defined as the tendency of an LLM to preferentially rely on information from either the user or assistant role when there is a conflict. We introduce a task-agnostic benchmark UserAssist and evaluate such bias in 52 frontier models. We observe that most of the instruction-tuned models exhibit strong user bias, whereas base and reasoning models are close to neutral. Using controlled fine-tuning experiments, we isolate which post-training recipes drive the observed user-assistant bias. We find that human-preference alignment amplifies user bias, while reasoning fine-tuning reduces it. Finally, we show that user-assistant bias can be bidirectionally controlled via direct preference optimization (DPO) on UserAssist-train, and that the resulting bias reliably generalizes to a more realistic multi-turn conversation dataset. These results reveal an underexplored consequence of role-tagged training and provide a principled framework to diagnose and control tag-induced biases in modern LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.07745v4">Chimera: Harnessing Multi-Agent LLMs for Automatic Insider Threat Simulation</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 Accepted by NDSS 2026
    </div>
    <details class="paper-abstract">
      Insider threats pose a persistent and critical security risk, yet are notoriously difficult to detect in complex enterprise environments, where malicious actions are often hidden within seemingly benign user behaviors. Although machine-learning-based insider threat detection (ITD) methods have shown promise, their effectiveness is fundamentally limited by the scarcity of high-quality and realistic training data. Enterprise internal data is highly sensitive and rarely accessible, while existing public and synthetic datasets are either small-scale or lack sufficient realism, semantic richness, and behavioral diversity. To address this challenge, we propose Chimera, an LLM-based multi-agent framework that automatically simulates both benign and malicious insider activities and generates comprehensive system logs across diverse enterprise environments. Chimera models each agent as an individual employee with fine-grained roles and supports group meetings, pairwise interactions, and self-organized scheduling to capture realistic organizational dynamics. Based on 15 insider attacks abstracted from real-world incidents, we deploy Chimera in three representative data-sensitive organizational scenarios and construct ChimeraLog, a new dataset for developing and evaluating ITD methods. We evaluate ChimeraLog through human studies and quantitative analyses, demonstrating its diversity and realism. Experiments with existing ITD methods show substantially lower detection performance on ChimeraLog compared to prior datasets, indicating a more challenging and realistic benchmark. Moreover, despite distribution shifts, models trained on ChimeraLog exhibit strong generalization, highlighting the practical value of LLM-based multi-agent simulation for advancing insider threat detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01233v1">Atomizer: An LLM-based Collaborative Multi-Agent Framework for Intent-Driven Commit Untangling</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 Accepted by ICSE 2026
    </div>
    <details class="paper-abstract">
      Composite commits, which entangle multiple unrelated concerns, are prevalent in software development and significantly hinder program comprehension and maintenance. Existing automated untangling methods, particularly state-of-the-art graph clustering-based approaches, are fundamentally limited by two issues. (1) They over-rely on structural information, failing to grasp the crucial semantic intent behind changes, and (2) they operate as ``single-pass'' algorithms, lacking a mechanism for the critical reflection and refinement inherent in human review processes. To overcome these challenges, we introduce Atomizer, a novel collaborative multi-agent framework for composite commit untangling. To address the semantic deficit, Atomizer employs an Intent-Oriented Chain-of-Thought (IO-CoT) strategy, which prompts large language models (LLMs) to infer the intent of each code change according to both the structure and the semantic information of code. To overcome the limitations of ``single-pass'' grouping, we employ two agents to establish a grouper-reviewer collaborative refinement loop, which mirrors human review practices by iteratively refining groupings until all changes in a cluster share the same underlying semantic intent. Extensive experiments on two benchmark C# and Java datasets demonstrate that Atomizer significantly outperforms several representative baselines. On average, it surpasses the state-of-the-art graph-based methods by over 6.0% on the C# dataset and 5.5% on the Java dataset. This superiority is particularly pronounced on complex commits, where Atomizer's performance advantage widens to over 16%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01215v1">Correctness isnt Efficiency: Runtime Memory Divergence in LLM-Generated Code</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 11 Pages, 11 figures, Accepted at ICSE SEIP
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can generate programs that pass unit tests, but passing tests does not guarantee reliable runtime behavior. We find that different correct solutions to the same task can show very different memory and performance patterns, which can lead to hidden operational risks. We present a framework to measure execution-time memory stability across multiple correct generations. At the solution level, we introduce Dynamic Mean Pairwise Distance (DMPD), which uses Dynamic Time Warping to compare the shapes of memory-usage traces after converting them into Monotonic Peak Profiles (MPPs) to reduce transient noise. Aggregating DMPD across tasks yields a model-level Model Instability Score (MIS). Experiments on BigOBench and CodeContests show substantial runtime divergence among correct solutions. Instability often increases with higher sampling temperature even when pass@1 improves. We also observe correlations between our stability measures and software engineering indicators such as cognitive and cyclomatic complexity, suggesting links between operational behavior and maintainability. Our results support stability-aware selection among passing candidates in CI/CD to reduce operational risk without sacrificing correctness. Artifacts are available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2512.23489v2">The Gaining Paths to Investment Success: Information-Driven LLM Graph Reasoning for Venture Capital Prediction</a></div>
    <div class="paper-meta">
      📅 2026-01-03
    </div>
    <details class="paper-abstract">
      Most venture capital (VC) investments fail, while a few deliver outsized returns. Accurately predicting startup success requires synthesizing complex relational evidence, including company disclosures, investor track records, and investment network structures, through explicit reasoning to form coherent, interpretable investment theses. Traditional machine learning and graph neural networks both lack this reasoning capability. Large language models (LLMs) offer strong reasoning but face a modality mismatch with graphs. Recent graph-LLM methods target in-graph tasks where answers lie within the graph, whereas VC prediction is off-graph: the target exists outside the network. The core challenge is selecting graph paths that maximize predictor performance on an external objective while enabling step-by-step reasoning. We present MIRAGE-VC, a multi-perspective retrieval-augmented generation framework that addresses two obstacles: path explosion (thousands of candidate paths overwhelm LLM context) and heterogeneous evidence fusion (different startups need different analytical emphasis). Our information-gain-driven path retriever iteratively selects high-value neighbors, distilling investment networks into compact chains for explicit reasoning. A multi-agent architecture integrates three evidence streams via a learnable gating mechanism based on company attributes. Under strict anti-leakage controls, MIRAGE-VC achieves +5.0% F1 and +16.6% PrecisionAt5, and sheds light on other off-graph prediction tasks such as recommendation and risk assessment. Code: https://anonymous.4open.science/r/MIRAGE-VC-323F.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01196v1">EduSim-LLM: An Educational Platform Integrating Large Language Models and Robotic Simulation for Beginners</a></div>
    <div class="paper-meta">
      📅 2026-01-03
    </div>
    <details class="paper-abstract">
      In recent years, the rapid development of Large Language Models (LLMs) has significantly enhanced natural language understanding and human-computer interaction, creating new opportunities in the field of robotics. However, the integration of natural language understanding into robotic control is an important challenge in the rapid development of human-robot interaction and intelligent automation industries. This challenge hinders intuitive human control over complex robotic systems, limiting their educational and practical accessibility. To address this, we present the EduSim-LLM, an educational platform that integrates LLMs with robot simulation and constructs a language-drive control model that translates natural language instructions into executable robot behavior sequences in CoppeliaSim. We design two human-robot interaction models: direct control and autonomous control, conduct systematic simulations based on multiple language models, and evaluate multi-robot collaboration, motion planning, and manipulation capabilities. Experiential results show that LLMs can reliably convert natural language into structured robot actions; after applying prompt-engineering templates instruction-parsing accuracy improves significantly; as task complexity increases, overall accuracy rate exceeds 88.9% in the highest complexity tests.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.22732v2">Reasoning Beyond Limits: Advances and Open Problems for LLMs</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 The paper is published ICT Express Volume 11, Issue 6, December 2025, Pages 1054-1096
    </div>
    <details class="paper-abstract">
      Recent breakthroughs in generative reasoning have fundamentally reshaped how large language models (LLMs) address complex tasks, enabling them to dynamically retrieve, refine, and organize information into coherent multi-step reasoning chains. Techniques such as inference-time scaling, reinforcement learning, supervised fine-tuning, and distillation have been effectively applied to state-of-the-art models, including DeepSeek-R1, OpenAI o1 and o3, GPT-4o, Qwen-32B, and various Llama variants, significantly enhancing their reasoning capabilities. In this paper, we present a comprehensive review of the top 27 LLMs released between 2023 and 2025, such as Mistral AI Small 3 24B, DeepSeek-R1, Search-o1, QwQ-32B, and Phi-4, and analyze their core innovations and performance improvements. We also provide a detailed overview of recent advancements in multilingual large language models (MLLMs), emphasizing methods that improve cross-lingual reasoning and address the limitations of English-centric training. In parallel, we present a comprehensive review of progress in state space model (SSM)-based architectures, including models such as Mamba, which demonstrate improved efficiency for long-context processing compared to transformer-based approaches. Our analysis covers training strategies including general optimization techniques, mixture-of-experts (MoE) configurations, retrieval-augmented generation (RAG), chain-of-thought prompting, self-improvement methods, and test-time compute scaling and distillation frameworks. Finally, we identify key challenges for future research, including enabling multi-step reasoning without human supervision, improving robustness in chained task execution, balancing structured prompting with generative flexibility, and enhancing the integration of long-context retrieval and external tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.23019v4">Stairway to Success: An Online Floor-Aware Zero-Shot Object-Goal Navigation Framework via LLM-Driven Coarse-to-Fine Exploration</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 Accepted to IEEE Robotics and Automation Letters (RAL). Project Page at https://zeying-gong.github.io/projects/ascent
    </div>
    <details class="paper-abstract">
      Deployable service and delivery robots struggle to navigate multi-floor buildings to reach object goals, as existing systems fail due to single-floor assumptions and requirements for offline, globally consistent maps. Multi-floor environments pose unique challenges including cross-floor transitions and vertical spatial reasoning, especially navigating unknown buildings. Object-Goal Navigation benchmarks like HM3D and MP3D also capture this multi-floor reality, yet current methods lack support for online, floor-aware navigation. To bridge this gap, we propose \textbf{\textit{ASCENT}}, an online framework for Zero-Shot Object-Goal Navigation that enables robots to operate without pre-built maps or retraining on new object categories. It introduces: (1) a \textbf{Multi-Floor Abstraction} module that dynamically constructs hierarchical representations with stair-aware obstacle mapping and cross-floor topology modeling, and (2) a \textbf{Coarse-to-Fine Reasoning} module that combines frontier ranking with LLM-driven contextual analysis for multi-floor navigation decisions. We evaluate on HM3D and MP3D benchmarks, outperforming state-of-the-art zero-shot approaches, and demonstrate real-world deployment on a quadruped robot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.05887v2">A three-Level Framework for LLM-Enhanced eXplainable AI: From technical explanations to natural language</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 22 pages, 5 figures
    </div>
    <details class="paper-abstract">
      The growing application of artificial intelligence in sensitive domains has intensified the demand for systems that are not only accurate but also explainable and trustworthy. Although explainable AI (XAI) methods have proliferated, many do not consider the diverse audiences that interact with AI systems: from developers and domain experts to end-users and society. This paper addresses how trust in AI is influenced by the design and delivery of explanations and proposes a multilevel framework that aligns explanations with the epistemic, contextual, and ethical expectations of different stakeholders. The framework consists of three layers: algorithmic and domain-based, human-centered, and social explainability, with Large Language Models serving as crucial mediators that transform technical outputs of AI explanations into accessible, contextual narratives across all levels. We show how LLMs enable dynamic, conversational explanations that bridge the gap between complex model behavior and human understanding, facilitating interactive dialogue and enhancing societal transparency. Through comprehensive case studies, we show how this LLM-enhanced approach achieves technical fidelity, user engagement, and societal accountability, reframing XAI as a dynamic, trust-building process that leverages natural language capabilities to democratize AI explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.01129v1">RovoDev Code Reviewer: A Large-Scale Online Evaluation of LLM-based Code Review Automation at Atlassian</a></div>
    <div class="paper-meta">
      📅 2026-01-03
      | 💬 Accepted at the 48th International Conference on Software Engineering (ICSE'26), SEIP Track. 12 Pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs)-powered code review automation has the potential to transform code review workflows. Despite the advances of LLM-powered code review comment generation approaches, several practical challenges remain for designing enterprise-grade code review automation tools. In particular, this paper aims at answering the practical question: how can we design a review-guided, context-aware, quality-checked code review comment generation without fine-tuning? In this paper, we present RovoDev Code Reviewer, an enterprise-grade LLM-based code review automation tool designed and deployed at scale within Atlassian's development ecosystem with seamless integration into Atlassian's Bitbucket. Through the offline, online, user feedback evaluations over a one-year period, we conclude that RovoDev Code Reviewer is (1) effective in generating code review comments that could lead to code resolution for 38.70% (i.e., comments that triggered code changes in the subsequent commits); and (2) offers the promise of accelerating feedback cycles (i.e., decreasing the PR cycle time by 30.8%), alleviating reviewer workload (i.e., reducing the number of human-written comments by 35.6%), and improving overall software quality (i.e., finding errors with actionable suggestions).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00930v1">AlignUSER: Human-Aligned LLM Agents via World Models for Recommender System Evaluation</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      Evaluating recommender systems remains challenging due to the gap between offline metrics and real user behavior, as well as the scarcity of interaction data. Recent work explores large language model (LLM) agents as synthetic users, yet they typically rely on few-shot prompting, which yields a shallow understanding of the environment and limits their ability to faithfully reproduce user actions. We introduce AlignUSER, a framework that learns world-model-driven agents from human interactions. Given rollout sequences of actions and states, we formalize world modeling as a next state prediction task that helps the agent internalize the environment. To align actions with human personas, we generate counterfactual trajectories around demonstrations and prompt the LLM to compare its decisions with human choices, identify suboptimal actions, and extract lessons. The learned policy is then used to drive agent interactions with the recommender system. We evaluate AlignUSER across multiple datasets and demonstrate closer alignment with genuine humans than prior work, both at the micro and macro levels.
    </details>
</div>
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
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.17972v4">LABOR-LLM: Language-Based Occupational Representations with Large Language Models</a></div>
    <div class="paper-meta">
      📅 2026-01-02
    </div>
    <details class="paper-abstract">
      This paper builds an empirical model that predicts a worker's next occupation as a function of the worker's occupational history. Because histories are sequences of occupations, the covariate space is high-dimensional, and further, the outcome (the next occupation) is a discrete choice that can take on many values. To estimate the parameters of the model, we leverage an approach from generative artificial intelligence. Estimation begins from a ``foundation model'' trained on non-representative data and then ``fine-tunes'' the estimation using data about careers from a representative survey. We convert tabular data from the survey into text files that resemble resumes and fine-tune the parameters of the foundation model, a large language model (LLM), using these text files with the objective of predicting the next token (word). The resulting fine-tuned LLM is used to calculate estimates of worker transition probabilities. Its predictive performance surpasses all prior models, both for the task of granularly predicting the next occupation as well as for specific tasks such as predicting whether the worker changes occupations or stays in the labor force. We quantify the value of fine-tuning and further show that by adding more career data from a different population, fine-tuning smaller LLMs (fewer parameters) surpasses the performance of fine-tuning larger models. When we omit the English language occupational title and replace it with a unique code, predictive performance declines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.23323v2">LLM Interpretability with Identifiable Temporal-Instantaneous Representation</a></div>
    <div class="paper-meta">
      📅 2026-01-02
      | 💬 NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Despite Large Language Models' remarkable capabilities, understanding their internal representations remains challenging. Mechanistic interpretability tools such as sparse autoencoders (SAEs) were developed to extract interpretable features from LLMs but lack temporal dependency modeling, instantaneous relation representation, and more importantly theoretical guarantees, undermining both the theoretical foundations and the practical confidence necessary for subsequent analyses. While causal representation learning (CRL) offers theoretically grounded approaches for uncovering latent concepts, existing methods cannot scale to LLMs' rich conceptual space due to inefficient computation. To bridge the gap, we introduce an identifiable temporal causal representation learning framework specifically designed for LLMs' high-dimensional concept space, capturing both time-delayed and instantaneous causal relations. Our approach provides theoretical guarantees and demonstrates efficacy on synthetic datasets scaled to match real-world complexity. By extending SAE techniques with our temporal causal framework, we successfully discover meaningful concept relationships in LLM activations. Our findings show that modeling both temporal and instantaneous conceptual relationships advances the interpretability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.22922v3">Improving Human Verification of LLM Reasoning through Interactive Explanation Interfaces</a></div>
    <div class="paper-meta">
      📅 2026-01-02
      | 💬 19 pages, 14 figures
    </div>
    <details class="paper-abstract">
      The reasoning capabilities of Large Language Models (LLMs) have led to their increasing employment in several critical applications, particularly education, where they support problem-solving, tutoring, and personalized study. Chain-of-thought (CoT) reasoning capabilities [1, 2] are well-known to help LLMs decompose a problem into steps and explore the solution spaces more effectively, leading to impressive performance on mathematical and reasoning benchmarks. As the length of CoT tokens per question increases substantially to even thousands of tokens per question [ 1], it is unknown how users could comprehend LLM reasoning and detect errors or hallucinations. To address this problem and understand how reasoning can improve human-AI interaction, we present three new interactive reasoning interfaces: interactive CoT (iCoT), interactive Program-of-Thought (iPoT), and interactive Graph (iGraph). That is, we ask LLMs themselves to generate an interactive web interface wrapped around the original CoT content, which may be presented in text (iCoT), graphs (iGraph) or code (iPoT). This interface allows users to interact with and provide a novel experience in reading and validating the reasoning chains of LLMs. Across a study of 125 participants, interactive interfaces significantly improve user performance. Specifically, iGraph users score the highest error detection rate (85.6%), followed by iPoT (82.5%), iCoT (80.6%), all outperforming standard CoT (73.5%). Interactive interfaces also lead to faster user validation time-iGraph users are faster (57.9 secs per question) than the users of iCoT and iPoT (60 secs) and the standard CoT (64.7 secs). A post-study questionnaire shows that users prefer iGraph, citing its superior ability to enable them to follow the LLM's reasoning. We discuss the implications of these results and provide recommendations for the future design of reasoning models.
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
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00926v1">MACA: A Framework for Distilling Trustworthy LLMs into Efficient Retrievers</a></div>
    <div class="paper-meta">
      📅 2026-01-01
    </div>
    <details class="paper-abstract">
      Modern enterprise retrieval systems must handle short, underspecified queries such as ``foreign transaction fee refund'' and ``recent check status''. In these cases, semantic nuance and metadata matter but per-query large language model (LLM) re-ranking and manual labeling are costly. We present Metadata-Aware Cross-Model Alignment (MACA), which distills a calibrated metadata aware LLM re-ranker into a compact student retriever, avoiding online LLM calls. A metadata-aware prompt verifies the teacher's trustworthiness by checking consistency under permutations and robustness to paraphrases, then supplies listwise scores, hard negatives, and calibrated relevance margins. The student trains with MACA's MetaFusion objective, which combines a metadata conditioned ranking loss with a cross model margin loss so it learns to push the correct answer above semantically similar candidates with mismatched topic, sub-topic, or entity. On a proprietary consumer banking FAQ corpus and BankFAQs, the MACA teacher surpasses a MAFA baseline at Accuracy@1 by five points on the proprietary set and three points on BankFAQs. MACA students substantially outperform pretrained encoders; e.g., on the proprietary corpus MiniLM Accuracy@1 improves from 0.23 to 0.48, while keeping inference free of LLM calls and supporting retrieval-augmented generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.00912v1">The Discovery Gap: How Product Hunt Startups Vanish in LLM Organic Discovery Queries</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 20 pages, 7 figures. Based on M.Tech thesis research, Indian Institute of Technology Patna, 2025
    </div>
    <details class="paper-abstract">
      When someone asks ChatGPT to recommend a project management tool, which products show up in the response? And more importantly for startup founders: will their newly launched product ever appear? This research set out to answer these questions. I randomly selected 112 startups from the top 500 products featured on the 2025 Product Hunt leaderboard and tested each one across 2,240 queries to two different large language models: ChatGPT (gpt-4o-mini) and Perplexity (sonar with web search). The results were striking. When users asked about products by name, both LLMs recognized them almost perfectly: 99.4% for ChatGPT and 94.3% for Perplexity. But when users asked discovery-style questions like "What are the best AI tools launched this year?" the success rates collapsed to 3.32% and 8.29% respectively. That's a gap of 30-to-1 for ChatGPT. Perhaps the most surprising finding was that Generative Engine Optimization (GEO), the practice of optimizing website content for AI visibility, showed no correlation with actual discovery rates. Products with high GEO scores were no more likely to appear in organic queries than products with low scores. What did matter? For Perplexity, traditional SEO signals like referring domains (r = +0.319, p < 0.001) and Product Hunt ranking (r = -0.286, p = 0.002) predicted visibility. After cleaning the Reddit data for false positives, community presence also emerged as significant (r = +0.395, p = 0.002). The practical takeaway is counterintuitive: don't optimize for AI discovery directly. Instead, build the SEO foundation first and LLM visibility will follow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2601.02407v1">Evolving Personalities in Chaos: An LLM-Augmented Framework for Character Discovery in the Iterated Prisoners Dilemma under Environmental Stress</a></div>
    <div class="paper-meta">
      📅 2026-01-01
      | 💬 10 pages, 5 figures. Project assignment; exploratory study on LLM-based adaptive agents
    </div>
    <details class="paper-abstract">
      Standard simulations of the Iterated Prisoners Dilemma (IPD) operate in deterministic, noise-free environments, producing strategies that may be theoretically optimal but fragile when confronted with real-world uncertainty. This paper addresses two critical gaps in evolutionary game theory research: (1) the absence of realistic environmental stressors during strategy evolution, and (2) the Interpretability Gap, where evolved genetic strategies remain opaque binary sequences devoid of semantic meaning. We introduce a novel framework combining stochastic environmental perturbations (God Mode) with Large Language Model (LLM)-based behavioral profiling to transform evolved genotypes into interpretable character archetypes. Our experiments demonstrate that strategies evolved under chaotic conditions exhibit superior resilience and present distinct behavioral phenotypes, ranging from Ruthless Capitalists to Diplomatic Enforcers. These phenotypes are readily classified by LLMs but remain nearly impossible to interpret through manual genome inspection alone. This work bridges evolutionary computation with explainable AI and provides a template for automated agent characterization in multi-agent systems.
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
