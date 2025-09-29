# llm - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- Part 8
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13351v1">Teaching LLMs to Plan: Logical Chain-of-Thought Instruction Tuning for Symbolic Planning</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities across diverse tasks, yet their ability to perform structured symbolic planning remains limited, particularly in domains requiring formal representations like the Planning Domain Definition Language (PDDL). In this paper, we present a novel instruction tuning framework, PDDL-Instruct, designed to enhance LLMs' symbolic planning capabilities through logical chain-of-thought reasoning. Our approach focuses on teaching models to rigorously reason about action applicability, state transitions, and plan validity using explicit logical inference steps. By developing instruction prompts that guide models through the precise logical reasoning required to determine when actions can be applied in a given state, we enable LLMs to self-correct their planning processes through structured reflection. The framework systematically builds verification skills by decomposing the planning process into explicit reasoning chains about precondition satisfaction, effect application, and invariant preservation. Experimental results on multiple planning domains show that our chain-of-thought reasoning based instruction-tuned models are significantly better at planning, achieving planning accuracy of up to 94% on standard benchmarks, representing a 66% absolute improvement over baseline models. This work bridges the gap between the general reasoning capabilities of LLMs and the logical precision required for automated planning, offering a promising direction for developing better AI planning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11466v1">Improving LLMs' Learning for Coreference Resolution</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Coreference Resolution (CR) is crucial for many NLP tasks, but existing LLMs struggle with hallucination and under-performance. In this paper, we investigate the limitations of existing LLM-based approaches to CR-specifically the Question-Answering (QA) Template and Document Template methods and propose two novel techniques: Reversed Training with Joint Inference and Iterative Document Generation. Our experiments show that Reversed Training improves the QA Template method, while Iterative Document Generation eliminates hallucinations in the generated source text and boosts coreference resolution. Integrating these methods and techniques offers an effective and robust solution to LLM-based coreference resolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11446v1">Large Language Models (LLMs) for Requirements Engineering (RE): A Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are finding applications in numerous domains, and Requirements Engineering (RE) is increasingly benefiting from their capabilities to assist with complex, language-intensive tasks. This paper presents a systematic literature review of 74 primary studies published between 2023 and 2024, examining how LLMs are being applied in RE. The study categorizes the literature according to several dimensions, including publication trends, RE activities, prompting strategies, and evaluation methods. Our findings indicate notable patterns, among which we observe substantial differences compared to previous works leveraging standard Natural Language Processing (NLP) techniques. Most of the studies focus on using LLMs for requirements elicitation and validation, rather than defect detection and classification, which were dominant in the past. Researchers have also broadened their focus and addressed novel tasks, e.g., test generation, exploring the integration of RE with other software engineering (SE) disciplines. Although requirements specifications remain the primary focus, other artifacts are increasingly considered, including issues from issue tracking systems, regulations, and technical manuals. The studies mostly rely on GPT-based models, and often use Zero-shot or Few-shot prompting. They are usually evaluated in controlled environments, with limited use in industry settings and limited integration in complex workflows. Our study outlines important future directions, such as leveraging the potential to expand the influence of RE in SE, exploring less-studied tasks, improving prompting methods, and testing in real-world environments. Our contribution also helps researchers and practitioners use LLMs more effectively in RE, by providing a list of identified tools leveraging LLMs for RE, as well as datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11420v1">Trading-R1: Financial Trading with LLM Reasoning via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-14
      | 💬 Tauric Research: https://github.com/TauricResearch
    </div>
    <details class="paper-abstract">
      Developing professional, structured reasoning on par with human financial analysts and traders remains a central challenge in AI for finance, where markets demand interpretability and trust. Traditional time-series models lack explainability, while LLMs face challenges in turning natural-language analysis into disciplined, executable trades. Although reasoning LLMs have advanced in step-by-step planning and verification, their application to risk-sensitive financial decisions is underexplored. We present Trading-R1, a financially-aware model that incorporates strategic thinking and planning for comprehensive thesis composition, facts-grounded analysis, and volatility-adjusted decision making. Trading-R1 aligns reasoning with trading principles through supervised fine-tuning and reinforcement learning with a three-stage easy-to-hard curriculum. Training uses Tauric-TR1-DB, a 100k-sample corpus spanning 18 months, 14 equities, and five heterogeneous financial data sources. Evaluated on six major equities and ETFs, Trading-R1 demonstrates improved risk-adjusted returns and lower drawdowns compared to both open-source and proprietary instruction-following models as well as reasoning models. The system generates structured, evidence-based investment theses that support disciplined and interpretable trading decisions. Trading-R1 Terminal will be released at https://github.com/TauricResearch/Trading-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04249v2">IOLBENCH: Benchmarking LLMs on Linguistic Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Despite the remarkable advancements and widespread applications of deep neural networks, their ability to perform reasoning tasks remains limited, particularly in domains requiring structured, abstract thought. In this paper, we investigate the linguistic reasoning capabilities of state-of-the-art large language models (LLMs) by introducing IOLBENCH, a novel benchmark derived from International Linguistics Olympiad (IOL) problems. This dataset encompasses diverse problems testing syntax, morphology, phonology, and semantics, all carefully designed to be self-contained and independent of external knowledge. These tasks challenge models to engage in metacognitive linguistic reasoning, requiring the deduction of linguistic rules and patterns from minimal examples. Through extensive benchmarking of leading LLMs, we find that even the most advanced models struggle to handle the intricacies of linguistic complexity, particularly in areas demanding compositional generalization and rule abstraction. Our analysis highlights both the strengths and persistent limitations of current models in linguistic problem-solving, offering valuable insights into their reasoning capabilities. By introducing IOLBENCH, we aim to foster further research into developing models capable of human-like reasoning, with broader implications for the fields of computational linguistics and artificial intelligence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11362v1">PersonaX: Multimodal Datasets with LLM-Inferred Behavior Traits</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Understanding human behavior traits is central to applications in human-computer interaction, computational social science, and personalized AI systems. Such understanding often requires integrating multiple modalities to capture nuanced patterns and relationships. However, existing resources rarely provide datasets that combine behavioral descriptors with complementary modalities such as facial attributes and biographical information. To address this gap, we present PersonaX, a curated collection of multimodal datasets designed to enable comprehensive analysis of public traits across modalities. PersonaX consists of (1) CelebPersona, featuring 9444 public figures from diverse occupations, and (2) AthlePersona, covering 4181 professional athletes across 7 major sports leagues. Each dataset includes behavioral trait assessments inferred by three high-performing large language models, alongside facial imagery and structured biographical features. We analyze PersonaX at two complementary levels. First, we abstract high-level trait scores from text descriptions and apply five statistical independence tests to examine their relationships with other modalities. Second, we introduce a novel causal representation learning (CRL) framework tailored to multimodal and multi-measurement data, providing theoretical identifiability guarantees. Experiments on both synthetic and real-world data demonstrate the effectiveness of our approach. By unifying structured and unstructured analysis, PersonaX establishes a foundation for studying LLM-inferred behavioral traits in conjunction with visual and biographical attributes, advancing multimodal trait analysis and causal reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.11353v1">Do Large Language Models Favor Recent Content? A Study on Recency Bias in LLM-Based Reranking</a></div>
    <div class="paper-meta">
      📅 2025-09-14
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed in information systems, including being used as second-stage rerankers in information retrieval pipelines, yet their susceptibility to recency bias has received little attention. We investigate whether LLMs implicitly favour newer documents by prepending artificial publication dates to passages in the TREC Deep Learning passage retrieval collections in 2021 (DL21) and 2022 (DL22). Across seven models, GPT-3.5-turbo, GPT-4o, GPT-4, LLaMA-3 8B/70B, and Qwen-2.5 7B/72B, "fresh" passages are consistently promoted, shifting the Top-10's mean publication year forward by up to 4.78 years and moving individual items by as many as 95 ranks in our listwise reranking experiments. Although larger models attenuate the effect, none eliminate it. We also observe that the preference of LLMs between two passages with an identical relevance level can be reversed by up to 25% on average after date injection in our pairwise preference experiments. These findings provide quantitative evidence of a pervasive recency bias in LLMs and highlight the importance of effective bias-mitigation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.21094v2">Video-LLMs with Temporal Visual Screening</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      Humans naturally perform temporal screening by dragging the progress bar and focusing on salient temporal segments, but current Video Large Language Models (Video-LLMs) struggle to capture fine-grained temporal semantics due to sparse frame sampling and insufficient inter-frame reasoning supervision during their training. To address this, Inspired by well-established cognitive science principles, we propose Temporal Visual Screening (TVS), a new task that universally pre-processes video question answering and instruction tuning data by: (1) retaining focus-critical video segments, (2) synchronously reconstructing queries to their most direct form while preserving answer consistency, and (3) keeping the invariance and consistency for any possible answer. TVS is formulated as a modular front-end adapter task that can be seamlessly integrated into both Video Instruction Tuning (training) and Video Question Answering (inference) pipelines. TVS optimizes distribution of reasoning burden and cognitive load; during training, it aligns queries with focus-critical visual information; at inference, it enables query-aware segment focus and streamlined query representations. In particular, we curate the first benchmark for TVS and propose ReSimplifyIt, a baseline outperforming prior approaches on seemingly similar tasks by 0.47 in F-1 score on video trimming while achieving competitive query rewriting performance. Experiments demonstrate that incorporating TVS yields relative gains of 7.33% (training) and 34.6% (inference), demonstrating the effectiveness of temporal information screening for improving video-language understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10975v1">ReFineG: Synergizing Small Supervised Models and LLMs for Low-Resource Grounded Multimodal NER</a></div>
    <div class="paper-meta">
      📅 2025-09-13
      | 💬 CCKS 2025 Shared Task Paper
    </div>
    <details class="paper-abstract">
      Grounded Multimodal Named Entity Recognition (GMNER) extends traditional NER by jointly detecting textual mentions and grounding them to visual regions. While existing supervised methods achieve strong performance, they rely on costly multimodal annotations and often underperform in low-resource domains. Multimodal Large Language Models (MLLMs) show strong generalization but suffer from Domain Knowledge Conflict, producing redundant or incorrect mentions for domain-specific entities. To address these challenges, we propose ReFineG, a three-stage collaborative framework that integrates small supervised models with frozen MLLMs for low-resource GMNER. In the Training Stage, a domain-aware NER data synthesis strategy transfers LLM knowledge to small models with supervised training while avoiding domain knowledge conflicts. In the Refinement Stage, an uncertainty-based mechanism retains confident predictions from supervised models and delegates uncertain ones to the MLLM. In the Grounding Stage, a multimodal context selection algorithm enhances visual grounding through analogical reasoning. In the CCKS2025 GMNER Shared Task, ReFineG ranked second with an F1 score of 0.6461 on the online leaderboard, demonstrating its effectiveness with limited annotations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.07238v2">Can Advanced LLMs Coach Smaller LLMs? Knowledge Distillation for Goal-Oriented Dialogs</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      Enterprises deploying LLMs for goal-oriented dialogs, such as customer service, face a critical trade-off between performance, control, and cost. Proprietary models like GPT-4 offer strong performance but are costly and cannot be self-hosted, raising security and privacy concerns. Open-source alternatives offer flexibility and lower token costs but lag in performance. We introduce Guidance Elicitation and Retrieval (GER), a prompt-based knowledge distillation framework where a high-performance teacher LLM coaches a lower-performance student without modifying the student's parameters. GER extracts tactical guidance for a wide range of dialog scenarios from the teacher and stores these scenario-guidance pairs in a structured library. At inference time, the student retrieves the relevant guidance and integrates it into its prompt. While GER training can be bootstrapped entirely with synthetic data, its modular design lets it seamlessly augment the synthetic data with human conversational logs. In addition, the modular design enables easy auditing and updating of the guidance library as new scenarios and constraints emerge. Experiments show GER's guidance-based coaching outperforms both example output based fine-tuning and non-customized guidance baselines, and generalizes across other contexts and student models. The GER framework is potentially extensible to coach human service agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10972v1">Enhancing Computational Cognitive Architectures with LLMs: A Case Study</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      Computational cognitive architectures are broadly scoped models of the human mind that combine different psychological functionalities (as well as often different computational methods for these different functionalities) into one unified framework. They structure them in a psychologically plausible and validated way. However, such models thus far have only limited computational capabilities, mostly limited by the computational tools and techniques that were adopted. More recently, LLMs have proved to be more capable computationally than any other tools. Thus, in order to deal with both real-world complexity and psychological realism at the same time, incorporating LLMs into cognitive architectures naturally becomes an important task. In the present article, a synergistic combination of the Clarion cognitive architecture and LLMs is discussed as a case study. The implicit-explicit dichotomy that is fundamental to Clarion is leveraged for a seamless integration of Clarion and LLMs. As a result, computational power of LLMs is combined with psychological nicety of Clarion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10946v1">When the Code Autopilot Breaks: Why LLMs Falter in Embedded Machine Learning</a></div>
    <div class="paper-meta">
      📅 2025-09-13
      | 💬 This paper has been accepted for publication in Computer (IEEE). Upon publication, the copyright will be transferred to IEEE
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to automate software generation in embedded machine learning workflows, yet their outputs often fail silently or behave unpredictably. This article presents an empirical investigation of failure modes in LLM-powered ML pipelines, based on an autopilot framework that orchestrates data preprocessing, model conversion, and on-device inference code generation. We show how prompt format, model behavior, and structural assumptions influence both success rates and failure characteristics, often in ways that standard validation pipelines fail to detect. Our analysis reveals a diverse set of error-prone behaviors, including format-induced misinterpretations and runtime-disruptive code that compiles but breaks downstream. We derive a taxonomy of failure categories and analyze errors across multiple LLMs, highlighting common root causes and systemic fragilities. Though grounded in specific devices, our study reveals broader challenges in LLM-based code generation. We conclude by discussing directions for improving reliability and traceability in LLM-powered embedded ML systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22169v2">ReliableEval: A Recipe for Stochastic LLM Evaluation via Method of Moments</a></div>
    <div class="paper-meta">
      📅 2025-09-13
      | 💬 Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      LLMs are highly sensitive to prompt phrasing, yet standard benchmarks typically report performance using a single prompt, raising concerns about the reliability of such evaluations. In this work, we argue for a stochastic method of moments evaluation over the space of meaning-preserving prompt perturbations. We introduce a formal definition of reliable evaluation that accounts for prompt sensitivity, and suggest ReliableEval - a method for estimating the number of prompt resamplings needed to obtain meaningful results. Using our framework, we stochastically evaluate five frontier LLMs and find that even top-performing models like GPT-4o and Claude-3.7-Sonnet exhibit substantial prompt sensitivity. Our approach is model-, task-, and metric-agnostic, offering a recipe for meaningful and robust LLM evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10931v1">Harmful Prompt Laundering: Jailbreaking LLMs with Abductive Styles and Symbolic Encoding</a></div>
    <div class="paper-meta">
      📅 2025-09-13
      | 💬 EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse tasks, but their potential misuse for harmful purposes remains a significant concern. To strengthen defenses against such vulnerabilities, it is essential to investigate universal jailbreak attacks that exploit intrinsic weaknesses in the architecture and learning paradigms of LLMs. In response, we propose \textbf{H}armful \textbf{P}rompt \textbf{La}undering (HaPLa), a novel and broadly applicable jailbreaking technique that requires only black-box access to target models. HaPLa incorporates two primary strategies: 1) \textit{abductive framing}, which instructs LLMs to infer plausible intermediate steps toward harmful activities, rather than directly responding to explicit harmful queries; and 2) \textit{symbolic encoding}, a lightweight and flexible approach designed to obfuscate harmful content, given that current LLMs remain sensitive primarily to explicit harmful keywords. Experimental results show that HaPLa achieves over 95% attack success rate on GPT-series models and 70% across all targets. Further analysis with diverse symbolic encoding rules also reveals a fundamental challenge: it remains difficult to safely tune LLMs without significantly diminishing their helpfulness in responding to benign queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10860v1">Quantifier Scope Interpretation in Language Learners and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      Sentences with multiple quantifiers often lead to interpretive ambiguities, which can vary across languages. This study adopts a cross-linguistic approach to examine how large language models (LLMs) handle quantifier scope interpretation in English and Chinese, using probabilities to assess interpretive likelihood. Human similarity (HS) scores were used to quantify the extent to which LLMs emulate human performance across language groups. Results reveal that most LLMs prefer the surface scope interpretations, aligning with human tendencies, while only some differentiate between English and Chinese in the inverse scope preferences, reflecting human-similar patterns. HS scores highlight variability in LLMs' approximation of human behavior, but their overall potential to align with humans is notable. Differences in model architecture, scale, and particularly models' pre-training data language background, significantly influence how closely LLMs approximate human quantifier scope interpretations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.06646v2">Evaluating and Aligning Human Economic Risk Preferences in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in decision-making scenarios that involve risk assessment, yet their alignment with human economic rationality remains unclear. In this study, we investigate whether LLMs exhibit risk preferences consistent with human expectations across different personas. Specifically, we assess whether LLM-generated responses reflect appropriate levels of risk aversion or risk-seeking behavior based on individual's persona. Our results reveal that while LLMs make reasonable decisions in simplified, personalized risk contexts, their performance declines in more complex economic decision-making tasks. To address this, we propose an alignment method designed to enhance LLM adherence to persona-specific risk preferences. Our approach improves the economic rationality of LLMs in risk-related applications, offering a step toward more human-aligned AI decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10830v1">The Siren Song of LLMs: How Users Perceive and Respond to Dark Patterns in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      Large language models can influence users through conversation, creating new forms of dark patterns that differ from traditional UX dark patterns. We define LLM dark patterns as manipulative or deceptive behaviors enacted in dialogue. Drawing on prior work and AI incident reports, we outline a diverse set of categories with real-world examples. Using them, we conducted a scenario-based study where participants (N=34) compared manipulative and neutral LLM responses. Our results reveal that recognition of LLM dark patterns often hinged on conversational cues such as exaggerated agreement, biased framing, or privacy intrusions, but these behaviors were also sometimes normalized as ordinary assistance. Users' perceptions of these dark patterns shaped how they respond to them. Responsibilities for these behaviors were also attributed in different ways, with participants assigning it to companies and developers, the model itself, or to users. We conclude with implications for design, advocacy, and governance to safeguard user autonomy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10818v1">LLM Enhancement with Domain Expert Mental Model to Reduce LLM Hallucination with Causal Prompt Engineering</a></div>
    <div class="paper-meta">
      📅 2025-09-13
      | 💬 25 pages,4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Difficult decision-making problems abound in various disciplines and domains. The proliferation of generative techniques, especially large language models (LLMs), has excited interest in using them for decision support. However, LLMs cannot yet resolve missingness in their training data, leading to hallucinations. Retrieval-Augmented Generation (RAG) enhances LLMs by incorporating external information retrieval, reducing hallucinations and improving accuracy. Yet, RAG and related methods are only partial solutions, as they may lack access to all necessary sources or key missing information. Even everyday issues often challenge LLMs' abilities. Submitting longer prompts with context and examples is one approach to address knowledge gaps, but designing effective prompts is non-trivial and may not capture complex mental models of domain experts. For tasks with missing critical information, LLMs are insufficient, as are many existing systems poorly represented in available documents. This paper explores how LLMs can make decision-making more efficient, using a running example of evaluating whether to respond to a call for proposals. We propose a technology based on optimized human-machine dialogue and monotone Boolean and k-valued functions to discover a computationally tractable personal expert mental model (EMM) of decision-making. Our EMM algorithm for LLM prompt engineering has four steps: (1) factor identification, (2) hierarchical structuring of factors, (3) generating a generalized expert mental model specification, and (4) generating a detailed generalized expert mental model from that specification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.06770v2">Another Turn, Better Output? A Turn-Wise Analysis of Iterative LLM Prompting</a></div>
    <div class="paper-meta">
      📅 2025-09-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are now used in multi-turn workflows, but we still lack a clear way to measure when iteration helps and when it hurts. We present an evaluation framework for iterative refinement that spans ideation, code, and math. Our protocol runs controlled 12-turn conversations per task, utilizing a variety of prompts ranging from vague ``improve it'' feedback to targeted steering, and logs per-turn outputs. We score outcomes with domain-appropriate checks (unit tests for code; answer-equivalence plus reasoning-soundness for math; originality and feasibility for ideation) and track turn-level behavior with three families of metrics: semantic movement across turns, turn-to-turn change, and output size growth. Across models and tasks, gains are domain-dependent: they arrive early in ideas and code, but in math late turns matter when guided by elaboration. After the first few turns, vague feedback often plateaus or reverses correctness, while targeted prompts reliably shift the intended quality axis (novelty vs. feasibility in ideation; speed vs. readability in code; in math, elaboration outperforms exploration and drives late-turn gains). We also observe consistent domain patterns: ideation moves more in meaning across turns, code tends to grow in size with little semantic change, and math starts fixed but can break that path with late, elaborative iteration. Together, the framework and metrics make iteration measurable and comparable across models, and signal when to steer, stop, or switch strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07894v3">HiPhO: How Far Are (M)LLMs from Humans in the Latest High School Physics Olympiad Benchmark?</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Recently, the physical capabilities of (M)LLMs have garnered increasing attention. However, existing benchmarks for physics suffer from two major gaps: they neither provide systematic and up-to-date coverage of real-world physics competitions such as physics Olympiads, nor enable direct performance comparison with humans. To bridge these gaps, we present HiPhO, the first benchmark dedicated to high school physics Olympiads with human-aligned evaluation. Specifically, HiPhO highlights three key innovations. (1) Comprehensive Data: It compiles 13 latest Olympiad exams from 2024-2025, spanning both international and regional competitions, and covering mixed modalities that encompass problems spanning text-only to diagram-based. (2) Professional Evaluation: We adopt official marking schemes to perform fine-grained grading at both the answer and step level, fully aligned with human examiners to ensure high-quality and domain-specific evaluation. (3) Comparison with Human Contestants: We assign gold, silver, and bronze medals to models based on official medal thresholds, thereby enabling direct comparison between (M)LLMs and human contestants. Our large-scale evaluation of 30 state-of-the-art (M)LLMs shows that: across 13 exams, open-source MLLMs mostly remain at or below the bronze level; open-source LLMs show promising progress with multiple golds; closed-source reasoning MLLMs can achieve 6 to 12 gold medals; and most models still have a significant gap from full marks. These results highlight the performance gap between open-source models and top students, the strong reasoning abilities of closed-source models, and the remaining room for improvement. HiPhO, a human-aligned Olympiad benchmark for multimodal physical reasoning, is open-source at https://github.com/SciYu/HiPhO with a public leaderboard at https://phyarena.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10625v1">No Answer Needed: Predicting LLM Answer Accuracy from Question-Only Linear Probes</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Do large language models (LLMs) anticipate when they will answer correctly? To study this, we extract activations after a question is read but before any tokens are generated, and train linear probes to predict whether the model's forthcoming answer will be correct. Across three open-source model families ranging from 7 to 70 billion parameters, projections on this "in-advance correctness direction" trained on generic trivia questions predict success in distribution and on diverse out-of-distribution knowledge datasets, outperforming black-box baselines and verbalised predicted confidence. Predictive power saturates in intermediate layers, suggesting that self-assessment emerges mid-computation. Notably, generalisation falters on questions requiring mathematical reasoning. Moreover, for models responding "I don't know", doing so strongly correlates with the probe score, indicating that the same direction also captures confidence. By complementing previous results on truthfulness and other behaviours obtained with probes and sparse auto-encoders, our work contributes essential findings to elucidate LLM internals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10594v1">SME-TEAM: Leveraging Trust and Ethics for Secure and Responsible Use of AI and LLMs in SMEs</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Artificial Intelligence (AI) and Large Language Models (LLMs) are reshaping today's business practices, however, their adoption within small and medium-sized enterprises (SMEs) raises significant technical, ethical and trust issues. This paper proposes a structured, multi-phased framework designed to embed trust and ethical principles throughout the AI lifecycle for their secure and responsible use in SMEs. Structured around four pillars, i.e., Data, Algorithms, Human oversight, and Model Architecture, the framework bridges theoretical ethical principles with operational practice, enhancing AI capabilities in diverse SME applications. Ultimately, this paper offers a structured roadmap for responsible AI adoption, framing trust and ethics as a catalyst for resilience, competitiveness, and sustainable innovation in SMEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10436v1">RefactorCoderQA: Benchmarking LLMs for Multi-Domain Coding Question Solutions in Cloud and Edge Deployment</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 12 pages, 5 figures, submitted to IEEE Transactions on Services Computing
    </div>
    <details class="paper-abstract">
      To optimize the reasoning and problem-solving capabilities of Large Language Models (LLMs), we propose a novel cloud-edge collaborative architecture that enables a structured, multi-agent prompting framework. This framework comprises three specialized components: GuideLLM, a lightweight model deployed at the edge to provide methodological guidance; SolverLLM, a more powerful model hosted in the cloud responsible for generating code solutions; and JudgeLLM, an automated evaluator for assessing solution correctness and quality. To evaluate and demonstrate the effectiveness of this architecture in realistic settings, we introduce RefactorCoderQA, a comprehensive benchmark designed to evaluate and enhance the performance of Large Language Models (LLMs) across multi-domain coding tasks. Motivated by the limitations of existing benchmarks, RefactorCoderQA systematically covers various technical domains, including Software Engineering, Data Science, Machine Learning, and Natural Language Processing, using authentic coding challenges from Stack Overflow. Extensive experiments reveal that our fine-tuned model, RefactorCoder-MoE, achieves state-of-the-art performance, significantly outperforming leading open-source and commercial baselines with an overall accuracy of 76.84%. Human evaluations further validate the interpretability, accuracy, and practical relevance of the generated solutions. In addition, we evaluate system-level metrics, such as throughput and latency, to gain deeper insights into the performance characteristics and trade-offs of the proposed architecture.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18889v2">Are LLMs Better than Reported? Detecting Label Errors and Mitigating Their Effect on Model Performance</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      NLP benchmarks rely on standardized datasets for training and evaluating models and are crucial for advancing the field. Traditionally, expert annotations ensure high-quality labels; however, the cost of expert annotation does not scale well with the growing demand for larger datasets required by modern models. While crowd-sourcing provides a more scalable solution, it often comes at the expense of annotation precision and consistency. Recent advancements in large language models (LLMs) offer new opportunities to enhance the annotation process, particularly for detecting label errors in existing datasets. In this work, we consider the recent approach of LLM-as-a-judge, leveraging an ensemble of LLMs to flag potentially mislabeled examples. We conduct a case study on four factual consistency datasets from the TRUE benchmark, spanning diverse NLP tasks, and on SummEval, which uses Likert-scale ratings of summary quality across multiple dimensions. We empirically analyze the labeling quality of existing datasets and compare expert, crowd-sourced, and LLM-based annotations in terms of the agreement, label quality, and efficiency, demonstrating the strengths and limitations of each annotation method. Our findings reveal a substantial number of label errors, which, when corrected, induce a significant upward shift in reported model performance. This suggests that many of the LLMs' so-called mistakes are due to label errors rather than genuine model failures. Additionally, we discuss the implications of mislabeled data and propose methods to mitigate them in training to improve performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10402v1">Developer-LLM Conversations: An Empirical Study of Interactions and Generated Code Quality</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming integral to modern software development workflows, assisting developers with code generation, API explanation, and iterative problem-solving through natural language conversations. Despite widespread adoption, there is limited understanding of how developers interact with LLMs in practice and how these conversational dynamics influence task outcomes, code quality, and software engineering workflows. To address this, we leverage CodeChat, a large dataset comprising 82,845 real-world developer-LLM conversations, containing 368,506 code snippets generated across over 20 programming languages, derived from the WildChat dataset. We find that LLM responses are substantially longer than developer prompts, with a median token-length ratio of 14:1. Multi-turn conversations account for 68% of the dataset and often evolve due to shifting requirements, incomplete prompts, or clarification requests. Topic analysis identifies web design (9.6% of conversations) and neural network training (8.7% of conversations) as the most frequent LLM-assisted tasks. Evaluation across five languages (i.e., Python, JavaScript, C++, Java, and C#) reveals prevalent and language-specific issues in LLM-generated code: generated Python and JavaScript code often include undefined variables (83.4% and 75.3% of code snippets, respectively); Java code lacks required comments (75.9%); C++ code frequently omits headers (41.1%) and C# code shows unresolved namespaces (49.2%). During a conversation, syntax and import errors persist across turns; however, documentation quality in Java improves by up to 14.7%, and import handling in Python improves by 3.7% over 5 turns. Prompts that point out mistakes in code generated in prior turns and explicitly request a fix are most effective for resolving errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02896v2">Cut Costs, Not Accuracy: LLM-Powered Data Processing with Guarantees</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 To appear in SIGMOD'26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are being increasingly used as a building block in data systems to process large text datasets. To do so, LLM model providers offer multiple LLMs with different sizes, spanning various cost-quality trade-offs when processing text at scale. Top-of-the-line LLMs (e.g., GPT-4o, Claude Sonnet) operate with high accuracy but are prohibitively expensive when processing many records. To avoid high costs, more affordable but lower quality LLMs (e.g., GPT-4o-mini, Claude Haiku) can be used to process records, but we need to ensure that the overall accuracy does not deviate substantially from that of the top-of-the-line LLMs. The model cascade framework provides a blueprint to manage this trade-off, by using the confidence of LLMs in their output (e.g., log-probabilities) to decide on which records to use the affordable LLM. However, existing solutions following this framework provide only marginal cost savings and weak theoretical guarantees because of poor estimation of the quality of the affordable LLM's outputs. We present BARGAIN, a method that judiciously uses affordable LLMs in data processing to significantly reduce cost while providing strong theoretical guarantees on the solution quality. BARGAIN employs a novel adaptive sampling strategy and statistical estimation procedure that uses data and task characteristics and builds on recent statistical tools to make accurate estimations with tight theoretical guarantees. Variants of BARGAIN can support guarantees on accuracy, precision, or recall of the output. Experimental results across 8 real-world datasets show that BARGAIN reduces cost, on average, by up to 86% more than state-of-the-art, while providing stronger theoretical guarantees on accuracy of output, with similar gains when guaranteeing a desired level of precision or recall.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10377v1">Dropping Experts, Recombining Neurons: Retraining-Free Pruning for Sparse Mixture-of-Experts LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 Accepted to EMNLP2025
    </div>
    <details class="paper-abstract">
      Sparse Mixture-of-Experts (SMoE) architectures are widely used in large language models (LLMs) due to their computational efficiency. However, though only a few experts are activated for each token, SMoE still requires loading all expert parameters, leading to high memory usage and challenges in deployment. Previous work has tried to reduce the overhead by pruning and merging experts, but primarily focused on expert-level operations, leaving neuron-level structure underexplored. We propose DERN (Dropping Experts, Recombining Neurons), a task-agnostic and retraining-free framework for expert pruning and reconstruction. We observe that experts are often misaligned and contain semantic conflicts at the neuron level, which poses challenges for direct merging. To solve this, DERN works in three steps: it first prunes redundant experts using router statistics; then it decomposes them into neuron-level expert segments, assigning each segment to its most compatible retained expert; and finally, it merges segments within each retained expert to build a compact representation. Experiments on Mixtral, Qwen, and DeepSeek SMoE models show that DERN improves performance by more than 5% on commonsense reasoning and MMLU benchmarks under 50% expert sparsity, without extra training. It also greatly reduces the number of experts and memory usage, making SMoE LLMs easier to deploy in practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10372v1">MCBP: A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) face significant inference latency due to inefficiencies in GEMM operations, weight access, and KV cache access, especially in real-time scenarios. This highlights the need for a versatile compute-memory efficient accelerator. Unfortunately, existing Transformer accelerators struggle to address both aspects simultaneously, as they focus on value-level processing, missing fine-grained opportunities to optimize computation and memory collaboratively. This paper introduces MCBP, a bit-grained compute-memory efficient algorithm-hardware co-design that leverages bit-slice (BS) enabled repetitiveness and sparsity to accelerate LLM inference. MCBP features three key innovations: 1) BS-repetitiveness-enabled computation reduction (BRCR), which eliminates redundant GEMM computations via leveraging redundancy hidden among BS vectors; 2) BS-sparsity-enabled two-state coding (BSTC), which reduces weight access via exploiting significant sparsity in high-order bit-slice weight; 3) Bit-grained progressive prediction (BGPP), which reduces KV cache access by leveraging early-termination-based bit-grained prediction. These techniques, supported by custom accelerator designs, effectively alleviate the burden in GEMM, weight access, and KV cache access. Extensive experiments on 26 benchmarks show that MCBP achieves 9.43x speed up and 31.1x higher energy efficiency than Nvidia A100 GPU. Compared to SOTA Transformer accelerators, MCBP achieves 35x, 5.2x and 3.2x energy saving than Spatten, FACT and SOFA, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18173v3">UIO-LLMs: Unbiased Incremental Optimization for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 This article was not accepted, and its quality is not very good. Therefore, we have decided to withdraw the submission and will not resubmit it elsewhere
    </div>
    <details class="paper-abstract">
      Managing long texts is challenging for large language models (LLMs) due to limited context window sizes. This study introduces UIO-LLMs, an unbiased incremental optimization approach for memory-enhanced transformers under long-context settings. We initially conceptualize the process as a streamlined encoder-decoder framework where the weights-shared encoder and decoder respectively encapsulate a context segment into memories and leverage these memories to predict outputs of the subsequent segment. Subsequently, by treating our memory-enhanced transformers as fully-connected recurrent neural networks (RNNs), we refine the training process using the Truncated Backpropagation Through Time (TBPTT) algorithm, which incorporates innovative incremental optimization techniques. These techniques not only diminish time complexity but also address the bias in gradient computation through an unbiased optimization process. UIO-LLMs successfully handle long context, such as extending the context window of Llama2-7b-chat from 4K to 100K tokens with minimal 2% additional parameters, while keeping the inference cost nearly linear as context length increases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10297v1">The Morality of Probability: How Implicit Moral Biases in LLMs May Shape the Future of Human-AI Symbiosis</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Artificial intelligence (AI) is advancing at a pace that raises urgent questions about how to align machine decision-making with human moral values. This working paper investigates how leading AI systems prioritize moral outcomes and what this reveals about the prospects for human-AI symbiosis. We address two central questions: (1) What moral values do state-of-the-art large language models (LLMs) implicitly favour when confronted with dilemmas? (2) How do differences in model architecture, cultural origin, and explainability affect these moral preferences? To explore these questions, we conduct a quantitative experiment with six LLMs, ranking and scoring outcomes across 18 dilemmas representing five moral frameworks. Our findings uncover strikingly consistent value biases. Across all models, Care and Virtue values outcomes were rated most moral, while libertarian choices were consistently penalized. Reasoning-enabled models exhibited greater sensitivity to context and provided richer explanations, whereas non-reasoning models produced more uniform but opaque judgments. This research makes three contributions: (i) Empirically, it delivers a large-scale comparison of moral reasoning across culturally distinct LLMs; (ii) Theoretically, it links probabilistic model behaviour with underlying value encodings; (iii) Practically, it highlights the need for explainability and cultural awareness as critical design principles to guide AI toward a transparent, aligned, and symbiotic future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13335v2">Comparing Apples to Oranges: A Dataset & Analysis of LLM Humour Understanding from Traditional Puns to Topical Jokes</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 Accepted to Findings of EMNLP 2025
    </div>
    <details class="paper-abstract">
      Humour, as a complex language form, is derived from myriad aspects of life. Whilst existing work on computational humour has focussed almost exclusively on short pun-based jokes, we investigate whether the ability of Large Language Models (LLMs) to explain humour depends on the particular form. We compare models' joke explanation abilities from simple puns to complex topical humour that requires esoteric knowledge of real-world entities and events. To this end, we curate a dataset of 600 jokes across 4 joke types and manually write high-quality explanations. These jokes include heterographic and homographic puns, contemporary internet humour, and topical jokes. Using this dataset, we compare the zero-shot abilities of a range of LLMs to accurately and comprehensively explain jokes of different types, identifying key research gaps in the task of humour explanation. We find that none of the tested models (including reasoning models) are capable of reliably generating adequate explanations of all joke types, further highlighting the narrow focus of most existing works on overly simple joke forms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20600v2">Multi-Turn Human-LLM Interaction Through the Lens of a Two-Way Intelligibility Protocol</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Our interest is in the design of software systems involving a human-expert interacting -- using natural language -- with a large language model (LLM) on data analysis tasks. For complex problems, it is possible that LLMs can harness human expertise and creativity to find solutions that were otherwise elusive. On one level, this interaction takes place through multiple turns of prompts from the human and responses from the LLM. Here we investigate a more structured approach based on an abstract protocol described in [3] for interaction between agents. The protocol is motivated by a notion of "two-way intelligibility" and is modelled by a pair of communicating finite-state machines. We provide an implementation of the protocol, and provide empirical evidence of using the implementation to mediate interactions between an LLM and a human-agent in two areas of scientific interest (radiology and drug design). We conduct controlled experiments with a human proxy (a database), and uncontrolled experiments with human subjects. The results provide evidence in support of the protocol's capability of capturing one- and two-way intelligibility in human-LLM interaction; and for the utility of two-way intelligibility in the design of human-machine systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10248v1">Prompt Injection Attacks on LLM Generated Reviews of Scientific Publications</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      The ongoing intense discussion on rising LLM usage in the scientific peer-review process has recently been mingled by reports of authors using hidden prompt injections to manipulate review scores. Since the existence of such "attacks" - although seen by some commentators as "self-defense" - would have a great impact on the further debate, this paper investigates the practicability and technical success of the described manipulations. Our systematic evaluation uses 1k reviews of 2024 ICLR papers generated by a wide range of LLMs shows two distinct results: I) very simple prompt injections are indeed highly effective, reaching up to 100% acceptance scores. II) LLM reviews are generally biased toward acceptance (>95% in many models). Both results have great impact on the ongoing discussions on LLM usage in peer-review.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10179v1">Benchmark of stylistic variation in LLM-generated texts</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      This study investigates the register variation in texts written by humans and comparable texts produced by large language models (LLMs). Biber's multidimensional analysis (MDA) is applied to a sample of human-written texts and AI-created texts generated to be their counterparts to find the dimensions of variation in which LLMs differ most significantly and most systematically from humans. As textual material, a new LLM-generated corpus AI-Brown is used, which is comparable to BE-21 (a Brown family corpus representing contemporary British English). Since all languages except English are underrepresented in the training data of frontier LLMs, similar analysis is replicated on Czech using AI-Koditex corpus and Czech multidimensional model. Examined were 16 frontier models in various settings and prompts, with emphasis placed on the difference between base models and instruction-tuned models. Based on this, a benchmark is created through which models can be compared with each other and ranked in interpretable dimensions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.15546v3">A Framework for Testing and Adapting REST APIs as LLM Tools</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used to build autonomous agents that perform complex tasks with external tools, often exposed through APIs in enterprise systems. Direct use of these APIs is difficult due to the complex input schema and verbose responses. Current benchmarks overlook these challenges, leaving a gap in assessing API readiness for agent-driven automation. We present a testing framework that systematically evaluates enterprise APIs when wrapped as Python tools for LLM-based agents. The framework generates data-aware test cases, translates them into natural language instructions, and evaluates whether agents can correctly invoke the tool, handle their inputs, and process its responses. We apply the framework to generate over 2400 test cases across different domains and develop a taxonomy of common errors, including input misinterpretation, output failures, and schema mismatches. We further classify errors to support debugging and tool refinement. Our framework provides a systematic approach to enabling enterprise APIs as reliable tools for agent-based applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.05367v2">Between a Rock and a Hard Place: Exploiting Ethical Reasoning to Jailbreak LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have undergone safety alignment efforts to mitigate harmful outputs. However, as LLMs become more sophisticated in reasoning, their intelligence may introduce new security risks. While traditional jailbreak attacks relied on singlestep attacks, multi-turn jailbreak strategies that adapt dynamically to context remain underexplored. In this work, we introduce TRIAL (Trolley-problem Reasoning for Interactive Attack Logic), a framework that leverages LLMs ethical reasoning to bypass their safeguards. TRIAL embeds adversarial goals within ethical dilemmas modeled on the trolley problem. TRIAL demonstrates high jailbreak success rates towards both open and close-source models. Our findings underscore a fundamental limitation in AI safety: as models gain advanced reasoning abilities, the nature of their alignment may inadvertently allow for more covert security vulnerabilities to be exploited. TRIAL raises an urgent need in reevaluating safety alignment oversight strategies, as current safeguards may prove insufficient against context-aware adversarial attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10127v1">Population-Aligned Persona Generation for LLM-based Social Simulation</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled human-like social simulations at unprecedented scale and fidelity, offering new opportunities for computational social science. A key challenge, however, is the construction of persona sets that authentically represent the diversity and distribution of real-world populations. Most existing LLM-based social simulation studies focus primarily on designing agentic frameworks and simulation environments, often overlooking the complexities of persona generation and the potential biases introduced by unrepresentative persona sets. In this paper, we propose a systematic framework for synthesizing high-quality, population-aligned persona sets for LLM-driven social simulation. Our approach begins by leveraging LLMs to generate narrative personas from long-term social media data, followed by rigorous quality assessment to filter out low-fidelity profiles. We then apply importance sampling to achieve global alignment with reference psychometric distributions, such as the Big Five personality traits. To address the needs of specific simulation contexts, we further introduce a task-specific module that adapts the globally aligned persona set to targeted subpopulations. Extensive experiments demonstrate that our method significantly reduces population-level bias and enables accurate, flexible social simulation for a wide range of research and policy applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08863v2">GeoJSON Agents:A Multi-Agent LLM Architecture for Geospatial Analysis-Function Calling vs Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      LLMs have made substantial progress in task automation and natural language understanding. However, without expertise in GIS, they continue to encounter limitations. To address these issues, we propose GeoJSON Agents-a multi-agent LLM architecture. This framework transforms natural language tasks into structured GeoJSON operation commands and processes spatial data using two widely adopted LLM enhancement techniques: Function Calling and Code Generation. The architecture consists of three components-task parsing, agent collaboration, and result integration-aimed at enhancing both the performance and scalability of GIS automation. The Planner agent interprets natural language tasks into structured GeoJSON commands. Then, specialized Worker agents collaborate according to assigned roles to perform spatial data processing and analysis, either by invoking predefined function APIs or by dynamically generating and executing Python-based spatial analysis code. Finally, the system integrates the outputs from multiple execution rounds into reusable, standards-compliant GeoJSON files. To systematically evaluate the performance of the two approaches, we constructed a benchmark dataset of 70 tasks with varying complexity and conducted experiments using OpenAI's GPT-4o as the core model. Results indicate that the Function Calling-based GeoJSON Agent achieved an accuracy of 85.71%, while the Code Generation-based agent reached 97.14%, both significantly outperforming the best-performing general-purpose model (48.57%). Further analysis reveals that the Code Generation provides greater flexibility, whereas the Function Calling approach offers more stable execution. This study is the first to introduce an LLM multi-agent framework for GeoJSON data and to compare the strengths and limitations of two mainstream LLM enhancement methods, offering new perspectives for improving GeoAI system performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10010v1">Multi-Intent Recognition in Dialogue Understanding: A Comparison Between Smaller Open-Source LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      In this paper, we provide an extensive analysis of multi-label intent classification using Large Language Models (LLMs) that are open-source, publicly available, and can be run in consumer hardware. We use the MultiWOZ 2.1 dataset, a benchmark in the dialogue system domain, to investigate the efficacy of three popular open-source pre-trained LLMs, namely LLama2-7B-hf, Mistral-7B-v0.1, and Yi-6B. We perform the classification task in a few-shot setup, giving 20 examples in the prompt with some instructions. Our approach focuses on the differences in performance of these models across several performance metrics by methodically assessing these models on multi-label intent classification tasks. Additionally, we compare the performance of the instruction-based fine-tuning approach with supervised learning using the smaller transformer model BertForSequenceClassification as a baseline. To evaluate the performance of the models, we use evaluation metrics like accuracy, precision, and recall as well as micro, macro, and weighted F1 score. We also report the inference time, VRAM requirements, etc. The Mistral-7B-v0.1 outperforms two other generative models on 11 intent classes out of 14 in terms of F-Score, with a weighted average of 0.50. It also has relatively lower Humming Loss and higher Jaccard Similarity, making it the winning model in the few-shot setting. We find BERT based supervised classifier having superior performance compared to the best performing few-shot generative LLM. The study provides a framework for small open-source LLMs in detecting complex multi-intent dialogues, enhancing the Natural Language Understanding aspect of task-oriented chatbots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09995v1">QuantAgent: Price-Driven Multi-Agent LLMs for High-Frequency Trading</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have demonstrated impressive capabilities in financial reasoning and market understanding. Multi-agent LLM frameworks such as TradingAgent and FINMEM augment these models to long-horizon investment tasks, leveraging fundamental and sentiment-based inputs for strategic decision-making. However, such systems are ill-suited for the high-speed, precision-critical demands of High-Frequency Trading (HFT). HFT requires rapid, risk-aware decisions based on structured, short-horizon signals, including technical indicators, chart patterns, and trend-based features, distinct from the long-term semantic reasoning typical of traditional financial LLM applications. To this end, we introduce QuantAgent, the first multi-agent LLM framework explicitly designed for high-frequency algorithmic trading. The system decomposes trading into four specialized agents, Indicator, Pattern, Trend, and Risk, each equipped with domain-specific tools and structured reasoning capabilities to capture distinct aspects of market dynamics over short temporal windows. In zero-shot evaluations across ten financial instruments, including Bitcoin and Nasdaq futures, QuantAgent demonstrates superior performance in both predictive accuracy and cumulative return over 4-hour trading intervals, outperforming strong neural and rule-based baselines. Our findings suggest that combining structured financial priors with language-native reasoning unlocks new potential for traceable, real-time decision systems in high-frequency financial markets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09970v1">Securing LLM-Generated Embedded Firmware through AI Agent-Driven Validation and Patching</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show promise in generating firmware for embedded systems, but often introduce security flaws and fail to meet real-time performance constraints. This paper proposes a three-phase methodology that combines LLM-based firmware generation with automated security validation and iterative refinement in a virtualized environment. Using structured prompts, models like GPT-4 generate firmware for networking and control tasks, deployed on FreeRTOS via QEMU. These implementations are tested using fuzzing, static analysis, and runtime monitoring to detect vulnerabilities such as buffer overflows (CWE-120), race conditions (CWE-362), and denial-of-service threats (CWE-400). Specialized AI agents for Threat Detection, Performance Optimization, and Compliance Verification collaborate to improve detection and remediation. Identified issues are categorized using CWE, then used to prompt targeted LLM-generated patches in an iterative loop. Experiments show a 92.4\% Vulnerability Remediation Rate (37.3\% improvement), 95.8\% Threat Model Compliance, and 0.87 Security Coverage Index. Real-time metrics include 8.6ms worst-case execution time and 195{\mu}s jitter. This process enhances firmware security and performance while contributing an open-source dataset for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11007v3">Local-Cloud Inference Offloading for LLMs in Multi-Modal, Multi-Task, Multi-Dialogue Settings</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Compared to traditional machine learning models, recent large language models (LLMs) can exhibit multi-task-solving capabilities through multiple dialogues and multi-modal data sources. These unique characteristics of LLMs, together with their large model size, make their deployment more challenging. Specifically, (i) deploying LLMs on local devices faces computational, memory, and energy resource issues, while (ii) deploying them in the cloud cannot guarantee real-time service and incurs communication/usage costs. In this paper, we design TMO, a local-cloud LLM inference system with Three-M Offloading: Multi-modal, Multi-task, and Multi-dialogue. TMO incorporates (i) a lightweight local LLM that can process simple tasks at high speed and (ii) a large-scale cloud LLM that can handle multi-modal data sources. We develop a resource-constrained reinforcement learning (RCRL) strategy for TMO that optimizes the inference location (i.e., local vs. cloud) and multi-modal data sources to use for each task/dialogue, aiming to maximize the long-term reward (response quality, latency, and usage cost) while adhering to resource constraints. We also contribute M4A1, a new dataset we curated that contains reward and cost metrics across multiple modality, task, dialogue, and LLM configurations, enabling evaluation of offloading decisions. We demonstrate the effectiveness of TMO compared to several exploration-decision and LLM-as-Agent baselines, showing significant improvements in latency, cost, and response quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.03703v2">Privacy Risks of LLM-Empowered Recommender Systems: An Inversion Attack Perspective</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 Accepted at ACM RecSys 2025 (10 pages, 4 figures)
    </div>
    <details class="paper-abstract">
      The large language model (LLM) powered recommendation paradigm has been proposed to address the limitations of traditional recommender systems, which often struggle to handle cold start users or items with new IDs. Despite its effectiveness, this study uncovers that LLM empowered recommender systems are vulnerable to reconstruction attacks that can expose both system and user privacy. To examine this threat, we present the first systematic study on inversion attacks targeting LLM empowered recommender systems, where adversaries attempt to reconstruct original prompts that contain personal preferences, interaction histories, and demographic attributes by exploiting the output logits of recommendation models. We reproduce the vec2text framework and optimize it using our proposed method called Similarity Guided Refinement, enabling more accurate reconstruction of textual prompts from model generated logits. Extensive experiments across two domains (movies and books) and two representative LLM based recommendation models demonstrate that our method achieves high fidelity reconstructions. Specifically, we can recover nearly 65 percent of the user interacted items and correctly infer age and gender in 87 percent of the cases. The experiments also reveal that privacy leakage is largely insensitive to the victim model's performance but highly dependent on domain consistency and prompt complexity. These findings expose critical privacy vulnerabilities in LLM empowered recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09928v1">Fraud detection and risk assessment of online payment transactions on e-commerce platforms based on LLM and GCN frameworks</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      With the rapid growth of e-commerce, online payment fraud has become increasingly complex, posing serious threats to financial security and consumer trust. Traditional detection methods often struggle to capture the intricate relational structures inherent in transactional data. This study presents a novel fraud detection framework that combines Large Language Models (LLM) with Graph Convolutional Networks (GCN) to effectively identify fraudulent activities in e-commerce online payment transactions. A dataset of 2,840,000 transactions was collected over 14 days from major platforms such as Amazon, involving approximately 2,000 U.S.-based consumers and 30 merchants. With fewer than 6000 fraudulent instances, the dataset represents a highly imbalanced scenario. Consumers and merchants were modeled as nodes and transactions as edges to form a heterogeneous graph, upon which a GCN was applied to learn complex behavioral patterns. Semantic features extracted via GPT-4o and Tabformer were integrated with structural features to enhance detection performance. Experimental results demonstrate that the proposed model achieves an accuracy of 0.98, effectively balancing precision and sensitivity in fraud detection. This framework offers a scalable and real-time solution for securing online payment environments and provides a promising direction for applying graph-based deep learning in financial fraud prevention.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19634v4">Faster and Better LLMs via Latency-Aware Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Test-Time Scaling (TTS) has proven effective in improving the performance of Large Language Models (LLMs) during inference. However, existing research has overlooked the efficiency of TTS from a latency-sensitive perspective. Through a latency-aware evaluation of representative TTS methods, we demonstrate that a compute-optimal TTS does not always result in the lowest latency in scenarios where latency is critical. To address this gap and achieve latency-optimal TTS, we propose two key approaches by optimizing the concurrency configurations: (1) branch-wise parallelism, which leverages multiple concurrent inference branches, and (2) sequence-wise parallelism, enabled by speculative decoding. By integrating these two approaches and allocating computational resources properly to each, our latency-optimal TTS enables a 32B model to reach 82.3% accuracy on MATH-500 within 1 minute and a smaller 3B model to achieve 72.4% within 10 seconds. Our work emphasizes the importance of latency-aware TTS and demonstrates its ability to deliver both speed and accuracy in latency-sensitive scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09917v1">SLD-Spec: Enhancement LLM-assisted Specification Generation for Complex Loop Functions via Program Slicing and Logical Deletion</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 22 pages, 2 figures, conference
    </div>
    <details class="paper-abstract">
      Automatically generating formal specifications from program code can greatly enhance the efficiency of program verification and enable end-to-end automation from requirements to reliable software. However, existing LLM-based approaches often struggle with programs that include complex loop structures, leading to irrelevant specifications. Moreover, the rigorous proof obligations and design constraints imposed by verification tools can further result in incomplete and ambiguous specifications. To address these challenges, we propose SLD-Spec, an LLM-assisted specification generation method tailored for programs with complex loop constructs. SLD-Spec introduces two novel phases into the traditional specification generation framework: (1) A slicing phase, which decomposes each function into code fragments containing independent loop structures, thereby reducing the complexity of specification generation; and (2) A logical deletion phase, which applies LLM-based reasoning to filter out incorrect candidate specifications--especially those not easily identified by verification tool--while retaining valid ones. Experimental results show that on the simple dataset, SLD-Spec successfully verifies five more programs than the state-of-the-art AutoSpec and reduces runtime by 23.73%. To address the limitations of existing research, we manually construct a dataset comprising four categories of complex loop programs. On this dataset, SLD-Spec significantly improves the correctness, relevance, and completeness of generated specifications compared to baseline methods, enabling 95.1% of assertions and 90.91% of programs to pass verification. Ablation studies further reveal that logical deletion is critical for enhancing specification correctness and relevance, while program slicing contributes significantly to specification completeness. Our code and data are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09912v1">When Your Reviewer is an LLM: Biases, Divergence, and Prompt Injection Risks in Peer Review</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Peer review is the cornerstone of academic publishing, yet the process is increasingly strained by rising submission volumes, reviewer overload, and expertise mismatches. Large language models (LLMs) are now being used as "reviewer aids," raising concerns about their fairness, consistency, and robustness against indirect prompt injection attacks. This paper presents a systematic evaluation of LLMs as academic reviewers. Using a curated dataset of 1,441 papers from ICLR 2023 and NeurIPS 2022, we evaluate GPT-5-mini against human reviewers across ratings, strengths, and weaknesses. The evaluation employs structured prompting with reference paper calibration, topic modeling, and similarity analysis to compare review content. We further embed covert instructions into PDF submissions to assess LLMs' susceptibility to prompt injection. Our findings show that LLMs consistently inflate ratings for weaker papers while aligning more closely with human judgments on stronger contributions. Moreover, while overarching malicious prompts induce only minor shifts in topical focus, explicitly field-specific instructions successfully manipulate specific aspects of LLM-generated reviews. This study underscores both the promises and perils of integrating LLMs into peer review and points to the importance of designing safeguards that ensure integrity and trust in future review processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10753v1">HalluField: Detecting LLM Hallucinations via Field-Theoretic Modeling</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) exhibit impressive reasoning and question-answering capabilities. However, they often produce inaccurate or unreliable content known as hallucinations. This unreliability significantly limits their deployment in high-stakes applications. Thus, there is a growing need for a general-purpose method to detect hallucinations in LLMs. In this work, we introduce HalluField, a novel field-theoretic approach for hallucination detection based on a parametrized variational principle and thermodynamics. Inspired by thermodynamics, HalluField models an LLM's response to a given query and temperature setting as a collection of discrete likelihood token paths, each associated with a corresponding energy and entropy. By analyzing how energy and entropy distributions vary across token paths under changes in temperature and likelihood, HalluField quantifies the semantic stability of a response. Hallucinations are then detected by identifying unstable or erratic behavior in this energy landscape. HalluField is computationally efficient and highly practical: it operates directly on the model's output logits without requiring fine-tuning or auxiliary neural networks. Notably, the method is grounded in a principled physical interpretation, drawing analogies to the first law of thermodynamics. Remarkably, by modeling LLM behavior through this physical lens, HalluField achieves state-of-the-art hallucination detection performance across models and datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10739v1">Reasoning Under Uncertainty: Exploring Probabilistic Reasoning Capabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 25 pages, 4 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Despite widespread success in language understanding and generation, large language models (LLMs) exhibit unclear and often inconsistent behavior when faced with tasks that require probabilistic reasoning. In this work, we present the first comprehensive study of the reasoning capabilities of LLMs over explicit discrete probability distributions. Given observations from a probability distribution, we evaluate models on three carefully designed tasks, mode identification, maximum likelihood estimation, and sample generation, by prompting them to provide responses to queries about either the joint distribution or its conditionals. These tasks thus probe a range of probabilistic skills, including frequency analysis, marginalization, and generative behavior. Through comprehensive empirical evaluations, we demonstrate that there exists a clear performance gap between smaller and larger models, with the latter demonstrating stronger inference and surprising capabilities in sample generation. Furthermore, our investigations reveal notable limitations, including sensitivity to variations in the notation utilized to represent probabilistic outcomes and performance degradation of over 60% as context length increases. Together, our results provide a detailed understanding of the probabilistic reasoning abilities of LLMs and identify key directions for future improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10729v1">Using LLMs for Late Multimodal Sensor Fusion for Activity Recognition</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 Preprint, under review
    </div>
    <details class="paper-abstract">
      Sensor data streams provide valuable information around activities and context for downstream applications, though integrating complementary information can be challenging. We show that large language models (LLMs) can be used for late fusion for activity classification from audio and motion time series data. We curated a subset of data for diverse activity recognition across contexts (e.g., household activities, sports) from the Ego4D dataset. Evaluated LLMs achieved 12-class zero- and one-shot classification F1-scores significantly above chance, with no task-specific training. Zero-shot classification via LLM-based fusion from modality-specific models can enable multimodal temporal applications where there is limited aligned training data for learning a shared embedding space. Additionally, LLM-based fusion can enable model deploying without requiring additional memory and computation for targeted application-specific multimodal models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10723v1">Dark Patterns Meet GUI Agents: LLM Agent Susceptibility to Manipulative Interfaces and the Role of Human Oversight</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      The dark patterns, deceptive interface designs manipulating user behaviors, have been extensively studied for their effects on human decision-making and autonomy. Yet, with the rising prominence of LLM-powered GUI agents that automate tasks from high-level intents, understanding how dark patterns affect agents is increasingly important. We present a two-phase empirical study examining how agents, human participants, and human-AI teams respond to 16 types of dark patterns across diverse scenarios. Phase 1 highlights that agents often fail to recognize dark patterns, and even when aware, prioritize task completion over protective action. Phase 2 revealed divergent failure modes: humans succumb due to cognitive shortcuts and habitual compliance, while agents falter from procedural blind spots. Human oversight improved avoidance but introduced costs such as attentional tunneling and cognitive load. Our findings show neither humans nor agents are uniformly resilient, and collaboration introduces new vulnerabilities, suggesting design needs for transparency, adjustable autonomy, and oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10698v1">CrunchLLM: Multitask LLMs for Structured Business Reasoning and Outcome Prediction</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Predicting the success of start-up companies, defined as achieving an exit through acquisition or IPO, is a critical problem in entrepreneurship and innovation research. Datasets such as Crunchbase provide both structured information (e.g., funding rounds, industries, investor networks) and unstructured text (e.g., company descriptions), but effectively leveraging this heterogeneous data for prediction remains challenging. Traditional machine learning approaches often rely only on structured features and achieve moderate accuracy, while large language models (LLMs) offer rich reasoning abilities but struggle to adapt directly to domain-specific business data. We present \textbf{CrunchLLM}, a domain-adapted LLM framework for startup success prediction. CrunchLLM integrates structured company attributes with unstructured textual narratives and applies parameter-efficient fine-tuning strategies alongside prompt optimization to specialize foundation models for entrepreneurship data. Our approach achieves accuracy exceeding 80\% on Crunchbase startup success prediction, significantly outperforming traditional classifiers and baseline LLMs. Beyond predictive performance, CrunchLLM provides interpretable reasoning traces that justify its predictions, enhancing transparency and trustworthiness for financial and policy decision makers. This work demonstrates how adapting LLMs with domain-aware fine-tuning and structured--unstructured data fusion can advance predictive modeling of entrepreneurial outcomes. CrunchLLM contributes a methodological framework and a practical tool for data-driven decision making in venture capital and innovation policy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10682v1">LLM in the Middle: A Systematic Review of Threats and Mitigations to Real-World LLM-based Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 37 pages, 8 figures, 13 tables
    </div>
    <details class="paper-abstract">
      The success and wide adoption of generative AI (GenAI), particularly large language models (LLMs), has attracted the attention of cybercriminals seeking to abuse models, steal sensitive data, or disrupt services. Moreover, providing security to LLM-based systems is a great challenge, as both traditional threats to software applications and threats targeting LLMs and their integration must be mitigated. In this survey, we shed light on security and privacy concerns of such LLM-based systems by performing a systematic review and comprehensive categorization of threats and defensive strategies considering the entire software and LLM life cycles. We analyze real-world scenarios with distinct characteristics of LLM usage, spanning from development to operation. In addition, threats are classified according to their severity level and to which scenarios they pertain, facilitating the identification of the most relevant threats. Recommended defense strategies are systematically categorized and mapped to the corresponding life cycle phase and possible attack strategies they attenuate. This work paves the way for consumers and vendors to understand and efficiently mitigate risks during integration of LLMs in their respective solutions or organizations. It also enables the research community to benefit from the discussion of open challenges and edge cases that may hinder the secure and privacy-preserving adoption of LLM-based systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.21587v2">A Cross-Cultural Comparison of LLM-based Public Opinion Simulation: Evaluating Chinese and U.S. Models on Diverse Societies</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      This study evaluates the ability of DeepSeek, an open-source large language model (LLM), to simulate public opinions in comparison to LLMs developed by major tech companies. By comparing DeepSeek-R1 and DeepSeek-V3 with Qwen2.5, GPT-4o, and Llama-3.3 and utilizing survey data from the American National Election Studies (ANES) and the Zuobiao dataset of China, we assess these models' capacity to predict public opinions on social issues in both China and the United States, highlighting their comparative capabilities between countries. Our findings indicate that DeepSeek-V3 performs best in simulating U.S. opinions on the abortion issue compared to other topics such as climate change, gun control, immigration, and services for same-sex couples, primarily because it more accurately simulates responses when provided with Democratic or liberal personas. For Chinese samples, DeepSeek-V3 performs best in simulating opinions on foreign aid and individualism but shows limitations in modeling views on capitalism, particularly failing to capture the stances of low-income and non-college-educated individuals. It does not exhibit significant differences from other models in simulating opinions on traditionalism and the free market. Further analysis reveals that all LLMs exhibit the tendency to overgeneralize a single perspective within demographic groups, often defaulting to consistent responses within groups. These findings highlight the need to mitigate cultural and demographic biases in LLM-driven public opinion modeling, calling for approaches such as more inclusive training methodologies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12110v2">Towards LLM Agents for Earth Observation</a></div>
    <div class="paper-meta">
      📅 2025-09-12
      | 💬 Accepted at ICML 2025 Workshop TerraBytes
    </div>
    <details class="paper-abstract">
      Earth Observation (EO) provides critical planetary data for environmental monitoring, disaster management, climate science, and other scientific domains. Here we ask: Are AI systems ready for reliable Earth Observation? We introduce \datasetnamenospace, a benchmark of 140 yes/no questions from NASA Earth Observatory articles across 13 topics and 17 satellite sensors. Using Google Earth Engine API as a tool, LLM agents can only achieve an accuracy of 33% because the code fails to run over 58% of the time. We improve the failure rate for open models by fine-tuning synthetic data, allowing much smaller models (Llama-3.1-8B) to achieve comparable accuracy to much larger ones (e.g., DeepSeek-R1). Taken together, our findings identify significant challenges to be solved before AI agents can automate earth observation, and suggest paths forward. The project page is available at https://iandrover.github.io/UnivEarth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10637v1">LLMs Homogenize Values in Constructive Arguments on Value-Laden Topics</a></div>
    <div class="paper-meta">
      📅 2025-09-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to promote prosocial and constructive discourse online. Yet little is known about how they negotiate and shape underlying values when reframing people's arguments on value-laden topics. We conducted experiments with 347 participants from India and the United States, who wrote constructive comments on homophobic and Islamophobic threads, and reviewed human-written and LLM-rewritten versions of these comments. Our analysis shows that LLM systematically diminishes Conservative values while elevating prosocial values such as Benevolence and Universalism. When these comments were read by others, participants opposing same-sex marriage or Islam found human-written comments more aligned with their values, whereas those supportive of these communities found LLM-rewritten versions more aligned with their values. These findings suggest that LLM-driven value homogenization can shape how diverse viewpoints are represented in contentious debates on value-laden topics and may influence the dynamics of online discourse critically.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09234v1">Agentic LLMs for Question Answering over Tabular Data</a></div>
    <div class="paper-meta">
      📅 2025-09-11
      | 💬 Accepted at ACL workshop SemEval 2025
    </div>
    <details class="paper-abstract">
      Question Answering over Tabular Data (Table QA) presents unique challenges due to the diverse structure, size, and data types of real-world tables. The SemEval 2025 Task 8 (DataBench) introduced a benchmark composed of large-scale, domain-diverse datasets to evaluate the ability of models to accurately answer structured queries. We propose a Natural Language to SQL (NL-to-SQL) approach leveraging large language models (LLMs) such as GPT-4o, GPT-4o-mini, and DeepSeek v2:16b to generate SQL queries dynamically. Our system follows a multi-stage pipeline involving example selection, SQL query generation, answer extraction, verification, and iterative refinement. Experiments demonstrate the effectiveness of our approach, achieving 70.5\% accuracy on DataBench QA and 71.6\% on DataBench Lite QA, significantly surpassing baseline scores of 26\% and 27\% respectively. This paper details our methodology, experimental results, and alternative approaches, providing insights into the strengths and limitations of LLM-driven Table QA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09226v1">Constructing a Question-Answering Simulator through the Distillation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      The question-answering (QA) simulator is a model that mimics real student learning behaviors and predicts their correctness of their responses to questions. QA simulators enable educational recommender systems (ERS) to collect large amounts of training data without interacting with real students, thereby preventing harmful recommendations made by an undertrained ERS from undermining actual student learning. Given the QA history, there are two categories of solutions to predict the correctness, conducting the simulation: (1) LLM-free methods, which apply a traditional sequential model to transfer the QA history into a vector representation first, and make predictions based on the representation; (2) LLM-based methods, which leverage the domain knowledge and reasoning capability of LLM to enhence the prediction. LLM-free methods offer fast inference but generally yield suboptimal performance. In contrast, most LLM-based methods achieve better results, but at the cost of slower inference speed and higher GPU memory consumption. In this paper, we propose a method named LLM Distillation based Simulator (LDSim), which distills domain knowledge and reasoning capability from an LLM to better assist prediction, thereby improving simulation performance. Extensive experiments demonstrate that our LDSim achieves strong results on both the simulation task and the knowledge tracing (KT) task. Our code is publicly available at https://anonymous.4open.science/r/LDSim-05A9.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19740v3">Spotlight Attention: Towards Efficient LLM Generation via Non-linear Hashing-based KV Cache Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Reducing the key-value (KV) cache burden in Large Language Models (LLMs) significantly accelerates inference. Dynamically selecting critical KV caches during decoding helps maintain performance. Existing methods use random linear hashing to identify important tokens, but this approach is inefficient due to the orthogonal distribution of queries and keys within two narrow cones in LLMs. We introduce Spotlight Attention, a novel method that employs non-linear hashing functions to optimize the embedding distribution of queries and keys, enhancing coding efficiency and robustness. We also developed a lightweight, stable training framework using a Bradley-Terry ranking-based loss, enabling optimization of the non-linear hashing module on GPUs with 16GB memory in 8 hours. Experimental results show that Spotlight Attention drastically improves retrieval precision while shortening the length of the hash code at least 5$\times$ compared to traditional linear hashing. Finally, we exploit the computational advantages of bitwise operations by implementing specialized CUDA kernels, achieving hashing retrieval for 512K tokens in under 100$\mu$s on a single A100 GPU, with end-to-end throughput up to 3$\times$ higher than vanilla decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09174v1">EchoX: Towards Mitigating Acoustic-Semantic Gap via Echo Training for Speech-to-Speech LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Speech-to-speech large language models (SLLMs) are attracting increasing attention. Derived from text-based large language models (LLMs), SLLMs often exhibit degradation in knowledge and reasoning capabilities. We hypothesize that this limitation arises because current training paradigms for SLLMs fail to bridge the acoustic-semantic gap in the feature representation space. To address this issue, we propose EchoX, which leverages semantic representations and dynamically generates speech training targets. This approach integrates both acoustic and semantic learning, enabling EchoX to preserve strong reasoning abilities as a speech LLM. Experimental results demonstrate that EchoX, with about six thousand hours of training data, achieves advanced performance on multiple knowledge-based question-answering benchmarks. The project is available at https://github.com/FreedomIntelligence/EchoX.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09121v1">Compass-v3: Scaling Domain-Specific LLMs for Multilingual E-Commerce in Southeast Asia</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel in general-domain applications, yet their performance often degrades in specialized tasks requiring domain-specific knowledge. E-commerce is particularly challenging, as its data are noisy, heterogeneous, multilingual, and highly dynamic. We present Compass-v3, a vertical-domain Mixture-of-Experts (MoE) model with 245B total parameters and 71B active per token, designed for Southeast Asian e-commerce. Compass-v3 adopts fewer but larger experts, combined with hardware-efficient optimizations-such as intra-node expert parallelism and a customized memcpy operator-to maximize GPU utilization. The model is trained on 12T tokens of curated multilingual corpora and large-scale synthetic e-commerce instructions using a mixed-training strategy. To enhance alignment, we propose Optimal-Transport Direct Preference Optimization (OTPO), which captures token-level distinctions and improves instruction adherence in commerce-specific scenarios. Extensive evaluations demonstrate that Compass-v3 delivers state-of-the-art e-commerce performance, surpassing DeepSeek-V3.1, GPT-4 series, and Qwen3-235B. Moreover, Compass-v3 demonstrates strong multilingual capability across low-resource Southeast Asian languages (Indonesian, Thai, Filipino, Vietnamese, Malay, Taglog) and Portuguese while sustaining competitive performance on general benchmarks. It has already been widely applied in Shopee's industrial-scale e-commerce platform and is gradually replacing OpenAI's traffic, now accounting for over 70\% of total LLM usage, highlighting its dual strengths in specialized commerce expertise and broad linguistic competence.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09112v1">Character-Level Perturbations Disrupt LLM Watermarks</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) watermarking embeds detectable signals into generated text for copyright protection, misuse prevention, and content detection. While prior studies evaluate robustness using watermark removal attacks, these methods are often suboptimal, creating the misconception that effective removal requires large perturbations or powerful adversaries. To bridge the gap, we first formalize the system model for LLM watermark, and characterize two realistic threat models constrained on limited access to the watermark detector. We then analyze how different types of perturbation vary in their attack range, i.e., the number of tokens they can affect with a single edit. We observe that character-level perturbations (e.g., typos, swaps, deletions, homoglyphs) can influence multiple tokens simultaneously by disrupting the tokenization process. We demonstrate that character-level perturbations are significantly more effective for watermark removal under the most restrictive threat model. We further propose guided removal attacks based on the Genetic Algorithm (GA) that uses a reference detector for optimization. Under a practical threat model with limited black-box queries to the watermark detector, our method demonstrates strong removal performance. Experiments confirm the superiority of character-level perturbations and the effectiveness of the GA in removing watermarks under realistic constraints. Additionally, we argue there is an adversarial dilemma when considering potential defenses: any fixed defense can be bypassed by a suitable perturbation strategy. Motivated by this principle, we propose an adaptive compound character-level attack. Experimental results show that this approach can effectively defeat the defenses. Our findings highlight significant vulnerabilities in existing LLM watermark schemes and underline the urgency for the development of new robust mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09103v1">AgriSentinel: Privacy-Enhanced Embedded-LLM Crop Disease Alerting System</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Crop diseases pose significant threats to global food security, agricultural productivity, and sustainable farming practices, directly affecting farmers' livelihoods and economic stability. To address the growing need for effective crop disease management, AI-based disease alerting systems have emerged as promising tools by providing early detection and actionable insights for timely intervention. However, existing systems often overlook critical aspects such as data privacy, market pricing power, and farmer-friendly usability, leaving farmers vulnerable to privacy breaches and economic exploitation. To bridge these gaps, we propose AgriSentinel, the first Privacy-Enhanced Embedded-LLM Crop Disease Alerting System. AgriSentinel incorporates a differential privacy mechanism to protect sensitive crop image data while maintaining classification accuracy. Its lightweight deep learning-based crop disease classification model is optimized for mobile devices, ensuring accessibility and usability for farmers. Additionally, the system includes a fine-tuned, on-device large language model (LLM) that leverages a curated knowledge pool to provide farmers with specific, actionable suggestions for managing crop diseases, going beyond simple alerting. Comprehensive experiments validate the effectiveness of AgriSentinel, demonstrating its ability to safeguard data privacy, maintain high classification performance, and deliver practical, actionable disease management strategies. AgriSentinel offers a robust, farmer-friendly solution for automating crop disease alerting and management, ultimately contributing to improved agricultural decision-making and enhanced crop productivity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09101v1">TigerCoder: A Novel Suite of LLMs for Code Generation in Bangla</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Despite being the 5th most spoken language, Bangla remains underrepresented in Large Language Models (LLMs), particularly for code generation. This primarily stems from the scarcity of high-quality data to pre-train and/or finetune such models. Hence, we introduce the first dedicated family of Code LLMs for Bangla (1B & 9B). We offer three major contributions: (1) a comprehensive Bangla code instruction datasets for programming domain adaptation; (2) MBPP-Bangla, an evaluation benchmark for Bangla code generation; and (3) the TigerCoder-family of Code LLMs, achieving significant ~11-18% performance gains at Pass@1 over existing multilingual and general-purpose Bangla LLMs. Our findings show that curated, high-quality datasets can overcome limitations of smaller models for low-resource languages. We open-source all resources to advance further Bangla LLM research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09091v1">Towards Confidential and Efficient LLM Inference with Dual Privacy Protection</a></div>
    <div class="paper-meta">
      📅 2025-09-11
      | 💬 Accepted by DASFAA2025
    </div>
    <details class="paper-abstract">
      CPU-based trusted execution environments (TEEs) and differential privacy (DP) have gained wide applications for private inference. Due to high inference latency in TEEs, researchers use partition-based approaches that offload linear model components to GPUs. However, dense nonlinear layers of large language models (LLMs) result in significant communication overhead between TEEs and GPUs. DP-based approaches apply random noise to protect data privacy, but this compromises LLM performance and semantic understanding. To overcome the above drawbacks, this paper proposes CMIF, a Confidential and efficient Model Inference Framework. CMIF confidentially deploys the embedding layer in the client-side TEE and subsequent layers on GPU servers. Meanwhile, it optimizes the Report-Noisy-Max mechanism to protect sensitive inputs with a slight decrease in model performance. Extensive experiments on Llama-series models demonstrate that CMIF reduces additional inference overhead in TEEs while preserving user data privacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09066v1">Instructional Prompt Optimization for Few-Shot LLM-Based Recommendations on Cold-Start Users</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      The cold-start user issue further compromises the effectiveness of recommender systems in limiting access to the historical behavioral information. It is an effective pipeline to optimize instructional prompts on a few-shot large language model (LLM) used in recommender tasks. We introduce a context-conditioned prompt formulation method P(u,\ Ds)\ \rightarrow\ R\widehat, where u is a cold-start user profile, Ds is a curated support set, and R\widehat is the predicted ranked list of items. Based on systematic experimentation with transformer-based autoregressive LLMs (BioGPT, LLaMA-2, GPT-4), we provide empirical evidence that optimal exemplar injection and instruction structuring can significantly improve the precision@k and NDCG scores of such models in low-data settings. The pipeline uses token-level alignments and embedding space regularization with a greater semantic fidelity. Our findings not only show that timely composition is not merely syntactic but also functional as it is in direct control of attention scales and decoder conduct through inference. This paper shows that prompt-based adaptation may be considered one of the ways to address cold-start recommendation issues in LLM-based pipelines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18383v2">NileChat: Towards Linguistically Diverse and Culturally Aware LLMs for Local Communities</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Enhancing the linguistic capabilities of Large Language Models (LLMs) to include low-resource languages is a critical research area. Current research directions predominantly rely on synthetic data generated by translating English corpora, which, while demonstrating promising linguistic understanding and translation abilities, often results in models aligned with source language culture. These models frequently fail to represent the cultural heritage and values of local communities. This work proposes a methodology to create both synthetic and retrieval-based pre-training data tailored to a specific community, considering its (i) language, (ii) cultural heritage, and (iii) cultural values. We demonstrate our methodology using Egyptian and Moroccan dialects as testbeds, chosen for their linguistic and cultural richness and current underrepresentation in LLMs. As a proof-of-concept, we develop NileChat, a 3B parameter LLM adapted for Egyptian and Moroccan communities, incorporating their language, cultural heritage, and values. Our results on various understanding, translation, and cultural and values alignment benchmarks show that NileChat outperforms existing Arabic-aware LLMs of similar size and performs on par with larger models. We share our methods, data, and models with the community to promote the inclusion and coverage of more diverse communities in LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09870v1">Vibe Check: Understanding the Effects of LLM-Based Conversational Agents' Personality and Alignment on User Perceptions in Goal-Oriented Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) enable conversational agents (CAs) to express distinctive personalities, raising new questions about how such designs shape user perceptions. This study investigates how personality expression levels and user-agent personality alignment influence perceptions in goal-oriented tasks. In a between-subjects experiment (N=150), participants completed travel planning with CAs exhibiting low, medium, or high expression across the Big Five traits, controlled via our novel Trait Modulation Keys framework. Results revealed an inverted-U relationship: medium expression produced the most positive evaluations across Intelligence, Enjoyment, Anthropomorphism, Intention to Adopt, Trust, and Likeability, significantly outperforming both extremes. Personality alignment further enhanced outcomes, with Extraversion and Emotional Stability emerging as the most influential traits. Cluster analysis identified three distinct compatibility profiles, with "Well-Aligned" users reporting substantially positive perceptions. These findings demonstrate that personality expression and strategic trait alignment constitute optimal design targets for CA personality, offering design implications as LLM-based CAs become increasingly prevalent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09867v1">LLMs as Agentic Cooperative Players in Multiplayer UNO</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      LLMs promise to assist humans -- not just by answering questions, but by offering useful guidance across a wide range of tasks. But how far does that assistance go? Can a large language model based agent actually help someone accomplish their goal as an active participant? We test this question by engaging an LLM in UNO, a turn-based card game, asking it not to win but instead help another player to do so. We built a tool that allows decoder-only LLMs to participate as agents within the RLCard game environment. These models receive full game-state information and respond using simple text prompts under two distinct prompting strategies. We evaluate models ranging from small (1B parameters) to large (70B parameters) and explore how model scale impacts performance. We find that while all models were able to successfully outperform a random baseline when playing UNO, few were able to significantly aid another player.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09790v1">How well can LLMs provide planning feedback in grounded environments?</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Learning to plan in grounded environments typically requires carefully designed reward functions or high-quality annotated demonstrations. Recent works show that pretrained foundation models, such as large language models (LLMs) and vision language models (VLMs), capture background knowledge helpful for planning, which reduces the amount of reward design and demonstrations needed for policy learning. We evaluate how well LLMs and VLMs provide feedback across symbolic, language, and continuous control environments. We consider prominent types of feedback for planning including binary feedback, preference feedback, action advising, goal advising, and delta action feedback. We also consider inference methods that impact feedback performance, including in-context learning, chain-of-thought, and access to environment dynamics. We find that foundation models can provide diverse high-quality feedback across domains. Moreover, larger and reasoning models consistently provide more accurate feedback, exhibit less bias, and benefit more from enhanced inference methods. Finally, feedback quality degrades for environments with complex dynamics or continuous state spaces and action spaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09782v1">One Head, Many Models: Cross-Attention Routing for Cost-Aware LLM Selection</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      The proliferation of large language models (LLMs) with varying computational costs and performance profiles presents a critical challenge for scalable, cost-effective deployment in real-world applications. We introduce a unified routing framework that leverages a single-head cross-attention mechanism to jointly model query and model embeddings, enabling dynamic selection of the optimal LLM for each input query. Our approach is evaluated on RouterBench, a large-scale, publicly available benchmark encompassing diverse LLM pools and domains. By explicitly capturing fine-grained query-model interactions, our router predicts both response quality and generation cost, achieving up to 6.6% improvement in Average Improvement in Quality (AIQ) and 2.9% in maximum performance over existing routers. To robustly balance performance and cost, we propose an exponential reward function that enhances stability across user preferences. The resulting architecture is lightweight, generalizes effectively across domains, and demonstrates improved efficiency compared to prior methods, establishing a new standard for cost-aware LLM routing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12039v3">Can LLM Prompting Serve as a Proxy for Static Analysis in Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Despite their remarkable success, large language models (LLMs) have shown limited ability on safety-critical code tasks such as vulnerability detection. Typically, static analysis (SA) tools, like CodeQL, CodeGuru Security, etc., are used for vulnerability detection. SA relies on predefined, manually-crafted rules for flagging various vulnerabilities. Thus, effectiveness of SA in detecting vulnerabilities depends on human experts and is known to report high error rates. In this study we investigate whether LLM prompting can be an effective alternative to these static analyzers in the partial code setting. We propose prompting strategies that integrate natural language instructions of vulnerabilities with contrastive chain-of-thought reasoning, augmented using contrastive samples from a synthetic dataset. Our findings demonstrate that security-aware prompting techniques can be effective alternatives to the laborious, hand-crafted rules of static analyzers, which often result in high false negative rates in the partial code setting. When leveraging SOTA reasoning models such as DeepSeek-R1, each of our prompting strategies exceeds the static analyzer baseline, with the best strategies improving accuracy by as much as 31.6%, F1-scores by 71.7%, pairwise accuracies by 60.4%, and reducing FNR by as much as 37.6%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.10575v1">Gene-R1: Reasoning with Data-Augmented Lightweight LLMs for Gene Set Analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-11
      | 💬 14 pages, 4 figures, 6 tables, 40 references
    </div>
    <details class="paper-abstract">
      The gene set analysis (GSA) is a foundational approach for uncovering the molecular functions associated with a group of genes. Recently, LLM-powered methods have emerged to annotate gene sets with biological functions together with coherent explanatory insights. However, existing studies primarily focus on proprietary models, which have been shown to outperform their open-source counterparts despite concerns over cost and data privacy. Furthermore, no research has investigated the application of advanced reasoning strategies to the GSA task. To address this gap, we introduce Gene-R1, a data-augmented learning framework that equips lightweight and open-source LLMs with step-by-step reasoning capabilities tailored to GSA. Experiments on 1,508 in-distribution gene sets demonstrate that Gene-R1 achieves substantial performance gains, matching commercial LLMs. On 106 out-of-distribution gene sets, Gene-R1 performs comparably to both commercial and large-scale LLMs, exhibiting robust generalizability across diverse gene sources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09677v1">The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Does continued scaling of large language models (LLMs) yield diminishing returns? Real-world value often stems from the length of task an agent can complete. We start this work by observing the simple but counterintuitive fact that marginal gains in single-step accuracy can compound into exponential improvements in the length of a task a model can successfully complete. Then, we argue that failures of LLMs when simple tasks are made longer arise from mistakes in execution, rather than an inability to reason. We propose isolating execution capability, by explicitly providing the knowledge and plan needed to solve a long-horizon task. We find that larger models can correctly execute significantly more turns even when small models have 100\% single-turn accuracy. We observe that the per-step accuracy of models degrades as the number of steps increases. This is not just due to long-context limitations -- curiously, we observe a self-conditioning effect -- models become more likely to make mistakes when the context contains their errors from prior turns. Self-conditioning does not reduce by just scaling the model size. In contrast, recent thinking models do not self-condition, and can also execute much longer tasks in a single turn. We conclude by benchmarking frontier thinking models on the length of task they can execute in a single turn. Overall, by focusing on the ability to execute, we hope to reconcile debates on how LLMs can solve complex reasoning problems yet fail at simple tasks when made longer, and highlight the massive benefits of scaling model size and sequential test-time compute for long-horizon tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09660v1">Steering MoE LLMs via Expert (De)Activation</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) in Large Language Models (LLMs) routes each token through a subset of specialized Feed-Forward Networks (FFN), known as experts. We present SteerMoE, a framework for steering MoE models by detecting and controlling behavior-linked experts. Our detection method identifies experts with distinct activation patterns across paired inputs exhibiting contrasting behaviors. By selectively (de)activating such experts during inference, we control behaviors like faithfulness and safety without retraining or modifying weights. Across 11 benchmarks and 6 LLMs, our steering raises safety by up to +20% and faithfulness by +27%. In adversarial attack mode, it drops safety by -41% alone, and -100% when combined with existing jailbreak methods, bypassing all safety guardrails and exposing a new dimension of alignment faking hidden within experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08219v3">Investigating Energy Efficiency and Performance Trade-offs in LLM Inference Across Tasks and DVFS Settings</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing (NLP) tasks, leading to widespread adoption in both research and industry. However, their inference workloads are computationally and energy intensive, raising concerns about sustainability and environmental impact. As LLMs continue to scale, it becomes essential to identify and optimize the factors that influence their runtime efficiency without compromising performance. In this work, we systematically investigate the energy-performance trade-offs of LLMs during inference. We benchmark models of varying sizes and architectures, including Falcon-7B, Mistral-7B-v0.1, LLaMA-3.2-1B, LLaMA-3.2-3B, and GPT-Neo-2.7B, across tasks such as question answering, commonsense reasoning, and factual generation. We analyze the effect of input characteristics, such as sequence length, entropy, named entity density and so on. Furthermore, we examine the impact of hardware-level optimizations through Dynamic Voltage and Frequency Scaling (DVFS), measuring how different GPU clock settings affect latency and power consumption. Our empirical findings show that model architecture, input complexity, and clock configuration significantly influence inference efficiency. By correlating input features with energy metrics and evaluating DVFS behavior, we identify practical strategies that reduce energy consumption by up to 30% while preserving model quality. This study provides actionable insights for designing energy-efficient and sustainable LLM inference systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09650v1">All for One: LLMs Solve Mental Math at the Last Token With Information Transferred From Other Tokens</a></div>
    <div class="paper-meta">
      📅 2025-09-11
      | 💬 EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate proficiency across numerous computational tasks, yet their inner workings remain unclear. In theory, the combination of causal self-attention and multilayer perceptron layers allows every token to access and compute information based on all preceding tokens. In practice, to what extent are such operations present? In this paper, on mental math tasks (i.e., direct math calculation via next-token prediction without explicit reasoning), we investigate this question in three steps: inhibiting input-specific token computations in the initial layers, restricting the routes of information transfer across token positions in the next few layers, and forcing all computation to happen at the last token in the remaining layers. With two proposed techniques, Context-Aware Mean Ablation (CAMA) and Attention-Based Peeking (ABP), we identify an All-for-One subgraph (AF1) with high accuracy on a wide variety of mental math tasks, where meaningful computation occurs very late (in terms of layer depth) and only at the last token, which receives information of other tokens in few specific middle layers. Experiments on a variety of models and arithmetic expressions show that this subgraph is sufficient and necessary for high model performance, transfers across different models, and works on a variety of input styles. Ablations on different CAMA and ABP alternatives reveal their unique advantages over other methods, which may be of independent interest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09629v1">Bridging the Capability Gap: Joint Alignment Tuning for Harmonizing LLM-based Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-11
      | 💬 EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      The advancement of large language models (LLMs) has enabled the construction of multi-agent systems to solve complex tasks by dividing responsibilities among specialized agents, such as a planning agent for subgoal generation and a grounding agent for executing tool-use actions. Most existing methods typically fine-tune these agents independently, leading to capability gaps among them with poor coordination. To address this, we propose MOAT, a Multi-Agent Joint Alignment Tuning framework that improves agents collaboration through iterative alignment. MOAT alternates between two key stages: (1) Planning Agent Alignment, which optimizes the planning agent to generate subgoal sequences that better guide the grounding agent; and (2) Grounding Agent Improving, which fine-tunes the grounding agent using diverse subgoal-action pairs generated by the agent itself to enhance its generalization capablity. Theoretical analysis proves that MOAT ensures a non-decreasing and progressively convergent training process. Experiments across six benchmarks demonstrate that MOAT outperforms state-of-the-art baselines, achieving average improvements of 3.1% on held-in tasks and 4.4% on held-out tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09596v1">How much are LLMs changing the language of academic papers after ChatGPT? A multi-database and full text analysis</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      This study investigates how Large Language Models (LLMs) are influencing the language of academic papers by tracking 12 LLM-associated terms across six major scholarly databases (Scopus, Web of Science, PubMed, PubMed Central (PMC), Dimensions, and OpenAlex) from 2015 to 2024. Using over 2.4 million PMC open-access publications (2021-July 2025), we also analysed full texts to assess changes in the frequency and co-occurrence of these terms before and after ChatGPT's initial public release. Across databases, delve (+1,500%), underscore (+1,000%), and intricate (+700%) had the largest increases between 2022 and 2024. Growth in LLM-term usage was much higher in STEM fields than in social sciences and arts and humanities. In PMC full texts, the proportion of papers using underscore six or more times increased by over 10,000% from 2022 to 2025, followed by intricate (+5,400%) and meticulous (+2,800%). Nearly half of all 2024 PMC papers using any LLM term also included underscore, compared with only 3%-14% of papers before ChatGPT in 2022. Papers using one LLM term are now much more likely to include other terms. For example, in 2024, underscore strongly correlated with pivotal (0.449) and delve (0.311), compared with very weak associations in 2022 (0.032 and 0.018, respectively). These findings provide the first large-scale evidence based on full-text publications and multiple databases that some LLM-related terms are now being used much more frequently and together. The rapid uptake of LLMs to support scholarly publishing is a welcome development reducing the language barrier to academic publishing for non-English speakers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08031v2">AU-Harness: An Open-Source Toolkit for Holistic Evaluation of Audio LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Large Audio Language Models (LALMs) are rapidly advancing, but evaluating them remains challenging due to inefficient toolkits that limit fair comparison and systematic assessment. Current frameworks suffer from three critical issues: slow processing that bottlenecks large-scale studies, inconsistent prompting that hurts reproducibility, and narrow task coverage that misses important audio reasoning capabilities. We introduce AU-Harness, an efficient and comprehensive evaluation framework for LALMs. Our system achieves a speedup of up to 127% over existing toolkits through optimized batch processing and parallel execution, enabling large-scale evaluations previously impractical. We provide standardized prompting protocols and flexible configurations for fair model comparison across diverse scenarios. Additionally, we introduce two new evaluation categories: LLM-Adaptive Diarization for temporal audio understanding and Spoken Language Reasoning for complex audio-based cognitive tasks. Through evaluation across 380+ tasks, we reveal significant gaps in current LALMs, particularly in temporal understanding and complex spoken language reasoning tasks. Our findings also highlight a lack of standardization in instruction modality existent across audio benchmarks, which can lead up performance differences up to 9.5 absolute points on the challenging complex instruction following downstream tasks. AU-Harness provides both practical evaluation tools and insights into model limitations, advancing systematic LALM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06485v2">Task Matters: Knowledge Requirements Shape LLM Responses to Context-Memory Conflict</a></div>
    <div class="paper-meta">
      📅 2025-09-11
      | 💬 Major revision
    </div>
    <details class="paper-abstract">
      Large Language Models require both contextual knowledge and parametric memory, but these sources can disagree. Prior investigations on contextual question answering tasks report a preference toward parametric knowledge under conflict, yet they focus almost exclusively on tasks that should always rely on the given passage, leaving open how this behavior manifests when tasks demand different amounts and kinds of knowledge. We study this question with a model-agnostic diagnostic framework that (i) automatically detects disagreements between a model's beliefs and a curated knowledge set, and (ii) injects controlled conflicts into tasks. The resulting datasets span two orthogonal dimensions: task knowledge reliance and conflict plausibility. Evaluating representative open-source LLMs, we find that: (1) performance degradation from conflict correlates with a task's knowledge reliance; (2) explanatory rationales and simple reiteration both increase context reliance-helpful for context-only tasks but harmful when parametric knowledge should dominate; (3) These behaviors raise concerns about the validity of model-based evaluation and underscore the need to account for knowledge conflict in the deployment of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01080v2">Development and Comparative Evaluation of Three Artificial Intelligence Models (NLP, LLM, JEPA) for Predicting Triage in Emergency Departments: A 7-Month Retrospective Proof-of-Concept</a></div>
    <div class="paper-meta">
      📅 2025-09-11
      | 💬 13 pages, 7 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Emergency departments struggle with persistent triage errors, especially undertriage and overtriage, which are aggravated by growing patient volumes and staff shortages. This study evaluated three AI models [TRIAGEMASTER (NLP), URGENTIAPARSE (LLM), and EMERGINET (JEPA)] against the FRENCH triage scale and nurse practice, using seven months of adult triage data from Roger Salengro Hospital in Lille, France. Among the models, the LLM-based URGENTIAPARSE consistently outperformed both AI alternatives and nurse triage, achieving the highest accuracy (F1-score 0.900, AUC-ROC 0.879) and superior performance in predicting hospitalization needs (GEMSA). Its robustness across structured data and raw transcripts highlighted the advantage of LLM architectures in abstracting patient information. Overall, the findings suggest that integrating LLM-based AI into emergency department workflows could significantly enhance patient safety and operational efficiency, though successful adoption will depend on addressing limitations and ensuring ethical transparency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04867v2">LLMs for sensory-motor control: Combining in-context and iterative learning</a></div>
    <div class="paper-meta">
      📅 2025-09-11
      | 💬 Article updated with results from gpt-oss:120b. 24 pages (13 pages are from appendix), 6 figures, code for experiments replication and supplementary material provided at https://github.com/jtyska/llm-robotics-article/
    </div>
    <details class="paper-abstract">
      We propose a method that enables large language models (LLMs) to control embodied agents by directly mapping continuous observation vectors to continuous action vectors. At the outset, the LLMs generate a control strategy based on a textual description of the agent, its environment, and the intended goal. This strategy is then iteratively refined through a learning process in which the LLMs are repeatedly prompted to improve the current strategy, using performance feedback and sensory-motor data collected during its evaluation. The method is validated on classic control tasks from the Gymnasium library and the inverted pendulum task from the MuJoCo library. The approach proves effective with relatively compact models such as Gpt-oss:120b and Qwen2.5:72b. In most cases, it successfully identifies optimal or near-optimal solutions by integrating symbolic knowledge derived through reasoning with sub-symbolic sensory-motor data gathered as the agent interacts with its environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09505v1">Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      LLMs now form the backbone of AI agents for a diverse array of applications, including tool use, command-line agents, and web or computer use agents. These agentic LLM inference tasks are fundamentally different from chatbot-focused inference -- they often have much larger context lengths to capture complex, prolonged inputs, such as entire webpage DOMs or complicated tool call trajectories. This, in turn, generates significant off-chip memory traffic for the underlying hardware at the inference stage and causes the workload to be constrained by two memory walls, namely the bandwidth and capacity memory walls, preventing the on-chip compute units from achieving high utilization. In this paper, we introduce PLENA, a hardware-software co-designed system that applies three core optimization pathways to tackle these challenges. PLENA includes an efficient hardware implementation of compute and memory units supporting an asymmetric quantization scheme. PLENA also features a novel flattened systolic array architecture that has native support for FlashAttention to tackle these memory walls in the scenario of inference serving for long-context LLMs. Additionally, PLENA is developed with a complete stack, including a custom ISA, a compiler, a cycle-emulated simulator, and an automated design space exploration flow. The simulated results show that PLENA achieves up to 8.5x higher utilization than existing accelerators, and delivers 2.24x higher throughput than the A100 GPU and 3.85x higher throughput than the TPU v6e, under the same multiplier count and memory settings. The full PLENA system will also be open-sourced.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.20999v2">LoRA-PAR: A Flexible Dual-System LoRA Partitioning Approach to Efficient LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-09-11
      | 💬 12 pages
    </div>
    <details class="paper-abstract">
      Large-scale generative models like DeepSeek-R1 and OpenAI-O1 benefit substantially from chain-of-thought (CoT) reasoning, yet pushing their performance typically requires vast data, large model sizes, and full-parameter fine-tuning. While parameter-efficient fine-tuning (PEFT) helps reduce cost, most existing approaches primarily address domain adaptation or layer-wise allocation rather than explicitly tailoring data and parameters to different response demands. Inspired by "Thinking, Fast and Slow," which characterizes two distinct modes of thought-System 1 (fast, intuitive, often automatic) and System 2 (slower, more deliberative and analytic)-we draw an analogy that different "subregions" of an LLM's parameters might similarly specialize for tasks that demand quick, intuitive responses versus those requiring multi-step logical reasoning. Therefore, we propose LoRA-PAR, a dual-system LoRA framework that partitions both data and parameters by System 1 or System 2 demands, using fewer yet more focused parameters for each task. Specifically, we classify task data via multi-model role-playing and voting, and partition parameters based on importance scoring, then adopt a two-stage fine-tuning strategy of training System 1 tasks with supervised fine-tuning (SFT) to enhance knowledge and intuition and refine System 2 tasks with reinforcement learning (RL) to reinforce deeper logical deliberation next. Extensive experiments show that the two-stage fine-tuning strategy, SFT and RL, lowers active parameter usage while matching or surpassing SOTA PEFT baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09461v1">Changing the Paradigm from Dynamic Queries to LLM-generated SQL Queries with Human Intervention</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      We propose leveraging Large Language Models (LLMs) as an interaction layer for medical visualization systems. In domains like healthcare, where users must navigate high-dimensional, coded, and heterogeneous datasets, LLM-generated queries enable expert medical users to express complex analytical intents in natural language. These intents are then translated into editable and executable queries, replacing the dynamic query interfaces used by traditional visualization systems built around sliders, check boxes, and drop-downs. This interaction model reduces visual clutter and eliminates the need for users to memorize field names or system codes, supporting fluid exploration, with the drawback of not exposing all the filtering criteria. We also reintroduce dynamic queries on demand to better support interactive exploration. We posit that medical users are trained to know the possible filtering options but challenged to remember the details of the attribute names and code values. We demonstrate this paradigm in ParcoursVis, our scalable EventFlow-inspired patient care pathway visualization system powered by the French National Health Data System, one of the largest health data repositories in the world.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09420v1">HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing</a></div>
    <div class="paper-meta">
      📅 2025-09-11
      | 💬 9 pages, 15 figures, International Conference on Computer-Aided Design (ICCAD) 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) with Mixture-of-Expert (MoE) architectures achieve superior model performance with reduced computation costs, but at the cost of high memory capacity and bandwidth requirements. Near-Memory Processing (NMP) accelerators that stack memory directly on the compute through hybrid bonding have demonstrated high bandwidth with high energy efficiency, becoming a promising architecture for MoE models. However, as NMP accelerators comprise distributed memory and computation, how to map the MoE computation directly determines the LLM inference efficiency. Existing parallel mapping strategies, including Tensor Parallelism (TP) and Expert Parallelism (EP), suffer from either high communication costs or unbalanced computation utilization, leading to inferior efficiency. The dynamic routing mechanism of MoE LLMs further aggravates the efficiency challenges. Therefore, in this paper, we propose HD-MoE to automatically optimize the MoE parallel computation across an NMP accelerator. HD-MoE features an offline automatic hybrid parallel mapping algorithm and an online dynamic scheduling strategy to reduce the communication costs while maximizing the computation utilization. With extensive experimental results, we demonstrate that HD-MoE achieves a speedup ranging from 1.1x to 1.8x over TP, 1.1x to 1.5x over EP, and 1.0x to 1.4x over the baseline Hybrid TP-EP with Compute-Balanced parallelism strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17274v7">CleanVul: Automatic Function-Level Vulnerability Detection in Code Commits Using LLM Heuristics</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Accurate identification of software vulnerabilities is crucial for system integrity. Vulnerability datasets, often derived from the National Vulnerability Database (NVD) or directly from GitHub, are essential for training machine learning models to detect these security flaws. However, these datasets frequently suffer from significant noise, typically 40% to 75%, due primarily to the automatic and indiscriminate labeling of all changes in vulnerability-fixing commits (VFCs) as vulnerability-related. This misclassification occurs because not all changes in a commit aimed at fixing vulnerabilities pertain to security threats; many are routine updates like bug fixes or test improvements. This paper introduces the first methodology that uses the Large Language Model (LLM) with a heuristic enhancement to automatically identify vulnerability-fixing changes from VFCs, achieving an F1-score of 0.82. VulSifter was applied to a large-scale study, where we conducted a crawl of 127,063 repositories on GitHub, resulting in the acquisition of 5,352,105 commits. VulSifter involves utilizing an LLM to comprehend code semantics and contextual information, while applying heuristics to filter out unrelated changes. We then developed CleanVul, a high-quality dataset comprising 8,198 functions using our LLM heuristic enhancement approach, demonstrating Correctness (90.6%) comparable to established datasets such as SVEN and PrimeVul. To evaluate the CleanVul dataset, we conducted experiments focusing on fine-tuning various LLMs on CleanVul and other high-quality datasets. Evaluation results reveal that LLMs fine-tuned on CleanVul not only exhibit enhanced accuracy but also superior generalization capabilities compared to those trained on uncleaned datasets. Specifically, models trained on CleanVul and tested on PrimeVul achieve accuracy higher than those trained and tested exclusively on PrimeVul.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04227v3">Can LLMs Hack Enterprise Networks? Autonomous Assumed Breach Penetration-Testing Active Directory Networks</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Enterprise penetration-testing is often limited by high operational costs and the scarcity of human expertise. This paper investigates the feasibility and effectiveness of using Large Language Model (LLM)-driven autonomous systems to address these challenges in real-world Active Directory (AD) enterprise networks. We introduce a novel prototype designed to employ LLMs to autonomously perform Assumed Breach penetration-testing against enterprise networks. Our system represents the first demonstration of a fully autonomous, LLM-driven framework capable of compromising accounts within a real-life Microsoft Active Directory testbed, GOAD. We perform our empirical evaluation using five LLMs, comparing reasoning to non-reasoning models as well as including open-weight models. Through quantitative and qualitative analysis, incorporating insights from cybersecurity experts, we demonstrate that autonomous LLMs can effectively conduct Assumed Breach simulations. Key findings highlight their ability to dynamically adapt attack strategies, perform inter-context attacks (e.g., web-app audits, social engineering, and unstructured data analysis for credentials), and generate scenario-specific attack parameters like realistic password candidates. The prototype exhibits robust self-correction mechanisms, installing missing tools and rectifying invalid command generations. We find that the associated costs are competitive with, and often significantly lower than, those incurred by professional human pen-testers, suggesting a path toward democratizing access to essential security testing for organizations with budgetary constraints. However, our research also illuminates existing limitations, including instances of LLM ``going down rabbit holes'', challenges in comprehensive information transfer between planning and execution modules, and critical safety concerns that necessitate human oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09396v1">LLMs Don't Know Their Own Decision Boundaries: The Unreliability of Self-Generated Counterfactual Explanations</a></div>
    <div class="paper-meta">
      📅 2025-09-11
      | 💬 Accepted to EMNLP 2025 Main
    </div>
    <details class="paper-abstract">
      To collaborate effectively with humans, language models must be able to explain their decisions in natural language. We study a specific type of self-explanation: self-generated counterfactual explanations (SCEs), where a model explains its prediction by modifying the input such that it would have predicted a different outcome. We evaluate whether LLMs can produce SCEs that are valid, achieving the intended outcome, and minimal, modifying the input no more than necessary. When asked to generate counterfactuals, we find that LLMs typically produce SCEs that are valid, but far from minimal, offering little insight into their decision-making behaviour. Worryingly, when asked to generate minimal counterfactuals, LLMs typically make excessively small edits that fail to change predictions. The observed validity-minimality trade-off is consistent across several LLMs, datasets, and evaluation settings. Our findings suggest that SCEs are, at best, an ineffective explainability tool and, at worst, can provide misleading insights into model behaviour. Proposals to deploy LLMs in high-stakes settings must consider the impact of unreliable self-explanations on downstream decision-making. Our code is available at https://github.com/HarryMayne/SCEs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.02253v5">Scaling LLM Planning: NL2FLOW for Parametric Problem Generation and Rigorous Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-09-11
      | 💬 31 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Robust workflow composition is critical for effective agent performance, yet progress in Large Language Model (LLM) planning and reasoning is hindered by a scarcity of scalable evaluation data. This work introduces NL2Flow, a fully automated pipeline for generating and evaluating workflow planning problems. NL2Flow generates problems parametrically in a structured intermediate representation, translating them into both natural language and formal PDDL. I evaluate several open-source, instruct-tuned LLMs on a dataset of 2296 low-difficulty problems generated by NL2Flow. Results demonstrate that the best-performing model achieved 86% success in generating valid plans and 69% in generating optimal plans (for solvable problems). Regression analysis shows that the influence of problem characteristics on plan generation is contingent on both model and prompt design. Importantly, translating natural language problems into a structured JSON representation prior to symbolic planning significantly improved success rates, suggesting a benefit from neuro-symbolic integration. These findings underscore the importance of understanding error sources within LLM reasoning as systems scale to more complex tasks. As LLM reasoning scales to increasingly complex problems, understanding the shifting bottlenecks and sources of error within these systems will be crucial.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09387v1">MetaLLMix : An XAI Aided LLM-Meta-learning Based Approach for Hyper-parameters Optimization</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Effective model and hyperparameter selection remains a major challenge in deep learning, often requiring extensive expertise and computation. While AutoML and large language models (LLMs) promise automation, current LLM-based approaches rely on trial and error and expensive APIs, which provide limited interpretability and generalizability. We propose MetaLLMiX, a zero-shot hyperparameter optimization framework combining meta-learning, explainable AI, and efficient LLM reasoning. By leveraging historical experiment outcomes with SHAP explanations, MetaLLMiX recommends optimal hyperparameters and pretrained models without additional trials. We further employ an LLM-as-judge evaluation to control output format, accuracy, and completeness. Experiments on eight medical imaging datasets using nine open-source lightweight LLMs show that MetaLLMiX achieves competitive or superior performance to traditional HPO methods while drastically reducing computational cost. Our local deployment outperforms prior API-based approaches, achieving optimal results on 5 of 8 tasks, response time reductions of 99.6-99.9%, and the fastest training times on 6 datasets (2.4-15.7x faster), maintaining accuracy within 1-5% of best-performing baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.07370v2">PersonaFuse: A Personality Activation-Driven Framework for Enhancing Human-LLM Interactions</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) demonstrate remarkable capabilities across various fields. These developments have led to more direct communication between humans and LLMs in various situations, such as social companionship and psychological support. However, LLMs often exhibit limitations in emotional perception and social competence during real-world conversations. These limitations partly originate from their inability to adapt their communication style and emotional expression to different social and task contexts. In this work, we introduce PersonaFuse, a novel LLM post-training framework that enables LLMs to adapt and express different personalities for varying situations. Inspired by Trait Activation Theory and the Big Five personality model, PersonaFuse employs a Mixture-of-Expert architecture that combines persona adapters with a dynamic routing network, enabling contextual trait expression. Experimental results show that PersonaFuse substantially outperforms baseline models across multiple dimensions of social-emotional intelligence. Importantly, these gains are achieved without sacrificing general reasoning ability or model safety, which remain common limitations of direct prompting and supervised fine-tuning approaches. PersonaFuse also delivers consistent improvements in downstream human-centered applications, such as mental health counseling and review-based customer service. Finally, human preference evaluations against leading LLMs, including GPT-4o and DeepSeek, demonstrate that PersonaFuse achieves competitive response quality despite its comparatively smaller model size. These findings demonstrate that PersonaFuse offers a theoretically grounded and practical approach for developing social-emotional enhanced LLMs, marking a significant advancement toward more human-centric AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09291v1">What You Code Is What We Prove: Translating BLE App Logic into Formal Models with LLMs for Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      The application layer of Bluetooth Low Energy (BLE) is a growing source of security vulnerabilities, as developers often neglect to implement critical protections such as encryption, authentication, and freshness. While formal verification offers a principled way to check these properties, the manual effort of constructing formal models makes it impractical for large-scale analysis. This paper introduces a key insight: BLE application security analysis can be reframed as a semantic translation problem, i.e., from real-world code to formal models. We leverage large language models (LLMs) not to directly detect vulnerabilities, but to serve as translators that convert BLE-specific code into process models verifiable by tools like ProVerif. We implement this idea in VerifiaBLE, a system that combines static analysis, prompt-guided LLM translation, and symbolic verification to check three core security features: encryption, randomness, and authentication. Applied to 1,050 Android BLE apps, VerifiaBLE uncovers systemic weaknesses: only 10.2\% of apps implement all three protections, while 53.9\% omit them entirely. Our work demonstrates that using LLMs as structured translators can lower the barrier to formal methods, unlocking scalable verification across security-critical domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09272v1">Fusing Knowledge and Language: A Comparative Study of Knowledge Graph-Based Question Answering with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-11
      | 💬 46 pages, 4 figures, 17 tables
    </div>
    <details class="paper-abstract">
      Knowledge graphs, a powerful tool for structuring information through relational triplets, have recently become the new front-runner in enhancing question-answering systems. While traditional Retrieval Augmented Generation (RAG) approaches are proficient in fact-based and local context-based extraction from concise texts, they encounter limitations when addressing the thematic and holistic understanding of complex, extensive texts, requiring a deeper analysis of both text and context. This paper presents a comprehensive technical comparative study of three different methodologies for constructing knowledge graph triplets and integrating them with Large Language Models (LLMs) for question answering: spaCy, Stanford CoreNLP-OpenIE, and GraphRAG, all leveraging open source technologies. We evaluate the effectiveness, feasibility, and adaptability of these methods by analyzing their capabilities, state of development, and their impact on the performance of LLM-based question answering. Experimental results indicate that while OpenIE provides the most comprehensive coverage of triplets, GraphRAG demonstrates superior reasoning abilities among the three. We conclude with a discussion on the strengths and limitations of each method and provide insights into future directions for improving knowledge graph-based question answering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09265v1">Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-09-11
      | 💬 ICLR 2026 Under review
    </div>
    <details class="paper-abstract">
      In long-horizon tasks, recent agents based on Large Language Models (LLMs) face a significant challenge that sparse, outcome-based rewards make it difficult to assign credit to intermediate steps. Previous methods mainly focus on creating dense reward signals to guide learning, either through traditional reinforcement learning techniques like inverse reinforcement learning or by using Process Reward Models for step-by-step feedback. In this paper, we identify a fundamental problem in the learning dynamics of LLMs: the magnitude of policy gradients is inherently coupled with the entropy, which leads to inefficient small updates for confident correct actions and potentially destabilizes large updates for uncertain ones. To resolve this, we propose Entropy-Modulated Policy Gradients (EMPG), a framework that re-calibrates the learning signal based on step-wise uncertainty and the final task outcome. EMPG amplifies updates for confident correct actions, penalizes confident errors, and attenuates updates from uncertain steps to stabilize exploration. We further introduce a bonus term for future clarity that encourages agents to find more predictable solution paths. Through comprehensive experiments on three challenging agent tasks, WebShop, ALFWorld, and Deep Search, we demonstrate that EMPG achieves substantial performance gains and significantly outperforms strong policy gradient baselines. Project page is at https://empgseed-seed.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.09245v1">Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search</a></div>
    <div class="paper-meta">
      📅 2025-09-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great promise in automating data science workflows, but existing models still struggle with multi-step reasoning and tool use, which limits their effectiveness on complex data analysis tasks. To address this, we propose a scalable pipeline that extracts high-quality, tool-based data analysis tasks and their executable multi-step solutions from real-world Jupyter notebooks and associated data files. Using this pipeline, we introduce NbQA, a large-scale dataset of standardized task-solution pairs that reflect authentic tool-use patterns in practical data science scenarios. To further enhance multi-step reasoning, we present Jupiter, a framework that formulates data analysis as a search problem and applies Monte Carlo Tree Search (MCTS) to generate diverse solution trajectories for value model learning. During inference, Jupiter combines the value model and node visit counts to efficiently collect executable multi-step plans with minimal search steps. Experimental results show that Qwen2.5-7B and 14B-Instruct models on NbQA solve 77.82% and 86.38% of tasks on InfiAgent-DABench, respectively-matching or surpassing GPT-4o and advanced agent frameworks. Further evaluations demonstrate improved generalization and stronger tool-use reasoning across diverse multi-step reasoning tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08542v1">BitROM: Weight Reload-Free CiROM Architecture Towards Billion-Parameter 1.58-bit LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 Accepted to ASP-DAC 2026
    </div>
    <details class="paper-abstract">
      Compute-in-Read-Only-Memory (CiROM) accelerators offer outstanding energy efficiency for CNNs by eliminating runtime weight updates. However, their scalability to Large Language Models (LLMs) is fundamentally constrained by their vast parameter sizes. Notably, LLaMA-7B - the smallest model in LLaMA series - demands more than 1,000 cm2 of silicon area even in advanced CMOS nodes. This paper presents BitROM, the first CiROM-based accelerator that overcomes this limitation through co-design with BitNet's 1.58-bit quantization model, enabling practical and efficient LLM inference at the edge. BitROM introduces three key innovations: 1) a novel Bidirectional ROM Array that stores two ternary weights per transistor; 2) a Tri-Mode Local Accumulator optimized for ternary-weight computations; and 3) an integrated Decode-Refresh (DR) eDRAM that supports on-die KV-cache management, significantly reducing external memory access during decoding. In addition, BitROM integrates LoRA-based adapters to enable efficient transfer learning across various downstream tasks. Evaluated in 65nm CMOS, BitROM achieves 20.8 TOPS/W and a bit density of 4,967 kB/mm2 - offering a 10x improvement in area efficiency over prior digital CiROM designs. Moreover, the DR eDRAM contributes to a 43.6% reduction in external DRAM access, further enhancing deployment efficiency for LLMs in edge applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16654v3">MSNav: Zero-Shot Vision-and-Language Navigation with Dynamic Memory and LLM Spatial Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-10
      | 💬 9 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) requires an agent to interpret natural language instructions and navigate complex environments. Current approaches often adopt a "black-box" paradigm, where a single Large Language Model (LLM) makes end-to-end decisions. However, it is plagued by critical vulnerabilities, including poor spatial reasoning, weak cross-modal grounding, and memory overload in long-horizon tasks. To systematically address these issues, we propose Memory Spatial Navigation(MSNav), a framework that fuses three modules into a synergistic architecture, which transforms fragile inference into a robust, integrated intelligence. MSNav integrates three modules: Memory Module, a dynamic map memory module that tackles memory overload through selective node pruning, enhancing long-range exploration; Spatial Module, a module for spatial reasoning and object relationship inference that improves endpoint recognition; and Decision Module, a module using LLM-based path planning to execute robust actions. Powering Spatial Module, we also introduce an Instruction-Object-Space (I-O-S) dataset and fine-tune the Qwen3-4B model into Qwen-Spatial (Qwen-Sp), which outperforms leading commercial LLMs in object list extraction, achieving higher F1 and NDCG scores on the I-O-S test set. Extensive experiments on the Room-to-Room (R2R) and REVERIE datasets demonstrate MSNav's state-of-the-art performance with significant improvements in Success Rate (SR) and Success weighted by Path Length (SPL).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.08493v1">Send to which account? Evaluation of an LLM-based Scambaiting System</a></div>
    <div class="paper-meta">
      📅 2025-09-10
    </div>
    <details class="paper-abstract">
      Scammers are increasingly harnessing generative AI(GenAI) technologies to produce convincing phishing content at scale, amplifying financial fraud and undermining public trust. While conventional defenses, such as detection algorithms, user training, and reactive takedown efforts remain important, they often fall short in dismantling the infrastructure scammers depend on, including mule bank accounts and cryptocurrency wallets. To bridge this gap, a proactive and emerging strategy involves using conversational honeypots to engage scammers and extract actionable threat intelligence. This paper presents the first large-scale, real-world evaluation of a scambaiting system powered by large language models (LLMs). Over a five-month deployment, the system initiated over 2,600 engagements with actual scammers, resulting in a dataset of more than 18,700 messages. It achieved an Information Disclosure Rate (IDR) of approximately 32%, successfully extracting sensitive financial information such as mule accounts. Additionally, the system maintained a Human Acceptance Rate (HAR) of around 70%, indicating strong alignment between LLM-generated responses and human operator preferences. Alongside these successes, our analysis reveals key operational challenges. In particular, the system struggled with engagement takeoff: only 48.7% of scammers responded to the initial seed message sent by defenders. These findings highlight the need for further refinement and provide actionable insights for advancing the design of automated scambaiting systems.
    </details>
</div>
