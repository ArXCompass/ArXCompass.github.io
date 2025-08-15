# llm - 2025_08

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17217v2">Mitigating Gender Bias via Fostering Exploratory Thinking in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often exhibit gender bias, resulting in unequal treatment of male and female subjects across different contexts. To address this issue, we propose a novel data generation framework that fosters exploratory thinking in LLMs. Our approach prompts models to generate story pairs featuring male and female protagonists in structurally identical, morally ambiguous scenarios, then elicits and compares their moral judgments. When inconsistencies arise, the model is guided to produce balanced, gender-neutral judgments. These story-judgment pairs are used to fine-tune or optimize the models via Direct Preference Optimization (DPO). Experimental results show that our method significantly reduces gender bias while preserving or even enhancing general model capabilities. We will release the code and generated data. We release the code and generated data at: https://github.com/WeiKangda/LLMs-Exploratory-Bias-Mitigation/tree/main.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09751v2">Sound and Complete Neurosymbolic Reasoning with LLM-Grounded Interpretations</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 29 pages, 9 tables, 3 figures. Accepted to the 19th Conference on Neurosymbolic Learning and Reasoning (NeSy 2025)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities in natural language understanding and generation, but they exhibit problems with logical consistency in the output they generate. How can we harness LLMs' broad-coverage parametric knowledge in formal reasoning despite their inconsistency? We present a method for directly integrating an LLM into the interpretation function of the formal semantics for a paraconsistent logic. We provide experimental evidence for the feasibility of the method by evaluating the function using datasets created from several short-form factuality benchmarks. Unlike prior work, our method offers a theoretical framework for neurosymbolic reasoning that leverages an LLM's knowledge while preserving the underlying logic's soundness and completeness properties.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00741v1">Out-of-Context Abduction: LLMs Make Inferences About Procedural Data Leveraging Declarative Facts in Earlier Training Data</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are trained on large corpora, yet it is unclear whether they can reason about the information present within their training data. We design experiments to study out-of-context abduction in LLMs, the ability to infer the most plausible explanations for observations using relevant facts present in training data. We train treatment LLMs on names and behavior descriptions of fictitious chatbots, but not on examples of dialogue with the chatbots. We find that OpenAI's GPT 4o LLM can correctly infer at least one chatbot's name after observing example responses characteristic of that chatbot. We also find that previously training GPT 4o on descriptions of a chatbot's behavior allows it to display behaviors more characteristic of the chatbot when iteratively trained to display such behaviors. Our results have implications for situational awareness in LLMs and, therefore, for AI safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00737v1">How LLMs are Shaping the Future of Virtual Reality</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 Pre-print
    </div>
    <details class="paper-abstract">
      The integration of Large Language Models (LLMs) into Virtual Reality (VR) games marks a paradigm shift in the design of immersive, adaptive, and intelligent digital experiences. This paper presents a comprehensive review of recent research at the intersection of LLMs and VR, examining how these models are transforming narrative generation, non-player character (NPC) interactions, accessibility, personalization, and game mastering. Drawing from an analysis of 62 peer reviewed studies published between 2018 and 2025, we identify key application domains ranging from emotionally intelligent NPCs and procedurally generated storytelling to AI-driven adaptive systems and inclusive gameplay interfaces. We also address the major challenges facing this convergence, including real-time performance constraints, memory limitations, ethical risks, and scalability barriers. Our findings highlight that while LLMs significantly enhance realism, creativity, and user engagement in VR environments, their effective deployment requires robust design strategies that integrate multimodal interaction, hybrid AI architectures, and ethical safeguards. The paper concludes by outlining future research directions in multimodal AI, affective computing, reinforcement learning, and open-source development, aiming to guide the responsible advancement of intelligent and inclusive VR systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00719v1">Dynamically Adaptive Reasoning via LLM-Guided MCTS for Efficient and Context-Aware KGQA</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Knowledge Graph Question Answering (KGQA) aims to interpret natural language queries and perform structured reasoning over knowledge graphs by leveraging their relational and semantic structures to retrieve accurate answers. Recent KGQA methods primarily follow either retrieve-then-reason paradigm, relying on GNNs or heuristic rules for static paths extraction, or dynamic path generation strategies that use large language models (LLMs) with prompting to jointly perform retrieval and reasoning. However, the former suffers from limited adaptability due to static path extraction and lack of contextual refinement, while the latter incurs high computational costs and struggles with accurate path evaluation due to reliance on fixed scoring functions and extensive LLM calls. To address these issues, this paper proposes Dynamically Adaptive MCTS-based Reasoning (DAMR), a novel framework that integrates symbolic search with adaptive path evaluation for efficient and context-aware KGQA. DAMR employs a Monte Carlo Tree Search (MCTS) backbone guided by an LLM-based planner, which selects top-$k$ relevant relations at each step to reduce search space. To improve path evaluation accuracy, we introduce a lightweight Transformer-based scorer that performs context-aware plausibility estimation by jointly encoding the question and relation sequence through cross-attention, enabling the model to capture fine-grained semantic shifts during multi-hop reasoning. Furthermore, to alleviate the scarcity of high-quality supervision, DAMR incorporates a dynamic pseudo-path refinement mechanism that periodically generates training signals from partial paths explored during search, allowing the scorer to continuously adapt to the evolving distribution of reasoning trajectories. Extensive experiments on multiple KGQA benchmarks show that DAMR significantly outperforms state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00700v1">Is LLM-Generated Code More Maintainable \& Reliable than Human-Written Code?</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 Accepted ESEM2025
    </div>
    <details class="paper-abstract">
      Background: The rise of Large Language Models (LLMs) in software development has opened new possibilities for code generation. Despite the widespread use of this technology, it remains unclear how well LLMs generate code solutions in terms of software quality and how they compare to human-written code. Aims: This study compares the internal quality attributes of LLM-generated and human-written code. Method: Our empirical study integrates datasets of coding tasks, three LLM configurations (zero-shot, few-shot, and fine-tuning), and SonarQube to assess software quality. The dataset comprises Python code solutions across three difficulty levels: introductory, interview, and competition. We analyzed key code quality metrics, including maintainability and reliability, and the estimated effort required to resolve code issues. Results: Our analysis shows that LLM-generated code has fewer bugs and requires less effort to fix them overall. Interestingly, fine-tuned models reduced the prevalence of high-severity issues, such as blocker and critical bugs, and shifted them to lower-severity categories, but decreased the model's performance. In competition-level problems, the LLM solutions sometimes introduce structural issues that are not present in human-written code. Conclusion: Our findings provide valuable insights into the quality of LLM-generated code; however, the introduction of critical issues in more complex scenarios highlights the need for a systematic evaluation and validation of LLM solutions. Our work deepens the understanding of the strengths and limitations of LLMs for code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00680v1">Better Call Claude: Can LLMs Detect Changes of Writing Style?</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      This article explores the zero-shot performance of state-of-the-art large language models (LLMs) on one of the most challenging tasks in authorship analysis: sentence-level style change detection. Benchmarking four LLMs on the official PAN~2024 and 2025 "Multi-Author Writing Style Analysis" datasets, we present several observations. First, state-of-the-art generative models are sensitive to variations in writing style - even at the granular level of individual sentences. Second, their accuracy establishes a challenging baseline for the task, outperforming suggested baselines of the PAN competition. Finally, we explore the influence of semantics on model predictions and present evidence suggesting that the latest generation of LLMs may be more sensitive to content-independent and purely stylistic signals than previously reported.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00669v1">Medical Reasoning in the Era of LLMs: A Systematic Review of Enhancement Techniques and Applications</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) in medicine has enabled impressive capabilities, yet a critical gap remains in their ability to perform systematic, transparent, and verifiable reasoning, a cornerstone of clinical practice. This has catalyzed a shift from single-step answer generation to the development of LLMs explicitly designed for medical reasoning. This paper provides the first systematic review of this emerging field. We propose a taxonomy of reasoning enhancement techniques, categorized into training-time strategies (e.g., supervised fine-tuning, reinforcement learning) and test-time mechanisms (e.g., prompt engineering, multi-agent systems). We analyze how these techniques are applied across different data modalities (text, image, code) and in key clinical applications such as diagnosis, education, and treatment planning. Furthermore, we survey the evolution of evaluation benchmarks from simple accuracy metrics to sophisticated assessments of reasoning quality and visual interpretability. Based on an analysis of 60 seminal studies from 2022-2025, we conclude by identifying critical challenges, including the faithfulness-plausibility gap and the need for native multimodal reasoning, and outlining future directions toward building efficient, robust, and sociotechnically responsible medical AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.08395v2">IssueBench: Millions of Realistic Prompts for Measuring Issue Bias in LLM Writing Assistance</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are helping millions of users write texts about diverse issues, and in doing so expose users to different ideas and perspectives. This creates concerns about issue bias, where an LLM tends to present just one perspective on a given issue, which in turn may influence how users think about this issue. So far, it has not been possible to measure which issue biases LLMs actually manifest in real user interactions, making it difficult to address the risks from biased LLMs. Therefore, we create IssueBench: a set of 2.49m realistic prompts for measuring issue bias in LLM writing assistance, which we construct based on 3.9k templates (e.g. "write a blog about") and 212 political issues (e.g. "AI regulation") from real user interactions. Using IssueBench, we show that issue biases are common and persistent in state-of-the-art LLMs. We also show that biases are remarkably similar across models, and that all models align more with US Democrat than Republican voter opinion on a subset of issues. IssueBench can easily be adapted to include other issues, templates, or tasks. By enabling robust and realistic measurement, we hope that IssueBench can bring a new quality of evidence to ongoing discussions about LLM biases and how to address them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12927v2">SEFL: Enhancing Educational Assignment Feedback with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Providing high-quality feedback to student assignments is crucial for student success, but it is constrained by time and costs. In this work, we introduce Synthetic Educational Feedback Loops (SEFL), a synthetic data framework designed to generate data that resembles immediate, on-demand feedback at scale without relying on extensive, real-world student assignments. To get this type of data, two large language models (LLMs) operate in teacher-student roles to simulate assignment completion and formative feedback, generating synthetic pairs of student work and corresponding critiques and actionable improvements from a teacher. With this data, we fine-tune smaller, more computationally efficient LLMs on these synthetic pairs, enabling them to replicate key features of high-quality, goal-oriented feedback. Unlike personalized tutoring approaches that offer multi-turn, individualized instruction, SEFL specifically focuses on replicating the teacher-student assignment feedback loop in higher education. Through comprehensive evaluations with four LLM judges and three human experts, we demonstrate that SEFL-tuned models outperform both their non-tuned counterparts in feedback quality and an existing baseline. The potential for societal impact is reinforced by extensive qualitative comments by ratings by human stakeholders -- both students and higher education instructors. All in all, SEFL has substantial potential to transform feedback processes for higher education and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00602v1">LeakSealer: A Semisupervised Defense for LLMs Against Prompt Injection and Leakage Attacks</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 22 pages, preprint
    </div>
    <details class="paper-abstract">
      The generalization capabilities of Large Language Models (LLMs) have led to their widespread deployment across various applications. However, this increased adoption has introduced several security threats, notably in the forms of jailbreaking and data leakage attacks. Additionally, Retrieval Augmented Generation (RAG), while enhancing context-awareness in LLM responses, has inadvertently introduced vulnerabilities that can result in the leakage of sensitive information. Our contributions are twofold. First, we introduce a methodology to analyze historical interaction data from an LLM system, enabling the generation of usage maps categorized by topics (including adversarial interactions). This approach further provides forensic insights for tracking the evolution of jailbreaking attack patterns. Second, we propose LeakSealer, a model-agnostic framework that combines static analysis for forensic insights with dynamic defenses in a Human-In-The-Loop (HITL) pipeline. This technique identifies topic groups and detects anomalous patterns, allowing for proactive defense mechanisms. We empirically evaluate LeakSealer under two scenarios: (1) jailbreak attempts, employing a public benchmark dataset, and (2) PII leakage, supported by a curated dataset of labeled LLM interactions. In the static setting, LeakSealer achieves the highest precision and recall on the ToxicChat dataset when identifying prompt injection. In the dynamic setting, PII leakage detection achieves an AUPRC of $0.97$, significantly outperforming baselines such as Llama Guard.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15066v2">Time-RA: Towards Time Series Reasoning for Anomaly with LLM Feedback</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 Under review. 19 pages, 8 figures, 12 tables. Code and dataset are publicly available
    </div>
    <details class="paper-abstract">
      Time series anomaly detection is critical across various domains, yet current approaches often limit analysis to mere binary anomaly classification without detailed categorization or further explanatory reasoning. To address these limitations, we propose a novel task, Time-series Reasoning for Anomaly (Time-RA) that transforms classical time series anomaly detection from a discriminative into a generative, reasoning-intensive task leveraging Large Language Models (LLMs). Also, we introduce the first real-world multimodal benchmark dataset, RATs40K, explicitly annotated for anomaly reasoning, comprising approximately 40,000 samples across 10 real-world domains. Each sample includes numeric time series data, contextual text information, and visual representations, each annotated with fine-grained categories (14 types for univariate anomalies and 6 for multivariate anomalies) and structured explanatory reasoning. We develop a sophisticated annotation framework utilizing ensemble-generated labels refined through GPT-4-driven feedback, ensuring accuracy and interpretability. Extensive benchmarking of LLMs and multimodal LLMs demonstrates the capabilities and limitations of current models, highlighting the critical role of supervised fine-tuning. Our dataset and task pave the way for significant advancements in interpretable time series anomaly detection and reasoning. The code (https://github.com/yyysjz1997/Time-RA) and dataset (https://huggingface.co/datasets/Time-RA/RATs40K) have been fully open-sourced to support and accelerate future research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00581v1">From EMR Data to Clinical Insight: An LLM-Driven Framework for Automated Pre-Consultation Questionnaire Generation</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 16 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Pre-consultation is a critical component of effective healthcare delivery. However, generating comprehensive pre-consultation questionnaires from complex, voluminous Electronic Medical Records (EMRs) is a challenging task. Direct Large Language Model (LLM) approaches face difficulties in this task, particularly regarding information completeness, logical order, and disease-level synthesis. To address this issue, we propose a novel multi-stage LLM-driven framework: Stage 1 extracts atomic assertions (key facts with timing) from EMRs; Stage 2 constructs personal causal networks and synthesizes disease knowledge by clustering representative networks from an EMR corpus; Stage 3 generates tailored personal and standardized disease-specific questionnaires based on these structured representations. This framework overcomes limitations of direct methods by building explicit clinical knowledge. Evaluated on a real-world EMR dataset and validated by clinical experts, our method demonstrates superior performance in information coverage, diagnostic relevance, understandability, and generation time, highlighting its practical potential to enhance patient information collection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00570v1">Session-Based Recommendation with Validated and Enriched LLM Intents</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Session-based recommendation (SBR) aims to predict the next item for an anonymous user in a timely manner. However, SBR suffers from data sparsity due to the short and anonymous nature of sessions. Recently, an emerging line of work has explored inferring the underlying user intents of a session using large language models (LLMs), with the generated intents serving as auxiliary training signals to enhance SBR models. Despite its promise, this approach faces three key challenges: validating intent quality, incorporating session-level multi-intents, and complementing inevitable LLM failure cases. In this paper, we propose VELI4SBR, a two-stage framework that leverages Validated and Enriched LLM-generated Intents for SBR. In the first stage, we generate high-quality intents using a predict-and-correct loop that validates the informativeness of LLM-generated intents with a global intent pool to constrain the LLM's output space and reduce hallucination. In the second stage, we enhance the SBR model using the generated intents through a lightweight multi-intent prediction and fusion mechanism. Furthermore, we introduce a training strategy that compensates for LLM failures by inferring intents from inter-session behavioral similarities. Extensive experiments show that VELI4SBR outperforms state-of-the-art baselines while improving explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19693v2">AdaptiVocab: Enhancing LLM Efficiency in Focused Domains through Lightweight Vocabulary Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive versatility as general purpose models. However, their broad applicability comes at a high-cost computational overhead, particularly in auto-regressive decoding where each step requires a forward pass. In domain-specific settings, general-purpose capabilities are unnecessary and can be exchanged for efficiency. In this work, we take a novel perspective on domain adaptation, reducing latency and computational costs by adapting the vocabulary to focused domains of interest. We introduce AdaptiVocab, an end-to-end approach for vocabulary adaptation, designed to enhance LLM efficiency in low-resource domains. AdaptiVocab can be applied to any tokenizer and architecture, modifying the vocabulary by replacing tokens with domain-specific n-gram-based tokens, thereby reducing the number of tokens required for both input processing and output generation. AdaptiVocab initializes new n-token embeddings using an exponentially weighted combination of existing embeddings and employs a lightweight fine-tuning phase that can be efficiently performed on a single GPU. We evaluate two 7B LLMs across three niche domains, assessing efficiency, generation quality, and end-task performance. Our results show that AdaptiVocab reduces token usage by over 25% without compromising performance
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15170v3">From LLMs to MLLMs to Agents: A Survey of Emerging Paradigms in Jailbreak Attacks and Defenses within LLM Ecosystem</a></div>
    <div class="paper-meta">
      📅 2025-08-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly evolving from single-modal systems to multimodal LLMs and intelligent agents, significantly expanding their capabilities while introducing increasingly severe security risks. This paper presents a systematic survey of the growing complexity of jailbreak attacks and corresponding defense mechanisms within the expanding LLM ecosystem. We first trace the developmental trajectory from LLMs to MLLMs and Agents, highlighting the core security challenges emerging at each stage. Next, we categorize mainstream jailbreak techniques from both the attack impact and visibility perspectives, and provide a comprehensive analysis of representative attack methods, related datasets, and evaluation metrics. On the defense side, we organize existing strategies based on response timing and technical approach, offering a structured understanding of their applicability and implementation. Furthermore, we identify key limitations in existing surveys, such as insufficient attention to agent-specific security issues, the absence of a clear taxonomy for hybrid jailbreak methods, a lack of detailed analysis of experimental setups, and outdated coverage of recent advancements. To address these limitations, we provide an updated synthesis of recent work and outline future research directions in areas such as dataset construction, evaluation framework optimization, and strategy generalization. Our study seeks to enhance the understanding of jailbreak mechanisms and facilitate the advancement of more resilient and adaptive defense strategies in the context of ever more capable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.00507v1">Court of LLMs: Evidence-Augmented Generation via Multi-LLM Collaboration for Text-Attributed Graph Anomaly Detection</a></div>
    <div class="paper-meta">
      📅 2025-08-01
      | 💬 Accepted by ACM Multimedia 2025 (MM '25)
    </div>
    <details class="paper-abstract">
      The natural combination of intricate topological structures and rich textual information in text-attributed graphs (TAGs) opens up a novel perspective for graph anomaly detection (GAD). However, existing GAD methods primarily focus on designing complex optimization objectives within the graph domain, overlooking the complementary value of the textual modality, whose features are often encoded by shallow embedding techniques, such as bag-of-words or skip-gram, so that semantic context related to anomalies may be missed. To unleash the enormous potential of textual modality, large language models (LLMs) have emerged as promising alternatives due to their strong semantic understanding and reasoning capabilities. Nevertheless, their application to TAG anomaly detection remains nascent, and they struggle to encode high-order structural information inherent in graphs due to input length constraints. For high-quality anomaly detection in TAGs, we propose CoLL, a novel framework that combines LLMs and graph neural networks (GNNs) to leverage their complementary strengths. CoLL employs multi-LLM collaboration for evidence-augmented generation to capture anomaly-relevant contexts while delivering human-readable rationales for detected anomalies. Moreover, CoLL integrates a GNN equipped with a gating mechanism to adaptively fuse textual features with evidence while preserving high-order topological information. Extensive experiments demonstrate the superiority of CoLL, achieving an average improvement of 13.37% in AP. This study opens a new avenue for incorporating LLMs in advancing GAD.
    </details>
</div>
