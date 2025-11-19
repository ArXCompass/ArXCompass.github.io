# llm - 2025_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- Part 5
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2408.09235v3">Reference-Guided Verdict: LLMs-as-Judges in Automatic Evaluation of Free-Form QA</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Proceedings of the 9th Widening NLP Workshop
    </div>
    <details class="paper-abstract">
      The emergence of Large Language Models (LLMs) as chat assistants capable of generating human-like conversations has amplified the need for robust evaluation methods, particularly for open-ended tasks. Conventional metrics such as EM and F1, while useful, are inadequate for capturing the full semantics and contextual depth of such generative outputs. We propose a reference-guided verdict method that automates the evaluation process by leveraging multiple LLMs as judges. Through experiments on free-form question-answering tasks, we demonstrate that combining multiple models improves the reliability and accuracy of evaluations, especially in tasks where a single model may struggle. The results indicate a strong correlation with human evaluations, establishing the proposed method as a reliable alternative to traditional metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.20166v2">CyberSOCEval: Benchmarking LLMs Capabilities for Malware Analysis and Threat Intelligence Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Today's cyber defenders are overwhelmed by a deluge of security alerts, threat intelligence signals, and shifting business context, creating an urgent need for AI systems to enhance operational security work. While Large Language Models (LLMs) have the potential to automate and scale Security Operations Center (SOC) operations, existing evaluations do not fully assess the scenarios most relevant to real-world defenders. This lack of informed evaluation impacts both AI developers and those applying LLMs to SOC automation. Without clear insight into LLM performance in real-world security scenarios, developers lack a north star for development, and users cannot reliably select the most effective models. Meanwhile, malicious actors are using AI to scale cyber attacks, highlighting the need for open source benchmarks to drive adoption and community-driven improvement among defenders and model developers. To address this, we introduce CyberSOCEval, a new suite of open source benchmarks within CyberSecEval 4. CyberSOCEval includes benchmarks tailored to evaluate LLMs in two tasks: Malware Analysis and Threat Intelligence Reasoning--core defensive domains with inadequate coverage in current benchmarks. Our evaluations show that larger, more modern LLMs tend to perform better, confirming the training scaling laws paradigm. We also find that reasoning models leveraging test time scaling do not achieve the same boost as in coding and math, suggesting these models have not been trained to reason about cybersecurity analysis, and pointing to a key opportunity for improvement. Finally, current LLMs are far from saturating our evaluations, showing that CyberSOCEval presents a significant challenge for AI developers to improve cyber defense capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.08542v2">CLEV: LLM-Based Evaluation Through Lightweight Efficient Voting for Free-Form Question-Answering</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted to AACL 2025
    </div>
    <details class="paper-abstract">
      Evaluating free-form Question Answering (QA) remains a challenge due to its diverse and open-ended nature. Traditional automatic metrics fail to capture semantic equivalence or accommodate the variability of open-ended responses. Leveraging Large Language Models (LLMs) as evaluators offers a promising alternative due to their strong language understanding and instruction-following capabilities. We propose Consensus via Lightweight Efficient Voting (CLEV), which employs two primary LLMs as judges and invokes a third judge only in cases of disagreement. This approach prioritizes evaluation reliability while reducing unnecessary computational demands. Through experiments, including human evaluation, we demonstrate CLEV's ability to provide consistent, scalable, and resource-efficient assessments, establishing it as a robust framework for evaluating LLMs on free-form QA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07555v1">LLM Optimization Unlocks Real-Time Pairwise Reranking</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Efficiently reranking documents retrieved from information retrieval (IR) pipelines to enhance overall quality of Retrieval-Augmented Generation (RAG) system remains an important yet challenging problem. Recent studies have highlighted the importance of Large Language Models (LLMs) in reranking tasks. In particular, Pairwise Reranking Prompting (PRP) has emerged as a promising plug-and-play approach due to its usability and effectiveness. However, the inherent complexity of the algorithm, coupled with the high computational demands and latency incurred due to LLMs, raises concerns about its feasibility in real-time applications. To address these challenges, this paper presents a focused study on pairwise reranking, demonstrating that carefully applied optimization methods can significantly mitigate these issues. By implementing these methods, we achieve a remarkable latency reduction of up to 166 times, from 61.36 seconds to 0.37 seconds per query, with an insignificant drop in performance measured by Recall@k. Our study highlights the importance of design choices that were previously overlooked, such as using smaller models, limiting the reranked set, using lower precision, reducing positional bias with one-directional order inference, and restricting output tokens. These optimizations make LLM-based reranking substantially more efficient and feasible for latency-sensitive, real-world deployments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07364v1">Self-Evaluating LLMs for Multi-Step Tasks: Stepwise Confidence Estimation for Failure Detection</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted at NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling
    </div>
    <details class="paper-abstract">
      Reliability and failure detection of large language models (LLMs) is critical for their deployment in high-stakes, multi-step reasoning tasks. Prior work explores confidence estimation for self-evaluating LLM-scorer systems, with confidence scorers estimating the likelihood of errors in LLM responses. However, most methods focus on single-step outputs and overlook the challenges of multi-step reasoning. In this work, we extend self-evaluation techniques to multi-step tasks, testing two intuitive approaches: holistic scoring and step-by-step scoring. Using two multi-step benchmark datasets, we show that stepwise evaluation generally outperforms holistic scoring in detecting potential errors, with up to 15% relative increase in AUC-ROC. Our findings demonstrate that self-evaluating LLM systems provide meaningful confidence estimates in complex reasoning, improving their trustworthiness and providing a practical framework for failure detection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07318v1">When Bias Pretends to Be Truth: How Spurious Correlations Undermine Hallucination Detection in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Despite substantial advances, large language models (LLMs) continue to exhibit hallucinations, generating plausible yet incorrect responses. In this paper, we highlight a critical yet previously underexplored class of hallucinations driven by spurious correlations -- superficial but statistically prominent associations between features (e.g., surnames) and attributes (e.g., nationality) present in the training data. We demonstrate that these spurious correlations induce hallucinations that are confidently generated, immune to model scaling, evade current detection methods, and persist even after refusal fine-tuning. Through systematically controlled synthetic experiments and empirical evaluations on state-of-the-art open-source and proprietary LLMs (including GPT-5), we show that existing hallucination detection methods, such as confidence-based filtering and inner-state probing, fundamentally fail in the presence of spurious correlations. Our theoretical analysis further elucidates why these statistical biases intrinsically undermine confidence-based detection techniques. Our findings thus emphasize the urgent need for new approaches explicitly designed to address hallucinations caused by spurious correlations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.01934v2">Tool Zero: Training Tool-Augmented LLMs via Pure RL from Scratch</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 EMNLP 2025 finding
    </div>
    <details class="paper-abstract">
      Training tool-augmented LLMs has emerged as a promising approach to enhancing language models' capabilities for complex tasks. The current supervised fine-tuning paradigm relies on constructing extensive domain-specific datasets to train models. However, this approach often struggles to generalize effectively to unfamiliar or intricate tool-use scenarios. Recently, reinforcement learning (RL) paradigm can endow LLMs with superior reasoning and generalization abilities. In this work, we address a key question: Can the pure RL be used to effectively elicit a model's intrinsic reasoning capabilities and enhance the tool-agnostic generalization? We propose a dynamic generalization-guided reward design for rule-based RL, which progressively shifts rewards from exploratory to exploitative tool-use patterns. Based on this design, we introduce the Tool-Zero series models. These models are trained to enable LLMs to autonomously utilize general tools by directly scaling up RL from Zero models (i.e., base models without post-training). Experimental results demonstrate that our models achieve over 7% performance improvement compared to both SFT and RL-with-SFT models under the same experimental settings. These gains are consistently replicated across cross-dataset and intra-dataset evaluations, validating the effectiveness and robustness of our methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2507.02735v2">Meta SecAlign: A Secure Foundation LLM Against Prompt Injection Attacks</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Prompt injection attack has been listed as the top-1 security threat to LLM-integrated applications, which interact with external environment data for complex tasks. The untrusted data may contain an injected prompt trying to arbitrarily manipulate the system. Model-level prompt injection defenses have shown strong effectiveness, but are currently deployed into commercial-grade models in a closed-source manner. We believe open-source secure models are needed by the AI security community, where co-development of attacks and defenses through open research drives scientific progress in mitigating prompt injection attacks. To this end, we develop Meta SecAlign, the first fully open-source LLM with built-in model-level defense that achieves commercial-grade performance, powerful enough for complex agentic tasks. We provide complete details of our training recipe, an improved version of the SOTA SecAlign defense. We perform the most comprehensive evaluation to date on 9 utility benchmarks and 7 security benchmarks on general knowledge, instruction following, and agentic workflows. Results show that Meta SecAlign, despite being trained on generic instruction-tuning samples only, surprisingly confers security in unseen downstream tasks, including tool-calling and web-navigation, in addition to general instruction-following. Our best model -- Meta-SecAlign-70B -- establishes a new frontier of utility-security trade-off for open-source LLMs. Even compared to closed-course commercial models such as GPT-5, our model is much securer than most of them. Below are links for the code (https://github.com/facebookresearch/Meta_SecAlign), Meta-SecAlign-70B(https://huggingface.co/facebook/Meta-SecAlign-70B), and Meta-SecAlign-8B(https://huggingface.co/facebook/Meta-SecAlign-8B) models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2501.13677v3">HumorReject: Decoupling LLM Safety from Refusal Prefix via A Little Humor</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) commonly rely on explicit refusal prefixes for safety, making them vulnerable to prefix injection attacks. We introduce HumorReject, a novel data-driven approach that reimagines LLM safety by decoupling it from refusal prefixes through humor as an indirect refusal strategy. Rather than explicitly rejecting harmful instructions, HumorReject responds with contextually appropriate humor that naturally defuses potentially dangerous requests. Our approach effectively addresses common "over-defense" issues while demonstrating superior robustness against various attack vectors. Our findings suggest that improvements in training data design can be as important as the alignment algorithm itself in achieving effective LLM safety. The code and dataset are available at https://github.com/wooozihui/HumorReject.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2509.08736v2">ChemBOMAS: Accelerated BO in Chemistry with LLM-Enhanced Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Bayesian optimization (BO) is a powerful tool for scientific discovery in chemistry, yet its efficiency is often hampered by the sparse experimental data and vast search space. Here, we introduce ChemBOMAS: a large language model (LLM)-enhanced multi-agent system that accelerates BO through synergistic data- and knowledge-driven strategies. Firstly, the data-driven strategy involves an 8B-scale LLM regressor fine-tuned on a mere 1% labeled samples for pseudo-data generation, robustly initializing the optimization process. Secondly, the knowledge-driven strategy employs a hybrid Retrieval-Augmented Generation approach to guide LLM in dividing the search space while mitigating LLM hallucinations. An Upper Confidence Bound algorithm then identifies high-potential subspaces within this established partition. Across the LLM-refined subspaces and supported by LLM-generated data, BO achieves the improvement of effectiveness and efficiency. Comprehensive evaluations across multiple scientific benchmarks demonstrate that ChemBOMAS set a new state-of-the-art, accelerating optimization efficiency by up to 5-fold compared to baseline methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07229v1">LLMServingSim2.0: A Unified Simulator for Heterogeneous Hardware and Serving Techniques in LLM Infrastructure</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 4 pages, 3 figures
    </div>
    <details class="paper-abstract">
      This paper introduces LLMServingSim2.0, a system simulator designed for exploring heterogeneous hardware in large-scale LLM serving systems. LLMServingSim2.0 addresses two key limitations of its predecessor: (1) integrating hardware models into system-level simulators is non-trivial due to the lack of a clear abstraction, and (2) existing simulators support only a narrow subset of serving techniques, leaving no infrastructure that captures the breadth of approaches in modern LLM serving. To overcome these issues, LLMServingSim2.0 adopts trace-driven performance modeling, accompanied by an operator-level latency profiler, enabling the integration of new accelerators with a single command. It further embeds up-to-date serving techniques while exposing flexible interfaces for request routing, cache management, and scheduling policies. In a TPU case study, our profiler requires 18.5x fewer LoC and outperforms the predecessor's hardware-simulator integration, demonstrating LLMServingSim2.0's low-effort hardware extensibility. Our experiments further show that LLMServingSim2.0 reproduces GPU-based LLM serving with 1.9% error, while maintaining practical simulation time, making it a comprehensive platform for both hardware developers and LLM service providers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07223v1">NoteEx: Interactive Visual Context Manipulation for LLM-Assisted Exploratory Data Analysis in Computational Notebooks</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Computational notebooks have become popular for Exploratory Data Analysis (EDA), augmented by LLM-based code generation and result interpretation. Effective LLM assistance hinges on selecting informative context -- the minimal set of cells whose code, data, or outputs suffice to answer a prompt. As notebooks grow long and messy, users can lose track of the mental model of their analysis. They thus fail to curate appropriate contexts for LLM tasks, causing frustration and tedious prompt engineering. We conducted a formative study (n=6) that surfaced challenges in LLM context selection and mental model maintenance. Therefore, we introduce NoteEx, a JupyterLab extension that provides a semantic visualization of the EDA workflow, allowing analysts to externalize their mental model, specify analysis dependencies, and enable interactive selection of task-relevant contexts for LLMs. A user study (n=12) against a baseline shows that NoteEx improved mental model retention and context selection, leading to more accurate and relevant LLM responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07204v1">Evaluating Online Moderation Via LLM-Powered Counterfactual Simulations</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted for publication at AAAI Conference on Artificial Intelligence 2026
    </div>
    <details class="paper-abstract">
      Online Social Networks (OSNs) widely adopt content moderation to mitigate the spread of abusive and toxic discourse. Nonetheless, the real effectiveness of moderation interventions remains unclear due to the high cost of data collection and limited experimental control. The latest developments in Natural Language Processing pave the way for a new evaluation approach. Large Language Models (LLMs) can be successfully leveraged to enhance Agent-Based Modeling and simulate human-like social behavior with unprecedented degree of believability. Yet, existing tools do not support simulation-based evaluation of moderation strategies. We fill this gap by designing a LLM-powered simulator of OSN conversations enabling a parallel, counterfactual simulation where toxic behavior is influenced by moderation interventions, keeping all else equal. We conduct extensive experiments, unveiling the psychological realism of OSN agents, the emergence of social contagion phenomena and the superior effectiveness of personalized moderation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2504.07863v3">Robust Hallucination Detection in LLMs via Adaptive Token Selection</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Hallucinations in large language models (LLMs) pose significant safety concerns that impede their broader deployment. Recent research in hallucination detection has demonstrated that LLMs' internal representations contain truthfulness hints, which can be harnessed for detector training. However, the performance of these detectors is heavily dependent on the internal representations of predetermined tokens, fluctuating considerably when working on free-form generations with varying lengths and sparse distributions of hallucinated entities. To address this, we propose HaMI, a novel approach that enables robust detection of hallucinations through adaptive selection and learning of critical tokens that are most indicative of hallucinations. We achieve this robustness by an innovative formulation of the Hallucination detection task as Multiple Instance (HaMI) learning over token-level representations within a sequence, thereby facilitating a joint optimisation of token selection and hallucination detection on generation sequences of diverse forms. Comprehensive experimental results on four hallucination benchmarks show that HaMI significantly outperforms existing state-of-the-art approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07168v1">LEAD: LLM-enhanced Engine for Author Disambiguation</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Author Name Disambiguation (AND) is a long-standing challenge in bibliometrics and scientometrics, as name ambiguity undermines the accuracy of bibliographic databases and the reliability of research evaluation. This study addresses the problem of cross-source disambiguation by linking academic career records from CercaUniversità, the official registry of Italian academics, with author profiles in Scopus. We introduce LEAD (LLM-enhanced Engine for Author Disambiguation), a novel hybrid framework that combines semantic features extracted through Large Language Models (LLMs) with structural evidence derived from co-authorship and citation networks. Using a gold standard of 606 ambiguous cases, we compare five methods: (i) Label Spreading on co-authorship networks; (ii) Bibliographic Coupling on citation networks; (iii) a standalone LLM-based approach; (iv) an LLM-enriched configuration; and (v) the proposed hybrid pipeline. LEAD achieves the best performance (F1 = 96.7%, accuracy = 95.7%) with lower computational cost than full LLM models. Bibliographic Coupling emerges as the fastest and strongest single-source method. These findings demonstrate that integrating semantic and structural signals within a selective hybrid strategy offers a robust and scalable solution to cross-database author identification. Beyond the Italian case, this work highlights the potential of hybrid LLM-based methods to improve data quality and reliability in scientometric analyses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07166v1">AdaRec: Adaptive Recommendation with LLMs via Narrative Profiling and Dual-Channel Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      We propose AdaRec, a few-shot in-context learning framework that leverages large language models for an adaptive personalized recommendation. AdaRec introduces narrative profiling, transforming user-item interactions into natural language representations to enable unified task handling and enhance human readability. Centered on a bivariate reasoning paradigm, AdaRec employs a dual-channel architecture that integrates horizontal behavioral alignment, discovering peer-driven patterns, with vertical causal attribution, highlighting decisive factors behind user preferences. Unlike existing LLM-based approaches, AdaRec eliminates manual feature engineering through semantic representations and supports rapid cross-task adaptation with minimal supervision. Experiments on real ecommerce datasets demonstrate that AdaRec outperforms both machine learning models and LLM-based baselines by up to eight percent in few-shot settings. In zero-shot scenarios, it achieves up to a nineteen percent improvement over expert-crafted profiling, showing effectiveness for long-tail personalization with minimal interaction data. Furthermore, lightweight fine-tuning on synthetic data generated by AdaRec matches the performance of fully fine-tuned models, highlighting its efficiency and generalization across diverse tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2510.19670v2">CoSense-LLM: Semantics at the Edge with Cost- and Uncertainty-Aware Cloud-Edge Cooperation</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 19 pages,8 figures
    </div>
    <details class="paper-abstract">
      We present CoSense-LLM, an edge-first framework that turns continuous multimodal sensor streams (for example Wi-Fi CSI, IMU, audio, RFID, and lightweight vision) into compact, verifiable semantic tokens and coordinates with large language models under explicit latency, energy, bandwidth, and privacy constraints. CoSense-LLM has four parts: (i) SenseFusion, a lightweight encoder that aligns sensor embeddings with language and compresses them into short discrete code sequences; (ii) Edge-RAG, a local hybrid retrieval layer that grounds generation in site specific policies and notes; (iii) PromptRouter, a cost and uncertainty aware policy that selects edge only generation, edge plus retrieval, or compact cloud escalation; and (iv) Secure Execution, an auditable redaction path that enforces data minimization so raw waveforms never leave the device. The system works with modern serving optimizations, including paged or streaming KV caches, FlashAttention style kernels, speculative decoding, and quantized LoRA adapters, and supports on device personalization and federated updates under non IID drift. Across home, office, and clinic deployments, CoSense-LLM delivers grounded explanations while meeting tight service level objectives: it sustains sub second (p95) end to end latency on edge dominant paths, reduces inter tier token and bandwidth costs by preferring local retrieval grounded responses, and preserves privacy by transmitting only discrete codes and redacted metadata. Ablations show that Edge-RAG improves factual consistency and reduces contradictions, calibrated uncertainty enables selective abstention and controlled escalations, and KV plus decoding accelerators lower energy per decision. The results support an edge first design that treats semantics, privacy, and predictable latency as co equal goals for large model deployments in interference prone environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07126v1">Saliency Map-Guided Knowledge Discovery for Subclass Identification with LLM-Based Symbolic Approximations</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      This paper proposes a novel neuro-symbolic approach for sensor signal-based knowledge discovery, focusing on identifying latent subclasses in time series classification tasks. The approach leverages gradient-based saliency maps derived from trained neural networks to guide the discovery process. Multiclass time series classification problems are transformed into binary classification problems through label subsumption, and classifiers are trained for each of these to yield saliency maps. The input signals, grouped by predicted class, are clustered under three distinct configurations. The centroids of the final set of clusters are provided as input to an LLM for symbolic approximation and fuzzy knowledge graph matching to discover the underlying subclasses of the original multiclass problem. Experimental results on well-established time series classification datasets demonstrate the effectiveness of our saliency map-driven method for knowledge discovery, outperforming signal-only baselines in both clustering and subclass identification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07107v1">MENTOR: A Metacognition-Driven Self-Evolution Framework for Uncovering and Mitigating Implicit Risks in LLMs on Domain Tasks</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Ensuring the safety and value alignment of large language models (LLMs) is critical for their deployment. Current alignment efforts primarily target explicit risks such as bias, hate speech, and violence. However, they often fail to address deeper, domain-specific implicit risks and lack a flexible, generalizable framework applicable across diverse specialized fields. Hence, we proposed MENTOR: A MEtacognition-driveN self-evoluTion framework for uncOvering and mitigating implicit Risks in LLMs on Domain Tasks. To address the limitations of labor-intensive human evaluation, we introduce a novel metacognitive self-assessment tool. This enables LLMs to reflect on potential value misalignments in their responses using strategies like perspective-taking and consequential thinking. We also release a supporting dataset of 9,000 risk queries spanning education, finance, and management to enhance domain-specific risk identification. Subsequently, based on the outcomes of metacognitive reflection, the framework dynamically generates supplementary rule knowledge graphs that extend predefined static rule trees. This enables models to actively apply validated rules to future similar challenges, establishing a continuous self-evolution cycle that enhances generalization by reducing maintenance costs and inflexibility of static systems. Finally, we employ activation steering during inference to guide LLMs in following the rules, a cost-effective method to robustly enhance enforcement across diverse contexts. Experimental results show MENTOR's effectiveness: In defensive testing across three vertical domains, the framework substantially reduces semantic attack success rates, enabling a new level of implicit risk mitigation for LLMs. Furthermore, metacognitive assessment not only aligns closely with baseline human evaluators but also delivers more thorough and insightful analysis of LLMs value alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07099v1">E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted to NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advancements in speech synthesis technology have enriched our daily lives, with high-quality and human-like audio widely adopted across real-world applications. However, malicious exploitation like voice-cloning fraud poses severe security risks. Existing defense techniques struggle to address the production large language model (LLM)-based speech synthesis. While previous studies have considered the protection for fine-tuning synthesizers, they assume manually annotated transcripts. Given the labor intensity of manual annotation, end-to-end (E2E) systems leveraging automatic speech recognition (ASR) to generate transcripts are becoming increasingly prevalent, e.g., voice cloning via commercial APIs. Therefore, this E2E speech synthesis also requires new security mechanisms. To tackle these challenges, we propose E2E-VGuard, a proactive defense framework for two emerging threats: (1) production LLM-based speech synthesis, and (2) the novel attack arising from ASR-driven E2E scenarios. Specifically, we employ the encoder ensemble with a feature extractor to protect timbre, while ASR-targeted adversarial examples disrupt pronunciation. Moreover, we incorporate the psychoacoustic model to ensure perturbative imperceptibility. For a comprehensive evaluation, we test 16 open-source synthesizers and 3 commercial APIs across Chinese and English datasets, confirming E2E-VGuard's effectiveness in timbre and pronunciation protection. Real-world deployment validation is also conducted. Our code and demo page are available at https://wxzyd123.github.io/e2e-vguard/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07086v1">LLM Driven Processes to Foster Explainable AI</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      We present a modular, explainable LLM-agent pipeline for decision support that externalizes reasoning into auditable artifacts. The system instantiates three frameworks: Vester's Sensitivity Model (factor set, signed impact matrix, systemic roles, feedback loops); normal-form games (strategies, payoff matrix, equilibria); and sequential games (role-conditioned agents, tree construction, backward induction), with swappable modules at every step. LLM components (default: GPT-5) are paired with deterministic analyzers for equilibria and matrix-based role classification, yielding traceable intermediates rather than opaque outputs. In a real-world logistics case (100 runs), mean factor alignment with a human baseline was 55.5\% over 26 factors and 62.9\% on the transport-core subset; role agreement over matches was 57\%. An LLM judge using an eight-criterion rubric (max 100) scored runs on par with a reconstructed human baseline. Configurable LLM pipelines can thus mimic expert workflows with transparent, inspectable steps.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07083v1">Increasing AI Explainability by LLM Driven Standard Processes</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      This paper introduces an approach to increasing the explainability of artificial intelligence (AI) systems by embedding Large Language Models (LLMs) within standardized analytical processes. While traditional explainable AI (XAI) methods focus on feature attribution or post-hoc interpretation, the proposed framework integrates LLMs into defined decision models such as Question-Option-Criteria (QOC), Sensitivity Analysis, Game Theory, and Risk Management. By situating LLM reasoning within these formal structures, the approach transforms opaque inference into transparent and auditable decision traces. A layered architecture is presented that separates the reasoning space of the LLM from the explainable process space above it. Empirical evaluations show that the system can reproduce human-level decision logic in decentralized governance, systems analysis, and strategic reasoning contexts. The results suggest that LLM-driven standard processes provide a foundation for reliable, interpretable, and verifiable AI-supported decision making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07074v1">Importance-Aware Data Selection for Efficient LLM Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted by AAAI 2026 Oral
    </div>
    <details class="paper-abstract">
      Instruction tuning plays a critical role in enhancing the performance and efficiency of Large Language Models (LLMs). Its success depends not only on the quality of the instruction data but also on the inherent capabilities of the LLM itself. Some studies suggest that even a small amount of high-quality data can achieve instruction fine-tuning results that are on par with, or even exceed, those from using a full-scale dataset. However, rather than focusing solely on calculating data quality scores to evaluate instruction data, there is a growing need to select high-quality data that maximally enhances the performance of instruction tuning for a given LLM. In this paper, we propose the Model Instruction Weakness Value (MIWV) as a novel metric to quantify the importance of instruction data in enhancing model's capabilities. The MIWV metric is derived from the discrepancies in the model's responses when using In-Context Learning (ICL), helping identify the most beneficial data for enhancing instruction tuning performance. Our experimental results demonstrate that selecting only the top 1\% of data based on MIWV can outperform training on the full dataset. Furthermore, this approach extends beyond existing research that focuses on data quality scoring for data selection, offering strong empirical evidence supporting the effectiveness of our proposed method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07070v1">RedOne 2.0: Rethinking Domain-specific LLM Post-Training in Social Networking Services</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      As a key medium for human interaction and information exchange, social networking services (SNS) pose unique challenges for large language models (LLMs): heterogeneous workloads, fast-shifting norms and slang, and multilingual, culturally diverse corpora that induce sharp distribution shift. Supervised fine-tuning (SFT) can specialize models but often triggers a ``seesaw'' between in-distribution gains and out-of-distribution robustness, especially for smaller models. To address these challenges, we introduce RedOne 2.0, an SNS-oriented LLM trained with a progressive, RL-prioritized post-training paradigm designed for rapid and stable adaptation. The pipeline consist in three stages: (1) Exploratory Learning on curated SNS corpora to establish initial alignment and identify systematic weaknesses; (2) Targeted Fine-Tuning that selectively applies SFT to the diagnosed gaps while mixing a small fraction of general data to mitigate forgetting; and (3) Refinement Learning that re-applies RL with SNS-centric signals to consolidate improvements and harmonize trade-offs across tasks. Across various tasks spanning three categories, our 4B scale model delivers an average improvements about 2.41 over the 7B sub-optimal baseline. Additionally, RedOne 2.0 achieves average performance lift about 8.74 from the base model with less than half the data required by SFT-centric method RedOne, evidencing superior data efficiency and stability at compact scales. Overall, RedOne 2.0 establishes a competitive, cost-effective baseline for domain-specific LLMs in SNS scenario, advancing capability without sacrificing robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07044v1">Evaluating LLMs for Anxiety, Depression, and Stress Detection Evaluating Large Language Models for Anxiety, Depression, and Stress Detection: Insights into Prompting Strategies and Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Mental health disorders affect over one-fifth of adults globally, yet detecting such conditions from text remains challenging due to the subtle and varied nature of symptom expression. This study evaluates multiple approaches for mental health detection, comparing Large Language Models (LLMs) such as Llama and GPT with classical machine learning and transformer-based architectures including BERT, XLNet, and Distil-RoBERTa. Using the DAIC-WOZ dataset of clinical interviews, we fine-tuned models for anxiety, depression, and stress classification and applied synthetic data generation to mitigate class imbalance. Results show that Distil-RoBERTa achieved the highest F1 score (0.883) for GAD-2, while XLNet outperformed others on PHQ tasks (F1 up to 0.891). For stress detection, a zero-shot synthetic approach (SD+Zero-Shot-Basic) reached an F1 of 0.884 and ROC AUC of 0.886. Findings demonstrate the effectiveness of transformer-based models and highlight the value of synthetic data in improving recall and generalization. However, careful calibration is required to prevent precision loss. Overall, this work emphasizes the potential of combining advanced language models and data augmentation to enhance automated mental health assessment from text.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07035v1">GoCkpt: Gradient-Assisted Multi-Step overlapped Checkpointing for Efficient LLM Training</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 12 pages, 10 figures, 1 tables
    </div>
    <details class="paper-abstract">
      The accuracy of large language models (LLMs) improves with increasing model size, but increasing model complexity also poses significant challenges to training stability. Periodic checkpointing is a key mechanism for fault recovery and is widely used in LLM training. However, traditional checkpointing strategies often pause or delay GPU computation during checkpoint saving for checkpoint GPU-CPU transfer, resulting in significant training interruptions and reduced training throughput. To address this issue, we propose GoCkpt, a method to overlap checkpoint saving with multiple training steps and restore the final checkpoint on the CPU. We transfer the checkpoint across multiple steps, each step transfers part of the checkpoint state, and we transfer some of the gradient data used for parameter updates. After the transfer is complete, each partial checkpoint state is updated to a consistent version on the CPU, thus avoiding the checkpoint state inconsistency problem caused by transferring checkpoints across multiple steps. Furthermore, we introduce a transfer optimization strategy to maximize GPU-CPU bandwidth utilization and SSD persistence throughput. This dual optimization overlapping saves across steps and maximizing I/O efficiency significantly reduces invalid training time. Experimental results show that GoCkpt can increase training throughput by up to 38.4% compared to traditional asynchronous checkpoint solutions in the industry. We also find that GoCkpt can reduce training interruption time by 86.7% compared to the state-of-the-art checkpoint transfer methods, which results in a 4.8% throughput improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.14496v2">LLM-Powered Swarms: A New Frontier or a Conceptual Stretch?</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 This is the author's version of a paper submitted to IEEE Intelligent Systems. 2 Tables, 2 Figures
    </div>
    <details class="paper-abstract">
      Swarm intelligence describes how simple, decentralized agents can collectively produce complex behaviors. Recently, the concept of swarming has been extended to large language model (LLM)-powered systems, such as OpenAI's Swarm (OAS) framework, where agents coordinate through natural language prompts. This paper evaluates whether such systems capture the fundamental principles of classical swarm intelligence: decentralization, simplicity, emergence, and scalability. Using OAS, we implement and compare classical and LLM-based versions of two well-established swarm algorithms: Boids and Ant Colony Optimization. Results indicate that while LLM-powered swarms can emulate swarm-like dynamics, they are constrained by substantial computational overhead. For instance, our LLM-based Boids simulation required roughly 300x more computation time than its classical counterpart, highlighting current limitations in applying LLM-driven swarms to real-time systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07033v1">Uncovering Pretraining Code in LLMs: A Syntax-Aware Attribution Approach</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Paper has been accepted by AAAI 2026
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly capable, concerns over the unauthorized use of copyrighted and licensed content in their training data have grown, especially in the context of code. Open-source code, often protected by open source licenses (e.g, GPL), poses legal and ethical challenges when used in pretraining. Detecting whether specific code samples were included in LLM training data is thus critical for transparency, accountability, and copyright compliance. We propose SynPrune, a syntax-pruned membership inference attack method tailored for code. Unlike prior MIA approaches that treat code as plain text, SynPrune leverages the structured and rule-governed nature of programming languages. Specifically, it identifies and excludes consequent tokens that are syntactically required and not reflective of authorship, from attribution when computing membership scores. Experimental results show that SynPrune consistently outperforms the state-of-the-arts. Our method is also robust across varying function lengths and syntax categories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.07017v1">Benchmarking LLMs for Fine-Grained Code Review with Enriched Context in Practice</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Code review is a cornerstone of software quality assurance, and recent advances in Large Language Models (LLMs) have shown promise in automating this process. However, existing benchmarks for LLM-based code review face three major limitations. (1) Lack of semantic context: most benchmarks provide only code diffs without textual information such as issue descriptions, which are crucial for understanding developer intent. (2) Data quality issues: without rigorous validation, many samples are noisy-e.g., reviews on outdated or irrelevant code-reducing evaluation reliability. (3) Coarse granularity: most benchmarks operate at the file or commit level, overlooking the fine-grained, line-level reasoning essential for precise review. We introduce ContextCRBench, a high-quality, context-rich benchmark for fine-grained LLM evaluation in code review. Our construction pipeline comprises: (1) Raw Data Crawling, collecting 153.7K issues and pull requests from top-tier repositories; (2) Comprehensive Context Extraction, linking issue-PR pairs for textual context and extracting the full surrounding function or class for code context; and (3) Multi-stage Data Filtering, combining rule-based and LLM-based validation to remove outdated, malformed, or low-value samples, resulting in 67,910 context-enriched entries. ContextCRBench supports three evaluation scenarios aligned with the review workflow: (1) hunk-level quality assessment, (2) line-level defect localization, and (3) line-level comment generation. Evaluating eight leading LLMs (four closed-source and four open-source) reveals that textual context yields greater performance gains than code context alone, while current LLMs remain far from human-level review ability. Deployed at ByteDance, ContextCRBench drives a self-evolving code review system, improving performance by 61.98% and demonstrating its robustness and industrial utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.17631v3">Time-Prompt: Integrated Heterogeneous Prompts for Unlocking LLMs in Time Series Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Time series forecasting aims to model temporal dependencies among variables for future state inference, holding significant importance and widespread applications in real-world scenarios. Although deep learning-based methods have achieved remarkable progress, they still exhibit suboptimal performance in long-term forecasting. Recent research demonstrates that large language models (LLMs) achieve promising performance in time series forecasting, but this progress is still met with skepticism about whether LLMs are truly useful for this task. To address this, we propose Time-Prompt, a framework for activating LLMs for time series forecasting. Specifically, we first construct a unified prompt paradigm with learnable soft prompts to guide the LLM's behavior and textualized hard prompts to enhance the time series representations. Second, to enhance LLM' comprehensive understanding of the forecasting task, we design a semantic space embedding and cross-modal alignment module to achieve fusion of temporal and textual data. Finally, we efficiently fine-tune the LLM's parameters using time series data. Furthermore, we focus on carbon emissions, aiming to provide a modest contribution to global carbon neutrality. Comprehensive evaluations on 6 public datasets and 3 carbon emission datasets demonstrate that Time-Prompt is a powerful framework for time series forecasting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2505.24826v2">LegalEval-Q: A New Benchmark for The Quality Evaluation of LLM-Generated Legal Text</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 10 pages, 11 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly used in legal applications, current evaluation benchmarks tend to focus mainly on factual accuracy while largely neglecting important linguistic quality aspects such as clarity, coherence, and terminology. To address this gap, we propose three steps: First, we develop a regression model to evaluate the quality of legal texts based on clarity, coherence, and terminology. Second, we create a specialized set of legal questions. Third, we analyze 49 LLMs using this evaluation framework. Our analysis identifies three key findings: First, model quality levels off at 14 billion parameters, with only a marginal improvement of $2.7\%$ noted at 72 billion parameters. Second, engineering choices such as quantization and context length have a negligible impact, as indicated by statistical significance thresholds above 0.016. Third, reasoning models consistently outperform base architectures. A significant outcome of our research is the release of a ranking list and Pareto analysis, which highlight the Qwen3 series as the optimal choice for cost-performance tradeoffs. This work not only establishes standardized evaluation protocols for legal LLMs but also uncovers fundamental limitations in current training data refinement approaches. Code and models are available at: https://github.com/lyxx3rd/LegalEval-Q.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.16040v2">Evaluating Test-Time Scaling LLMs for Legal Reasoning: OpenAI o1, DeepSeek-R1, and Beyond</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 23 pages, Published in Findings of the Association for Computational Linguistics: EMNLP 2025
    </div>
    <details class="paper-abstract">
      Recent advances in test-time scaling of large language models (LLMs), exemplified by DeepSeek-R1 and OpenAI's o1, show that extending the chain of thought during inference can significantly improve general reasoning performance. However, the impact of this paradigm on legal reasoning remains insufficiently explored. To address this gap, we present the first systematic evaluation of 12 LLMs, including both reasoning-focused and general-purpose models, across 17 Chinese and English legal tasks spanning statutory and case-law traditions. In addition, we curate a bilingual chain-of-thought dataset for legal reasoning through distillation from DeepSeek-R1 and develop Legal-R1, an open-source model specialized for the legal domain. Experimental results show that Legal-R1 delivers competitive performance across diverse tasks. DeepSeek-R1 exhibits clear advantages in Chinese legal reasoning, while OpenAI's o1 achieves comparable results on English tasks. We further conduct a detailed error analysis, which reveals recurring issues such as outdated legal knowledge, limited capacity for legal interpretation, and susceptibility to factual hallucinations. These findings delineate the main obstacles confronting legal-domain LLMs and suggest promising directions for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2511.06890v1">EduGuardBench: A Holistic Benchmark for Evaluating the Pedagogical Fidelity and Adversarial Safety of LLMs as Simulated Teachers</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 22 pages, 9 figures, accepted by AAAI2026 as oral paper
    </div>
    <details class="paper-abstract">
      Large Language Models for Simulating Professions (SP-LLMs), particularly as teachers, are pivotal for personalized education. However, ensuring their professional competence and ethical safety is a critical challenge, as existing benchmarks fail to measure role-playing fidelity or address the unique teaching harms inherent in educational scenarios. To address this, we propose EduGuardBench, a dual-component benchmark. It assesses professional fidelity using a Role-playing Fidelity Score (RFS) while diagnosing harms specific to the teaching profession. It also probes safety vulnerabilities using persona-based adversarial prompts targeting both general harms and, particularly, academic misconduct, evaluated with metrics including Attack Success Rate (ASR) and a three-tier Refusal Quality assessment. Our extensive experiments on 14 leading models reveal a stark polarization in performance. While reasoning-oriented models generally show superior fidelity, incompetence remains the dominant failure mode across most models. The adversarial tests uncovered a counterintuitive scaling paradox, where mid-sized models can be the most vulnerable, challenging monotonic safety assumptions. Critically, we identified a powerful Educational Transformation Effect: the safest models excel at converting harmful requests into teachable moments by providing ideal Educational Refusals. This capacity is strongly negatively correlated with ASR, revealing a new dimension of advanced AI safety. EduGuardBench thus provides a reproducible framework that moves beyond siloed knowledge tests toward a holistic assessment of professional, ethical, and pedagogical alignment, uncovering complex dynamics essential for deploying trustworthy AI in education. See https://github.com/YL1N/EduGuardBench for Materials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2508.10027v3">LLMCARE: early detection of cognitive impairment via transformer models enhanced by LLM-generated synthetic data</a></div>
    <div class="paper-meta">
      📅 2025-11-10
    </div>
    <details class="paper-abstract">
      Alzheimer's disease and related dementias(ADRD) affect nearly five million older adults in the United States, yet more than half remain undiagnosed. Speech-based natural language processing(NLP) offers a scalable approach for detecting early cognitive decline through subtle linguistic markers that may precede clinical diagnosis. This study develops and evaluates a speech-based screening pipeline integrating transformer embeddings with handcrafted linguistic features, synthetic augmentation using large language models(LLMs), and benchmarking of unimodal and multimodal classifiers. External validation assessed generalizability to a MCI-only cohort. Transcripts were drawn from the ADReSSo 2021 benchmark dataset(n=237, Pitt Corpus) and the DementiaBank Delaware corpus(n=205, MCI vs. controls). Ten transformer models were tested under three fine-tuning strategies. A late-fusion model combined embeddings from the top transformer with 110 linguistic features. Five LLMs(LLaMA8B/70B, MedAlpaca7B, Ministral8B,GPT-4o) generated label-conditioned synthetic speech for augmentation, and three multimodal LLMs(GPT-4o,Qwen-Omni,Phi-4) were evaluated in zero-shot and fine-tuned modes. On ADReSSo, the fusion model achieved F1=83.3(AUC=89.5), outperforming transformer-only and linguistic baselines. MedAlpaca7B augmentation(2x) improved F1=85.7, though larger scales reduced gains. Fine-tuning boosted unimodal LLMs(MedAlpaca7B F1=47.7=>78.7), while multimodal models performed lower (Phi-4=71.6;GPT-4o=67.6). On Delaware, the fusion plus 1x MedAlpaca7B model achieved F1=72.8(AUC=69.6). Integrating transformer and linguistic features enhances ADRD detection. LLM-based augmentation improves data efficiency but yields diminishing returns, while current multimodal models remain limited. Validation on an independent MCI cohort supports the pipeline's potential for scalable, clinically relevant early screening.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2411.19638v2">LLM Teacher-Student Framework for Text Classification With No Manually Annotated Data: A Case Study in IPTC News Topic Classification</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 This work has been accepted and published in the IEEE Access journal. This arXiv version is retained for archival purposes. Readers should use and cite the IEEE Access Version available at https://ieeexplore.ieee.org/document/10900365
    </div>
    <details class="paper-abstract">
      With the ever-increasing number of news stories available online, classifying them by topic, regardless of the language they are written in, has become crucial for enhancing readers' access to relevant content. To address this challenge, we propose a teacher-student framework based on large language models (LLMs) for developing multilingual news topic classification models of reasonable size with no need for manual data annotation. The framework employs a Generative Pretrained Transformer (GPT) model as the teacher model to develop a news topic training dataset through automatic annotation of 20,000 news articles in Slovenian, Croatian, Greek, and Catalan. Articles are classified into 17 main categories from the Media Topic schema, developed by the International Press Telecommunications Council (IPTC). The teacher model exhibits high zero-shot performance in all four languages. Its agreement with human annotators is comparable to that between the human annotators themselves. To mitigate the computational limitations associated with the requirement of processing millions of texts daily, smaller BERT-like student models are fine-tuned on the GPT-annotated dataset. These student models achieve high performance comparable to the teacher model. Furthermore, we explore the impact of the training data size on the performance of the student models and investigate their monolingual, multilingual, and zero-shot cross-lingual capabilities. The findings indicate that student models can achieve high performance with a relatively small number of training instances, and demonstrate strong zero-shot cross-lingual abilities. Finally, we publish the best-performing news topic classifier, enabling multilingual classification with the top-level categories of the IPTC Media Topic schema.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2506.00708v3">DrKGC: Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion across General and Biomedical Domains</a></div>
    <div class="paper-meta">
      📅 2025-11-10
      | 💬 Accepted at EMNLP 2025 Findings
    </div>
    <details class="paper-abstract">
      Knowledge graph completion (KGC) aims to predict missing triples in knowledge graphs (KGs) by leveraging existing triples and textual information. Recently, generative large language models (LLMs) have been increasingly employed for graph tasks. However, current approaches typically encode graph context in textual form, which fails to fully exploit the potential of LLMs for perceiving and reasoning about graph structures. To address this limitation, we propose DrKGC (Dynamic Subgraph Retrieval-Augmented LLMs for Knowledge Graph Completion). DrKGC employs a flexible lightweight model training strategy to learn structural embeddings and logical rules within the KG. It then leverages a novel bottom-up graph retrieval method to extract a subgraph for each query guided by the learned rules. Finally, a graph convolutional network (GCN) adapter uses the retrieved subgraph to enhance the structural embeddings, which are then integrated into the prompt for effective LLM fine-tuning. Experimental results on two general domain benchmark datasets and two biomedical datasets demonstrate the superior performance of DrKGC. Furthermore, a realistic case study in the biomedical domain highlights its interpretability and practical utility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.13557v5">MACO: A Multi-Agent LLM-Based Hardware/Software Co-Design Framework for CGRAs</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Coarse-grained Reconfigurable Arrays (CGRAs) are a promising computing architecture that can deliver high-performance, energy-efficient acceleration across diverse domains. By supporting reconfiguration at the functional unit level, CGRAs efficiently adapt to varying computational patterns and optimize resource utilization. However, designing CGRAs is highly challenging due to the vast design space, independent architectural parameters, and the time-consuming nature of manual design. Fortunately, the rapid advancement of large language models (LLMs) presents new opportunities to automate this process. In this work, we propose MACO-- an open-source multi-agent LLM-based framework for Hardware/Software (HW/SW) co-design of CGRAs. The framework employs LLM reasoning to generate CGRAs across four stages: HW/SW co-design, Design error correction, Best design selection, and Evaluation & Feedback. Furthermore, MACO iteratively optimizes the generated CGRAs, leveraging agent reasoning and feedback to achieve higher PPA (that is, power, performance, and area) design points for a given domain. In addition, we introduce an LLM self-learning mechanism that employs LLM-driven decision making to select the optimal CGRA to accelerate the design process. We evaluate the framework with state-of-the-art LLM-based methods and manual CGRA design, in terms of performance, power consumption, and area. Experimental results show that MACO efficiently generates high-quality CGRA architectures, significantly reducing manual design effort and demonstrating the potential of our framework for real-world CGRA design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21700v3">XBreaking: Understanding how LLMs security alignment can be broken</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Large Language Models are fundamental actors in the modern IT landscape dominated by AI solutions. However, security threats associated with them might prevent their reliable adoption in critical application scenarios such as government organizations and medical institutions. For this reason, commercial LLMs typically undergo a sophisticated censoring mechanism to eliminate any harmful output they could possibly produce. These mechanisms maintain the integrity of LLM alignment by guaranteeing that the models respond safely and ethically. In response to this, attacks on LLMs are a significant threat to such protections, and many previous approaches have already demonstrated their effectiveness across diverse domains. Existing LLM attacks mostly adopt a generate-and-test strategy to craft malicious input. To improve the comprehension of censoring mechanisms and design a targeted attack, we propose an Explainable-AI solution that comparatively analyzes the behavior of censored and uncensored models to derive unique exploitable alignment patterns. Then, we propose XBreaking, a novel approach that exploits these unique patterns to break the security and alignment constraints of LLMs by targeted noise injection. Our thorough experimental campaign returns important insights about the censoring mechanisms and demonstrates the effectiveness and performance of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.05311v1">Cleaning Maintenance Logs with LLM Agents for Improved Predictive Maintenance</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Economic constraints, limited availability of datasets for reproducibility and shortages of specialized expertise have long been recognized as key challenges to the adoption and advancement of predictive maintenance (PdM) in the automotive sector. Recent progress in large language models (LLMs) presents an opportunity to overcome these barriers and speed up the transition of PdM from research to industrial practice. Under these conditions, we explore the potential of LLM-based agents to support PdM cleaning pipelines. Specifically, we focus on maintenance logs, a critical data source for training well-performing machine learning (ML) models, but one often affected by errors such as typos, missing fields, near-duplicate entries, and incorrect dates. We evaluate LLM agents on cleaning tasks involving six distinct types of noise. Our findings show that LLMs are effective at handling generic cleaning tasks and offer a promising foundation for future industrial applications. While domain-specific errors remain challenging, these results highlight the potential for further improvements through specialized training and enhanced agentic capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.05269v1">TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted at ICML 2025 MAS Workshop. This version includes additional experiments and analysis
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated strong capabilities as autonomous agents through tool use, planning, and decision-making abilities, leading to their widespread adoption across diverse tasks. As task complexity grows, multi-agent LLM systems are increasingly used to solve problems collaboratively. However, safety and security of these systems remains largely under-explored. Existing benchmarks and datasets predominantly focus on single-agent settings, failing to capture the unique vulnerabilities of multi-agent dynamics and co-ordination. To address this gap, we introduce $\textbf{T}$hreats and $\textbf{A}$ttacks in $\textbf{M}$ulti-$\textbf{A}$gent $\textbf{S}$ystems ($\textbf{TAMAS}$), a benchmark designed to evaluate the robustness and safety of multi-agent LLM systems. TAMAS includes five distinct scenarios comprising 300 adversarial instances across six attack types and 211 tools, along with 100 harmless tasks. We assess system performance across ten backbone LLMs and three agent interaction configurations from Autogen and CrewAI frameworks, highlighting critical challenges and failure modes in current multi-agent deployments. Furthermore, we introduce Effective Robustness Score (ERS) to assess the tradeoff between safety and task effectiveness of these frameworks. Our findings show that multi-agent systems are highly vulnerable to adversarial attacks, underscoring the urgent need for stronger defenses. TAMAS provides a foundation for systematically studying and improving the safety of multi-agent LLM systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14450v2">LLM4FaaS: No-Code Application Development using LLMs and FaaS</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted for publication in 2025 IEEE/ACM 18th International Conference on Utility and Cloud Computing
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show great capabilities in generating code from natural language descriptions, bringing programming power closer to non-technical users. However, their lack of expertise in operating the generated code remains a key barrier to realizing customized applications. Function-as-a-Service (FaaS) platforms offer a high level of abstraction for code execution and deployment, allowing users to run LLM-generated code without requiring technical expertise or incurring operational overhead. In this paper, we present LLM4FaaS, a no-code application development approach that integrates LLMs and FaaS platforms to enable non-technical users to build and run customized applications using only natural language. By deploying LLM-generated code through FaaS, LLM4FaaS abstracts away infrastructure management and boilerplate code generation. We implement a proof-of-concept prototype based on an open-source FaaS platform, and evaluate it using real prompts from non-technical users. Experiments with GPT-4o show that LLM4FaaS can automatically build and deploy code in 71.47% of cases, outperforming a non-FaaS baseline at 43.48% and an existing LLM-based platform at 14.55%, narrowing the gap to human performance at 88.99%. Further analysis of code quality, programming language diversity, latency, and consistency demonstrates a balanced performance in terms of efficiency, maintainability and availability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17086v4">Mind the Blind Spots: A Focus-Level Evaluation Framework for LLM Reviews</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 EMNLP 2025 Oral
    </div>
    <details class="paper-abstract">
      Peer review underpins scientific progress, but it is increasingly strained by reviewer shortages and growing workloads. Large Language Models (LLMs) can automatically draft reviews now, but determining whether LLM-generated reviews are trustworthy requires systematic evaluation. Researchers have evaluated LLM reviews at either surface-level (e.g., BLEU and ROUGE) or content-level (e.g., specificity and factual accuracy). Yet it remains uncertain whether LLM-generated reviews attend to the same critical facets that human experts weigh -- the strengths and weaknesses that ultimately drive an accept-or-reject decision. We introduce a focus-level evaluation framework that operationalizes the focus as a normalized distribution of attention across predefined facets in paper reviews. Based on the framework, we developed an automatic focus-level evaluation pipeline based on two sets of facets: target (e.g., problem, method, and experiment) and aspect (e.g., validity, clarity, and novelty), leveraging 676 paper reviews (https://figshare.com/s/d5adf26c802527dd0f62) from OpenReview that consists of 3,657 strengths and weaknesses identified from human experts. The comparison of focus distributions between LLMs and human experts showed that the off-the-shelf LLMs consistently have a more biased focus towards examining technical validity while significantly overlooking novelty assessment when criticizing papers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.13142v3">Holistic Evaluation of Multimodal LLMs on Spatial Intelligence</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Codebase: https://github.com/EvolvingLMMs-Lab/EASI/
    </div>
    <details class="paper-abstract">
      Multimodal models have achieved remarkable progress in recent years. Nevertheless, they continue to exhibit notable limitations in spatial understanding and reasoning, the very capability that anchors artificial general intelligence in the physical world. With the recent release of GPT-5, allegedly the most powerful AI model to date, it is timely to examine where the leading models (GPT, Gemini, Grok, Seed, Qwen, and Intern) stand on the path toward spatial intelligence. We thus propose EASI for holistic Evaluation of multimodAl LLMs on Spatial Intelligence. EASI conceptualizes a comprehensive taxonomy of spatial tasks that unifies existing benchmarks and a standardized protocol for the fair evaluation of state-of-the-art proprietary and open-source models. In this report, we conduct the study across eight key benchmarks, at a cost exceeding ten billion total tokens. Our empirical study then reveals that (1) GPT-5 demonstrates unprecedented strength in spatial intelligence (SI), yet (2) still falls short of human performance significantly across a broad spectrum of SI-tasks. Moreover, we (3) show that SI-tasks expose greater model capability deficiency than non-SI tasks, to the extent that (4) proprietary models do not exhibit a decisive advantage when facing the most difficult ones. In addition, we conduct a qualitative evaluation across a diverse set of scenarios that are intuitive for humans, yet fail even the most advanced multimodal models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18469v6">Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted to NAACL 2025 Main (Oral)
    </div>
    <details class="paper-abstract">
      Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.25441v2">Grounded in Reality: Learning and Deploying Proactive LLM from Offline Logs</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 27 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel as passive responders, but teaching them to be proactive, goal-oriented partners, a critical capability in high-stakes domains, remains a major challenge. Current paradigms either myopically optimize single-turn attributes or rely on brittle, high-cost user simulators, creating a persistent ``reality gap''. To bridge this gap, we introduce \texttt{Learn-to-Ask}, a general, simulator-free framework for learning and deploying proactive dialogue agents \textit{directly from offline expert data}, bypassing the need to model complex user dynamics. Our key insight is to reframe the offline policy learning problem by leveraging the \textbf{observed future} of each expert trajectory. This allows us to infer a dense, turn-by-turn reward signal grounded in the expert's revealed strategy, decomposing the intractable long-horizon problem into a series of supervised learning tasks, and training a policy to output a structured \texttt{(action, state_assessment)} tuple, governing both \textbf{what to ask} and, crucially, \textbf{when to stop}. To ensure reward fidelity, our Automated Grader Calibration pipeline systematically purges noise from the LLM-based reward model with minimal human supervision. Empirically, we demonstrate the efficacy of \texttt{Learn-to-Ask} in a real-world medical dataset, using LLMs of varying sizes up to 32B. Our approach culminates in the successful deployment of LLMs into a live, large-scale online AI service. In rigorous in-house evaluations, our model was launched and achieved performance even superior to human experts, proving our framework's ability to translate offline data into tangible, real-world impact. We hope this work provides a practical and economically viable blueprint for transforming passive LLMs into proactive, goal-oriented LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16447v2">Boardwalk: Towards a Framework for Creating Board Games with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Presented at SBGames 2025
    </div>
    <details class="paper-abstract">
      Implementing board games in code can be a time-consuming task. However, Large Language Models (LLMs) have been proven effective at generating code for domain-specific tasks with simple contextual information. We aim to investigate whether LLMs can implement digital versions of board games from rules described in natural language. This would be a step towards an LLM-assisted framework for quick board game code generation. We expect to determine the main challenges for LLMs to implement the board games, and how different approaches and models compare to one another. We task three state-of-the-art LLMs (Claude, DeepSeek and ChatGPT) with coding a selection of 12 popular and obscure games in free-form and within Boardwalk, our proposed General Game Playing API. We anonymize the games and components to avoid evoking pre-trained LLM knowledge. The implementations are tested for playability and rule compliance. We evaluate success rate and common errors across LLMs and game popularity. Our approach proves viable, with the best performing model, Claude 3.7 Sonnet, yielding 55.6\% of games without any errors. While compliance with the API increases error frequency, the severity of errors is more significantly dependent on the LLM. We outline future steps for creating a framework to integrate this process, making the elaboration of board games more accessible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.05114v1">Usando LLMs para Programar Jogos de Tabuleiro e Variações</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted for presentation at the I Escola Regional de Aprendizado de M\'aquina e Intelig\^encia Artificial da Regi\~ao Sul, 2025, in Portuguese language
    </div>
    <details class="paper-abstract">
      Creating programs to represent board games can be a time-consuming task. Large Language Models (LLMs) arise as appealing tools to expedite this process, given their capacity to efficiently generate code from simple contextual information. In this work, we propose a method to test how capable three LLMs (Claude, DeepSeek and ChatGPT) are at creating code for board games, as well as new variants of existing games.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.22963v2">CompressionAttack: Exploiting Prompt Compression as a New Attack Surface in LLM-Powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      LLM-powered agents often use prompt compression to reduce inference costs, but this introduces a new security risk. Compression modules, which are optimized for efficiency rather than safety, can be manipulated by adversarial inputs, causing semantic drift and altering LLM behavior. This work identifies prompt compression as a novel attack surface and presents CompressionAttack, the first framework to exploit it. CompressionAttack includes two strategies: HardCom, which uses discrete adversarial edits for hard compression, and SoftCom, which performs latent-space perturbations for soft compression. Experiments on multiple LLMs show up to 80% attack success and 98% preference flips, while remaining highly stealthy and transferable. Case studies in VSCode Cline and Ollama confirm real-world impact, and current defenses prove ineffective, highlighting the need for stronger protections.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.05080v1">On Text Simplification Metrics and General-Purpose LLMs for Accessible Health Information, and A Potential Architectural Advantage of The Instruction-Tuned LLM class</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      The increasing health-seeking behavior and digital consumption of biomedical information by the general public necessitate scalable solutions for automatically adapting complex scientific and technical documents into plain language. Automatic text simplification solutions, including advanced large language models, however, continue to face challenges in reliably arbitrating the tension between optimizing readability performance and ensuring preservation of discourse fidelity. This report empirically assesses the performance of two major classes of general-purpose LLMs, demonstrating their linguistic capabilities and foundational readiness for the task compared to a human benchmark. Using a comparative analysis of the instruction-tuned Mistral 24B and the reasoning-augmented QWen2.5 32B, we identify a potential architectural advantage in the instruction-tuned LLM. Mistral exhibits a tempered lexical simplification strategy that enhances readability across a suite of metrics and the simplification-specific formula SARI (mean 42.46), while preserving human-level discourse with a BERTScore of 0.91. QWen also attains enhanced readability performance, but its operational strategy shows a disconnect in balancing between readability and accuracy, reaching a statistically significantly lower BERTScore of 0.89. Additionally, a comprehensive correlation analysis of 21 metrics spanning readability, discourse fidelity, content safety, and underlying distributional measures for mechanistic insights, confirms strong functional redundancies among five readability indices. This empirical evidence tracks baseline performance of the evolving LLMs for the task of text simplification, identifies the instruction-tuned Mistral 24B for simplification, provides necessary heuristics for metric selection, and points to lexical support as a primary domain-adaptation issue for simplification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.21150v2">String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      We introduce String Seed of Thought (SSoT), a novel prompting method for LLMs that improves Probabilistic Instruction Following (PIF). We define PIF as a task requiring an LLM to select its answer from a predefined set of options, each associated with a specific probability, such that the empirical distribution of the generated answers aligns with the target distribution when prompted multiple times. While LLMs excel at tasks with single, deterministic answers, they often fail at PIF, exhibiting biases problematic for applications requiring non-deterministic behaviors, such as human-behavior simulation, content diversification, and multiplayer games. It also harms the diversity of generated responses, a crucial factor in test-time scaling, by causing the outputs to collapse into a limited set of answers. To address this, we propose SSoT, a simple prompting method that instructs an LLM to first output a random string to generate sufficient entropy. SSoT also instructs the LLM to extract randomness by manipulating this string to derive a final answer, thereby preserving diversity while adhering to specific constraints. We demonstrate that SSoT significantly improves the PIF performance of LLMs, approaching the ideal performance of a pseudo-random number generator. Furthermore, our experiments on NoveltyBench show SSoT's benefits extend beyond closed-set tasks to open-ended tasks by enhancing response diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04412v5">Wider or Deeper? Scaling LLM Inference-Time Compute with Adaptive Branching Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted as a spotlight at NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advances demonstrate that increasing inference-time computation can significantly boost the reasoning capabilities of large language models (LLMs). Although repeated sampling (i.e., generating multiple candidate outputs) is a highly effective strategy, it does not leverage external feedback signals for refinement, which are often available in tasks like coding. In this work, we propose Adaptive Branching Monte Carlo Tree Search (AB-MCTS), a novel inference-time framework that generalizes repeated sampling with principled multi-turn exploration and exploitation. At each node in the search tree, AB-MCTS dynamically decides whether to "go wider" by expanding new candidate responses or "go deeper" by revisiting existing ones based on external feedback signals. We evaluate our method on complex coding and engineering tasks using frontier models. Empirical results show that AB-MCTS consistently outperforms both repeated sampling and standard MCTS, underscoring the importance of combining the response diversity of LLMs with multi-turn solution refinement for effective inference-time scaling. Code is available at https://github.com/SakanaAI/treequest .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16395v3">AIRepr: An Analyst-Inspector Framework for Evaluating Reproducibility of LLMs in Data Science</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 Accepted to 2025 EMNLP findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used to automate data analysis through executable code generation. Yet, data science tasks often admit multiple statistically valid solutions, e.g. different modeling strategies, making it critical to understand the reasoning behind analyses, not just their outcomes. While manual review of LLM-generated code can help ensure statistical soundness, it is labor-intensive and requires expertise. A more scalable approach is to evaluate the underlying workflows-the logical plans guiding code generation. However, it remains unclear how to assess whether an LLM-generated workflow supports reproducible implementations. To address this, we present AIRepr, an Analyst-Inspector framework for automatically evaluating and improving the reproducibility of LLM-generated data analysis workflows. Our framework is grounded in statistical principles and supports scalable, automated assessment. We introduce two novel reproducibility-enhancing prompting strategies and benchmark them against standard prompting across 15 analyst-inspector LLM pairs and 1,032 tasks from three public benchmarks. Our findings show that workflows with higher reproducibility also yield more accurate analyses, and that reproducibility-enhancing prompts substantially improve both metrics. This work provides a foundation for transparent, reliable, and efficient human-AI collaboration in data science. Our code is publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04962v1">Too Good to be Bad: On the Failure of LLMs to Role-Play Villains</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly tasked with creative generation, including the simulation of fictional characters. However, their ability to portray non-prosocial, antagonistic personas remains largely unexamined. We hypothesize that the safety alignment of modern LLMs creates a fundamental conflict with the task of authentically role-playing morally ambiguous or villainous characters. To investigate this, we introduce the Moral RolePlay benchmark, a new dataset featuring a four-level moral alignment scale and a balanced test set for rigorous evaluation. We task state-of-the-art LLMs with role-playing characters from moral paragons to pure villains. Our large-scale evaluation reveals a consistent, monotonic decline in role-playing fidelity as character morality decreases. We find that models struggle most with traits directly antithetical to safety principles, such as ``Deceitful'' and ``Manipulative'', often substituting nuanced malevolence with superficial aggression. Furthermore, we demonstrate that general chatbot proficiency is a poor predictor of villain role-playing ability, with highly safety-aligned models performing particularly poorly. Our work provides the first systematic evidence of this critical limitation, highlighting a key tension between model safety and creative fidelity. Our benchmark and findings pave the way for developing more nuanced, context-aware alignment methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.01602v3">L2T-Tune:LLM-Guided Hybrid Database Tuning with LHS and TD3</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Configuration tuning is critical for database performance. Although recent advancements in database tuning have shown promising results in throughput and latency improvement, challenges remain. First, the vast knob space makes direct optimization unstable and slow to converge. Second, reinforcement learning pipelines often lack effective warm-start guidance and require long offline training. Third, transferability is limited: when hardware or workloads change, existing models typically require substantial retraining to recover performance. To address these limitations, we propose L2T-Tune, a new LLM-guided hybrid database tuning framework that features a three-stage pipeline: Stage one performs a warm start that simultaneously generates uniform samples across the knob space and logs them into a shared pool; Stage two leverages a large language model to mine and prioritize tuning hints from manuals and community documents for rapid convergence. Stage three uses the warm-start sample pool to reduce the dimensionality of knobs and state features, then fine-tunes the configuration with the Twin Delayed Deep Deterministic Policy Gradient algorithm. We conduct experiments on L2T-Tune and the state-of-the-art models. Compared with the best-performing alternative, our approach improves performance by an average of 37.1% across all workloads, and by up to 73% on TPC-C. Compared with models trained with reinforcement learning, it achieves rapid convergence in the offline tuning stage on a single server. Moreover, during the online tuning stage, it only takes 30 steps to achieve best results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04934v1">Leak@$k$: Unlearning Does Not Make LLMs Forget Under Probabilistic Decoding</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Unlearning in large language models (LLMs) is critical for regulatory compliance and for building ethical generative AI systems that avoid producing private, toxic, illegal, or copyrighted content. Despite rapid progress, in this work we show that \textit{almost all} existing unlearning methods fail to achieve true forgetting in practice. Specifically, while evaluations of these `unlearned' models under deterministic (greedy) decoding often suggest successful knowledge removal using standard benchmarks (as has been done in the literature), we show that sensitive information reliably resurfaces when models are sampled with standard probabilistic decoding. To rigorously capture this vulnerability, we introduce \texttt{leak@$k$}, a new meta-evaluation metric that quantifies the likelihood of forgotten knowledge reappearing when generating $k$ samples from the model under realistic decoding strategies. Using three widely adopted benchmarks, TOFU, MUSE, and WMDP, we conduct the first large-scale, systematic study of unlearning reliability using our newly defined \texttt{leak@$k$} metric. Our findings demonstrate that knowledge leakage persists across methods and tasks, underscoring that current state-of-the-art unlearning techniques provide only limited forgetting and highlighting the urgent need for more robust approaches to LLM unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.20514v2">SciTopic: Enhancing Topic Discovery in Scientific Literature through Advanced LLM</a></div>
    <div class="paper-meta">
      📅 2025-11-07
    </div>
    <details class="paper-abstract">
      Topic discovery in scientific literature provides valuable insights for researchers to identify emerging trends and explore new avenues for investigation, facilitating easier scientific information retrieval. Many machine learning methods, particularly deep embedding techniques, have been applied to discover research topics. However, most existing topic discovery methods rely on word embedding to capture the semantics and lack a comprehensive understanding of scientific publications, struggling with complex, high-dimensional text relationships. Inspired by the exceptional comprehension of textual information by large language models (LLMs), we propose an advanced topic discovery method enhanced by LLMs to improve scientific topic identification, namely SciTopic. Specifically, we first build a textual encoder to capture the content from scientific publications, including metadata, title, and abstract. Next, we construct a space optimization module that integrates entropy-based sampling and triplet tasks guided by LLMs, enhancing the focus on thematic relevance and contextual intricacies between ambiguous instances. Then, we propose to fine-tune the textual encoder based on the guidance from the LLMs by optimizing the contrastive loss of the triplets, forcing the text encoder to better discriminate instances of different topics. Finally, extensive experiments conducted on three real-world datasets of scientific publications demonstrate that SciTopic outperforms the state-of-the-art (SOTA) scientific topic discovery methods, enabling researchers to gain deeper and faster insights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04921v1">AgentExpt: Automating AI Experiment Design with LLM-based Resource Retrieval Agent</a></div>
    <div class="paper-meta">
      📅 2025-11-07
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Large language model agents are becoming increasingly capable at web-centric tasks such as information retrieval, complex reasoning. These emerging capabilities have given rise to surge research interests in developing LLM agent for facilitating scientific quest. One key application in AI research is to automate experiment design through agentic dataset and baseline retrieval. However, prior efforts suffer from limited data coverage, as recommendation datasets primarily harvest candidates from public portals and omit many datasets actually used in published papers, and from an overreliance on content similarity that biases model toward superficial similarity and overlooks experimental suitability. Harnessing collective perception embedded in the baseline and dataset citation network, we present a comprehensive framework for baseline and dataset recommendation. First, we design an automated data-collection pipeline that links roughly one hundred thousand accepted papers to the baselines and datasets they actually used. Second, we propose a collective perception enhanced retriever. To represent the position of each dataset or baseline within the scholarly network, it concatenates self-descriptions with aggregated citation contexts. To achieve efficient candidate recall, we finetune an embedding model on these representations. Finally, we develop a reasoning-augmented reranker that exact interaction chains to construct explicit reasoning chains and finetunes a large language model to produce interpretable justifications and refined rankings. The dataset we curated covers 85\% of the datasets and baselines used at top AI conferences over the past five years. On our dataset, the proposed method outperforms the strongest prior baseline with average gains of +5.85\% in Recall@20, +8.30\% in HitRate@5. Taken together, our results advance reliable, interpretable automation of experimental design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04157v1">Are We Aligned? A Preliminary Investigation of the Alignment of Responsible AI Values between LLMs and Human Judgment</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed in software engineering tasks such as requirements elicitation, design, and evaluation, raising critical questions regarding their alignment with human judgments on responsible AI values. This study investigates how closely LLMs' value preferences align with those of two human groups: a US-representative sample and AI practitioners. We evaluate 23 LLMs across four tasks: (T1) selecting key responsible AI values, (T2) rating their importance in specific contexts, (T3) resolving trade-offs between competing values, and (T4) prioritizing software requirements that embody those values. The results show that LLMs generally align more closely with AI practitioners than with the US-representative sample, emphasizing fairness, privacy, transparency, safety, and accountability. However, inconsistencies appear between the values that LLMs claim to uphold (Tasks 1-3) and the way they prioritize requirements (Task 4), revealing gaps in faithfulness between stated and applied behavior. These findings highlight the practical risk of relying on LLMs in requirements engineering without human oversight and motivate the need for systematic approaches to benchmark, interpret, and monitor value alignment in AI-assisted software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04136v1">Implementation of transformer-based LLMs with large-scale optoelectronic neurons on a CMOS image sensor platform</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      The recent rapid deployment of datacenter infrastructures for performing large language models (LLMs) and related artificial intelligence (AI) applications in the clouds is predicted to incur an exponentially growing energy consumption in the near-term future. In this paper, we propose and analyze the implementation of the transformer model, which is the cornerstone of the modern LLMs, with novel large-scale optoelectronic neurons (OENs) constructed over the commercially available complementary metal-oxide-semiconductor (CMOS) image sensor (CIS) platform. With all of the required optoelectronic devices and electronic circuits integrated in a chiplet only about 2 cm by 3 cm in size, 175 billon parameters in the case of GPT-3 are shown to perform inference at an unprecedented speed of 12.6 POPS using only a 40 nm CMOS process node, along with a high power efficiency of 74 TOPS/W and a high area efficiency of 19 TOPS/mm2, both surpassing the related digital electronics by roughly two orders of magnitude. The influence of the quantization formats and the hardware induced errors are numerically investigated, and are shown to have a minimal impact. Our study presents a new yet practical path toward analog neural processing units (NPUs) to complement existing digital processing units.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04087v1">E-CARE: An Efficient LLM-based Commonsense-Augmented Framework for E-Commerce</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Finding relevant products given a user query plays a pivotal role in an e-commerce platform, as it can spark shopping behaviors and result in revenue gains. The challenge lies in accurately predicting the correlation between queries and products. Recently, mining the cross-features between queries and products based on the commonsense reasoning capacity of Large Language Models (LLMs) has shown promising performance. However, such methods suffer from high costs due to intensive real-time LLM inference during serving, as well as human annotations and potential Supervised Fine Tuning (SFT). To boost efficiency while leveraging the commonsense reasoning capacity of LLMs for various e-commerce tasks, we propose the Efficient Commonsense-Augmented Recommendation Enhancer (E-CARE). During inference, models augmented with E-CARE can access commonsense reasoning with only a single LLM forward pass per query by utilizing a commonsense reasoning factor graph that encodes most of the reasoning schema from powerful LLMs. The experiments on 2 downstream tasks show an improvement of up to 12.1% on precision@5.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04064v1">Benchmarking and Studying the LLM-based Agent System in End-to-End Software Development</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      The development of LLM-based autonomous agents for end-to-end software development represents a significant paradigm shift in software engineering. However, the scientific evaluation of these systems is hampered by significant challenges, including overly simplistic benchmarks and the difficulty of conducting fair comparisons between different agent architectures due to confounding implementation variables. To address these limitations, we first construct a challenging and dynamically curated E2EDevBench to simulate realistic development scenarios. Second, we propose a hybrid evaluation framework that combines test-case-based functional assessment with fine-grained, LLM-based requirement verification. Using this framework, we conduct a controlled empirical study on three representative agent architectures implemented upon a unified foundation to isolate the impact of workflow design. Our findings reveal that state-of-the-art agents can fulfill approximately 50\% of requirements on \bench{}, but their success is critically dependent on the architectural strategy for task decomposition and collaboration. Furthermore, our analysis indicates that the primary bottleneck is the omission of requirements and inadequate self-verification. This work provides the community with a more realistic benchmark, a comprehensive evaluation framework, and crucial insights into the current capabilities and core challenges of software development agents, guiding future research toward enhancing requirement comprehension and planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00730v2">Teaching LLMs to See and Guide: Context-Aware Real-Time Assistance in Augmented Reality</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 This work is intended for submission to the IEEE Transactions on Systems, Man, and Cybernetics: Systems for possible publication
    </div>
    <details class="paper-abstract">
      The growing adoption of augmented and virtual reality (AR and VR) technologies in industrial training and on-the-job assistance has created new opportunities for intelligent, context-aware support systems. As workers perform complex tasks guided by AR and VR, these devices capture rich streams of multimodal data, including gaze, hand actions, and task progression, that can reveal user intent and task state in real time. Leveraging this information effectively remains a major challenge. In this work, we present a context-aware large language model (LLM) assistant that integrates diverse data modalities, such as hand actions, task steps, and dialogue history, into a unified framework for real-time question answering. To systematically study how context influences performance, we introduce an incremental prompting framework, where each model version receives progressively richer contextual inputs. Using the HoloAssist dataset, which records AR-guided task executions, we evaluate how each modality contributes to the assistant's effectiveness. Our experiments show that incorporating multimodal context significantly improves the accuracy and relevance of responses. These findings highlight the potential of LLM-driven multimodal integration to enable adaptive, intuitive assistance for AR and VR-based industrial training and assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04042v1">An LLM-based Framework for Human-Swarm Teaming Cognition in Disaster Search and Rescue</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large-scale disaster Search And Rescue (SAR) operations are persistently challenged by complex terrain and disrupted communications. While Unmanned Aerial Vehicle (UAV) swarms offer a promising solution for tasks like wide-area search and supply delivery, yet their effective coordination places a significant cognitive burden on human operators. The core human-machine collaboration bottleneck lies in the ``intention-to-action gap'', which is an error-prone process of translating a high-level rescue objective into a low-level swarm command under high intensity and pressure. To bridge this gap, this study proposes a novel LLM-CRF system that leverages Large Language Models (LLMs) to model and augment human-swarm teaming cognition. The proposed framework initially captures the operator's intention through natural and multi-modal interactions with the device via voice or graphical annotations. It then employs the LLM as a cognitive engine to perform intention comprehension, hierarchical task decomposition, and mission planning for the UAV swarm. This closed-loop framework enables the swarm to act as a proactive partner, providing active feedback in real-time while reducing the need for manual monitoring and control, which considerably advances the efficacy of the SAR task. We evaluate the proposed framework in a simulated SAR scenario. Experimental results demonstrate that, compared to traditional order and command-based interfaces, the proposed LLM-driven approach reduced task completion time by approximately $64.2\%$ and improved task success rate by $7\%$. It also leads to a considerable reduction in subjective cognitive workload, with NASA-TLX scores dropping by $42.9\%$. This work establishes the potential of LLMs to create more intuitive and effective human-swarm collaborations in high-stakes scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04036v1">PICNIC: Silicon Photonic Interconnected Chiplets with Computational Network and In-memory Computing for LLM Inference Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      This paper presents a 3D-stacked chiplets based large language model (LLM) inference accelerator, consisting of non-volatile in-memory-computing processing elements (PEs) and Inter-PE Computational Network (IPCN), interconnected via silicon photonic to effectively address the communication bottlenecks. A LLM mapping scheme was developed to optimize hardware scheduling and workload mapping. Simulation results show it achieves $3.95\times$ speedup and $30\times$ efficiency improvement over the Nvidia A100 before chiplet clustering and power gating scheme (CCPG). Additionally, the system achieves further scalability and efficiency improvement with the implementation of CCPG to accommodate larger models, attaining $57\times$ efficiency improvement over Nvidia H100 at similar throughput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2510.27140v2">Measuring the Security of Mobile LLM Agents under Adversarial Prompts from Untrusted Third-Party Channels</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed software development, enabling AI-powered applications known as LLM-based agents that promise to automate tasks across diverse apps and workflows. Yet, the security implications of deploying such agents in adversarial mobile environments remain poorly understood. In this paper, we present the first systematic study of security risks in mobile LLM agents. We design and evaluate a suite of adversarial case studies, ranging from opportunistic manipulations such as pop-up advertisements to advanced, end-to-end workflows involving malware installation and cross-app data exfiltration. Our evaluation covers eight state-of-the-art mobile agents across three architectures, with over 2,000 adversarial and paired benign trials. The results reveal systemic vulnerabilities: low-barrier vectors such as fraudulent ads succeed with over 80% reliability, while even workflows requiring the circumvention of operating-system warnings, such as malware installation, are consistently completed by advanced multi-app agents. By mapping these attacks to the MITRE ATT&CK Mobile framework, we uncover novel privilege-escalation and persistence pathways unique to LLM-driven automation. Collectively, our findings provide the first end-to-end evidence that mobile LLM agents are exploitable in realistic adversarial settings, where untrusted third-party channels (e.g., ads, embedded webviews, cross-app notifications) are an inherent part of the mobile ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04023v1">LLM-Driven Adaptive Source-Sink Identification and False Positive Mitigation for Static Analysis</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Static analysis is effective for discovering software vulnerabilities but notoriously suffers from incomplete source--sink specifications and excessive false positives (FPs). We present \textsc{AdaTaint}, an LLM-driven taint analysis framework that adaptively infers source/sink specifications and filters spurious alerts through neuro-symbolic reasoning. Unlike LLM-only detectors, \textsc{AdaTaint} grounds model suggestions in program facts and constraint validation, ensuring both adaptability and determinism. We evaluate \textsc{AdaTaint} on Juliet 1.3, SV-COMP-style C benchmarks, and three large real-world projects. Results show that \textsc{AdaTaint} reduces false positives by \textbf{43.7\%} on average and improves recall by \textbf{11.2\%} compared to state-of-the-art baselines (CodeQL, Joern, and LLM-only pipelines), while maintaining competitive runtime overhead. These findings demonstrate that combining LLM inference with symbolic validation offers a practical path toward more accurate and reliable static vulnerability analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.02263v3">LA-MARRVEL: A Knowledge-Grounded and Language-Aware LLM Reranker for AI-MARRVEL in Rare Disease Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Diagnosing rare diseases requires linking gene findings with often unstructured reference text. Current pipelines collect many candidate genes, but clinicians still spend a lot of time filtering false positives and combining evidence from papers and databases. A key challenge is language: phenotype descriptions and inheritance patterns are written in prose, not fully captured by tables. Large language models (LLMs) can read such text, but clinical use needs grounding in citable knowledge and stable, repeatable behavior. We explore a knowledge-grounded and language-aware reranking layer on top of a high-recall first-stage pipeline. The goal is to improve precision and explainability, not to replace standard bioinformatics steps. We use expert-built context and a consensus method to reduce LLM variability, producing shorter, better-justified gene lists for expert review. LA-MARRVEL achieves the highest accuracy, outperforming other methods -- including traditional bioinformatics diagnostic tools (AI-MARRVEL, Exomiser, LIRICAL) and naive large language models (e.g., Anthropic Claude) -- with an average Recall@5 of 94.10%, a +3.65 percentage-point improvement over AI-MARRVEL. The LLM-generated reasoning provides clear prose on phenotype matching and inheritance patterns, making clinical review faster and easier. LA-MARRVEL has three parts: expert-engineered context that enriches phenotype and disease information; a ranked voting algorithm that combines multiple LLM runs to choose a consensus ranked gene list; and the AI-MARRVEL pipeline that provides first-stage ranks and gene annotations, already known as a state-of-the-art method in Rare Disease Diagnosis on BG, DDD, and UDN cohorts. The online AI-MARRVEL includes LA-MARRVEL as an LLM feature at https://ai.marrvel.org . We evaluate LA-MARRVEL on three datasets from independent cohorts of real-world diagnosed patients.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00847v3">Pay for The Second-Best Service: A Game-Theoretic Approach Against Dishonest LLM Providers</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) through Application Programming Interfaces (APIs) induces a critical vulnerability: the potential for dishonest manipulation by service providers. This manipulation can manifest in various forms, such as secretly substituting a proclaimed high-performance model with a low-cost alternative, or inflating responses with meaningless tokens to increase billing. This work tackles the issue through the lens of algorithmic game theory and mechanism design. We are the first to propose a formal economic model for a realistic user-provider ecosystem, where a user can iteratively delegate $T$ queries to multiple model providers, and providers can engage in a range of strategic behaviors. As our central contribution, we prove that for a continuous strategy space and any $\epsilon\in(0,\frac12)$, there exists an approximate incentive-compatible mechanism with an additive approximation ratio of $O(T^{1-\epsilon}\log T)$, and a guaranteed quasi-linear second-best user utility. We also prove an impossibility result, stating that no mechanism can guarantee an expected user utility that is asymptotically better than our mechanism. Furthermore, we demonstrate the effectiveness of our mechanism in simulation experiments with real-world API settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03995v1">Hybrid Fuzzing with LLM-Guided Input Mutation and Semantic Feedback</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Software fuzzing has become a cornerstone in automated vulnerability discovery, yet existing mutation strategies often lack semantic awareness, leading to redundant test cases and slow exploration of deep program states. In this work, I present a hybrid fuzzing framework that integrates static and dynamic analysis with Large Language Model (LLM)-guided input mutation and semantic feedback. Static analysis extracts control-flow and data-flow information, which is transformed into structured prompts for the LLM to generate syntactically valid and semantically diverse inputs. During execution, I augment traditional coverage-based feedback with semantic feedback signals-derived from program state changes, exception types, and output semantics-allowing the fuzzer to prioritize inputs that trigger novel program behaviors beyond mere code coverage. I implement our approach atop AFL++, combining program instrumentation with embedding-based semantic similarity metrics to guide seed selection. Evaluation on real-world open-source targets, including libpng, tcpdump, and sqlite, demonstrates that our method achieves faster time-to-first-bug, higher semantic diversity, and a competitive number of unique bugs compared to state-of-the-art fuzzers. This work highlights the potential of combining LLM reasoning with semantic-aware feedback to accelerate and deepen vulnerability discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03980v1">LLMs and Cultural Values: the Impact of Prompt Language and Explicit Cultural Framing</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Preprint under review at Computational Linguistics. Accepted with minor revisions (10/10/2025); second round
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are rapidly being adopted by users across the globe, who interact with them in a diverse range of languages. At the same time, there are well-documented imbalances in the training data and optimisation objectives of this technology, raising doubts as to whether LLMs can represent the cultural diversity of their broad user base. In this study, we look at LLMs and cultural values and examine how prompt language and cultural framing influence model responses and their alignment with human values in different countries. We probe 10 LLMs with 63 items from the Hofstede Values Survey Module and World Values Survey, translated into 11 languages, and formulated as prompts with and without different explicit cultural perspectives. Our study confirms that both prompt language and cultural perspective produce variation in LLM outputs, but with an important caveat: While targeted prompting can, to a certain extent, steer LLM responses in the direction of the predominant values of the corresponding countries, it does not overcome the models' systematic bias toward the values associated with a restricted set of countries in our dataset: the Netherlands, Germany, the US, and Japan. All tested models, regardless of their origin, exhibit remarkably similar patterns: They produce fairly neutral responses on most topics, with selective progressive stances on issues such as social tolerance. Alignment with cultural values of human respondents is improved more with an explicit cultural perspective than with a targeted prompt language. Unexpectedly, combining both approaches is no more effective than cultural framing with an English prompt. These findings reveal that LLMs occupy an uncomfortable middle ground: They are responsive enough to changes in prompts to produce variation, but too firmly anchored to specific cultural defaults to adequately represent cultural diversity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.03934v1">PEFA-AI: Advancing Open-source LLMs for RTL generation using Progressive Error Feedback Agentic-AI</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Appeared in the Design Automation Conference (DAC) 2025, Workshop Poster on June 22, 2025
    </div>
    <details class="paper-abstract">
      We present an agentic flow consisting of multiple agents that combine specialized LLMs and hardware simulation tools to collaboratively complete the complex task of Register Transfer Level (RTL) generation without human intervention. A key feature of the proposed flow is the progressive error feedback system of agents (PEFA), a self-correcting mechanism that leverages iterative error feedback to progressively increase the complexity of the approach. The generated RTL includes checks for compilation, functional correctness, and synthesizable constructs. To validate this adaptive approach to code generation, benchmarking is performed using two opensource natural language-to-RTL datasets. We demonstrate the benefits of the proposed approach implemented on an open source agentic framework, using both open- and closed-source LLMs, effectively bridging the performance gap between them. Compared to previously published methods, our approach sets a new benchmark, providing state-of-the-art pass rates while being efficient in token counts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22913v2">Mustafar: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 20 pages, 9 figures, NeurIPS 2025
    </div>
    <details class="paper-abstract">
      We demonstrate that unstructured sparsity significantly improves KV cache compression for LLMs, enabling sparsity levels up to 70% without compromising accuracy or requiring fine-tuning. We conduct a systematic exploration of pruning strategies and find per-token magnitude-based pruning as highly effective for both Key and Value caches under unstructured sparsity, surpassing prior structured pruning schemes. The Key cache benefits from prominent outlier elements, while the Value cache surprisingly benefits from a simple magnitude-based pruning despite its uniform distribution. KV cache size is the major bottleneck in decode performance due to high memory overhead for large context lengths. To address this, we use a bitmap-based sparse format and a custom attention kernel capable of compressing and directly computing over compressed caches pruned to arbitrary sparsity patterns, significantly accelerating memory-bound operations in decode computations and thereby compensating for the overhead of runtime pruning and compression. Our custom attention kernel coupled with the bitmap-based format delivers substantial compression of KV cache upto 45% of dense inference and thereby enables longer context length and increased tokens/sec throughput of upto 2.23x compared to dense inference. Our pruning mechanism and sparse attention kernel is available at https://github.com/dhjoo98/mustafar.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04875v1">Minimal and Mechanistic Conditions for Behavioral Self-Awareness in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Recent studies have revealed that LLMs can exhibit behavioral self-awareness: the ability to accurately describe or predict their own learned behaviors without explicit supervision. This capability raises safety concerns as it may, for example, allow models to better conceal their true abilities during evaluation. We attempt to characterize the minimal conditions under which such self-awareness emerges, and the mechanistic processes through which it manifests. Through controlled finetuning experiments on instruction-tuned LLMs with low-rank adapters (LoRA), we find: (1) that self-awareness can be reliably induced using a single rank-1 LoRA adapter; (2) that the learned self-aware behavior can be largely captured by a single steering vector in activation space, recovering nearly all of the fine-tune's behavioral effect; and (3) that self-awareness is non-universal and domain-localized, with independent representations across tasks. Together, these findings suggest that behavioral self-awareness emerges as a domain-specific, linear feature that can be easily induced and modulated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04869v1">Trained on Tokens, Calibrated on Concepts: The Emergence of Semantic Calibration in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often lack meaningful confidence estimates for their outputs. While base LLMs are known to exhibit next-token calibration, it remains unclear whether they can assess confidence in the actual meaning of their responses beyond the token level. We find that, when using a certain sampling-based notion of semantic calibration, base LLMs are remarkably well-calibrated: they can meaningfully assess confidence in open-domain question-answering tasks, despite not being explicitly trained to do so. Our main theoretical contribution establishes a mechanism for why semantic calibration emerges as a byproduct of next-token prediction, leveraging a recent connection between calibration and local loss optimality. The theory relies on a general definition of "B-calibration," which is a notion of calibration parameterized by a choice of equivalence classes (semantic or otherwise). This theoretical mechanism leads to a testable prediction: base LLMs will be semantically calibrated when they can easily predict their own distribution over semantic answer classes before generating a response. We state three implications of this prediction, which we validate through experiments: (1) Base LLMs are semantically calibrated across question-answering tasks, (2) RL instruction-tuning systematically breaks this calibration, and (3) chain-of-thought reasoning breaks calibration. To our knowledge, our work provides the first principled explanation of when and why semantic calibration emerges in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.15809v2">Chain-of-Query: Unleashing the Power of LLMs in SQL-Aided Table Understanding via Multi-Agent Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 AACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Table understanding requires structured, multi-step reasoning. Large Language Models (LLMs) struggle with it due to the structural complexity of tabular data. Recently, multi-agent frameworks for SQL generation have shown promise in tackling the challenges of understanding tabular data, but existing approaches often suffer from limitations such as the inability to comprehend table structure for reliable SQL generation, error propagation that results in invalid queries, and over-reliance on execution correctness. To address these issues, we propose Chain-of-Query (CoQ), a novel multi-agent framework for SQL-aided table understanding. CoQ adopts natural-language-style representations of table schemas to abstract away structural noise and enhance understanding. It employs a clause-by-clause SQL generation strategy to improve query quality and introduces a hybrid reasoning division that separates SQL-based mechanical reasoning from LLM-based logical inference, thereby reducing reliance on execution outcomes. Extensive experiments across four models and five widely used benchmarks demonstrate that CoQ achieves substantial accuracy improvements and significantly lowers invalid SQL rates compared to prior generic LLM-based, SQL-aided, and hybrid baselines, confirming its superior effectiveness in table understanding. The code is available at https://github.com/SongyuanSui/ChainofQuery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23842v3">Fair Document Valuation in LLM Summaries via Shapley Values</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in systems that retrieve and summarize content from multiple sources, such as search engines and AI assistants. While these systems enhance user experience through coherent summaries, they obscure the individual contributions of original content creators, raising concerns about credit attribution and compensation. We address the challenge of valuing individual documents used in LLM-generated summaries by proposing a Shapley value-based framework for fair document valuation. Although theoretically appealing, exact Shapley value computation is prohibitively expensive at scale. To improve efficiency, we develop Cluster Shapley, a simple approximation algorithm that leverages semantic similarity among documents to reduce computation while maintaining attribution accuracy. Using Amazon product review data, we empirically show that off-the-shelf Shapley approximations, such as Monte Carlo sampling and Kernel SHAP, perform suboptimally in LLM settings, whereas Cluster Shapley substantially improves the efficiency-accuracy frontier. Moreover, simple attribution rules (e.g., equal or relevance-based allocation), though computationally cheap, lead to highly unfair outcomes. Together, our findings highlight the potential of structure-aware Shapley approximations tailored to LLM summarization and offer guidance for platforms seeking scalable and fair content attribution mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04847v1">Grounded Test-Time Adaptation for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      Large language model (LLM)-based agents struggle to generalize to novel and complex environments, such as unseen websites or new sets of functions, due to a fundamental mismatch between their pre-training and test-time conditions. This challenge stems from two distinct failure modes: a syntactic misunderstanding of environment-specific components like observation formats, and a semantic misunderstanding of state-transition dynamics, which are only revealed at test time. To address these issues, we propose two distinct and complementary strategies for adapting LLM agents by leveraging environment-specific information available during deployment. First, an online distributional adaptation method parameterizes environmental nuances by learning a lightweight adaptation vector that biases the model's output distribution, enabling rapid alignment with an environment response format. Second, a deployment-time dynamics grounding method employs a persona-driven exploration phase to systematically probe and learn the environment's causal dynamics before task execution, equipping the agent with a nonparametric world model. We evaluate these strategies across diverse agentic benchmarks, including function calling and web navigation. Our empirical results show the effectiveness of both strategies across all benchmarks with minimal computational cost. We find that dynamics grounding is particularly effective in complex environments where unpredictable dynamics pose a major obstacle, demonstrating a robust path toward more generalizable and capable LLM-based agents. For example, on the WebArena multi-site split, this method increases the agent's success rate from 2% to 23%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.16204v2">Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 To Appear in NeurIPS 2025 Datasets & Benchmarks Track
    </div>
    <details class="paper-abstract">
      Modern engineering, spanning electrical, mechanical, aerospace, civil, and computer disciplines, stands as a cornerstone of human civilization and the foundation of our society. However, engineering design poses a fundamentally different challenge for large language models (LLMs) compared with traditional textbook-style problem solving or factual question answering. Although existing benchmarks have driven progress in areas such as language understanding, code synthesis, and scientific problem solving, real-world engineering design demands the synthesis of domain knowledge, navigation of complex trade-offs, and management of the tedious processes that consume much of practicing engineers' time. Despite these shared challenges across engineering disciplines, no benchmark currently captures the unique demands of engineering design work. In this work, we introduce EngDesign, an Engineering Design benchmark that evaluates LLMs' abilities to perform practical design tasks across nine engineering domains. Unlike existing benchmarks that focus on factual recall or question answering, EngDesign uniquely emphasizes LLMs' ability to synthesize domain knowledge, reason under constraints, and generate functional, objective-oriented engineering designs. Each task in EngDesign represents a real-world engineering design problem, accompanied by a detailed task description specifying design goals, constraints, and performance requirements. EngDesign pioneers a simulation-based evaluation paradigm that moves beyond textbook knowledge to assess genuine engineering design capabilities and shifts evaluation from static answer checking to dynamic, simulation-driven functional verification, marking a crucial step toward realizing the vision of engineering Artificial General Intelligence (AGI).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24095v2">SkyWalker: A Locality-Aware Cross-Region Load Balancer for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Serving Large Language Models (LLMs) efficiently in multi-region setups remains a challenge. Due to cost and GPU availability concerns, providers typically deploy LLMs in multiple regions using instance with long-term commitments, like reserved instances or on-premise clusters, which are often underutilized due to their region-local traffic handling and diurnal traffic variance. In this paper, we introduce SkyWalker, a multi-region load balancer for LLM inference that aggregates regional diurnal patterns through cross-region traffic handling. By doing so, SkyWalker enables providers to reserve instances based on expected global demand, rather than peak demand in each individual region. Meanwhile, SkyWalker preserves KV-Cache locality and load balancing, ensuring cost efficiency without sacrificing performance. SkyWalker achieves this with a cache-aware cross-region traffic handler and a selective pushing based load balancing mechanism. Our evaluation on real-world workloads shows that it achieves 1.12-2.06x higher throughput and 1.74-6.30x lower latency compared to existing load balancers, while reducing total serving cost by 25%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04791v1">DuetServe: Harmonizing Prefill and Decode for LLM Serving via Adaptive GPU Multiplexing</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Modern LLM serving systems must sustain high throughput while meeting strict latency SLOs across two distinct inference phases: compute-intensive prefill and memory-bound decode phases. Existing approaches either (1) aggregate both phases on shared GPUs, leading to interference between prefill and decode phases, which degrades time-between-tokens (TBT); or (2) disaggregate the two phases across GPUs, improving latency but wasting resources through duplicated models and KV cache transfers. We present DuetServe, a unified LLM serving framework that achieves disaggregation-level isolation within a single GPU. DuetServe operates in aggregated mode by default and dynamically activates SM-level GPU spatial multiplexing when TBT degradation is predicted. Its key idea is to decouple prefill and decode execution only when needed through fine-grained, adaptive SM partitioning that provides phase isolation only when contention threatens latency service level objectives (SLOs). DuetServe integrates (1) an attention-aware roofline model to forecast iteration latency, (2) a partitioning optimizer that selects the optimal SM split to maximize throughput under TBT constraints, and (3) an interruption-free execution engine that eliminates CPU-GPU synchronization overhead. Evaluations show that DuetServe improves total throughput by up to 1.3x while maintaining low generation latency compared to state-of-the-art frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05447v3">TOBUGraph: Knowledge Graph-Based Retrieval for Enhanced LLM Performance Beyond RAG</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) is one of the leading and most widely used techniques for enhancing LLM retrieval capabilities, but it still faces significant limitations in commercial use cases. RAG primarily relies on the query-chunk text-to-text similarity in the embedding space for retrieval and can fail to capture deeper semantic relationships across chunks, is highly sensitive to chunking strategies, and is prone to hallucinations. To address these challenges, we propose TOBUGraph, a graph-based retrieval framework that first constructs the knowledge graph from unstructured data dynamically and automatically. Using LLMs, TOBUGraph extracts structured knowledge and diverse relationships among data, going beyond RAG's text-to-text similarity. Retrieval is achieved through graph traversal, leveraging the extracted relationships and structures to enhance retrieval accuracy, eliminating the need for chunking configurations while reducing hallucination. We demonstrate TOBUGraph's effectiveness in TOBU, a real-world application in production for personal memory organization and retrieval. Our evaluation using real user data demonstrates that TOBUGraph outperforms multiple RAG implementations in both precision and recall, significantly improving user experience through improved retrieval accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2410.20749v3">Matryoshka Pilot: Learning to Drive Black-Box LLMs with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Despite the impressive generative abilities of black-box large language models (LLMs), their inherent opacity hinders further advancements in capabilities such as reasoning, planning, and personalization. Existing works aim to enhance LLM capabilities via domain-specific adaptation, which require additional training on accessible model parameters, an infeasible option for black-box LLMs. To address this challenge, we introduce Matryoshka Pilot (M-Pilot), a lightweight white-box LLM controller that guides a large-scale black-box LLM generator by decomposing complex tasks into a series of intermediate outputs. Specifically, we consider the black-box LLM as an environment, with M-Pilot serving as a policy to provide intermediate guidance through prompts for driving the black-box LLM. M-Pilot is trained to pivot the outputs of the black-box LLM aligning with preferences during iterative interaction, which enables controllable multi-turn generation and self-improvement in optimizing intermediate guidance. Empirical evaluations on diverse tasks demonstrate that our method effectively enhances the capabilities of black-box LLMs in complex, long-horizon tasks. Our code is publicly available at: https://github.com/lichangh20/Matryoshka.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05410v2">Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 14 pages,7 figures;Accepted by NeurIPS 2025
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have highlighted the critical importance of extending context length, yet the quadratic complexity of attention mechanisms poses significant challenges for efficient long-context modeling. KV cache compression has emerged as a key approach to address this challenge. Through extensive empirical analysis, we reveal a fundamental yet previously overlooked asymmetry in KV caches: while adjacent keys receive similar attention weights ({\it local homogeneity}), adjacent values demonstrate distinct {\it heterogeneous} distributions. This key-value asymmetry reveals a critical limitation in existing compression methods that treat keys and values uniformly. To address the limitation, we propose a training-free compression framework (AsymKV) that combines homogeneity-based key merging with a mathematically proven lossless value compression. Extensive experiments demonstrate that AsymKV consistently outperforms existing long-context methods across various tasks and base models. For example, on LLaMA3.1-8B, AsymKV achieves an average score of 43.95 on LongBench, surpassing SOTA methods like H$_2$O (38.89) by a large margin.Our code can be found in this link:https://github.com/the-scale-lab/Asymkv.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04541v1">LLM-as-a-Judge: Toward World Models for Slate Recommendation Systems</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Modeling user preferences across domains remains a key challenge in slate recommendation (i.e. recommending an ordered sequence of items) research. We investigate how Large Language Models (LLM) can effectively act as world models of user preferences through pairwise reasoning over slates. We conduct an empirical study involving several LLMs on three tasks spanning different datasets. Our results reveal relationships between task performance and properties of the preference function captured by LLMs, hinting towards areas for improvement and highlighting the potential of LLMs as world models in recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04538v1">From Model to Breach: Towards Actionable LLM-Generated Vulnerabilities Reporting</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      As the role of Large Language Models (LLM)-based coding assistants in software development becomes more critical, so does the role of the bugs they generate in the overall cybersecurity landscape. While a number of LLM code security benchmarks have been proposed alongside approaches to improve the security of generated code, it remains unclear to what extent they have impacted widely used coding LLMs. Here, we show that even the latest open-weight models are vulnerable in the earliest reported vulnerability scenarios in a realistic use setting, suggesting that the safety-functionality trade-off has until now prevented effective patching of vulnerabilities. To help address this issue, we introduce a new severity metric that reflects the risk posed by an LLM-generated vulnerability, accounting for vulnerability severity, generation chance, and the formulation of the prompt that induces vulnerable code generation - Prompt Exposure (PE). To encourage the mitigation of the most serious and prevalent vulnerabilities, we use PE to define the Model Exposure (ME) score, which indicates the severity and prevalence of vulnerabilities a model generates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.17737v2">LLM Targeted Underperformance Disproportionately Impacts Vulnerable Users</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 Paper accepted at AAAI 2026
    </div>
    <details class="paper-abstract">
      While state-of-the-art large language models (LLMs) have shown impressive performance on many tasks, there has been extensive research on undesirable model behavior such as hallucinations and bias. In this work, we investigate how the quality of LLM responses changes in terms of information accuracy, truthfulness, and refusals depending on three user traits: English proficiency, education level, and country of origin. We present extensive experimentation on three state-of-the-art LLMs and two different datasets targeting truthfulness and factuality. Our findings suggest that undesirable behaviors in state-of-the-art LLMs occur disproportionately more for users with lower English proficiency, of lower education status, and originating from outside the US, rendering these models unreliable sources of information towards their most vulnerable users.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04491v1">RUST-BENCH: Benchmarking LLM Reasoning on Unstructured Text within Structured Tables</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Existing tabular reasoning benchmarks mostly test models on small, uniform tables, underrepresenting the complexity of real-world data and giving an incomplete view of Large Language Models' (LLMs) reasoning abilities. Real tables are long, heterogeneous, and domain-specific, mixing structured fields with free text and requiring multi-hop reasoning across thousands of tokens. To address this gap, we introduce RUST-BENCH, a benchmark of 7966 questions from 2031 real-world tables spanning two domains: i) RB-Science (NSF grant records) and ii) RB-Sports (NBA statistics). Unlike prior work, RUST-BENCH evaluates LLMs jointly across scale, heterogeneity, domain specificity, and reasoning complexity. Experiments with open-source and proprietary models show that LLMs struggle with heterogeneous schemas and complex multi-hop inference, revealing persistent weaknesses in current architectures and prompting strategies. RUST-BENCH establishes a challenging new testbed for advancing tabular reasoning research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04486v1">EDIT-Bench: Evaluating LLM Abilities to Perform Real-World Instructed Code Edits</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Instructed code editing, where LLMs directly modify a developer's existing code based on a user instruction, is becoming a widely used interaction mode in AI coding assistants. However, few benchmarks directly evaluate this capability and current datasets often rely on artificial sources. We introduce EDIT-Bench, a benchmark for evaluating LLM code editing capabilities grounded in real-world usage, i.e., user instructions and code contexts collected in the wild. EDIT-Bench comprises of 545 problems, multiple natural and programming languages, and a diverse set of real-world use cases, ranging from resolving errors to adding features. EDIT-Bench introduces context-dependent problems that require the model to understand code context, highlighted code, and cursor position in addition to the user instruction. We evaluate 40 diverse LLMs and observe that EDIT-Bench is a challenging set of problems where only 5 models score over 60%. We find that model performance varies across different categories of user instructions. Further, we find that varying levels of contextual information greatly affect task success rate, with performance varying up to 11%, indicating the importance of evaluating with realistic context.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04478v1">Generate, Evaluate, Iterate: Synthetic Data for Human-in-the-Loop Refinement of LLM Judges</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 29 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The LLM-as-a-judge paradigm enables flexible, user-defined evaluation, but its effectiveness is often limited by the scarcity of diverse, representative data for refining criteria. We present a tool that integrates synthetic data generation into the LLM-as-a-judge workflow, empowering users to create tailored and challenging test cases with configurable domains, personas, lengths, and desired outcomes, including borderline cases. The tool also supports AI-assisted inline editing of existing test cases. To enhance transparency and interpretability, it reveals the prompts and explanations behind each generation. In a user study (N=24), 83% of participants preferred the tool over manually creating or selecting test cases, as it allowed them to rapidly generate diverse synthetic data without additional workload. The generated synthetic data proved as effective as hand-crafted data for both refining evaluation criteria and aligning with human preferences. These findings highlight synthetic data as a promising alternative, particularly in contexts where efficiency and scalability are critical.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04477v1">Enabling Dynamic Sparsity in Quantized LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Deploying large language models (LLMs) on end-user devices is gaining importance due to benefits in responsiveness, privacy, and operational cost. Yet the limited memory and compute capability of mobile and desktop GPUs make efficient execution difficult. Recent observations suggest that the internal activations of LLMs are often dynamically sparse, meaning that for each input, only part of the network contributes significantly to the output. Such sparsity could reduce computation, but it interacts poorly with group-wise quantization, which remains the dominant approach for fitting LLMs onto resource-constrained hardware. To reconcile these two properties, this study proposes a set of techniques that realize dynamic sparse inference under low-bit quantization. The method features: (1) a zigzag-patterned quantization layout that organizes weights in a way consistent with activation sparsity and improves GPU memory locality; (2) a specialized GEMV kernel designed for this layout to fully utilize parallel compute units; and (3) a compact runtime mechanism that gathers sparse indices with minimal overhead. Across several model scales and hardware configurations, the approach achieves up to 1.55x faster decoding throughput while maintaining accuracy comparable to dense quantized inference, showing that structured sparsity and quantization can effectively coexist on commodity GPUs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04847v2">Benchmarking LLM Faithfulness in RAG with Evolving Leaderboards</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 EMNLP Industry Track 2025
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) aims to reduce hallucinations by grounding responses in external context, yet large language models (LLMs) still frequently introduce unsupported information or contradictions even when provided with relevant context. This paper presents two complementary efforts at Vectara to measure and benchmark LLM faithfulness in RAG. First, we describe our original hallucination leaderboard, which has tracked hallucination rates for LLMs since 2023 using our HHEM hallucination detection model. Motivated by limitations observed in current hallucination detection methods, we introduce FaithJudge, an LLM-as-a-judge framework that leverages a pool of diverse human-annotated hallucination examples to substantially improve the automated hallucination evaluation of LLMs. We introduce an enhanced hallucination leaderboard centered on FaithJudge that benchmarks LLMs on RAG faithfulness in summarization, question-answering, and data-to-text generation tasks. FaithJudge enables a more reliable benchmarking of LLM hallucinations in RAG and supports the development of more trustworthy generative AI systems: https://github.com/vectara/FaithJudge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04473v1">Ground-Truth Subgraphs for Better Training and Evaluation of Knowledge Graph Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Retrieval of information from graph-structured knowledge bases represents a promising direction for improving the factuality of LLMs. While various solutions have been proposed, a comparison of methods is difficult due to the lack of challenging QA datasets with ground-truth targets for graph retrieval. We present SynthKGQA, a framework for generating high-quality synthetic Knowledge Graph Question Answering datasets from any Knowledge Graph, providing the full set of ground-truth facts in the KG to reason over each question. We show how, in addition to enabling more informative benchmarking of KG retrievers, the data produced with SynthKGQA also allows us to train better models. We apply SynthKGQA to Wikidata to generate GTSQA, a new dataset designed to test zero-shot generalization abilities of KG retrievers with respect to unseen graph structures and relation types, and benchmark popular solutions for KG-augmented LLMs on it.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.00783v2">When Semantics Connect the Swarm: LLM-Driven Fuzzy Control for Cooperative Multi-Robot Underwater Coverage</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 This paper has been submitted to IEEE Transactions on Mobile Computing. Jingzehua Xu, Weihang Zhang, and Yangyang Li contributed equally to this work and are recognized as the co-first authors of the paper
    </div>
    <details class="paper-abstract">
      Underwater multi-robot cooperative coverage remains challenging due to partial observability, limited communication, environmental uncertainty, and the lack of access to global localization. To address these issues, this paper presents a semantics-guided fuzzy control framework that couples Large Language Models (LLMs) with interpretable control and lightweight coordination. Raw multimodal observations are compressed by the LLM into compact, human-interpretable semantic tokens that summarize obstacles, unexplored regions, and Objects Of Interest (OOIs) under uncertain perception. A fuzzy inference system with pre-defined membership functions then maps these tokens into smooth and stable steering and gait commands, enabling reliable navigation without relying on global positioning. Then, we further coordinate multiple robots by introducing semantic communication that shares intent and local context in linguistic form, enabling agreement on who explores where while avoiding redundant revisits. Extensive simulations in unknown reef-like environments show that, under limited sensing and communication, the proposed framework achieves robust OOI-oriented navigation and cooperative coverage with improved efficiency and adaptability, narrowing the gap between semantic cognition and distributed underwater control in GPS-denied, map-free conditions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.06991v2">Evaluating LLM-Contaminated Crowdsourcing Data Without Ground Truth</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 32 pages, 7 figures
    </div>
    <details class="paper-abstract">
      The recent success of generative AI highlights the crucial role of high-quality human feedback in building trustworthy AI systems. However, the increasing use of large language models (LLMs) by crowdsourcing workers poses a significant challenge: datasets intended to reflect human input may be compromised by LLM-generated responses. Existing LLM detection approaches often rely on high-dimensional training data such as text, making them unsuitable for annotation tasks like multiple-choice labeling. In this work, we investigate the potential of peer prediction -- a mechanism that evaluates the information within workers' responses without using ground truth -- to mitigate LLM-assisted cheating in crowdsourcing with a focus on annotation tasks. Our approach quantifies the correlations between worker answers while conditioning on (a subset of) LLM-generated labels available to the requester. Building on prior research, we propose a training-free scoring mechanism with theoretical guarantees under a crowdsourcing model that accounts for LLM collusion. We establish conditions under which our method is effective and empirically demonstrate its robustness in detecting low-effort cheating on real-world crowdsourcing datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04432v1">If I Could Turn Back Time: Temporal Reframing as a Historical Reasoning Task for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 8 pages, 1 figure, 3 tables, submitted to aconference
    </div>
    <details class="paper-abstract">
      In this study, we experiment with the ability of LLMs to do temporal reasoning. Using a Norwegian book from 1940 containing trivia questions, we prompt the LLMs to answer the questions as if it were 1940. We also pose the questions in both English and Norwegian. Correct answers are often presented as sentences, and grading is done by means of LLM-as-judge, with sampled checks by a native speaker. Prompting in English consistently gave better results than in Norwegian, an unexpected result. In contrast, using larger LLMs improved results. We tested the DeepSeek-R1, Gemma3, Qwen3, and Llama3.1 model families, and also the largest available LLM especially crafted for Norwegian.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04427v1">Speed at the Cost of Quality? The Impact of LLM Agent Assistance on Software Development</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated the promise to revolutionize the field of software engineering. Among other things, LLM agents are rapidly gaining momentum in their application to software development, with practitioners claiming a multifold productivity increase after adoption. Yet, empirical evidence is lacking around these claims. In this paper, we estimate the causal effect of adopting a widely popular LLM agent assistant, namely Cursor, on development velocity and software quality. The estimation is enabled by a state-of-the-art difference-in-differences design comparing Cursor-adopting GitHub projects with a matched control group of similar GitHub projects that do not use Cursor. We find that the adoption of Cursor leads to a significant, large, but transient increase in project-level development velocity, along with a significant and persistent increase in static analysis warnings and code complexity. Further panel generalized method of moments estimation reveals that the increase in static analysis warnings and code complexity acts as a major factor causing long-term velocity slowdown. Our study carries implications for software engineering practitioners, LLM agent assistant designers, and researchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04418v1">The Illusion of Certainty: Uncertainty quantification for LLMs fails under ambiguity</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Accurate uncertainty quantification (UQ) in Large Language Models (LLMs) is critical for trustworthy deployment. While real-world language is inherently ambiguous, reflecting aleatoric uncertainty, existing UQ methods are typically benchmarked against tasks with no ambiguity. In this work, we demonstrate that while current uncertainty estimators perform well under the restrictive assumption of no ambiguity, they degrade to close-to-random performance on ambiguous data. To this end, we introduce MAQA* and AmbigQA*, the first ambiguous question-answering (QA) datasets equipped with ground-truth answer distributions estimated from factual co-occurrence. We find this performance deterioration to be consistent across different estimation paradigms: using the predictive distribution itself, internal representations throughout the model, and an ensemble of models. We show that this phenomenon can be theoretically explained, revealing that predictive-distribution and ensemble-based estimators are fundamentally limited under ambiguity. Overall, our study reveals a key shortcoming of current UQ methods for LLMs and motivates a rethinking of current modeling paradigms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04393v1">Post-Training LLMs as Better Decision-Making Agents: A Regret-Minimization Approach</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly deployed as "agents" for decision-making (DM) in interactive and dynamic environments. Yet, since they were not originally designed for DM, recent studies show that LLMs can struggle even in basic online DM problems, failing to achieve low regret or an effective exploration-exploitation tradeoff. To address this, we introduce Iterative Regret-Minimization Fine-Tuning (Iterative RMFT), a post-training procedure that repeatedly distills low-regret decision trajectories back into the base model. At each iteration, the model rolls out multiple decision trajectories, selects the k-lowest regret ones, and fine-tunes itself on them. Unlike prior methods that (a) distill action sequences from known DM algorithms or (b) rely on manually crafted chain-of-thought templates, our approach leverages the regret metric to elicit the model's own DM ability and reasoning rationales. This reliance on model-generated reasoning avoids rigid output engineering and provides more flexible, natural-language training signals. Empirical results show that Iterative RMFT improves LLMs' DM performance across diverse models - from Transformers with numerical input/output, to open-weight LLMs, and advanced closed-weight models like GPT-4o mini. Its flexibility in output and reasoning formats enables generalization across tasks with varying horizons, action spaces, reward processes, and natural-language contexts. Finally, we provide theoretical insight showing that a single-layer Transformer under this paradigm can act as a no-regret learner in a simplified setting. Overall, Iterative RMFT offers a principled and general post-training framework for enhancing LLMs' decision-making capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01827v3">APRMCTS: Improving LLM-based Automated Program Repair with Iterative Tree Search</a></div>
    <div class="paper-meta">
      📅 2025-11-06
    </div>
    <details class="paper-abstract">
      Automated Program Repair (APR) attempts to fix software bugs without human intervention, which plays a crucial role in software development and maintenance. Recently, with the advances in Large Language Models (LLMs), a rapidly increasing number of APR techniques have been proposed with remarkable performance. However, existing LLM-based APR techniques typically adopt trial-and-error strategies, which suffer from two major drawbacks: (1) inherently limited patch effectiveness due to local exploration, and (2) low search efficiency due to redundant exploration. In this paper, we propose APRMCTS, which uses iterative tree search to improve LLM-based APR. APRMCTS incorporates Monte Carlo Tree Search (MCTS) into patch searching by performing a global evaluation of the explored patches and selecting the most promising one for subsequent refinement and generation. APRMCTS effectively resolves the problems of falling into local optima and thus helps improve the efficiency of patch searching. Our experiments on 835 bugs from Defects4J demonstrate that, when integrated with GPT-3.5, APRMCTS can fix a total of 201 bugs, which outperforms all state-of-the-art baselines. Besides, APRMCTS helps GPT-4o-mini, GPT-3.5, Yi-Coder-9B, and Qwen2.5-Coder-7B to fix 30, 27, 37, and 28 more bugs, respectively. More importantly, APRMCTS boasts a significant performance advantage while employing small patch size (16 and 32), notably fewer than the 500 and 10,000 patches adopted in previous studies. In terms of cost, compared to existing state-of-the-art LLM-based APR methods, APRMCTS has time and monetary costs of less than 20% and 50%, respectively. Our extensive study demonstrates that APRMCTS exhibits good effectiveness and efficiency, with particular advantages in addressing complex bugs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2511.04355v1">Where Do LLMs Still Struggle? An In-Depth Analysis of Code Generation Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-11-06
      | 💬 To be published in Proceedings of 2025 2nd IEEE/ACM International Conference on AI-powered Software (AIware), Data & Benchmark Track
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success in code generation, and the race to improve their performance has become a central focus of AI research. Benchmarks and leaderboards are increasingly popular, offering quantitative rankings of LLMs. However, they provide limited insight into the tasks that LLMs consistently fail to solve - information that is crucial for understanding current limitations and guiding the development of more capable models. To address this gap, we examined code generation tasks across four popular benchmarks, identifying those that major LLMs are most likely to fail. To understand the causes of these failures, we investigated whether the static complexity of solution code contributes to them, followed by a systematic inspection of 114 tasks that LLMs consistently struggled with. Our analysis revealed four recurring patterns of weaknesses in LLMs, as well as common complications within benchmark tasks that most often lead to failure.
    </details>
</div>
