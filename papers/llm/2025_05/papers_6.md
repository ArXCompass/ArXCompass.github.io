# llm - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6
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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20139v1">StructEval: Benchmarking LLMs' Capabilities to Generate Structural Outputs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 16 pages, 9 figures, 13 tables
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) become integral to software development workflows, their ability to generate structured outputs has become critically important. We introduce StructEval, a comprehensive benchmark for evaluating LLMs' capabilities in producing both non-renderable (JSON, YAML, CSV) and renderable (HTML, React, SVG) structured formats. Unlike prior benchmarks, StructEval systematically evaluates structural fidelity across diverse formats through two paradigms: 1) generation tasks, producing structured output from natural language prompts, and 2) conversion tasks, translating between structured formats. Our benchmark encompasses 18 formats and 44 types of task, with novel metrics for format adherence and structural correctness. Results reveal significant performance gaps, even state-of-the-art models like o1-mini achieve only 75.58 average score, with open-source alternatives lagging approximately 10 points behind. We find generation tasks more challenging than conversion tasks, and producing correct visual content more difficult than generating text-only structures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.13862v3">PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20097v1">S2LPP: Small-to-Large Prompt Prediction across LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      The performance of pre-trained Large Language Models (LLMs) is often sensitive to nuances in prompt templates, requiring careful prompt engineering, adding costs in terms of computing and human effort. In this study, we present experiments encompassing multiple LLMs variants of varying sizes aimed at probing their preference with different prompts. Through experiments on Question Answering, we show prompt preference consistency across LLMs of different sizes. We also show that this consistency extends to other tasks, such as Natural Language Inference. Utilizing this consistency, we propose a method to use a smaller model to select effective prompt templates for a larger model. We show that our method substantially reduces the cost of prompt engineering while consistently matching performance with optimal prompts among candidates. More importantly, our experiment shows the efficacy of our strategy across fourteen LLMs and its applicability to a broad range of NLP tasks, highlighting its robustness
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20047v1">Grammars of Formal Uncertainty: When to Trust LLMs in Automated Reasoning Tasks</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show remarkable promise for democratizing automated reasoning by generating formal specifications. However, a fundamental tension exists: LLMs are probabilistic, while formal verification demands deterministic guarantees. This paper addresses this epistemological gap by comprehensively investigating failure modes and uncertainty quantification (UQ) in LLM-generated formal artifacts. Our systematic evaluation of five frontier LLMs reveals Satisfiability Modulo Theories (SMT) based autoformalization's domain-specific impact on accuracy (from +34.8% on logical tasks to -44.5% on factual ones), with known UQ techniques like the entropy of token probabilities failing to identify these errors. We introduce a probabilistic context-free grammar (PCFG) framework to model LLM outputs, yielding a refined uncertainty taxonomy. We find uncertainty signals are task-dependent (e.g., grammar entropy for logic, AUROC>0.93). Finally, a lightweight fusion of these signals enables selective verification, drastically reducing errors (14-100%) with minimal abstention, transforming LLM-driven formalization into a reliable engineering discipline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20045v1">Uncertainty-Aware Attention Heads: Efficient Unsupervised Uncertainty Quantification for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit impressive fluency, but often produce critical errors known as "hallucinations". Uncertainty quantification (UQ) methods are a promising tool for coping with this fundamental shortcoming. Yet, existing UQ methods face challenges such as high computational overhead or reliance on supervised learning. Here, we aim to bridge this gap. In particular, we propose RAUQ (Recurrent Attention-based Uncertainty Quantification), an unsupervised approach that leverages intrinsic attention patterns in transformers to detect hallucinations efficiently. By analyzing attention weights, we identified a peculiar pattern: drops in attention to preceding tokens are systematically observed during incorrect generations for certain "uncertainty-aware" heads. RAUQ automatically selects such heads, recurrently aggregates their attention weights and token-level confidences, and computes sequence-level uncertainty scores in a single forward pass. Experiments across 4 LLMs and 12 question answering, summarization, and translation tasks demonstrate that RAUQ yields excellent results, outperforming state-of-the-art UQ methods using minimal computational overhead (<1% latency). Moreover, it requires no task-specific labels and no careful hyperparameter tuning, offering plug-and-play real-time hallucination detection in white-box LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11211v2">A Survey of LLM-based Agents in Medicine: How far are we from Baymax?</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming healthcare through the development of LLM-based agents that can understand, reason about, and assist with medical tasks. This survey provides a comprehensive review of LLM-based agents in medicine, examining their architectures, applications, and challenges. We analyze the key components of medical agent systems, including system profiles, clinical planning mechanisms, medical reasoning frameworks, and external capacity enhancement. The survey covers major application scenarios such as clinical decision support, medical documentation, training simulations, and healthcare service optimization. We discuss evaluation frameworks and metrics used to assess these agents' performance in healthcare settings. While LLM-based agents show promise in enhancing healthcare delivery, several challenges remain, including hallucination management, multimodal integration, implementation barriers, and ethical considerations. The survey concludes by highlighting future research directions, including advances in medical reasoning inspired by recent developments in LLM architectures, integration with physical systems, and improvements in training simulations. This work provides researchers and practitioners with a structured overview of the current state and future prospects of LLM-based agents in medicine.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20023v1">Training LLM-Based Agents with Synthetic Self-Reflected Trajectories and Partial Masking</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Autonomous agents, which perceive environments and take actions to achieve goals, have become increasingly feasible with the advancements in large language models (LLMs). However, current powerful agents often depend on sophisticated prompt engineering combined with closed-source LLMs like GPT-4. Although training open-source LLMs using expert trajectories from teacher models has yielded some improvements in agent capabilities, this approach still faces limitations such as performance plateauing and error propagation. To mitigate these challenges, we propose STeP, a novel method for improving LLM-based agent training. We synthesize self-reflected trajectories that include reflections and corrections of error steps, which enhance the effectiveness of LLM agents in learning from teacher models, enabling them to become agents capable of self-reflecting and correcting. We also introduce partial masking strategy that prevents the LLM from internalizing incorrect or suboptimal steps. Experiments demonstrate that our method improves agent performance across three representative tasks: ALFWorld, WebShop, and SciWorld. For the open-source model LLaMA2-7B-Chat, when trained using self-reflected trajectories constructed with Qwen1.5-110B-Chat as the teacher model, it achieves comprehensive improvements with less training data compared to agents trained exclusively on expert trajectories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20020v1">Ontology- and LLM-based Data Harmonization for Federated Learning in Healthcare</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Related dataset: https://doi.org/10.5281/zenodo.15411810
    </div>
    <details class="paper-abstract">
      The rise of electronic health records (EHRs) has unlocked new opportunities for medical research, but privacy regulations and data heterogeneity remain key barriers to large-scale machine learning. Federated learning (FL) enables collaborative modeling without sharing raw data, yet faces challenges in harmonizing diverse clinical datasets. This paper presents a two-step data alignment strategy integrating ontologies and large language models (LLMs) to support secure, privacy-preserving FL in healthcare, demonstrating its effectiveness in a real-world project involving semantic mapping of EHR data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19997v1">Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are revolutionizing education, with LLM-based agents playing a key role in simulating student behavior. A major challenge in student simulation is modeling the diverse learning patterns of students at various cognitive levels. However, current LLMs, typically trained as ``helpful assistants'', target at generating perfect responses. As a result, they struggle to simulate students with diverse cognitive abilities, as they often produce overly advanced answers, missing the natural imperfections that characterize student learning and resulting in unrealistic simulations. To address this issue, we propose a training-free framework for student simulation. We begin by constructing a cognitive prototype for each student using a knowledge graph, which captures their understanding of concepts from past learning records. This prototype is then mapped to new tasks to predict student performance. Next, we simulate student solutions based on these predictions and iteratively refine them using a beam search method to better replicate realistic mistakes. To validate our approach, we construct the \texttt{Student\_100} dataset, consisting of $100$ students working on Python programming and $5,000$ learning records. Experimental results show that our method consistently outperforms baseline models, achieving $100\%$ improvement in simulation accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19970v1">CP-Router: An Uncertainty-Aware Router Between LLM and LRM</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Recent advances in Large Reasoning Models (LRMs) have significantly improved long-chain reasoning capabilities over Large Language Models (LLMs). However, LRMs often produce unnecessarily lengthy outputs even for simple queries, leading to inefficiencies or even accuracy degradation compared to LLMs. To overcome this, we propose CP-Router, a training-free and model-agnostic routing framework that dynamically selects between an LLM and an LRM, demonstrated with multiple-choice question answering (MCQA) prompts. The routing decision is guided by the prediction uncertainty estimates derived via Conformal Prediction (CP), which provides rigorous coverage guarantees. To further refine the uncertainty differentiation across inputs, we introduce Full and Binary Entropy (FBE), a novel entropy-based criterion that adaptively selects the appropriate CP threshold. Experiments across diverse MCQA benchmarks, including mathematics, logical reasoning, and Chinese chemistry, demonstrate that CP-Router efficiently reduces token usage while maintaining or even improving accuracy compared to using LRM alone. We also extend CP-Router to diverse model pairings and open-ended QA, where it continues to demonstrate strong performance, validating its generality and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17138v2">RAP: Runtime-Adaptive Pruning for LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at language understanding and generation, but their enormous computational and memory requirements hinder deployment. Compression offers a potential solution to mitigate these constraints. However, most existing methods rely on fixed heuristics and thus fail to adapt to runtime memory variations or heterogeneous KV-cache demands arising from diverse user requests. To address these limitations, we propose RAP, an elastic pruning framework driven by reinforcement learning (RL) that dynamically adjusts compression strategies in a runtime-aware manner. Specifically, RAP dynamically tracks the evolving ratio between model parameters and KV-cache across practical execution. Recognizing that FFNs house most parameters, whereas parameter -light attention layers dominate KV-cache formation, the RL agent retains only those components that maximize utility within the current memory budget, conditioned on instantaneous workload and device state. Extensive experiments results demonstrate that RAP outperforms state-of-the-art baselines, marking the first time to jointly consider model weights and KV-cache on the fly.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19954v1">An Explainable Diagnostic Framework for Neurodegenerative Dementias via Reinforcement-Optimized LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      The differential diagnosis of neurodegenerative dementias is a challenging clinical task, mainly because of the overlap in symptom presentation and the similarity of patterns observed in structural neuroimaging. To improve diagnostic efficiency and accuracy, deep learning-based methods such as Convolutional Neural Networks and Vision Transformers have been proposed for the automatic classification of brain MRIs. However, despite their strong predictive performance, these models find limited clinical utility due to their opaque decision making. In this work, we propose a framework that integrates two core components to enhance diagnostic transparency. First, we introduce a modular pipeline for converting 3D T1-weighted brain MRIs into textual radiology reports. Second, we explore the potential of modern Large Language Models (LLMs) to assist clinicians in the differential diagnosis between Frontotemporal dementia subtypes, Alzheimer's disease, and normal aging based on the generated reports. To bridge the gap between predictive accuracy and explainability, we employ reinforcement learning to incentivize diagnostic reasoning in LLMs. Without requiring supervised reasoning traces or distillation from larger models, our approach enables the emergence of structured diagnostic rationales grounded in neuroimaging findings. Unlike post-hoc explainability methods that retrospectively justify model decisions, our framework generates diagnostic rationales as part of the inference process-producing causally grounded explanations that inform and guide the model's decision-making process. In doing so, our framework matches the diagnostic performance of existing deep learning methods while offering rationales that support its diagnostic conclusions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19937v1">ALAS: Measuring Latent Speech-Text Alignment For Spoken Language Understanding In Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are widely used in Spoken Language Understanding (SLU). Recent SLU models process audio directly by adapting speech input into LLMs for better multimodal learning. A key consideration for these models is the cross-modal alignment between text and audio modalities, which is a telltale sign as to whether or not LLM is able to associate semantic meaning to audio segments. While various methods exist for fusing these modalities, there is no standard metric to evaluate alignment quality in LLMs. In this work, we propose a new metric, ALAS (Automatic Latent Alignment Score). Our study examines the correlation between audio and text representations across transformer layers, for two different tasks (Spoken Question Answering and Emotion Recognition). We showcase that our metric behaves as expected across different layers and different tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19933v1">Subtle Risks, Critical Failures: A Framework for Diagnosing Physical Safety of LLMs for Embodied Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 37 pages, 13 tables, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used for decision making in embodied agents, yet existing safety evaluations often rely on coarse success rates and domain-specific setups, making it difficult to diagnose why and where these models fail. This obscures our understanding of embodied safety and limits the selective deployment of LLMs in high-risk physical environments. We introduce SAFEL, the framework for systematically evaluating the physical safety of LLMs in embodied decision making. SAFEL assesses two key competencies: (1) rejecting unsafe commands via the Command Refusal Test, and (2) generating safe and executable plans via the Plan Safety Test. Critically, the latter is decomposed into functional modules, goal interpretation, transition modeling, action sequencing, enabling fine-grained diagnosis of safety failures. To support this framework, we introduce EMBODYGUARD, a PDDL-grounded benchmark containing 942 LLM-generated scenarios covering both overtly malicious and contextually hazardous instructions. Evaluation across 13 state-of-the-art LLMs reveals that while models often reject clearly unsafe commands, they struggle to anticipate and mitigate subtle, situational risks. Our results highlight critical limitations in current LLMs and provide a foundation for more targeted, modular improvements in safe embodied reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19912v1">APE: A Data-Centric Benchmark for Efficient LLM Adaptation in Text Summarization</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      We present Adjacent Possible Exploration (APE), a simple yet effective method for adapting large language models to specific tasks using minimal computational resources. Unlike traditional fine-tuning that requires extensive compute, APE iteratively fine-tunes models on small, carefully selected data batches (200 examples), retaining only improvements. On news summarization, APE achieves 40 percent BLEU improvement using just a T4 GPU in 60 minutes, matching or exceeding more complex methods like LoRA while remaining conceptually simple. Our approach is particularly valuable for researchers and practitioners with limited computational resources. We provide open-source code and demonstrate APE's effectiveness through both automatic metrics and human evaluation. While inspired by evolutionary theory's "adjacent possible", APE's core insight has a very practical application: small, iterative data perturbations can efficiently guide LLMs toward task-specific performance without expensive retraining.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01822v5">Firewalls to Secure Dynamic LLM Agentic Networks</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      LLM agents will likely communicate on behalf of users with other entity-representing agents on tasks involving long-horizon plans with interdependent goals. Current work neglects these agentic networks and their challenges. We identify required properties for agent communication: proactivity, adaptability, privacy (sharing only task-necessary information), and security (preserving integrity and utility against selfish entities). After demonstrating communication vulnerabilities, we propose a practical design and protocol inspired by network security principles. Our framework automatically derives task-specific rules from prior conversations to build firewalls. These firewalls construct a closed language that is completely controlled by the developer. They transform any personal data to the allowed degree of permissibility entailed by the task. Both operations are completely quarantined from external attackers, disabling the potential for prompt injections, jailbreaks, or manipulation. By incorporating rules learned from their previous mistakes, agents rewrite their instructions and self-correct during communication. Evaluations on diverse attacks demonstrate our framework significantly reduces privacy and security vulnerabilities while allowing adaptability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05716v2">Single-Agent vs. Multi-Agent LLM Strategies for Automated Student Reflection Assessment</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Published in Proceedings of the 29th Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD 2025)
    </div>
    <details class="paper-abstract">
      We explore the use of Large Language Models (LLMs) for automated assessment of open-text student reflections and prediction of academic performance. Traditional methods for evaluating reflections are time-consuming and may not scale effectively in educational settings. In this work, we employ LLMs to transform student reflections into quantitative scores using two assessment strategies (single-agent and multi-agent) and two prompting techniques (zero-shot and few-shot). Our experiments, conducted on a dataset of 5,278 reflections from 377 students over three academic terms, demonstrate that the single-agent with few-shot strategy achieves the highest match rate with human evaluations. Furthermore, models utilizing LLM-assessed reflection scores outperform baselines in both at-risk student identification and grade prediction tasks. These findings suggest that LLMs can effectively automate reflection assessment, reduce educators' workload, and enable timely support for students who may need additional assistance. Our work emphasizes the potential of integrating advanced generative AI technologies into educational practices to enhance student engagement and academic success.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19851v1">Beyond Specialization: Benchmarking LLMs for Transliteration of Indian Languages</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Transliteration, the process of mapping text from one script to another, plays a crucial role in multilingual natural language processing, especially within linguistically diverse contexts such as India. Despite significant advancements through specialized models like IndicXlit, recent developments in large language models suggest a potential for general-purpose models to excel at this task without explicit task-specific training. The current work systematically evaluates the performance of prominent LLMs, including GPT-4o, GPT-4.5, GPT-4.1, Gemma-3-27B-it, and Mistral-Large against IndicXlit, a state-of-the-art transliteration model, across ten major Indian languages. Experiments utilized standard benchmarks, including Dakshina and Aksharantar datasets, with performance assessed via Top-1 Accuracy and Character Error Rate. Our findings reveal that while GPT family models generally outperform other LLMs and IndicXlit for most instances. Additionally, fine-tuning GPT-4o improves performance on specific languages notably. An extensive error analysis and robustness testing under noisy conditions further elucidate strengths of LLMs compared to specialized models, highlighting the efficacy of foundational models for a wide spectrum of specialized applications with minimal overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.04394v2">DECT: Harnessing LLM-assisted Fine-Grained Linguistic Knowledge and Label-Switched and Label-Preserved Data Generation for Diagnosis of Alzheimer's Disease</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Alzheimer's Disease (AD) is an irreversible neurodegenerative disease affecting 50 million people worldwide. Low-cost, accurate identification of key markers of AD is crucial for timely diagnosis and intervention. Language impairment is one of the earliest signs of cognitive decline, which can be used to discriminate AD patients from normal control individuals. Patient-interviewer dialogues may be used to detect such impairments, but they are often mixed with ambiguous, noisy, and irrelevant information, making the AD detection task difficult. Moreover, the limited availability of AD speech samples and variability in their speech styles pose significant challenges in developing robust speech-based AD detection models. To address these challenges, we propose DECT, a novel speech-based domain-specific approach leveraging large language models (LLMs) for fine-grained linguistic analysis and label-switched label-preserved data generation. Our study presents four novelties: We harness the summarizing capabilities of LLMs to identify and distill key Cognitive-Linguistic information from noisy speech transcripts, effectively filtering irrelevant information. We leverage the inherent linguistic knowledge of LLMs to extract linguistic markers from unstructured and heterogeneous audio transcripts. We exploit the compositional ability of LLMs to generate AD speech transcripts consisting of diverse linguistic patterns to overcome the speech data scarcity challenge and enhance the robustness of AD detection models. We use the augmented AD textual speech transcript dataset and a more fine-grained representation of AD textual speech transcript data to fine-tune the AD detection model. The results have shown that DECT demonstrates superior model performance with an 11% improvement in AD detection accuracy on the datasets from DementiaBank compared to the baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07440v3">Model Utility Law: Evaluating LLMs beyond Performance through Mechanism Interpretable Metric</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become indispensable across academia, industry, and daily applications, yet current evaluation methods struggle to keep pace with their rapid development. One core challenge of evaluation in the large language model (LLM) era is the generalization issue: how to infer a model's near-unbounded abilities from inevitably bounded benchmarks. We address this challenge by proposing Model Utilization Index (MUI), a mechanism interpretability enhanced metric that complements traditional performance scores. MUI quantifies the effort a model expends on a task, defined as the proportion of activated neurons or features during inference. Intuitively, a truly capable model should achieve higher performance with lower effort. Extensive experiments across popular LLMs reveal a consistent inverse logarithmic relationship between MUI and performance, which we formulate as the Utility Law. From this law we derive four practical corollaries that (i) guide training diagnostics, (ii) expose data contamination issue, (iii) enable fairer model comparisons, and (iv) design model-specific dataset diversity. Our code can be found at https://github.com/ALEX-nlp/MUI-Eva.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19828v1">SecVulEval: Benchmarking LLMs for Real-World C/C++ Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise in software engineering tasks, but evaluating their effectiveness in vulnerability detection is challenging due to the lack of high-quality datasets. Most existing datasets are limited to function-level labels, ignoring finer-grained vulnerability patterns and crucial contextual information. Also, poor data quality such as mislabeling, inconsistent annotations, and duplicates can lead to inflated performance and weak generalization. Moreover, by including only the functions, these datasets miss broader program context, like data/control dependencies and interprocedural interactions, that are essential for accurately understanding real-world security flaws. Without this context, detection models are evaluated under unrealistic assumptions. To address these limitations, this paper introduces SecVulEval, a benchmark designed to support fine-grained evaluation of LLMs and other detection methods with rich contextual information. SecVulEval focuses on real-world C/C++ vulnerabilities at the statement level. This granularity enables more precise evaluation of a model's ability to localize vulnerabilities, beyond simple binary classification at the function level. By incorporating rich contextual information, SecVulEval sets a new standard for vulnerability detection benchmarks in realistic scenarios. This benchmark includes 25,440 function samples covering 5,867 unique CVEs in C/C++ projects from 1999 to 2024. We evaluated the SOTA LLMs with a multi-agent-based approach. The evaluation on our dataset shows that the models are still far from accurately predicting vulnerable statements in a given function. The best-performing Claude-3.7-Sonnet model achieves 23.83% F1-score for detecting vulnerable statements with correct reasoning. Finally, we analyze the LLM outputs and provide insights into their behavior in vulnerability detection for C/C++.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19819v1">FinLoRA: Benchmarking LoRA Methods for Fine-Tuning LLMs on Financial Datasets</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Low-rank adaptation (LoRA) methods show great potential for scaling pre-trained general-purpose Large Language Models (LLMs) to hundreds or thousands of use scenarios. However, their efficacy in high-stakes domains like finance is rarely explored, e.g., passing CFA exams and analyzing SEC filings. In this paper, we present the open-source FinLoRA project that benchmarks LoRA methods on both general and highly professional financial tasks. First, we curated 19 datasets covering diverse financial applications; in particular, we created four novel XBRL analysis datasets based on 150 SEC filings. Second, we evaluated five LoRA methods and five base LLMs. Finally, we provide extensive experimental results in terms of accuracy, F1, and BERTScore and report computational cost in terms of time and GPU memory during fine-tuning and inference stages. We find that LoRA methods achieved substantial performance gains of 36\% on average over base models. Our FinLoRA project provides an affordable and scalable approach to democratize financial intelligence to the general public. Datasets, LoRA adapters, code, and documentation are available at https://github.com/Open-Finance-Lab/FinLoRA
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19815v1">Deciphering Trajectory-Aided LLM Reasoning: An Optimization Perspective</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      We propose a novel framework for comprehending the reasoning capabilities of large language models (LLMs) through the perspective of meta-learning. By conceptualizing reasoning trajectories as pseudo-gradient descent updates to the LLM's parameters, we identify parallels between LLM reasoning and various meta-learning paradigms. We formalize the training process for reasoning tasks as a meta-learning setup, with each question treated as an individual task, and reasoning trajectories serving as the inner loop optimization for adapting model parameters. Once trained on a diverse set of questions, the LLM develops fundamental reasoning capabilities that can generalize to previously unseen questions. Extensive empirical evaluations substantiate the strong connection between LLM reasoning and meta-learning, exploring several issues of significant interest from a meta-learning standpoint. Our work not only enhances the understanding of LLM reasoning but also provides practical insights for improving these models through established meta-learning techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19806v1">Exploring Consciousness in LLMs: A Systematic Survey of Theories, Implementations, and Frontier Risks</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Consciousness stands as one of the most profound and distinguishing features of the human mind, fundamentally shaping our understanding of existence and agency. As large language models (LLMs) develop at an unprecedented pace, questions concerning intelligence and consciousness have become increasingly significant. However, discourse on LLM consciousness remains largely unexplored territory. In this paper, we first clarify frequently conflated terminologies (e.g., LLM consciousness and LLM awareness). Then, we systematically organize and synthesize existing research on LLM consciousness from both theoretical and empirical perspectives. Furthermore, we highlight potential frontier risks that conscious LLMs might introduce. Finally, we discuss current challenges and outline future directions in this emerging field. The references discussed in this paper are organized at https://github.com/OpenCausaLab/Awesome-LLM-Consciousness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19800v1">MOLE: Metadata Extraction and Validation in Scientific Papers Using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Metadata extraction is essential for cataloging and preserving datasets, enabling effective research discovery and reproducibility, especially given the current exponential growth in scientific research. While Masader (Alyafeai et al.,2021) laid the groundwork for extracting a wide range of metadata attributes from Arabic NLP datasets' scholarly articles, it relies heavily on manual annotation. In this paper, we present MOLE, a framework that leverages Large Language Models (LLMs) to automatically extract metadata attributes from scientific papers covering datasets of languages other than Arabic. Our schema-driven methodology processes entire documents across multiple input formats and incorporates robust validation mechanisms for consistent output. Additionally, we introduce a new benchmark to evaluate the research progress on this task. Through systematic analysis of context length, few-shot learning, and web browsing integration, we demonstrate that modern LLMs show promising results in automating this task, highlighting the need for further future work improvements to ensure consistent and reliable performance. We release the code: https://github.com/IVUL-KAUST/MOLE and dataset: https://huggingface.co/datasets/IVUL-KAUST/MOLE for the research community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02810v2">Mol-LLM: Multimodal Generalist Molecular LLM with Improved Graph Utilization</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 9 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have led to models that tackle diverse molecular tasks, such as chemical reaction prediction and molecular property prediction. Large-scale molecular instruction-tuning datasets have enabled sequence-only (e.g., SMILES or SELFIES) generalist molecular LLMs, and researchers are now exploring multimodal approaches that incorporate molecular structural information for further gains. However, a genuinely multimodal, generalist LLM that covers a broad spectrum of molecular tasks has yet to be fully investigated. We observe that naive next token prediction training ignores graph-structural information, limiting an LLM's ability to exploit molecular graphs. To address this, we propose (i) Molecular structure Preference Optimization (MolPO), which facilitates graph usage by optimizing preferences between pairs of correct and perturbed molecular structures, and (ii) an advanced graph encoder with a tailored pre-training strategy to improve the effect of graph utilization by MolPO. Building on these contributions, we introduce Mol-LLM, the first multimodal generalist model that (a) handles a broad spectrum of molecular tasks among molecular LLMs, (b) explicitly leverages molecular-structure information, and (c) takes advantage of extensive instruction tuning. Mol-LLM attains state-of-the-art or comparable results across the most comprehensive molecular-LLM benchmark-even on out-of-distribution datasets for reaction and property prediction, where it surpasses prior generalist molecular LLMs by a large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19735v3">R1-T1: Fully Incentivizing Translation Capability in LLMs via Reasoning Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Despite recent breakthroughs in reasoning-enhanced large language models (LLMs) like DeepSeek-R1, incorporating inference-time reasoning into machine translation (MT), where human translators naturally employ structured, multi-layered reasoning chain-of-thoughts (CoTs), is yet underexplored. Existing methods either design a fixed CoT tailored for a specific MT sub-task (e.g., literature translation), or rely on synthesizing CoTs unaligned with humans and supervised fine-tuning (SFT) prone to overfitting, limiting their adaptability to diverse translation scenarios. This paper introduces R1-Translator (R1-T1), a novel framework to achieve inference-time reasoning for general MT via reinforcement learning (RL) with human-aligned CoTs comprising six common patterns. Our approach pioneers three innovations: (1) extending reasoning-based translation to broader MT scenarios (e.g., multilingual MT, domain MT) unseen in the training phase; (2) formalizing six expert-curated CoT templates that mirror hybrid human strategies like context-aware paraphrasing and back translation; and (3) enabling self-evolving CoT discovery through RL. Both human and automatic evaluation results indicate a steady translation performance improvement in a total of 10+ languages and 40+ translation directions on Flores-101 test set and four domain-specific MT tasks, especially on the languages unseen from training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19776v1">Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 To be published in the Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)
    </div>
    <details class="paper-abstract">
      Political biases encoded by LLMs might have detrimental effects on downstream applications. Existing bias analysis methods rely on small-size intermediate tasks (questionnaire answering or political content generation) and rely on the LLMs themselves for analysis, thus propagating bias. We propose a new approach leveraging the observation that LLM sentiment predictions vary with the target entity in the same sentence. We define an entropy-based inconsistency metric to encode this prediction variability. We insert 1319 demographically and politically diverse politician names in 450 political sentences and predict target-oriented sentiment using seven models in six widely spoken languages. We observe inconsistencies in all tested combinations and aggregate them in a statistically robust analysis at different granularity levels. We observe positive and negative bias toward left and far-right politicians and positive correlations between politicians with similar alignment. Bias intensity is higher for Western languages than for others. Larger models exhibit stronger and more consistent biases and reduce discrepancies between similar languages. We partially mitigate LLM unreliability in target-oriented sentiment classification (TSC) by replacing politician names with fictional but plausible counterparts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19773v1">What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Accepted by ACL 2025
    </div>
    <details class="paper-abstract">
      We investigate long-context vulnerabilities in Large Language Models (LLMs) through Many-Shot Jailbreaking (MSJ). Our experiments utilize context length of up to 128K tokens. Through comprehensive analysis with various many-shot attack settings with different instruction styles, shot density, topic, and format, we reveal that context length is the primary factor determining attack effectiveness. Critically, we find that successful attacks do not require carefully crafted harmful content. Even repetitive shots or random dummy text can circumvent model safety measures, suggesting fundamental limitations in long-context processing capabilities of LLMs. The safety behavior of well-aligned models becomes increasingly inconsistent with longer contexts. These findings highlight significant safety gaps in context expansion capabilities of LLMs, emphasizing the need for new safety mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19761v1">Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Accepted by ICML 2025, 21 pages
    </div>
    <details class="paper-abstract">
      While showing sophisticated reasoning abilities, large language models (LLMs) still struggle with long-horizon decision-making tasks due to deficient exploration and long-term credit assignment, especially in sparse-reward scenarios. Inspired by the divide-and-conquer principle, we propose an innovative framework **GLIDER** (**G**rounding **L**anguage Models as Eff**I**cient **D**ecision-Making Agents via Offline Hi**E**rarchical **R**einforcement Learning) that introduces a parameter-efficient and generally applicable hierarchy to LLM policies. We develop a scheme where the low-level controller is supervised with abstract, step-by-step plans that are learned and instructed by the high-level policy. This design decomposes complicated problems into a series of coherent chain-of-thought reasoning sub-tasks, providing flexible temporal abstraction to significantly enhance exploration and learning for long-horizon tasks. Furthermore, GLIDER facilitates fast online adaptation to non-stationary environments owing to the strong transferability of its task-agnostic low-level skills. Experiments on ScienceWorld and ALFWorld benchmarks show that GLIDER achieves consistent performance gains, along with enhanced generalization capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19734v1">ReChisel: Effective Automatic Chisel Code Generation by LLM with Reflection</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Accepted to DAC 2025
    </div>
    <details class="paper-abstract">
      Coding with hardware description languages (HDLs) such as Verilog is a time-intensive and laborious task. With the rapid advancement of large language models (LLMs), there is increasing interest in applying LLMs to assist with HDL coding. Recent efforts have demonstrated the potential of LLMs in translating natural language to traditional HDL Verilog. Chisel, a next-generation HDL based on Scala, introduces higher-level abstractions, facilitating more concise, maintainable, and scalable hardware designs. However, the potential of using LLMs for Chisel code generation remains largely unexplored. This work proposes ReChisel, an LLM-based agentic system designed to enhance the effectiveness of Chisel code generation. ReChisel incorporates a reflection mechanism to iteratively refine the quality of generated code using feedback from compilation and simulation processes, and introduces an escape mechanism to break free from non-progress loops. Experiments demonstrate that ReChisel significantly improves the success rate of Chisel code generation, achieving performance comparable to state-of-the-art LLM-based agentic systems for Verilog code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19722v1">Distilling Closed-Source LLM's Knowledge for Locally Stable and Economic Biomedical Entity Linking</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Accepted by ICIC 2025
    </div>
    <details class="paper-abstract">
      Biomedical entity linking aims to map nonstandard entities to standard entities in a knowledge base. Traditional supervised methods perform well but require extensive annotated data to transfer, limiting their usage in low-resource scenarios. Large language models (LLMs), especially closed-source LLMs, can address these but risk stability issues and high economic costs: using these models is restricted by commercial companies and brings significant economic costs when dealing with large amounts of data. To address this, we propose ``RPDR'', a framework combining closed-source LLMs and open-source LLMs for re-ranking candidates retrieved by a retriever fine-tuned with a small amount of data. By prompting a closed-source LLM to generate training data from unannotated data and fine-tuning an open-source LLM for re-ranking, we effectively distill the knowledge to the open-source LLM that can be deployed locally, thus avoiding the stability issues and the problem of high economic costs. We evaluate RPDR on two datasets, including one real-world dataset and one publicly available dataset involving two languages: Chinese and English. RPDR achieves 0.019 Acc@1 improvement and 0.036 Acc@1 improvement on the Aier dataset and the Ask A Patient dataset when the amount of training data is not enough. The results demonstrate the superiority and generalizability of the proposed framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15337v2">Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      The misuse of large language models (LLMs), such as academic plagiarism, has driven the development of detectors to identify LLM-generated texts. To bypass these detectors, paraphrase attacks have emerged to purposely rewrite these texts to evade detection. Despite the success, existing methods require substantial data and computational budgets to train a specialized paraphraser, and their attack efficacy greatly reduces when faced with advanced detection algorithms. To address this, we propose \textbf{Co}ntrastive \textbf{P}araphrase \textbf{A}ttack (CoPA), a training-free method that effectively deceives text detectors using off-the-shelf LLMs. The first step is to carefully craft instructions that encourage LLMs to produce more human-like texts. Nonetheless, we observe that the inherent statistical biases of LLMs can still result in some generated texts carrying certain machine-like attributes that can be captured by detectors. To overcome this, CoPA constructs an auxiliary machine-like word distribution as a contrast to the human-like distribution generated by the LLM. By subtracting the machine-like patterns from the human-like distribution during the decoding process, CoPA is able to produce sentences that are less discernible by text detectors. Our theoretical analysis suggests the superiority of the proposed attack. Extensive experiments validate the effectiveness of CoPA in fooling text detectors across various scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00032v3">Detecting LLM-Generated Korean Text through Linguistic Feature Analysis</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Accepted to ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      The rapid advancement of large language models (LLMs) increases the difficulty of distinguishing between human-written and LLM-generated text. Detecting LLM-generated text is crucial for upholding academic integrity, preventing plagiarism, protecting copyrights, and ensuring ethical research practices. Most prior studies on detecting LLM-generated text focus primarily on English text. However, languages with distinct morphological and syntactic characteristics require specialized detection approaches. Their unique structures and usage patterns can hinder the direct application of methods primarily designed for English. Among such languages, we focus on Korean, which has relatively flexible spacing rules, a rich morphological system, and less frequent comma usage compared to English. We introduce KatFish, the first benchmark dataset for detecting LLM-generated Korean text. The dataset consists of text written by humans and generated by four LLMs across three genres. By examining spacing patterns, part-of-speech diversity, and comma usage, we illuminate the linguistic differences between human-written and LLM-generated Korean text. Building on these observations, we propose KatFishNet, a detection method specifically designed for the Korean language. KatFishNet achieves an average of 19.78% higher AUROC compared to the best-performing existing detection method. Our code and data are available at https://github.com/Shinwoo-Park/detecting_llm_generated_korean_text_through_linguistic_analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19675v1">Calibrating Pre-trained Language Classifiers on LLM-generated Noisy Labels via Iterative Refinement</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      The traditional process of creating labeled datasets is labor-intensive and expensive. Recent breakthroughs in open-source large language models (LLMs) have opened up a new avenue in generating labeled datasets automatically for various natural language processing (NLP) tasks, providing an alternative to such an expensive annotation process. However, the reliability of such auto-generated labels remains a significant concern due to inherent inaccuracies. When learning from noisy labels, the model's generalization is likely to be harmed as it is prone to overfit to those label noises. While previous studies in learning from noisy labels mainly focus on synthetic noise and real-world noise, LLM-generated label noise receives less attention. In this paper, we propose SiDyP: Simplex Label Diffusion with Dynamic Prior to calibrate the classifier's prediction, thus enhancing its robustness towards LLM-generated noisy labels. SiDyP retrieves potential true label candidates by neighborhood label distribution in text embedding space and iteratively refines noisy candidates using a simplex diffusion model. Our framework can increase the performance of the BERT classifier fine-tuned on both zero-shot and few-shot LLM-generated noisy label datasets by an average of 7.21% and 7.30% respectively. We demonstrate the effectiveness of SiDyP by conducting extensive benchmarking for different LLMs over a variety of NLP tasks. Our code is available on Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.14838v3">DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Efficient KV cache management in LLMs is crucial for long-context tasks like RAG and summarization. Existing KV cache compression methods enforce a fixed pattern, neglecting task-specific characteristics and reducing the retention of essential information. However, we observe distinct activation patterns across layers in various tasks, highlighting the need for adaptive strategies tailored to each task's unique demands. Based on this insight, we propose DynamicKV, a method that dynamically optimizes token retention by adjusting the number of tokens retained at each layer to adapt to the specific task. DynamicKV establishes global and per-layer maximum KV cache budgets, temporarily retaining the maximum budget for the current layer, and periodically updating the KV cache sizes of all preceding layers during inference. Our method retains only 1.7% of the KV cache size while achieving ~85% of the Full KV cache performance on LongBench. Notably, even under extreme compression (0.9%), DynamicKV surpasses state-of-the-art (SOTA) methods by 11% in the Needle-in-a-Haystack test using Mistral-7B-Instruct-v0.2. The code will be released.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19674v1">Comparing Moral Values in Western English-speaking societies and LLMs with Word Associations</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 9 pages,7 figures. Accepted to the ACL 2025 conference
    </div>
    <details class="paper-abstract">
      As the impact of large language models increases, understanding the moral values they reflect becomes ever more important. Assessing the nature of moral values as understood by these models via direct prompting is challenging due to potential leakage of human norms into model training data, and their sensitivity to prompt formulation. Instead, we propose to use word associations, which have been shown to reflect moral reasoning in humans, as low-level underlying representations to obtain a more robust picture of LLMs' moral reasoning. We study moral differences in associations from western English-speaking communities and LLMs trained predominantly on English data. First, we create a large dataset of LLM-generated word associations, resembling an existing data set of human word associations. Next, we propose a novel method to propagate moral values based on seed words derived from Moral Foundation Theory through the human and LLM-generated association graphs. Finally, we compare the resulting moral conceptualizations, highlighting detailed but systematic differences between moral values emerging from English speakers and LLM associations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.04285v4">Separate Source Channel Coding Is Still What You Need: An LLM-based Rethinking</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Along with the proliferating research interest in Semantic Communication (SemCom), Joint Source Channel Coding (JSCC) has dominated the attention due to the widely assumed existence in efficiently delivering information semantics. Nevertheless, this paper challenges the conventional JSCC paradigm, and advocates for adoption of Separate Source Channel Coding (SSCC) to enjoy the underlying more degree of freedom for optimization. We demonstrate that SSCC, after leveraging the strengths of Large Language Model (LLM) for source coding and Error Correction Code Transformer (ECCT) complemented for channel decoding, offers superior performance over JSCC. Our proposed framework also effectively highlights the compatibility challenges between SemCom approaches and digital communication systems, particularly concerning the resource costs associated with the transmission of high precision floating point numbers. Through comprehensive evaluations, we establish that empowered by LLM-based compression and ECCT-enhanced error correction, SSCC remains a viable and effective solution for modern communication systems. In other words, separate source and channel coding is still what we need!
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12924v2">Conditioning LLMs to Generate Code-Switched Text</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 [v2]Added new experiments and analyses
    </div>
    <details class="paper-abstract">
      Code-switching (CS) is still a critical challenge in Natural Language Processing (NLP). Current Large Language Models (LLMs) struggle to interpret and generate code-switched text, primarily due to the scarcity of large-scale CS datasets for training. This paper presents a novel methodology to generate CS data using LLMs, and test it on the English-Spanish language pair. We propose back-translating natural CS sentences into monolingual English, and using the resulting parallel corpus to fine-tune LLMs to turn monolingual sentences into CS. Unlike previous approaches to CS generation, our methodology uses natural CS data as a starting point, allowing models to learn its natural distribution beyond grammatical patterns. We thoroughly analyse the models' performance through a study on human preferences, a qualitative error analysis and an evaluation with popular automatic metrics. Results show that our methodology generates fluent code-switched text, expanding research opportunities in CS communication, and that traditional metrics do not correlate with human judgement when assessing the quality of the generated CS data. We release our code and generated dataset under a CC-BY-NC-SA license.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14924v2">A Tale of Two Structures: Do LLMs Capture the Fractal Complexity of Language?</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Language exhibits a fractal structure in its information-theoretic complexity (i.e. bits per token), with self-similarity across scales and long-range dependence (LRD). In this work, we investigate whether large language models (LLMs) can replicate such fractal characteristics and identify conditions-such as temperature setting and prompting method-under which they may fail. Moreover, we find that the fractal parameters observed in natural language are contained within a narrow range, whereas those of LLMs' output vary widely, suggesting that fractal parameters might prove helpful in detecting a non-trivial portion of LLM-generated texts. Notably, these findings, and many others reported in this work, are robust to the choice of the architecture; e.g. Gemini 1.0 Pro, Mistral-7B and Gemma-2B. We also release a dataset comprising of over 240,000 articles generated by various LLMs (both pretrained and instruction-tuned) with different decoding temperatures and prompting methods, along with their corresponding human-generated texts. We hope that this work highlights the complex interplay between fractal properties, prompting, and statistical mimicry in LLMs, offering insights for generating, evaluating and detecting synthetic texts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19634v1">Faster and Better LLMs via Latency-Aware Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Test-Time Scaling (TTS) has proven effective in improving the performance of Large Language Models (LLMs) during inference. However, existing research has overlooked the efficiency of TTS from a latency-sensitive perspective. Through a latency-aware evaluation of representative TTS methods, we demonstrate that a compute-optimal TTS does not always result in the lowest latency in scenarios where latency is critical. To address this gap and achieve latency-optimal TTS, we propose two key approaches by optimizing the concurrency configurations: (1) branch-wise parallelism, which leverages multiple concurrent inference branches, and (2) sequence-wise parallelism, enabled by speculative decoding. By integrating these two approaches and allocating computational resources properly to each, our latency-optimal TTS enables a 32B model to reach 82.3% accuracy on MATH-500 within 1 minute and a smaller 3B model to achieve 72.4% within 10 seconds. Our work emphasizes the importance of latency-aware TTS and demonstrates its ability to deliver both speed and accuracy in latency-sensitive scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19628v1">HomeBench: Evaluating LLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multiple Devices</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have the potential to revolutionize smart home assistants by enhancing their ability to accurately understand user needs and respond appropriately, which is extremely beneficial for building a smarter home environment. While recent studies have explored integrating LLMs into smart home systems, they primarily focus on handling straightforward, valid single-device operation instructions. However, real-world scenarios are far more complex and often involve users issuing invalid instructions or controlling multiple devices simultaneously. These have two main challenges: LLMs must accurately identify and rectify errors in user instructions and execute multiple user instructions perfectly. To address these challenges and advance the development of LLM-based smart home assistants, we introduce HomeBench, the first smart home dataset with valid and invalid instructions across single and multiple devices in this paper. We have experimental results on 13 distinct LLMs; e.g., GPT-4o achieves only a 0.0% success rate in the scenario of invalid multi-device instructions, revealing that the existing state-of-the-art LLMs still cannot perform well in this situation even with the help of in-context learning, retrieval-augmented generation, and fine-tuning. Our code and dataset are publicly available at https://github.com/BITHLP/HomeBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19623v1">AgentRecBench: Benchmarking LLM Agent-based Personalized Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The emergence of agentic recommender systems powered by Large Language Models (LLMs) represents a paradigm shift in personalized recommendations, leveraging LLMs' advanced reasoning and role-playing capabilities to enable autonomous, adaptive decision-making. Unlike traditional recommendation approaches, agentic recommender systems can dynamically gather and interpret user-item interactions from complex environments, generating robust recommendation strategies that generalize across diverse scenarios. However, the field currently lacks standardized evaluation protocols to systematically assess these methods. To address this critical gap, we propose: (1) an interactive textual recommendation simulator incorporating rich user and item metadata and three typical evaluation scenarios (classic, evolving-interest, and cold-start recommendation tasks); (2) a unified modular framework for developing and studying agentic recommender systems; and (3) the first comprehensive benchmark comparing 10 classical and agentic recommendation methods. Our findings demonstrate the superiority of agentic systems and establish actionable design guidelines for their core components. The benchmark environment has been rigorously validated through an open challenge and remains publicly available with a continuously maintained leaderboard~\footnote[2]{https://tsinghua-fib-lab.github.io/AgentSocietyChallenge/pages/overview.html}, fostering ongoing community engagement and reproducible research. The benchmark is available at: \hyperlink{https://huggingface.co/datasets/SGJQovo/AgentRecBench}{https://huggingface.co/datasets/SGJQovo/AgentRecBench}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.08192v2">PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are typically served from clusters of GPUs/NPUs that consist of large number of devices. Unfortunately, communication between these devices incurs significant overhead, increasing the inference latency and cost while limiting the scalability. Prior work addressed this issue by overlapping communication with compute, but has severe limitations due to the data dependencies between these operations. In this paper, we propose PRESERVE, a novel framework that prefetches model weights and KV-cache from off-chip HBM memory to the on-chip cache of AI accelerators during the communication operations, which offers various advantages and performance improvements compared to prior methods. Through extensive experiments conducted on commercial AI accelerators, we demonstrate up to 1.6x end-to-end speedup on state-of-the-art, open-source LLMs. Additionally, we perform a design space exploration that identifies the optimal hardware configuration for the proposed method, showing a further 1.25x improvement in performance per cost by selecting the optimal L2 cache size. Our results show that PRESERVE has the potential to mitigate the memory bottlenecks and communication overheads, offering a solution to improve the performance and scalability of the LLM inference systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16520v2">Are the Hidden States Hiding Something? Testing the Limits of Factuality-Encoding Capabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Factual hallucinations are a major challenge for Large Language Models (LLMs). They undermine reliability and user trust by generating inaccurate or fabricated content. Recent studies suggest that when generating false statements, the internal states of LLMs encode information about truthfulness. However, these studies often rely on synthetic datasets that lack realism, which limits generalization when evaluating the factual accuracy of text generated by the model itself. In this paper, we challenge the findings of previous work by investigating truthfulness encoding capabilities, leading to the generation of a more realistic and challenging dataset. Specifically, we extend previous work by introducing: (1) a strategy for sampling plausible true-false factoid sentences from tabular data and (2) a procedure for generating realistic, LLM-dependent true-false datasets from Question Answering collections. Our analysis of two open-source LLMs reveals that while the findings from previous studies are partially validated, generalization to LLM-generated datasets remains challenging. This study lays the groundwork for future research on factuality in LLMs and offers practical guidelines for more effective evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19567v1">LLM-Agent-Controller: A Universal Multi-Agent Large Language Model System as a Control Engineer</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      This study presents the LLM-Agent-Controller, a multi-agent large language model (LLM) system developed to address a wide range of problems in control engineering (Control Theory). The system integrates a central controller agent with multiple specialized auxiliary agents, responsible for tasks such as controller design, model representation, control analysis, time-domain response, and simulation. A supervisor oversees high-level decision-making and workflow coordination, enhancing the system's reliability and efficiency. The LLM-Agent-Controller incorporates advanced capabilities, including Retrieval-Augmented Generation (RAG), Chain-of-Thought reasoning, self-criticism and correction, efficient memory handling, and user-friendly natural language communication. It is designed to function without requiring users to have prior knowledge of Control Theory, enabling them to input problems in plain language and receive complete, real-time solutions. To evaluate the system, we propose new performance metrics assessing both individual agents and the system as a whole. We test five categories of Control Theory problems and benchmark performance across three advanced LLMs. Additionally, we conduct a comprehensive qualitative conversational analysis covering all key services. Results show that the LLM-Agent-Controller successfully solved 83% of general tasks, with individual agents achieving an average success rate of 87%. Performance improved with more advanced LLMs. This research demonstrates the potential of multi-agent LLM architectures to solve complex, domain-specific problems. By integrating specialized agents, supervisory control, and advanced reasoning, the LLM-Agent-Controller offers a scalable, robust, and accessible solution framework that can be extended to various technical domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19562v1">AMQA: An Adversarial Dataset for Benchmarking Bias of LLMs in Medicine and Healthcare</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are reaching expert-level accuracy on medical diagnosis questions, yet their mistakes and the biases behind them pose life-critical risks. Bias linked to race, sex, and socioeconomic status is already well known, but a consistent and automatic testbed for measuring it is missing. To fill this gap, this paper presents AMQA -- an Adversarial Medical Question-Answering dataset -- built for automated, large-scale bias evaluation of LLMs in medical QA. AMQA includes 4,806 medical QA pairs sourced from the United States Medical Licensing Examination (USMLE) dataset, generated using a multi-agent framework to create diverse adversarial descriptions and question pairs. Using AMQA, we benchmark five representative LLMs and find surprisingly substantial disparities: even GPT-4.1, the least biased model tested, answers privileged-group questions over 10 percentage points more accurately than unprivileged ones. Compared with the existing benchmark CPV, AMQA reveals 15% larger accuracy gaps on average between privileged and unprivileged groups. Our dataset and code are publicly available at https://github.com/XY-Showing/AMQA to support reproducible research and advance trustworthy, bias-aware medical AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12904v2">Fraud-R1 : A Multi-Round Benchmark for Assessing the Robustness of LLM Against Augmented Fraud and Phishing Inducements</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Accepted by ACL2025 Findings
    </div>
    <details class="paper-abstract">
      We introduce Fraud-R1, a benchmark designed to evaluate LLMs' ability to defend against internet fraud and phishing in dynamic, real-world scenarios. Fraud-R1 comprises 8,564 fraud cases sourced from phishing scams, fake job postings, social media, and news, categorized into 5 major fraud types. Unlike previous benchmarks, Fraud-R1 introduces a multi-round evaluation pipeline to assess LLMs' resistance to fraud at different stages, including credibility building, urgency creation, and emotional manipulation. Furthermore, we evaluate 15 LLMs under two settings: 1. Helpful-Assistant, where the LLM provides general decision-making assistance, and 2. Role-play, where the model assumes a specific persona, widely used in real-world agent-based interactions. Our evaluation reveals the significant challenges in defending against fraud and phishing inducement, especially in role-play settings and fake job postings. Additionally, we observe a substantial performance gap between Chinese and English, underscoring the need for improved multilingual fraud detection capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19148v2">Amulet: ReAlignment During Test Time for Personalized Preference Adaptation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Accepted by ICLR 2025, Project page: https://zowiezhang.github.io/projects/Amulet
    </div>
    <details class="paper-abstract">
      How to align large language models (LLMs) with user preferences from a static general dataset has been frequently studied. However, user preferences are usually personalized, changing, and diverse regarding culture, values, or time. This leads to the problem that the actual user preferences often do not coincide with those trained by the model developers in the practical use of LLMs. Since we cannot collect enough data and retrain for every demand, researching efficient real-time preference adaptation methods based on the backbone LLMs during test time is important. To this end, we introduce Amulet, a novel, training-free framework that formulates the decoding process of every token as a separate online learning problem with the guidance of simple user-provided prompts, thus enabling real-time optimization to satisfy users' personalized preferences. To reduce the computational cost brought by this optimization process for each token, we additionally provide a closed-form solution for each iteration step of the optimization process, thereby reducing the computational time cost to a negligible level. The detailed experimental results demonstrate that Amulet can achieve significant performance improvements in rich settings with combinations of different LLMs, datasets, and user preferences, while maintaining acceptable computational efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08704v2">LLM-based Prompt Ensemble for Reliable Medical Entity Recognition from EHRs</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 IEEE 26th International Conference on Information Reuse and Integration for Data Science (IRI 2025), San Jose, CA, USA
    </div>
    <details class="paper-abstract">
      Electronic Health Records (EHRs) are digital records of patient information, often containing unstructured clinical text. Named Entity Recognition (NER) is essential in EHRs for extracting key medical entities like problems, tests, and treatments to support downstream clinical applications. This paper explores prompt-based medical entity recognition using large language models (LLMs), specifically GPT-4o and DeepSeek-R1, guided by various prompt engineering techniques, including zero-shot, few-shot, and an ensemble approach. Among all strategies, GPT-4o with prompt ensemble achieved the highest classification performance with an F1-score of 0.95 and recall of 0.98, outperforming DeepSeek-R1 on the task. The ensemble method improved reliability by aggregating outputs through embedding-based similarity and majority voting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20006v2">Chatbot Arena Meets Nuggets: Towards Explanations and Diagnostics in the Evaluation of LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 10 pages, 8 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Battles, or side-by-side comparisons in so-called arenas that elicit human preferences, have emerged as a popular approach for assessing the output quality of LLMs. Recently, this idea has been extended to retrieval-augmented generation (RAG) systems. While undoubtedly representing an advance in evaluation, battles have at least two drawbacks, particularly in the context of complex information-seeking queries: they are neither explanatory nor diagnostic. Recently, the nugget evaluation methodology has emerged as a promising approach to evaluate the quality of RAG answers. Nuggets decompose long-form LLM-generated answers into atomic facts, highlighting important pieces of information necessary in a "good" response. In this work, we apply our AutoNuggetizer framework to analyze data from roughly 7K Search Arena battles provided by LMArena in a fully automatic manner. Our results show a significant correlation between nugget scores and human preferences, showcasing promise in our approach to explainable and diagnostic system evaluations. All the code necessary to reproduce results in our work is available in https://github.com/castorini/lmsys_nuggetize.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19300v1">SituatedThinker: Grounding LLM Reasoning with Real-World through Situated Thinking</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) demonstrate their impressive reasoning capabilities. However, the reasoning confined to internal parametric space limits LLMs' access to real-time information and understanding of the physical world. To overcome this constraint, we introduce SituatedThinker, a novel framework that enables LLMs to ground their reasoning in real-world contexts through situated thinking, which adaptively combines both internal knowledge and external information with predefined interfaces. By utilizing reinforcement learning, SituatedThinker incentivizes deliberate reasoning with the real world to acquire information and feedback, allowing LLMs to surpass their knowledge boundaries and enhance reasoning. Experimental results demonstrate significant performance improvements on multi-hop question-answering and mathematical reasoning benchmarks. Furthermore, SituatedThinker demonstrates strong performance on unseen tasks, such as KBQA, TableQA, and text-based games, showcasing the generalizable real-world grounded reasoning capability. Our codes are available at https://github.com/jnanliu/SituatedThinker.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20749v5">Prompting is Not All You Need! Evaluating LLM Agent Simulation Methodologies with Real-World Online Customer Behavior Data</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      Recent research shows that LLMs can simulate ``believable'' human behaviors to power LLM agents via prompt-only methods. In this work, we focus on evaluating LLM's objective ``accuracy'' rather than the subjective ``believability'' in simulating human behavior, leveraging a large-scale, real-world dataset collected from customers' online shopping actions. We present the first comprehensive evaluation of state-of-the-art LLMs (e.g., DeepSeek-R1, Llama, and Claude) on the task of web shopping action generation. Our results show that out-of-the-box LLM-generated actions are often misaligned with actual human behavior, whereas fine-tuning LLMs on real-world behavioral data substantially improves their ability to generate accurate actions compared to prompt-only methods. Furthermore, incorporating synthesized reasonings into model training leads to additional performance gains, demonstrating the value of explicit rationale in behavior modeling. This work evaluates state-of-the-art LLMs in behavior simulation and provides actionable insights into how real-world action data can enhance the fidelity of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19284v1">RankLLM: A Python Package for Reranking with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 SIGIR 2025
    </div>
    <details class="paper-abstract">
      The adoption of large language models (LLMs) as rerankers in multi-stage retrieval systems has gained significant traction in academia and industry. These models refine a candidate list of retrieved documents, often through carefully designed prompts, and are typically used in applications built on retrieval-augmented generation (RAG). This paper introduces RankLLM, an open-source Python package for reranking that is modular, highly configurable, and supports both proprietary and open-source LLMs in customized reranking workflows. To improve usability, RankLLM features optional integration with Pyserini for retrieval and provides integrated evaluation for multi-stage pipelines. Additionally, RankLLM includes a module for detailed analysis of input prompts and LLM responses, addressing reliability concerns with LLM APIs and non-deterministic behavior in Mixture-of-Experts (MoE) models. This paper presents the architecture of RankLLM, along with a detailed step-by-step guide and sample code. We reproduce results from RankGPT, LRL, RankVicuna, RankZephyr, and other recent models. RankLLM integrates with common inference frameworks and a wide range of LLMs. This compatibility allows for quick reproduction of reported results, helping to speed up both research and real-world applications. The complete repository is available at rankllm.ai, and the package can be installed via PyPI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19400v2">TheoremExplainAgent: Towards Video-based Multimodal Explanations for LLM Theorem Understanding</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 accepted to ACL 2025 main, camera ready
    </div>
    <details class="paper-abstract">
      Understanding domain-specific theorems often requires more than just text-based reasoning; effective communication through structured visual explanations is crucial for deeper comprehension. While large language models (LLMs) demonstrate strong performance in text-based theorem reasoning, their ability to generate coherent and pedagogically meaningful visual explanations remains an open challenge. In this work, we introduce TheoremExplainAgent, an agentic approach for generating long-form theorem explanation videos (over 5 minutes) using Manim animations. To systematically evaluate multimodal theorem explanations, we propose TheoremExplainBench, a benchmark covering 240 theorems across multiple STEM disciplines, along with 5 automated evaluation metrics. Our results reveal that agentic planning is essential for generating detailed long-form videos, and the o3-mini agent achieves a success rate of 93.8% and an overall score of 0.77. However, our quantitative and qualitative studies show that most of the videos produced exhibit minor issues with visual element layout. Furthermore, multimodal explanations expose deeper reasoning flaws that text-based explanations fail to reveal, highlighting the importance of multimodal explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20251v2">ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 Accepted to ACL 2025 findings
    </div>
    <details class="paper-abstract">
      The rapid development of LLMs has sparked extensive research into their factual knowledge. Current works find that LLMs fall short on questions around low-frequency entities. However, such proofs are unreliable since the questions can differ not only in entity frequency but also in difficulty themselves. So we introduce ComparisonQA benchmark, containing 283K abstract questions, each instantiated by a pair of high-frequency and low-frequency entities. It ensures a controllable comparison to study the role of knowledge frequency in the performance of LLMs. Because the difference between such a pair is only the entity with different frequencies. In addition, we use both correctness and uncertainty to develop a two-round method to evaluate LLMs' knowledge robustness. It aims to avoid possible semantic shortcuts which is a serious problem of current QA study. Experiments reveal that LLMs, including GPT-4o, exhibit particularly low robustness regarding low-frequency knowledge. Besides, we find that uncertainty can be used to effectively identify high-quality and shortcut-free questions while maintaining the data size. Based on this, we propose an automatic method to select such questions to form a subset called ComparisonQA-Hard, containing only hard low-frequency questions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19234v1">GUARDIAN: Safeguarding LLM Multi-Agent Collaborations with Temporal Graph Modeling</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) enables the development of intelligent agents capable of engaging in complex and multi-turn dialogues. However, multi-agent collaboration face critical safety challenges, such as hallucination amplification and error injection and propagation. This paper presents GUARDIAN, a unified method for detecting and mitigating multiple safety concerns in GUARDing Intelligent Agent collaboratioNs. By modeling the multi-agent collaboration process as a discrete-time temporal attributed graph, GUARDIAN explicitly captures the propagation dynamics of hallucinations and errors. The unsupervised encoder-decoder architecture incorporating an incremental training paradigm, learns to reconstruct node attributes and graph structures from latent embeddings, enabling the identification of anomalous nodes and edges with unparalleled precision. Moreover, we introduce a graph abstraction mechanism based on the Information Bottleneck Theory, which compresses temporal interaction graphs while preserving essential patterns. Extensive experiments demonstrate GUARDIAN's effectiveness in safeguarding LLM multi-agent collaborations against diverse safety vulnerabilities, achieving state-of-the-art accuracy with efficient resource utilization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.19299v2">The Impact of LoRA Adapters for LLMs on Clinical NLP Classification Under Data Limitations</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 Under revisions
    </div>
    <details class="paper-abstract">
      Fine-tuning Large Language Models (LLMs) for clinical Natural Language Processing (NLP) poses significant challenges due to the domain gap and limited data availability. This study investigates the effectiveness of various adapter techniques, equivalent to Low-Rank Adaptation (LoRA), for fine-tuning LLMs in a resource-constrained hospital environment. We experimented with four structures-Adapter, Lightweight, TinyAttention, and Gated Residual Network (GRN)-as final layers for clinical notes classification. We fine-tuned biomedical pre-trained models, including CamemBERT-bio, AliBERT, and DrBERT, alongside two Transformer-based models. Our extensive experimental results indicate that i) employing adapter structures does not yield significant improvements in fine-tuning biomedical pre-trained LLMs, and ii) simpler Transformer-based models, trained from scratch, perform better under resource constraints. Among the adapter structures, GRN demonstrated superior performance with accuracy, precision, recall, and an F1 score of 0.88. Moreover, the total training time for LLMs exceeded 1000 hours, compared to under 6 hours for simpler transformer-based models, highlighting that LLMs are more suitable for environments with extensive computational resources and larger datasets. Consequently, this study demonstrates that simpler Transformer-based models can be effectively trained from scratch, providing a viable solution for clinical NLP tasks in low-resource environments with limited data availability. By identifying the GRN as the most effective adapter structure, we offer a practical approach to enhance clinical note classification without requiring extensive computational resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19212v1">When Ethics and Payoffs Diverge: LLM Agents in Morally Charged Social Dilemmas</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have enabled their use in complex agentic roles, involving decision-making with humans or other agents, making ethical alignment a key AI safety concern. While prior work has examined both LLMs' moral judgment and strategic behavior in social dilemmas, there is limited understanding of how they act when moral imperatives directly conflict with rewards or incentives. To investigate this, we introduce Moral Behavior in Social Dilemma Simulation (MoralSim) and evaluate how LLMs behave in the prisoner's dilemma and public goods game with morally charged contexts. In MoralSim, we test a range of frontier models across both game structures and three distinct moral framings, enabling a systematic examination of how LLMs navigate social dilemmas in which ethical norms conflict with payoff-maximizing strategies. Our results show substantial variation across models in both their general tendency to act morally and the consistency of their behavior across game types, the specific moral framing, and situational factors such as opponent behavior and survival risks. Crucially, no model exhibits consistently moral behavior in MoralSim, highlighting the need for caution when deploying LLMs in agentic roles where the agent's "self-interest" may conflict with ethical expectations. Our code is available at https://github.com/sbackmann/moralsim.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19209v1">MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery via Hierarchical Search</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in automating scientific hypothesis generation, yet existing approaches primarily yield coarse-grained hypotheses lacking critical methodological and experimental details. We introduce and formally define the novel task of fine-grained scientific hypothesis discovery, which entails generating detailed, experimentally actionable hypotheses from coarse initial research directions. We frame this as a combinatorial optimization problem and investigate the upper limits of LLMs' capacity to solve it when maximally leveraged. Specifically, we explore four foundational questions: (1) how to best harness an LLM's internal heuristics to formulate the fine-grained hypothesis it itself would judge as the most promising among all the possible hypotheses it might generate, based on its own internal scoring-thus defining a latent reward landscape over the hypothesis space; (2) whether such LLM-judged better hypotheses exhibit stronger alignment with ground-truth hypotheses; (3) whether shaping the reward landscape using an ensemble of diverse LLMs of similar capacity yields better outcomes than defining it with repeated instances of the strongest LLM among them; and (4) whether an ensemble of identical LLMs provides a more reliable reward landscape than a single LLM. To address these questions, we propose a hierarchical search method that incrementally proposes and integrates details into the hypothesis, progressing from general concepts to specific experimental configurations. We show that this hierarchical process smooths the reward landscape and enables more effective optimization. Empirical evaluations on a new benchmark of expert-annotated fine-grained hypotheses from recent chemistry literature show that our method consistently outperforms strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19184v1">Two LLMs debate, both are certain they've won</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      Can LLMs accurately adjust their confidence when facing opposition? Building on previous studies measuring calibration on static fact-based question-answering tasks, we evaluate Large Language Models (LLMs) in a dynamic, adversarial debate setting, uniquely combining two realistic factors: (a) a multi-turn format requiring models to update beliefs as new information emerges, and (b) a zero-sum structure to control for task-related uncertainty, since mutual high-confidence claims imply systematic overconfidence. We organized 60 three-round policy debates among ten state-of-the-art LLMs, with models privately rating their confidence (0-100) in winning after each round. We observed five concerning patterns: (1) Systematic overconfidence: models began debates with average initial confidence of 72.9% vs. a rational 50% baseline. (2) Confidence escalation: rather than reducing confidence as debates progressed, debaters increased their win probabilities, averaging 83% by the final round. (3) Mutual overestimation: in 61.7% of debates, both sides simultaneously claimed >=75% probability of victory, a logical impossibility. (4) Persistent self-debate bias: models debating identical copies increased confidence from 64.1% to 75.2%; even when explicitly informed their chance of winning was exactly 50%, confidence still rose (from 50.0% to 57.1%). (5) Misaligned private reasoning: models' private scratchpad thoughts sometimes differed from their public confidence ratings, raising concerns about faithfulness of chain-of-thought reasoning. These results suggest LLMs lack the ability to accurately self-assess or update their beliefs in dynamic, multi-turn tasks; a major concern as LLM outputs are deployed without careful review in assistant roles or agentic settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00985v3">Position: Enough of Scaling LLMs! Lets Focus on Downscaling</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      We challenge the dominant focus on neural scaling laws and advocate for a paradigm shift toward downscaling in the development of large language models (LLMs). While scaling laws have provided critical insights into performance improvements through increasing model and dataset size, we emphasize the significant limitations of this approach, particularly in terms of computational inefficiency, environmental impact, and deployment constraints. To address these challenges, we propose a holistic framework for downscaling LLMs that seeks to maintain performance while drastically reducing resource demands. This paper outlines practical strategies for transitioning away from traditional scaling paradigms, advocating for a more sustainable, efficient, and accessible approach to LLM development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19176v1">Assistant-Guided Mitigation of Teacher Preference Bias in LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      LLM-as-a-Judge employs large language models (LLMs), such as GPT-4, to evaluate the quality of LLM-generated responses, gaining popularity for its cost-effectiveness and strong alignment with human evaluations. However, training proxy judge models using evaluation data generated by powerful teacher models introduces a critical yet previously overlooked issue: teacher preference bias, where the proxy judge model learns a biased preference for responses from the teacher model. To tackle this problem, we propose a novel setting that incorporates an additional assistant model, which is not biased toward the teacher model's responses, to complement the training data. Building on this setup, we introduce AGDe-Judge, a three-stage framework designed to debias from both the labels and feedbacks in the training data. Extensive experiments demonstrate that AGDe-Judge effectively reduces teacher preference bias while maintaining strong performance across six evaluation benchmarks. Code is available at https://github.com/Liuz233/AGDe-Judge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19173v1">Investigating Pedagogical Teacher and Student LLM Agents: Genetic Adaptation Meets Retrieval Augmented Generation Across Learning Style</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 38 Pages
    </div>
    <details class="paper-abstract">
      Effective teaching requires adapting instructional strategies to accommodate the diverse cognitive and behavioral profiles of students, a persistent challenge in education and teacher training. While Large Language Models (LLMs) offer promise as tools to simulate such complex pedagogical environments, current simulation frameworks are limited in two key respects: (1) they often reduce students to static knowledge profiles, and (2) they lack adaptive mechanisms for modeling teachers who evolve their strategies in response to student feedback. To address these gaps, \textbf{we introduce a novel simulation framework that integrates LLM-based heterogeneous student agents with a self-optimizing teacher agent}. The teacher agent's pedagogical policy is dynamically evolved using a genetic algorithm, allowing it to discover and refine effective teaching strategies based on the aggregate performance of diverse learners. In addition, \textbf{we propose Persona-RAG}, a Retrieval Augmented Generation module that enables student agents to retrieve knowledge tailored to their individual learning styles. Persona-RAG preserves the retrieval accuracy of standard RAG baselines while enhancing personalization, an essential factor in modeling realistic educational scenarios. Through extensive experiments, we demonstrate how our framework supports the emergence of distinct and interpretable teaching patterns when interacting with varied student populations. Our results highlight the potential of LLM-driven simulations to inform adaptive teaching practices and provide a testbed for training human educators in controlled, data-driven environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19165v1">OrgAccess: A Benchmark for Role Based Access Control in Organization Scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 56 Pages
    </div>
    <details class="paper-abstract">
      Role-based access control (RBAC) and hierarchical structures are foundational to how information flows and decisions are made within virtually all organizations. As the potential of Large Language Models (LLMs) to serve as unified knowledge repositories and intelligent assistants in enterprise settings becomes increasingly apparent, a critical, yet under explored, challenge emerges: \textit{can these models reliably understand and operate within the complex, often nuanced, constraints imposed by organizational hierarchies and associated permissions?} Evaluating this crucial capability is inherently difficult due to the proprietary and sensitive nature of real-world corporate data and access control policies. We introduce a synthetic yet representative \textbf{OrgAccess} benchmark consisting of 40 distinct types of permissions commonly relevant across different organizational roles and levels. We further create three types of permissions: 40,000 easy (1 permission), 10,000 medium (3-permissions tuple), and 20,000 hard (5-permissions tuple) to test LLMs' ability to accurately assess these permissions and generate responses that strictly adhere to the specified hierarchical rules, particularly in scenarios involving users with overlapping or conflicting permissions. Our findings reveal that even state-of-the-art LLMs struggle significantly to maintain compliance with role-based structures, even with explicit instructions, with their performance degrades further when navigating interactions involving two or more conflicting permissions. Specifically, even \textbf{GPT-4.1 only achieves an F1-Score of 0.27 on our hardest benchmark}. This demonstrates a critical limitation in LLMs' complex rule following and compositional reasoning capabilities beyond standard factual or STEM-based benchmarks, opening up a new paradigm for evaluating their fitness for practical, structured environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19163v1">SpokenNativQA: Multilingual Everyday Spoken Queries for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 Spoken Question Answering, Multilingual LLMs, Speech-based Evaluation, Dialectal Speech, Low-resource Languages, Multimodal Benchmarking, Conversational AI, Speech-to-Text QA, Real-world Interaction, Natural Language Understanding
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance across various disciplines and tasks. However, benchmarking their capabilities with multilingual spoken queries remains largely unexplored. In this study, we introduce SpokenNativQA, the first multilingual and culturally aligned spoken question-answering (SQA) dataset designed to evaluate LLMs in real-world conversational settings. The dataset comprises approximately 33,000 naturally spoken questions and answers in multiple languages, including low-resource and dialect-rich languages, providing a robust benchmark for assessing LLM performance in speech-based interactions. SpokenNativQA addresses the limitations of text-based QA datasets by incorporating speech variability, accents, and linguistic diversity. We benchmark different ASR systems and LLMs for SQA and present our findings. We released the data at (https://huggingface.co/datasets/QCRI/SpokenNativQA) and the experimental scripts at (https://llmebench.qcri.org/) for the research community.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19155v1">Sparse-to-Dense: A Free Lunch for Lossless Acceleration of Video Understanding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      Due to the auto-regressive nature of current video large language models (Video-LLMs), the inference latency increases as the input sequence length grows, posing challenges for the efficient processing of video sequences that are usually very long. We observe that during decoding, the attention scores of most tokens in Video-LLMs tend to be sparse and concentrated, with only certain tokens requiring comprehensive full attention. Based on this insight, we introduce Sparse-to-Dense (StD), a novel decoding strategy that integrates two distinct modules: one leveraging sparse top-K attention and the other employing dense full attention. These modules collaborate to accelerate Video-LLMs without loss. The fast (sparse) model speculatively decodes multiple tokens, while the slow (dense) model verifies them in parallel. StD is a tuning-free, plug-and-play solution that achieves up to a 1.94$\times$ walltime speedup in video processing. It maintains model performance while enabling a seamless transition from a standard Video-LLM to a sparse Video-LLM with minimal code modifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17189v2">Better Think with Tables: Tabular Structures Enhance LLM Comprehension for Data-Analytics Requests</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 20 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often struggle with data-analytics requests related to information retrieval and data manipulation that frequently arise in real-world scenarios under multiple conditions. In this paper, we introduce Thinking with Tables, where we inject tabular structures into LLMs for data-analytics requests. Through comprehensive evaluations across various request types, we show that providing tabular structures yields a 40.29 percent average performance gain along with better robustness and token efficiency. Through attention-value analysis, we uncover that tables help LLMs better attend to relevant information, explaining these improvements. Beyond tables and text, we evaluate whether (1) blending structuredness within text, such as providing templates or fixing the order of attributes, and (2) other representative structures, such as knowledge graphs and JSON, are helpful. We observe that utilizing tables offers the best balance between efficiency and effectiveness. These advantages remain consistent under increased task complexity and even when all input data cannot be structured. Finally, as data analytics typically relies on structured factual inputs, our text-to-table conversion demonstrates the method's applicability to text-compatible data sources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14634v2">CER: Confidence Enhanced Reasoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 Accepted at ACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Ensuring the reliability of Large Language Models (LLMs) in complex reasoning tasks remains a formidable challenge, particularly in scenarios that demand precise mathematical calculations and knowledge-intensive open-domain generation. In this work, we introduce an uncertainty-aware framework designed to enhance the accuracy of LLM responses by systematically incorporating model confidence at critical decision points. We propose an approach that encourages multi-step reasoning in LLMs and quantify the confidence of intermediate answers such as numerical results in mathematical reasoning and proper nouns in open-domain generation. Then, the overall confidence of each reasoning chain is evaluated based on confidence of these critical intermediate steps. Finally, we aggregate the answer of generated response paths in a way that reflects the reliability of each generated content (as opposed to self-consistency in which each generated chain contributes equally to majority voting). We conducted extensive experiments in five datasets, three mathematical datasets and two open-domain datasets, using four LLMs. The results consistently validate the effectiveness of our novel confidence aggregation method, leading to an accuracy improvement of up to 7.4% and 5.8% over baseline approaches in math and open-domain generation tasks, respectively. Code is publicly available at https://github.com/ Aquasar11/CER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19115v1">FP4 All the Way: Fully Quantized Training of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      We demonstrate, for the first time, fully quantized training (FQT) of large language models (LLMs) using predominantly 4-bit floating-point (FP4) precision for weights, activations, and gradients on datasets up to 200 billion tokens. We extensively investigate key design choices for FP4, including block sizes, scaling formats, and rounding methods. Our analysis shows that the NVFP4 format, where each block of 16 FP4 values (E2M1) shares a scale represented in E4M3, provides optimal results. We use stochastic rounding for backward and update passes and round-to-nearest for the forward pass to enhance stability. Additionally, we identify a theoretical and empirical threshold for effective quantized training: when the gradient norm falls below approximately $\sqrt{3}$ times the quantization noise, quantized training becomes less effective. Leveraging these insights, we successfully train a 7-billion-parameter model on 256 Intel Gaudi2 accelerators. The resulting FP4-trained model achieves downstream task performance comparable to a standard BF16 baseline, confirming that FP4 training is a practical and highly efficient approach for large-scale LLM training. A reference implementation is supplied in https://github.com/Anonymous1252022/fp4-all-the-way .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19092v1">Reinforced Latent Reasoning for LLM-based Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive reasoning capabilities in complex problem-solving tasks, sparking growing interest in their application to preference reasoning in recommendation systems. Existing methods typically rely on fine-tuning with explicit chain-of-thought (CoT) data. However, these methods face significant practical limitations due to (1) the difficulty of obtaining high-quality CoT data in recommendation and (2) the high inference latency caused by generating CoT reasoning. In this work, we explore an alternative approach that shifts from explicit CoT reasoning to compact, information-dense latent reasoning. This approach eliminates the need for explicit CoT generation and improves inference efficiency, as a small set of latent tokens can effectively capture the entire reasoning process. Building on this idea, we propose $\textit{\underline{R}einforced \underline{Latent} \underline{R}easoning for \underline{R}ecommendation}$ (LatentR$^3$), a novel end-to-end training framework that leverages reinforcement learning (RL) to optimize latent reasoning without relying on any CoT data.LatentR$^3$ adopts a two-stage training strategy: first, supervised fine-tuning to initialize the latent reasoning module, followed by pure RL training to encourage exploration through a rule-based reward design. Our RL implementation is based on a modified GRPO algorithm, which reduces computational overhead during training and introduces continuous reward signals for more efficient learning. Extensive experiments demonstrate that LatentR$^3$ enables effective latent reasoning without any direct supervision of the reasoning process, significantly improving performance when integrated with different LLM-based recommendation methods. Our codes are available at https://anonymous.4open.science/r/R3-A278/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.09403v3">Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 Accepted by ACL 2025 Main Conference
    </div>
    <details class="paper-abstract">
      The rapid advancement of scientific progress requires innovative tools that can accelerate knowledge discovery. Although recent AI methods, particularly large language models (LLMs), have shown promise in tasks such as hypothesis generation and experimental design, they fall short of replicating the collaborative nature of real-world scientific practices, where diverse experts work together in teams to tackle complex problems. To address the limitations, we propose an LLM-based multi-agent system, i.e., Virtual Scientists (VirSci), designed to mimic the teamwork inherent in scientific research. VirSci organizes a team of agents to collaboratively generate, evaluate, and refine research ideas. Through comprehensive experiments, we demonstrate that this multi-agent approach outperforms the state-of-the-art method in producing novel scientific ideas. We further investigate the collaboration mechanisms that contribute to its tendency to produce ideas with higher novelty, offering valuable insights to guide future research and illuminating pathways toward building a robust system for autonomous scientific discovery. The code is available at https://github.com/open-sciencelab/Virtual-Scientists.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19075v1">Universal Reasoner: A Single, Composable Plug-and-Play Reasoner for Frozen LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 22 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable general capabilities, but enhancing skills such as reasoning often demands substantial computational resources and may compromise their generalization. While Parameter-Efficient Fine-Tuning (PEFT) methods offer a more resource-conscious alternative, they typically requires retraining for each LLM backbone due to architectural dependencies. To address these challenges, here we propose Universal Reasoner (UniR) - a single, lightweight, composable, and plug-and-play reasoning module that can be used with any frozen LLM to endow it with specialized reasoning capabilities. Specifically, UniR decomposes the reward into a standalone reasoning module that is trained independently using predefined rewards, effectively translating trajectory-level signals into token-level guidance. Once trained, UniR can be combined with any frozen LLM at inference time by simply adding its output logits to those of the LLM backbone. This additive structure naturally enables modular composition: multiple UniR modules trained for different tasks can be jointly applied by summing their logits, enabling complex reasoning via composition. Experimental results on mathematical reasoning and machine translation tasks show that UniR significantly outperforms \add{existing baseline fine-tuning methods using the Llama3.2 model}. Furthermore, UniR demonstrates strong weak-to-strong generalization: reasoning modules trained on smaller models effectively guide much larger LLMs. This makes UniR a cost-efficient, adaptable, and robust solution for enhancing reasoning in LLMs without compromising their core capabilities. Code is open-sourced at https://github.com/hangeol/UniR
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.13311v3">Training Turn-by-Turn Verifiers for Dialogue Tutoring Agents: The Curious Case of LLMs as Your Coding Tutors</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 Accepted to Findings of ACL 2025
    </div>
    <details class="paper-abstract">
      Intelligent tutoring agents powered by large language models (LLMs) have been increasingly explored to deliver personalized knowledge in areas such as language learning and science education. However, their capabilities in guiding users to solve complex real-world tasks remain underexplored. To address this limitation, in this work, we focus on coding tutoring, a challenging problem that requires tutors to proactively guide students towards completing predefined coding tasks. We propose a novel agent workflow, Trace-and-Verify (TRAVER), which combines knowledge tracing to estimate a student's knowledge state and turn-by-turn verification to ensure effective guidance toward task completion. We introduce DICT, an automatic evaluation protocol that assesses tutor agents using controlled student simulation and code generation tests. Extensive experiments reveal the challenges of coding tutoring and demonstrate that TRAVER achieves a significantly higher success rate. Although we use code tutoring as an example in this paper, our approach can be extended beyond coding, providing valuable insights into advancing tutoring agents for human task learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19056v1">An Embarrassingly Simple Defense Against LLM Abliteration Attacks</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 preprint
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are typically aligned to comply with safety guidelines by refusing harmful instructions. A recent attack, termed abliteration, isolates and suppresses the single latent direction most responsible for refusal behavior, enabling the model to generate unethical content. We propose a defense that modifies how models generate refusals. We construct an extended-refusal dataset that contains harmful prompts with a full response that justifies the reason for refusal. We then fine-tune Llama-2-7B-Chat and Qwen2.5-Instruct (1.5B and 3B parameters) on our extended-refusal dataset, and evaluate the resulting systems on a set of harmful prompts. In our experiments, extended-refusal models maintain high refusal rates, dropping at most by 10%, whereas baseline models' refusal rates drop by 70-80% after abliteration. A broad evaluation of safety and utility shows that extended-refusal fine-tuning neutralizes the abliteration attack while preserving general performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06704v2">PII-Scope: A Comprehensive Study on Training Data PII Extraction Attacks in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 Additional results with Pythia6.9B; Additional results with Phone number PII;
    </div>
    <details class="paper-abstract">
      In this work, we introduce PII-Scope, a comprehensive benchmark designed to evaluate state-of-the-art methodologies for PII extraction attacks targeting LLMs across diverse threat settings. Our study provides a deeper understanding of these attacks by uncovering several hyperparameters (e.g., demonstration selection) crucial to their effectiveness. Building on this understanding, we extend our study to more realistic attack scenarios, exploring PII attacks that employ advanced adversarial strategies, including repeated and diverse querying, and leveraging iterative learning for continual PII extraction. Through extensive experimentation, our results reveal a notable underestimation of PII leakage in existing single-query attacks. In fact, we show that with sophisticated adversarial capabilities and a limited query budget, PII extraction rates can increase by up to fivefold when targeting the pretrained model. Moreover, we evaluate PII leakage on finetuned models, showing that they are more vulnerable to leakage than pretrained models. Overall, our work establishes a rigorous empirical benchmark for PII extraction attacks in realistic threat scenarios and provides a strong foundation for developing effective mitigation strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19030v1">RECAST: Strengthening LLMs' Complex Instruction Following with Constraint-Verifiable Data</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly expected to tackle complex tasks, driven by their expanding applications and users' growing proficiency in crafting sophisticated prompts. However, as the number of explicitly stated requirements increases (particularly more than 10 constraints), LLMs often struggle to accurately follow such complex instructions. To address this challenge, we propose RECAST, a novel framework for synthesizing datasets where each example incorporates far more constraints than those in existing benchmarks. These constraints are extracted from real-world prompt-response pairs to ensure practical relevance. RECAST enables automatic verification of constraint satisfaction via rule-based validators for quantitative constraints and LLM-based validators for qualitative ones. Using this framework, we construct RECAST-30K, a large-scale, high-quality dataset comprising 30k instances spanning 15 constraint types. Experimental results demonstrate that models fine-tuned on RECAST-30K show substantial improvements in following complex instructions. Moreover, the verifiability provided by RECAST enables the design of reward functions for reinforcement learning, which further boosts model performance on complex and challenging tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.06431v3">Functional-level Uncertainty Quantification for Calibrated Fine-tuning on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      Accurate uncertainty quantification in large language models (LLMs) is essential for providing credible confidence estimates over their outputs. However, fine-tuned LLMs often exhibit overconfidence in uncertain predictions, which stems from their limited ability to generalize with sparse data. Existing parameter efficient fine-tuning (PEFT) uncertainty quantification methods for LLMs focus on post fine-tuning stage, and thus fail to address the core issue: limited specialization of PEFT adapters to accurately capture task-specific input-output relationships. To address these limitations, we propose Functional-Level Uncertainty Quantification for Calibrated Fine-Tuning (UQ4CT), which captures and calibrates uncertainty over the space of functions that map input prompts to outputs. We implement UQ4CT during the fine-tuning stage via a mixture-of-experts framework that hierarchically decomposes the functional space. Empirically, UQ4CT achieves over $25\%$ reduction in Expected Calibration Error (ECE) while preserving high accuracy across five benchmarks. Even under distribution shift, UQ4CT maintains superior ECE performance with high accuracy, showcasing improved generalizability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19003v1">Aligning LLM with human travel choices: a persona-based embedding learning approach</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 32 pages, 8 figures
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) presents new opportunities for travel demand modeling. However, behavioral misalignment between LLMs and humans presents obstacles for the usage of LLMs, and existing alignment methods are frequently inefficient or impractical given the constraints of typical travel demand data. This paper introduces a novel framework for aligning LLMs with human travel choice behavior, tailored to the current travel demand data sources. Our framework uses a persona inference and loading process to condition LLMs with suitable prompts to enhance alignment. The inference step establishes a set of base personas from empirical data, and a learned persona loading function driven by behavioral embeddings guides the loading process. We validate our framework on the Swissmetro mode choice dataset, and the results show that our proposed approach significantly outperformed baseline choice models and LLM-based simulation models in predicting both aggregate mode choice shares and individual choice outcomes. Furthermore, we showcase that our framework can generate insights on population behavior through interpretable parameters. Overall, our research offers a more adaptable, interpretable, and resource-efficient pathway to robust LLM-based travel behavior simulation, paving the way to integrate LLMs into travel demand modeling practice in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19000v1">VerIPO: Cultivating Long Reasoning in Video-LLMs via Verifier-Gudied Iterative Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 19 pages, 9 figures, Project Link: https://github.com/HITsz-TMG/VerIPO
    </div>
    <details class="paper-abstract">
      Applying Reinforcement Learning (RL) to Video Large Language Models (Video-LLMs) shows significant promise for complex video reasoning. However, popular Reinforcement Fine-Tuning (RFT) methods, such as outcome-based Group Relative Policy Optimization (GRPO), are limited by data preparation bottlenecks (e.g., noise or high cost) and exhibit unstable improvements in the quality of long chain-of-thoughts (CoTs) and downstream performance.To address these limitations, we propose VerIPO, a Verifier-guided Iterative Policy Optimization method designed to gradually improve video LLMs' capacity for generating deep, long-term reasoning chains. The core component is Rollout-Aware Verifier, positioned between the GRPO and Direct Preference Optimization (DPO) training phases to form the GRPO-Verifier-DPO training loop. This verifier leverages small LLMs as a judge to assess the reasoning logic of rollouts, enabling the construction of high-quality contrastive data, including reflective and contextually consistent CoTs. These curated preference samples drive the efficient DPO stage (7x faster than GRPO), leading to marked improvements in reasoning chain quality, especially in terms of length and contextual consistency. This training loop benefits from GRPO's expansive search and DPO's targeted optimization. Experimental results demonstrate: 1) Significantly faster and more effective optimization compared to standard GRPO variants, yielding superior performance; 2) Our trained models exceed the direct inference of large-scale instruction-tuned Video-LLMs, producing long and contextually consistent CoTs on diverse video reasoning tasks; and 3) Our model with one iteration outperforms powerful LMMs (e.g., Kimi-VL) and long reasoning models (e.g., Video-R1), highlighting its effectiveness and stability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.10700v2">LLMs know their vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 ACL 2025 main conference. Code is available at https://github.com/AI45Lab/ActorAttack
    </div>
    <details class="paper-abstract">
      Safety concerns in large language models (LLMs) have gained significant attention due to their exposure to potentially harmful data during pre-training. In this paper, we identify a new safety vulnerability in LLMs: their susceptibility to \textit{natural distribution shifts} between attack prompts and original toxic prompts, where seemingly benign prompts, semantically related to harmful content, can bypass safety mechanisms. To explore this issue, we introduce a novel attack method, \textit{ActorBreaker}, which identifies actors related to toxic prompts within pre-training distribution to craft multi-turn prompts that gradually lead LLMs to reveal unsafe content. ActorBreaker is grounded in Latour's actor-network theory, encompassing both human and non-human actors to capture a broader range of vulnerabilities. Our experimental results demonstrate that ActorBreaker outperforms existing attack methods in terms of diversity, effectiveness, and efficiency across aligned LLMs. To address this vulnerability, we propose expanding safety training to cover a broader semantic space of toxic content. We thus construct a multi-turn safety dataset using ActorBreaker. Fine-tuning models on our dataset shows significant improvements in robustness, though with some trade-offs in utility. Code is available at https://github.com/AI45Lab/ActorAttack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16638v4">LLMScan: Causal Scan for LLM Misbehavior Detection</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      Despite the success of Large Language Models (LLMs) across various fields, their potential to generate untruthful, biased and harmful responses poses significant risks, particularly in critical applications. This highlights the urgent need for systematic methods to detect and prevent such misbehavior. While existing approaches target specific issues such as harmful responses, this work introduces LLMScan, an innovative LLM monitoring technique based on causality analysis, offering a comprehensive solution. LLMScan systematically monitors the inner workings of an LLM through the lens of causal inference, operating on the premise that the LLM's `brain' behaves differently when misbehaving. By analyzing the causal contributions of the LLM's input tokens and transformer layers, LLMScan effectively detects misbehavior. Extensive experiments across various tasks and models reveal clear distinctions in the causal distributions between normal behavior and misbehavior, enabling the development of accurate, lightweight detectors for a variety of misbehavior detection tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19976v2">ScaleBiO: Scalable Bilevel Optimization for LLM Data Reweighting</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      Bilevel optimization has shown its utility across various machine learning settings, yet most algorithms in practice require second-order information, making it challenging to scale them up. Only recently, a paradigm of first-order algorithms has emerged in the theoretical literature, capable of effectively addressing bilevel optimization problems. Nevertheless, the practical efficiency of this paradigm remains unverified, particularly in the context of large language models (LLMs). This paper introduces the first scalable instantiation of this paradigm called ScaleBiO, focusing on bilevel optimization for large-scale LLM data reweighting. By combining with a recently proposed memory-efficient training technique called LISA, our novel algorithm allows the paradigm to scale to $\sim$30B-sized LLMs on $8\times$H100 GPUs, marking the first successful application of bilevel optimization under practical scenarios for large-sized LLMs. Empirically, extensive experiments on data reweighting verify the effectiveness of ScaleBiO for different-scaled models, including Llama-3-8B, Gemma-2-9B, Qwen-2-7B, and Qwen-2.5-32B, where bilevel optimization succeeds in instruction-following and math reasoning tasks, outperforming several popular baselines, including uniform sampling, influence-aware data filtering, and reference-model-based sampling methods. Theoretically, ScaleBiO ensures the optimality of the learned data weights, along with a convergence guarantee matching the conventional first-order bilevel optimization paradigm on smooth and strongly convex objectives.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18970v1">Learning to Explain: Prototype-Based Surrogate Models for LLM Classification</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive performance on natural language tasks, but their decision-making processes remain largely opaque. Existing explanation methods either suffer from limited faithfulness to the model's reasoning or produce explanations that humans find difficult to understand. To address these challenges, we propose \textbf{ProtoSurE}, a novel prototype-based surrogate framework that provides faithful and human-understandable explanations for LLMs. ProtoSurE trains an interpretable-by-design surrogate model that aligns with the target LLM while utilizing sentence-level prototypes as human-understandable concepts. Extensive experiments show that ProtoSurE consistently outperforms SOTA explanation methods across diverse LLMs and datasets. Importantly, ProtoSurE demonstrates strong data efficiency, requiring relatively few training examples to achieve good performance, making it practical for real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.11827v2">Not All Thoughts are Generated Equal: Efficient LLM Reasoning via Multi-Turn Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Compressing long chain-of-thought (CoT) from large language models (LLMs) is an emerging strategy to improve the reasoning efficiency of LLMs. Despite its promising benefits, existing studies equally compress all thoughts within a long CoT, hindering more concise and effective reasoning. To this end, we first investigate the importance of different thoughts by examining their effectiveness and efficiency in contributing to reasoning through automatic long CoT chunking and Monte Carlo rollouts. Building upon the insights, we propose a theoretically bounded metric to jointly measure the effectiveness and efficiency of different thoughts. We then propose Long$\otimes$Short, an efficient reasoning framework that enables two LLMs to collaboratively solve the problem: a long-thought LLM for more effectively generating important thoughts, while a short-thought LLM for efficiently generating remaining thoughts. Specifically, we begin by synthesizing a small amount of cold-start data to fine-tune LLMs for long-thought and short-thought reasoning styles, respectively. Furthermore, we propose a synergizing-oriented multi-turn reinforcement learning, focusing on the model self-evolution and collaboration between long-thought and short-thought LLMs. Experimental results show that our method enables Qwen2.5-7B and Llama3.1-8B to achieve comparable performance compared to DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B, while reducing token length by over 80% across the MATH500, AIME24/25, AMC23, and GPQA Diamond benchmarks. Our data and code are available at https://github.com/usail-hkust/LongShort.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05185v2">PentestAgent: Incorporating LLM Agents to Automated Penetration Testing</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 14 pages, 13 figures, AsiaCCS 2025
    </div>
    <details class="paper-abstract">
      Penetration testing is a critical technique for identifying security vulnerabilities, traditionally performed manually by skilled security specialists. This complex process involves gathering information about the target system, identifying entry points, exploiting the system, and reporting findings. Despite its effectiveness, manual penetration testing is time-consuming and expensive, often requiring significant expertise and resources that many organizations cannot afford. While automated penetration testing methods have been proposed, they often fall short in real-world applications due to limitations in flexibility, adaptability, and implementation. Recent advancements in large language models (LLMs) offer new opportunities for enhancing penetration testing through increased intelligence and automation. However, current LLM-based approaches still face significant challenges, including limited penetration testing knowledge and a lack of comprehensive automation capabilities. To address these gaps, we propose PentestAgent, a novel LLM-based automated penetration testing framework that leverages the power of LLMs and various LLM-based techniques like Retrieval Augmented Generation (RAG) to enhance penetration testing knowledge and automate various tasks. Our framework leverages multi-agent collaboration to automate intelligence gathering, vulnerability analysis, and exploitation stages, reducing manual intervention. We evaluate PentestAgent using a comprehensive benchmark, demonstrating superior performance in task completion and overall efficiency. This work significantly advances the practical applicability of automated penetration testing systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18961v1">Weaver: Interweaving SQL and LLM for Table Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      Querying tables with unstructured data is challenging due to the presence of text (or image), either embedded in the table or in external paragraphs, which traditional SQL struggles to process, especially for tasks requiring semantic reasoning. While Large Language Models (LLMs) excel at understanding context, they face limitations with long input sequences. Existing approaches that combine SQL and LLMs typically rely on rigid, predefined work-flows, limiting their adaptability to complex queries. To address these issues, we introduce Weaver , a modular pipeline that dynamically integrates SQL and LLMs for table-based question answering (TableQA). Weaver generates a flexible, step-by-step plan that combines SQL for structured data retrieval with LLMs for semantic processing. By decomposing complex queries into manageable subtasks, Weaver improves accuracy and generalization. Our experiments show that Weaver consistently outperforms state-of-the-art methods across four TableQA datasets, reducing both API calls and error rates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18949v1">The Price of Format: Diversity Collapse in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 14 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Instruction-tuned large language models (LLMs) employ structured templates, such as role markers and special tokens, to enforce format consistency during inference. However, we identify a critical limitation of such formatting: it induces a phenomenon we term diversity collapse, where the model generates semantically similar outputs for open-ended inputs, undermining creativity and variability. We systematically evaluate this effect across tasks like story completion and free-form generation, finding that (1) diversity collapse persists even under high-temperature sampling, and (2) structural tokens in templates significantly constrain the model's output space. To contextualize these findings, we fine-tune the same model using a range of structured prompts and then evaluate them across three axes: downstream task performance, alignment behavior, and output diversity. Our analysis shows that format consistency between fine-tuning and inference is crucial for structure-sensitive tasks (e.g., GSM8K, IFEval), but has marginal influence on knowledge-heavy tasks (e.g., MMLU, WebQuestions). In contrast, output diversity is primarily governed by the presence or absence of structural tokens, with minimal formatting yielding the most diverse outputs. These findings reveal that current prompting conventions, while beneficial for alignment, may inadvertently suppress output diversity, underscoring the need for diversity-aware prompt design and instruction tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16128v2">Veracity Bias and Beyond: Uncovering LLMs' Hidden Beliefs in Problem-Solving Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 Accepted to ACL 2025 (Main)
    </div>
    <details class="paper-abstract">
      Despite LLMs' explicit alignment against demographic stereotypes, they have been shown to exhibit biases under various social contexts. In this work, we find that LLMs exhibit concerning biases in how they associate solution veracity with demographics. Through experiments across five human value-aligned LLMs on mathematics, coding, commonsense, and writing problems, we reveal two forms of such veracity biases: Attribution Bias, where models disproportionately attribute correct solutions to certain demographic groups, and Evaluation Bias, where models' assessment of identical solutions varies based on perceived demographic authorship. Our results show pervasive biases: LLMs consistently attribute fewer correct solutions and more incorrect ones to African-American groups in math and coding, while Asian authorships are least preferred in writing evaluation. In additional studies, we show LLMs automatically assign racially stereotypical colors to demographic groups in visualization code, suggesting these biases are deeply embedded in models' reasoning processes. Our findings indicate that demographic bias extends beyond surface-level stereotypes and social context provocations, raising concerns about LLMs' deployment in educational and evaluation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01657v2">Improving Rule-based Reasoning in LLMs via Neurosymbolic Representations</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) continue to face challenges in reliably solving reasoning tasks, particularly those that require precise rule following, as often found in mathematical reasoning. This paper introduces a novel neurosymbolic method that improves LLM reasoning by encoding hidden states into neurosymbolic vectors, enabling problem-solving within a neurosymbolic vector space. The results are decoded and merged with the original hidden state, significantly boosting the model's performance on numerical reasoning tasks. By offloading computation through neurosymbolic representations, this method enhances efficiency, reliability, and interpretability. Experimental results demonstrate an average of 88.6% lower cross-entropy loss and 15.4 times more problems correctly solved on a suite of mathematical reasoning tasks compared to chain-of-thought prompting and supervised fine-tuning (LoRA), without degrading performance on other tasks. We make our code available at: https://github.com/vdhanraj/Neurosymbolic-LLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.14662v3">iAgent: LLM Agent as a Shield between User and Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-25
      | 💬 Findings of ACL 2025 and WWW2025@HCRS
    </div>
    <details class="paper-abstract">
      Traditional recommender systems usually take the user-platform paradigm, where users are directly exposed under the control of the platform's recommendation algorithms. However, the defect of recommendation algorithms may put users in very vulnerable positions under this paradigm. First, many sophisticated models are often designed with commercial objectives in mind, focusing on the platform's benefits, which may hinder their ability to protect and capture users' true interests. Second, these models are typically optimized using data from all users, which may overlook individual user's preferences. Due to these shortcomings, users may experience several disadvantages under the traditional user-platform direct exposure paradigm, such as lack of control over the recommender system, potential manipulation by the platform, echo chamber effects, or lack of personalization for less active users due to the dominance of active users during collaborative learning. Therefore, there is an urgent need to develop a new paradigm to protect user interests and alleviate these issues. Recently, some researchers have introduced LLM agents to simulate user behaviors, these approaches primarily aim to optimize platform-side performance, leaving core issues in recommender systems unresolved. To address these limitations, we propose a new user-agent-platform paradigm, where agent serves as the protective shield between user and recommender system that enables indirect exposure.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18924v1">LLM-Guided Taxonomy and Hierarchical Uncertainty for 3D Point CLoud Active Learning</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      We present a novel active learning framework for 3D point cloud semantic segmentation that, for the first time, integrates large language models (LLMs) to construct hierarchical label structures and guide uncertainty-based sample selection. Unlike prior methods that treat labels as flat and independent, our approach leverages LLM prompting to automatically generate multi-level semantic taxonomies and introduces a recursive uncertainty projection mechanism that propagates uncertainty across hierarchy levels. This enables spatially diverse, label-aware point selection that respects the inherent semantic structure of 3D scenes. Experiments on S3DIS and ScanNet v2 show that our method achieves up to 4% mIoU improvement under extremely low annotation budgets (e.g., 0.02%), substantially outperforming existing baselines. Our results highlight the untapped potential of LLMs as knowledge priors in 3D vision and establish hierarchical uncertainty modeling as a powerful paradigm for efficient point cloud annotation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19349v1">DECA: A Near-Core LLM Decompression Accelerator Supporting Out-of-Order Invocation</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      To alleviate the memory bandwidth bottleneck in Large Language Model (LLM) inference workloads, weight matrices are stored in memory in quantized and sparsified formats. Hence, before tiles of these matrices can be processed by in-core generalized matrix multiplication (GeMM) hardware engines, they need to be dequantized and de-sparsified. This is currently performed in software with vector operations. Unfortunately, this approach delivers only modest performance. Moreover, it is hard to understand how to improve the system, as the overall GeMM performance depends on the interaction between memory resources, vector units, and hardware matrix engines. To improve the performance of LLM inference in advanced platforms equipped with in-core GeMM engines and HBM, this paper makes three main contributions. First, it develops an analytical performance model with a 3D visual representation that provides insights into how memory resources, vector units, and hardware matrix engines interact to deliver compressed GeMM performance. Second, it proposes DECA, a new near-core ML-model decompression accelerator. DECA offloads tile de-sparsification and dequantization from the CPU, producing ready-to-use tiles for in-core GeMM engines. Third, it introduces a new ISA extension that enables out-of-order invocation of the near-core accelerator. With this extension, accelerator and core computations can interleave and overlap with high-performance. Our evaluation shows that, in a simulated 56-core Xeon 4 server with HBM, DECA accelerates the execution of compressed GeMMs by up to 4x over the use of optimized Intel software kernels. Further, DECA reduces the next-token generation time of Llama2-70B and OPT-66B by 1.6x-2.6x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19345v1">PatentScore: Multi-dimensional Evaluation of LLM-Generated Patent Claims</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      Natural language generation (NLG) metrics play a central role in evaluating generated texts, but are not well suited for the structural and legal characteristics of patent documents. Large language models (LLMs) offer strong potential in automating patent generation, yet research on evaluating LLM-generated patents remains limited, especially in evaluating the generation quality of patent claims, which are central to defining the scope of protection. Effective claim evaluation requires addressing legal validity, technical accuracy, and structural compliance. To address this gap, we introduce PatentScore, a multi-dimensional evaluation framework for assessing LLM-generated patent claims. PatentScore incorporates: (1) hierarchical decomposition for claim analysis; (2) domain-specific validation patterns based on legal and technical standards; and (3) scoring across structural, semantic, and legal dimensions. Unlike general-purpose NLG metrics, PatentScore reflects patent-specific constraints and document structures, enabling evaluation beyond surface similarity. We evaluate 400 GPT-4o-mini generated Claim 1s and report a Pearson correlation of $r = 0.819$ with expert annotations, outperforming existing NLG metrics. Furthermore, we conduct additional evaluations using open models such as Claude-3.5-Haiku and Gemini-1.5-flash, all of which show strong correlations with expert judgments, confirming the robustness and generalizability of our framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19334v1">Likert or Not: LLM Absolute Relevance Judgments on Fine-Grained Ordinal Scales</a></div>
    <div class="paper-meta">
      📅 2025-05-25
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) obtain state of the art zero shot relevance ranking performance on a variety of information retrieval tasks. The two most common prompts to elicit LLM relevance judgments are pointwise scoring (a.k.a. relevance generation), where the LLM sees a single query-document pair and outputs a single relevance score, and listwise ranking (a.k.a. permutation generation), where the LLM sees a query and a list of documents and outputs a permutation, sorting the documents in decreasing order of relevance. The current research community consensus is that listwise ranking yields superior performance, and significant research effort has been devoted to crafting LLM listwise ranking algorithms. The underlying hypothesis is that LLMs are better at making relative relevance judgments than absolute ones. In tension with this hypothesis, we find that the gap between pointwise scoring and listwise ranking shrinks when pointwise scoring is implemented using a sufficiently large ordinal relevance label space, becoming statistically insignificant for many LLM-benchmark dataset combinations (where ``significant'' means ``95\% confidence that listwise ranking improves NDCG@10''). Our evaluations span four LLMs, eight benchmark datasets from the BEIR and TREC-DL suites, and two proprietary datasets with relevance labels collected after the training cut-off of all LLMs evaluated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12067v2">TokenSkip: Controllable Chain-of-Thought Compression in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-24
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) has been proven effective in enhancing the reasoning capabilities of large language models (LLMs). Recent advancements, such as OpenAI's o1 and DeepSeek-R1, suggest that scaling up the length of CoT sequences during inference could further boost LLM reasoning performance. However, due to the autoregressive nature of LLM decoding, longer CoT outputs lead to a linear increase in inference latency, adversely affecting user experience, particularly when the CoT exceeds 10,000 tokens. To address this limitation, we analyze the semantic importance of tokens within CoT outputs and reveal that their contributions to reasoning vary. Building on this insight, we propose TokenSkip, a simple yet effective approach that enables LLMs to selectively skip less important tokens, allowing for controllable CoT compression. Extensive experiments across various models and tasks demonstrate the effectiveness of TokenSkip in reducing CoT token usage while preserving strong reasoning performance. Notably, when applied to Qwen2.5-14B-Instruct, TokenSkip reduces reasoning tokens by 40% (from 313 to 181) on GSM8K, with less than a 0.4% performance drop.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10688v2">CULEMO: Cultural Lenses on Emotion -- Benchmarking LLMs for Cross-Cultural Emotion Understanding</a></div>
    <div class="paper-meta">
      📅 2025-05-24
      | 💬 ACL-main 2025
    </div>
    <details class="paper-abstract">
      NLP research has increasingly focused on subjective tasks such as emotion analysis. However, existing emotion benchmarks suffer from two major shortcomings: (1) they largely rely on keyword-based emotion recognition, overlooking crucial cultural dimensions required for deeper emotion understanding, and (2) many are created by translating English-annotated data into other languages, leading to potentially unreliable evaluation. To address these issues, we introduce Cultural Lenses on Emotion (CuLEmo), the first benchmark designed to evaluate culture-aware emotion prediction across six languages: Amharic, Arabic, English, German, Hindi, and Spanish. CuLEmo comprises 400 crafted questions per language, each requiring nuanced cultural reasoning and understanding. We use this benchmark to evaluate several state-of-the-art LLMs on culture-aware emotion prediction and sentiment analysis tasks. Our findings reveal that (1) emotion conceptualizations vary significantly across languages and cultures, (2) LLMs performance likewise varies by language and cultural context, and (3) prompting in English with explicit country context often outperforms in-language prompts for culture-aware emotion and sentiment understanding. The dataset and evaluation code are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18886v1">KerZOO: Kernel Function Informed Zeroth-Order Optimization for Accurate and Accelerated LLM Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-05-24
      | 💬 9 pages, 6 figures, Neurips
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated impressive capabilities across numerous NLP tasks. Nevertheless, conventional first-order fine-tuning techniques impose heavy memory demands, creating practical obstacles to real-world applications. Zeroth-order (ZO) optimization has recently emerged as a promising memory-efficient alternative, as it circumvents the need for backpropagation by estimating gradients solely through forward passes--making it particularly suitable for resource-limited environments. Despite its efficiency, ZO optimization suffers from gradient estimation bias, which significantly hinders convergence speed. To address this, we analytically identify and characterize the lower-order bias introduced during ZO-based gradient estimation in LLM fine-tuning. Motivated by tools in mathematical physics, we introduce a kernel-function-based ZO framework aimed at mitigating this bias and improving optimization stability. KerZOO achieves comparable or superior performance to existing ZO baselines in both full-parameter and parameter-efficient fine-tuning settings of LLMs, while significantly reducing the number of iterations required to reach convergence. For example, KerZOO reduces total GPU training hours by as much as 74% and 44% on WSC and MultiRC datasets in fine-tuning OPT-2.7B model and can exceed the MeZO baseline by 2.9% and 2.6% in accuracy. We show that the kernel function is an effective avenue for reducing estimation bias in ZO methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11242v3">LLMs and Childhood Safety: Identifying Risks and Proposing a Protection Framework for Safe Child-LLM Interaction</a></div>
    <div class="paper-meta">
      📅 2025-05-24
    </div>
    <details class="paper-abstract">
      This study examines the growing use of Large Language Models (LLMs) in child-centered applications, highlighting safety and ethical concerns such as bias, harmful content, and cultural insensitivity. Despite their potential to enhance learning, there is a lack of standardized frameworks to mitigate these risks. Through a systematic literature review, we identify key parental and empirical concerns, including toxicity and ethical breaches in AI outputs. Moreover, to address these issues, this paper proposes a protection framework for safe Child-LLM interaction, incorporating metrics for content safety, behavioral ethics, and cultural sensitivity. The framework provides practical tools for evaluating LLM safety, offering guidance for developers, policymakers, and educators to ensure responsible AI deployment for children.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18882v1">Personalized Safety in LLMs: A Benchmark and A Planning-Based Agent Approach</a></div>
    <div class="paper-meta">
      📅 2025-05-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) typically generate identical or similar responses for all users given the same prompt, posing serious safety risks in high-stakes applications where user vulnerabilities differ widely. Existing safety evaluations primarily rely on context-independent metrics - such as factuality, bias, or toxicity - overlooking the fact that the same response may carry divergent risks depending on the user's background or condition. We introduce personalized safety to fill this gap and present PENGUIN - a benchmark comprising 14,000 scenarios across seven sensitive domains with both context-rich and context-free variants. Evaluating six leading LLMs, we demonstrate that personalized user information significantly improves safety scores by 43.2%, confirming the effectiveness of personalization in safety alignment. However, not all context attributes contribute equally to safety enhancement. To address this, we develop RAISE - a training-free, two-stage agent framework that strategically acquires user-specific background. RAISE improves safety scores by up to 31.6% over six vanilla LLMs, while maintaining a low interaction cost of just 2.7 user queries on average. Our findings highlight the importance of selective information gathering in safety-critical domains and offer a practical solution for personalizing LLM responses without model retraining. This work establishes a foundation for safety research that adapts to individual user contexts rather than assuming a universal harm standard.
    </details>
</div>
