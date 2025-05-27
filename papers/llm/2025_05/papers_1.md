# llm - 2025_05

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20155v1">Pangu Light: Weight Re-Initialization for Pruning and Accelerating LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) deliver state-of-the-art capabilities across numerous tasks, but their immense size and inference costs pose significant computational challenges for practical deployment. While structured pruning offers a promising avenue for model compression, existing methods often struggle with the detrimental effects of aggressive, simultaneous width and depth reductions, leading to substantial performance degradation. This paper argues that a critical, often overlooked, aspect in making such aggressive joint pruning viable is the strategic re-initialization and adjustment of remaining weights to improve the model post-pruning training accuracies. We introduce Pangu Light, a framework for LLM acceleration centered around structured pruning coupled with novel weight re-initialization techniques designed to address this ``missing piece''. Our framework systematically targets multiple axes, including model width, depth, attention heads, and RMSNorm, with its effectiveness rooted in novel re-initialization methods like Cross-Layer Attention Pruning (CLAP) and Stabilized LayerNorm Pruning (SLNP) that mitigate performance drops by providing the network a better training starting point. Further enhancing efficiency, Pangu Light incorporates specialized optimizations such as absorbing Post-RMSNorm computations and tailors its strategies to Ascend NPU characteristics. The Pangu Light models consistently exhibit a superior accuracy-efficiency trade-off, outperforming prominent baseline pruning methods like Nemotron and established LLMs like Qwen3 series. For instance, on Ascend NPUs, Pangu Light-32B's 81.6 average score and 2585 tokens/s throughput exceed Qwen3-32B's 80.9 average score and 2225 tokens/s.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15873v2">Abstractions-of-Thought: Intermediate Representations for LLM Reasoning in Hardware Design</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved impressive proficiency on logic and programming tasks, often rivaling expert-level performance. However, generating functionally correct hardware description language (HDL) code from natural language specifications remains challenging, primarily in data-scarce domains. Therefore, we present Abstractions-of-Thought (AoT) - a training-free, inference-only prompting framework to mitigate misinterpretations and reasoning pitfalls of LLMs through a series of task-based abstractions within the prompting procedure, assisting in the transition from high-level to low-level representations of hardware. Furthermore, AoT consists of the following stages: (1) an LLM-based classification of hardware design patterns, (2) a structured intermediate representation (IR) to separate functional decomposition from code syntax, and (3) a line-by-line pseudocode solution enabling a more direct mapping to the final Verilog implementation. Experimental results on the VerilogEval benchmark depict that AoT demonstrates improvements in functionality when applied to large non-reasoning models (such as GPT-4o, outperforming all baseline techniques (including 1-shot, Chain-of-Thought, and Tree-of-Thought) while significantly reducing the generated tokens by 1.8-5.2x compared to popular Tree-of-Thought prompting.
    </details>
</div>
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
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17726v2">Slot-MLLM: Object-Centric Visual Tokenization for Multimodal LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Recently, multimodal large language models (MLLMs) have emerged as a key approach in achieving artificial general intelligence. In particular, vision-language MLLMs have been developed to generate not only text but also visual outputs from multimodal inputs. This advancement requires efficient image tokens that LLMs can process effectively both in input and output. However, existing image tokenization methods for MLLMs typically capture only global abstract concepts or uniformly segmented image patches, restricting MLLMs' capability to effectively understand or generate detailed visual content, particularly at the object level. To address this limitation, we propose an object-centric visual tokenizer based on Slot Attention specifically for MLLMs. In particular, based on the Q-Former encoder, diffusion decoder, and residual vector quantization, our proposed discretized slot tokens can encode local visual details while maintaining high-level semantics, and also align with textual data to be integrated seamlessly within a unified next-token prediction framework of LLMs. The resulting Slot-MLLM demonstrates significant performance improvements over baselines with previous visual tokenizers across various vision-language tasks that entail local detailed comprehension and generation. Notably, this work is the first demonstration of the feasibility of object-centric slot attention performed with MLLMs and in-the-wild natural images.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19510v1">LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs? A Benchmark and Empirical Study</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      The remarkable reasoning and generalization capabilities of Large Language Models (LLMs) have paved the way for their expanding applications in embodied AI, robotics, and other real-world tasks. To effectively support these applications, grounding in spatial and temporal understanding in multimodal environments is essential. To this end, recent works have leveraged scene graphs, a structured representation that encodes entities, attributes, and their relationships in a scene. However, a comprehensive evaluation of LLMs' ability to utilize scene graphs remains limited. In this work, we introduce Text-Scene Graph (TSG) Bench, a benchmark designed to systematically assess LLMs' ability to (1) understand scene graphs and (2) generate them from textual narratives. With TSG Bench we evaluate 11 LLMs and reveal that, while models perform well on scene graph understanding, they struggle with scene graph generation, particularly for complex narratives. Our analysis indicates that these models fail to effectively decompose discrete scenes from a complex narrative, leading to a bottleneck when generating scene graphs. These findings underscore the need for improved methodologies in scene graph generation and provide valuable insights for future research. The demonstration of our benchmark is available at https://tsg-bench.netlify.app. Additionally, our code and evaluation data are publicly available at https://anonymous.4open.science/r/TSG-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.15107v2">StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 20 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Efficient multi-hop reasoning requires Large Language Models (LLMs) based agents to acquire high-value external knowledge iteratively. Previous work has explored reinforcement learning (RL) to train LLMs to perform search-based document retrieval, achieving notable improvements in QA performance, but underperform on complex, multi-hop QA resulting from the sparse rewards from global signal only. To address this gap in existing research, we introduce StepSearch, a framework for search LLMs that trained with step-wise proximal policy optimization method. It consists of richer and more detailed intermediate search rewards and token-level process supervision based on information gain and redundancy penalties to better guide each search step. We constructed a fine-grained question-answering dataset containing sub-question-level search trajectories based on open source datasets through a set of data pipeline method. On standard multi-hop QA benchmarks, it significantly outperforms global-reward baselines, achieving 11.2% and 4.2% absolute improvements for 3B and 7B models over various search with RL baselines using only 19k training data, demonstrating the effectiveness of fine-grained, stepwise supervision in optimizing deep search LLMs. Our code will be released on https://github.com/Zillwang/StepSearch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19504v1">DOGe: Defensive Output Generation for LLM Protection Against Knowledge Distillation</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Code is available at https://github.com/UNITES-Lab/DOGe
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) represent substantial intellectual and economic investments, yet their effectiveness can inadvertently facilitate model imitation via knowledge distillation (KD).In practical scenarios, competitors can distill proprietary LLM capabilities by simply observing publicly accessible outputs, akin to reverse-engineering a complex performance by observation alone. Existing protective methods like watermarking only identify imitation post-hoc, while other defenses assume the student model mimics the teacher's internal logits, rendering them ineffective against distillation purely from observed output text. This paper confronts the challenge of actively protecting LLMs within the realistic constraints of API-based access. We introduce an effective and efficient Defensive Output Generation (DOGe) strategy that subtly modifies the output behavior of an LLM. Its outputs remain accurate and useful for legitimate users, yet are designed to be misleading for distillation, significantly undermining imitation attempts. We achieve this by fine-tuning only the final linear layer of the teacher LLM with an adversarial loss. This targeted training approach anticipates and disrupts distillation attempts during inference time. Our experiments show that, while preserving or even improving the original performance of the teacher model, student models distilled from the defensively generated teacher outputs demonstrate catastrophically reduced performance, demonstrating our method's effectiveness as a practical safeguard against KD-based model imitation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17070v2">Robo-Troj: Attacking LLM-based Task Planners</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Robots need task planning methods to achieve goals that require more than individual actions. Recently, large language models (LLMs) have demonstrated impressive performance in task planning. LLMs can generate a step-by-step solution using a description of actions and the goal. Despite the successes in LLM-based task planning, there is limited research studying the security aspects of those systems. In this paper, we develop Robo-Troj, the first multi-trigger backdoor attack for LLM-based task planners, which is the main contribution of this work. As a multi-trigger attack, Robo-Troj is trained to accommodate the diversity of robot application domains. For instance, one can use unique trigger words, e.g., "herical", to activate a specific malicious behavior, e.g., cutting hand on a kitchen robot. In addition, we develop an optimization method for selecting the trigger words that are most effective. Through demonstrating the vulnerability of LLM-based planners, we aim to promote the development of secured robot systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19489v1">Benchmarking and Enhancing LLM Agents in Localizing Linux Kernel Bugs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      The Linux kernel is a critical system, serving as the foundation for numerous systems. Bugs in the Linux kernel can cause serious consequences, affecting billions of users. Fault localization (FL), which aims at identifying the buggy code elements in software, plays an essential role in software quality assurance. While recent LLM agents have achieved promising accuracy in FL on recent benchmarks like SWE-bench, it remains unclear how well these methods perform in the Linux kernel, where FL is much more challenging due to the large-scale code base, limited observability, and diverse impact factors. In this paper, we introduce LinuxFLBench, a FL benchmark constructed from real-world Linux kernel bugs. We conduct an empirical study to assess the performance of state-of-the-art LLM agents on the Linux kernel. Our initial results reveal that existing agents struggle with this task, achieving a best top-1 accuracy of only 41.6% at file level. To address this challenge, we propose LinuxFL$^+$, an enhancement framework designed to improve FL effectiveness of LLM agents for the Linux kernel. LinuxFL$^+$ substantially improves the FL accuracy of all studied agents (e.g., 7.2% - 11.2% accuracy increase) with minimal costs. Data and code are available at https://github.com/FudanSELab/LinuxFLBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19484v1">CulFiT: A Fine-grained Cultural-aware LLM Training Paradigm via Multilingual Critique Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they often exhibit a specific cultural biases, neglecting the values and linguistic diversity of low-resource regions. This cultural bias not only undermines universal equality, but also risks reinforcing stereotypes and perpetuating discrimination. To address this, we propose CulFiT, a novel culturally-aware training paradigm that leverages multilingual data and fine-grained reward modeling to enhance cultural sensitivity and inclusivity. Our approach synthesizes diverse cultural-related questions, constructs critique data in culturally relevant languages, and employs fine-grained rewards to decompose cultural texts into verifiable knowledge units for interpretable evaluation. We also introduce GlobalCultureQA, a multilingual open-ended question-answering dataset designed to evaluate culturally-aware responses in a global context. Extensive experiments on three existing benchmarks and our GlobalCultureQA demonstrate that CulFiT achieves state-of-the-art open-source model performance in cultural alignment and general reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19481v1">Win Fast or Lose Slow: Balancing Speed and Accuracy in Latency-Sensitive Decisions of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable performance across diverse reasoning and generation tasks, and are increasingly deployed as agents in dynamic environments such as code generation and recommendation systems. However, many real-world applications, such as high-frequency trading and real-time competitive gaming, require decisions under strict latency constraints, where faster responses directly translate into higher rewards. Despite the importance of this latency quality trade off, it remains underexplored in the context of LLM based agents. In this work, we present the first systematic study of this trade off in real time decision making tasks. To support our investigation, we introduce two new benchmarks: HFTBench, a high frequency trading simulation, and StreetFighter, a competitive gaming platform. Our analysis reveals that optimal latency quality balance varies by task, and that sacrificing quality for lower latency can significantly enhance downstream performance. To address this, we propose FPX, an adaptive framework that dynamically selects model size and quantization level based on real time demands. Our method achieves the best performance on both benchmarks, improving win rate by up to 80% in Street Fighter and boosting daily yield by up to 26.52% in trading, underscoring the need for latency aware evaluation and deployment strategies for LLM based agents. These results demonstrate the critical importance of latency aware evaluation and deployment strategies for real world LLM based agents. Our benchmarks are available at Latency Sensitive Benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17135v2">When can isotropy help adapt LLMs' next word prediction to numerical domains?</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Recent studies have shown that vector representations of contextual embeddings learned by pre-trained large language models (LLMs) are effective in various downstream tasks in numerical domains. Despite their significant benefits, the tendency of LLMs to hallucinate in such domains can have severe consequences in applications such as energy, nature, finance, healthcare, retail and transportation, among others. To guarantee prediction reliability and accuracy in numerical domains, it is necessary to open the black-box and provide performance guarantees through explanation. However, there is little theoretical understanding of when pre-trained language models help solve numeric downstream tasks. This paper seeks to bridge this gap by understanding when the next-word prediction capability of LLMs can be adapted to numerical domains through a novel analysis based on the concept of isotropy in the contextual embedding space. Specifically, we consider a log-linear model for LLMs in which numeric data can be predicted from its context through a network with softmax in the output layer of LLMs (i.e., language model head in self-attention). We demonstrate that, in order to achieve state-of-the-art performance in numerical domains, the hidden representations of the LLM embeddings must possess a structure that accounts for the shift-invariance of the softmax function. By formulating a gradient structure of self-attention in pre-trained models, we show how the isotropic property of LLM embeddings in contextual embedding space preserves the underlying structure of representations, thereby resolving the shift-invariance problem and providing a performance guarantee. Experiments show that different characteristics of numeric data and model architecture could have different impacts on isotropy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19473v1">Improving Recommendation Fairness without Sensitive Attributes Using Multi-Persona LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 18 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Despite the success of recommender systems in alleviating information overload, fairness issues have raised concerns in recent years, potentially leading to unequal treatment for certain user groups. While efforts have been made to improve recommendation fairness, they often assume that users' sensitive attributes are available during model training. However, collecting sensitive information can be difficult, especially on platforms that involve no personal information disclosure. Therefore, we aim to improve recommendation fairness without any access to sensitive attributes. However, this is a non-trivial task because uncovering latent sensitive patterns from complicated user behaviors without explicit sensitive attributes can be difficult. Consequently, suboptimal estimates of sensitive distributions can hinder the fairness training process. To address these challenges, leveraging the remarkable reasoning abilities of Large Language Models (LLMs), we propose a novel LLM-enhanced framework for Fair recommendation withOut Sensitive Attributes (LLMFOSA). A Multi-Persona Sensitive Information Inference module employs LLMs with distinct personas that mimic diverse human perceptions to infer and distill sensitive information. Furthermore, a Confusion-Aware Sensitive Representation Learning module incorporates inference results and rationales to develop robust sensitive representations, considering the mislabeling confusion and collective consensus among agents. The model is then optimized by a formulated mutual information objective. Extensive experiments on two public datasets validate the effectiveness of LLMFOSA in improving fairness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11393v2">HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 ACL 2025 Findings
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capabilities in commonsense reasoning; however, some variations in questions can trigger incorrect responses. Do these models truly understand commonsense knowledge, or just memorize expression patterns? To investigate this question, we present the first extensive robustness evaluation of LLMs in commonsense reasoning. We introduce HellaSwag-Pro, a large-scale bilingual benchmark consisting of 11,200 cases, by designing and compiling seven types of question variants. To construct this benchmark, we propose a two-stage method to develop Chinese HellaSwag, a finely annotated dataset comprising 12,000 instances across 56 categories. We conduct extensive experiments on 41 representative LLMs, revealing that these LLMs are far from robust in commonsense reasoning. Furthermore, this robustness varies depending on the language in which the LLM is tested. This work establishes a high-quality evaluation benchmark, with extensive experiments offering valuable insights to the community in commonsense reasoning for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19466v1">Origin Tracer: A Method for Detecting LoRA Fine-Tuning Origins in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to advance, their deployment often involves fine-tuning to enhance performance on specific downstream tasks. However, this customization is sometimes accompanied by misleading claims about the origins, raising significant concerns about transparency and trust within the open-source community. Existing model verification techniques typically assess functional, representational, and weight similarities. However, these approaches often struggle against obfuscation techniques, such as permutations and scaling transformations. To address this limitation, we propose a novel detection method Origin-Tracer that rigorously determines whether a model has been fine-tuned from a specified base model. This method includes the ability to extract the LoRA rank utilized during the fine-tuning process, providing a more robust verification framework. This framework is the first to provide a formalized approach specifically aimed at pinpointing the sources of model fine-tuning. We empirically validated our method on thirty-one diverse open-source models under conditions that simulate real-world obfuscation scenarios. We empirically analyze the effectiveness of our framework and finally, discuss its limitations. The results demonstrate the effectiveness of our approach and indicate its potential to establish new benchmarks for model verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19464v1">LLMs as Better Recommenders with Natural Language Collaborative Signals: A Self-Assessing Retrieval Approach</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 13 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Incorporating collaborative information (CI) effectively is crucial for leveraging LLMs in recommendation tasks. Existing approaches often encode CI using soft tokens or abstract identifiers, which introduces a semantic misalignment with the LLM's natural language pretraining and hampers knowledge integration. To address this, we propose expressing CI directly in natural language to better align with LLMs' semantic space. We achieve this by retrieving a curated set of the most relevant user behaviors in natural language form. However, identifying informative CI is challenging due to the complexity of similarity and utility assessment. To tackle this, we introduce a Self-assessing COllaborative REtrieval framework (SCORE) following the retrieve-rerank paradigm. First, a Collaborative Retriever (CAR) is developed to consider both collaborative patterns and semantic similarity. Then, a Self-assessing Reranker (SARE) leverages LLMs' own reasoning to assess and prioritize retrieved behaviors. Finally, the selected behaviors are prepended to the LLM prompt as natural-language CI to guide recommendation. Extensive experiments on two public datasets validate the effectiveness of SCORE in improving LLM-based recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19457v1">BizFinBench: A Business-Driven Real-World Financial Benchmark for Evaluating LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Project Page: https://hithink-research.github.io/BizFinBench/
    </div>
    <details class="paper-abstract">
      Large language models excel in general tasks, yet assessing their reliability in logic-heavy, precision-critical domains like finance, law, and healthcare remains challenging. To address this, we introduce BizFinBench, the first benchmark specifically designed to evaluate LLMs in real-world financial applications. BizFinBench consists of 6,781 well-annotated queries in Chinese, spanning five dimensions: numerical calculation, reasoning, information extraction, prediction recognition, and knowledge-based question answering, grouped into nine fine-grained categories. The benchmark includes both objective and subjective metrics. We also introduce IteraJudge, a novel LLM evaluation method that reduces bias when LLMs serve as evaluators in objective metrics. We benchmark 25 models, including both proprietary and open-source systems. Extensive experiments show that no model dominates across all tasks. Our evaluation reveals distinct capability patterns: (1) In Numerical Calculation, Claude-3.5-Sonnet (63.18) and DeepSeek-R1 (64.04) lead, while smaller models like Qwen2.5-VL-3B (15.92) lag significantly; (2) In Reasoning, proprietary models dominate (ChatGPT-o3: 83.58, Gemini-2.0-Flash: 81.15), with open-source models trailing by up to 19.49 points; (3) In Information Extraction, the performance spread is the largest, with DeepSeek-R1 scoring 71.46, while Qwen3-1.7B scores 11.23; (4) In Prediction Recognition, performance variance is minimal, with top models scoring between 39.16 and 50.00. We find that while current LLMs handle routine finance queries competently, they struggle with complex scenarios requiring cross-concept reasoning. BizFinBench offers a rigorous, business-aligned benchmark for future research. The code and dataset are available at https://github.com/HiThink-Research/BizFinBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19436v1">Task Memory Engine: Spatial Memory for Robust Multi-Step LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Under review. 9 pages main content, 15 pages appendix, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) falter in multi-step interactions -- often hallucinating, repeating actions, or misinterpreting user corrections -- due to reliance on linear, unstructured context. This fragility stems from the lack of persistent memory to track evolving goals and task dependencies, undermining trust in autonomous agents. We introduce the Task Memory Engine (TME), a modular memory controller that transforms existing LLMs into robust, revision-aware agents without fine-tuning. TME implements a spatial memory framework that replaces flat context with graph-based structures to support consistent, multi-turn reasoning. Departing from linear concatenation and ReAct-style prompting, TME builds a dynamic task graph -- either a tree or directed acyclic graph (DAG) -- to map user inputs to subtasks, align them with prior context, and enable dependency-tracked revisions. Its Task Representation and Intent Management (TRIM) component models task semantics and user intent to ensure accurate interpretation. Across four multi-turn scenarios-trip planning, cooking, meeting scheduling, and shopping cart editing -- TME eliminates 100% of hallucinations and misinterpretations in three tasks, and reduces hallucinations by 66.7% and misinterpretations by 83.3% across 27 user turns, outperforming ReAct. TME's modular design supports plug-and-play deployment and domain-specific customization, adaptable to both personal assistants and enterprise automation. We release TME's codebase, benchmarks, and components as open-source resources, enabling researchers to develop reliable LLM agents. TME's scalable architecture addresses a critical gap in agent performance across complex, interactive settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04856v2">One-Shot is Enough: Consolidating Multi-Turn Attacks into Efficient Single-Turn Prompts for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      We introduce a novel framework for consolidating multi-turn adversarial ``jailbreak'' prompts into single-turn queries, significantly reducing the manual overhead required for adversarial testing of large language models (LLMs). While multi-turn human jailbreaks have been shown to yield high attack success rates, they demand considerable human effort and time. Our multi-turn-to-single-turn (M2S) methods -- Hyphenize, Numberize, and Pythonize -- systematically reformat multi-turn dialogues into structured single-turn prompts. Despite removing iterative back-and-forth interactions, these prompts preserve and often enhance adversarial potency: in extensive evaluations on the Multi-turn Human Jailbreak (MHJ) dataset, M2S methods achieve attack success rates from 70.6 percent to 95.9 percent across several state-of-the-art LLMs. Remarkably, the single-turn prompts outperform the original multi-turn attacks by as much as 17.5 percentage points while cutting token usage by more than half on average. Further analysis shows that embedding malicious requests in enumerated or code-like structures exploits ``contextual blindness'', bypassing both native guardrails and external input-output filters. By converting multi-turn conversations into concise single-turn prompts, the M2S framework provides a scalable tool for large-scale red teaming and reveals critical weaknesses in contemporary LLM defenses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19433v1">Can Compressed LLMs Truly Act? An Empirical Evaluation of Agentic Capabilities in LLM Compression</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Accepted by ICML2025 as Poster
    </div>
    <details class="paper-abstract">
      Post-training compression reduces the computational and memory costs of large language models (LLMs), enabling resource-efficient deployment. However, existing compression benchmarks only focus on language modeling (e.g., perplexity) and natural language understanding tasks (e.g., GLUE accuracy), ignoring the agentic capabilities - workflow, tool use/function call, long-context understanding and real-world application. We introduce the Agent Compression Benchmark (ACBench), the first comprehensive benchmark for evaluating how compression impacts LLMs' agentic abilities. ACBench spans (1) 12 tasks across 4 capabilities (e.g., WorfBench for workflow generation, Needle-in-Haystack for long-context retrieval), (2) quantization (GPTQ, AWQ) and pruning (Wanda, SparseGPT), and (3) 15 models, including small (Gemma-2B), standard (Qwen2.5 7B-32B), and distilled reasoning LLMs (DeepSeek-R1-Distill). Our experiments reveal compression tradeoffs: 4-bit quantization preserves workflow generation and tool use (1%-3% drop) but degrades real-world application accuracy by 10%-15%. We introduce ERank, Top-k Ranking Correlation and Energy to systematize analysis. ACBench provides actionable insights for optimizing LLM compression in agentic scenarios. The code can be found in https://github.com/pprp/ACBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11478v3">Each Graph is a New Language: Graph Learning with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Recent efforts leverage Large Language Models (LLMs) for modeling text-attributed graph structures in node classification tasks. These approaches describe graph structures for LLMs to understand or aggregate LLM-generated textual attribute embeddings through graph structure. However, these approaches face two main limitations in modeling graph structures with LLMs. (i) Graph descriptions become verbose in describing high-order graph structure. (ii) Textual attributes alone do not contain adequate graph structure information. It is challenging to model graph structure concisely and adequately with LLMs. LLMs lack built-in mechanisms to model graph structures directly. They also struggle with complex long-range dependencies between high-order nodes and target nodes. Inspired by the observation that LLMs pre-trained on one language can achieve exceptional performance on another with minimal additional training, we propose \textbf{G}raph-\textbf{D}efined \textbf{L}anguage for \textbf{L}arge \textbf{L}anguage \textbf{M}odel (GDL4LLM). This novel framework enables LLMs to transfer their powerful language understanding capabilities to graph-structured data. GDL4LLM translates graphs into a graph language corpus instead of graph descriptions and pre-trains LLMs on this corpus to adequately understand graph structures. During fine-tuning, this corpus describes the structural information of target nodes concisely with only a few tokens. By treating graphs as a new language, GDL4LLM enables LLMs to model graph structures adequately and concisely for node classification tasks. Extensive experiments on three real-world datasets demonstrate that GDL4LLM outperforms description-based and textual attribute embeddings-based baselines by efficiently modeling different orders of graph structure with LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19419v1">It's Not Just Labeling" -- A Research on LLM Generated Feedback Interpretability and Image Labeling Sketch Features</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      The quality of training data is critical to the performance of machine learning applications in domains like transportation, healthcare, and robotics. Accurate image labeling, however, often relies on time-consuming, expert-driven methods with limited feedback. This research introduces a sketch-based annotation approach supported by large language models (LLMs) to reduce technical barriers and enhance accessibility. Using a synthetic dataset, we examine how sketch recognition features relate to LLM feedback metrics, aiming to improve the reliability and interpretability of LLM-assisted labeling. We also explore how prompting strategies and sketch variations influence feedback quality. Our main contribution is a sketch-based virtual assistant that simplifies annotation for non-experts and advances LLM-driven labeling tools in terms of scalability, accessibility, and explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19410v1">Self-Reflective Planning with Knowledge Graphs: Enhancing LLM Reasoning Reliability for Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      Recently, large language models (LLMs) have demonstrated remarkable capabilities in natural language processing tasks, yet they remain prone to hallucinations when reasoning with insufficient internal knowledge. While integrating LLMs with knowledge graphs (KGs) provides access to structured, verifiable information, existing approaches often generate incomplete or factually inconsistent reasoning paths. To this end, we propose Self-Reflective Planning (SRP), a framework that synergizes LLMs with KGs through iterative, reference-guided reasoning. Specifically, given a question and topic entities, SRP first searches for references to guide planning and reflection. In the planning process, it checks initial relations and generates a reasoning path. After retrieving knowledge from KGs through a reasoning path, it implements iterative reflection by judging the retrieval result and editing the reasoning path until the answer is correctly retrieved. Extensive experiments on three public datasets demonstrate that SRP surpasses various strong baselines and further underscore its reliable reasoning ability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.19405v1">CoTGuard: Using Chain-of-Thought Triggering for Copyright Protection in Multi-Agent LLM Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 18 pages, 1 figure
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) evolve into autonomous agents capable of collaborative reasoning and task execution, multi-agent LLM systems have emerged as a powerful paradigm for solving complex problems. However, these systems pose new challenges for copyright protection, particularly when sensitive or copyrighted content is inadvertently recalled through inter-agent communication and reasoning. Existing protection techniques primarily focus on detecting content in final outputs, overlooking the richer, more revealing reasoning processes within the agents themselves. In this paper, we introduce CoTGuard, a novel framework for copyright protection that leverages trigger-based detection within Chain-of-Thought (CoT) reasoning. Specifically, we can activate specific CoT segments and monitor intermediate reasoning steps for unauthorized content reproduction by embedding specific trigger queries into agent prompts. This approach enables fine-grained, interpretable detection of copyright violations in collaborative agent scenarios. We evaluate CoTGuard on various benchmarks in extensive experiments and show that it effectively uncovers content leakage with minimal interference to task performance. Our findings suggest that reasoning-level monitoring offers a promising direction for safeguarding intellectual property in LLM-based agent systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.18791v3">Can LLMs Help Uncover Insights about LLMs? A Large-Scale, Evolving Literature Analysis of Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      The surge of LLM studies makes synthesizing their findings challenging. Analysis of experimental results from literature can uncover important trends across studies, but the time-consuming nature of manual data extraction limits its use. Our study presents a semi-automated approach for literature analysis that accelerates data extraction using LLMs. It automatically identifies relevant arXiv papers, extracts experimental results and related attributes, and organizes them into a structured dataset, LLMEvalDB. We then conduct an automated literature analysis of frontier LLMs, reducing the effort of paper surveying and data extraction by more than 93% compared to manual approaches. We validate LLMEvalDB by showing that it reproduces key findings from a recent manual analysis of Chain-of-Thought (CoT) reasoning and also uncovers new insights that go beyond it, showing, for example, that in-context examples benefit coding & multimodal tasks but offer limited gains in math reasoning tasks compared to zero-shot CoT. Our automatically updatable dataset enables continuous tracking of target models by extracting evaluation studies as new data becomes available. Through LLMEvalDB and empirical analysis, we provide insights into LLMs while facilitating ongoing literature analyses of their behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12632v3">What External Knowledge is Preferred by LLMs? Characterizing and Exploring Chain of Evidence in Imperfect Context for Multi-Hop QA</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Incorporating external knowledge has emerged as a promising way to mitigate outdated knowledge and hallucinations in LLM. However, external knowledge is often imperfect, encompassing substantial extraneous or even inaccurate content, which interferes with the LLM's utilization of useful knowledge in the context. This paper seeks to characterize the features of preferred external knowledge and perform empirical studies in imperfect contexts. Inspired by the chain of evidence (CoE), we characterize that the knowledge preferred by LLMs should maintain both relevance to the question and mutual support among the textual pieces. Accordingly, we propose a CoE discrimination approach and conduct a comparative analysis between CoE and Non-CoE samples across significance, deceptiveness, and robustness, revealing the LLM's preference for external knowledge that aligns with CoE features. Furthermore, we selected three representative tasks (RAG-based multi-hop QA, external knowledge poisoning and poisoning defense), along with corresponding SOTA or prevalent baselines. By integrating CoE features, the variants achieved significant improvements over the original baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16254v2">Reassessing Collaborative Writing Theories and Frameworks in the Age of LLMs: What Still Applies and What We Must Leave Behind</a></div>
    <div class="paper-meta">
      📅 2025-05-26
    </div>
    <details class="paper-abstract">
      In this paper, we conduct a critical review of existing theories and frameworks on human-human collaborative writing to assess their relevance to the current human-AI paradigm in professional contexts, and draw seven insights along with design implications for human-AI collaborative writing tools. We found that, as LLMs nudge the writing process more towards an empirical "trial and error" process analogous to prototyping, the non-linear cognitive process of writing will stay the same, but more rigor will be required for revision methodologies. This shift would shed further light on the importance of coherence support, but the large language model (LLM)'s unprecedented semantic capabilities can bring novel approaches to this ongoing challenge. We argue that teamwork-related factors such as group awareness, consensus building and authorship - which have been central in human-human collaborative writing studies - should not apply to the human-AI paradigm due to excessive anthropomorphism. With the LLM's text generation capabilities becoming essentially indistinguishable from human-written ones, we are entering an era where, for the first time in the history of computing, we are engaging in collaborative writing with AI at workplaces on a daily basis. We aim to bring theoretical grounding and practical design guidance to the interaction designs of human-AI collaborative writing, with the goal of enhancing future human-AI writing software.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.18152v2">Fann or Flop: A Multigenre, Multiera Benchmark for Arabic Poetry Understanding in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-26
      | 💬 Github:https://github.com/mbzuai-oryx/FannOrFlop, Dataset:https://huggingface.co/datasets/omkarthawakar/FannOrFlop
    </div>
    <details class="paper-abstract">
      Arabic poetry is one of the richest and most culturally rooted forms of expression in the Arabic language, known for its layered meanings, stylistic diversity, and deep historical continuity. Although large language models (LLMs) have demonstrated strong performance across languages and tasks, their ability to understand Arabic poetry remains largely unexplored. In this work, we introduce \emph{Fann or Flop}, the first benchmark designed to assess the comprehension of Arabic poetry by LLMs in 12 historical eras, covering 14 core poetic genres and a variety of metrical forms, from classical structures to contemporary free verse. The benchmark comprises a curated corpus of poems with explanations that assess semantic understanding, metaphor interpretation, prosodic awareness, and cultural context. We argue that poetic comprehension offers a strong indicator for testing how good the LLM understands classical Arabic through Arabic poetry. Unlike surface-level tasks, this domain demands deeper interpretive reasoning and cultural sensitivity. Our evaluation of state-of-the-art LLMs shows that most models struggle with poetic understanding despite strong results on standard Arabic benchmarks. We release "Fann or Flop" along with the evaluation suite as an open-source resource to enable rigorous evaluation and advancement for Arabic language models. Code is available at: https://github.com/mbzuai-oryx/FannOrFlop.
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
