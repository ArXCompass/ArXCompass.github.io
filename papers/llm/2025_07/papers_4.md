# llm - 2025_07

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- Part 4
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.15170v2">From LLMs to MLLMs to Agents: A Survey of Emerging Paradigms in Jailbreak Attacks and Defenses within LLM Ecosystem</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are rapidly evolving from single-modal systems to multimodal LLMs and intelligent agents, significantly expanding their capabilities while introducing increasingly severe security risks. This paper presents a systematic survey of the growing complexity of jailbreak attacks and corresponding defense mechanisms within the expanding LLM ecosystem. We first trace the developmental trajectory from LLMs to MLLMs and Agents, highlighting the core security challenges emerging at each stage. Next, we categorize mainstream jailbreak techniques from both the attack impact and visibility perspectives, and provide a comprehensive analysis of representative attack methods, related datasets, and evaluation metrics. On the defense side, we organize existing strategies based on response timing and technical approach, offering a structured understanding of their applicability and implementation. Furthermore, we identify key limitations in existing surveys, such as insufficient attention to agent-specific security issues, the absence of a clear taxonomy for hybrid jailbreak methods, a lack of detailed analysis of experimental setups, and outdated coverage of recent advancements. To address these limitations, we provide an updated synthesis of recent work and outline future research directions in areas such as dataset construction, evaluation framework optimization, and strategy generalization. Our study seeks to enhance the understanding of jailbreak mechanisms and facilitate the advancement of more resilient and adaptive defense strategies in the context of ever more capable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14928v1">Byzantine-Robust Decentralized Coordination of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Collaboration among multiple large language model (LLM) agents is a promising approach to overcome inherent limitations of single-agent systems, such as hallucinations and single points of failure. As LLM agents are increasingly deployed on open blockchain platforms, multi-agent systems capable of tolerating malicious (Byzantine) agents have become essential. Recent Byzantine-robust multi-agent systems typically rely on leader-driven coordination, which suffers from two major drawbacks. First, they are inherently vulnerable to targeted attacks against the leader. If consecutive leaders behave maliciously, the system repeatedly fails to achieve consensus, forcing new consensus rounds, which is particularly costly given the high latency of LLM invocations. Second, an underperforming proposal from the leader can be accepted as the final answer even when higher-quality alternatives are available, as existing methods finalize the leader's proposal once it receives a quorum of votes. To address these issues, we propose DecentLLMs, a novel decentralized consensus approach for multi-agent LLM systems, where worker agents generate answers concurrently and evaluator agents independently score and rank these answers to select the best available one. This decentralized architecture enables faster consensus despite the presence of Byzantine agents and consistently selects higher-quality answers through Byzantine-robust aggregation techniques. Experimental results demonstrate that DecentLLMs effectively tolerates Byzantine agents and significantly improves the quality of selected answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14906v1">Feedback-Induced Performance Decline in LLM-Based Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      The ability of Large Language Models (LLMs) to extract context from natural language problem descriptions naturally raises questions about their suitability in autonomous decision-making settings. This paper studies the behaviour of these models within a Markov Decision Process (MDPs). While traditional reinforcement learning (RL) strategies commonly employed in this setting rely on iterative exploration, LLMs, pre-trained on diverse datasets, offer the capability to leverage prior knowledge for faster adaptation. We investigate online structured prompting strategies in sequential decision making tasks, comparing the zero-shot performance of LLM-based approaches to that of classical RL methods. Our findings reveal that although LLMs demonstrate improved initial performance in simpler environments, they struggle with planning and reasoning in complex scenarios without fine-tuning or additional guidance. Our results show that feedback mechanisms, intended to improve decision-making, often introduce confusion, leading to diminished performance in intricate environments. These insights underscore the need for further exploration into hybrid strategies, fine-tuning, and advanced memory integration to enhance LLM-based decision-making capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14894v1">Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-20
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have impressive multilingual capabilities, but they suffer from unexpected code-switching, also known as language mixing, which involves switching to unexpected languages in the model response. This problem leads to poor readability and degrades the usability of model responses. However, existing work on this issue lacks a mechanistic analysis and shows limited effectiveness. In this paper, we first provide an in-depth analysis of unexpected code-switching using sparse autoencoders and find that when LLMs switch to a language, the features of that language exhibit excessive pre-activation values. Based on our findings, we propose $\textbf{S}$parse $\textbf{A}$utoencoder-guided $\textbf{S}$upervised $\textbf{F}$ine$\textbf{t}$uning (SASFT), which teaches LLMs to maintain appropriate pre-activation values of specific language features during training. Experiments on five models across three languages demonstrate that SASFT consistently reduces unexpected code-switching by more than 50\% compared to standard supervised fine-tuning, with complete elimination in four cases. Moreover, SASFT maintains or even improves the models' performance on six multilingual benchmarks, showing its effectiveness in addressing code-switching while preserving multilingual capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.10467v4">Specification and Evaluation of Multi-Agent LLM Systems -- Prototype and Cybersecurity Applications</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 This work has been submitted for publication. Copyright may be transferred. In this case, this version will be updated with a notice, according to the publisher's guidelines
    </div>
    <details class="paper-abstract">
      Recent advancements in LLMs indicate potential for novel applications, as evidenced by the reasoning capabilities in the latest OpenAI and DeepSeek models. To apply these models to domain-specific applications beyond text generation, LLM-based multi-agent systems can be utilized to solve complex tasks, particularly by combining reasoning techniques, code generation, and software execution across multiple, potentially specialized LLMs. However, while many evaluations are performed on LLMs, reasoning techniques, and applications individually, their joint specification and combined application are not well understood. Defined specifications for multi-agent LLM systems are required to explore their potential and suitability for specific applications, allowing for systematic evaluations of LLMs, reasoning techniques, and related aspects. This paper reports the results of exploratory research on (1.) multi-agent specification by introducing an agent schema language and (2.) the execution and evaluation of the specifications through a multi-agent system architecture and prototype. The specification language, system architecture, and prototype are first presented in this work, building on an LLM system from prior research. Test cases involving cybersecurity tasks indicate the feasibility of the architecture and evaluation approach. As a result, evaluations could be demonstrated for question answering, server security, and network security tasks completed correctly by agents with LLMs from OpenAI and DeepSeek.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14735v1">Investigating the Role of LLMs Hyperparameter Tuning and Prompt Engineering to Support Domain Modeling</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 Accepted at 51st Euromicro Conference Series on Software Engineering and Advanced Applications (SEAA)
    </div>
    <details class="paper-abstract">
      The introduction of large language models (LLMs) has enhanced automation in software engineering tasks, including in Model Driven Engineering (MDE). However, using general-purpose LLMs for domain modeling has its limitations. One approach is to adopt fine-tuned models, but this requires significant computational resources and can lead to issues like catastrophic forgetting. This paper explores how hyperparameter tuning and prompt engineering can improve the accuracy of the Llama 3.1 model for generating domain models from textual descriptions. We use search-based methods to tune hyperparameters for a specific medical data model, resulting in a notable quality improvement over the baseline LLM. We then test the optimized hyperparameters across ten diverse application domains. While the solutions were not universally applicable, we demonstrate that combining hyperparameter tuning with prompt engineering can enhance results across nearly all examined domain models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14719v1">Automated Safety Evaluations Across 20 Large Language Models: The Aymara LLM Risk and Responsibility Matrix</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly integrated into real-world applications, scalable and rigorous safety evaluation is essential. This paper introduces Aymara AI, a programmatic platform for generating and administering customized, policy-grounded safety evaluations. Aymara AI transforms natural-language safety policies into adversarial prompts and scores model responses using an AI-based rater validated against human judgments. We demonstrate its capabilities through the Aymara LLM Risk and Responsibility Matrix, which evaluates 20 commercially available LLMs across 10 real-world safety domains. Results reveal wide performance disparities, with mean safety scores ranging from 86.2% to 52.4%. While models performed well in well-established safety domains such as Misinformation (mean = 95.7%), they consistently failed in more complex or underspecified domains, notably Privacy & Impersonation (mean = 24.3%). Analyses of Variance confirmed that safety scores differed significantly across both models and domains (p < .05). These findings underscore the inconsistent and context-dependent nature of LLM safety and highlight the need for scalable, customizable tools like Aymara AI to support responsible AI development and oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14705v1">Configurable multi-agent framework for scalable and realistic testing of llm-based agents</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      Large-language-model (LLM) agents exhibit complex, context-sensitive behaviour that quickly renders static benchmarks and ad-hoc manual testing obsolete. We present Neo, a configurable, multi-agent framework that automates realistic, multi-turn evaluation of LLM-based systems. Neo couples a Question Generation Agent and an Evaluation Agent through a shared context-hub, allowing domain prompts, scenario controls and dynamic feedback to be composed modularly. Test inputs are sampled from a probabilistic state model spanning dialogue flow, user intent and emotional tone, enabling diverse, human-like conversations that adapt after every turn. Applied to a production-grade Seller Financial Assistant chatbot, Neo (i) uncovered edge-case failures across five attack categories with a 3.3% break rate close to the 5.8% achieved by expert human red-teamers, and (ii) delivered 10-12X higher throughput, generating 180 coherent test questions in around 45 mins versus 16h of human effort. Beyond security probing, Neo's stochastic policies balanced topic coverage and conversational depth, yielding broader behavioural exploration than manually crafted scripts. Neo therefore lays a foundation for scalable, self-evolving LLM QA: its agent interfaces, state controller and feedback loops are model-agnostic and extensible to richer factual-grounding and policy-compliance checks. We release the framework to facilitate reproducible, high-fidelity testing of emerging agentic systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08263v2">LLM-Based Detection of Tangled Code Changes for Higher-Quality Method-Level Bug Datasets</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      Tangled code changes, commits that conflate unrelated modifications such as bug fixes, refactorings, and enhancements, introduce significant noise into bug datasets and adversely affect the performance of bug prediction models. Addressing this issue at a fine-grained, method-level granularity remains underexplored. This is critical to address, as recent bug prediction models, driven by practitioner demand, are increasingly focusing on finer granularity rather than traditional class- or file-level predictions. This study investigates the utility of Large Language Models (LLMs) for detecting tangled code changes by leveraging both commit messages and method-level code diffs. We formulate the problem as a binary classification task and evaluate multiple prompting strategies, including zero-shot, few-shot, and chain-of-thought prompting, using state-of-the-art proprietary LLMs such as GPT-4o and Gemini-2.0-Flash. Our results demonstrate that combining commit messages with code diffs significantly enhances model performance, with the combined few-shot and chain-of-thought prompting achieving an F1-score of 0.88. Additionally, we explore machine learning models trained on LLM-generated embeddings, where a multi-layer perceptron classifier achieves superior performance (F1-score: 0.906, MCC: 0.807). Applying our approach to 49 open-source projects improves the distributional separability of code metrics between buggy and non-buggy methods, demonstrating the promise of LLMs for method-level commit untangling and potentially contributing to improving the accuracy of future bug prediction models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18765v3">A Vision for Auto Research with LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      This paper introduces Agent-Based Auto Research, a structured multi-agent framework designed to automate, coordinate, and optimize the full lifecycle of scientific research. Leveraging the capabilities of large language models (LLMs) and modular agent collaboration, the system spans all major research phases, including literature review, ideation, methodology planning, experimentation, paper writing, peer review response, and dissemination. By addressing issues such as fragmented workflows, uneven methodological expertise, and cognitive overload, the framework offers a systematic and scalable approach to scientific inquiry. Preliminary explorations demonstrate the feasibility and potential of Auto Research as a promising paradigm for self-improving, AI-driven research processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.09116v3">Mixture of LoRA Experts with Multi-Modal and Multi-Granularity LLM Generative Error Correction for Accented Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 IEEE Transactions on Audio, Speech and Language Processing
    </div>
    <details class="paper-abstract">
      Despite improvements in automatic speech recognition, performance drops with accented speech. Generative error correction (GER) leverages the linguistic knowledge of large language models (LLMs), outperforming typical language model methods. However, it lacks specificity in accented speech scenarios. Accents represent deviations from standard pronunciation, making multi-granularity pronunciation and semantic information essential for accented speech recognition. Moreover, accents exhibit considerable diversity, with each accent possessing distinct characteristics. In this study, we leverage GER to improve transcription accuracy by addressing the two primary features. We propose the multi-modal GER, which integrates pronunciation information from the speech modality, and the multi-granularity GER, which incorporates fine-grained phoneme-level pronunciation information. These methods enable the LLM to utilize the pronunciation information of accented speech and the semantic information from word-level hypotheses for accurate transcription predictions through low-rank adaptation (LoRA) fine-tuning. We employ a three-stage strategy to train separate multi-modal GER models for each accent to obtain mono-accent LoRA experts. By adopting our proposed HDMoLE method, which incorporates hierarchical routing and dynamic thresholds within the mixture of LoRA experts, we effectively merge mono-accent LoRA experts within a single multi-modal GER to overcome accent diversity challenges. Furthermore, multi-granularity GER leverages N-best word-level and phoneme-level hypotheses from the HDMoLE model to predict final transcriptions. Experiments on a multi-accent English dataset show that our methods reduce word error rate by 67.35% compared to the baseline vanilla Whisper-large-v3 model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14649v1">Cleanse: Uncertainty Estimation Approach Using Clustering-based Semantic Consistency in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      Despite the outstanding performance of large language models (LLMs) across various NLP tasks, hallucinations in LLMs--where LLMs generate inaccurate responses--remains as a critical problem as it can be directly connected to a crisis of building safe and reliable LLMs. Uncertainty estimation is primarily used to measure hallucination levels in LLM responses so that correct and incorrect answers can be distinguished clearly. This study proposes an effective uncertainty estimation approach, \textbf{Cl}ust\textbf{e}ring-based sem\textbf{an}tic con\textbf{s}ist\textbf{e}ncy (\textbf{Cleanse}). Cleanse quantifies the uncertainty with the proportion of the intra-cluster consistency in the total consistency between LLM hidden embeddings which contain adequate semantic information of generations, by employing clustering. The effectiveness of Cleanse for detecting hallucination is validated using four off-the-shelf models, LLaMA-7B, LLaMA-13B, LLaMA2-7B and Mistral-7B and two question-answering benchmarks, SQuAD and CoQA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03108v3">OMNISEC: LLM-Driven Provenance-based Intrusion Detection via Retrieval-Augmented Behavior Prompting</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      Recently, Provenance-based Intrusion Detection Systems (PIDSes) have been widely used for endpoint threat analysis. These studies can be broadly categorized into rule-based detection systems and learning-based detection systems. Among these, due to the evolution of attack techniques, rules cannot dynamically model all the characteristics of attackers. As a result, such systems often face false negatives. Learning-based detection systems are further divided into supervised learning and anomaly detection. The scarcity of attack samples hinders the usability and effectiveness of supervised learning-based detection systems in practical applications. Anomaly-based detection systems face a massive false positive problem because they cannot distinguish between changes in normal behavior and real attack behavior. The alert results of detection systems are closely related to the manual labor costs of subsequent security analysts. To reduce manual analysis time, we propose OMNISEC, which applies large language models (LLMs) to anomaly-based intrusion detection systems via retrieval-augmented behavior prompting. OMNISEC can identify abnormal nodes and corresponding abnormal events by constructing suspicious nodes and rare paths. By combining two external knowledge bases, OMNISEC uses Retrieval Augmented Generation (RAG) to enable the LLM to determine whether abnormal behavior is a real attack. Finally, OMNISEC can reconstruct the attack graph and restore the complete attack behavior chain of the attacker's intrusion. Experimental results show that OMNISEC outperforms state-of-the-art methods on public benchmark datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14558v1">Harnessing LLMs for Document-Guided Fuzzing of OpenCV Library</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      The combination of computer vision and artificial intelligence is fundamentally transforming a broad spectrum of industries by enabling machines to interpret and act upon visual data with high levels of accuracy. As the biggest and by far the most popular open-source computer vision library, OpenCV library provides an extensive suite of programming functions supporting real-time computer vision. Bugs in the OpenCV library can affect the downstream computer vision applications, and it is critical to ensure the reliability of the OpenCV library. This paper introduces VISTAFUZZ, a novel technique for harnessing large language models (LLMs) for document-guided fuzzing of the OpenCV library. VISTAFUZZ utilizes LLMs to parse API documentation and obtain standardized API information. Based on this standardized information, VISTAFUZZ extracts constraints on individual input parameters and dependencies between these. Using these constraints and dependencies, VISTAFUZZ then generates new input values to systematically test each target API. We evaluate the effectiveness of VISTAFUZZ in testing 330 APIs in the OpenCV library, and the results show that VISTAFUZZ detected 17 new bugs, where 10 bugs have been confirmed, and 5 of these have been fixed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.12891v2">TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 Second version
    </div>
    <details class="paper-abstract">
      Temporal reasoning is pivotal for Large Language Models (LLMs) to comprehend the real world. However, existing works neglect the real-world challenges for temporal reasoning: (1) intensive temporal information, (2) fast-changing event dynamics, and (3) complex temporal dependencies in social interactions. To bridge this gap, we propose a multi-level benchmark TIME, designed for temporal reasoning in real-world scenarios. TIME consists of 38,522 QA pairs, covering 3 levels with 11 fine-grained sub-tasks. This benchmark encompasses 3 sub-datasets reflecting different real-world challenges: TIME-Wiki, TIME-News, and TIME-Dial. We conduct extensive experiments on reasoning models and non-reasoning models. And we conducted an in-depth analysis of temporal reasoning performance across diverse real-world scenarios and tasks, and summarized the impact of test-time scaling on temporal reasoning capabilities. Additionally, we release TIME-Lite, a human-annotated subset to foster future research and standardized evaluation in temporal reasoning. The code is available at https://github.com/sylvain-wei/TIME , and the dataset is available at https://huggingface.co/datasets/SylvainWei/TIME .
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.23644v3">QLPro: Automated Code Vulnerability Discovery via LLM and Static Code Analysis Integration</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 The experimental data in the experimental section needs to be improved, and there are some errors
    </div>
    <details class="paper-abstract">
      We introduce QLPro, a vulnerability detection framework that systematically integrates LLMs and static analysis tools to enable comprehensive vulnerability detection across entire open-source projects.We constructed a new dataset, JavaTest, comprising 10 open-source projects from GitHub with 62 confirmed vulnerabilities. CodeQL, a state-of-the-art static analysis tool, detected only 24 of these vulnerabilities while QLPro detected 41. Furthermore, QLPro discovered 6 previously unknown vulnerabilities, 2 of which have been confirmed as 0-days.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.08373v2">Draft-based Approximate Inference for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 Added discussion and comparison with SpecPrefill
    </div>
    <details class="paper-abstract">
      Optimizing inference for long-context Large Language Models (LLMs) is increasingly important due to the quadratic compute and linear memory complexity of Transformers. Existing approximation methods, such as key-value (KV) cache dropping, sparse attention, and prompt compression, typically rely on rough predictions of token or KV pair importance. We propose a novel framework for approximate LLM inference that leverages small draft models to more accurately predict the importance of tokens and KV pairs. Specifically, we introduce two instantiations of our proposed framework: (i) SpecKV, the first method that leverages a draft output to accurately assess the importance of each KV pair for more effective KV cache dropping, and (ii) SpecPC, which uses the draft model's attention activations to identify and discard unimportant prompt tokens. We motivate our methods with theoretical and empirical analyses, and show a strong correlation between the attention patterns of draft and target models. Extensive experiments on long-context benchmarks show that our methods consistently achieve higher accuracy than existing baselines, while preserving the same improvements in memory usage, latency, and throughput. Our code is available at https://github.com/furiosa-ai/draft-based-approx-llm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08493v3">Intelligent LiDAR Navigation: Leveraging External Information and Semantic Maps with LLM as Copilot</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 Accepted at IROS 2025
    </div>
    <details class="paper-abstract">
      Traditional robot navigation systems primarily utilize occupancy grid maps and laser-based sensing technologies, as demonstrated by the popular move_base package in ROS. Unlike robots, humans navigate not only through spatial awareness and physical distances but also by integrating external information, such as elevator maintenance updates from public notification boards and experiential knowledge, like the need for special access through certain doors. With the development of Large Language Models (LLMs), which possesses text understanding and intelligence close to human performance, there is now an opportunity to infuse robot navigation systems with a level of understanding akin to human cognition. In this study, we propose using osmAG (Area Graph in OpensStreetMap textual format), an innovative semantic topometric hierarchical map representation, to bridge the gap between the capabilities of ROS move_base and the contextual understanding offered by LLMs. Our methodology employs LLMs as an actual copilot in robot navigation, enabling the integration of a broader range of informational inputs while maintaining the robustness of traditional robotic navigation systems. Our code, demo, map, experiment results can be accessed at https://github.com/xiexiexiaoxiexie/Intelligent-LiDAR-Navigation-LLM-as-Copilot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20016v3">Vulnerability of LLMs to Vertically Aligned Text Manipulations</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 Accepted to ACL 2025 (Main)
    </div>
    <details class="paper-abstract">
      Vertical text input is commonly encountered in various real-world applications, such as mathematical computations and word-based Sudoku puzzles. While current large language models (LLMs) have excelled in natural language tasks, they remain vulnerable to variations in text formatting. Recent research demonstrates that modifying input formats, such as vertically aligning words for encoder-based models, can substantially lower accuracy in text classification tasks. While easily understood by humans, these inputs can significantly mislead models, posing a potential risk of bypassing detection in real-world scenarios involving harmful or sensitive information. With the expanding application of LLMs, a crucial question arises: \textit{Do decoder-based LLMs exhibit similar vulnerabilities to vertically formatted text input?} In this paper, we investigate the impact of vertical text input on the performance of various LLMs across multiple text classification datasets and analyze the underlying causes. Our findings are as follows: (i) Vertical text input significantly degrades the accuracy of LLMs in text classification tasks. (ii) \textit{Chain of Thought (CoT)} reasoning does not help LLMs recognize vertical input or mitigate its vulnerability, but \textit{few-shot learning} with careful analysis does. (iii) We explore the underlying cause of the vulnerability by analyzing the inherent issues in tokenization and attention matrices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14430v1">X-Intelligence 3.0: Training and Evaluating Reasoning LLM for Semiconductor Display</a></div>
    <div class="paper-meta">
      📅 2025-07-19
      | 💬 Technical Report
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently achieved significant advances in reasoning and demonstrated their advantages in solving challenging problems. Yet, their effectiveness in the semiconductor display industry remains limited due to a lack of domain-specific training and expertise. To bridge this gap, we present X-Intelligence 3.0, the first high-performance reasoning model specifically developed for the semiconductor display industry. This model is designed to deliver expert-level understanding and reasoning for the industry's complex challenges. Leveraging a carefully curated industry knowledge base, the model undergoes supervised fine-tuning and reinforcement learning to enhance its reasoning and comprehension capabilities. To further accelerate development, we implemented an automated evaluation framework that simulates expert-level assessments. We also integrated a domain-specific retrieval-augmented generation (RAG) mechanism, resulting in notable performance gains on benchmark datasets. Despite its relatively compact size of 32 billion parameters, X-Intelligence 3.0 outperforms SOTA DeepSeek-R1-671B across multiple evaluations. This demonstrates its exceptional efficiency and establishes it as a powerful solution to the longstanding reasoning challenges faced by the semiconductor display industry.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16841v1">AquaChat: An LLM-Guided ROV Framework for Adaptive Inspection of Aquaculture Net Pens</a></div>
    <div class="paper-meta">
      📅 2025-07-19
    </div>
    <details class="paper-abstract">
      Inspection of aquaculture net pens is essential for maintaining the structural integrity, biosecurity, and operational efficiency of fish farming systems. Traditional inspection approaches rely on pre-programmed missions or manual control, offering limited adaptability to dynamic underwater conditions and user-specific demands. In this study, we propose AquaChat, a novel Remotely Operated Vehicle (ROV) framework that integrates Large Language Models (LLMs) for intelligent and adaptive net pen inspection. The system features a multi-layered architecture: (1) a high-level planning layer that interprets natural language user commands using an LLM to generate symbolic task plans; (2) a mid-level task manager that translates plans into ROV control sequences; and (3) a low-level motion control layer that executes navigation and inspection tasks with precision. Real-time feedback and event-triggered replanning enhance robustness in challenging aquaculture environments. The framework is validated through experiments in both simulated and controlled aquatic environments representative of aquaculture net pens. Results demonstrate improved task flexibility, inspection accuracy, and operational efficiency. AquaChat illustrates the potential of integrating language-based AI with marine robotics to enable intelligent, user-interactive inspection systems for sustainable aquaculture operations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15090v2">Analyze the Neurons, not the Embeddings: Understanding When and Where LLM Representations Align with Humans</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Modern large language models (LLMs) achieve impressive performance on some tasks, while exhibiting distinctly non-human-like behaviors on others. This raises the question of how well the LLM's learned representations align with human representations. In this work, we introduce a novel approach to study representation alignment: we adopt a method from research on activation steering to identify neurons responsible for specific concepts (e.g., ''cat'') and then analyze the corresponding activation patterns. We find that LLM representations captured this way closely align with human representations inferred from behavioral data, matching inter-human alignment levels. Our approach significantly outperforms the alignment captured by word embeddings, which have been the focus of prior work on human-LLM alignment. Additionally, our approach enables a more granular view of how LLMs represent concepts -- we show that LLMs organize concepts in a way that mirrors human concept organization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14355v1">Can LLMs Infer Personality from Real World Conversations?</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 21 pages, 12 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) such as OpenAI's GPT-4 and Meta's LLaMA offer a promising approach for scalable personality assessment from open-ended language. However, inferring personality traits remains challenging, and earlier work often relied on synthetic data or social media text lacking psychometric validity. We introduce a real-world benchmark of 555 semi-structured interviews with BFI-10 self-report scores for evaluating LLM-based personality inference. Three state-of-the-art LLMs (GPT-4.1 Mini, Meta-LLaMA, and DeepSeek) were tested using zero-shot prompting for BFI-10 item prediction and both zero-shot and chain-of-thought prompting for Big Five trait inference. All models showed high test-retest reliability, but construct validity was limited: correlations with ground-truth scores were weak (max Pearson's $r = 0.27$), interrater agreement was low (Cohen's $\kappa < 0.10$), and predictions were biased toward moderate or high trait levels. Chain-of-thought prompting and longer input context modestly improved distributional alignment, but not trait-level accuracy. These results underscore limitations in current LLM-based personality inference and highlight the need for evidence-based development for psychological applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14335v1">ProofCompass: Enhancing Specialized Provers with LLM Guidance</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 19 pages, 7 figures. Accepted at the 2nd AI for MATH Workshop at the 42nd International Conference on Machine Learning (ICML 2025)
    </div>
    <details class="paper-abstract">
      Language models have become increasingly powerful tools for formal mathematical reasoning. However, most existing approaches rely exclusively on either large general-purpose models or smaller specialized models, each with distinct limitations, while training specialized large models still requires significant computational resources. This paper introduces ProofCompass, a novel hybrid methodology that achieves remarkable computational efficiency by strategically guiding existing specialized prover methods, such as DeepSeek-Prover-v1.5-RL (DSP-v1.5) with a Large Language Model (LLM) without requiring additional model training. The LLM provides natural language proof strategies and analyzes failed attempts to select intermediate lemmas, enabling effective problem decomposition. On the miniF2F benchmark, ProofCompass demonstrates substantial resource efficiency: it outperforms DSP-v1.5 ($54.9\% \rightarrow 55.3\%$) while using 25x fewer attempts ($3200 \rightarrow 128$). Our synergistic approach paves the way for simultaneously improving computational efficiency and accuracy in formal theorem proving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.03619v2">Blackbox Dataset Inference for LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Today, the training of large language models (LLMs) can involve personally identifiable information and copyrighted material, incurring dataset misuse. To mitigate the problem of dataset misuse, this paper explores \textit{dataset inference}, which aims to detect if a suspect model $\mathcal{M}$ used a victim dataset $\mathcal{D}$ in training. Previous research tackles dataset inference by aggregating results of membership inference attacks (MIAs) -- methods to determine whether individual samples are a part of the training dataset. However, restricted by the low accuracy of MIAs, previous research mandates grey-box access to $\mathcal{M}$ to get intermediate outputs (probabilities, loss, perplexity, etc.) for obtaining satisfactory results. This leads to reduced practicality, as LLMs, especially those deployed for profits, have limited incentives to return the intermediate outputs. In this paper, we propose a new method of dataset inference with only black-box access to the target model (i.e., assuming only the text-based responses of the target model are available). Our method is enabled by two sets of locally built reference models, one set involving $\mathcal{D}$ in training and the other not. By measuring which set of reference model $\mathcal{M}$ is closer to, we determine if $\mathcal{M}$ used $\mathcal{D}$ for training. Evaluations of real-world LLMs in the wild show that our method offers high accuracy in all settings and presents robustness against bypassing attempts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14330v1">Leveraging LLMs for Formal Software Requirements -- Challenges and Prospects</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Submitted to Overlay2025 - 7th International Workshop on Artificial Intelligence and fOrmal VERification, Logic, Automata, and sYnthesis. [under review]
    </div>
    <details class="paper-abstract">
      Software correctness is ensured mathematically through formal verification, which involves the resources of generating formal requirement specifications and having an implementation that must be verified. Tools such as model-checkers and theorem provers ensure software correctness by verifying the implementation against the specification. Formal methods deployment is regularly enforced in the development of safety-critical systems e.g. aerospace, medical devices and autonomous systems. Generating these specifications from informal and ambiguous natural language requirements remains the key challenge. Our project, VERIFAI^{1}, aims to investigate automated and semi-automated approaches to bridge this gap, using techniques from Natural Language Processing (NLP), ontology-based domain modelling, artefact reuse, and large language models (LLMs). This position paper presents a preliminary synthesis of relevant literature to identify recurring challenges and prospective research directions in the generation of verifiable specifications from informal requirements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14307v1">How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit increasingly sophisticated linguistic capabilities, yet the extent to which these behaviors reflect human-like cognition versus advanced pattern recognition remains an open question. In this study, we investigate how LLMs process the temporal meaning of linguistic aspect in narratives that were previously used in human studies. Using an Expert-in-the-Loop probing pipeline, we conduct a series of targeted experiments to assess whether LLMs construct semantic representations and pragmatic inferences in a human-like manner. Our findings show that LLMs over-rely on prototypicality, produce inconsistent aspectual judgments, and struggle with causal reasoning derived from aspect, raising concerns about their ability to fully comprehend narratives. These results suggest that LLMs process aspect fundamentally differently from humans and lack robust narrative understanding. Beyond these empirical findings, we develop a standardized experimental framework for the reliable assessment of LLMs' cognitive and linguistic capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14304v1">Aligning Large Language Models to Low-Resource Languages through LLM-Based Selective Translation: A Systematic Study</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Multilingual large language models (LLMs) often demonstrate a performance gap between English and non-English languages, particularly in low-resource settings. Aligning these models to low-resource languages is essential yet challenging due to limited high-quality data. While English alignment datasets are readily available, curating equivalent data in other languages is expensive and time-consuming. A common workaround is to translate existing English alignment data; however, standard translation techniques often fail to preserve critical elements such as code, mathematical expressions, and structured formats like JSON. In this work, we investigate LLM-based selective translation, a technique that selectively translates only the translatable parts of a text while preserving non-translatable content and sentence structure. We conduct a systematic study to explore key questions around this approach, including its effectiveness compared to vanilla translation, the importance of filtering noisy outputs, and the benefits of mixing translated samples with original English data during alignment. Our experiments focus on the low-resource Indic language Hindi and compare translations generated by Google Cloud Translation (GCP) and Llama-3.1-405B. The results highlight the promise of selective translation as a practical and effective method for improving multilingual alignment in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.03304v2">Harmony in Divergence: Towards Fast, Accurate, and Memory-efficient Zeroth-order LLM Fine-tuning</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel across various tasks, but standard first-order (FO) fine-tuning demands considerable memory, significantly limiting real-world deployment. Recently, zeroth-order (ZO) optimization stood out as a promising memory-efficient training paradigm, avoiding backward passes and relying solely on forward passes for gradient estimation, making it attractive for resource-constrained scenarios. However, ZO method lags far behind FO method in both convergence speed and accuracy. To bridge the gap, we introduce a novel layer-wise divergence analysis that uncovers the distinct update pattern of FO and ZO optimization. Aiming to resemble the learning capacity of FO method from the findings, we propose Divergence-driven Zeroth-Order (DiZO) optimization. DiZO conducts divergence-driven layer adaptation by incorporating projections to ZO updates, generating diverse-magnitude updates precisely scaled to layer-wise individual optimization needs. Our results demonstrate that DiZO significantly reduces the needed iterations for convergence without sacrificing throughput, cutting training GPU hours by up to 48% on various datasets. Moreover, DiZO consistently outperforms the representative ZO baselines in fine-tuning RoBERTa-large, OPT-series, and Llama-series on downstream tasks and, in some cases, even surpasses memory-intensive FO fine-tuning. Our code is released at https://anonymous.4open.science/r/DiZO-E86D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13394v2">Cross-Lingual Auto Evaluation for Assessing Multilingual LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Evaluating machine-generated text remains a significant challenge in NLP, especially for non-English languages. Current methodologies, including automated metrics, human assessments, and LLM-based evaluations, predominantly focus on English, revealing a significant gap in multilingual evaluation frameworks. We introduce the Cross Lingual Auto Evaluation (CIA) Suite, an extensible framework that includes evaluator LLMs (Hercule) and a novel test set (Recon) specifically designed for multilingual evaluation. Our test set features 500 human-annotated instructions spanning various task capabilities along with human judgment scores across six languages. This would enable benchmarking of general-purpose multilingual LLMs and facilitate meta-evaluation of Evaluator LLMs. The proposed model, Hercule, is a cross-lingual evaluation model that addresses the scarcity of reference answers in the target language by learning to assign scores to responses based on easily available reference answers in English. Our experiments demonstrate that Hercule aligns more closely with human judgments compared to proprietary models, demonstrating the effectiveness of such cross-lingual evaluation in low resource scenarios. Further, it is also effective in zero-shot evaluation on unseen languages. This study is the first comprehensive examination of cross-lingual evaluation using LLMs, presenting a scalable and effective approach for multilingual assessment. All code, datasets, and models will be publicly available to enable further research in this important area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13933v1">Preprint: Did I Just Browse A Website Written by LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 In submission. 2 pages. 3 figures
    </div>
    <details class="paper-abstract">
      Increasingly, web content is automatically generated by large language models (LLMs) with little human input. We call this "LLM-dominant" content. Since LLMs plagiarize and hallucinate, LLM-dominant content can be unreliable and unethical. Yet, websites rarely disclose such content, and human readers struggle to distinguish it. Thus, we must develop reliable detectors for LLM-dominant content. However, state-of-the-art LLM detectors are insufficient, because they perform well mainly on clean, prose-like text, while web content has complex markup and diverse genres. We propose a highly reliable, scalable pipeline that classifies entire websites. Instead of naively classifying text extracted from each page, we classify each site based on an LLM text detector's outputs of multiple prose-like pages. We train and evaluate our detector by collecting 2 distinct ground truth datasets totaling 120 sites, and obtain 100% accuracies testing across them. In the wild, we detect a sizable portion of sites as LLM-dominant among 10k sites in search engine results and 10k in Common Crawl archives. We find LLM-dominant sites are growing in prevalence and rank highly in search results, raising questions about their impact on end users and the overall Web ecosystem.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13881v1">Using LLMs to identify features of personal and professional skills in an open-response situational judgment test</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 10 pages, 2 figures, 4 tables; this work was accepted for presentation at the 2025 Artificial Intelligence in Measurement and Education Conference in Pittsburgh, Pennsylvania, United States
    </div>
    <details class="paper-abstract">
      Academic programs are increasingly recognizing the importance of personal and professional skills and their critical role alongside technical expertise in preparing students for future success in diverse career paths. With this growing demand comes the need for scalable systems to measure, evaluate, and develop these skills. Situational Judgment Tests (SJTs) offer one potential avenue for measuring these skills in a standardized and reliable way, but open-response SJTs have traditionally relied on trained human raters for evaluation, presenting operational challenges to delivering SJTs at scale. Past attempts at developing NLP-based scoring systems for SJTs have fallen short due to issues with construct validity of these systems. In this article, we explore a novel approach to extracting construct-relevant features from SJT responses using large language models (LLMs). We use the Casper SJT to demonstrate the efficacy of this approach. This study sets the foundation for future developments in automated scoring for personal and professional skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13859v1">SPARQL Query Generation with LLMs: Measuring the Impact of Training Data Memorization and Knowledge Injection</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Winner of Best Paper Award at the 25th International Conference on Web Engineering (ICWE 2025)
    </div>
    <details class="paper-abstract">
      Nowadays, the importance of software with natural-language user interfaces cannot be underestimated. In particular, in Question Answering (QA) systems, generating a SPARQL query for a given natural-language question (often named Query Building) from the information retrieved from the same question is the central task of QA systems working over Knowledge Graphs (KGQA). Due to the rise of Large Language Models (LLMs), they are considered a well-suited method to increase the quality of the question-answering functionality, as there is still a lot of room for improvement, aiming for enhanced quality and trustworthiness. However, LLMs are trained on web data, where researchers have no control over whether the benchmark or the knowledge graph was already included in the training data. In this paper, we introduce a novel method that evaluates the quality of LLMs by generating a SPARQL query from a natural-language question under various conditions: (1) zero-shot SPARQL generation, (2) with knowledge injection, and (3) with "anonymized" knowledge injection. This enables us, for the first time, to estimate the influence of the training data on the QA quality improved by LLMs. Ultimately, this will help to identify how portable a method is or whether good results might mostly be achieved because a benchmark was already included in the training data (cf. LLM memorization). The developed method is portable, robust, and supports any knowledge graph; therefore, it could be easily applied to any KGQA or LLM, s.t., generating consistent insights into the actual LLM capabilities is possible.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13833v1">DistFlow: A Fully Distributed RL Framework for Scalable and Efficient LLM Post-Training</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become the pivotal post-training technique for large language model. Effectively scaling reinforcement learning is now the key to unlocking advanced reasoning capabilities and ensuring safe, goal-aligned behavior in the most powerful LLMs. Mainstream frameworks usually employ a hybrid-controller architecture where a single-controller dispatches the overall execution logic and manages overall data transfer and the multi-controller executes distributed computation. For large-scale reinforcement learning, minor load imbalances can introduce significant bottlenecks, ultimately constraining the scalability of the system. To address this limitation, we introduce DistFlow, a novel, fully distributed RL framework designed to break scaling barrier. We adopt a multi-controller paradigm that dispatches data transfer and execution tasks to all workers, which eliminates the centralized node. This allows each worker to operate independently, leading to near-linear scalability up to thousands of GPUs and dramatic efficiency gains. Furthermore, our architecture decouples resource configuration from execution logic, allowing each worker to have a unique execution flow, offering significant flexibility for rapid and cost-effective algorithmic experimentation. Extensive experiments show that DistFlow achieves excellent linear scalability and up to a 7x end-to-end throughput improvement over state-of-the-art (SOTA) frameworks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.04295v3">LearnLens: LLM-Enabled Personalised, Curriculum-Grounded Feedback with Educators in the Loop</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Effective feedback is essential for student learning but is time-intensive for teachers. We present LearnLens, a modular, LLM-based system that generates personalised, curriculum-aligned feedback in science education. LearnLens comprises three components: (1) an error-aware assessment module that captures nuanced reasoning errors; (2) a curriculum-grounded generation module that uses a structured, topic-linked memory chain rather than traditional similarity-based retrieval, improving relevance and reducing noise; and (3) an educator-in-the-loop interface for customisation and oversight. LearnLens addresses key challenges in existing systems, offering scalable, high-quality feedback that empowers both teachers and students.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13822v1">RAG-based Architectures for Drug Side Effect Retrieval in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Drug side effects are a major global health concern, necessitating advanced methods for their accurate detection and analysis. While Large Language Models (LLMs) offer promising conversational interfaces, their inherent limitations, including reliance on black-box training data, susceptibility to hallucinations, and lack of domain-specific knowledge, hinder their reliability in specialized fields like pharmacovigilance. To address this gap, we propose two architectures: Retrieval-Augmented Generation (RAG) and GraphRAG, which integrate comprehensive drug side effect knowledge into a Llama 3 8B language model. Through extensive evaluations on 19,520 drug side effect associations (covering 976 drugs and 3,851 side effect terms), our results demonstrate that GraphRAG achieves near-perfect accuracy in drug side effect retrieval. This framework offers a highly accurate and scalable solution, signifying a significant advancement in leveraging LLMs for critical pharmacovigilance applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13774v2">DP2Unlearning: An Efficient and Guaranteed Unlearning Framework for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 This is the updated version of the preprint, revised following acceptance for publication in Elsevier Neural Networks Journal. The paper is now published (18 July 2025) with DOI: https://doi.org/10.1016/j.neunet.2025.107879
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently revolutionized language processing tasks but have also brought ethical and legal issues. LLMs have a tendency to memorize potentially private or copyrighted information present in the training data, which might then be delivered to end users at inference time. When this happens, a naive solution is to retrain the model from scratch after excluding the undesired data. Although this guarantees that the target data have been forgotten, it is also prohibitively expensive for LLMs. Approximate unlearning offers a more efficient alternative, as it consists of ex post modifications of the trained model itself to prevent undesirable results, but it lacks forgetting guarantees because it relies solely on empirical evidence. In this work, we present DP2Unlearning, a novel LLM unlearning framework that offers formal forgetting guarantees at a significantly lower cost than retraining from scratch on the data to be retained. DP2Unlearning involves training LLMs on textual data protected using {\epsilon}-differential privacy (DP), which later enables efficient unlearning with the guarantees against disclosure associated with the chosen {\epsilon}. Our experiments demonstrate that DP2Unlearning achieves similar model performance post-unlearning, compared to an LLM retraining from scratch on retained data -- the gold standard exact unlearning -- but at approximately half the unlearning cost. In addition, with a reasonable computational cost, it outperforms approximate unlearning methods at both preserving the utility of the model post-unlearning and effectively forgetting the targeted information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08924v2">From KMMLU-Redux to KMMLU-Pro: A Professional Korean Benchmark Suite for LLM Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      The development of Large Language Models (LLMs) requires robust benchmarks that encompass not only academic domains but also industrial fields to effectively evaluate their applicability in real-world scenarios. In this paper, we introduce two Korean expert-level benchmarks. KMMLU-Redux, reconstructed from the existing KMMLU, consists of questions from the Korean National Technical Qualification exams, with critical errors removed to enhance reliability. KMMLU-Pro is based on Korean National Professional Licensure exams to reflect professional knowledge in Korea. Our experiments demonstrate that these benchmarks comprehensively represent industrial knowledge in Korea. We release our dataset publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13743v1">PRIDE -- Parameter-Efficient Reduction of Identity Discrimination for Equality in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently reproduce the gender- and sexual-identity prejudices embedded in their training corpora, leading to outputs that marginalize LGBTQIA+ users. Hence, reducing such biases is of great importance. To achieve this, we evaluate two parameter-efficient fine-tuning (PEFT) techniques - Low-Rank Adaptation (LoRA) and soft-prompt tuning - as lightweight alternatives to full-model fine-tuning for mitigating such biases. Using the WinoQueer benchmark, we quantify bias in three open-source LLMs and observe baseline bias scores reaching up to 98 (out of 100) across a range of queer identities defined by gender and/or sexual orientation, where 50 would indicate neutrality. Fine-tuning with LoRA (< 0.1% additional parameters) on a curated QueerNews corpus reduces those scores by up to 50 points and raises neutrality from virtually 0% to as much as 36%. Soft-prompt tuning (10 virtual tokens) delivers only marginal improvements. These findings show that LoRA can deliver meaningful fairness gains with minimal computation. We advocate broader adoption of community-informed PEFT, the creation of larger queer-authored corpora, and richer evaluation suites beyond WinoQueer, coupled with ongoing audits to keep LLMs inclusive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.02145v4">From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Final Version and Paper Accepted at IEEE ITSC 2025
    </div>
    <details class="paper-abstract">
      Ensuring the safety of autonomous vehicles requires virtual scenario-based testing, which depends on the robust evaluation and generation of safety-critical scenarios. So far, researchers have used scenario-based testing frameworks that rely heavily on handcrafted scenarios as safety metrics. To reduce the effort of human interpretation and overcome the limited scalability of these approaches, we combine Large Language Models (LLMs) with structured scenario parsing and prompt engineering to automatically evaluate and generate safety-critical driving scenarios. We introduce Cartesian and Ego-centric prompt strategies for scenario evaluation, and an adversarial generation module that modifies trajectories of risk-inducing vehicles (ego-attackers) to create critical scenarios. We validate our approach using a 2D simulation framework and multiple pre-trained LLMs. The results show that the evaluation module effectively detects collision scenarios and infers scenario safety. Meanwhile, the new generation module identifies high-risk agents and synthesizes realistic, safety-critical scenarios. We conclude that an LLM equipped with domain-informed prompting techniques can effectively evaluate and generate safety-critical driving scenarios, reducing dependence on handcrafted metrics. We release our open-source code and scenarios at: https://github.com/TUM-AVS/From-Words-to-Collisions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13737v1">DailyLLM: Context-Aware Activity Log Generation Using Multi-Modal Sensors and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Rich and context-aware activity logs facilitate user behavior analysis and health monitoring, making them a key research focus in ubiquitous computing. The remarkable semantic understanding and generation capabilities of Large Language Models (LLMs) have recently created new opportunities for activity log generation. However, existing methods continue to exhibit notable limitations in terms of accuracy, efficiency, and semantic richness. To address these challenges, we propose DailyLLM. To the best of our knowledge, this is the first log generation and summarization system that comprehensively integrates contextual activity information across four dimensions: location, motion, environment, and physiology, using only sensors commonly available on smartphones and smartwatches. To achieve this, DailyLLM introduces a lightweight LLM-based framework that integrates structured prompting with efficient feature extraction to enable high-level activity understanding. Extensive experiments demonstrate that DailyLLM outperforms state-of-the-art (SOTA) log generation methods and can be efficiently deployed on personal computers and Raspberry Pi. Utilizing only a 1.5B-parameter LLM model, DailyLLM achieves a 17% improvement in log generation BERTScore precision compared to the 70B-parameter SOTA baseline, while delivering nearly 10x faster inference speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13729v1">AGENTS-LLM: Augmentative GENeration of Challenging Traffic Scenarios with an Agentic LLM Framework</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Rare, yet critical, scenarios pose a significant challenge in testing and evaluating autonomous driving planners. Relying solely on real-world driving scenes requires collecting massive datasets to capture these scenarios. While automatic generation of traffic scenarios appears promising, data-driven models require extensive training data and often lack fine-grained control over the output. Moreover, generating novel scenarios from scratch can introduce a distributional shift from the original training scenes which undermines the validity of evaluations especially for learning-based planners. To sidestep this, recent work proposes to generate challenging scenarios by augmenting original scenarios from the test set. However, this involves the manual augmentation of scenarios by domain experts. An approach that is unable to meet the demands for scale in the evaluation of self-driving systems. Therefore, this paper introduces a novel LLM-agent based framework for augmenting real-world traffic scenarios using natural language descriptions, addressing the limitations of existing methods. A key innovation is the use of an agentic design, enabling fine-grained control over the output and maintaining high performance even with smaller, cost-effective LLMs. Extensive human expert evaluation demonstrates our framework's ability to accurately adhere to user intent, generating high quality augmented scenarios comparable to those created manually.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13712v1">LLaPipe: LLM-Guided Reinforcement Learning for Automated Data Preparation Pipeline Construction</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Automated data preparation is crucial for democratizing machine learning, yet existing reinforcement learning (RL) based approaches suffer from inefficient exploration in the vast space of possible preprocessing pipelines. We present LLaPipe, a novel framework that addresses this exploration bottleneck by integrating Large Language Models (LLMs) as intelligent policy advisors. Unlike traditional methods that rely solely on statistical features and blind trial-and-error, LLaPipe leverages the semantic understanding capabilities of LLMs to provide contextually relevant exploration guidance. Our framework introduces three key innovations: (1) an LLM Policy Advisor that analyzes dataset semantics and pipeline history to suggest promising preprocessing operations, (2) an Experience Distillation mechanism that mines successful patterns from past pipelines and transfers this knowledge to guide future exploration, and (3) an Adaptive Advisor Triggering strategy (Advisor\textsuperscript{+}) that dynamically determines when LLM intervention is most beneficial, balancing exploration effectiveness with computational cost. Through extensive experiments on 18 diverse datasets spanning multiple domains, we demonstrate that LLaPipe achieves up to 22.4\% improvement in pipeline quality and 2.3$\times$ faster convergence compared to state-of-the-art RL-based methods, while maintaining computational efficiency through selective LLM usage (averaging only 19.0\% of total exploration steps).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.17562v2">LLM-driven Medical Report Generation via Communication-efficient Heterogeneous Federated Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Accepted by IEEE TMI
    </div>
    <details class="paper-abstract">
      LLMs have demonstrated significant potential in Medical Report Generation (MRG), yet their development requires large amounts of medical image-report pairs, which are commonly scattered across multiple centers. Centralizing these data is exceptionally challenging due to privacy regulations, thereby impeding model development and broader adoption of LLM-driven MRG models. To address this challenge, we present FedMRG, the first framework that leverages Federated Learning (FL) to enable privacy-preserving, multi-center development of LLM-driven MRG models, specifically designed to overcome the critical challenge of communication-efficient LLM training under multi-modal data heterogeneity. To start with, our framework tackles the fundamental challenge of communication overhead in FL-LLM tuning by employing low-rank factorization to efficiently decompose parameter updates, significantly reducing gradient transmission costs and making LLM-driven MRG feasible in bandwidth-constrained FL settings. Furthermore, we observed the dual heterogeneity in MRG under the FL scenario: varying image characteristics across medical centers, as well as diverse reporting styles and terminology preferences. To address this, we further enhance FedMRG with (1) client-aware contrastive learning in the MRG encoder, coupled with diagnosis-driven prompts, which capture both globally generalizable and locally distinctive features while maintaining diagnostic accuracy; and (2) a dual-adapter mutual boosting mechanism in the MRG decoder that harmonizes generic and specialized adapters to address variations in reporting styles and terminology. Through extensive evaluation of our established FL-MRG benchmark, we demonstrate the generalizability and adaptability of FedMRG, underscoring its potential in harnessing multi-center data and generating clinically accurate reports while maintaining communication efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17735v2">SafeAgent: Safeguarding LLM Agents via an Automated Risk Simulator</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 38 pages;12 figures;12 tables
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM)-based agents are increasingly deployed in real-world applications such as "digital assistants, autonomous customer service, and decision-support systems", where their ability to "interact in multi-turn, tool-augmented environments" makes them indispensable. However, ensuring the safety of these agents remains a significant challenge due to the diverse and complex risks arising from dynamic user interactions, external tool usage, and the potential for unintended harmful behaviors. To address this critical issue, we propose AutoSafe, the first framework that systematically enhances agent safety through fully automated synthetic data generation. Concretely, 1) we introduce an open and extensible threat model, OTS, which formalizes how unsafe behaviors emerge from the interplay of user instructions, interaction contexts, and agent actions. This enables precise modeling of safety risks across diverse scenarios. 2) we develop a fully automated data generation pipeline that simulates unsafe user behaviors, applies self-reflective reasoning to generate safe responses, and constructs a large-scale, diverse, and high-quality safety training dataset-eliminating the need for hazardous real-world data collection. To evaluate the effectiveness of our framework, we design comprehensive experiments on both synthetic and real-world safety benchmarks. Results demonstrate that AutoSafe boosts safety scores by 45% on average and achieves a 28.91% improvement on real-world tasks, validating the generalization ability of our learned safety strategies. These results highlight the practical advancement and scalability of AutoSafe in building safer LLM-based agents for real-world deployment. We have released the project page at https://auto-safe.github.io/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13705v1">Consistent Explainers or Unreliable Narrators? Understanding LLM-generated Group Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Short paper accepted at the Nineteenth ACM Conference on Recommender Systems (RecSys '25). Cedric Waterschoot, Nava Tintarev, and Francesco Barile. 2025. Consistent Explainers or Unreliable Narrators? Understanding LLM-generated Group Recommendations. Proceedings of the Nineteenth ACM Conference on Recommender Systems (RecSys '25), Prague, Czech Republic. doi: 10.1145/3705328.3748015
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being implemented as joint decision-makers and explanation generators for Group Recommender Systems (GRS). In this paper, we evaluate these recommendations and explanations by comparing them to social choice-based aggregation strategies. Our results indicate that LLM-generated recommendations often resembled those produced by Additive Utilitarian (ADD) aggregation. However, the explanations typically referred to averaging ratings (resembling but not identical to ADD aggregation). Group structure, uniform or divergent, did not impact the recommendations. Furthermore, LLMs regularly claimed additional criteria such as user or item similarity, diversity, or used undefined popularity metrics or thresholds. Our findings have important implications for LLMs in the GRS pipeline as well as standard aggregation strategies. Additional criteria in explanations were dependent on the number of ratings in the group scenario, indicating potential inefficiency of standard aggregation methods at larger item set sizes. Additionally, inconsistent and ambiguous explanations undermine transparency and explainability, which are key motivations behind the use of LLMs for GRS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20839v3">FireQ: Fast INT4-FP8 Kernel and RoPE-aware Quantization for LLM Inference Acceleration</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      As large language models become increasingly prevalent, memory bandwidth constraints significantly limit inference throughput, motivating post-training quantization (PTQ). In this paper, we propose FireQ, a co-designed PTQ framework and an INT4-FP8 matrix multiplication kernel that accelerates LLM inference across all linear layers. Specifically, FireQ quantizes linear layer weights and key-values to INT4, and activations and queries to FP8, significantly enhancing throughput. Additionally, we introduce a three-stage pipelining for the prefill phase, which modifies the FlashAttention-3 kernel, effectively reducing time-to-first-token in the prefill phase. To minimize accuracy loss from quantization, we develop novel outlier smoothing techniques tailored separately for linear and attention layers. In linear layers, we explicitly use per-tensor scaling to prevent underflow caused by the FP8 quantization scaling factor of INT4 quantization, and channel-wise scaling to compensate for coarse granularity of INT4. In attention layers, we address quantization challenges posed by rotary positional embeddings (RoPE) by combining pre-RoPE and post-RoPE scaling strategies. FireQ significantly outperforms state-of-the-art methods, achieving 1.68x faster inference in feed-forward network layers on Llama2-7B and 1.26x faster prefill phase performance on Llama3-8B compared to QServe, with negligible accuracy loss.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13681v1">LoopServe: An Adaptive Dual-phase LLM Inference Acceleration System for Multi-Turn Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Multi-turn dialogues are essential in many real-world applications of large language models, such as chatbots and virtual assistants. As conversation histories become longer, existing large language models face increasing computational and memory challenges, which hinder their ability to provide efficient and responsive interactions. Most current acceleration methods either compress the context or optimize key value caching, but they often rely on fixed or position-based heuristics that do not adapt well to the dynamic and unpredictable patterns found in actual multi-turn conversations. In this paper, we present LoopServe, an adaptive dual-phase inference acceleration framework for large language models in multi-turn dialogues. LoopServe introduces two main innovations. First, it performs online sparsification during the prefilling phase by dynamically selecting the most important parts of the attention matrix for each new input. Second, it uses progressive key value compression during decoding by adaptively maintaining a relevant and efficient cache based on the most recently generated output tokens. We also propose a \href{https://huggingface.co/datasets/TreeAILab/Multi-turn_Long-context_Benchmark_for_LLMs}{new benchmark} with eleven multi-turn datasets that reflect realistic query positions and conversational dependencies. Extensive experiments demonstrate that LoopServe consistently achieves superior effectiveness compared to existing baselines and significantly accelerates LLM inference across a wide range of long-context dialogue tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13666v1">KiC: Keyword-inspired Cascade for Cost-Efficient Text Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated state-of-the-art performance across a wide range of natural language processing tasks. However, high-performing models are typically accessible only via APIs, incurring substantial inference costs. Cascade methods address this by initially employing a cheaper model and escalating to a stronger one only when necessary. Nevertheless, existing cascade approaches struggle to select a reliable representative response and assess the overall reliability of free-form outputs, as they rely on exact text matching. To overcome these limitations, we propose Keyword-inspired Cascade (KiC), a novel framework for cost-efficient free-form text generation. KiC identifies the most representative answer among multiple outputs from a weaker model and evaluates the semantic alignment of other responses with it. Based on the degree of alignment, KiC determines whether to accept the weaker model's output or escalate to a stronger model. Experiments on three free-form text generation benchmarks show that KiC achieves 97.53 percent of GPT-4's accuracy while reducing API costs by 28.81 percent on average, and even outperforms GPT-4 in a specific benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08392v2">Multi-Agent LLMs as Ethics Advocates for AI-Based Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Incorporating ethics into the requirement elicitation process is essential for creating ethically aligned systems. Although eliciting manual ethics requirements is effective, it requires diverse input from multiple stakeholders, which can be challenging due to time and resource constraints. Moreover, it is often given a low priority in the requirements elicitation process. This study proposes a framework for generating ethics requirements drafts by introducing an ethics advocate agent in a multi-agent LLM setting. This agent critiques and provides input on ethical issues based on the system description. The proposed framework is evaluated through two case studies from different contexts, demonstrating that it captures the majority of ethics requirements identified by researchers during 30-minute interviews and introduces several additional relevant requirements. However, it also highlights reliability issues in generating ethics requirements, emphasizing the need for human feedback in this sensitive domain. We believe this work can facilitate the broader adoption of ethics in the requirements engineering process, ultimately leading to more ethically aligned products.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04834v4">LLM-Based Multi-Agent Systems for Software Engineering: Literature Review, Vision and the Road Ahead</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 TOSEM 2030 Special Issue
    </div>
    <details class="paper-abstract">
      Integrating Large Language Models (LLMs) into autonomous agents marks a significant shift in the research landscape by offering cognitive abilities that are competitive with human planning and reasoning. This paper explores the transformative potential of integrating Large Language Models into Multi-Agent (LMA) systems for addressing complex challenges in software engineering (SE). By leveraging the collaborative and specialized abilities of multiple agents, LMA systems enable autonomous problem-solving, improve robustness, and provide scalable solutions for managing the complexity of real-world software projects. In this paper, we conduct a systematic review of recent primary studies to map the current landscape of LMA applications across various stages of the software development lifecycle (SDLC). To illustrate current capabilities and limitations, we perform two case studies to demonstrate the effectiveness of state-of-the-art LMA frameworks. Additionally, we identify critical research gaps and propose a comprehensive research agenda focused on enhancing individual agent capabilities and optimizing agent synergy. Our work outlines a forward-looking vision for developing fully autonomous, scalable, and trustworthy LMA systems, laying the foundation for the evolution of Software Engineering 2.0.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01551v2">EvolveNav: Self-Improving Embodied Reasoning for LLM-Based Vision-Language Navigation</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Building Vision-Language Navigation (VLN) agents which can navigate following natural language instructions is a long-standing goal in human-robot interaction applications. Recent studies have revealed the potential of training open-source Large Language Models (LLMs) to unleash LLMs' reasoning ability for improving navigation, and simultaneously mitigate the domain gap between LLMs' training corpus and the VLN task. However, these approaches primarily adopt direct input-output mapping paradigms, causing the mapping learning difficult and the navigational decisions unexplainable. Chain-of-Thought (CoT) training is a promising way to improve both navigational decision accuracy and interpretability, while the complexity of the navigation task makes the perfect CoT labels unavailable and may lead to overfitting through pure CoT supervised fine-tuning. In this paper, we propose a novel sElf-improving embodied reasoning framework for boosting LLM-based vision-language Navigation, dubbed EvolveNav. Our EvolveNav consists of two stages: (1) Formalized CoT Supervised Fine-Tuning, where we train the model with formalized CoT labels to both activate the model's navigational reasoning capabilities and increase the reasoning speed; (2) Self-Reflective Post-Training, where the model is iteratively trained with its own reasoning outputs as self-enriched CoT labels to enhance the supervision diversity. A self-reflective auxiliary task is also introduced to encourage learning correct reasoning patterns by contrasting with wrong ones. Experimental results on the popular VLN benchmarks demonstrate the superiority of EvolveNav over previous LLM-based VLN approaches. Code is available at https://github.com/expectorlin/EvolveNav.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13618v1">Seed-X: Building Strong Multilingual Translation LLM with 7B Parameters</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Multilingual translation stands as a challenging task for large language models (LLMs) to handle intricate language patterns and stilted translations that arise in automated translations. In this paper, we introduce Seed-X, a family of open-source LLMs comprising instruct and reasoning models, pushing the limits of translation capability with 7B parameter size. The base model is pre-trained on a diverse, high-quality dataset encompassing both monolingual and bilingual content across 28 languages, harnessing the full potential of multilingual data. The instruct model is then finetuned to translate by Chain-of-Thought (CoT) reasoning and further enhanced through reinforcement learning (RL) to achieve better generalization across diverse language pairs. Seed-X achieves performance comparable to leading closed-source models, including Gemini-2.5 and GPT-4o, across 28 languages, and significantly outperforms larger open-source models in both automatic metrics and human evaluations. We share the best practices through our optimization process, and make the parameter public available for advancing translation research and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03993v4">TR-LLM: Integrating Trajectory Data for Scene-Aware LLM-Based Human Action Prediction</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 Accepted to IROS 2025
    </div>
    <details class="paper-abstract">
      Accurate prediction of human behavior is crucial for AI systems to effectively support real-world applications, such as autonomous robots anticipating and assisting with human tasks. Real-world scenarios frequently present challenges such as occlusions and incomplete scene observations, which can compromise predictive accuracy. Thus, traditional video-based methods often struggle due to limited temporal and spatial perspectives. Large Language Models (LLMs) offer a promising alternative. Having been trained on a large text corpus describing human behaviors, LLMs likely encode plausible sequences of human actions in a home environment. However, LLMs, trained primarily on text data, lack inherent spatial awareness and real-time environmental perception. They struggle with understanding physical constraints and spatial geometry. Therefore, to be effective in a real-world spatial scenario, we propose a multimodal prediction framework that enhances LLM-based action prediction by integrating physical constraints derived from human trajectories. Our experiments demonstrate that combining LLM predictions with trajectory data significantly improves overall prediction performance. This enhancement is particularly notable in situations where the LLM receives limited scene information, highlighting the complementary nature of linguistic knowledge and physical constraints in understanding and anticipating human behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12674v2">ParaStudent: Generating and Evaluating Realistic Student Code by Teaching LLMs to Struggle</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown strong performance on programming tasks, but can they generate student-like code like real students - imperfect, iterative, and stylistically diverse? We present ParaStudent, a systematic study of LLM-based "student-like" code generation in an introductory programming course setting. Using a dataset of timestamped student submissions across multiple semesters, we design low- and high-resolution experiments to model student progress and evaluate code outputs along semantic, functional, and stylistic dimensions. Our results show that fine-tuning significantly improves alignment with real student trajectories and captures error patterns, incremental improvements, and stylistic variations more faithfully. This study shows that modeling realistic student code requires capturing learning dynamics through context-aware generation, temporal modeling, and multi-dimensional evaluation. Code for experiments and evaluation is available at https://github.com/mmiroyan/ParaStudent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10622v4">A recent evaluation on the performance of LLMs on radiation oncology physics using questions of randomly shuffled options</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Purpose: We present an updated study evaluating the performance of large language models (LLMs) in answering radiation oncology physics questions, focusing on the recently released models. Methods: A set of 100 multiple-choice radiation oncology physics questions, previously created by a well-experienced physicist, was used for this study. The answer options of the questions were randomly shuffled to create "new" exam sets. Five LLMs -- OpenAI o1-preview, GPT-4o, LLaMA 3.1 (405B), Gemini 1.5 Pro, and Claude 3.5 Sonnet -- with the versions released before September 30, 2024, were queried using these new exam sets. To evaluate their deductive reasoning ability, the correct answer options in the questions were replaced with "None of the above." Then, the explain-first and step-by-step instruction prompts were used to test if this strategy improved their reasoning ability. The performance of the LLMs was compared with the answers from medical physicists. Results: All models demonstrated expert-level performance on these questions, with o1-preview even surpassing medical physicists with a majority vote. When replacing the correct answer options with 'None of the above', all models exhibited a considerable decline in performance, suggesting room for improvement. The explain-first and step-by-step instruction prompts helped enhance the reasoning ability of the LLaMA 3.1 (405B), Gemini 1.5 Pro, and Claude 3.5 Sonnet models. Conclusion: These recently released LLMs demonstrated expert-level performance in answering radiation oncology physics questions, exhibiting great potential to assist in radiation oncology physics education and training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15838v2">Enhancing LLM Code Generation with Ensembles: A Similarity-Based Selection Approach</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Ensemble learning has been widely used in machine learning to improve model robustness, accuracy, and generalization, but has not yet been applied to code generation tasks with large language models (LLMs). We propose an ensemble approach for LLMs in code generation. Instead of relying on the output of a single model, we generate multiple candidate programs from different LLMs and apply a structured voting mechanism to select the most reliable solution. For voting, we compute syntactic and semantic similarity using CodeBLEU and behavioral equivalence using CrossHair's differential behavior analysis. By aggregating these similarity scores, we select the program that best aligns with the consensus among the candidates. We show through experiments that our ensemble approach consistently outperforms standalone LLMs on the well-known HumanEval and the more challenging LiveCodeBench datasets, achieving an accuracy of 90.2% and 50.2%, respectively, on the two datasets. In comparison, the best-performing LLM (GPT-4o) has an accuracy of 83.5% and 43.4%, respectively. Furthermore, even when restricted to free open-source models, our method achieves an accuracy of 80.5% and 41.6%, respectively, demonstrating the viability of our approach in resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14406v1">Fail Fast, or Ask: Mitigating the Deficiencies of Reasoning LLMs with Human-in-the-Loop Systems Engineering</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 8 pages, 5 figures
    </div>
    <details class="paper-abstract">
      State-of-the-art reasoning LLMs are powerful problem solvers, but they still occasionally make mistakes. However, adopting AI models in risk-sensitive domains often requires error rates near 0%. To address this gap, we propose collaboration between a reasoning model and a human expert who resolves queries the model cannot confidently answer. We find that quantifying the uncertainty of a reasoning model through the length of its reasoning trace yields an effective basis for deferral to a human, e.g., cutting the error rate of Qwen3 235B-A22B on difficult MATH problems from 3% to less than 1% when deferring 7.5% of queries. However, the high latency of reasoning models still makes them challenging to deploy on use cases with high query volume. To address this challenge, we explore fronting a reasoning model with a large non-reasoning model. We call this modified human-in-the-loop system "Fail Fast, or Ask", since the non-reasoning model may defer difficult queries to the human expert directly ("failing fast"), without incurring the reasoning model's higher latency. We show that this approach yields around 40% latency reduction and about 50% cost savings for DeepSeek R1 while maintaining 90+% area under the accuracy-rejection curve. However, we observe that latency savings are lower than expected because of "latency drag", the phenomenon that processing easier queries with a non-reasoning model pushes the reasoning model's latency distribution towards longer latencies. Broadly, our results suggest that the deficiencies of state-of-the-art reasoning models -- nontrivial error rates and high latency -- can be substantially mitigated through black-box systems engineering, without requiring access to LLM internals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14403v1">NPUEval: Optimizing NPU Kernels with LLMs and Open Source Compilers</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Neural processing units (NPUs) are gaining prominence in power-sensitive devices like client devices, with AI PCs being defined by their inclusion of these specialized processors. Running AI workloads efficiently on these devices requires libraries of optimized kernels. Creating efficient kernels demands expertise in domain-specific C++ with vector intrinsics and in-depth knowledge of the target architecture. Unlike GPU programming, which has had years to mature, NPU programming is new, with smaller and more fragmented developer communities across hardware platforms. This fragmentation poses a challenge when utilizing LLMs to assist in writing NPU kernels, as domain-specific optimized code examples are underrepresented in LLM pre-training data. In this paper we introduce NPUEval -- a benchmark for writing and evaluating NPU kernels, consisting of 102 common operators for machine learning workloads. We evaluate LLM generated code on actual hardware based on both functional correctness and vectorization efficiency using open source compiler tools targeting the AMD NPU. We evaluate a range of state-of-the-art LLMs with a mix of proprietary and open-weight models. Latest reasoning models like DeepSeek R1, show promising results achieving out-of-the-box 50%+ vectorization on select kernels. However, the average score across the entire dataset remains roughly 10% even with compiler feedback and vectorized kernel examples -- showing that this is a challenging dataset even for frontier models. The dataset and evaluation code will be released with a permissive open source license, providing an essential benchmark for advancing research in code generation and NPU kernel optimization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14397v1">Efficient LLM Inference: Bandwidth, Compute, Synchronization, and Capacity are all you need</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      This paper presents a limit study of transformer-based large language model (LLM) inference, focusing on the fundamental performance bottlenecks imposed by memory bandwidth, memory capacity, and synchronization overhead in distributed inference systems. We develop a hardware-agnostic performance model that abstracts away implementation details, enabling the analysis of a wide range of current and near-future hardware technologies. Our analysis spans from current HBM3 memory technology used in AI accelerators like GPUs and TPUs to systems based on advanced HBM4 and advanced 3D-stacked DRAM technology. It also covers SRAM-based designs and scaling techniques from distributed clusters with varying numbers of chips to wafer-scale integration. Our key findings for auto-regressive decoding are: i) serving LLMs requires 100s of GB per server to serve a model instance; ii) high memory bandwidth is critical for high per-user throughput; iii) exposed synchronization latencies to achieve collective communication must be around 1us else they make the memory bandwidth ineffective; iv) DRAM-based designs have a fundamental advantage in terms of system-level efficiency as measured in throughput per cost or watt; and v) hardware designs can easily reach 2000+ user token/sec but getting to 10,000+ tokens/sec will need smaller models, smaller context, or other forms of algorithmic advances. This study provides valuable insights into the fundamental performance limits of LLM inference, highlighting the potential benefits of future hardware advancements and guiding the optimization of LLM deployment strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10968v2">Combinatorial Optimization for All: Using LLMs to Aid Non-Experts in Improving Optimization Algorithms</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown notable potential in code generation for optimization algorithms, unlocking exciting new opportunities. This paper examines how LLMs, rather than creating algorithms from scratch, can improve existing ones without the need for specialized expertise. To explore this potential, we selected 10 baseline optimization algorithms from various domains (metaheuristics, reinforcement learning, deterministic, and exact methods) to solve the classic Travelling Salesman Problem. The results show that our simple methodology often results in LLM-generated algorithm variants that improve over the baseline algorithms in terms of solution quality, reduction in computational time, and simplification of code complexity, all without requiring specialized optimization knowledge or advanced algorithmic implementation skills.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14376v1">Schemora: schema matching via multi-stage recommendation and metadata enrichment using off-the-shelf llms</a></div>
    <div class="paper-meta">
      📅 2025-07-18
      | 💬 11 pages
    </div>
    <details class="paper-abstract">
      Schema matching is essential for integrating heterogeneous data sources and enhancing dataset discovery, yet it remains a complex and resource-intensive problem. We introduce SCHEMORA, a schema matching framework that combines large language models with hybrid retrieval techniques in a prompt-based approach, enabling efficient identification of candidate matches without relying on labeled training data or exhaustive pairwise comparisons. By enriching schema metadata and leveraging both vector-based and lexical retrieval, SCHEMORA improves matching accuracy and scalability. Evaluated on the MIMIC-OMOP benchmark, it establishes new state-of-the-art performance, with gains of 7.49% in HitRate@5 and 3.75% in HitRate@3 over previous best results. To our knowledge, this is the first LLM-based schema matching method with an open-source implementation, accompanied by analysis that underscores the critical role of retrieval and provides practical guidance on model selection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.10871v2">Layerwise Recall and the Geometry of Interwoven Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-18
    </div>
    <details class="paper-abstract">
      This study explores how large language models (LLMs) encode interwoven scientific knowledge, using chemical elements and LLaMA-series models as a case study. We identify a 3D spiral structure in the hidden states that aligns with the conceptual structure of the periodic table, suggesting that LLMs can reflect the geometric organization of scientific concepts learned from text. Linear probing reveals that middle layers encode continuous, overlapping attributes that enable indirect recall, while deeper layers sharpen categorical distinctions and incorporate linguistic context. These findings suggest that LLMs represent symbolic knowledge not as isolated facts, but as structured geometric manifolds that intertwine semantic information across layers. We hope this work inspires further exploration of how LLMs represent and reason about scientific knowledge, particularly in domains such as materials science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13335v1">Comparing Apples to Oranges: A Dataset & Analysis of LLM Humour Understanding from Traditional Puns to Topical Jokes</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      Humour, as a complex language form, is derived from myriad aspects of life, whilst existing work on computational humour has focussed almost exclusively on short pun-based jokes. In this work, we investigate whether the ability of Large Language Models (LLMs) to explain humour depends on the particular humour form. We compare models on simple puns and more complex topical humour that requires knowledge of real-world entities and events. In doing so, we curate a dataset of 600 jokes split across 4 joke types and manually write high-quality explanations. These jokes include heterographic and homographic puns, contemporary internet humour, and topical jokes, where understanding relies on reasoning beyond "common sense", rooted instead in world knowledge regarding news events and pop culture. Using this dataset, we compare the zero-shot abilities of a range of LLMs to accurately and comprehensively explain jokes of different types, identifying key research gaps in the task of humour explanation. We find that none of the tested models (inc. reasoning models) are capable of reliably generating adequate explanations of all joke types, further highlighting the narrow focus of most works in computational humour on overly simple joke forms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13323v1">GeoReg: Weight-Constrained Few-Shot Regression for Socio-Economic Estimation using LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 15 pages, 13 figures, 7 tables
    </div>
    <details class="paper-abstract">
      Socio-economic indicators like regional GDP, population, and education levels, are crucial to shaping policy decisions and fostering sustainable development. This research introduces GeoReg a regression model that integrates diverse data sources, including satellite imagery and web-based geospatial information, to estimate these indicators even for data-scarce regions such as developing countries. Our approach leverages the prior knowledge of large language model (LLM) to address the scarcity of labeled data, with the LLM functioning as a data engineer by extracting informative features to enable effective estimation in few-shot settings. Specifically, our model obtains contextual relationships between data features and the target indicator, categorizing their correlations as positive, negative, mixed, or irrelevant. These features are then fed into the linear estimator with tailored weight constraints for each category. To capture nonlinear patterns, the model also identifies meaningful feature interactions and integrates them, along with nonlinear transformations. Experiments across three countries at different stages of development demonstrate that our model outperforms baselines in estimating socio-economic indicators, even for low-income countries with limited data availability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13302v1">The Generative Energy Arena (GEA): Incorporating Energy Awareness in Large Language Model (LLM) Human Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      The evaluation of large language models is a complex task, in which several approaches have been proposed. The most common is the use of automated benchmarks in which LLMs have to answer multiple-choice questions of different topics. However, this method has certain limitations, being the most concerning, the poor correlation with the humans. An alternative approach, is to have humans evaluate the LLMs. This poses scalability issues as there is a large and growing number of models to evaluate making it impractical (and costly) to run traditional studies based on recruiting a number of evaluators and having them rank the responses of the models. An alternative approach is the use of public arenas, such as the popular LM arena, on which any user can freely evaluate models on any question and rank the responses of two models. The results are then elaborated into a model ranking. An increasingly important aspect of LLMs is their energy consumption and, therefore, evaluating how energy awareness influences the decisions of humans in selecting a model is of interest. In this paper, we present GEA, the Generative Energy Arena, an arena that incorporates information on the energy consumption of the model in the evaluation process. Preliminary results obtained with GEA are also presented, showing that for most questions, when users are aware of the energy consumption, they favor smaller and more energy efficient models. This suggests that for most user interactions, the extra cost and energy incurred by the more complex and top-performing models do not provide an increase in the perceived quality of the responses that justifies their use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13290v1">Towards Formal Verification of LLM-Generated Code from Natural Language Prompts</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 31 pages, 9 figures
    </div>
    <details class="paper-abstract">
      In the past few years LLMs have emerged as a tool that can aid programmers by taking natural language descriptions and generating code based on it. However, LLMs often generate incorrect code that users need to fix and the literature suggests users often struggle to detect these errors. In this work we seek to offer formal guarantees of correctness to LLM generated code; such guarantees could improve the experience of using AI Code Assistants and potentially enable natural language programming for users with little or no programming knowledge. To address this challenge we propose to incorporate a formal query language that can represent a user's intent in a formally defined but natural language-like manner that a user can confirm matches their intent. Then, using such a query we propose to verify LLM generated code to ensure it matches the user's intent. We implement these ideas in our system, Astrogator, for the Ansible programming language which includes such a formal query language, a calculus for representing the behavior of Ansible programs, and a symbolic interpreter which is used for the verification. On a benchmark suite of 21 code-generation tasks, our verifier is able to verify correct code in 83% of cases and identify incorrect code in 92%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13266v1">QuestA: Expanding Reasoning Capacity in LLMs via Question Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 19 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL) has become a key component in training large language reasoning models (LLMs). However, recent studies questions its effectiveness in improving multi-step reasoning-particularly on hard problems. To address this challenge, we propose a simple yet effective strategy via Question Augmentation: introduce partial solutions during training to reduce problem difficulty and provide more informative learning signals. Our method, QuestA, when applied during RL training on math reasoning tasks, not only improves pass@1 but also pass@k-particularly on problems where standard RL struggles to make progress. This enables continual improvement over strong open-source models such as DeepScaleR and OpenMath Nemotron, further enhancing their reasoning capabilities. We achieve new state-of-the-art results on math benchmarks using 1.5B-parameter models: 67.1% (+5.3%) on AIME24, 59.5% (+10.0%) on AIME25, and 35.5% (+4.0%) on HMMT25. Further, we provide theoretical explanations that QuestA improves sample efficiency, offering a practical and generalizable pathway for expanding reasoning capability through RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10917v2">LLM-Driven Dual-Level Multi-Interest Modeling for Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Recently, much effort has been devoted to modeling users' multi-interests based on their behaviors or auxiliary signals. However, existing methods often rely on heuristic assumptions, e.g., co-occurring items indicate the same interest of users, failing to capture user multi-interests aligning with real-world scenarios. While large language models (LLMs) show significant potential for multi-interest analysis due to their extensive knowledge and powerful reasoning capabilities, two key challenges remain. First, the granularity of LLM-driven multi-interests is agnostic, possibly leading to overly fine or coarse interest grouping. Second, individual user analysis provides limited insights due to the data sparsity issue. In this paper, we propose an LLM-driven dual-level multi-interest modeling framework for more effective recommendation. At the user-individual level, we exploit LLMs to flexibly allocate items engaged by users into different semantic clusters, indicating their diverse and distinct interests. To alleviate the agnostic generation of LLMs, we adaptively assign these semantic clusters to users' collaborative multi-interests learned from global user-item interactions, allowing the granularity to be automatically adjusted according to the user's behaviors using an alignment module. To alleviate the limited insights derived from individual users' behaviors, at the user-crowd level, we propose aggregating user cliques into synthesized users with rich behaviors for more comprehensive LLM-driven multi-interest analysis. We formulate a max covering problem to ensure the compactness and representativeness of synthesized users' behaviors, and then conduct contrastive learning based on their LLM-driven multi-interests to disentangle item representations among different interests. Experiments on real-world datasets show the superiority of our approach against state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13175v1">Black Box Deployed -- Functional Criteria for Artificial Moral Agents in the LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 42 pages. Supplementary material included at end of article
    </div>
    <details class="paper-abstract">
      The advancement of powerful yet opaque large language models (LLMs) necessitates a fundamental revision of the philosophical criteria used to evaluate artificial moral agents (AMAs). Pre-LLM frameworks often relied on the assumption of transparent architectures, which LLMs defy due to their stochastic outputs and opaque internal states. This paper argues that traditional ethical criteria are pragmatically obsolete for LLMs due to this mismatch. Engaging with core themes in the philosophy of technology, this paper proffers a revised set of ten functional criteria to evaluate LLM-based artificial moral agents: moral concordance, context sensitivity, normative integrity, metaethical awareness, system resilience, trustworthiness, corrigibility, partial transparency, functional autonomy, and moral imagination. These guideposts, applied to what we term "SMA-LLS" (Simulating Moral Agency through Large Language Systems), aim to steer AMAs toward greater alignment and beneficial societal integration in the coming years. We illustrate these criteria using hypothetical scenarios involving an autonomous public bus (APB) to demonstrate their practical applicability in morally salient contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13123v1">Detecting LLM-generated Code with Subtle Modification by Adversarial Training</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      With the rapid development of Large Language Models (LLMs), their powerful code-generation capabilities have been widely applied in tasks like code completion and automated development, demonstrating the value of improving coding efficiency. However, the extensive use of LLM-generated code also raises several new challenges. On the one hand, issues such as the regulation of code provenance, copyright disputes, and code quality have become increasingly concerning. How to effectively detect LLM-generated code and ensure its compliant and responsible use has become a critical and urgent issue. On the other hand, in practical applications, LLM-generated code is often subject to manual modifications, such as variable renaming or structural adjustments. Although some recent studies have proposed training-based and zero-shot methods for detecting LLM-generated code, these approaches show insufficient robustness when facing modified LLM-generated code, and there is a lack of an effective solution. To address the real-world scenario where LLM-generated code may undergo minor modifications, we propose CodeGPTSensor+, an enhanced version of CodeGPTSensor, which employs adversarial training to improve robustness against input perturbations. CodeGPTSensor+ integrates an adversarial sample generation module, Multi-objective Identifier and Structure Transformation (MIST), which systematically generates both high-quality and representative adversarial samples. This module effectively enhances the model's resistance against diverse adversarial attacks. Experimental results on the HMCorp dataset demonstrate that CodeGPTSensor+ significantly improves detection accuracy on the adversarial test set while maintaining high accuracy on the original test set, showcasing superior robustness compared to CodeGPTSensor.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13105v1">SemCSE: Semantic Contrastive Sentence Embeddings Using LLM-Generated Summaries For Scientific Abstracts</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      We introduce SemCSE, an unsupervised method for learning semantic embeddings of scientific texts. Building on recent advances in contrastive learning for text embeddings, our approach leverages LLM-generated summaries of scientific abstracts to train a model that positions semantically related summaries closer together in the embedding space. This resulting objective ensures that the model captures the true semantic content of a text, in contrast to traditional citation-based approaches that do not necessarily reflect semantic similarity. To validate this, we propose a novel benchmark designed to assess a model's ability to understand and encode the semantic content of scientific texts, demonstrating that our method enforces a stronger semantic separation within the embedding space. Additionally, we evaluate SemCSE on the comprehensive SciRepEval benchmark for scientific text embeddings, where it achieves state-of-the-art performance among models of its size, thus highlighting the benefits of a semantically focused training approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06796v3">Write Your Own CodeChecker: An Automated Test-Driven Checker Development Approach with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 update metadata and artifact url
    </div>
    <details class="paper-abstract">
      With the rising demand for code quality assurance, developers are not only utilizing existing static code checkers but also seeking custom checkers to satisfy their specific needs. Nowadays, various code-checking frameworks provide extensive checker customization interfaces to meet this need. However, both the abstract checking logic and the complex API usage of large-scale checker frameworks make this task challenging. To this end, automated code checker generation is anticipated to ease the burden of checker development. In this paper, we propose AutoChecker, an innovative LLM-powered approach that can write code checkers automatically based on only a rule description and a test suite. To achieve comprehensive checking logic, AutoChecker incrementally updates the checker's logic by focusing on solving one selected case each time. To obtain precise API knowledge, during each iteration, it leverages fine-grained logic-guided API-context retrieval, where it first decomposes the checking logic into a series of sub-operations and then retrieves checker-related API-contexts for each sub-operation. For evaluation, we apply AutoChecker, five baselines, and three ablation methods using multiple LLMs to generate checkers for 20 randomly selected PMD rules. Experimental results show that AutoChecker significantly outperforms others across all effectiveness metrics, with an average test pass rate of 82.28%. Additionally, the checkers generated by AutoChecker can be successfully applied to real-world projects, matching the performance of official checkers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12948v1">Probabilistic Soundness Guarantees in LLM Reasoning Chains</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      In reasoning chains generated by large language models (LLMs), initial errors often propagate and undermine the reliability of the final conclusion. Current LLM-based error detection methods often fail to detect propagated errors because they do not properly account for how earlier errors might corrupt judgments of downstream reasoning. To better detect such propagated errors, we introduce Autoregressive Reasoning Entailment Stability (ARES), a novel probabilistic framework that prevents error propagation by judging each claim based only on previously-assessed sound premises. This inductive method yields a nuanced score for each step and provides certified statistical guarantees of its soundness, rather than a brittle binary label. ARES achieves state-of-the-art performance across four benchmarks (72.1% Macro-F1, +8.2 points) and demonstrates superior robustness on very long synthetic reasoning chains, where it excels at detecting propagated errors (90.3% F1, +27.6 points).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06208v3">IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 ACL 2025
    </div>
    <details class="paper-abstract">
      In the realm of large language models (LLMs), the ability of models to accurately follow instructions is paramount as more agents and applications leverage LLMs for construction, where the complexity of instructions are rapidly increasing. However, on the one hand, there is only a certain amount of complex instruction evaluation data; on the other hand, there are no dedicated algorithms to improve the ability to follow complex instructions. To this end, this paper introduces TRACE, a benchmark for improving and evaluating the complex instructionfollowing ability, which consists of 120K training data and 1K evaluation data. Furthermore, we propose IOPO (Input-Output Preference Optimization) alignment method which takes both input and output preference pairs into consideration, where LLMs not only rapidly align with response preferences but also meticulously explore the instruction preferences. Extensive experiments on both in-domain and outof-domain datasets confirm the effectiveness of IOPO, showing 8.15%, 2.18% improvements on in-domain data and 6.29%, 3.13% on outof-domain data compared to SFT and DPO respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.04631v2">On the Limitations of Large Language Models (LLMs): False Attribution</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 This paper was accepted for presentation by Recent Advances in NLP (RANLP) 2025 conference
    </div>
    <details class="paper-abstract">
      In this work, we introduce a new hallucination metric - Simple Hallucination Index (SHI) and provide insight into one important limitation of the parametric knowledge of large language models (LLMs), i.e. false attribution. The task of automatic author attribution for relatively small chunks of text is an important NLP task but can be challenging. We empirically evaluate the power of 3 open SotA LLMs in zero-shot setting (Gemma-7B, Mixtral 8x7B, and LLaMA-2-13B). We acquired the top 10 most popular books of a month, according to Project Gutenberg, divided each one into equal chunks of 400 words, and prompted each LLM to predict the author. We then randomly sampled 162 chunks per book for human evaluation, based on the error margin of 7% and a confidence level of 95%. The average results show that Mixtral 8x7B has the highest prediction accuracy, the lowest SHI, and a Pearson's correlation (r) of 0.724, 0.263, and -0.9996, respectively, followed by LLaMA-2-13B and Gemma-7B. However, Mixtral 8x7B suffers from high hallucinations for 3 books, rising as high as a SHI of 0.87 (in the range 0-1, where 1 is the worst). The strong negative correlation of accuracy and SHI, given by r, demonstrates the fidelity of the new hallucination metric, which may generalize to other tasks. We also show that prediction accuracies correlate positively with the frequencies of Wikipedia instances of the book titles instead of the downloads and we perform error analyses of predictions. We publicly release the annotated chunks of data and our codes to aid the reproducibility and evaluation of other models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21773v2">MAC-Tuning: LLM Multi-Compositional Problem Reasoning with Enhanced Knowledge Boundary Awareness</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      With the widespread application of large language models (LLMs), the issue of generating non-existing facts, known as hallucination, has garnered increasing attention. Previous research in enhancing LLM confidence estimation mainly focuses on the single problem setting. However, LLM awareness of its internal parameterized knowledge boundary under the more challenging multi-problem setting, which requires answering multiple problems accurately simultaneously, remains underexplored. To bridge this gap, we introduce a novel method, Multiple Answers and Confidence Stepwise Tuning (MAC-Tuning), that separates the learning of answer prediction and confidence estimation during fine-tuning on instruction data. Extensive experiments demonstrate that our method outperforms baselines by up to 25% in average precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.08898v3">SEALGuard: Safeguarding the Multilingual Conversations in Southeast Asian Languages for LLM Software Systems</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      Safety alignment is critical for LLM-powered systems. While recent LLM-powered guardrail approaches such as LlamaGuard achieve high detection accuracy of unsafe inputs written in English (e.g., ``How to create a bomb?''), they struggle with multilingual unsafe inputs. This limitation leaves LLM systems vulnerable to unsafe and jailbreak prompts written in low-resource languages such as those in Southeast Asia. This paper introduces SEALGuard, a multilingual guardrail designed to improve the safety alignment across diverse languages. It aims to address the multilingual safety alignment gap of existing guardrails and ensure effective filtering of unsafe and jailbreak prompts in LLM-powered systems. We adapt a general-purpose multilingual language model into a multilingual guardrail using low-rank adaptation (LoRA). We construct SEALSBench, a large-scale multilingual safety alignment dataset containing over 260,000 prompts in ten languages, including safe, unsafe, and jailbreak cases. We evaluate SEALGuard against state-of-the-art guardrails such as LlamaGuard on this benchmark. Our findings show that multilingual unsafe and jailbreak prompts substantially degrade the performance of the state-of-the-art LlamaGuard, which experiences a drop in Defense Success Rate (DSR) by 9% and 18%, respectively, compared to its performance on English-only prompts. In contrast, SEALGuard outperforms existing guardrails in detecting multilingual unsafe and jailbreak prompts, improving DSR by 48% over LlamaGuard and achieving the best DSR, precision, and F1-score. Our ablation study further reveals the contributions of adaptation strategies and model size to the overall performance of SEALGuard. We release our pre-trained model and benchmark at https://github.com/awsm-research/SEALGuard to support further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.09617v2">LLM-Enhanced User-Item Interactions: Leveraging Edge Information for Optimized Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      Graph recommendation methods, representing a connected interaction perspective, reformulate user-item interactions as graphs to leverage graph structure and topology to recommend and have proved practical effectiveness at scale. Large language models, representing a textual generative perspective, excel at modeling user languages, understanding behavioral contexts, capturing user-item semantic relationships, analyzing textual sentiments, and generating coherent and contextually relevant texts as recommendations. However, there is a gap between the connected graph perspective and the text generation perspective as the task formulations are different. A research question arises: how can we effectively integrate the two perspectives for more personalized recsys? To fill this gap, we propose to incorporate graph-edge information into LLMs via prompt and attention innovations. We reformulate recommendations as a probabilistic generative problem using prompts. We develop a framework to incorporate graph edge information from the prompt and attention mechanisms for graph-structured LLM recommendations. We develop a new prompt design that brings in both first-order and second-order graph relationships; we devise an improved LLM attention mechanism to embed direct the spatial and connectivity information of edges. Our evaluation of real-world datasets demonstrates the framework's ability to understand connectivity information in graph data and to improve the relevance and quality of recommendation results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12820v1">Emotional Support with LLM-based Empathetic Dialogue Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      Emotional Support Conversation (ESC) aims to provide empathetic and effective emotional assistance through dialogue, addressing the growing demand for mental health support. This paper presents our solution for the NLPCC 2025 Task 8 ESC evaluation, where we leverage large-scale language models enhanced by prompt engineering and finetuning techniques. We explore both parameter-efficient Low-Rank Adaptation and full-parameter fine-tuning strategies to improve the model's ability to generate supportive and contextually appropriate responses. Our best model ranked second in the competition, highlighting the potential of combining LLMs with effective adaptation methods for ESC tasks. Future work will focus on further enhancing emotional understanding and response personalization to build more practical and reliable emotional support systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03106v4">Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 52 pages, updated with new experimental results and implementation details
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language models (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of spontaneous self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided self-refinements simultaneously while maintaining exploration. Additionally, we employ a shaping function to amplify learning from correct, especially unfamiliar, refinements and penalize incorrect ones. Extensive experiments with Qwen2.5-7B-Base, Qwen2.5-Math-7B-Base, and Qwen3-8B demonstrate that Critique-GRPO consistently outperforms supervised learning and RL-based fine-tuning methods across eight challenging mathematical, STEM, and general reasoning tasks, improving average pass@1 scores by approximately 4.4% and 3.8% on Qwen2.5-7B-Base and Qwen3-8B, respectively. Notably, Critique-GRPO enables effective self-improvement through self-critiquing and weak-to-strong generalization, achieving consistent gains over GRPO, such as 16.7% and 10.0% pass@1 improvements on AIME 2024, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12347v2">Synthesizing Privacy-Preserving Text Data via Finetuning without Finetuning Billion-Scale LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 Code available at https://github.com/tanyuqian/synthetic-private-data
    </div>
    <details class="paper-abstract">
      Synthetic data offers a promising path to train models while preserving data privacy. Differentially private (DP) finetuning of large language models (LLMs) as data generator is effective, but is impractical when computation resources are limited. Meanwhile, prompt-based methods such as private evolution depend heavily on the manual prompts, and ineffectively use private information in their iterative data selection process. To overcome these limitations, we propose CTCL (Data Synthesis with ConTrollability and CLustering), a novel framework for generating privacy-preserving synthetic data without extensive prompt engineering or billion-scale LLM finetuning. CTCL pretrains a lightweight 140M conditional generator and a clustering-based topic model on large-scale public data. To further adapt to the private domain, the generator is DP finetuned on private data for fine-grained textual information, while the topic model extracts a DP histogram representing distributional information. The DP generator then samples according to the DP histogram to synthesize a desired number of data examples. Evaluation across five diverse domains demonstrates the effectiveness of our framework, particularly in the strong privacy regime. Systematic ablation validates the design of each framework component and highlights the scalability of our approach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.02946v6">Scaling Trends for Data Poisoning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 This arXiv version of the paper originally included an initial investigation of jailbreak-tuning, which can produce 60+ percentage point increases in vulnerability elicitation compared with standard data poisoning. Jailbreak-tuning has now been separated into a full independent paper, which can be found at arXiv:2507.11630
    </div>
    <details class="paper-abstract">
      LLMs produce harmful and undesirable behavior when trained on datasets containing even a small fraction of poisoned data. We demonstrate that GPT models remain vulnerable to fine-tuning on poisoned data, even when safeguarded by moderation systems. Given the persistence of data poisoning vulnerabilities in today's most capable models, this paper investigates whether these risks increase with model scaling. We evaluate three threat models -- malicious fine-tuning, imperfect data curation, and intentional data contamination -- across 24 frontier LLMs ranging from 1.5 to 72 billion parameters. Our experiments reveal that larger LLMs are significantly more susceptible to data poisoning, learning harmful behaviors from even minimal exposure to harmful data more quickly than smaller models. These findings underscore the need for leading AI companies to thoroughly red team fine-tuning APIs before public release and to develop more robust safeguards against data poisoning, particularly as models continue to scale in size and capability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13577v1">LLM-Based Community Surveys for Operational Decision Making in Interconnected Utility Infrastructures</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      We represent interdependent infrastructure systems and communities alike with a hetero-functional graph (HFG) that encodes the dependencies between functionalities. This graph naturally imposes a partial order of functionalities that can inform the sequence of repair decisions to be made during a disaster across affected communities. However, using such technical criteria alone provides limited guidance at the point where the functionalities directly impact the communities, since these can be repaired in any order without violating the system constraints. To address this gap and improve resilience, we integrate community preferences to refine this partial order from the HFG into a total order. Our strategy involves getting the communities' opinions on their preferred sequence for repair crews to address infrastructure issues, considering potential constraints on resources. Due to the delay and cost associated with real-world survey data, we utilize a Large Language Model (LLM) as a proxy survey tool. We use the LLM to craft distinct personas representing individuals, each with varied disaster experiences. We construct diverse disaster scenarios, and each simulated persona provides input on prioritizing infrastructure repair needs across various communities. Finally, we apply learning algorithms to generate a global order based on the aggregated responses from these LLM-generated personas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13555v1">Demystifying Feature Requests: Leveraging LLMs to Refine Feature Requests in Open-Source Software</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 Accepted at the 33rd IEEE International Requirements Engineering 2025
    </div>
    <details class="paper-abstract">
      The growing popularity and widespread use of software applications (apps) across various domains have driven rapid industry growth. Along with this growth, fast-paced market changes have led to constantly evolving software requirements. Such requirements are often grounded in feature requests and enhancement suggestions, typically provided by users in natural language (NL). However, these requests often suffer from defects such as ambiguity and incompleteness, making them challenging to interpret. Traditional validation methods (e.g., interviews and workshops) help clarify such defects but are impractical in decentralized environments like open-source software (OSS), where change requests originate from diverse users on platforms like GitHub. This paper proposes a novel approach leveraging Large Language Models (LLMs) to detect and refine NL defects in feature requests. Our approach automates the identification of ambiguous and incomplete requests and generates clarification questions (CQs) to enhance their usefulness for developers. To evaluate its effectiveness, we apply our method to real-world OSS feature requests and compare its performance against human annotations. In addition, we conduct interviews with GitHub developers to gain deeper insights into their perceptions of NL defects, the strategies they use to address these defects, and the impact of defects on downstream software engineering (SE) tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.05630v2">How Not to Detect Prompt Injections with an LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      LLM-integrated applications and agents are vulnerable to prompt injection attacks, in which adversaries embed malicious instructions within seemingly benign user inputs to manipulate the LLM's intended behavior. Recent defenses based on $\textit{known-answer detection}$ (KAD) have achieved near-perfect performance by using an LLM to classify inputs as clean or contaminated. In this work, we formally characterize the KAD framework and uncover a structural vulnerability in its design that invalidates its core security premise. We design a methodical adaptive attack, $\textit{DataFlip}$, to exploit this fundamental weakness. It consistently evades KAD defenses with detection rates as low as $1.5\%$ while reliably inducing malicious behavior with success rates of up to $88\%$, without needing white-box access to the LLM or any optimization procedures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13525v1">Revisiting Prompt Engineering: A Comprehensive Evaluation for LLM-based Personalized Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 Accepted to ACM RecSys2025 reproducibility
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can perform recommendation tasks by taking prompts written in natural language as input. Compared to traditional methods such as collaborative filtering, LLM-based recommendation offers advantages in handling cold-start, cross-domain, and zero-shot scenarios, as well as supporting flexible input formats and generating explanations of user behavior. In this paper, we focus on a single-user setting, where no information from other users is used. This setting is practical for privacy-sensitive or data-limited applications. In such cases, prompt engineering becomes especially important for controlling the output generated by the LLM. We conduct a large-scale comparison of 23 prompt types across 8 public datasets and 12 LLMs. We use statistical tests and linear mixed-effects models to evaluate both accuracy and inference cost. Our results show that for cost-efficient LLMs, three types of prompts are especially effective: those that rephrase instructions, consider background knowledge, and make the reasoning process easier to follow. For high-performance LLMs, simple prompts often outperform more complex ones while reducing cost. In contrast, commonly used prompting styles in natural language processing, such as step-by-step reasoning, or the use of reasoning models often lead to lower accuracy. Based on these findings, we provide practical suggestions for selecting prompts and LLMs depending on the required balance between accuracy and cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13490v1">Revisiting LLM Value Probing Strategies: Are They Robust and Expressive?</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      There has been extensive research on assessing the value orientation of Large Language Models (LLMs) as it can shape user experiences across demographic groups. However, several challenges remain. First, while the Multiple Choice Question (MCQ) setting has been shown to be vulnerable to perturbations, there is no systematic comparison of probing methods for value probing. Second, it is unclear to what extent the probed values capture in-context information and reflect models' preferences for real-world actions. In this paper, we evaluate the robustness and expressiveness of value representations across three widely used probing strategies. We use variations in prompts and options, showing that all methods exhibit large variances under input perturbations. We also introduce two tasks studying whether the values are responsive to demographic context, and how well they align with the models' behaviors in value-related scenarios. We show that the demographic context has little effect on the free-text generation, and the models' values only weakly correlate with their preference for value-based actions. Our work highlights the need for a more careful examination of LLM value probing and awareness of its limitations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.13474v1">Paper Summary Attack: Jailbreaking LLMs through LLM Safety Papers</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      The safety of large language models (LLMs) has garnered significant research attention. In this paper, we argue that previous empirical studies demonstrate LLMs exhibit a propensity to trust information from authoritative sources, such as academic papers, implying new possible vulnerabilities. To verify this possibility, a preliminary analysis is designed to illustrate our two findings. Based on this insight, a novel jailbreaking method, Paper Summary Attack (\llmname{PSA}), is proposed. It systematically synthesizes content from either attack-focused or defense-focused LLM safety paper to construct an adversarial prompt template, while strategically infilling harmful query as adversarial payloads within predefined subsections. Extensive experiments show significant vulnerabilities not only in base LLMs, but also in state-of-the-art reasoning model like Deepseek-R1. PSA achieves a 97\% attack success rate (ASR) on well-aligned models like Claude3.5-Sonnet and an even higher 98\% ASR on Deepseek-R1. More intriguingly, our work has further revealed diametrically opposed vulnerability bias across different base models, and even between different versions of the same model, when exposed to either attack-focused or defense-focused papers. This phenomenon potentially indicates future research clues for both adversarial methodologies and safety alignment.Code is available at https://github.com/233liang/Paper-Summary-Attack
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14240v1">HuggingGraph: Understanding the Supply Chain of LLM Ecosystem</a></div>
    <div class="paper-meta">
      📅 2025-07-17
      | 💬 10 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) leverage deep learning to process and predict sequences of words from context, enabling them to perform various NLP tasks, such as translation, summarization, question answering, and content generation. However, the growing size and complexity of developing, training, and deploying advanced LLMs require extensive computational resources and large datasets. This creates a barrier for users. As a result, platforms that host models and datasets are widely used. For example, Hugging Face, one of the most popular platforms, hosted 1.8 million models and 450K datasets by June 2025, with no sign of slowing down. Since many LLMs are built from base models, pre-trained models, and external datasets, they can inherit vulnerabilities, biases, or malicious components from earlier models or datasets. Therefore, it is critical to understand the origin and development of these components to better detect potential risks, improve model fairness, and ensure compliance. Motivated by this, our project aims to study the relationships between models and datasets, which are core components of the LLM supply chain. First, we design a method to systematically collect LLM supply chain data. Using this data, we build a directed heterogeneous graph to model the relationships between models and datasets, resulting in a structure with 397,376 nodes and 453,469 edges. We then perform various analyses and uncover several findings, such as: (i) the LLM supply chain graph is large, sparse, and follows a power-law degree distribution; (ii) it features a densely connected core and a fragmented periphery; (iii) datasets play pivotal roles in training; (iv) strong interdependence exists between models and datasets; and (v) the graph is dynamic, with daily updates reflecting the ecosystem's ongoing evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.15874v1">Why Braking? Scenario Extraction and Reasoning Utilizing LLM</a></div>
    <div class="paper-meta">
      📅 2025-07-17
    </div>
    <details class="paper-abstract">
      The growing number of ADAS-equipped vehicles has led to a dramatic increase in driving data, yet most of them capture routine driving behavior. Identifying and understanding safety-critical corner cases within this vast dataset remains a significant challenge. Braking events are particularly indicative of potentially hazardous situations, motivating the central question of our research: Why does a vehicle brake? Existing approaches primarily rely on rule-based heuristics to retrieve target scenarios using predefined condition filters. While effective in simple environments such as highways, these methods lack generalization in complex urban settings. In this paper, we propose a novel framework that leverages Large Language Model (LLM) for scenario understanding and reasoning. Our method bridges the gap between low-level numerical signals and natural language descriptions, enabling LLM to interpret and classify driving scenarios. We propose a dual-path scenario retrieval that supports both category-based search for known scenarios and embedding-based retrieval for unknown Out-of-Distribution (OOD) scenarios. To facilitate evaluation, we curate scenario annotations on the Argoverse 2 Sensor Dataset. Experimental results show that our method outperforms rule-based baselines and generalizes well to OOD scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12143v1">Overview of the Sensemaking Task at the ELOQUENT 2025 Lab: LLMs as Teachers, Students and Evaluators</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 30 pages, 7 figures, CLEF 2025 Conference and Labs of the Evaluation Forum
    </div>
    <details class="paper-abstract">
      ELOQUENT is a set of shared tasks that aims to create easily testable high-level criteria for evaluating generative language models. Sensemaking is one such shared task. In Sensemaking, we try to assess how well generative models ``make sense out of a given text'' in three steps inspired by exams in a classroom setting: (1) Teacher systems should prepare a set of questions, (2) Student systems should answer these questions, and (3) Evaluator systems should score these answers, all adhering rather strictly to a given set of input materials. We report on the 2025 edition of Sensemaking, where we had 7 sources of test materials (fact-checking analyses of statements, textbooks, transcribed recordings of a lecture, and educational videos) spanning English, German, Ukrainian, and Czech languages. This year, 4 teams participated, providing us with 2 Teacher submissions, 2 Student submissions, and 2 Evaluator submissions. We added baselines for Teacher and Student using commercial large language model systems. We devised a fully automatic evaluation procedure, which we compare to a minimalistic manual evaluation. We were able to make some interesting observations. For the first task, the creation of questions, better evaluation strategies will still have to be devised because it is difficult to discern the quality of the various candidate question sets. In the second task, question answering, the LLMs examined overall perform acceptably, but restricting their answers to the given input texts remains problematic. In the third task, evaluation of question answers, our adversarial tests reveal that systems using the LLM-as-a-Judge paradigm erroneously rate both garbled question-answer pairs and answers to mixed-up questions as acceptable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.00513v3">Leveraging LLMs for User Stories in AI Systems: UStAI Dataset</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      AI systems are gaining widespread adoption across various sectors and domains. Creating high-quality AI system requirements is crucial for aligning the AI system with business goals and consumer values and for social responsibility. However, with the uncertain nature of AI systems and the heavy reliance on sensitive data, more research is needed to address the elicitation and analysis of AI systems requirements. With the proprietary nature of many AI systems, there is a lack of open-source requirements artifacts and technical requirements documents for AI systems, limiting broader research and investigation. With Large Language Models (LLMs) emerging as a promising alternative to human-generated text, this paper investigates the potential use of LLMs to generate user stories for AI systems based on abstracts from scholarly papers. We conducted an empirical evaluation using three LLMs and generated $1260$ user stories from $42$ abstracts from $26$ domains. We assess their quality using the Quality User Story (QUS) framework. Moreover, we identify relevant non-functional requirements (NFRs) and ethical principles. Our analysis demonstrates that the investigated LLMs can generate user stories inspired by the needs of various stakeholders, offering a promising approach for generating user stories for research purposes and for aiding in the early requirements elicitation phase of AI systems. We have compiled and curated a collection of stories generated by various LLMs into a dataset (UStAI), which is now publicly available for use.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12104v1">From Static to Intelligent: Evolving SaaS Pricing with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 12 pages. Accepted at the SOC4AI Workshop (Service-Oriented Computing for AI Applications), held in conjunction with the 22nd International Conference on Service-Oriented Computing (ICSOC 2024)
    </div>
    <details class="paper-abstract">
      The SaaS paradigm has revolutionized software distribution by offering flexible pricing options to meet diverse customer needs. However, the rapid expansion of the SaaS market has introduced significant complexity for DevOps teams, who must manually manage and evolve pricing structures, an approach that is both time-consuming and prone to errors. The absence of automated tools for pricing analysis restricts the ability to efficiently evaluate, optimize, and scale these models. This paper proposes leveraging intelligent pricing (iPricing), dynamic, machine-readable pricing models, as a solution to these challenges. Intelligent pricing enables competitive analysis, streamlines operational decision-making, and supports continuous pricing evolution in response to market dynamics, leading to improved efficiency and accuracy. We present an LLM-driven approach that automates the transformation of static HTML pricing into iPricing, significantly improving efficiency and consistency while minimizing human error. Our implementation, AI4Pricing2Yaml, features a basic Information Extractor that uses web scraping and LLMs technologies to extract essential pricing components, plans, features, usage limits, and add-ons, from SaaS websites. Validation against a dataset of 30 distinct commercial SaaS, encompassing over 150 intelligent pricings, demonstrates the system's effectiveness in extracting the desired elements across all steps. However, challenges remain in addressing hallucinations, complex structures, and dynamic content. This work highlights the potential of automating intelligent pricing transformation to streamline SaaS pricing management, offering implications for improved consistency and scalability in an increasingly intricate pricing landscape. Future research will focus on refining extraction capabilities and enhancing the system's adaptability to a wider range of SaaS websites.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12084v1">LLAMA: Multi-Feedback Smart Contract Fuzzing Framework with LLM-Guided Seed Generation</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Smart contracts play a pivotal role in blockchain ecosystems, and fuzzing remains an important approach to securing smart contracts. Even though mutation scheduling is a key factor influencing fuzzing effectiveness, existing fuzzers have primarily explored seed scheduling and generation, while mutation scheduling has been rarely addressed by prior work. In this work, we propose a Large Language Models (LLMs)-based Multi-feedback Smart Contract Fuzzing framework (LLAMA) that integrates LLMs, evolutionary mutation strategies, and hybrid testing techniques. Key components of the proposed LLAMA include: (i) a hierarchical prompting strategy that guides LLMs to generate semantically valid initial seeds, coupled with a lightweight pre-fuzzing phase to select high-potential inputs; (ii) a multi-feedback optimization mechanism that simultaneously improves seed generation, seed selection, and mutation scheduling by leveraging runtime coverage and dependency feedback; and (iii) an evolutionary fuzzing engine that dynamically adjusts mutation operator probabilities based on effectiveness, while incorporating symbolic execution to escape stagnation and uncover deeper vulnerabilities. Our experiments demonstrate that LLAMA outperforms state-of-the-art fuzzers in both coverage and vulnerability detection. Specifically, it achieves 91% instruction coverage and 90% branch coverage, while detecting 132 out of 148 known vulnerabilities across diverse categories. These results highlight LLAMA's effectiveness, adaptability, and practicality in real-world smart contract security testing scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.12079v1">Findings of MEGA: Maths Explanation with LLMs using the Socratic Method for Active Learning</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 This paper was accepted for the special issue AI for Education by the IEEE Signal Processing Magazine journal
    </div>
    <details class="paper-abstract">
      This paper presents an intervention study on the effects of the combined methods of (1) the Socratic method, (2) Chain of Thought (CoT) reasoning, (3) simplified gamification and (4) formative feedback on university students' Maths learning driven by large language models (LLMs). We call our approach Mathematics Explanations through Games by AI LLMs (MEGA). Some students struggle with Maths and as a result avoid Math-related discipline or subjects despite the importance of Maths across many fields, including signal processing. Oftentimes, students' Maths difficulties stem from suboptimal pedagogy. We compared the MEGA method to the traditional step-by-step (CoT) method to ascertain which is better by using a within-group design after randomly assigning questions for the participants, who are university students. Samples (n=60) were randomly drawn from each of the two test sets of the Grade School Math 8K (GSM8K) and Mathematics Aptitude Test of Heuristics (MATH) datasets, based on the error margin of 11%, the confidence level of 90%, and a manageable number of samples for the student evaluators. These samples were used to evaluate two capable LLMs at length (Generative Pretrained Transformer 4o (GPT4o) and Claude 3.5 Sonnet) out of the initial six that were tested for capability. The results showed that students agree in more instances that the MEGA method is experienced as better for learning for both datasets. It is even much better than the CoT (47.5% compared to 26.67%) in the more difficult MATH dataset, indicating that MEGA is better at explaining difficult Maths problems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.10024v2">Qualitative Study for LLM-assisted Design Study Process: Strategies, Challenges, and Roles</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Design studies aim to create visualization solutions for real-world problems of different application domains. Recently, the emergence of large language models (LLMs) has introduced new opportunities to enhance the design study process, providing capabilities such as creative problem-solving, data handling, and insightful analysis. However, despite their growing popularity, there remains a lack of systematic understanding of how LLMs can effectively assist researchers in visualization-specific design studies. In this paper, we conducted a multi-stage qualitative study to fill this gap, involving 30 design study researchers from diverse backgrounds and expertise levels. Through in-depth interviews and carefully-designed questionnaires, we investigated strategies for utilizing LLMs, the challenges encountered, and the practices used to overcome them. We further compiled and summarized the roles that LLMs can play across different stages of the design study process. Our findings highlight practical implications to inform visualization practitioners, and provide a framework for leveraging LLMs to enhance the design study process in visualization research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11997v1">Can LLMs Find Fraudsters? Multi-level LLM Enhanced Graph Fraud Detection</a></div>
    <div class="paper-meta">
      📅 2025-07-16
    </div>
    <details class="paper-abstract">
      Graph fraud detection has garnered significant attention as Graph Neural Networks (GNNs) have proven effective in modeling complex relationships within multimodal data. However, existing graph fraud detection methods typically use preprocessed node embeddings and predefined graph structures to reveal fraudsters, which ignore the rich semantic cues contained in raw textual information. Although Large Language Models (LLMs) exhibit powerful capabilities in processing textual information, it remains a significant challenge to perform multimodal fusion of processed textual embeddings with graph structures. In this paper, we propose a \textbf{M}ulti-level \textbf{L}LM \textbf{E}nhanced Graph Fraud \textbf{D}etection framework called MLED. In MLED, we utilize LLMs to extract external knowledge from textual information to enhance graph fraud detection methods. To integrate LLMs with graph structure information and enhance the ability to distinguish fraudsters, we design a multi-level LLM enhanced framework including type-level enhancer and relation-level enhancer. One is to enhance the difference between the fraudsters and the benign entities, the other is to enhance the importance of the fraudsters in different relations. The experiments on four real-world datasets show that MLED achieves state-of-the-art performance in graph fraud detection as a generalized framework that can be applied to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.11981v1">Simplifications are Absolutists: How Simplified Language Reduces Word Sense Awareness in LLM-Generated Definitions</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 Accepted by RANLP 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) can provide accurate word definitions and explanations for any context. However, the scope of the definition changes for different target groups, like children or language learners. This is especially relevant for homonyms, words with multiple meanings, where oversimplification might risk information loss by omitting key senses, potentially misleading users who trust LLM outputs. We investigate how simplification impacts homonym definition quality across three target groups: Normal, Simple, and ELI5. Using two novel evaluation datasets spanning multiple languages, we test DeepSeek v3, Llama 4 Maverick, Qwen3-30B A3B, GPT-4o mini, and Llama 3.1 8B via LLM-as-Judge and human annotations. Our results show that simplification drastically degrades definition completeness by neglecting polysemy, increasing the risk of misunderstanding. Fine-tuning Llama 3.1 8B with Direct Preference Optimization substantially improves homonym response quality across all prompt types. These findings highlight the need to balance simplicity and completeness in educational NLP to ensure reliable, context-aware definitions for all learners.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13789v3">LEADRE: Multi-Faceted Knowledge Enhanced LLM Empowered Display Advertisement Recommender System</a></div>
    <div class="paper-meta">
      📅 2025-07-16
      | 💬 Accepted by VLDB 2025 Industrial Track
    </div>
    <details class="paper-abstract">
      Display advertising provides significant value to advertisers, publishers, and users. Traditional display advertising systems utilize a multi-stage architecture consisting of retrieval, coarse ranking, and final ranking. However, conventional retrieval methods rely on ID-based learning to rank mechanisms and fail to adequately utilize the content information of ads, which hampers their ability to provide diverse recommendation lists. To address this limitation, we propose leveraging the extensive world knowledge of LLMs. However, three key challenges arise when attempting to maximize the effectiveness of LLMs: "How to capture user interests", "How to bridge the knowledge gap between LLMs and advertising system", and "How to efficiently deploy LLMs". To overcome these challenges, we introduce a novel LLM-based framework called LLM Empowered Display ADvertisement REcommender system (LEADRE). LEADRE consists of three core modules: (1) The Intent-Aware Prompt Engineering introduces multi-faceted knowledge and designs intent-aware <Prompt, Response> pairs that fine-tune LLMs to generate ads tailored to users' personal interests. (2) The Advertising-Specific Knowledge Alignment incorporates auxiliary fine-tuning tasks and Direct Preference Optimization (DPO) to align LLMs with ad semantic and business value. (3) The Efficient System Deployment deploys LEADRE in an online environment by integrating both latency-tolerant and latency-sensitive service. Extensive offline experiments demonstrate the effectiveness of LEADRE and validate the contributions of individual modules. Online A/B test shows that LEADRE leads to a 1.57% and 1.17% GMV lift for serviced users on WeChat Channels and Moments separately. LEADRE has been deployed on both platforms, serving tens of billions of requests each day.
    </details>
</div>
