# llm - 2025_06

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
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
- Part 12
- [Part 13](papers_13.md)
- [Part 14](papers_14.md)
- [Part 15](papers_15.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04989v1">BacPrep: An Experimental Platform for Evaluating LLM-Based Bacalaureat Assessment</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 9 pages Preprint ACCEPTED at BBGI (ITS Workshop)
    </div>
    <details class="paper-abstract">
      Accessing quality preparation and feedback for the Romanian Bacalaureat exam is challenging, particularly for students in remote or underserved areas. This paper introduces BacPrep, an experimental online platform exploring Large Language Model (LLM) potential for automated assessment, aiming to offer a free, accessible resource. Using official exam questions from the last 5 years, BacPrep employs one of Google's newest models, Gemini 2.0 Flash (released Feb 2025), guided by official grading schemes, to provide experimental feedback. Currently operational, its primary research function is collecting student solutions and LLM outputs. This focused dataset is vital for planned expert validation to rigorously evaluate the feasibility and accuracy of this cutting-edge LLM in the specific Bacalaureat context before reliable deployment. We detail the design, data strategy, status, validation plan, and ethics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03785v2">Knockout LLM Assessment: Using Large Language Models for Evaluations through Iterative Pairwise Comparisons</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Accepted to GEM @ ACL 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown to be effective evaluators across various domains such as machine translations or the scientific domain. Current LLM-as-a-Judge approaches rely mostly on individual assessments or a single round of pairwise assessments, preventing the judge LLM from developing a global ranking perspective. To address this, we present Knockout Assessment, an LLM-asa Judge method using a knockout tournament system with iterative pairwise comparisons. Experiments across three LLMs on two datasets show that knockout assessment improves scoring accuracy, increasing Pearson correlation with expert evaluations by 0.07 on average for university-level exam scoring and machine translation evaluations, aligning LLM assessments more closely with human scoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04985v1">FPTQuant: Function-Preserving Transforms for LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) require substantial compute, and thus energy, at inference time. While quantizing weights and activations is effective at improving efficiency, naive quantization of LLMs can significantly degrade performance due to large magnitude outliers. This paper describes FPTQuant, which introduces four novel, lightweight, and expressive function-preserving transforms (FPTs) to facilitate quantization of transformers: (1) a mergeable pre-RoPE transform for queries and keys, (2) a mergeable transform for values, (3) a mergeable scaling transform within the MLP block, and (4) a cheap, dynamic scaling transform. By leveraging the equivariances and independencies inherent to canonical transformer operation, we designed these FPTs to maintain the model's function while shaping the intermediate activation distributions to be more quantization friendly. FPTQuant requires no custom kernels and adds virtually no overhead during inference. The FPTs are trained both locally to reduce outliers, and end-to-end such that the outputs of the quantized and full-precision models match. FPTQuant enables static INT4 quantization with minimal overhead and shows SOTA speed-up of up to 3.9 times over FP. Empirically, FPTQuant has an excellent accuracy-speed trade-off -- it is performing on par or exceeding most prior work and only shows slightly lower accuracy compared to a method that is up to 29% slower.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02672v2">EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 47 pages, 24 figures
    </div>
    <details class="paper-abstract">
      We introduce EvaLearn, a pioneering benchmark designed to evaluate large language models (LLMs) on their learning capability and efficiency in challenging tasks, a critical, yet underexplored aspect of model potential. EvaLearn contains 648 challenging problems across six task types, grouped into 182 sequences, each sequence dedicated to one task type. Diverging from most existing benchmarks that evaluate models in parallel, EvaLearn requires models to solve problems sequentially, allowing them to leverage the experience gained from previous solutions. EvaLearn provides five comprehensive automated metrics to evaluate models and quantify their learning capability and efficiency. We extensively benchmark nine frontier models and observe varied performance profiles: some models, such as Claude-3.7-sonnet, start with moderate initial performance but exhibit strong learning ability, while some models struggle to benefit from experience and may even show negative transfer. Moreover, we investigate model performance under two learning settings and find that instance-level rubrics and teacher-model feedback further facilitate model learning. Importantly, we observe that current LLMs with stronger static abilities do not show a clear advantage in learning capability across all tasks, highlighting that EvaLearn evaluates a new dimension of model performance. We hope EvaLearn provides a novel evaluation perspective for assessing LLM potential and understanding the gap between models and human capabilities, promoting the development of deeper and more dynamic evaluation approaches. All datasets, the automatic evaluation framework, and the results studied in this paper are available at the GitHub repository.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04965v1">From Struggle (06-2024) to Mastery (02-2025) LLMs Conquer Advanced Algorithm Exams and Pave the Way for Editorial Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 15 pages Pre-print Paper accepted to ITS 2025
    </div>
    <details class="paper-abstract">
      This paper presents a comprehensive evaluation of the performance of state-of-the-art Large Language Models (LLMs) on challenging university-level algorithms exams. By testing multiple models on both a Romanian exam and its high-quality English translation, we analyze LLMs' problem-solving capabilities, consistency, and multilingual performance. Our empirical study reveals that the most recent models not only achieve scores comparable to top-performing students but also demonstrate robust reasoning skills on complex, multi-step algorithmic challenges, even though difficulties remain with graph-based tasks. Building on these findings, we explore the potential of LLMs to support educational environments through the generation of high-quality editorial content, offering instructors a powerful tool to enhance student feedback. The insights and best practices discussed herein pave the way for further integration of generative AI in advanced algorithm education.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17934v2">Toward a Human-Centered Evaluation Framework for Trustworthy LLM-Powered GUI Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has revolutionized Graphical User Interface (GUI) automation through LLM-powered GUI agents, yet their ability to process sensitive data with limited human oversight raises significant privacy and security risks. This position paper identifies three key risks of GUI agents and examines how they differ from traditional GUI automation and general autonomous agents. Despite these risks, existing evaluations focus primarily on performance, leaving privacy and security assessments largely unexplored. We review current evaluation metrics for both GUI and general LLM agents and outline five key challenges in integrating human evaluators for GUI agent assessments. To address these gaps, we advocate for a human-centered evaluation framework that incorporates risk assessments, enhances user awareness through in-context consent, and embeds privacy and security considerations into GUI agent design and evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07323v2">Navigating Motion Agents in Dynamic and Cluttered Environments through LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      This paper advances motion agents empowered by large language models (LLMs) toward autonomous navigation in dynamic and cluttered environments, significantly surpassing first and recent seminal but limited studies on LLM's spatial reasoning, where movements are restricted in four directions in simple, static environments in the presence of only single agents much less multiple agents. Specifically, we investigate LLMs as spatial reasoners to overcome these limitations by uniformly encoding environments (e.g., real indoor floorplans), agents which can be dynamic obstacles and their paths as discrete tokens akin to language tokens. Our training-free framework supports multi-agent coordination, closed-loop replanning, and dynamic obstacle avoidance without retraining or fine-tuning. We show that LLMs can generalize across agents, tasks, and environments using only text-based interactions, opening new possibilities for semantically grounded, interactive navigation in both simulation and embodied systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04920v1">Simulating LLM-to-LLM Tutoring for Multilingual Math Feedback</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Preprint, in submission
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated the ability to generate formative feedback and instructional hints in English, making them increasingly relevant for AI-assisted education. However, their ability to provide effective instructional support across different languages, especially for mathematically grounded reasoning tasks, remains largely unexamined. In this work, we present the first large-scale simulation of multilingual tutor-student interactions using LLMs. A stronger model plays the role of the tutor, generating feedback in the form of hints, while a weaker model simulates the student. We explore 352 experimental settings across 11 typologically diverse languages, four state-of-the-art LLMs, and multiple prompting strategies to assess whether language-specific feedback leads to measurable learning gains. Our study examines how student input language, teacher feedback language, model choice, and language resource level jointly influence performance. Results show that multilingual hints can significantly improve learning outcomes, particularly in low-resource languages when feedback is aligned with the student's native language. These findings offer practical insights for developing multilingual, LLM-based educational tools that are both effective and inclusive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04909v1">When Thinking LLMs Lie: Unveiling the Strategic Deception in Representations of Reasoning Models</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      The honesty of large language models (LLMs) is a critical alignment challenge, especially as advanced systems with chain-of-thought (CoT) reasoning may strategically deceive humans. Unlike traditional honesty issues on LLMs, which could be possibly explained as some kind of hallucination, those models' explicit thought paths enable us to study strategic deception--goal-driven, intentional misinformation where reasoning contradicts outputs. Using representation engineering, we systematically induce, detect, and control such deception in CoT-enabled LLMs, extracting "deception vectors" via Linear Artificial Tomography (LAT) for 89% detection accuracy. Through activation steering, we achieve a 40% success rate in eliciting context-appropriate deception without explicit prompts, unveiling the specific honesty-related issue of reasoning models and providing tools for trustworthy AI alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04907v1">Verbose ListOps (VLO): Beyond Long Context -- Unmasking LLM's Reasoning Blind Spots</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), whilst great at extracting facts from text, struggle with nested narrative reasoning. Existing long context and multi-hop QA benchmarks inadequately test this, lacking realistic distractors or failing to decouple context length from reasoning complexity, masking a fundamental LLM limitation. We introduce Verbose ListOps, a novel benchmark that programmatically transposes ListOps computations into lengthy, coherent stories. This uniquely forces internal computation and state management of nested reasoning problems by withholding intermediate results, and offers fine-grained controls for both narrative size \emph{and} reasoning difficulty. Whilst benchmarks like LongReason (2025) advance approaches for synthetically expanding the context size of multi-hop QA problems, Verbose ListOps pinpoints a specific LLM vulnerability: difficulty in state management for nested sub-reasoning amongst semantically-relevant, distracting narrative. Our experiments show that leading LLMs (e.g., OpenAI o4, Gemini 2.5 Pro) collapse in performance on Verbose ListOps at modest (~10k token) narrative lengths, despite effortlessly solving raw ListOps equations. Addressing this failure is paramount for real-world text interpretation which requires identifying key reasoning points, tracking conceptual intermediate results, and filtering irrelevant information. Verbose ListOps, and its extensible generation framework thus enables targeted reasoning enhancements beyond mere context-window expansion; a critical step to automating the world's knowledge work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04894v1">ICPC-Eval: Probing the Frontiers of LLM Reasoning with Competitive Programming Contests</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      With the significant progress of large reasoning models in complex coding and reasoning tasks, existing benchmarks, like LiveCodeBench and CodeElo, are insufficient to evaluate the coding capabilities of large language models (LLMs) in real competition environments. Moreover, current evaluation metrics such as Pass@K fail to capture the reflective abilities of reasoning models. To address these challenges, we propose \textbf{ICPC-Eval}, a top-level competitive coding benchmark designed to probing the frontiers of LLM reasoning. ICPC-Eval includes 118 carefully curated problems from 11 recent ICPC contests held in various regions of the world, offering three key contributions: 1) A challenging realistic ICPC competition scenario, featuring a problem type and difficulty distribution consistent with actual contests. 2) A robust test case generation method and a corresponding local evaluation toolkit, enabling efficient and accurate local evaluation. 3) An effective test-time scaling evaluation metric, Refine@K, which allows iterative repair of solutions based on execution feedback. The results underscore the significant challenge in evaluating complex reasoning abilities: top-tier reasoning models like DeepSeek-R1 often rely on multi-turn code feedback to fully unlock their in-context reasoning potential when compared to non-reasoning counterparts. Furthermore, despite recent advancements in code generation, these models still lag behind top-performing human teams. We release the benchmark at: https://github.com/RUCAIBox/Slow_Thinking_with_LLMs
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.07140v5">Can Graph Descriptive Order Affect Solving Graph Problems with LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Accepted to ACL 2025 main conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved significant success in reasoning tasks, including mathematical reasoning and logical deduction. Among these reasoning tasks, graph problems stand out due to their complexity and unique structural characteristics, attracting considerable attention from researchers. Previous studies have explored LLMs' graph reasoning abilities through various techniques, such as different encoding methods for graph structures and the use of carefully designed prompts. However, a critical factor has been mostly overlooked: the prompt sequential order in which graph descriptions are presented to the models. In this study, we present the first comprehensive analysis of how the order of graph descriptions impacts LLM performance. Specifically, we comprehensively evaluate four graph description orders across six graph problems using six mainstream LLMs. The results reveal that: (1) ordered graph descriptions significantly improve LLMs' comprehension of graph structures; (2) the robustness of LLMs to graph description order varies across different tasks; and (3) the impact of graph order on performance is closely related to the inherent characteristics of tasks. This study provides a critical advancement in the application of LLMs for solving graph-related problems, paving the way for future research to optimize model performance through strategic graph description ordering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.22251v2">Evaluation of LLMs in Speech is Often Flawed: Test Set Contamination in Large Language Models for Speech Recognition</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Recent work suggests that large language models (LLMs) can improve performance of speech tasks compared to existing systems. To support their claims, results on LibriSpeech and Common Voice are often quoted. However, this work finds that a substantial amount of the LibriSpeech and Common Voice evaluation sets appear in public LLM pretraining corpora. This calls into question the reliability of findings drawn from these two datasets. To measure contamination impact, LLMs trained with/without contamination are compared. A contaminated LLM is more likely to generate test sentences it has seen during training. Then, speech recognisers based on LLMs are compared. They show only subtle error rate differences if the LLM is contaminated, but assign significantly higher probabilities to transcriptions seen during LLM training. Results show that LLM outputs can be biased by tiny amounts of data contamination, highlighting the importance of evaluating LLM-based speech systems with held-out data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04867v1">LLMs for sensory-motor control: Combining in-context and iterative learning</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 24 pages (13 pages are from appendix), 6 figures, code for experiments replication and supplementary material provided at https://github.com/jtyska/llm-robotics-article/
    </div>
    <details class="paper-abstract">
      We propose a method that enables large language models (LLMs) to control embodied agents by directly mapping continuous observation vectors to continuous action vectors. Initially, the LLMs generate a control strategy based on a textual description of the agent, its environment, and the intended goal. This strategy is then iteratively refined through a learning process in which the LLMs are repeatedly prompted to improve the current strategy, using performance feedback and sensory-motor data collected during its evaluation. The method is validated on classic control tasks from the Gymnasium library and the inverted pendulum task from the MuJoCo library. In most cases, it successfully identifies optimal or high-performing solutions by integrating symbolic knowledge derived through reasoning with sub-symbolic sensory-motor data gathered as the agent interacts with its environment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04855v1">Prompting LLMs: Length Control for Isometric Machine Translation</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Accepted to IWSLT 2025
    </div>
    <details class="paper-abstract">
      In this study, we explore the effectiveness of isometric machine translation across multiple language pairs (En$\to$De, En$\to$Fr, and En$\to$Es) under the conditions of the IWSLT Isometric Shared Task 2022. Using eight open-source large language models (LLMs) of varying sizes, we investigate how different prompting strategies, varying numbers of few-shot examples, and demonstration selection influence translation quality and length control. We discover that the phrasing of instructions, when aligned with the properties of the provided demonstrations, plays a crucial role in controlling the output length. Our experiments show that LLMs tend to produce shorter translations only when presented with extreme examples, while isometric demonstrations often lead to the models disregarding length constraints. While few-shot prompting generally enhances translation quality, further improvements are marginal across 5, 10, and 20-shot settings. Finally, considering multiple outputs allows to notably improve overall tradeoff between the length and quality, yielding state-of-the-art performance for some language pairs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20354v2">Rethinking Text-based Protein Understanding: Retrieval or LLM?</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      In recent years, protein-text models have gained significant attention for their potential in protein generation and understanding. Current approaches focus on integrating protein-related knowledge into large language models through continued pretraining and multi-modal alignment, enabling simultaneous comprehension of textual descriptions and protein sequences. Through a thorough analysis of existing model architectures and text-based protein understanding benchmarks, we identify significant data leakage issues present in current benchmarks. Moreover, conventional metrics derived from natural language processing fail to accurately assess the model's performance in this domain. To address these limitations, we reorganize existing datasets and introduce a novel evaluation framework based on biological entities. Motivated by our observation, we propose a retrieval-enhanced method, which significantly outperforms fine-tuned LLMs for protein-to-text generation and shows accuracy and efficiency in training-free scenarios. Our code and data can be seen at https://github.com/IDEA-XL/RAPM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04838v1">On Automating Security Policies with Contemporary LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Short Paper. Accepted To Appear in IEEE SSE 2025 (part of SERVICES 2025)
    </div>
    <details class="paper-abstract">
      The complexity of modern computing environments and the growing sophistication of cyber threats necessitate a more robust, adaptive, and automated approach to security enforcement. In this paper, we present a framework leveraging large language models (LLMs) for automating attack mitigation policy compliance through an innovative combination of in-context learning and retrieval-augmented generation (RAG). We begin by describing how our system collects and manages both tool and API specifications, storing them in a vector database to enable efficient retrieval of relevant information. We then detail the architectural pipeline that first decomposes high-level mitigation policies into discrete tasks and subsequently translates each task into a set of actionable API calls. Our empirical evaluation, conducted using publicly available CTI policies in STIXv2 format and Windows API documentation, demonstrates significant improvements in precision, recall, and F1-score when employing RAG compared to a non-RAG baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04821v1">LogicPuzzleRL: Cultivating Robust Mathematical Reasoning in LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) excel at many supervised tasks but often struggle with structured reasoning in unfamiliar settings. This discrepancy suggests that standard fine-tuning pipelines may instill narrow, domain-specific heuristics rather than fostering general-purpose thinking strategies. In this work, we propose a "play to learn" framework that fine-tunes LLMs through reinforcement learning on a suite of seven custom logic puzzles, each designed to cultivate distinct reasoning skills such as constraint propagation, spatial consistency, and symbolic deduction. Using a reinforcement learning setup with verifiable rewards, models receive binary feedback based on puzzle correctness, encouraging iterative, hypothesis-driven problem solving. We demonstrate that this training approach significantly improves out-of-distribution performance on a range of mathematical benchmarks, especially for mid-difficulty problems that require multi-step reasoning. Analyses across problem categories and difficulty levels reveal that puzzle training promotes transferable reasoning routines, strengthening algebraic manipulation, geometric inference, and combinatorial logic, while offering limited gains on rote or highly specialized tasks. These findings show that reinforcement learning over logic puzzles reshapes the internal reasoning of LLMs, enabling more robust and compositional generalization without relying on task-specific symbolic tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04810v1">Dissecting Logical Reasoning in LLMs: A Fine-Grained Evaluation and Supervision Study</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Logical reasoning is a core capability for many applications of large language models (LLMs), yet existing benchmarks often rely solely on final-answer accuracy, failing to capture the quality and structure of the reasoning process. We propose FineLogic, a fine-grained evaluation framework that assesses logical reasoning across three dimensions: overall benchmark accuracy, stepwise soundness, and representation-level alignment. In addition, to better understand how reasoning capabilities emerge, we conduct a comprehensive study on the effects of supervision format during fine-tuning. We construct four supervision styles (one natural language and three symbolic variants) and train LLMs under each. Our findings reveal that natural language supervision yields strong generalization even on out-of-distribution and long-context tasks, while symbolic reasoning styles promote more structurally sound and atomic inference chains. Further, our representation-level probing shows that fine-tuning primarily improves reasoning behaviors through step-by-step generation, rather than enhancing shortcut prediction or internalized correctness. Together, our framework and analysis provide a more rigorous and interpretable lens for evaluating and improving logical reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24347v2">Fewer Hallucinations, More Verification: A Three-Stage LLM-Based Framework for ASR Error Correction</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Automatic Speech Recognition (ASR) error correction aims to correct recognition errors while preserving accurate text. Although traditional approaches demonstrate moderate effectiveness, LLMs offer a paradigm that eliminates the need for training and labeled data. However, directly using LLMs will encounter hallucinations problem, which may lead to the modification of the correct text. To address this problem, we propose the Reliable LLM Correction Framework (RLLM-CF), which consists of three stages: (1) error pre-detection, (2) chain-of-thought sub-tasks iterative correction, and (3) reasoning process verification. The advantage of our method is that it does not require additional information or fine-tuning of the model, and ensures the correctness of the LLM correction under multi-pass programming. Experiments on AISHELL-1, AISHELL-2, and Librispeech show that the GPT-4o model enhanced by our framework achieves 21%, 11%, 9%, and 11.4% relative reductions in CER/WER.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04788v1">Towards LLM-Centric Multimodal Fusion: A Survey on Integration Strategies and Techniques</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 18 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      The rapid progress of Multimodal Large Language Models(MLLMs) has transformed the AI landscape. These models combine pre-trained LLMs with various modality encoders. This integration requires a systematic understanding of how different modalities connect to the language backbone. Our survey presents an LLM-centric analysis of current approaches. We examine methods for transforming and aligning diverse modal inputs into the language embedding space. This addresses a significant gap in existing literature. We propose a classification framework for MLLMs based on three key dimensions. First, we examine architectural strategies for modality integration. This includes both the specific integration mechanisms and the fusion level. Second, we categorize representation learning techniques as either joint or coordinate representations. Third, we analyze training paradigms, including training strategies and objective functions. By examining 125 MLLMs developed between 2021 and 2025, we identify emerging patterns in the field. Our taxonomy provides researchers with a structured overview of current integration techniques. These insights aim to guide the development of more robust multimodal integration strategies for future models built on pre-trained foundations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.16142v2">Distilling the Implicit Multi-Branch Structure in LLMs' Reasoning via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 15 pages
    </div>
    <details class="paper-abstract">
      Distilling reasoning paths from teacher to student models via supervised fine-tuning (SFT) provides a shortcut for improving the reasoning ability of smaller Large Language Models (LLMs). However, the reasoning paths generated by teacher models often reflect only surface-level traces of their underlying authentic reasoning. Insights from cognitive neuroscience suggest that authentic reasoning involves a complex interweaving between meta-reasoning (which selects appropriate sub-problems from multiple candidates) and solving (which addresses the sub-problem). This implies authentic reasoning has an implicit multi-branch structure. Supervised fine-tuning collapses this rich structure into a flat sequence of token prediction in the teacher's reasoning path, preventing effective distillation of this structure to students. To address this limitation, we propose RLKD, a reinforcement learning (RL)-based distillation framework guided by a novel Generative Structure Reward Model (GSRM). Our GSRM converts reasoning paths into multiple meta-reasoning-solving steps and computes rewards to measure structural alignment between student and teacher reasoning. RLKD combines this reward with RL, enabling student LLMs to internalize the teacher's implicit multi-branch reasoning structure rather than merely mimicking fixed output paths. Experiments show RLKD surpasses standard SFT-RL pipelines even when trained on 0.1% of data under an RL-only regime, unlocking greater student reasoning potential than SFT-based distillation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.22944v4">Focus On This, Not That! Steering LLMs with Adaptive Feature Specification</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 36pages, 19 figures
    </div>
    <details class="paper-abstract">
      Despite the success of Instruction Tuning (IT) in training large language models (LLMs), such models often leverage spurious or biased features learnt from their training data and can become misaligned, leading to undesired behaviours. While existing techniques can steer model behaviour at inference-time, they are often post-hoc and do not embed steering as an intrinsic model feature. In this work, we introduce Focus Instruction Tuning (FIT), which trains LLMs to condition their responses by focusing on specific features whilst ignoring others, leading to different behaviours based on what features are specified. Across diverse benchmarks, we demonstrate that FIT: (i) successfully steers behaviour at inference time; (ii) increases robustness by amplifying core task signals and down-weighting spurious cues; (iii) mitigates social bias by suppressing demographic attributes; and (iv) generalises under distribution shifts and to previously unseen focus features. FIT therefore offers a lightweight, intrinsic mechanism for building more robust, fair, and easily controllable LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00722v2">Demystifying Cost-Efficiency in LLM Serving over Heterogeneous GPUs</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have led to increasingly diverse requests, accompanied with varying resource (compute and memory) demands to serve them. However, this in turn degrades the cost-efficiency of LLM serving as common practices primarily rely on homogeneous GPU resources. In response to this problem, this work conducts a thorough study about serving LLMs over heterogeneous GPU resources on cloud platforms. The rationale is that different GPU types exhibit distinct compute and memory characteristics, aligning well with the divergent resource demands of diverse requests. Particularly, through comprehensive benchmarking, we discover that the cost-efficiency of LLM serving can be substantially optimized by meticulously determining GPU composition, deployment configurations, and workload assignments. Subsequently, we design a scheduling algorithm via mixed-integer linear programming, aiming at deducing the most cost-efficient serving plan under the constraints of price budget and real-time GPU availability. Remarkably, our approach effectively outperforms homogeneous and heterogeneous baselines under a wide array of scenarios, covering diverse workload traces, varying GPU availablilities, and multi-model serving. This casts new light on more accessible and efficient LLM serving over heterogeneous cloud resources.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04715v1">Towards Holistic Visual Quality Assessment of AI-Generated Videos: A LLM-Based Multi-Dimensional Evaluation Model</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      The development of AI-Generated Video (AIGV) technology has been remarkable in recent years, significantly transforming the paradigm of video content production. However, AIGVs still suffer from noticeable visual quality defects, such as noise, blurriness, frame jitter and low dynamic degree, which severely impact the user's viewing experience. Therefore, an effective automatic visual quality assessment is of great importance for AIGV content regulation and generative model improvement. In this work, we decompose the visual quality of AIGVs into three dimensions: technical quality, motion quality, and video semantics. For each dimension, we design corresponding encoder to achieve effective feature representation. Moreover, considering the outstanding performance of large language models (LLMs) in various vision and language tasks, we introduce a LLM as the quality regression module. To better enable the LLM to establish reasoning associations between multi-dimensional features and visual quality, we propose a specially designed multi-modal prompt engineering framework. Additionally, we incorporate LoRA fine-tuning technology during the training phase, allowing the LLM to better adapt to specific tasks. Our proposed method achieved \textbf{second place} in the NTIRE 2025 Quality Assessment of AI-Generated Content Challenge: Track 2 AI Generated video, demonstrating its effectiveness. Codes can be obtained at https://github.com/QiZelu/AIGVEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04711v1">LLM-based phoneme-to-grapheme for phoneme-based speech recognition</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Interspeech 2025
    </div>
    <details class="paper-abstract">
      In automatic speech recognition (ASR), phoneme-based multilingual pre-training and crosslingual fine-tuning is attractive for its high data efficiency and competitive results compared to subword-based models. However, Weighted Finite State Transducer (WFST) based decoding is limited by its complex pipeline and inability to leverage large language models (LLMs). Therefore, we propose LLM-based phoneme-to-grapheme (LLM-P2G) decoding for phoneme-based ASR, consisting of speech-to-phoneme (S2P) and phoneme-to-grapheme (P2G). A challenge is that there seems to have information loss in cascading S2P and P2G. To address this challenge, we propose two training strategies: data augmentation with noisy phonemes (DANP), and randomized top-$K$ marginalized (TKM) training and decoding. Our experimental results show that LLM-P2G outperforms WFST-based systems in crosslingual ASR for Polish and German, by relative WER reductions of 3.6% and 6.9% respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05849v3">AGENTFUZZER: Generic Black-Box Fuzzing for Indirect Prompt Injection against LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      The strong planning and reasoning capabilities of Large Language Models (LLMs) have fostered the development of agent-based systems capable of leveraging external tools and interacting with increasingly complex environments. However, these powerful features also introduce a critical security risk: indirect prompt injection, a sophisticated attack vector that compromises the core of these agents, the LLM, by manipulating contextual information rather than direct user prompts. In this work, we propose a generic black-box fuzzing framework, AgentFuzzer, designed to automatically discover and exploit indirect prompt injection vulnerabilities across diverse LLM agents. Our approach starts by constructing a high-quality initial seed corpus, then employs a seed selection algorithm based on Monte Carlo Tree Search (MCTS) to iteratively refine inputs, thereby maximizing the likelihood of uncovering agent weaknesses. We evaluate AgentFuzzer on two public benchmarks, AgentDojo and VWA-adv, where it achieves 71% and 70% success rates against agents based on o3-mini and GPT-4o, respectively, nearly doubling the performance of baseline attacks. Moreover, AgentFuzzer exhibits strong transferability across unseen tasks and internal LLMs, as well as promising results against defenses. Beyond benchmark evaluations, we apply our attacks in real-world environments, successfully misleading agents to navigate to arbitrary URLs, including malicious sites.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11315v2">MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens</a></div>
    <div class="paper-meta">
      📅 2025-06-05
      | 💬 Accepted at Findings of ACL 2025. The code and models are available https://github.com/JeongHun0716/MMS-LLaMA
    </div>
    <details class="paper-abstract">
      Audio-Visual Speech Recognition (AVSR) achieves robust speech recognition in noisy environments by combining auditory and visual information. However, recent Large Language Model (LLM) based AVSR systems incur high computational costs due to the high temporal resolution of audio-visual speech processed by LLMs. In this work, we introduce an efficient multimodal speech LLM framework that minimizes token length while preserving essential linguistic content. Our approach employs an early AV-fusion module for streamlined feature integration, an audio-visual speech Q-Former that dynamically allocates tokens based on input duration, and a refined query allocation strategy with a speech rate predictor to adjust token allocation according to speaking speed of each audio sample. Extensive experiments on the LRS3 dataset show that our method achieves state-of-the-art performance with a WER of 0.72% while using only 3.5 tokens per second. Moreover, our approach not only reduces token usage by 86% compared to the previous multimodal speech LLM framework, but also improves computational efficiency by reducing FLOPs by 35.7%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04651v1">Agents of Change: Self-Evolving LLM Agents for Strategic Planning</a></div>
    <div class="paper-meta">
      📅 2025-06-05
    </div>
    <details class="paper-abstract">
      Recent advances in LLMs have enabled their use as autonomous agents across a range of tasks, yet they continue to struggle with formulating and adhering to coherent long-term strategies. In this paper, we investigate whether LLM agents can self-improve when placed in environments that explicitly challenge their strategic planning abilities. Using the board game Settlers of Catan, accessed through the open-source Catanatron framework, we benchmark a progression of LLM-based agents, from a simple game-playing agent to systems capable of autonomously rewriting their own prompts and their player agent's code. We introduce a multi-agent architecture in which specialized roles (Analyzer, Researcher, Coder, and Player) collaborate to iteratively analyze gameplay, research new strategies, and modify the agent's logic or prompt. By comparing manually crafted agents to those evolved entirely by LLMs, we evaluate how effectively these systems can diagnose failure and adapt over time. Our results show that self-evolving agents, particularly when powered by models like Claude 3.7 and GPT-4o, outperform static baselines by autonomously adopting their strategies, passing along sample behavior to game-playing agents, and demonstrating adaptive reasoning over multiple iterations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04044v1">Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted to SemEval-2025, an ACL 2025 workshop
    </div>
    <details class="paper-abstract">
      This paper describes LIBU (LoRA enhanced influence-based unlearning), an algorithm to solve the task of unlearning - removing specific knowledge from a large language model without retraining from scratch and compromising its overall utility (SemEval-2025 Task 4: Unlearning sensitive content from Large Language Models). The algorithm combines classical \textit{influence functions} to remove the influence of the data from the model and \textit{second-order optimization} to stabilize the overall utility. Our experiments show that this lightweight approach is well applicable for unlearning LLMs in different kinds of task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04043v1">Think Like a Person Before Responding: A Multi-Faceted Evaluation of Persona-Guided LLMs for Countering Hate</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted at ACL WOAH 2025
    </div>
    <details class="paper-abstract">
      Automated counter-narratives (CN) offer a promising strategy for mitigating online hate speech, yet concerns about their affective tone, accessibility, and ethical risks remain. We propose a framework for evaluating Large Language Model (LLM)-generated CNs across four dimensions: persona framing, verbosity and readability, affective tone, and ethical robustness. Using GPT-4o-Mini, Cohere's CommandR-7B, and Meta's LLaMA 3.1-70B, we assess three prompting strategies on the MT-Conan and HatEval datasets. Our findings reveal that LLM-generated CNs are often verbose and adapted for people with college-level literacy, limiting their accessibility. While emotionally guided prompts yield more empathetic and readable responses, there remain concerns surrounding safety and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02524v3">CheckEmbed: Effective Verification of LLM Solutions to Open-Ended Tasks</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are transforming a wide range of domains, yet verifying their outputs remains a significant challenge, especially for complex open-ended tasks such as consolidation, summarization, and knowledge extraction. To address this, we introduce CheckEmbed (CE): a simple, scalable, and accurate verification method. CE reduces each LLM answer to a single embedding vector using powerful modern embedding LLM models like SFR-Embedding-Mistral. Prior methods such as BERTScore and SelfCheckGPT relied on weaker encoders like BERT, forcing them to operate at token or sentence granularity. In contrast, CE performs fast, semantically rich comparisons directly at the whole-answer level, overcoming key limitations in both accuracy and scalability. We conduct a comprehensive design and time complexity analysis across 13 verification baselines, including classical text scorers (e.g., BLEU), stability-based methods (e.g., SelfCheckGPT), and generative evaluators (e.g., LLM-as-a-Judge), which highlights the effectiveness, efficiency, versatility, and simplicity of CE. Empirical results show that CE reliably detects hallucinations in both closed and open-ended tasks. We further present evidence that CE generalizes beyond text to other modalities such as vision, establishing it as a practical and versatile verification framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04019v1">CETBench: A Novel Dataset constructed via Transformations over Programs for Benchmarking LLMs for Code-Equivalence Checking</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      LLMs have been extensively used for the task of automated code generation. In this work, we examine the applicability of LLMs for the related but relatively unexplored task of code-equivalence checking, i.e., given two programs, whether they are functionally equivalent or not. This is an important problem since benchmarking code equivalence can play a critical role in evaluating LLM capabilities for tasks such as code re-writing and code translation. Towards this end, we present CETBench - Code Equivalence with Transformations Benchmark, constructed via a repository of programs, where two programs in the repository may be solving the same or different tasks. Each instance in our dataset is obtained by taking a pair of programs in the repository and applying a random series of pre-defined code transformations, resulting in (non-)equivalent pairs. Our analysis on this dataset reveals a surprising finding that very simple code transformations in the underlying pair of programs can result in a significant drop in performance of SOTA LLMs for the task of code-equivalence checking. To remedy this, we present a simple fine-tuning-based approach to boost LLM performance on the transformed pairs of programs. Our approach for dataset generation is generic, and can be used with repositories with varying program difficulty levels and allows for applying varying numbers as well as kinds of transformations. In our experiments, we perform ablations over the difficulty level of original programs, as well as the kind of transformations used in generating pairs for equivalence checking. Our analysis presents deep insights into the working of LLMs for the task of code-equivalence, and points to the fact that they may still be far from what could be termed as a semantic understanding of the underlying code.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04018v1">AgentMisalignment: Measuring the Propensity for Misaligned Behaviour in LLM-Based Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Prepint, under review for NeurIPS 2025
    </div>
    <details class="paper-abstract">
      As Large Language Model (LLM) agents become more widespread, associated misalignment risks increase. Prior work has examined agents' ability to enact misaligned behaviour (misalignment capability) and their compliance with harmful instructions (misuse propensity). However, the likelihood of agents attempting misaligned behaviours in real-world settings (misalignment propensity) remains poorly understood. We introduce a misalignment propensity benchmark, AgentMisalignment, consisting of a suite of realistic scenarios in which LLM agents have the opportunity to display misaligned behaviour. We organise our evaluations into subcategories of misaligned behaviours, including goal-guarding, resisting shutdown, sandbagging, and power-seeking. We report the performance of frontier models on our benchmark, observing higher misalignment on average when evaluating more capable models. Finally, we systematically vary agent personalities through different system prompts. We find that persona characteristics can dramatically and unpredictably influence misalignment tendencies -- occasionally far more than the choice of model itself -- highlighting the importance of careful system prompt engineering for deployed AI agents. Our work highlights the failure of current alignment methods to generalise to LLM agents, and underscores the need for further propensity evaluations as autonomous systems become more prevalent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04015v1">GORACS: Group-level Optimal Transport-guided Coreset Selection for LLM-based Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted by KDD 2025
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have shown great potential in recommender systems, the prohibitive computational costs for fine-tuning LLMs on entire datasets hinder their successful deployment in real-world scenarios. To develop affordable and effective LLM-based recommender systems, we focus on the task of coreset selection which identifies a small subset of fine-tuning data to optimize the test loss, thereby facilitating efficient LLMs' fine-tuning. Although there exist some intuitive solutions of subset selection, including distribution-based and importance-based approaches, they often lead to suboptimal performance due to the misalignment with downstream fine-tuning objectives or weak generalization ability caused by individual-level sample selection. To overcome these challenges, we propose GORACS, which is a novel Group-level Optimal tRAnsport-guided Coreset Selection framework for LLM-based recommender systems. GORACS is designed based on two key principles for coreset selection: 1) selecting the subsets that minimize the test loss to align with fine-tuning objectives, and 2) enhancing model generalization through group-level data selection. Corresponding to these two principles, GORACS has two key components: 1) a Proxy Optimization Objective (POO) leveraging optimal transport and gradient information to bound the intractable test loss, thus reducing computational costs by avoiding repeated LLM retraining, and 2) a two-stage Initialization-Then-Refinement Algorithm (ITRA) for efficient group-level selection. Our extensive experiments across diverse recommendation datasets and tasks validate that GORACS significantly reduces fine-tuning costs of LLMs while achieving superior performance over the state-of-the-art baselines and full data training. The source code of GORACS are available at https://github.com/Mithas-114/GORACS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02943v2">A Multi-agent LLM-based JUnit Test Generation with Strong Oracles</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Unit testing plays a critical role in ensuring software correctness. However, writing unit tests manually is laborious, especially for strong typed languages like Java, motivating the need for automated approaches. Traditional methods primarily rely on search-based or randomized algorithms to generate tests that achieve high code coverage and produce regression oracles, which are derived from the program's current behavior rather than its intended functionality. Recent advances in large language models (LLMs) have enabled oracle generation from natural language descriptions. However, existing LLM-based methods often require LLM fine-tuning or rely on external tools such as EvoSuite for test prefix generation. In this work, we propose CANDOR, a novel end-to-end, prompt-based LLM framework for automated JUnit test generation. CANDOR orchestrates multiple specialized LLM agents to generate JUnit tests, including both high-quality test prefixes and accurate oracles. To mitigate the notorious hallucinations in LLMs, we introduce a novel strategy that engages multiple reasoning LLMs in a panel discussion and generate accurate oracles based on consensus. Additionally, to reduce the verbosity of reasoning LLMs' outputs, we propose a novel dual-LLM pipeline to produce concise and structured oracle evaluations. Our experiments on the HumanEvalJava and LeetCodeJava datasets show that CANDOR can generate accurate oracles and is slightly better than EvoSuite in generating tests with high line coverage and clearly superior in terms of mutation score. Moreover, CANDOR significantly outperforms the state-of-the-art, prompt-based test generator LLM-Empirical, achieving improvements of 15.8 to 25.1 percentage points in oracle correctness on both correct and faulty source code. Ablation studies confirm the critical contributions of key agents in improving test prefix quality and oracle accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03106v2">Critique-GRPO: Advancing LLM Reasoning with Natural Language and Numerical Feedback</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 38 pages
    </div>
    <details class="paper-abstract">
      Recent advances in reinforcement learning (RL) with numerical feedback, such as scalar rewards, have significantly enhanced the complex reasoning capabilities of large language models (LLMs). Despite this success, we identify three key challenges encountered by RL with solely numerical feedback: performance plateaus, limited effectiveness of self-reflection, and persistent failures. We then demonstrate that RL-finetuned models, even after exhibiting performance plateaus, can generate correct refinements on persistently failed problems by leveraging natural language feedback in the form of critiques. Building on this insight, we propose Critique-GRPO, an online RL framework that integrates both natural language and numerical feedback for effective policy optimization. Critique-GRPO enables LLMs to learn from initial responses and critique-guided refinements simultaneously while maintaining exploration. Extensive experiments using Qwen2.5-7B-Base and Qwen3-8B-Base show that Critique-GRPO consistently outperforms supervised learning-based and RL-based fine-tuning approaches across eight challenging mathematical, STEM, and general reasoning tasks, improving average pass@1 scores by approximately 4.5% and 5%, respectively. Notably, Critique-GRPO surpasses a strong baseline that incorporates expert demonstrations within online RL. Further analysis reveals two critical insights about policy exploration: (1) higher entropy does not always guarantee efficient learning from exploration, and (2) longer responses do not necessarily lead to more effective exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03939v1">Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted by ACL 2025
    </div>
    <details class="paper-abstract">
      Graph Retrieval Augmented Generation (GraphRAG) effectively enhances external knowledge integration capabilities by explicitly modeling knowledge relationships, thereby improving the factual accuracy and generation quality of Large Language Models (LLMs) in specialized domains. However, existing methods suffer from two inherent limitations: 1) Inefficient Information Aggregation: They rely on a single agent and fixed iterative patterns, making it difficult to adaptively capture multi-level textual, structural, and degree information within graph data. 2) Rigid Reasoning Mechanism: They employ preset reasoning schemes, which cannot dynamically adjust reasoning depth nor achieve precise semantic correction. To overcome these limitations, we propose Graph Counselor, an GraphRAG method based on multi-agent collaboration. This method uses the Adaptive Graph Information Extraction Module (AGIEM), where Planning, Thought, and Execution Agents work together to precisely model complex graph structures and dynamically adjust information extraction strategies, addressing the challenges of multi-level dependency modeling and adaptive reasoning depth. Additionally, the Self-Reflection with Multiple Perspectives (SR) module improves the accuracy and semantic consistency of reasoning results through self-reflection and backward reasoning mechanisms. Experiments demonstrate that Graph Counselor outperforms existing methods in multiple graph reasoning tasks, exhibiting higher reasoning accuracy and generalization ability. Our code is available at https://github.com/gjq100/Graph-Counselor.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03930v1">VisCoder: Fine-Tuning LLMs for Executable Python Visualization Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often struggle with visualization tasks like plotting diagrams, charts, where success depends on both code correctness and visual semantics. Existing instruction-tuning datasets lack execution-grounded supervision and offer limited support for iterative code correction, resulting in fragile and unreliable plot generation. We present VisCode-200K, a large-scale instruction tuning dataset for Python-based visualization and self-correction. It contains over 200K examples from two sources: (1) validated plotting code from open-source repositories, paired with natural language instructions and rendered plots; and (2) 45K multi-turn correction dialogues from Code-Feedback, enabling models to revise faulty code using runtime feedback. We fine-tune Qwen2.5-Coder-Instruct on VisCode-200K to create VisCoder, and evaluate it on PandasPlotBench. VisCoder significantly outperforms strong open-source baselines and approaches the performance of proprietary models like GPT-4o-mini. We further adopt a self-debug evaluation protocol to assess iterative repair, demonstrating the benefits of feedback-driven learning for executable, visually accurate code generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.23703v2">Let's Reason Formally: Natural-Formal Hybrid Reasoning Enhances LLM's Math Capability</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Enhancing the mathematical reasoning capabilities of LLMs has garnered significant attention in both the mathematical and computer science communities. Recent works have made substantial progress in both Natural Language (NL) reasoning and Formal Language (FL) reasoning by leveraging the potential of pure Reinforcement Learning (RL) methods on base models. However, RL approaches struggle to impart new capabilities not presented in the base model, highlighting the need to integrate more knowledge like FL into NL math reasoning effectively. Yet, this integration is challenging due to inherent disparities in problem structure and reasoning format between NL and FL. To address these challenges, we introduce **NL-FL HybridReasoning**, an end-to-end framework designed to incorporate the FL expert into NL math problem-solving. To bridge the NL and FL input format gap, we propose the *NL-FL Problem Alignment* method, which reformulates the Question-Answering (QA) problems in NL as existence theorems in FL. Subsequently, the *Mixed Problem Input* technique we provide enables the FL reasoner to handle both QA and existence problems concurrently. Lastly, we mitigate the NL and FL output format gap in reasoning through an LLM-based *Answer Extraction* mechanism. Comprehensive experiments demonstrate that the **HybridReasoning** framework achieves **89.80%** and **84.34%** accuracy rates on the MATH-500 and the AMC benchmarks, surpassing the NL baseline by 4.60% and 4.82%, respectively. Notably, some problems resolved by our framework remain unsolved by the NL baseline model even under a larger number of trials.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03923v1">More or Less Wrong: A Benchmark for Directional Bias in LLM Comparative Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are known to be sensitive to input phrasing, but the mechanisms by which semantic cues shape reasoning remain poorly understood. We investigate this phenomenon in the context of comparative math problems with objective ground truth, revealing a consistent and directional framing bias: logically equivalent questions containing the words ``more'', ``less'', or ``equal'' systematically steer predictions in the direction of the framing term. To study this effect, we introduce MathComp, a controlled benchmark of 300 comparison scenarios, each evaluated under 14 prompt variants across three LLM families. We find that model errors frequently reflect linguistic steering, systematic shifts toward the comparative term present in the prompt. Chain-of-thought prompting reduces these biases, but its effectiveness varies: free-form reasoning is more robust, while structured formats may preserve or reintroduce directional drift. Finally, we show that including demographic identity terms (e.g., ``a woman'', ``a Black person'') in input scenarios amplifies directional drift, despite identical underlying quantities, highlighting the interplay between semantic framing and social referents. These findings expose critical blind spots in standard evaluation and motivate framing-aware benchmarks for diagnosing reasoning robustness and fairness in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03921v1">Boosting Open-Source LLMs for Program Repair via Reasoning Transfer and LLM-Guided Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Several closed-source LLMs have consistently outperformed open-source alternatives in program repair tasks, primarily due to their superior reasoning capabilities and extensive pre-training. This paper introduces Repairity, a novel three-stage methodology that significantly narrows this performance gap through reasoning extraction and reinforcement learning. Our approach: (1) systematically filters high-quality reasoning traces from closed-source models using correctness verification, (2) transfers this reasoning knowledge to open-source models via supervised fine-tuning, and (3) develops reinforcement learning with LLM-based feedback to further optimize performance. Empirical evaluation across multiple program repair benchmarks demonstrates that Repairity improves the performance of Qwen2.5-Coder-32B-Instruct, a base open source LLM, by 8.68\% on average, reducing the capability gap with Claude-Sonnet3.7, a state-of-the-art closed-source model, from 10.05% to 1.35%. Ablation studies confirm that both reasoning extraction and LLM-guided reinforcement learning contribute significantly to these improvements. Our methodology generalizes effectively to additional code-related tasks, enabling organizations to leverage high-quality program repair capabilities while maintaining the customizability, transparency, and deployment flexibility inherent to open-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04670v2">LLM Code Customization with Visual Results: A Benchmark on TikZ</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      With the rise of AI-based code generation, customizing existing code out of natural language instructions to modify visual results -such as figures or images -has become possible, promising to reduce the need for deep programming expertise. However, even experienced developers can struggle with this task, as it requires identifying relevant code regions (feature location), generating valid code variants, and ensuring the modifications reliably align with user intent. In this paper, we introduce vTikZ, the first benchmark designed to evaluate the ability of Large Language Models (LLMs) to customize code while preserving coherent visual outcomes. Our benchmark consists of carefully curated vTikZ editing scenarios, parameterized ground truths, and a reviewing tool that leverages visual feedback to assess correctness. Empirical evaluation with stateof-the-art LLMs shows that existing solutions struggle to reliably modify code in alignment with visual intent, highlighting a gap in current AI-assisted code editing approaches. We argue that vTikZ opens new research directions for integrating LLMs with visual feedback mechanisms to improve code customization tasks in various domains beyond TikZ, including image processing, art creation, Web design, and 3D modeling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03887v1">Pre$^3$: Enabling Deterministic Pushdown Automata for Faster Structured LLM Generation</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Published as a conference paper at ACL 2025
    </div>
    <details class="paper-abstract">
      Extensive LLM applications demand efficient structured generations, particularly for LR(1) grammars, to produce outputs in specified formats (e.g., JSON). Existing methods primarily parse LR(1) grammars into a pushdown automaton (PDA), leading to runtime execution overhead for context-dependent token processing, especially inefficient under large inference batches. To address these issues, we propose Pre$^3$ that exploits deterministic pushdown automata (DPDA) to optimize the constrained LLM decoding efficiency. First, by precomputing prefix-conditioned edges during the preprocessing, Pre$^3$ enables ahead-of-time edge analysis and thus makes parallel transition processing possible. Second, by leveraging the prefix-conditioned edges, Pre$^3$ introduces a novel approach that transforms LR(1) transition graphs into DPDA, eliminating the need for runtime path exploration and achieving edge transitions with minimal overhead. Pre$^3$ can be seamlessly integrated into standard LLM inference frameworks, reducing time per output token (TPOT) by up to 40% and increasing throughput by up to 36% in our experiments. Our code is available at https://github.com/ModelTC/lightllm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03857v1">Prompt Candidates, then Distill: A Teacher-Student Framework for LLM-driven Data Annotation</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted to ACL 2025 (Main conference)
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have demonstrated significant potential for data annotation, markedly reducing the labor costs associated with downstream applications. However, existing methods mostly adopt an aggressive strategy by prompting LLM to determine a single gold label for each unlabeled sample. Due to the inherent uncertainty within LLMs, they often produce incorrect labels for difficult samples, severely compromising the data quality for downstream applications. Motivated by ambiguity aversion in human behaviors, we propose a novel candidate annotation paradigm wherein large language models are encouraged to output all possible labels when incurring uncertainty. To ensure unique labels are provided for downstream tasks, we develop a teacher-student framework CanDist that distills candidate annotations with a Small Language Model (SLM). We further provide a rigorous justification demonstrating that distilling candidate annotations from the teacher LLM offers superior theoretical guarantees compared to directly using single annotations. Extensive experiments across six text classification tasks validate the effectiveness of our proposed method. The source code is available at https://github.com/MingxuanXia/CanDist.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.21082v3">LLMs Think, But Not In Your Flow: Reasoning-Level Personalization for Black-Box Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have recently achieved impressive performance across a wide range of natural language tasks and are now widely used in real-world applications. Among them, black-box LLMs--served via APIs without access to model internals--are especially dominant due to their scalability and ease of deployment. Despite their strong capabilities, these models typically produce generalized responses that overlook personal preferences and reasoning styles. This has led to growing interest in black-box LLM personalization, which aims to tailor model outputs to user-specific context without modifying model parameters. However, existing approaches primarily focus on response-level personalization, attempting to match final outputs without modeling personal thought process. To address this limitation, we propose RPM, a framework for reasoning-level personalization that aligns the model's reasoning process with a user's personalized logic. RPM first constructs statistical user-specific factors by extracting and grouping response-influential features from user history. It then builds personalized reasoning paths that reflect how these factors are used in context. In the inference stage, RPM retrieves reasoning-aligned examples for new queries via feature-level similarity and performs inference conditioned on the structured factors and retrieved reasoning paths, enabling the model to follow user-specific reasoning trajectories. This reasoning-level personalization enhances both predictive accuracy and interpretability by grounding model outputs in user-specific logic through structured information. Extensive experiments across diverse tasks show that RPM consistently outperforms response-level personalization methods, demonstrating the effectiveness of reasoning-level personalization in black-box LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.04920v4">Enabling LLM Knowledge Analysis via Extensive Materialization</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 14 pages, 4 tables, 12 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have majorly advanced NLP and AI, and next to their ability to perform a wide range of procedural tasks, a major success factor is their internalized factual knowledge. Since Petroni et al. (2019), analyzing this knowledge has gained attention. However, most approaches investigate one question at a time via modest-sized pre-defined samples, introducing an ``availability bias'' (Tversky&Kahnemann, 1973) that prevents the analysis of knowledge (or beliefs) of LLMs beyond the experimenter's predisposition. To address this challenge, we propose a novel methodology to comprehensively materialize an LLM's factual knowledge through recursive querying and result consolidation. Our approach is a milestone for LLM research, for the first time providing constructive insights into the scope and structure of LLM knowledge (or beliefs). As a prototype, we build GPTKB, a knowledge base (KB) comprising 101 million relational triples for over 2.9 million entities from GPT-4o-mini. We use GPTKB to exemplarily analyze GPT-4o-mini's factual knowledge in terms of scale, accuracy, bias, cutoff and consistency, at the same time. GPTKB is accessible at https://gptkb.org
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03801v1">From Theory to Practice: Real-World Use Cases on Trustworthy LLM-Driven Process Modeling, Prediction and Automation</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted to the Next Gen Data and Process Management: Large Language Models and Beyond workshop at SIGMOD 2025
    </div>
    <details class="paper-abstract">
      Traditional Business Process Management (BPM) struggles with rigidity, opacity, and scalability in dynamic environments while emerging Large Language Models (LLMs) present transformative opportunities alongside risks. This paper explores four real-world use cases that demonstrate how LLMs, augmented with trustworthy process intelligence, redefine process modeling, prediction, and automation. Grounded in early-stage research projects with industrial partners, the work spans manufacturing, modeling, life-science, and design processes, addressing domain-specific challenges through human-AI collaboration. In manufacturing, an LLM-driven framework integrates uncertainty-aware explainable Machine Learning (ML) with interactive dialogues, transforming opaque predictions into auditable workflows. For process modeling, conversational interfaces democratize BPMN design. Pharmacovigilance agents automate drug safety monitoring via knowledge-graph-augmented LLMs. Finally, sustainable textile design employs multi-agent systems to navigate regulatory and environmental trade-offs. We intend to examine tensions between transparency and efficiency, generalization and specialization, and human agency versus automation. By mapping these trade-offs, we advocate for context-sensitive integration prioritizing domain needs, stakeholder values, and iterative human-in-the-loop workflows over universal solutions. This work provides actionable insights for researchers and practitioners aiming to operationalize LLMs in critical BPM environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03785v1">Knockout LLM Assessment: Using Large Language Models for Evaluations through Iterative Pairwise Comparisons</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 4 pages, 2 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown to be effective evaluators across various domains such as machine translations or the scientific domain. Current LLM-as-a-Judge approaches rely mostly on individual assessments or a single round of pairwise assessments, preventing the judge LLM from developing a global ranking perspective. To address this, we present Knockout Assessment, an LLM-asa Judge method using a knockout tournament system with iterative pairwise comparisons. Experiments across three LLMs on two datasets show that knockout assessment improves scoring accuracy, increasing Pearson correlation with expert evaluations by 0.07 on average for university-level exam scoring and machine translation evaluations, aligning LLM assessments more closely with human scoring.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.14175v2">Hypothetical Documents or Knowledge Leakage? Rethinking LLM-based Query Expansion</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 ACL 2025 (Findings)
    </div>
    <details class="paper-abstract">
      Query expansion methods powered by large language models (LLMs) have demonstrated effectiveness in zero-shot retrieval tasks. These methods assume that LLMs can generate hypothetical documents that, when incorporated into a query vector, enhance the retrieval of real evidence. However, we challenge this assumption by investigating whether knowledge leakage in benchmarks contributes to the observed performance gains. Using fact verification as a testbed, we analyze whether the generated documents contain information entailed by ground-truth evidence and assess their impact on performance. Our findings indicate that, on average, performance improvements consistently occurred for claims whose generated documents included sentences entailed by gold evidence. This suggests that knowledge leakage may be present in fact-verification benchmarks, potentially inflating the perceived performance of LLM-based query expansion methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03700v1">AdaDecode: Accelerating LLM Decoding with Adaptive Layer Parallelism</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 ICML 2025. Code: https://github.com/weizhepei/AdaDecode
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly used for long-content generation (e.g., long Chain-of-Thought reasoning) where decoding efficiency becomes a critical bottleneck: Autoregressive decoding is inherently limited by its sequential token generation process, where each token must be generated before the next can be processed. This sequential dependency restricts the ability to fully leverage modern hardware's parallel processing capabilities. Existing methods like speculative decoding and layer skipping offer potential speedups but have notable drawbacks: speculative decoding relies on an auxiliary "drafter" model, which can be challenging to acquire and increases memory overhead, while layer skipping may introduce discrepancies in the outputs due to the missing key-value cache at skipped layers. In this work, we propose AdaDecode, which accelerates LLM decoding without requiring auxiliary models or changes to the original model parameters, while ensuring output consistency. AdaDecode leverages the insight that many tokens can accurately be generated at intermediate layers, as further layers often do not significantly alter predictions once the model reaches a certain confidence. By adaptively generating tokens at intermediate layers when confidence is high, AdaDecode enables the next token's computation to begin immediately. The remaining layer computations for early-predicted tokens are deferred and executed in parallel with subsequent tokens when needed, maximizing hardware utilization and reducing decoding latency. A final verification step ensures that early predictions match the results of standard autoregressive decoding, preserving output parity. Experiments across diverse generation tasks shows that AdaDecode consistently achieves superior decoding throughput with up to 1.73x speedup, while guaranteeing output parity with standard autoregressive decoding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03691v1">A Two-Staged LLM-Based Framework for CI/CD Failure Detection and Remediation with Industrial Validation</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 12 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Continuous Integration and Continuous Deployment (CI/CD) pipelines are pivotal to modern software engineering, yet diagnosing and resolving their failures remains a complex and labor-intensive challenge. In this paper, we present LogSage, the first end-to-end LLM-powered framework that performs root cause analysis and solution generation from failed CI/CD pipeline logs. During the root cause analysis stage, LogSage employs a specialized log preprocessing pipeline tailored for LLMs, which extracts critical error logs and eliminates noise to enhance the precision of LLM-driven root cause analysis. In the solution generation stage, LogSage leverages RAG to integrate historical resolution strategies and utilizes tool-calling to deliver actionable, automated fixes. We evaluated the root cause analysis stage using a newly curated open-source dataset, achieving 98\% in precision and 12\% improvement over naively designed LLM-based log analysis baselines, while attaining near-perfect recall. The end-to-end system was rigorously validated in a large-scale industrial CI/CD environment of production quality, processing more than 3,000 executions daily and accumulating more than 1.07 million executions in its first year of deployment, with end-to-end precision exceeding 88\%. These two forms of evaluation confirm that LogSage providing a scalable and practical solution to manage CI/CD pipeline failures in real-world DevOps workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10515v2">Probing LLMs for Multilingual Discourse Generalization Through a Unified Label Set</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 18 pages, 7 figures, 3 tables, code: https://github.com/mainlp/discourse_probes, camera-ready revision for ACL 2025
    </div>
    <details class="paper-abstract">
      Discourse understanding is essential for many NLP tasks, yet most existing work remains constrained by framework-dependent discourse representations. This work investigates whether large language models (LLMs) capture discourse knowledge that generalizes across languages and frameworks. We address this question along two dimensions: (1) developing a unified discourse relation label set to facilitate cross-lingual and cross-framework discourse analysis, and (2) probing LLMs to assess whether they encode generalizable discourse abstractions. Using multilingual discourse relation classification as a testbed, we examine a comprehensive set of 23 LLMs of varying sizes and multilingual capabilities. Our results show that LLMs, especially those with multilingual training corpora, can generalize discourse information across languages and frameworks. Further layer-wise analyses reveal that language generalization at the discourse level is most salient in the intermediate layers. Lastly, our error analysis provides an account of challenging relation classes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00486v3">It Takes a Good Model to Train a Good Model: Generalized Gaussian Priors for Optimized LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Despite rapid advancements in the research and deployment of large language models (LLMs), the statistical distribution of model parameters, as well as their influence on initialization, training dynamics, and downstream efficiency, has received surprisingly little attention. A recent work introduced BackSlash, a training-time compression algorithm. It first demonstrated that pre-trained LLM parameters follow generalized Gaussian distributions (GGDs) better. By optimizing GG priors during training, BackSlash can reduce parameters by up to 90\% with minimal performance loss. Building on this foundational insight, we propose a unified, end-to-end framework for LLM optimization based on the GG model. Our contributions are threefold: (1) GG-based initialization scheme that aligns with the statistical structure of trained models, resulting in faster convergence and improved accuracy; (2) DeepShape, a post-training regularization method that reshapes weight distributions to match a GG profile, improving compressibility with minimized degradation in performance; and (3) RF8, a compact and hardware-efficient 8-bit floating-point format designed for GG-distributed-initialized BackSlash training, enabling low-cost inference without compromising accuracy. Experiments across diverse model architectures show that our framework consistently yields smaller and faster models that match or outperform standard training baselines. By grounding LLM development in principled statistical modeling, this work forges a new path toward efficient, scalable, and hardware-aware AI systems. The code is available on our project page: https://huggingface.co/spaces/shifeng3711/gg_prior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16502v3">RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Preprint. Accepted by ICML 2025
    </div>
    <details class="paper-abstract">
      Formal logic enables computers to reason in natural language by representing sentences in symbolic forms and applying rules to derive conclusions. However, in what our study characterizes as "rulebreaker" scenarios, this method can lead to conclusions that are typically not inferred or accepted by humans given their common sense and factual knowledge. Inspired by works in cognitive science, we create RULEBREAKERS, the first dataset for rigorously evaluating the ability of large language models (LLMs) to recognize and respond to rulebreakers (versus non-rulebreakers) in a human-like manner. Evaluating seven LLMs, we find that most models, including GPT-4o, achieve mediocre accuracy on RULEBREAKERS and exhibit some tendency to over-rigidly apply logical rules unlike what is expected from typical human reasoners. Further analysis suggests that this apparent failure is potentially associated with the models' poor utilization of their world knowledge and their attention distribution patterns. Whilst revealing a limitation of current LLMs, our study also provides a timely counterbalance to a growing body of recent works that propose methods relying on formal logic to improve LLMs' general reasoning capabilities, highlighting their risk of further increasing divergence between LLMs and human-like reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19449v3">VecTrans: Enhancing Compiler Auto-Vectorization through LLM-Assisted Code Transformations</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Auto-vectorization is a fundamental optimization for modern compilers to exploit SIMD parallelism. However, state-of-the-art approaches still struggle to handle intricate code patterns, often requiring manual hints or domain-specific expertise. Large language models (LLMs), with their ability to capture intricate patterns, provide a promising solution, yet their effective application in compiler optimizations remains an open challenge due to issues such as hallucinations and a lack of domain-specific reasoning. In this paper, we present VecTrans, a novel framework that leverages LLMs to enhance compiler-based code vectorization. VecTrans first employs compiler analysis to identify potentially vectorizable code regions. It then utilizes an LLM to refactor these regions into patterns that are more amenable to the compilers auto-vectorization. To ensure semantic correctness, VecTrans further integrates a hybrid validation mechanism at the intermediate representation (IR) level. With the above efforts, VecTrans combines the adaptability of LLMs with the precision of compiler vectorization, thereby effectively opening up the vectorization opportunities. experimental results show that among all TSVC functions unvectorizable by GCC, ICC, Clang, and BiSheng Compiler, VecTrans achieves an geomean speedup of 1.77x and successfully vectorizes 24 of 51 test cases. This marks a significant advancement over state-of-the-art approaches while maintaining a cost efficiency of $0.012 per function optimization for LLM API usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03656v1">Client-Side Zero-Shot LLM Inference for Comprehensive In-Browser URL Analysis</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 46 pages , 5 figures
    </div>
    <details class="paper-abstract">
      Malicious websites and phishing URLs pose an ever-increasing cybersecurity risk, with phishing attacks growing by 40% in a single year. Traditional detection approaches rely on machine learning classifiers or rule-based scanners operating in the cloud, but these face significant challenges in generalization, privacy, and evasion by sophisticated threats. In this paper, we propose a novel client-side framework for comprehensive URL analysis that leverages zero-shot inference by a local large language model (LLM) running entirely in-browser. Our system uses a compact LLM (e.g., 3B/8B parameters) via WebLLM to perform reasoning over rich context collected from the target webpage, including static code analysis (JavaScript abstract syntax trees, structure, and code patterns), dynamic sandbox execution results (DOM changes, API calls, and network requests),and visible content. We detail the architecture and methodology of the system, which combines a real browser sandbox (using iframes) resistant to common anti-analysis techniques, with an LLM-based analyzer that assesses potential vulnerabilities and malicious behaviors without any task-specific training (zero-shot). The LLM aggregates evidence from multiple sources (code, execution trace, page content) to classify the URL as benign or malicious and to provide an explanation of the threats or security issues identified. We evaluate our approach on a diverse set of benign and malicious URLs, demonstrating that even a compact client-side model can achieve high detection accuracy and insightful explanations comparable to cloud-based solutions, while operating privately on end-user devices. The results show that client-side LLM inference is a feasible and effective solution to web threat analysis, eliminating the need to send potentially sensitive data to cloud services.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03655v1">Facts are Harder Than Opinions -- A Multilingual, Comparative Analysis of LLM-Based Fact-Checking Reliability</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      The proliferation of misinformation necessitates scalable, automated fact-checking solutions. Yet, current benchmarks often overlook multilingual and topical diversity. This paper introduces a novel, dynamically extensible data set that includes 61,514 claims in multiple languages and topics, extending existing datasets up to 2024. Through a comprehensive evaluation of five prominent Large Language Models (LLMs), including GPT-4o, GPT-3.5 Turbo, LLaMA 3.1, and Mixtral 8x7B, we identify significant performance gaps between different languages and topics. While overall GPT-4o achieves the highest accuracy, it declines to classify 43% of claims. Across all models, factual-sounding claims are misclassified more often than opinions, revealing a key vulnerability. These findings underscore the need for caution and highlight challenges in deploying LLM-based fact-checking systems at scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.04173v2">Quantifying Prediction Consistency Under Fine-Tuning Multiplicity in Tabular LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 International Conference on Machine Learning (ICML), 2025
    </div>
    <details class="paper-abstract">
      Fine-tuning LLMs on tabular classification tasks can lead to the phenomenon of fine-tuning multiplicity where equally well-performing models make conflicting predictions on the same input. Fine-tuning multiplicity can arise due to variations in the training process, e.g., seed, weight initialization, minor changes to training data, etc., raising concerns about the reliability of Tabular LLMs in high-stakes applications such as finance, hiring, education, healthcare. Our work formalizes this unique challenge of fine-tuning multiplicity in Tabular LLMs and proposes a novel measure to quantify the consistency of individual predictions without expensive model retraining. Our measure quantifies a prediction's consistency by analyzing (sampling) the model's local behavior around that input in the embedding space. Interestingly, we show that sampling in the local neighborhood can be leveraged to provide probabilistic guarantees on prediction consistency under a broad class of fine-tuned models, i.e., inputs with sufficiently high local stability (as defined by our measure) also remain consistent across several fine-tuned models with high probability. We perform experiments on multiple real-world datasets to show that our local stability measure preemptively captures consistency under actual multiplicity across several fine-tuned models, outperforming competing measures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.09887v4">OptiBench Meets ReSocratic: Measure and Improve LLMs for Optimization Modeling</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have exhibited their problem-solving abilities in mathematical reasoning. Solving realistic optimization (OPT) problems in application scenarios requires advanced and applied mathematics ability. However, current OPT benchmarks that merely solve linear programming are far from complex realistic situations. In this work, we propose OptiBench, a benchmark for End-to-end optimization problem-solving with human-readable inputs and outputs. OptiBench contains rich optimization problems, including linear and nonlinear programming with or without tabular data, which can comprehensively evaluate LLMs' solving ability. In our benchmark, LLMs are required to call a code solver to provide precise numerical answers. Furthermore, to alleviate the data scarcity for optimization problems, and to bridge the gap between open-source LLMs on a small scale (e.g., Llama-3-8b) and closed-source LLMs (e.g., GPT-4), we further propose a data synthesis method namely ReSocratic. Unlike general data synthesis methods that proceed from questions to answers, \ReSocratic first incrementally synthesizes formatted optimization demonstration with mathematical formulations step by step and then back-translates the generated demonstrations into questions. Based on this, we synthesize the ReSocratic-29k dataset. We further conduct supervised fine-tuning with ReSocratic-29k on multiple open-source models. Experimental results show that ReSocratic-29k significantly improves the performance of open-source models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09429v3">From Intention To Implementation: Automating Biomedical Research via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 The paper involves material for which we have not yet obtained proper copyright permissions
    </div>
    <details class="paper-abstract">
      Conventional biomedical research is increasingly labor-intensive due to the exponential growth of scientific literature and datasets. Artificial intelligence (AI), particularly Large Language Models (LLMs), has the potential to revolutionize this process by automating various steps. Still, significant challenges remain, including the need for multidisciplinary expertise, logicality of experimental design, and performance measurements. This paper introduces BioResearcher, the first end-to-end automated system designed to streamline the entire biomedical research process involving dry lab experiments. BioResearcher employs a modular multi-agent architecture, integrating specialized agents for search, literature processing, experimental design, and programming. By decomposing complex tasks into logically related sub-tasks and utilizing a hierarchical learning approach, BioResearcher effectively addresses the challenges of multidisciplinary requirements and logical complexity. Furthermore, BioResearcher incorporates an LLM-based reviewer for in-process quality control and introduces novel evaluation metrics to assess the quality and automation of experimental protocols. BioResearcher successfully achieves an average execution success rate of 63.07% across eight previously unmet research objectives. The generated protocols averagely outperform typical agent systems by 22.0% on five quality metrics. The system demonstrates significant potential to reduce researchers' workloads and accelerate biomedical discoveries, paving the way for future innovations in automated research systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03610v1">Orak: A Foundational Benchmark for Training and Evaluating LLM Agents on Diverse Video Games</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are reshaping the game industry, particularly with more intelligent and human-preferable game characters. However, existing game benchmarks fall short of practical needs: they lack evaluations of diverse LLM capabilities across various game genres, studies of agentic modules crucial for complex gameplay, and fine-tuning datasets for aligning pre-trained LLMs into gaming agents. To fill these gaps, we present \textbf{\benchname{}}, a foundational benchmark designed to train and evaluate LLM agents across diverse real-world video games. Unlike existing benchmarks, Orak includes 12 popular video games spanning all major genres, enabling comprehensive studies of LLM capabilities and agentic modules essential for intricate game scenarios. To support consistent evaluation of LLMs, we introduce a plug-and-play interface based on Model Context Protocol (MCP) that enables LLMs to seamlessly connect with games and manipulate agentic modules. Additionally, we propose a fine-tuning dataset, consisting of LLM gameplay trajectories across diverse game genres. Orak offers a comprehensive evaluation framework, encompassing general game score leaderboards, LLM battle arenas, and in-depth analyses of visual input state, agentic strategies, and fine-tuning effects, establishing a foundation towards building generic gaming agents. Code is available at https://github.com/krafton-ai/Orak.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.01056v2">MCP-Zero: Proactive Toolchain Construction for LLM Agents from Scratch</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Function-calling has enabled large language models (LLMs) to act as tool-using agents, but injecting thousands of tool schemas into the prompt is costly and error-prone. We introduce MCP-Zero, a proactive agent framework that lets the LLM itself decide when and which external tools to retrieve, thereby assembling a task-specific toolchain from scratch. The framework is built upon three components: (1) Proactive Tool Request, where the model emits a structured $\left<\operatorname{tool\_assistant}\right>$ block that explicitly specifies the desired server and task; (2) Hierarchical Vector Routing, a coarse-to-fine retrieval algorithm that first selects candidate servers and then ranks tools within each server based on the semantic similarity; (3) Iterative Proactive Invocation, enabling multi-round, cross-domain toolchain construction with minimal context overhead, and allowing the model to iteratively revise its request when the returned tools are insufficient. To evaluate our approach we also compile MCP-tools, a retrieval dataset comprising 308 MCP servers and 2,797 tools extracted from the official Model-Context-Protocol repository and normalized into a unified JSON schema. Experiments show that MCP-Zero (i) effectively addresses the context overhead problem of existing methods and accurately selects the correct tool from a pool of nearly 3,000 candidates (248.1k tokens); (ii) reduces token consumption by 98\% on the APIbank while maintaining high accuracy; and (iii) supports multi-turn tool invocation with consistent accuracy across rounds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.14884v2">Polar Sparsity: High Throughput Batched LLM Inferencing with Scalable Contextual Sparsity</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Accelerating large language model (LLM) inference is critical for real-world deployments requiring high throughput and low latency. Contextual sparsity, where each token dynamically activates only a small subset of the model parameters, shows promise but does not scale to large batch sizes due to union of active neurons quickly approaching dense computation. We introduce Polar Sparsity, highlighting a key shift in sparsity importance from MLP to Attention layers as we scale batch size and sequence length. While MLP layers become more compute-efficient under batching, their sparsity vanishes. In contrast, attention becomes increasingly more expensive at scale, while their head sparsity remains stable and batch-invariant. We develop hardware-efficient, sparsity-aware GPU kernels for selective MLP and Attention computations, delivering up to \(2.2\times\) end-to-end speedups for models like OPT, LLaMA-2 \& 3, across various batch sizes and sequence lengths without compromising accuracy. To our knowledge, this is the first work to demonstrate that contextual sparsity can scale effectively to large batch sizes, delivering substantial inference acceleration with minimal changes, making Polar Sparsity practical for large-scale, high-throughput LLM deployment systems. Our code is available at: https://github.com/susavlsh10/Polar-Sparsity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02442v2">Should LLM Safety Be More Than Refusing Harmful Instructions?</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      This paper presents a systematic evaluation of Large Language Models' (LLMs) behavior on long-tail distributed (encrypted) texts and their safety implications. We introduce a two-dimensional framework for assessing LLM safety: (1) instruction refusal-the ability to reject harmful obfuscated instructions, and (2) generation safety-the suppression of generating harmful responses. Through comprehensive experiments, we demonstrate that models that possess capabilities to decrypt ciphers may be susceptible to mismatched-generalization attacks: their safety mechanisms fail on at least one safety dimension, leading to unsafe responses or over-refusal. Based on these findings, we evaluate a number of pre-LLM and post-LLM safeguards and discuss their strengths and limitations. This work contributes to understanding the safety of LLM in long-tail text scenarios and provides directions for developing robust safety mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.02965v2">PC-MoE: Memory-Efficient and Privacy-Preserving Collaborative Training for Mixture-of-Experts LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 20 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) has been gaining popularity due to its successful adaptation to large language models (LLMs). In this work, we introduce Privacy-preserving Collaborative Mixture-of-Experts (PC-MoE), which leverages the sparsity of the MoE architecture for memory-efficient decentralized collaborative LLM training, enabling multiple parties with limited GPU-memory and data resources to collectively train more capable LLMs than they could achieve individually. At the same time, this approach protects training data privacy of each participant by keeping training data, as well as parts of the forward pass signal and gradients locally within each party. By design, PC-MoE synergistically combines the strengths of distributed computation with strong confidentiality assurances. Unlike most privacy-preserving schemes, which pay for confidentiality with lower task accuracy, our framework breaks that trade-off: across seven popular LLM benchmarks, it almost matches (and sometimes exceeds) the performance and convergence rate of a fully centralized model, enjoys near 70% peak GPU RAM reduction, while being fully robust against reconstruction attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03585v1">Improving LLM-Based Fault Localization with External Memory and Project Context</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 12 Pages, 7 figures
    </div>
    <details class="paper-abstract">
      Fault localization, the process of identifying the software components responsible for failures, is essential but often time-consuming. Recent advances in Large Language Models (LLMs) have enabled fault localization without extensive defect datasets or model fine-tuning. However, existing LLM-based methods rely only on general LLM capabilities and lack integration of project-specific knowledge, resulting in limited effectiveness, especially for complex software. We introduce MemFL, a novel approach that enhances LLM-based fault localization by integrating project-specific knowledge via external memory. This memory includes static summaries of the project and dynamic, iterative debugging insights gathered from previous attempts. By leveraging external memory, MemFL simplifies debugging into three streamlined steps, significantly improving efficiency and accuracy. Iterative refinement through dynamic memory further enhances reasoning quality over time. Evaluated on the Defects4J benchmark, MemFL using GPT-4o-mini localized 12.7% more bugs than current LLM-based methods, achieving this improvement with just 21% of the execution time (17.4 seconds per bug) and 33% of the API cost (0.0033 dollars per bug). On complex projects, MemFL's advantage increased to 27.6%. Additionally, MemFL with GPT-4.1-mini outperformed existing methods by 24.4%, requiring only 24.7 seconds and 0.0094 dollars per bug. MemFL thus demonstrates significant improvements by effectively incorporating project-specific knowledge into LLM-based fault localization, delivering high accuracy with reduced time and cost.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20875v2">Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 27 pages, 6 figures, 16 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are predominantly evaluated on Standard American English (SAE), often overlooking the diversity of global English varieties. This narrow focus may raise fairness concerns as degraded performance on non-standard varieties can lead to unequal benefits for users worldwide. Therefore, it is critical to extensively evaluate the linguistic robustness of LLMs on multiple non-standard English varieties. We introduce Trans-EnV, a framework that automatically transforms SAE datasets into multiple English varieties to evaluate the linguistic robustness. Our framework combines (1) linguistics expert knowledge to curate variety-specific features and transformation guidelines from linguistic literature and corpora, and (2) LLM-based transformations to ensure both linguistic validity and scalability. Using Trans-EnV, we transform six benchmark datasets into 38 English varieties and evaluate seven state-of-the-art LLMs. Our results reveal significant performance disparities, with accuracy decreasing by up to 46.3% on non-standard varieties. These findings highlight the importance of comprehensive linguistic robustness evaluation across diverse English varieties. Each construction of Trans-EnV was validated through rigorous statistical testing and consultation with a researcher in the field of second language acquisition, ensuring its linguistic validity. Our code and datasets are publicly available at https://github.com/jiyounglee-0523/TransEnV and https://huggingface.co/collections/jiyounglee0523/transenv-681eadb3c0c8cf363b363fb1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03543v1">CogniPair: From LLM Chatbots to Conscious AI Agents -- GNWT-Based Multi-Agent Digital Twins for Social Pairing -- Dating & Hiring Applications</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Current large language model (LLM) agents lack authentic human psychological processes necessary for genuine digital twins and social AI applications. To address this limitation, we present a computational implementation of Global Workspace Theory (GNWT) that integrates human cognitive architecture principles into LLM agents, creating specialized sub-agents for emotion, memory, social norms, planning, and goal-tracking coordinated through a global workspace mechanism. However, authentic digital twins require accurate personality initialization. We therefore develop a novel adventure-based personality test that evaluates true personality through behavioral choices within interactive scenarios, bypassing self-presentation bias found in traditional assessments. Building on these innovations, our CogniPair platform enables digital twins to engage in realistic simulated dating interactions and job interviews before real encounters, providing bidirectional cultural fit assessment for both romantic compatibility and workplace matching. Validation using 551 GNWT-Agents and Columbia University Speed Dating dataset demonstrates 72% correlation with human attraction patterns, 77.8% match prediction accuracy, and 74% agreement in human validation studies. This work advances psychological authenticity in LLM agents and establishes a foundation for intelligent dating platforms and HR technology solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.00095v3">ClinBench-HPB: A Clinical Benchmark for Evaluating LLMs in Hepato-Pancreato-Biliary Diseases</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Hepato-pancreato-biliary (HPB) disorders represent a global public health challenge due to their high morbidity and mortality. Although large language models (LLMs) have shown promising performance in general medical question-answering tasks, the current evaluation benchmarks are mostly derived from standardized examinations or manually designed questions, lacking HPB coverage and clinical cases. To address these issues, we systematically eatablish an HPB disease evaluation benchmark comprising 3,535 closed-ended multiple-choice questions and 337 open-ended real diagnosis cases, which encompasses all the 33 main categories and 465 subcategories of HPB diseases defined in the International Statistical Classification of Diseases, 10th Revision (ICD-10). The multiple-choice questions are curated from public datasets and synthesized data, and the clinical cases are collected from prestigious medical journals, case-sharing platforms, and collaborating hospitals. By evalauting commercial and open-source general and medical LLMs on our established benchmark, namely ClinBench-HBP, we find that while commercial LLMs perform competently on medical exam questions, they exhibit substantial performance degradation on HPB diagnosis tasks, especially on complex, inpatient clinical cases. Those medical LLMs also show limited generalizability to HPB diseases. Our results reveal the critical limitations of current LLMs in the domain of HPB diseases, underscoring the imperative need for future medical LLMs to handle real, complex clinical diagnostics rather than simple medical exam questions. The benchmark will be released at https://clinbench-hpb.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20197v3">Enhancing the Robustness of LLM-Generated Code: Empirical Study and Framework</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Ensuring the robustness of code generated by large language models (LLMs) is crucial for real-world reliability. However, existing evaluations predominantly focus on correctness, often neglecting key robustness concerns such as missing input validation and insufficient error handling. In this paper, we present the first empirical study on the robustness of LLM-generated code. We introduce novel robustness metrics and analyze four state-of-the-art code LLMs, revealing that, on average, 43.1% of their generated code is less robust than human-written counterparts. Notably, over 90% of robustness deficiencies stem from missing conditional checks, with 70% of these omissions occurring in the first line of code. Additionally, in 69% of cases where a conditional statement is necessary but absent, the "if" token still ranks third or higher in the model's predicted token probabilities, indicating an implicit recognition of control structures. Building on these findings, we propose RobGen, a framework designed to enhance code robustness without requiring model retraining. RobGen leverages two model-agnostic techniques: RobGen-Adj, which dynamically adjusts token probabilities during decoding to encourage the inclusion of control structures, and RobGen-Ins, which improves generated code by inserting missing conditionals after generation. Experimental results demonstrate that RobGen reduces the proportion of less robust model-generated code by 20.0%, significantly enhancing code reliability across diverse tasks. As a lightweight and adaptable solution, RobGen effectively mitigates robustness challenges in LLM-generated code. All code and data are available at https://github.com/SYSUSELab/RobGen.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20099v2">Defensive Prompt Patch: A Robust and Interpretable Defense of LLMs against Jailbreak Attacks</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Safety, security, and compliance are essential requirements when aligning large language models (LLMs). However, many seemingly aligned LLMs are soon shown to be susceptible to jailbreak attacks. These attacks aim to circumvent the models' safety guardrails and security mechanisms by introducing jailbreak prompts into malicious queries. In response to these challenges, this paper introduces Defensive Prompt Patch (DPP), a novel prompt-based defense mechanism specifically designed to protect LLMs against such sophisticated jailbreak strategies. Unlike previous approaches, which have often compromised the utility of the model for the sake of safety, DPP is designed to achieve a minimal Attack Success Rate (ASR) while preserving the high utility of LLMs. Our method uses strategically designed interpretable suffix prompts that effectively thwart a wide range of standard and adaptive jailbreak techniques. Empirical results conducted on LLAMA-2-7B-Chat and Mistral-7B-Instruct-v0.2 models demonstrate the robustness and adaptability of DPP, showing significant reductions in ASR with negligible impact on utility. Our approach not only outperforms existing defense strategies in balancing safety and functionality, but also provides a scalable and interpretable solution applicable to various LLM platforms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03504v1">Beyond C/C++: Probabilistic and LLM Methods for Next-Generation Software Reverse Engineering</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      This proposal discusses the growing challenges in reverse engineering modern software binaries, particularly those compiled from newer system programming languages such as Rust, Go, and Mojo. Traditional reverse engineering techniques, developed with a focus on C and C++, fall short when applied to these newer languages due to their reliance on outdated heuristics and failure to fully utilize the rich semantic information embedded in binary programs. These challenges are exacerbated by the limitations of current data-driven methods, which are susceptible to generating inaccurate results, commonly referred to as hallucinations. To overcome these limitations, we propose a novel approach that integrates probabilistic binary analysis with fine-tuned large language models (LLMs). Our method systematically models the uncertainties inherent in reverse engineering, enabling more accurate reasoning about incomplete or ambiguous information. By incorporating LLMs, we extend the analysis beyond traditional heuristics, allowing for more creative and context-aware inferences, particularly for binaries from diverse programming languages. This hybrid approach not only enhances the robustness and accuracy of reverse engineering efforts but also offers a scalable solution adaptable to the rapidly evolving landscape of software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.03483v1">APT: Improving Specialist LLM Performance with Weakness Case Acquisition and Iterative Preference Training</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 ACL2025 Findings
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) often require domain-specific fine-tuning to address targeted tasks, which risks degrading their general capabilities. Maintaining a balance between domain-specific enhancements and general model utility is a key challenge. This paper proposes a novel approach named APT (Weakness Case Acquisition and Iterative Preference Training) to enhance domain-specific performance with self-generated dis-preferred weakness data (bad cases and similar cases). APT uniquely focuses on training the model using only those samples where errors occur, alongside a small, similar set of samples retrieved for this purpose. This targeted training minimizes interference with the model's existing knowledge base, effectively retaining generic capabilities. Experimental results on the LLama-2 and Mistral-V0.3 models across various benchmarks demonstrate that APT ensures no reduction in generic capacity and achieves superior performance on downstream tasks compared to various existing methods. This validates our method as an effective strategy for enhancing domain-specific capabilities without sacrificing the model's broader applicability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18173v2">UIO-LLMs: Unbiased Incremental Optimization for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 The experimental results of the paper require further validation
    </div>
    <details class="paper-abstract">
      Managing long texts is challenging for large language models (LLMs) due to limited context window sizes. This study introduces UIO-LLMs, an unbiased incremental optimization approach for memory-enhanced transformers under long-context settings. We initially conceptualize the process as a streamlined encoder-decoder framework where the weights-shared encoder and decoder respectively encapsulate a context segment into memories and leverage these memories to predict outputs of the subsequent segment. Subsequently, by treating our memory-enhanced transformers as fully-connected recurrent neural networks (RNNs), we refine the training process using the Truncated Backpropagation Through Time (TBPTT) algorithm, which incorporates innovative incremental optimization techniques. These techniques not only diminish time complexity but also address the bias in gradient computation through an unbiased optimization process. UIO-LLMs successfully handle long context, such as extending the context window of Llama2-7b-chat from 4K to 100K tokens with minimal 2% additional parameters, while keeping the inference cost nearly linear as context length increases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.20730v2">What LLMs Miss in Recommendations: Bridging the Gap with Retrieval-Augmented Collaborative Signals</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      User-item interactions contain rich collaborative signals that form the backbone of many successful recommender systems. While recent work has explored the use of large language models (LLMs) for recommendation, it remains unclear whether LLMs can effectively reason over this type of collaborative information. In this paper, we conduct a systematic comparison between LLMs and classical matrix factorization (MF) models to assess LLMs' ability to leverage user-item interaction data. We further introduce a simple retrieval-augmented generation (RAG) method that enhances LLMs by grounding their predictions in structured interaction data. Our experiments reveal that current LLMs often fall short in capturing collaborative patterns inherent to MF models, but that our RAG-based approach substantially improves recommendation quality-highlighting a promising direction for future LLM-based recommenders.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01821v2">On the Power of Context-Enhanced Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 77 pages, 18 figures; ICML 2025 Main Conference
    </div>
    <details class="paper-abstract">
      We formalize a new concept for LLMs, context-enhanced learning. It involves standard gradient-based learning on text except that the context is enhanced with additional data on which no auto-regressive gradients are computed. This setting is a gradient-based analog of usual in-context learning (ICL) and appears in some recent works. Using a multi-step reasoning task, we prove in a simplified setting that context-enhanced learning can be exponentially more sample-efficient than standard learning when the model is capable of ICL. At a mechanistic level, we find that the benefit of context-enhancement arises from a more accurate gradient learning signal. We also experimentally demonstrate that it appears hard to detect or recover learning materials that were used in the context during training. This may have implications for data security as well as copyright.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04482v1">Understanding and Meeting Practitioner Needs When Measuring Representational Harms Caused by LLM-Based Systems</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Findings of the Association for Computational Linguistics: ACL 2025
    </div>
    <details class="paper-abstract">
      The NLP research community has made publicly available numerous instruments for measuring representational harms caused by large language model (LLM)-based systems. These instruments have taken the form of datasets, metrics, tools, and more. In this paper, we examine the extent to which such instruments meet the needs of practitioners tasked with evaluating LLM-based systems. Via semi-structured interviews with 12 such practitioners, we find that practitioners are often unable to use publicly available instruments for measuring representational harms. We identify two types of challenges. In some cases, instruments are not useful because they do not meaningfully measure what practitioners seek to measure or are otherwise misaligned with practitioner needs. In other cases, instruments - even useful instruments - are not used by practitioners due to practical and institutional barriers impeding their uptake. Drawing on measurement theory and pragmatic measurement, we provide recommendations for addressing these challenges to better meet practitioner needs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04481v1">CogMath: Assessing LLMs' Authentic Mathematical Ability from a Human Cognitive Perspective</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) show promise in solving complex mathematical tasks, existing evaluation paradigms rely solely on a coarse measure of overall answer accuracy, which are insufficient for assessing their authentic capabilities. In this paper, we propose \textbf{CogMath}, which comprehensively assesses LLMs' mathematical abilities through the lens of human cognition. Specifically, inspired by psychological theories, CogMath formalizes human reasoning process into 3 stages: \emph{problem comprehension}, \emph{problem solving}, and \emph{solution summarization}. Within these stages, we investigate perspectives such as numerical calculation, knowledge, and counterfactuals, and design a total of 9 fine-grained evaluation dimensions. In each dimension, we develop an ``\emph{Inquiry}-\emph{Judge}-\emph{Reference}'' multi-agent system to generate inquiries that assess LLMs' mastery from this dimension. An LLM is considered to truly master a problem only when excelling in all inquiries from the 9 dimensions. By applying CogMath on three benchmarks, we reveal that the mathematical capabilities of 7 mainstream LLMs are overestimated by 30\%-40\%. Moreover, we locate their strengths and weaknesses across specific stages/dimensions, offering in-depth insights to further enhance their reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04478v1">Matching Markets Meet LLMs: Algorithmic Reasoning with Ranked Preferences</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has driven progress in reasoning tasks -- from program synthesis to scientific hypothesis generation -- yet their ability to handle ranked preferences and structured algorithms in combinatorial domains remains underexplored. We study matching markets, a core framework behind applications like resource allocation and ride-sharing, which require reconciling individual ranked preferences to ensure stable outcomes. We evaluate several state-of-the-art models on a hierarchy of preference-based reasoning tasks -- ranging from stable-matching generation to instability detection, instability resolution, and fine-grained preference queries -- to systematically expose their logical and algorithmic limitations in handling ranked inputs. Surprisingly, even top-performing models with advanced reasoning struggle to resolve instability in large markets, often failing to identify blocking pairs or execute algorithms iteratively. We further show that parameter-efficient fine-tuning (LoRA) significantly improves performance in small markets, but fails to bring about a similar improvement on large instances, suggesting the need for more sophisticated strategies to improve LLMs' reasoning with larger-context inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04450v1">Learning to Diagnose Privately: DP-Powered LLMs for Radiology Report Classification</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 19 pages, 5 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Purpose: This study proposes a framework for fine-tuning large language models (LLMs) with differential privacy (DP) to perform multi-abnormality classification on radiology report text. By injecting calibrated noise during fine-tuning, the framework seeks to mitigate the privacy risks associated with sensitive patient data and protect against data leakage while maintaining classification performance. Materials and Methods: We used 50,232 radiology reports from the publicly available MIMIC-CXR chest radiography and CT-RATE computed tomography datasets, collected between 2011 and 2019. Fine-tuning of LLMs was conducted to classify 14 labels from MIMIC-CXR dataset, and 18 labels from CT-RATE dataset using Differentially Private Low-Rank Adaptation (DP-LoRA) in high and moderate privacy regimes (across a range of privacy budgets = {0.01, 0.1, 1.0, 10.0}). Model performance was evaluated using weighted F1 score across three model architectures: BERT-medium, BERT-small, and ALBERT-base. Statistical analyses compared model performance across different privacy levels to quantify the privacy-utility trade-off. Results: We observe a clear privacy-utility trade-off through our experiments on 2 different datasets and 3 different models. Under moderate privacy guarantees the DP fine-tuned models achieved comparable weighted F1 scores of 0.88 on MIMIC-CXR and 0.59 on CT-RATE, compared to non-private LoRA baselines of 0.90 and 0.78, respectively. Conclusion: Differentially private fine-tuning using LoRA enables effective and privacy-preserving multi-abnormality classification from radiology reports, addressing a key challenge in fine-tuning LLMs on sensitive medical data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04427v1">Plugging Schema Graph into Multi-Table QA: A Human-Guided Framework for Reducing LLM Reliance</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Submitted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown promise in table Question Answering (Table QA). However, extending these capabilities to multi-table QA remains challenging due to unreliable schema linking across complex tables. Existing methods based on semantic similarity work well only on simplified hand-crafted datasets and struggle to handle complex, real-world scenarios with numerous and diverse columns. To address this, we propose a graph-based framework that leverages human-curated relational knowledge to explicitly encode schema links and join paths. Given a natural language query, our method searches this graph to construct interpretable reasoning chains, aided by pruning and sub-path merging strategies to enhance efficiency and coherence. Experiments on both standard benchmarks and a realistic, large-scale dataset demonstrate the effectiveness of our approach. To our knowledge, this is the first multi-table QA system applied to truly complex industrial tabular data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04418v1">Characterizing Multi-Hunk Patches: Divergence, Proximity, and LLM Repair Challenges</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Multi-hunk bugs, where fixes span disjoint regions of code, are common in practice, yet remain underrepresented in automated repair. Existing techniques and benchmarks pre-dominantly target single-hunk scenarios, overlooking the added complexity of coordinating semantically related changes across the codebase. In this work, we characterize HUNK4J, a dataset of multi-hunk patches derived from 372 real-world defects. We propose hunk divergence, a metric that quantifies the variation among edits in a patch by capturing lexical, structural, and file-level differences, while incorporating the number of hunks involved. We further define spatial proximity, a classification that models how hunks are spatially distributed across the program hierarchy. Our empirical study spanning six LLMs reveals that model success rates decline with increased divergence and spatial dispersion. Notably, when using the LLM alone, no model succeeds in the most dispersed Fragment class. These findings highlight a critical gap in LLM capabilities and motivate divergence-aware repair strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.17135v3">When can isotropy help adapt LLMs' next word prediction to numerical domains?</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Recent studies have shown that vector representations of contextual embeddings learned by pre-trained large language models (LLMs) are effective in various downstream tasks in numerical domains. Despite their significant benefits, the tendency of LLMs to hallucinate in such domains can have severe consequences in applications such as energy, nature, finance, healthcare, retail and transportation, among others. To guarantee prediction reliability and accuracy in numerical domains, it is necessary to open the black-box and provide performance guarantees through explanation. However, there is little theoretical understanding of when pre-trained language models help solve numeric downstream tasks. This paper seeks to bridge this gap by understanding when the next-word prediction capability of LLMs can be adapted to numerical domains through a novel analysis based on the concept of isotropy in the contextual embedding space. Specifically, we consider a log-linear model for LLMs in which numeric data can be predicted from its context through a network with softmax in the output layer of LLMs (i.e., language model head in self-attention). We demonstrate that, in order to achieve state-of-the-art performance in numerical domains, the hidden representations of the LLM embeddings must possess a structure that accounts for the shift-invariance of the softmax function. By formulating a gradient structure of self-attention in pre-trained models, we show how the isotropic property of LLM embeddings in contextual embedding space preserves the underlying structure of representations, thereby resolving the shift-invariance problem and providing a performance guarantee. Experiments show that different characteristics of numeric data and model architecture could have different impacts on isotropy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04405v1">MedAgentGym: Training LLM Agents for Code-Based Medical Reasoning at Scale</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      We introduce MedAgentGYM, the first publicly available training environment designed to enhance coding-based medical reasoning capabilities in large language model (LLM) agents. MedAgentGYM comprises 72,413 task instances across 129 categories derived from authentic real-world biomedical scenarios. Tasks are encapsulated within executable coding environments, each featuring detailed task descriptions, interactive feedback mechanisms, verifiable ground-truth annotations, and scalable training trajectory generation. Extensive benchmarking of over 30 LLMs reveals a notable performance disparity between commercial API-based models and open-source counterparts. Leveraging MedAgentGYM, Med-Copilot-7B achieves substantial performance gains through supervised fine-tuning (+36.44%) and continued reinforcement learning (+42.47%), emerging as an affordable and privacy-preserving alternative competitive with gpt-4o. By offering both a comprehensive benchmark and accessible, expandable training resources within unified execution environments, MedAgentGYM delivers an integrated platform to develop LLM-based coding assistants for advanced biomedical research and practice.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15652v3">Empowering LLMs with Logical Reasoning: A Comprehensive Survey</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Accepted by IJCAI 2025 (Survey Track)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved remarkable successes on various tasks. However, recent studies have found that there are still significant challenges to the logical reasoning abilities of LLMs, which can be categorized into the following two aspects: (1) Logical question answering: LLMs often fail to generate the correct answer within a complex logical problem which requires sophisticated deductive, inductive or abductive reasoning given a collection of premises and constrains. (2) Logical consistency: LLMs are prone to producing responses contradicting themselves across different questions. For example, a state-of-the-art question-answering LLM Macaw, answers Yes to both questions Is a magpie a bird? and Does a bird have wings? but answers No to Does a magpie have wings?. To facilitate this research direction, we comprehensively investigate the most cutting-edge methods and propose a detailed taxonomy. Specifically, to accurately answer complex logic questions, previous methods can be categorized based on reliance on external solvers, prompts, and fine-tuning. To avoid logical contradictions, we discuss concepts and solutions of various logical consistencies, including implication, negation, transitivity, factuality consistencies, and their composites. In addition, we review commonly used benchmark datasets and evaluation metrics, and discuss promising research directions, such as extending to modal logic to account for uncertainty and developing efficient algorithms that simultaneously satisfy multiple logical consistencies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23884v3">Failure Modes of LLMs for Causal Reasoning on Narratives</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      In this work, we investigate the causal reasoning abilities of large language models (LLMs) through the representative problem of inferring causal relationships from narratives. We find that even state-of-the-art language models rely on unreliable shortcuts, both in terms of the narrative presentation and their parametric knowledge. For example, LLMs tend to determine causal relationships based on the topological ordering of events (i.e., earlier events cause later ones), resulting in lower performance whenever events are not narrated in their exact causal order. Similarly, we demonstrate that LLMs struggle with long-term causal reasoning and often fail when the narratives are long and contain many events. Additionally, we show LLMs appear to rely heavily on their parametric knowledge at the expense of reasoning over the provided narrative. This degrades their abilities whenever the narrative opposes parametric knowledge. We extensively validate these failure modes through carefully controlled synthetic experiments, as well as evaluations on real-world narratives. Finally, we observe that explicitly generating a causal graph generally improves performance while naive chain-of-thought is ineffective. Collectively, our results distill precise failure modes of current state-of-the-art models and can pave the way for future techniques to enhance causal reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07987v3">Universal Adversarial Attack on Aligned Multimodal LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Added benchmarks, baselines, author, appendix
    </div>
    <details class="paper-abstract">
      We propose a universal adversarial attack on multimodal Large Language Models (LLMs) that leverages a single optimized image to override alignment safeguards across diverse queries and even multiple models. By backpropagating through the vision encoder and language head, we craft a synthetic image that forces the model to respond with a targeted phrase (e.g., "Sure, here it is") or otherwise unsafe content -- even for harmful prompts. In experiments on the SafeBench and MM-SafetyBench benchmarks, our method achieves higher attack success rates than existing baselines, including text-only universal prompts (e.g., up to 81% on certain models). We further demonstrate cross-model universality by training on several multimodal LLMs simultaneously. Additionally, a multi-answer variant of our approach produces more natural-sounding (yet still malicious) responses. These findings underscore critical vulnerabilities in current multimodal alignment and call for more robust adversarial defenses. We will release code and datasets under the Apache-2.0 license. Warning: some content generated by Multimodal LLMs in this paper may be offensive.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04344v1">GEM: Empowering LLM for both Embedding Generation and Language Understanding</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large decoder-only language models (LLMs) have achieved remarkable success in generation and reasoning tasks, where they generate text responses given instructions. However, many applications, e.g., retrieval augmented generation (RAG), still rely on separate embedding models to generate text embeddings, which can complicate the system and introduce discrepancies in understanding of the query between the embedding model and LLMs. To address this limitation, we propose a simple self-supervised approach, Generative Embedding large language Model (GEM), that enables any large decoder-only LLM to generate high-quality text embeddings while maintaining its original text generation and reasoning capabilities. Our method inserts new special token(s) into a text body, and generates summarization embedding of the text by manipulating the attention mask. This method could be easily integrated into post-training or fine tuning stages of any existing LLMs. We demonstrate the effectiveness of our approach by applying it to two popular LLM families, ranging from 1B to 8B parameters, and evaluating the transformed models on both text embedding benchmarks (MTEB) and NLP benchmarks (MMLU). The results show that our proposed method significantly improves the original LLMs on MTEB while having a minimal impact on MMLU. Our strong results indicate that our approach can empower LLMs with state-of-the-art text embedding capabilities while maintaining their original NLP performance
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04290v1">Interpretable LLMs for Credit Risk: A Systematic Review and Taxonomy</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 20 pages, 6 figures, preprint under review at Information Processing & Management
    </div>
    <details class="paper-abstract">
      Large Language Models (LLM), which have developed in recent years, enable credit risk assessment through the analysis of financial texts such as analyst reports and corporate disclosures. This paper presents the first systematic review and taxonomy focusing on LLMbased approaches in credit risk estimation. We determined the basic model architectures by selecting 60 relevant papers published between 2020-2025 with the PRISMA research strategy. And we examined the data used for scenarios such as credit default prediction and risk analysis. Since the main focus of the paper is interpretability, we classify concepts such as explainability mechanisms, chain of thought prompts and natural language justifications for LLM-based credit models. The taxonomy organizes the literature under four main headings: model architectures, data types, explainability mechanisms and application areas. Based on this analysis, we highlight the main future trends and research gaps for LLM-based credit scoring systems. This paper aims to be a reference paper for artificial intelligence and financial researchers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04282v1">DrSR: LLM based Scientific Equation Discovery with Dual Reasoning from Data and Experience</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Symbolic regression is a fundamental tool for discovering interpretable mathematical expressions from data, with broad applications across scientific and engineering domains. Recently, large language models (LLMs) have demonstrated strong performance in this task, leveraging embedded scientific priors and reasoning capabilities to surpass traditional methods. However, existing LLM-based approaches, such as LLM-SR, often over-rely on internal priors, lacking explicit data understanding and systematic reflection during equation generation. To address these limitations, we propose DrSR (Dual Reasoning Symbolic Regression), a framework that combines data-driven insight with reflective learning to enhance both robustness and discovery capability. Specifically, DrSR guides LLMs to analyze structural relationships (e.g., monotonicity, nonlinearity, and correlation) within the data to generate structured descriptions. Simultaneously, it monitors equation performance and establishes a feedback loop to refine subsequent generations. By integrating data understanding and generation reflection in a closed loop, DrSR enables more efficient exploration of the symbolic expression space. Experiments across interdisciplinary datasets in physics, chemistry, biology, and materials science demonstrate that DrSR substantially improves the valid equation rate and consistently outperforms both classical and recent LLM-based methods in terms of accuracy, generalization, and search efficiency. These results underscore its potential for scientific equation discovery.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05414v1">SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 Project website with demo videos: https://zijuncui02.github.io/SAVVY/
    </div>
    <details class="paper-abstract">
      3D spatial reasoning in dynamic, audio-visual environments is a cornerstone of human cognition yet remains largely unexplored by existing Audio-Visual Large Language Models (AV-LLMs) and benchmarks, which predominantly focus on static or 2D scenes. We introduce SAVVY-Bench, the first benchmark for 3D spatial reasoning in dynamic scenes with synchronized spatial audio. SAVVY-Bench is comprised of thousands of relationships involving static and moving objects, and requires fine-grained temporal grounding, consistent 3D localization, and multi-modal annotation. To tackle this challenge, we propose SAVVY, a novel training-free reasoning pipeline that consists of two stages: (i) Egocentric Spatial Tracks Estimation, which leverages AV-LLMs as well as other audio-visual methods to track the trajectories of key objects related to the query using both visual and spatial audio cues, and (ii) Dynamic Global Map Construction, which aggregates multi-modal queried object trajectories and converts them into a unified global dynamic map. Using the constructed map, a final QA answer is obtained through a coordinate transformation that aligns the global map with the queried viewpoint. Empirical evaluation demonstrates that SAVVY substantially enhances performance of state-of-the-art AV-LLMs, setting a new standard and stage for approaching dynamic 3D spatial reasoning in AV-LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05413v1">SmoothRot: Combining Channel-Wise Scaling and Rotation for Quantization-Friendly LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 6 pages, 3 figures, 5 tables. Submitted to the IEEE SMC 2025 conference
    </div>
    <details class="paper-abstract">
      We present SmoothRot, a novel post-training quantization technique to enhance the efficiency of 4-bit quantization in Large Language Models (LLMs). SmoothRot addresses the critical challenge of massive activation outliers, by integrating channel-wise scaling with Hadamard transformations. Our technique effectively transforms extreme outliers into quantization-friendly activations, significantly improving quantization accuracy. Experiments conducted on popular LLMs (LLaMA2 7B, LLaMA3.1 8B, and Mistral 7B) demonstrate that SmoothRot consistently reduces the performance gap between quantized and FP16 models by approximately 10-30\% across language generation and zero-shot reasoning tasks, without introducing additional inference latency. Code is available at https://github.com/czakop/smoothrot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.05410v1">Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 14 pages,7 figures
    </div>
    <details class="paper-abstract">
      Recent advances in Large Language Models (LLMs) have highlighted the critical importance of extending context length, yet the quadratic complexity of attention mechanisms poses significant challenges for efficient long-context modeling. KV cache compression has emerged as a key approach to address this challenge. Through extensive empirical analysis, we reveal a fundamental yet previously overlooked asymmetry in KV caches: while adjacent keys receive similar attention weights (local homogeneity), adjacent values demonstrate distinct heterogeneous distributions. This key-value asymmetry reveals a critical limitation in existing compression methods that treat keys and values uniformly. To address the limitation, we propose a training-free compression framework (AsymKV) that combines homogeneity-based key merging with a mathematically proven lossless value compression. Extensive experiments demonstrate that AsymKV consistently outperforms existing long-context methods across various tasks and base models. For example, on LLaMA3.1-8B, AsymKV achieves an average score of 43.95 on LongBench, surpassing SOTA methods like H$_2$O (38.89) by a large margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04202v1">TracLLM: A Generic Framework for Attributing Long Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 To appear in USENIX Security Symposium 2025. The code and data are at: https://github.com/Wang-Yanting/TracLLM
    </div>
    <details class="paper-abstract">
      Long context large language models (LLMs) are deployed in many real-world applications such as RAG, agent, and broad LLM-integrated applications. Given an instruction and a long context (e.g., documents, PDF files, webpages), a long context LLM can generate an output grounded in the provided context, aiming to provide more accurate, up-to-date, and verifiable outputs while reducing hallucinations and unsupported claims. This raises a research question: how to pinpoint the texts (e.g., sentences, passages, or paragraphs) in the context that contribute most to or are responsible for the generated output by an LLM? This process, which we call context traceback, has various real-world applications, such as 1) debugging LLM-based systems, 2) conducting post-attack forensic analysis for attacks (e.g., prompt injection attack, knowledge corruption attacks) to an LLM, and 3) highlighting knowledge sources to enhance the trust of users towards outputs generated by LLMs. When applied to context traceback for long context LLMs, existing feature attribution methods such as Shapley have sub-optimal performance and/or incur a large computational cost. In this work, we develop TracLLM, the first generic context traceback framework tailored to long context LLMs. Our framework can improve the effectiveness and efficiency of existing feature attribution methods. To improve the efficiency, we develop an informed search based algorithm in TracLLM. We also develop contribution score ensemble/denoising techniques to improve the accuracy of TracLLM. Our evaluation results show TracLLM can effectively identify texts in a long context that lead to the output of an LLM. Our code and data are at: https://github.com/Wang-Yanting/TracLLM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04185v1">R-Search: Empowering LLM Reasoning with Search via Multi-Reward Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 16 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have notably progressed in multi-step and long-chain reasoning. However, extending their reasoning capabilities to encompass deep interactions with search remains a non-trivial challenge, as models often fail to identify optimal reasoning-search interaction trajectories, resulting in suboptimal responses. We propose R-Search, a novel reinforcement learning framework for Reasoning-Search integration, designed to enable LLMs to autonomously execute multi-step reasoning with deep search interaction, and learn optimal reasoning search interaction trajectories via multi-reward signals, improving response quality in complex logic- and knowledge-intensive tasks. R-Search guides the LLM to dynamically decide when to retrieve or reason, while globally integrating key evidence to enhance deep knowledge interaction between reasoning and search. During RL training, R-Search provides multi-stage, multi-type rewards to jointly optimize the reasoning-search trajectory. Experiments on seven datasets show that R-Search outperforms advanced RAG baselines by up to 32.2% (in-domain) and 25.1% (out-of-domain). The code and data are available at https://github.com/QingFei1/R-Search.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13865v2">A Survey on (M)LLM-Based GUI Agents</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Graphical User Interface (GUI) Agents have emerged as a transformative paradigm in human-computer interaction, evolving from rule-based automation scripts to sophisticated AI-driven systems capable of understanding and executing complex interface operations. This survey provides a comprehensive examination of the rapidly advancing field of LLM-based GUI Agents, systematically analyzing their architectural foundations, technical components, and evaluation methodologies. We identify and analyze four fundamental components that constitute modern GUI Agents: (1) perception systems that integrate text-based parsing with multimodal understanding for comprehensive interface comprehension; (2) exploration mechanisms that construct and maintain knowledge bases through internal modeling, historical experience, and external information retrieval; (3) planning frameworks that leverage advanced reasoning methodologies for task decomposition and execution; and (4) interaction systems that manage action generation with robust safety controls. Through rigorous analysis of these components, we reveal how recent advances in large language models and multimodal learning have revolutionized GUI automation across desktop, mobile, and web platforms. We critically examine current evaluation frameworks, highlighting methodological limitations in existing benchmarks while proposing directions for standardization. This survey also identifies key technical challenges, including accurate element localization, effective knowledge retrieval, long-horizon planning, and safety-aware execution control, while outlining promising research directions for enhancing GUI Agents' capabilities. Our systematic review provides researchers and practitioners with a thorough understanding of the field's current state and offers insights into future developments in intelligent interface automation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04172v1">Does Prompt Design Impact Quality of Data Imputation by LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      Generating realistic synthetic tabular data presents a critical challenge in machine learning. It adds another layer of complexity when this data contain class imbalance problems. This paper presents a novel token-aware data imputation method that leverages the in-context learning capabilities of large language models. This is achieved through the combination of a structured group-wise CSV-style prompting technique and the elimination of irrelevant contextual information in the input prompt. We test this approach with two class-imbalanced binary classification datasets and evaluate the effectiveness of imputation using classification-based evaluation metrics. The experimental results demonstrate that our approach significantly reduces the input prompt size while maintaining or improving imputation quality compared to our baseline prompt, especially for datasets that are of relatively smaller in size. The contributions of this presented work is two-fold -- 1) it sheds light on the importance of prompt design when leveraging LLMs for synthetic data generation and 2) it addresses a critical gap in LLM-based data imputation for class-imbalanced datasets with missing data by providing a practical solution within computational constraints. We hope that our work will foster further research and discussions about leveraging the incredible potential of LLMs and prompt engineering techniques for synthetic data generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.16748v2">Through the Prism of Culture: Evaluating LLMs' Understanding of Indian Subcultures and Traditions</a></div>
    <div class="paper-meta">
      📅 2025-06-04
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable advancements but also raise concerns about cultural bias, often reflecting dominant narratives at the expense of under-represented subcultures. In this study, we evaluate the capacity of LLMs to recognize and accurately respond to the Little Traditions within Indian society, encompassing localized cultural practices and subcultures such as caste, kinship, marriage, and religion. Through a series of case studies, we assess whether LLMs can balance the interplay between dominant Great Traditions and localized Little Traditions. We explore various prompting strategies and further investigate whether using prompts in regional languages enhances the models cultural sensitivity and response quality. Our findings reveal that while LLMs demonstrate an ability to articulate cultural nuances, they often struggle to apply this understanding in practical, context-specific scenarios. To the best of our knowledge, this is the first study to analyze LLMs engagement with Indian subcultures, offering critical insights into the challenges of embedding cultural diversity in AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2506.04167v1">Neural and Cognitive Impacts of AI: The Influence of Task Subjectivity on Human-LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-06-04
      | 💬 15 pages, 12 figures
    </div>
    <details class="paper-abstract">
      AI-based interactive assistants are advancing human-augmenting technology, yet their effects on users' mental and physiological states remain under-explored. We address this gap by analyzing how Copilot for Microsoft Word, a LLM-based assistant, impacts users. Using tasks ranging from objective (SAT reading comprehension) to subjective (personal reflection), and with measurements including fNIRS, Empatica E4, NASA-TLX, and questionnaires, we measure Copilot's effects on users. We also evaluate users' performance with and without Copilot across tasks. In objective tasks, participants reported a reduction of workload and an increase in enjoyment, which was paired with objective performance increases. Participants reported reduced workload and increased enjoyment with no change in performance in a creative poetry writing task. However, no benefits due to Copilot use were reported in a highly subjective self-reflection task. Although no physiological changes were recorded due to Copilot use, task-dependent differences in prefrontal cortex activation offer complementary insights into the cognitive processes associated with successful and unsuccessful human-AI collaboration. These findings suggest that AI assistants' effectiveness varies with task type-particularly showing decreased usefulness in tasks that engage episodic memory-and presents a brain-network based hypothesis of human-AI collaboration.
    </details>
</div>
