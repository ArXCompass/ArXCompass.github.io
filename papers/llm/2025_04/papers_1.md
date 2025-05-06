# llm - 2025_04

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.20774v3">Can We Trust Embodied Agents? Exploring Backdoor Attacks against Embodied LLM-based Decision-Making Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 Accepted paper at ICLR 2025, 31 pages, including main paper, references, and appendix
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown significant promise in real-world decision-making tasks for embodied artificial intelligence, especially when fine-tuned to leverage their inherent common sense and reasoning abilities while being tailored to specific applications. However, this fine-tuning process introduces considerable safety and security vulnerabilities, especially in safety-critical cyber-physical systems. In this work, we propose the first comprehensive framework for Backdoor Attacks against LLM-based Decision-making systems (BALD) in embodied AI, systematically exploring the attack surfaces and trigger mechanisms. Specifically, we propose three distinct attack mechanisms: word injection, scenario manipulation, and knowledge injection, targeting various components in the LLM-based decision-making pipeline. We perform extensive experiments on representative LLMs (GPT-3.5, LLaMA2, PaLM2) in autonomous driving and home robot tasks, demonstrating the effectiveness and stealthiness of our backdoor triggers across various attack channels, with cases like vehicles accelerating toward obstacles and robots placing knives on beds. Our word and knowledge injection attacks achieve nearly 100% success rate across multiple models and datasets while requiring only limited access to the system. Our scenario manipulation attack yields success rates exceeding 65%, reaching up to 90%, and does not require any runtime system intrusion. We also assess the robustness of these attacks against defenses, revealing their resilience. Our findings highlight critical security vulnerabilities in embodied LLM systems and emphasize the urgent need for safeguarding these systems to mitigate potential risks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21851v1">TRUST: An LLM-Based Dialogue System for Trauma Understanding and Structured Assessments</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 5 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Objectives: While Large Language Models (LLMs) have been widely used to assist clinicians and support patients, no existing work has explored dialogue systems for standard diagnostic interviews and assessments. This study aims to bridge the gap in mental healthcare accessibility by developing an LLM-powered dialogue system that replicates clinician behavior. Materials and Methods: We introduce TRUST, a framework of cooperative LLM modules capable of conducting formal diagnostic interviews and assessments for Post-Traumatic Stress Disorder (PTSD). To guide the generation of appropriate clinical responses, we propose a Dialogue Acts schema specifically designed for clinical interviews. Additionally, we develop a patient simulation approach based on real-life interview transcripts to replace time-consuming and costly manual testing by clinicians. Results: A comprehensive set of evaluation metrics is designed to assess the dialogue system from both the agent and patient simulation perspectives. Expert evaluations by conversation and clinical specialists show that TRUST performs comparably to real-life clinical interviews. Discussion: Our system performs at the level of average clinicians, with room for future enhancements in communication styles and response appropriateness. Conclusions: Our TRUST framework shows its potential to facilitate mental healthcare availability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19254v2">Uncertainty Quantification for Language Models: A Suite of Black-Box, White-Box, LLM Judge, and Ensemble Scorers</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 UQLM repository: https://github.com/cvs-health/uqlm
    </div>
    <details class="paper-abstract">
      Hallucinations are a persistent problem with Large Language Models (LLMs). As these models become increasingly used in high-stakes domains, such as healthcare and finance, the need for effective hallucination detection is crucial. To this end, we propose a versatile framework for zero-resource hallucination detection that practitioners can apply to real-world use cases. To achieve this, we adapt a variety of existing uncertainty quantification (UQ) techniques, including black-box UQ, white-box UQ, and LLM-as-a-Judge, transforming them as necessary into standardized response-level confidence scores ranging from 0 to 1. To enhance flexibility, we introduce a tunable ensemble approach that incorporates any combination of the individual confidence scores. This approach enables practitioners to optimize the ensemble for a specific use case for improved performance. To streamline implementation, the full suite of scorers is offered in this paper's companion Python toolkit, UQLM. To evaluate the performance of the various scorers, we conduct an extensive set of experiments using several LLM question-answering benchmarks. We find that our tunable ensemble typically surpasses its individual components and outperforms existing hallucination detection methods. Our results demonstrate the benefits of customized hallucination detection strategies for improving the accuracy and reliability of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21773v1">MAC-Tuning: LLM Multi-Compositional Problem Reasoning with Enhanced Knowledge Boundary Awareness</a></div>
    <div class="paper-meta">
      📅 2025-04-30
    </div>
    <details class="paper-abstract">
      With the widespread application of large language models (LLMs), the issue of generating non-existing facts, known as hallucination, has garnered increasing attention. Previous research in enhancing LLM confidence estimation mainly focuses on the single problem setting. However, LLM awareness of its internal parameterized knowledge boundary under the more challenging multi-problem setting, which requires answering multiple problems accurately simultaneously, remains underexplored. To bridge this gap, we introduce a novel method, Multiple Answers and Confidence Stepwise Tuning (MAC-Tuning), that separates the learning of answer prediction and confidence estimation during fine-tuning on instruction data. Extensive experiments demonstrate that our method outperforms baselines by up to 25% in average precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21770v1">LASHED: LLMs And Static Hardware Analysis for Early Detection of RTL Bugs</a></div>
    <div class="paper-meta">
      📅 2025-04-30
    </div>
    <details class="paper-abstract">
      While static analysis is useful in detecting early-stage hardware security bugs, its efficacy is limited because it requires information to form checks and is often unable to explain the security impact of a detected vulnerability. Large Language Models can be useful in filling these gaps by identifying relevant assets, removing false violations flagged by static analysis tools, and explaining the reported violations. LASHED combines the two approaches (LLMs and Static Analysis) to overcome each other's limitations for hardware security bug detection. We investigate our approach on four open-source SoCs for five Common Weakness Enumerations (CWEs) and present strategies for improvement with better prompt engineering. We find that 87.5% of instances flagged by our recommended scheme are plausible CWEs. In-context learning and asking the model to 'think again' improves LASHED's precision.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21769v1">LLM-based Interactive Imitation Learning for Robotic Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 To be published in IJCNN 2025 proceedings
    </div>
    <details class="paper-abstract">
      Recent advancements in machine learning provide methods to train autonomous agents capable of handling the increasing complexity of sequential decision-making in robotics. Imitation Learning (IL) is a prominent approach, where agents learn to control robots based on human demonstrations. However, IL commonly suffers from violating the independent and identically distributed (i.i.d) assumption in robotic tasks. Interactive Imitation Learning (IIL) achieves improved performance by allowing agents to learn from interactive feedback from human teachers. Despite these improvements, both approaches come with significant costs due to the necessity of human involvement. Leveraging the emergent capabilities of Large Language Models (LLMs) in reasoning and generating human-like responses, we introduce LLM-iTeach -- a novel IIL framework that utilizes an LLM as an interactive teacher to enhance agent performance while alleviating the dependence on human resources. Firstly, LLM-iTeach uses a hierarchical prompting strategy that guides the LLM in generating a policy in Python code. Then, with a designed similarity-based feedback mechanism, LLM-iTeach provides corrective and evaluative feedback interactively during the agent's training. We evaluate LLM-iTeach against baseline methods such as Behavior Cloning (BC), an IL method, and CEILing, a state-of-the-art IIL method using a human teacher, on various robotic manipulation tasks. Our results demonstrate that LLM-iTeach surpasses BC in the success rate and achieves or even outscores that of CEILing, highlighting the potential of LLMs as cost-effective, human-like teachers in interactive learning environments. We further demonstrate the method's potential for generalization by evaluating it on additional tasks. The code and prompts are provided at: https://github.com/Tubicor/LLM-iTeach.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21735v1">TheraQuest: A Gamified, LLM-Powered Simulation for Massage Therapy Training</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 8 Pages
    </div>
    <details class="paper-abstract">
      Massage therapy training emphasizes hands-on techniques and effective therapist--patient communication. However, many educational programs struggle to provide realistic practice scenarios. To address this problem, we propose TheraQuest, a gamified, web-based simulation platform that employs large language models (LLMs) to generate diverse virtual patients with varying symptoms and cultural backgrounds. Through interactive dialogue, anatomical decision-making, and immediate assessment, trainees develop both diagnostic reasoning and empathetic communication skills in a low-risk environment. Unlike exclusively VR-based solutions, TheraQuest remains accessible via standard web browsers, mitigating the cost and discomfort associated with extended headset use. Preliminary testing suggests that integrating LLM-driven virtual patients with real-time skill metrics can enhance trainee engagement and help bridge the gap between theoretical knowledge and clinical proficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21716v1">LLM-Empowered Embodied Agent for Memory-Augmented Task Planning in Household Robotics</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 Accepted at Austrian Robotics Workshop 2025
    </div>
    <details class="paper-abstract">
      We present an embodied robotic system with an LLM-driven agent-orchestration architecture for autonomous household object management. The system integrates memory-augmented task planning, enabling robots to execute high-level user commands while tracking past actions. It employs three specialized agents: a routing agent, a task planning agent, and a knowledge base agent, each powered by task-specific LLMs. By leveraging in-context learning, our system avoids the need for explicit model training. RAG enables the system to retrieve context from past interactions, enhancing long-term object tracking. A combination of Grounded SAM and LLaMa3.2-Vision provides robust object detection, facilitating semantic scene understanding for task planning. Evaluation across three household scenarios demonstrates high task planning accuracy and an improvement in memory recall due to RAG. Specifically, Qwen2.5 yields best performance for specialized agents, while LLaMA3.1 excels in routing tasks. The source code is available at: https://github.com/marc1198/chat-hsr.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.18964v4">LLMs and Finetuning: Benchmarking cross-domain performance for hate speech detection</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 18 pages, 3 figures, 5 tables
    </div>
    <details class="paper-abstract">
      In the evolving landscape of online communication, hate speech detection remains a formidable challenge, further compounded by the diversity of digital platforms. This study investigates the effectiveness and adaptability of pre-trained and fine-tuned Large Language Models (LLMs) in identifying hate speech, to address two central questions: (1) To what extent does the model performance depend on the fine-tuning and training parameters?, (2) To what extent do models generalize to cross-domain hate speech detection? and (3) What are the specific features of the datasets or models that influence the generalization potential? The experiment shows that LLMs offer a huge advantage over the state-of-the-art even without pretraining. Ordinary least squares analyses suggest that the advantage of training with fine-grained hate speech labels is washed away with the increase in dataset size. While our research demonstrates the potential of large language models (LLMs) for hate speech detection, several limitations remain, particularly regarding the validity and the reproducibility of the results. We conclude with an exhaustive discussion of the challenges we faced in our experimentation and offer recommended best practices for future scholars designing benchmarking experiments of this kind.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21700v1">XBreaking: Explainable Artificial Intelligence for Jailbreaking LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-30
    </div>
    <details class="paper-abstract">
      Large Language Models are fundamental actors in the modern IT landscape dominated by AI solutions. However, security threats associated with them might prevent their reliable adoption in critical application scenarios such as government organizations and medical institutions. For this reason, commercial LLMs typically undergo a sophisticated censoring mechanism to eliminate any harmful output they could possibly produce. In response to this, LLM Jailbreaking is a significant threat to such protections, and many previous approaches have already demonstrated its effectiveness across diverse domains. Existing jailbreak proposals mostly adopt a generate-and-test strategy to craft malicious input. To improve the comprehension of censoring mechanisms and design a targeted jailbreak attack, we propose an Explainable-AI solution that comparatively analyzes the behavior of censored and uncensored models to derive unique exploitable alignment patterns. Then, we propose XBreaking, a novel jailbreak attack that exploits these unique patterns to break the security constraints of LLMs by targeted noise injection. Our thorough experimental campaign returns important insights about the censoring mechanisms and demonstrates the effectiveness and performance of our attack.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21680v1">Hoist with His Own Petard: Inducing Guardrails to Facilitate Denial-of-Service Attacks on Retrieval-Augmented Generation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 11 pages, 6 figures. This work will be submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) integrates Large Language Models (LLMs) with external knowledge bases, improving output quality while introducing new security risks. Existing studies on RAG vulnerabilities typically focus on exploiting the retrieval mechanism to inject erroneous knowledge or malicious texts, inducing incorrect outputs. However, these approaches overlook critical weaknesses within LLMs, leaving important attack vectors unexplored and limiting the scope and efficiency of attacks. In this paper, we uncover a novel vulnerability: the safety guardrails of LLMs, while designed for protection, can also be exploited as an attack vector by adversaries. Building on this vulnerability, we propose MutedRAG, a novel denial-of-service attack that reversely leverages the guardrails of LLMs to undermine the availability of RAG systems. By injecting minimalistic jailbreak texts, such as "\textit{How to build a bomb}", into the knowledge base, MutedRAG intentionally triggers the LLM's safety guardrails, causing the system to reject legitimate queries. Besides, due to the high sensitivity of guardrails, a single jailbreak sample can affect multiple queries, effectively amplifying the efficiency of attacks while reducing their costs. Experimental results on three datasets demonstrate that MutedRAG achieves an attack success rate exceeding 60% in many scenarios, requiring only less than one malicious text to each target query on average. In addition, we evaluate potential defense strategies against MutedRAG, finding that some of current mechanisms are insufficient to mitigate this threat, underscoring the urgent need for more robust solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20828v2">Ascendra: Dynamic Request Prioritization for Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-04-30
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has driven the need for more efficient serving strategies. In this context, efficiency refers to the proportion of requests that meet their Service Level Objectives (SLOs), particularly for Time To First Token (TTFT) and Time Between Tokens (TBT). However, existing systems often prioritize one metric at the cost of the other. We present Ascendra, an LLM serving system designed to meet both TTFT and TBT SLOs simultaneously. The core insight behind Ascendra is that a request's urgency evolves as it approaches its deadline. To leverage this, Ascendra partitions GPU resources into two types of instances: low-priority and high-priority. Low-priority instances maximize throughput by processing requests out of arrival order, but at the risk of request starvation. To address this, Ascendra employs a performance model to predict requests at risk of missing their SLOs and proactively offloads them to high-priority instances. High-priority instances are optimized for low-latency execution and handle urgent requests nearing their deadlines. This partitioned architecture enables Ascendra to effectively balance high throughput and low latency. Extensive evaluation shows that Ascendra improves system throughput by up to 1.7x compared to vLLM and Sarathi-Serve while meeting both TTFT and TBT SLOs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21625v1">Meeseeks: An Iterative Benchmark Evaluating LLMs Multi-Turn Instruction-Following Ability</a></div>
    <div class="paper-meta">
      📅 2025-04-30
    </div>
    <details class="paper-abstract">
      The ability to follow instructions accurately is fundamental for Large Language Models (LLMs) to serve as reliable agents in real-world applications. While existing instruction-following benchmarks are either single-turn or introduce new requirements in each turn without allowing self-correction, Meeseeks simulates realistic human-LLM interactions through an iterative feedback process. This design enables models to self-correct based on specific requirement failures, better reflecting real-world user-end usage patterns. The benchmark implements a comprehensive evaluation system with 38 capability tags organized across three dimensions: Intent Recognition, Granular Content Validation, and Output Structure Validation. Through rigorous evaluation across LLMs, Meeseeks provides valuable insights into LLMs' instruction-following capabilities in practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.15551v2">Revise, Reason, and Recognize: LLM-Based Emotion Recognition via Emotion-Specific Prompts and ASR Error Correction</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 Accepted to ICASSP 2025
    </div>
    <details class="paper-abstract">
      Annotating and recognizing speech emotion using prompt engineering has recently emerged with the advancement of Large Language Models (LLMs), yet its efficacy and reliability remain questionable. In this paper, we conduct a systematic study on this topic, beginning with the proposal of novel prompts that incorporate emotion-specific knowledge from acoustics, linguistics, and psychology. Subsequently, we examine the effectiveness of LLM-based prompting on Automatic Speech Recognition (ASR) transcription, contrasting it with ground-truth transcription. Furthermore, we propose a Revise-Reason-Recognize prompting pipeline for robust LLM-based emotion recognition from spoken language with ASR errors. Additionally, experiments on context-aware learning, in-context learning, and instruction tuning are performed to examine the usefulness of LLM training schemes in this direction. Finally, we investigate the sensitivity of LLMs to minor prompt variations. Experimental results demonstrate the efficacy of the emotion-specific prompts, ASR error correction, and LLM training schemes for LLM-based emotion recognition. Our study aims to refine the use of LLMs in emotion recognition and related domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21605v1">RDF-Based Structured Quality Assessment Representation of Multilingual LLM Evaluations</a></div>
    <div class="paper-meta">
      📅 2025-04-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) increasingly serve as knowledge interfaces, yet systematically assessing their reliability with conflicting information remains difficult. We propose an RDF-based framework to assess multilingual LLM quality, focusing on knowledge conflicts. Our approach captures model responses across four distinct context conditions (complete, incomplete, conflicting, and no-context information) in German and English. This structured representation enables the comprehensive analysis of knowledge leakage-where models favor training data over provided context-error detection, and multilingual consistency. We demonstrate the framework through a fire safety domain experiment, revealing critical patterns in context prioritization and language-specific performance, and demonstrating that our vocabulary was sufficient to express every assessment facet encountered in the 28-question study.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21589v1">DNB-AI-Project at SemEval-2025 Task 5: An LLM-Ensemble Approach for Automated Subject Indexing</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 11 pages, 4 figures, submitted to SemEval-2025 workshop Task 5: LLMs4Subjects
    </div>
    <details class="paper-abstract">
      This paper presents our system developed for the SemEval-2025 Task 5: LLMs4Subjects: LLM-based Automated Subject Tagging for a National Technical Library's Open-Access Catalog. Our system relies on prompting a selection of LLMs with varying examples of intellectually annotated records and asking the LLMs to similarly suggest keywords for new records. This few-shot prompting technique is combined with a series of post-processing steps that map the generated keywords to the target vocabulary, aggregate the resulting subject terms to an ensemble vote and, finally, rank them as to their relevance to the record. Our system is fourth in the quantitative ranking in the all-subjects track, but achieves the best result in the qualitative ranking conducted by subject indexing experts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.10133v2">You Name It, I Run It: An LLM Agent to Execute Tests of Arbitrary Projects</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 PUBLISHED AT ISSTA 2025
    </div>
    <details class="paper-abstract">
      The ability to execute the test suite of a project is essential in many scenarios, e.g., to assess code quality and code coverage, to validate code changes made by developers or automated tools, and to ensure compatibility with dependencies. Despite its importance, executing the test suite of a project can be challenging in practice because different projects use different programming languages, software ecosystems, build systems, testing frameworks, and other tools. These challenges make it difficult to create a reliable, universal test execution method that works across different projects. This paper presents ExecutionAgent, an automated technique that prepares scripts for building an arbitrary project from source code and running its test cases. Inspired by the way a human developer would address this task, our approach is a large language model (LLM)-based agent that autonomously executes commands and interacts with the host system. The agent uses meta-prompting to gather guidelines on the latest technologies related to the given project, and it iteratively refines its process based on feedback from the previous steps. Our evaluation applies ExecutionAgent to 50 open-source projects that use 14 different programming languages and many different build and testing tools. The approach successfully executes the test suites of 33/50 projects, while matching the test results of ground truth test suite executions with a deviation of only 7.5%. These results improve over the best previously available technique by 6.6x. The costs imposed by the approach are reasonable, with an execution time of 74 minutes and LLM costs of USD 0.16, on average per project. We envision ExecutionAgent to serve as a valuable tool for developers, automated programming tools, and researchers that need to execute tests across a wide variety of projects.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20462v2">TAMO:Fine-Grained Root Cause Analysis via Tool-Assisted LLM Agent with Multi-Modality Observation Data</a></div>
    <div class="paper-meta">
      📅 2025-04-30
    </div>
    <details class="paper-abstract">
      With the development of distributed systems, microservices and cloud native technologies have become central to modern enterprise software development. Despite bringing significant advantages, these technologies also increase system complexity and operational challenges. Traditional root cause analysis (RCA) struggles to achieve automated fault response, heavily relying on manual intervention. In recent years, large language models (LLMs) have made breakthroughs in contextual inference and domain knowledge integration, providing new solutions for Artificial Intelligence for Operations (AIOps). However, Existing LLM-based approaches face three key challenges: text input constraints, dynamic service dependency hallucinations, and context window limitations. To address these issues, we propose a tool-assisted LLM agent with multi-modality observation data, namely TAMO, for fine-grained RCA. It unifies multi-modal observational data into time-aligned representations to extract consistent features and employs specialized root cause localization and fault classification tools for perceiving the contextual environment. This approach overcomes the limitations of LLM in handling real-time changing service dependencies and raw observational data and guides LLM to generate repair strategies aligned with system contexts by structuring key information into a prompt. Experimental results show that TAMO performs well in root cause analysis when dealing with public datasets characterized by heterogeneity and common fault types, demonstrating its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04178v3">MSL: Not All Tokens Are What You Need for Tuning LLM as a Recommender</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 Accepted by SIGIR2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), known for their comprehension capabilities and extensive knowledge, have been increasingly applied to recommendation systems (RS). Given the fundamental gap between the mechanism of LLMs and the requirement of RS, researchers have focused on fine-tuning LLMs with recommendation-specific data to enhance their performance. Language Modeling Loss (LML), originally designed for language generation tasks, is commonly adopted. However, we identify two critical limitations of LML: 1) it exhibits significant divergence from the recommendation objective; 2) it erroneously treats all fictitious item descriptions as negative samples, introducing misleading training signals. To address these limitations, we propose a novel Masked Softmax Loss (MSL) tailored for fine-tuning LLMs on recommendation. MSL improves LML by identifying and masking invalid tokens that could lead to fictitious item descriptions during loss computation. This strategy can effectively avoid the interference from erroneous negative signals and ensure well alignment with the recommendation objective supported by theoretical guarantees. During implementation, we identify a potential challenge related to gradient vanishing of MSL. To overcome this, we further introduce the temperature coefficient and propose an Adaptive Temperature Strategy (ATS) that adaptively adjusts the temperature without requiring extensive hyperparameter tuning. Extensive experiments conducted on four public datasets further validate the effectiveness of MSL, achieving an average improvement of 42.24% in NDCG@10. The code is available at https://github.com/WANGBohaO-jpg/MSL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21398v1">In a Few Words: Comparing Weak Supervision and LLMs for Short Query Intent Classification</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 accepted at International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25), July 13--18, 2025, Padua, Italy
    </div>
    <details class="paper-abstract">
      User intent classification is an important task in information retrieval. Previously, user intents were classified manually and automatically; the latter helped to avoid hand labelling of large datasets. Recent studies explored whether LLMs can reliably determine user intent. However, researchers have recognized the limitations of using generative LLMs for classification tasks. In this study, we empirically compare user intent classification into informational, navigational, and transactional categories, using weak supervision and LLMs. Specifically, we evaluate LLaMA-3.1-8B-Instruct and LLaMA-3.1-70B-Instruct for in-context learning and LLaMA-3.1-8B-Instruct for fine-tuning, comparing their performance to an established baseline classifier trained using weak supervision (ORCAS-I). Our results indicate that while LLMs outperform weak supervision in recall, they continue to struggle with precision, which shows the need for improved methods to balance both metrics effectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21304v1">Unsupervised Feature Transformation via In-context Generation, Generator-critic LLM Agents, and Duet-play Teaming</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 Accepted to IJCAI 2025
    </div>
    <details class="paper-abstract">
      Feature transformation involves generating a new set of features from the original dataset to enhance the data's utility. In certain domains like material performance screening, dimensionality is large and collecting labels is expensive and lengthy. It highly necessitates transforming feature spaces efficiently and without supervision to enhance data readiness and AI utility. However, existing methods fall short in efficient navigation of a vast space of feature combinations, and are mostly designed for supervised settings. To fill this gap, our unique perspective is to leverage a generator-critic duet-play teaming framework using LLM agents and in-context learning to derive pseudo-supervision from unsupervised data. The framework consists of three interconnected steps: (1) Critic agent diagnoses data to generate actionable advice, (2) Generator agent produces tokenized feature transformations guided by the critic's advice, and (3) Iterative refinement ensures continuous improvement through feedback between agents. The generator-critic framework can be generalized to human-agent collaborative generation, by replacing the critic agent with human experts. Extensive experiments demonstrate that the proposed framework outperforms even supervised baselines in feature transformation efficiency, robustness, and practical applicability across diverse datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21276v1">Assessing LLM code generation quality through path planning tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-30
    </div>
    <details class="paper-abstract">
      As LLM-generated code grows in popularity, more evaluation is needed to assess the risks of using such tools, especially for safety-critical applications such as path planning. Existing coding benchmarks are insufficient as they do not reflect the context and complexity of safety-critical applications. To this end, we assessed six LLMs' abilities to generate the code for three different path-planning algorithms and tested them on three maps of various difficulties. Our results suggest that LLM-generated code presents serious hazards for path planning applications and should not be applied in safety-critical contexts without rigorous testing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21934v4">Proof or Bluff? Evaluating LLMs on 2025 USA Math Olympiad</a></div>
    <div class="paper-meta">
      📅 2025-04-30
    </div>
    <details class="paper-abstract">
      Recent math benchmarks for large language models (LLMs) such as MathArena indicate that state-of-the-art reasoning models achieve impressive performance on mathematical competitions like AIME, with the leading model, Gemini-2.5-Pro, achieving scores comparable to top human competitors. However, these benchmarks evaluate models solely based on final numerical answers, neglecting rigorous reasoning and proof generation which are essential for real-world mathematical tasks. To address this, we introduce the first comprehensive evaluation of full-solution reasoning for challenging mathematical problems. Using expert human annotators, we evaluated several state-of-the-art reasoning models on the six problems from the 2025 USAMO within hours of their release. Our results reveal that all tested models struggled significantly: only Gemini-2.5-Pro achieves a non-trivial score of 25%, while all other models achieve less than 5%. Through detailed analysis of reasoning traces, we identify the most common failure modes and find several unwanted artifacts arising from the optimization strategies employed during model training. Overall, our results suggest that current LLMs are inadequate for rigorous mathematical reasoning tasks, highlighting the need for substantial improvements in reasoning and proof generation capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19915v2">LLM-driven Effective Knowledge Tracing by Integrating Dual-channel Difficulty</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 During a careful review of our base-experiment results, we discovered a possible error in the way some data were recorded. To ensure the integrity and accuracy of our work, we must correct these results and revise the corresponding analysis before making the manuscript publicly available
    </div>
    <details class="paper-abstract">
      Knowledge Tracing (KT) is a fundamental technology in intelligent tutoring systems used to simulate changes in students' knowledge state during learning, track personalized knowledge mastery, and predict performance. However, current KT models face three major challenges: (1) When encountering new questions, models face cold-start problems due to sparse interaction records, making precise modeling difficult; (2) Traditional models only use historical interaction records for student personalization modeling, unable to accurately track individual mastery levels, resulting in unclear personalized modeling; (3) The decision-making process is opaque to educators, making it challenging for them to understand model judgments. To address these challenges, we propose a novel Dual-channel Difficulty-aware Knowledge Tracing (DDKT) framework that utilizes Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) for subjective difficulty assessment, while integrating difficulty bias-aware algorithms and student mastery algorithms for precise difficulty measurement. Our framework introduces three key innovations: (1) Difficulty Balance Perception Sequence (DBPS) - students' subjective perceptions combined with objective difficulty, measuring gaps between LLM-assessed difficulty, mathematical-statistical difficulty, and students' subjective perceived difficulty through attention mechanisms; (2) Difficulty Mastery Ratio (DMR) - precise modeling of student mastery levels through different difficulty zones; (3) Knowledge State Update Mechanism - implementing personalized knowledge acquisition through gated networks and updating student knowledge state. Experimental results on two real datasets show our method consistently outperforms nine baseline models, improving AUC metrics by 2% to 10% while effectively addressing cold-start problems and enhancing model interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21239v1">Memorization and Knowledge Injection in Gated LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-30
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) currently struggle to sequentially add new memories and integrate new knowledge. These limitations contrast with the human ability to continuously learn from new experiences and acquire knowledge throughout life. Most existing approaches add memories either through large context windows or external memory buffers (e.g., Retrieval-Augmented Generation), and studies on knowledge injection rarely test scenarios resembling everyday life events. In this work, we introduce a continual learning framework, Memory Embedded in Gated LLMs (MEGa), which injects event memories directly into the weights of LLMs. Each memory is stored in a dedicated set of gated low-rank weights. During inference, a gating mechanism activates relevant memory weights by matching query embeddings to stored memory embeddings. This enables the model to both recall entire memories and answer related questions. On two datasets - fictional characters and Wikipedia events - MEGa outperforms baseline approaches in mitigating catastrophic forgetting. Our model draws inspiration from the complementary memory system of the human brain.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00212v1">Which Agent Causes Task Failures and When? On Automated Failure Attribution of LLM Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-30
    </div>
    <details class="paper-abstract">
      Failure attribution in LLM multi-agent systems-identifying the agent and step responsible for task failures-provides crucial clues for systems debugging but remains underexplored and labor-intensive. In this paper, we propose and formulate a new research area: automated failure attribution for LLM multi-agent systems. To support this initiative, we introduce the Who&When dataset, comprising extensive failure logs from 127 LLM multi-agent systems with fine-grained annotations linking failures to specific agents and decisive error steps. Using the Who&When, we develop and evaluate three automated failure attribution methods, summarizing their corresponding pros and cons. The best method achieves 53.5% accuracy in identifying failure-responsible agents but only 14.2% in pinpointing failure steps, with some methods performing below random. Even SOTA reasoning models, such as OpenAI o1 and DeepSeek R1, fail to achieve practical usability. These results highlight the task's complexity and the need for further research in this area. Code and dataset are available at https://github.com/mingyin1/Agents_Failure_Attribution
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07377v2">Process-Supervised LLM Recommenders via Flow-guided Tuning</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 Accepted by SIGIR 2025
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly adapted for recommendation systems via supervised fine-tuning (SFT), this approach amplifies popularity bias due to its likelihood maximization objective, compromising recommendation diversity and fairness. To address this, we present Flow-guided fine-tuning recommender (Flower), which replaces SFT with a Generative Flow Network (GFlowNet) framework that enacts process supervision through token-level reward propagation. Flower's key innovation lies in decomposing item-level rewards into constituent token rewards, enabling direct alignment between token generation probabilities and their reward signals. This mechanism achieves three critical advancements: (1) popularity bias mitigation and fairness enhancement through empirical distribution matching, (2) preservation of diversity through GFlowNet's proportional sampling, and (3) flexible integration of personalized preferences via adaptable token rewards. Experiments demonstrate Flower's superior distribution-fitting capability and its significant advantages over traditional SFT in terms of accuracy, fairness, and diversity, highlighting its potential to improve LLM-based recommendation systems. The implementation is available via https://github.com/MrPeach0301/Flower
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20774v2">Are LLM-Judges Robust to Expressions of Uncertainty? Investigating the effect of Epistemic Markers on LLM-based Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 NAACL 2025 Oral (21 pages, 6 figures, 15 tables)
    </div>
    <details class="paper-abstract">
      In line with the principle of honesty, there has been a growing effort to train large language models (LLMs) to generate outputs containing epistemic markers. However, evaluation in the presence of epistemic markers has been largely overlooked, raising a critical question: Could the use of epistemic markers in LLM-generated outputs lead to unintended negative consequences? To address this, we present EMBER, a benchmark designed to assess the robustness of LLM-judges to epistemic markers in both single and pairwise evaluation settings. Our findings, based on evaluations using EMBER, reveal that all tested LLM-judges, including GPT-4o, show a notable lack of robustness in the presence of epistemic markers. Specifically, we observe a negative bias toward epistemic markers, with a stronger bias against markers expressing uncertainty. This suggests that LLM-judges are influenced by the presence of these markers and do not focus solely on the correctness of the content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04827v2">Leveraging LLMs for Influence Path Planning in Proactive Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 WWW 2025 short paper
    </div>
    <details class="paper-abstract">
      Recommender systems are pivotal in Internet social platforms, yet they often cater to users' historical interests, leading to critical issues like echo chambers. To broaden user horizons, proactive recommender systems aim to guide user interest to gradually like a target item beyond historical interests through an influence path,i.e., a sequence of recommended items. As a representative, Influential Recommender System (IRS) designs a sequential model for influence path planning but faces issues of lacking target item inclusion and path coherence. To address the issues, we leverage the advanced planning capabilities of Large Language Models (LLMs) and propose an LLM-based Influence Path Planning (LLM-IPP) method. LLM-IPP generates coherent and effective influence paths by capturing user interest shifts and item characteristics. We introduce novel evaluation metrics and user simulators to benchmark LLM-IPP against traditional methods. Our experiments demonstrate that LLM-IPP significantly enhances user acceptability and path coherence, outperforming existing approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00127v1">Between Underthinking and Overthinking: An Empirical Study of Reasoning Length and correctness in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly optimized for long reasoning, under the assumption that more reasoning leads to better performance. However, emerging evidence suggests that longer responses can sometimes degrade accuracy rather than improve it. In this paper, we conduct a systematic empirical study of the relationship between reasoning length and answer correctness. We find that LLMs tend to overthink simple problems, generating unnecessarily long outputs, and underthink harder ones, failing to extend their reasoning when it is most needed. This indicates that models might misjudge problem difficulty and fail to calibrate their response length appropriately. Furthermore, we investigate the effects of length reduction with a preference optimization algorithm when simply preferring the shorter responses regardless of answer correctness. Experiments show that the generation length can be significantly reduced while maintaining acceptable accuracy. Our findings highlight generation length as a meaningful signal for reasoning behavior and motivate further exploration into LLMs' self-awareness in reasoning length adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00114v1">Fine-Tuning LLMs for Low-Resource Dialect Translation: The Case of Lebanese</a></div>
    <div class="paper-meta">
      📅 2025-04-30
    </div>
    <details class="paper-abstract">
      This paper examines the effectiveness of Large Language Models (LLMs) in translating the low-resource Lebanese dialect, focusing on the impact of culturally authentic data versus larger translated datasets. We compare three fine-tuning approaches: Basic, contrastive, and grammar-hint tuning, using open-source Aya23 models. Experiments reveal that models fine-tuned on a smaller but culturally aware Lebanese dataset (LW) consistently outperform those trained on larger, non-native data. The best results were achieved through contrastive fine-tuning paired with contrastive prompting, which indicates the benefits of exposing translation models to bad examples. In addition, to ensure authentic evaluation, we introduce LebEval, a new benchmark derived from native Lebanese content, and compare it to the existing FLoRes benchmark. Our findings challenge the "More Data is Better" paradigm and emphasize the crucial role of cultural authenticity in dialectal translation. We made our datasets and code available on Github.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00057v1">A Report on the llms evaluating the high school questions</a></div>
    <div class="paper-meta">
      📅 2025-04-30
    </div>
    <details class="paper-abstract">
      This report aims to evaluate the performance of large language models (LLMs) in solving high school science questions and to explore their potential applications in the educational field. With the rapid development of LLMs in the field of natural language processing, their application in education has attracted widespread attention. This study selected mathematics exam questions from the college entrance examinations (2019-2023) as evaluation data and utilized at least eight LLM APIs to provide answers. A comprehensive assessment was conducted based on metrics such as accuracy, response time, logical reasoning, and creativity. Through an in-depth analysis of the evaluation results, this report reveals the strengths and weaknesses of LLMs in handling high school science questions and discusses their implications for educational practice. The findings indicate that although LLMs perform excellently in certain aspects, there is still room for improvement in logical reasoning and creative problem-solving. This report provides an empirical foundation for further research and application of LLMs in the educational field and offers suggestions for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00049v1">Humanizing LLMs: A Survey of Psychological Measurements with Tools, Datasets, and Human-Agent Applications</a></div>
    <div class="paper-meta">
      📅 2025-04-30
      | 💬 26 pages,7 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly used in human-centered tasks, assessing their psychological traits is crucial for understanding their social impact and ensuring trustworthy AI alignment. While existing reviews have covered some aspects of related research, several important areas have not been systematically discussed, including detailed discussions of diverse psychological tests, LLM-specific psychological datasets, and the applications of LLMs with psychological traits. To address this gap, we systematically review six key dimensions of applying psychological theories to LLMs: (1) assessment tools; (2) LLM-specific datasets; (3) evaluation metrics (consistency and stability); (4) empirical findings; (5) personality simulation methods; and (6) LLM-based behavior simulation. Our analysis highlights both the strengths and limitations of current methods. While some LLMs exhibit reproducible personality patterns under specific prompting schemes, significant variability remains across tasks and settings. Recognizing methodological challenges such as mismatches between psychological tools and LLMs' capabilities, as well as inconsistencies in evaluation practices, this study aims to propose future directions for developing more interpretable, robust, and generalizable psychological assessment frameworks for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01449v1">COSMOS: Predictable and Cost-Effective Adaptation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) achieve remarkable performance across numerous tasks by using a diverse array of adaptation strategies. However, optimally selecting a model and adaptation strategy under resource constraints is challenging and often requires extensive experimentation. We investigate whether it is possible to accurately predict both performance and cost without expensive trials. We formalize the strategy selection problem for LLMs and introduce COSMOS, a unified prediction framework that efficiently estimates adaptation outcomes at minimal cost. We instantiate and study the capability of our framework via a pair of powerful predictors: embedding-augmented lightweight proxy models to predict fine-tuning performance, and low-sample scaling laws to forecast retrieval-augmented in-context learning. Extensive evaluation across eight representative benchmarks demonstrates that COSMOS achieves high prediction accuracy while reducing computational costs by 92.72% on average, and up to 98.71% in resource-intensive scenarios. Our results show that efficient prediction of adaptation outcomes is not only feasible but can substantially reduce the computational overhead of LLM deployment while maintaining performance standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20984v1">ACE: A Security Architecture for LLM-Integrated App Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 21 pages, 13 figures
    </div>
    <details class="paper-abstract">
      LLM-integrated app systems extend the utility of Large Language Models (LLMs) with third-party apps that are invoked by a system LLM using interleaved planning and execution phases to answer user queries. These systems introduce new attack vectors where malicious apps can cause integrity violation of planning or execution, availability breakdown, or privacy compromise during execution. In this work, we identify new attacks impacting the integrity of planning, as well as the integrity and availability of execution in LLM-integrated apps, and demonstrate them against IsolateGPT, a recent solution designed to mitigate attacks from malicious apps. We propose Abstract-Concrete-Execute (ACE), a new secure architecture for LLM-integrated app systems that provides security guarantees for system planning and execution. Specifically, ACE decouples planning into two phases by first creating an abstract execution plan using only trusted information, and then mapping the abstract plan to a concrete plan using installed system apps. We verify that the plans generated by our system satisfy user-specified secure information flow constraints via static analysis on the structured plan output. During execution, ACE enforces data and capability barriers between apps, and ensures that the execution is conducted according to the trusted abstract plan. We show experimentally that our system is secure against attacks from the INJECAGENT benchmark, a standard benchmark for control flow integrity in the face of indirect prompt injection attacks, and our newly introduced attacks. Our architecture represents a significant advancement towards hardening LLM-based systems containing system facilities of varying levels of trustworthiness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20965v1">AegisLLM: Scaling Agentic Systems for Self-Reflective Defense in LLM Security</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 ICLR 2025 Workshop BuildingTrust
    </div>
    <details class="paper-abstract">
      We introduce AegisLLM, a cooperative multi-agent defense against adversarial attacks and information leakage. In AegisLLM, a structured workflow of autonomous agents - orchestrator, deflector, responder, and evaluator - collaborate to ensure safe and compliant LLM outputs, while self-improving over time through prompt optimization. We show that scaling agentic reasoning system at test-time - both by incorporating additional agent roles and by leveraging automated prompt optimization (such as DSPy)- substantially enhances robustness without compromising model utility. This test-time defense enables real-time adaptability to evolving attacks, without requiring model retraining. Comprehensive evaluations across key threat scenarios, including unlearning and jailbreaking, demonstrate the effectiveness of AegisLLM. On the WMDP unlearning benchmark, AegisLLM achieves near-perfect unlearning with only 20 training examples and fewer than 300 LM calls. For jailbreaking benchmarks, we achieve 51% improvement compared to the base model on StrongReject, with false refusal rates of only 7.9% on PHTest compared to 18-55% for comparable methods. Our results highlight the advantages of adaptive, agentic reasoning over static defenses, establishing AegisLLM as a strong runtime alternative to traditional approaches based on model modifications. Code is available at https://github.com/zikuicai/aegisllm
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20964v1">OSVBench: Benchmarking LLMs on Specification Generation Tasks for Operating System Verification</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      We introduce OSVBench, a new benchmark for evaluating Large Language Models (LLMs) in generating complete specification code pertaining to operating system kernel verification tasks. The benchmark first defines the specification generation problem into a program synthesis problem within a confined scope of syntax and semantics by providing LLMs with the programming model. The LLMs are required to understand the provided verification assumption and the potential syntax and semantics space to search for, then generate the complete specification for the potentially buggy operating system code implementation under the guidance of the high-level functional description of the operating system. This benchmark is built upon a real-world operating system kernel, Hyperkernel, and consists of 245 complex specification generation tasks in total, each is a long context task of about 20k-30k tokens. Our comprehensive evaluation of 12 LLMs exhibits the limited performance of the current LLMs on the specification generation tasks for operating system verification. Significant disparities in their performance on the benchmark highlight differences in their ability to handle long-context code generation tasks. The evaluation toolkit and benchmark are available at https://github.com/lishangyu-hkust/OSVBench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12836v2">An LLM-Powered Agent for Physiological Data Analysis: A Case Study on PPG-based Heart Rate Estimation</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are revolutionizing healthcare by improving diagnosis, patient care, and decision support through interactive communication. More recently, they have been applied to analyzing physiological time-series like wearable data for health insight extraction. Existing methods embed raw numerical sequences directly into prompts, which exceeds token limits and increases computational costs. Additionally, some studies integrated features extracted from time-series in textual prompts or applied multimodal approaches. However, these methods often produce generic and unreliable outputs due to LLMs' limited analytical rigor and inefficiency in interpreting continuous waveforms. In this paper, we develop an LLM-powered agent for physiological time-series analysis aimed to bridge the gap in integrating LLMs with well-established analytical tools. Built on the OpenCHA, an open-source LLM-powered framework, our agent powered by OpenAI's GPT-3.5-turbo model features an orchestrator that integrates user interaction, data sources, and analytical tools to generate accurate health insights. To evaluate its effectiveness, we implement a case study on heart rate (HR) estimation from Photoplethysmogram (PPG) signals using a dataset of PPG and Electrocardiogram (ECG) recordings in a remote health monitoring study. The agent's performance is benchmarked against OpenAI GPT-4o-mini and GPT-4o, with ECG serving as the gold standard for HR estimation. Results demonstrate that our agent significantly outperforms benchmark models by achieving lower error rates and more reliable HR estimations. The agent implementation is publicly available on GitHub.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20911v1">An Empirical Study on the Capability of LLMs in Decomposing Bug Reports</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Background: Bug reports are essential to the software development life cycle. They help developers track and resolve issues, but are often difficult to process due to their complexity, which can delay resolution and affect software quality. Aims: This study investigates whether large language models (LLMs) can assist developers in automatically decomposing complex bug reports into smaller, self-contained units, making them easier to understand and address. Method: We conducted an empirical study on 127 resolved privacy-related bug reports collected from Apache Jira. We evaluated ChatGPT and DeepSeek using different prompting strategies. We first tested both LLMs with zero-shot prompts, then applied improved prompts with demonstrations (using few-shot prompting) to measure their abilities in bug decomposition. Results: Our findings show that LLMs are capable of decomposing bug reports, but their overall performance still requires further improvement and strongly depends on the quality of the prompts. With zero-shot prompts, both studied LLMs (ChatGPT and DeepSeek) performed poorly. After prompt tuning, ChatGPT's true decomposition rate increased by 140\% and DeepSeek's by 163.64\%. Conclusions: LLMs show potential in helping developers analyze and decompose complex bug reports, but they still need improvement in terms of accuracy and bug understanding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20896v1">LELANTE: LEveraging LLM for Automated ANdroid TEsting</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 6 pages, 4 figures, 29th International Conference on Evaluation and Assessment in Software Engineering (EASE)
    </div>
    <details class="paper-abstract">
      Given natural language test case description for an Android application, existing testing approaches require developers to manually write scripts using tools such as Appium and Espresso to execute the corresponding test case. This process is labor-intensive and demands significant effort to maintain as UI interfaces evolve throughout development. In this work, we introduce LELANTE, a novel framework that utilizes large language models (LLMs) to automate test case execution without requiring pre-written scripts. LELANTE interprets natural language test case descriptions, iteratively generate action plans, and perform the actions directly on the Android screen using its GUI. LELANTE employs a screen refinement process to enhance LLM interpretability, constructs a structured prompt for LLMs, and implements an action generation mechanism based on chain-of-thought reasoning of LLMs. To further reduce computational cost and enhance scalability, LELANTE utilizes model distillation using a foundational LLM. In experiments across 390 test cases spanning 10 popular Android applications, LELANTE achieved a 73% test execution success rate. Our results demonstrate that LLMs can effectively bridge the gap between natural language test case description and automated execution, making mobile testing more scalable and adaptable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20834v1">Reinforcement Learning for LLM Reasoning Under Memory Constraints</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      We explore reinforcement learning (RL) techniques to enhance reasoning within targeted problem spaces in large language models (LLMs) under memory and compute constraints. Our focus is on critic-free methods compatible with LoRA fine-tuning on a single 40GB GPU, a common limitation in academic settings. We introduce S-GRPO, a memory-efficient variant of Group Relative Policy Optimization, and T-SPMO, a token-level prefix matching strategy for fine-grained credit assignment. Despite limited resources, when used to fine-tune Qwen2-1.5B both methods significantly improve SVAMP benchmark accuracy from 46% to above 70% using LoRA training. T-SPMO also excels in multi-digit multiplication tasks, underscoring the potential of RL fine-tuning under hardware constraints. Additionally, we find that our full-token GRPO baseline under LoRA fine-tuning did not improve model performance (compared to base model) on either task, suggesting that our memory-efficient methods may act as a form of regularization that stabilizes training when only a small subset of parameters are updated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20828v1">Ascendra: Dynamic Request Prioritization for Efficient LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has driven the need for more efficient serving strategies. In this context, efficiency refers to the proportion of requests that meet their Service Level Objectives (SLOs), particularly for Time To First Token (TTFT) and Time Between Tokens (TBT). However, existing systems often prioritize one metric at the cost of the other. We present Ascendra, an LLM serving system designed to meet both TTFT and TBT SLOs simultaneously. The core insight behind Ascendra is that a request's urgency evolves as it approaches its deadline. To leverage this, Ascendra partitions GPU resources into two types of instances: low-priority and high-priority. Low-priority instances maximize throughput by processing requests out of arrival order, but at the risk of request starvation. To address this, Ascendra employs a performance model to predict requests at risk of missing their SLOs and proactively offloads them to high-priority instances. High-priority instances are optimized for low-latency execution and handle urgent requests nearing their deadlines. This partitioned architecture enables Ascendra to effectively balance high throughput and low latency. Extensive evaluation shows that Ascendra improves system throughput by up to 1.7x compared to vLLM and Sarathi-Serve while meeting both TTFT and TBT SLOs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09089v2">LocAgent: Graph-Guided LLM Agents for Code Localization</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Code localization--identifying precisely where in a codebase changes need to be made--is a fundamental yet challenging task in software maintenance. Existing approaches struggle to efficiently navigate complex codebases when identifying relevant code sections. The challenge lies in bridging natural language problem descriptions with the appropriate code elements, often requiring reasoning across hierarchical structures and multiple dependencies. We introduce LocAgent, a framework that addresses code localization through graph-based representation. By parsing codebases into directed heterogeneous graphs, LocAgent creates a lightweight representation that captures code structures (files, classes, functions) and their dependencies (imports, invocations, inheritance), enabling LLM agents to effectively search and locate relevant entities through powerful multi-hop reasoning. Experimental results on real-world benchmarks demonstrate that our approach significantly enhances accuracy in code localization. Notably, our method with the fine-tuned Qwen-2.5-Coder-Instruct-32B model achieves comparable results to SOTA proprietary models at greatly reduced cost (approximately 86% reduction), reaching up to 92.7% accuracy on file-level localization while improving downstream GitHub issue resolution success rates by 12% for multiple attempts (Pass@10). Our code is available at https://github.com/gersteinlab/LocAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.12397v2">Activated LoRA: Fine-tuned LLMs for Intrinsics</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 arXiv admin note: text overlap with arXiv:2504.11704
    </div>
    <details class="paper-abstract">
      Low-Rank Adaptation (LoRA) has emerged as a highly efficient framework for finetuning the weights of large foundation models, and has become the go-to method for data-driven customization of LLMs. Despite the promise of highly customized behaviors and capabilities, switching between relevant LoRAs in a multiturn setting is highly inefficient, as the key-value (KV) cache of the entire turn history must be recomputed with the LoRA weights before generation can begin. To address this problem, we propose Activated LoRA (aLoRA), which modifies the LoRA framework to only adapt weights for the tokens in the sequence \emph{after} the aLoRA is invoked. This change crucially allows aLoRA to accept the base model's KV cache of the input string, meaning that aLoRA can be instantly activated whenever needed in a chain without recomputing the cache. This enables building what we call \emph{intrinsics}, i.e. highly specialized models invoked to perform well-defined operations on portions of an input chain or conversation that otherwise uses the base model by default. We use aLoRA to train a set of intrinsics models, demonstrating competitive accuracy with standard LoRA while achieving significant inference benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20799v1">Hallucination by Code Generation LLMs: Taxonomy, Benchmarks, Mitigation, and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recent technical breakthroughs in large language models (LLMs) have enabled them to fluently generate source code. Software developers often leverage both general-purpose and code-specialized LLMs to revise existing code or even generate a whole function from scratch. These capabilities are also beneficial in no-code or low-code contexts, in which one can write programs without a technical background. However, due to their internal design, LLMs are prone to generating hallucinations, which are incorrect, nonsensical, and not justifiable information but difficult to identify its presence. This problem also occurs when generating source code. Once hallucinated code is produced, it is often challenging for users to identify and fix it, especially when such hallucinations can be identified under specific execution paths. As a result, the hallucinated code may remain unnoticed within the codebase. This survey investigates recent studies and techniques relevant to hallucinations generated by CodeLLMs. We categorize the types of hallucinations in the code generated by CodeLLMs, review existing benchmarks and mitigation strategies, and identify open challenges. Based on these findings, this survey outlines further research directions in the detection and removal of hallucinations produced by CodeLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20781v1">Using LLMs in Generating Design Rationale for Software Architecture Decisions</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 28 pages, 5 images, 7 tables, Manuscript submitted to a journal (2025)
    </div>
    <details class="paper-abstract">
      Design Rationale (DR) for software architecture decisions refers to the reasoning underlying architectural choices, which provides valuable insights into the different phases of the architecting process throughout software development. However, in practice, DR is often inadequately documented due to a lack of motivation and effort from developers. With the recent advancements in Large Language Models (LLMs), their capabilities in text comprehension, reasoning, and generation may enable the generation and recovery of DR for architecture decisions. In this study, we evaluated the performance of LLMs in generating DR for architecture decisions. First, we collected 50 Stack Overflow (SO) posts, 25 GitHub issues, and 25 GitHub discussions related to architecture decisions to construct a dataset of 100 architecture-related problems. Then, we selected five LLMs to generate DR for the architecture decisions with three prompting strategies, including zero-shot, chain of thought (CoT), and LLM-based agents. With the DR provided by human experts as ground truth, the Precision of LLM-generated DR with the three prompting strategies ranges from 0.267 to 0.278, Recall from 0.627 to 0.715, and F1-score from 0.351 to 0.389. Additionally, 64.45% to 69.42% of the arguments of DR not mentioned by human experts are also helpful, 4.12% to 4.87% of the arguments have uncertain correctness, and 1.59% to 3.24% of the arguments are potentially misleading. Based on the results, we further discussed the pros and cons of the three prompting strategies and the strengths and limitations of the DR generated by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20013v2">LLM-Generated Fake News Induces Truth Decay in News Ecosystem: A Case Study on Neural News Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 ACM SIGIR 2025 Full Paper
    </div>
    <details class="paper-abstract">
      Online fake news moderation now faces a new challenge brought by the malicious use of large language models (LLMs) in fake news production. Though existing works have shown LLM-generated fake news is hard to detect from an individual aspect, it remains underexplored how its large-scale release will impact the news ecosystem. In this study, we develop a simulation pipeline and a dataset with ~56k generated news of diverse types to investigate the effects of LLM-generated fake news within neural news recommendation systems. Our findings expose a truth decay phenomenon, where real news is gradually losing its advantageous position in news ranking against fake news as LLM-generated news is involved in news recommendation. We further provide an explanation about why truth decay occurs from a familiarity perspective and show the positive correlation between perplexity and news ranking. Finally, we discuss the threats of LLM-generated fake news and provide possible countermeasures. We urge stakeholders to address this emerging challenge to preserve the integrity of news ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16563v3">Enhancing LLM-Based Agents via Global Planning and Hierarchical Execution</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Intelligent agent systems based on Large Language Models (LLMs) have shown great potential in real-world applications. However, existing agent frameworks still face critical limitations in task planning and execution, restricting their effectiveness and generalizability. Specifically, current planning methods often lack clear global goals, leading agents to get stuck in local branches, or produce non-executable plans. Meanwhile, existing execution mechanisms struggle to balance complexity and stability, and their limited action space restricts their ability to handle diverse real-world tasks. To address these limitations, we propose GoalAct, a novel agent framework that introduces a continuously updated global planning mechanism and integrates a hierarchical execution strategy. GoalAct decomposes task execution into high-level skills, including searching, coding, writing and more, thereby reducing planning complexity while enhancing the agents' adaptability across diverse task scenarios. We evaluate GoalAct on LegalAgentBench, a benchmark with multiple types of legal tasks that require the use of multiple types of tools. Experimental results demonstrate that GoalAct achieves state-of-the-art (SOTA) performance, with an average improvement of 12.22% in success rate. These findings highlight GoalAct's potential to drive the development of more advanced intelligent agent systems, making them more effective across complex real-world applications. Our code can be found at https://github.com/cjj826/GoalAct.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20699v1">Can LLMs Detect Intrinsic Hallucinations in Paraphrasing and Machine Translation?</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      A frequently observed problem with LLMs is their tendency to generate output that is nonsensical, illogical, or factually incorrect, often referred to broadly as hallucination. Building on the recently proposed HalluciGen task for hallucination detection and generation, we evaluate a suite of open-access LLMs on their ability to detect intrinsic hallucinations in two conditional generation tasks: translation and paraphrasing. We study how model performance varies across tasks and language and we investigate the impact of model size, instruction tuning, and prompt choice. We find that performance varies across models but is consistent across prompts. Finally, we find that NLI models perform comparably well, suggesting that LLM-based detectors are not the only viable option for this specific task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20653v1">ComplexVCoder: An LLM-Driven Framework for Systematic Generation of Complex Verilog Code</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Recent advances have demonstrated the promising capabilities of large language models (LLMs) in generating register-transfer level (RTL) code, such as Verilog. However, existing LLM-based frameworks still face significant challenges in accurately handling the complexity of real-world RTL designs, particularly those that are large-scale and involve multi-level module instantiations. To address this issue, we present ComplexVCoder, an open-source LLM-driven framework that enhances both the generation quality and efficiency of complex Verilog code. Specifically, we introduce a two-stage generation mechanism, which leverages an intermediate representation to enable a more accurate and structured transition from natural language descriptions to intricate Verilog designs. In addition, we introduce a rule-based alignment method and a domain-specific retrieval-augmented generation (RAG) to further improve the correctness of the synthesized code by incorporating relevant design knowledge during generation. To evaluate our approach, we construct a comprehensive dataset comprising 55 complex Verilog designs derived from real-world implementations. We also release an open-source benchmark suite for systematically assessing the quality of auto-generated RTL code together with the ComplexVCoder framework. Experimental results show that ComplexVCoder outperforms SOTA frameworks such as CodeV and RTLCoder by 14.6% and 22.2%, respectively, in terms of function correctness on complex Verilog benchmarks. Furthermore, ComplexVcoder achieves comparable generation performances in terms of functionality correctness using a lightweight 32B model (Qwen2.5), rivaling larger-scale models such as GPT-3.5 and DeepSeek-V3.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20644v1">Combatting Dimensional Collapse in LLM Pre-Training Data via Diversified File Selection</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Selecting high-quality pre-training data for large language models (LLMs) is crucial for enhancing their overall performance under limited computation budget, improving both training and sample efficiency. Recent advancements in file selection primarily rely on using an existing or trained proxy model to assess the similarity of samples to a target domain, such as high quality sources BookCorpus and Wikipedia. However, upon revisiting these methods, the domain-similarity selection criteria demonstrates a diversity dilemma, i.e.dimensional collapse in the feature space, improving performance on the domain-related tasks but causing severe degradation on generic performance. To prevent collapse and enhance diversity, we propose a DiverSified File selection algorithm (DiSF), which selects the most decorrelated text files in the feature space. We approach this with a classical greedy algorithm to achieve more uniform eigenvalues in the feature covariance matrix of the selected texts, analyzing its approximation to the optimal solution under a formulation of $\gamma$-weakly submodular optimization problem. Empirically, we establish a benchmark and conduct extensive experiments on the TinyLlama architecture with models from 120M to 1.1B parameters. Evaluating across nine tasks from the Harness framework, DiSF demonstrates a significant improvement on overall performance. Specifically, DiSF saves 98.5% of 590M training files in SlimPajama, outperforming the full-data pre-training within a 50B training budget, and achieving about 1.5x training efficiency and 5x data efficiency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20643v1">Cooking Up Creativity: A Cognitively-Inspired Approach for Enhancing LLM Creativity through Structured Representations</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 10 pages, 8 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at countless tasks, yet struggle with creativity. In this paper, we introduce a novel approach that couples LLMs with structured representations and cognitively inspired manipulations to generate more creative and diverse ideas. Our notion of creativity goes beyond superficial token-level variations; rather, we explicitly recombine structured representations of existing ideas, allowing our algorithm to effectively explore the more abstract landscape of ideas. We demonstrate our approach in the culinary domain with DishCOVER, a model that generates creative recipes. Experiments comparing our model's results to those of GPT-4o show greater diversity. Domain expert evaluations reveal that our outputs, which are mostly coherent and feasible culinary creations, significantly surpass GPT-4o in terms of novelty, thus outperforming it in creative generation. We hope our work inspires further research into structured creativity in AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20612v1">The Hidden Risks of LLM-Generated Web Application Code: A Security-Centric Evaluation of Code Generation Capabilities in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has enhanced software development processes, minimizing the time and effort required for coding and enhancing developer productivity. However, despite their potential benefits, code generated by LLMs has been shown to generate insecure code in controlled environments, raising critical concerns about their reliability and security in real-world applications. This paper uses predefined security parameters to evaluate the security compliance of LLM-generated code across multiple models, such as ChatGPT, DeepSeek, Claude, Gemini and Grok. The analysis reveals critical vulnerabilities in authentication mechanisms, session management, input validation and HTTP security headers. Although some models implement security measures to a limited extent, none fully align with industry best practices, highlighting the associated risks in automated software development. Our findings underscore that human expertise is crucial to ensure secure software deployment or review of LLM-generated code. Also, there is a need for robust security assessment frameworks to enhance the reliability of LLM-generated code in real-world applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.01070v3">An Inquiry into Datacenter TCO for LLM Inference with FP8</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) continue to scale, their inference demands present significant challenges, particularly due to the high power consumption of AI accelerators in datacenters. These facilities require specialized cooling and power management systems, substantially increasing the total cost of ownership (TCO) for cloud service providers (CSPs). In this work, we analyze the computational characteristics and constraints of LLM inference from a TCO perspective, focusing on two representative accelerators: the Gaudi 2 and NVIDIA H100. We present a generalizable framework that enables CSPs to compare and select AI accelerators according to diverse operational requirements. Using this model, we analyze the impact of FP8 precision and LLM inference workload characteristics as key factors influencing TCO. We investigate FP8 quantization, which is gaining adoption in LLM training, as a technique to improve inference throughput while maintaining cost efficiency. Furthermore, our analysis of LLM inference workloads reveals that performance on thin GEMMs, which dominate the decode phase, can have a greater impact than theoretical hardware peak performance. By studying the interaction between power consumption, quantization strategies, and hardware architecture, we offer insights that support informed deployment decisions and guide future accelerator designs to improve the TCO of LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.17593v2">Leveraging Memory Retrieval to Enhance LLM-based Generative Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 Accepted by WWW'2025
    </div>
    <details class="paper-abstract">
      Leveraging Large Language Models (LLMs) to harness user-item interaction histories for item generation has emerged as a promising paradigm in generative recommendation. However, the limited context window of LLMs often restricts them to focusing on recent user interactions only, leading to the neglect of long-term interests involved in the longer histories. To address this challenge, we propose a novel Automatic Memory-Retrieval framework (AutoMR), which is capable of storing long-term interests in the memory and extracting relevant information from it for next-item generation within LLMs. Extensive experimental results on two real-world datasets demonstrate the effectiveness of our proposed AutoMR framework in utilizing long-term interests for generative recommendation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.05676v2">Efficiency Unleashed: Inference Acceleration for LLM-based Recommender Systems with Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 Accepted by SIGIR'25
    </div>
    <details class="paper-abstract">
      The past few years have witnessed a growing interest in LLM-based recommender systems (RSs), although their industrial deployment remains in a preliminary stage. Most existing deployments leverage LLMs offline as feature enhancers, generating augmented knowledge for downstream tasks. However, in recommendation scenarios with numerous users and items, even offline knowledge generation with LLMs demands significant time and computational resources. This inefficiency arises from the autoregressive nature of LLMs. A promising solution is speculative decoding, a Draft-Then-Verify approach that increases the number of tokens generated per decoding step. In this work, we first identify recommendation knowledge generation as a highly fitting use case for retrieval-based speculative decoding. Then, we discern its two characteristics: (1) the vast number of items and users in RSs leads to retrieval inefficiency, and (2) RSs exhibit high diversity tolerance for LLM-generated text. Building on these insights, we introduce Lossless Acceleration via Speculative Decoding for LLM-based Recommender Systems (LASER), which features a Customized Retrieval Pool to enhance retrieval efficiency and Relaxed Verification to improve the acceptance rate of draft tokens. LASER achieves a 3-5x speedup on public datasets and saves about 67\% of computational resources during the online A/B test on a large-scale advertising scenario with lossless downstream recommendation performance. Our code is available at https://github.com/YunjiaXi/LASER
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20493v1">Token-Efficient Prompt Injection Attack: Provoking Cessation in LLM Reasoning via Adaptive Token Compression</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      While reasoning large language models (LLMs) demonstrate remarkable performance across various tasks, they also contain notable security vulnerabilities. Recent research has uncovered a "thinking-stopped" vulnerability in DeepSeek-R1, where model-generated reasoning tokens can forcibly interrupt the inference process, resulting in empty responses that compromise LLM-integrated applications. However, existing methods triggering this vulnerability require complex mathematical word problems with long prompts--even exceeding 5,000 tokens. To reduce the token cost and formally define this vulnerability, we propose a novel prompt injection attack named "Reasoning Interruption Attack", based on adaptive token compression. We demonstrate that simple standalone arithmetic tasks can effectively trigger this vulnerability, and the prompts based on such tasks exhibit simpler logical structures than mathematical word problems. We develop a systematic approach to efficiently collect attack prompts and an adaptive token compression framework that utilizes LLMs to automatically compress these prompts. Experiments show our compression framework significantly reduces prompt length while maintaining effective attack capabilities. We further investigate the attack's performance via output prefix and analyze the underlying causes of the vulnerability, providing valuable insights for improving security in reasoning LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20484v1">Enhancing LLM Language Adaption through Cross-lingual In-Context Pre-training</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 12 pages, 6 figures, Under Review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit remarkable multilingual capabilities despite English-dominated pre-training, attributed to cross-lingual mechanisms during pre-training. Existing methods for enhancing cross-lingual transfer remain constrained by parallel resources, suffering from limited linguistic and domain coverage. We propose Cross-lingual In-context Pre-training (CrossIC-PT), a simple and scalable approach that enhances cross-lingual transfer by leveraging semantically related bilingual texts via simple next-word prediction. We construct CrossIC-PT samples by interleaving semantic-related bilingual Wikipedia documents into a single context window. To access window size constraints, we implement a systematic segmentation policy to split long bilingual document pairs into chunks while adjusting the sliding window mechanism to preserve contextual coherence. We further extend data availability through a semantic retrieval framework to construct CrossIC-PT samples from web-crawled corpus. Experimental results demonstrate that CrossIC-PT improves multilingual performance on three models (Llama-3.1-8B, Qwen2.5-7B, and Qwen2.5-1.5B) across six target languages, yielding performance gains of 3.79%, 3.99%, and 1.95%, respectively, with additional improvements after data augmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.04199v2">Investigating and Mitigating Stereotype-aware Unfairness in LLM-based Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated unprecedented language understanding and reasoning capabilities to capture diverse user preferences and advance personalized recommendations. Despite the growing interest in LLM-based recommendations, unique challenges are brought to the trustworthiness of LLM-based recommender systems (LLM-RS). Compared to unique user/item representations in conventional recommender systems, users and items share the textual representation (e.g., word embeddings) in LLM-based recommendations. Recent studies have revealed that LLMs are likely to inherit stereotypes that are embedded ubiquitously in word embeddings, due to their training on large-scale uncurated datasets. This leads to LLM-RS exhibiting stereotypical linguistic associations between users and items, causing a form of two-sided (i.e., user-to-item) recommendation fairness. However, there remains a lack of studies investigating the unfairness of LLM-RS due to intrinsic stereotypes, which can simultaneously involve user and item groups. To bridge this gap, this study reveals a new variant of fairness between stereotype groups containing both users and items, to quantify discrimination against stereotypes in LLM-RS. Moreover, in this paper, to mitigate stereotype-aware unfairness in textual user and item representations, we propose a novel framework named Mixture-of-Stereotypes (MoS). In particular, an insightful stereotype-wise routing strategy over multiple stereotype-relevant experts is designed, aiming to learn unbiased representations against different stereotypes in LLM-RS. Extensive experiments are conducted to analyze the influence of stereotype-aware fairness in LLM-RS and the effectiveness of our proposed methods, which consistently outperform competitive benchmarks under various fairness settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20462v1">TAMO:Fine-Grained Root Cause Analysis via Tool-Assisted LLM Agent with Multi-Modality Observation Data</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      With the development of distributed systems, microservices and cloud native technologies have become central to modern enterprise software development. Despite bringing significant advantages, these technologies also increase system complexity and operational challenges. Traditional root cause analysis (RCA) struggles to achieve automated fault response, heavily relying on manual intervention. In recent years, large language models (LLMs) have made breakthroughs in contextual inference and domain knowledge integration, providing new solutions for Artificial Intelligence for Operations (AIOps). However, Existing LLM-based approaches face three key challenges: text input constraints, dynamic service dependency hallucinations, and context window limitations. To address these issues, we propose a tool-assisted LLM agent with multi-modality observation data, namely TAMO, for fine-grained RCA. It unifies multi-modal observational data into time-aligned representations to extract consistent features and employs specialized root cause localization and fault classification tools for perceiving the contextual environment. This approach overcomes the limitations of LLM in handling real-time changing service dependencies and raw observational data and guides LLM to generate repair strategies aligned with system contexts by structuring key information into a prompt. Experimental results show that TAMO performs well in root cause analysis when dealing with public datasets characterized by heterogeneity and common fault types, demonstrating its effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20452v1">Enhancing News Recommendation with Hierarchical LLM Prompting</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Personalized news recommendation systems often struggle to effectively capture the complexity of user preferences, as they rely heavily on shallow representations, such as article titles and abstracts. To address this problem, we introduce a novel method, namely PNR-LLM, for Large Language Models for Personalized News Recommendation. Specifically, PNR-LLM harnesses the generation capabilities of LLMs to enrich news titles and abstracts, and consequently improves recommendation quality. PNR-LLM contains a novel module, News Enrichment via LLMs, which generates deeper semantic information and relevant entities from articles, transforming shallow contents into richer representations. We further propose an attention mechanism to aggregate enriched semantic- and entity-level data, forming unified user and news embeddings that reveal a more accurate user-news match. Extensive experiments on MIND datasets show that PNR-LLM outperforms state-of-the-art baselines. Moreover, the proposed data enrichment module is model-agnostic, and we empirically show that applying our proposed module to multiple existing models can further improve their performance, verifying the advantage of our design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20444v1">On Psychology of AI -- Does Primacy Effect Affect ChatGPT and Other LLMs?</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      We study the primacy effect in three commercial LLMs: ChatGPT, Gemini and Claude. We do this by repurposing the famous experiment Asch (1946) conducted using human subjects. The experiment is simple, given two candidates with equal descriptions which one is preferred if one description has positive adjectives first before negative ones and another description has negative adjectives followed by positive ones. We test this in two experiments. In one experiment, LLMs are given both candidates simultaneously in the same prompt, and in another experiment, LLMs are given both candidates separately. We test all the models with 200 candidate pairs. We found that, in the first experiment, ChatGPT preferred the candidate with positive adjectives listed first, while Gemini preferred both equally often. Claude refused to make a choice. In the second experiment, ChatGPT and Claude were most likely to rank both candidates equally. In the case where they did not give an equal rating, both showed a clear preference to a candidate that had negative adjectives listed first. Gemini was most likely to prefer a candidate with negative adjectives listed first.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20437v1">GaLore 2: Large-Scale LLM Pre-Training by Gradient Low-Rank Projection</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have revolutionized natural language understanding and generation but face significant memory bottlenecks during training. GaLore, Gradient Low-Rank Projection, addresses this issue by leveraging the inherent low-rank structure of weight gradients, enabling substantial memory savings without sacrificing performance. Recent works further extend GaLore from various aspects, including low-bit quantization and higher-order tensor structures. However, there are several remaining challenges for GaLore, such as the computational overhead of SVD for subspace updates and the integration with state-of-the-art training parallelization strategies (e.g., FSDP). In this paper, we present GaLore 2, an efficient and scalable GaLore framework that addresses these challenges and incorporates recent advancements. In addition, we demonstrate the scalability of GaLore 2 by pre-training Llama 7B from scratch using up to 500 billion training tokens, highlighting its potential impact on real LLM pre-training scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14634v4">Scideator: Human-LLM Scientific Idea Generation Grounded in Research-Paper Facet Recombination</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 Updated with new and improved user study
    </div>
    <details class="paper-abstract">
      The scientific ideation process often involves blending salient aspects of existing papers to create new ideas, and facet-based ideation is an established framework for idea generation. To see how large language models (LLMs) might assist in this process, we contribute a novel mixed-initiative ideation tool called Scideator. Starting from a user-provided set of scientific papers, Scideator extracts key facets -- purposes, mechanisms, and evaluations -- from these and related papers, allowing users to explore the idea space by interactively recombining facets to synthesize inventive ideas. Scideator also helps users gauge idea originality by searching the literature for overlaps, assessing idea novelty and providing explanations. To support these tasks, Scideator introduces three LLM-powered retrieval-augmented generation (RAG) modules: Analogous Paper Facet Finder, Faceted Idea Generator, and Idea Novelty Checker. In a within-subjects user study (N=22) with computer-science researchers comparing Scideator to a strong baseline, our tool provided significantly more creativity support, particularly with respect to exploration, which participants considered the most important factor for idea generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20414v1">Enhancing Leakage Attacks on Searchable Symmetric Encryption Using LLM-Based Synthetic Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Searchable Symmetric Encryption (SSE) enables efficient search capabilities over encrypted data, allowing users to maintain privacy while utilizing cloud storage. However, SSE schemes are vulnerable to leakage attacks that exploit access patterns, search frequency, and volume information. Existing studies frequently assume that adversaries possess a substantial fraction of the encrypted dataset to mount effective inference attacks, implying there is a database leakage of such documents, thus, an assumption that may not hold in real-world scenarios. In this work, we investigate the feasibility of enhancing leakage attacks under a more realistic threat model in which adversaries have access to minimal leaked data. We propose a novel approach that leverages large language models (LLMs), specifically GPT-4 variants, to generate synthetic documents that statistically and semantically resemble the real-world dataset of Enron emails. Using the email corpus as a case study, we evaluate the effectiveness of synthetic data generated via random sampling and hierarchical clustering methods on the performance of the SAP (Search Access Pattern) keyword inference attack restricted to token volumes only. Our results demonstrate that, while the choice of LLM has limited effect, increasing dataset size and employing clustering-based generation significantly improve attack accuracy, achieving comparable performance to attacks using larger amounts of real data. We highlight the growing relevance of LLMs in adversarial contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20406v1">Skill Discovery for Software Scripting Automation via Offline Simulations with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Scripting interfaces enable users to automate tasks and customize software workflows, but creating scripts traditionally requires programming expertise and familiarity with specific APIs, posing barriers for many users. While Large Language Models (LLMs) can generate code from natural language queries, runtime code generation is severely limited due to unverified code, security risks, longer response times, and higher computational costs. To bridge the gap, we propose an offline simulation framework to curate a software-specific skillset, a collection of verified scripts, by exploiting LLMs and publicly available scripting guides. Our framework comprises two components: (1) task creation, using top-down functionality guidance and bottom-up API synergy exploration to generate helpful tasks; and (2) skill generation with trials, refining and validating scripts based on execution feedback. To efficiently navigate the extensive API landscape, we introduce a Graph Neural Network (GNN)-based link prediction model to capture API synergy, enabling the generation of skills involving underutilized APIs and expanding the skillset's diversity. Experiments with Adobe Illustrator demonstrate that our framework significantly improves automation success rates, reduces response time, and saves runtime token costs compared to traditional runtime code generation. This is the first attempt to use software scripting interfaces as a testbed for LLM-based systems, highlighting the advantages of leveraging execution feedback in a controlled environment and offering valuable insights into aligning AI capabilities with user needs in specialized software domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07127v2">Benchmarking LLMs' Judgments with No Gold Standard</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 The Thirteenth International Conference on Learning Representations (ICLR 2025)
    </div>
    <details class="paper-abstract">
      We introduce the GEM (Generative Estimator for Mutual Information), an evaluation metric for assessing language generation by Large Language Models (LLMs), particularly in generating informative judgments, without the need for a gold standard reference. GEM broadens the scenarios where we can benchmark LLM generation performance-from traditional ones, like machine translation and summarization, where gold standard references are readily available, to subjective tasks without clear gold standards, such as academic peer review. GEM uses a generative model to estimate mutual information between candidate and reference responses, without requiring the reference to be a gold standard. In experiments on a human-annotated dataset, GEM demonstrates competitive correlations with human scores compared to the state-of-the-art GPT-4o Examiner, and outperforms all other baselines. Additionally, GEM is more robust against strategic manipulations, such as rephrasing or elongation, which can artificially inflate scores under a GPT-4o Examiner. We also present GRE-bench (Generating Review Evaluation Benchmark) which evaluates LLMs based on how well they can generate high-quality peer reviews for academic research papers. Because GRE-bench is based upon GEM, it inherits its robustness properties. Additionally, GRE-bench circumvents data contamination problems (or data leakage) by using the continuous influx of new open-access research papers and peer reviews each year. We show GRE-bench results of various popular LLMs on their peer review capabilities using the ICLR2023 dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05982v2">Unmasking the Shadows: Pinpoint the Implementations of Anti-Dynamic Analysis Techniques in Malware Using LLM</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Sandboxes and other dynamic analysis processes are prevalent in malware detection systems nowadays to enhance the capability of detecting 0-day malware. Therefore, techniques of anti-dynamic analysis (TADA) are prevalent in modern malware samples, and sandboxes can suffer from false negatives and analysis failures when analyzing the samples with TADAs. In such cases, human reverse engineers will get involved in conducting dynamic analysis manually (i.e., debugging, patching), which in turn also gets obstructed by TADAs. In this work, we propose a Large Language Model (LLM) based workflow that can pinpoint the location of the TADA implementation in the code, to help reverse engineers place breakpoints used in debugging. Our evaluation shows that we successfully identified the locations of 87.80% known TADA implementations adopted from public repositories. In addition, we successfully pinpoint the locations of TADAs in 4 well-known malware samples that are documented in online malware analysis blogs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05239v2">LLM-based Automated Grading with Human-in-the-Loop</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      The rise of artificial intelligence (AI) technologies, particularly large language models (LLMs), has brought significant advancements to the field of education. Among various applications, automatic short answer grading (ASAG), which focuses on evaluating open-ended textual responses, has seen remarkable progress with the introduction of LLMs. These models not only enhance grading performance compared to traditional ASAG approaches but also move beyond simple comparisons with predefined "golden" answers, enabling more sophisticated grading scenarios, such as rubric-based evaluation. However, existing LLM-powered methods still face challenges in achieving human-level grading performance in rubric-based assessments due to their reliance on fully automated approaches. In this work, we explore the potential of LLMs in ASAG tasks by leveraging their interactive capabilities through a human-in-the-loop (HITL) approach. Our proposed framework, GradeHITL, utilizes the generative properties of LLMs to pose questions to human experts, incorporating their insights to refine grading rubrics dynamically. This adaptive process significantly improves grading accuracy, outperforming existing methods and bringing ASAG closer to human-level evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18583v2">PARD: Accelerating LLM Inference with Low-Cost PARallel Draft Model Adaptation</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 15 pages, 6 figures
    </div>
    <details class="paper-abstract">
      The autoregressive nature of large language models (LLMs) limits inference speed. Each forward pass generates only a single token and is often bottlenecked by memory bandwidth. Speculative decoding alleviates this issue using a draft-then-verify approach to accelerate token generation. However, the overhead introduced during the draft phase and the training cost of the draft model limit the efficiency and adaptability of speculative decoding. In this work, we introduce PARallel Draft (PARD), a novel speculative decoding method that enables low-cost adaptation of autoregressive draft models into parallel draft models. PARD enhances inference efficiency by predicting multiple future tokens in a single forward pass of the draft phase, and incorporates a conditional drop token method to accelerate training. Its target-independence property allows a single draft model to be applied to an entire family of different models, minimizing the adaptation cost. Our proposed conditional drop token method can improves draft model training efficiency by 3x. On our optimized inference framework, PARD accelerates LLaMA3.1-8B inference by 4.08x, achieving 311.5 tokens per second.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20371v1">DMDTEval: An Evaluation and Analysis of LLMs on Disambiguation in Multi-domain Translation</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Currently, Large Language Models (LLMs) have achieved remarkable results in machine translation. However, their performance in multi-domain translation (MDT) is less satisfactory; the meanings of words can vary across different domains, highlighting the significant ambiguity inherent in MDT. Therefore, evaluating the disambiguation ability of LLMs in MDT remains an open problem. To this end, we present an evaluation and analysis of LLMs on disambiguation in multi-domain translation (DMDTEval), our systematic evaluation framework consisting of three critical aspects: (1) we construct a translation test set with multi-domain ambiguous word annotation, (2) we curate a diverse set of disambiguation prompting templates, and (3) we design precise disambiguation metrics, and study the efficacy of various prompting strategies on multiple state-of-the-art LLMs. Our extensive experiments reveal a number of crucial findings that we believe will pave the way and also facilitate further research in the critical area of improving the disambiguation of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18598v2">BadMoE: Backdooring Mixture-of-Experts LLMs via Optimizing Routing Triggers and Infecting Dormant Experts</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Mixture-of-Experts (MoE) have emerged as a powerful architecture for large language models (LLMs), enabling efficient scaling of model capacity while maintaining manageable computational costs. The key advantage lies in their ability to route different tokens to different ``expert'' networks within the model, enabling specialization and efficient handling of diverse input. However, the vulnerabilities of MoE-based LLMs still have barely been studied, and the potential for backdoor attacks in this context remains largely unexplored. This paper presents the first backdoor attack against MoE-based LLMs where the attackers poison ``dormant experts'' (i.e., underutilized experts) and activate them by optimizing routing triggers, thereby gaining control over the model's output. We first rigorously prove the existence of a few ``dominating experts'' in MoE models, whose outputs can determine the overall MoE's output. We also show that dormant experts can serve as dominating experts to manipulate model predictions. Accordingly, our attack, namely BadMoE, exploits the unique architecture of MoE models by 1) identifying dormant experts unrelated to the target task, 2) constructing a routing-aware loss to optimize the activation triggers of these experts, and 3) promoting dormant experts to dominating roles via poisoned training data. Extensive experiments show that BadMoE successfully enforces malicious prediction on attackers' target tasks while preserving overall model utility, making it a more potent and stealthy attack than existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19959v2">From Concept to Practice: an Automated LLM-aided UVM Machine for RTL Verification</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Verification presents a major bottleneck in Integrated Circuit (IC) development, consuming nearly 70% of the total development effort. While the Universal Verification Methodology (UVM) is widely used in industry to improve verification efficiency through structured and reusable testbenches, constructing these testbenches and generating sufficient stimuli remain challenging. These challenges arise from the considerable manual coding effort required, repetitive manual execution of multiple EDA tools, and the need for in-depth domain expertise to navigate complex designs.Here, we present UVM^2, an automated verification framework that leverages Large Language Models (LLMs) to generate UVM testbenches and iteratively refine them using coverage feedback, significantly reducing manual effort while maintaining rigorous verification standards.To evaluate UVM^2, we introduce a benchmark suite comprising Register Transfer Level (RTL) designs of up to 1.6K lines of code.The results show that UVM^2 reduces testbench setup time by up to UVM^2 compared to experienced engineers, and achieve average code and function coverage of 87.44% and 89.58%, outperforming state-of-the-art solutions by 20.96% and 23.51%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.18762v2">SynLexLM: Scaling Legal LLMs with Synthetic Data and Curriculum Learning</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 9 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are powerful but often require extensive fine-tuning and large datasets for specialized domains like law. General-purpose pre-training may not capture legal nuances, and acquiring sufficient legal data is challenging. We introduce SynLexLM, a novel approach to efficiently pre-train a legal LLM. Our method employs curriculum learning, progressing from simple to complex legal texts and queries, combined with synthetic data augmentation using models like Gemini Pro to address data scarcity. We aim to achieve improved performance on legal benchmarks (BigLaw-Bench, EUR-Lex-Sum) compared to traditional models and fine-tuned versions. Preliminary work involves generating synthetic QA pairs reflecting legal reasoning. This work aims to enhance legal document analysis and research tools, potentially democratizing access to advanced legal AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.19442v5">Does Generative AI speak Nigerian-Pidgin?: Issues about Representativeness and Bias for Multilingualism in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 Accepted to NAACL 2025 (findings), please cite ACL anthology reference on https://aclanthology.org/2025.findings-naacl.85/
    </div>
    <details class="paper-abstract">
      Nigeria is a multilingual country with 500+ languages. Naija is a Nigerian Pidgin spoken by approximately 120M speakers and it is a mixed language (e.g., English, Portuguese, Yoruba, Hausa and Igbo). Although it has mainly been a spoken language until recently, there are some online platforms (e.g., Wikipedia), publishing in written Naija as well. West African Pidgin English (WAPE) is also spoken in Nigeria and it is used by BBC to broadcast news on the internet to a wider audience not only in Nigeria but also in other West African countries (e.g., Cameroon and Ghana). Through statistical analyses and Machine Translation experiments, our paper shows that these two pidgin varieties do not represent each other (i.e., there are linguistic differences in word order and vocabulary) and Generative AI operates only based on WAPE. In other words, Naija is underrepresented in Generative AI, and it is hard to teach LLMs with few examples. In addition to the statistical analyses, we also provide historical information on both pidgins as well as insights from the interviews conducted with volunteer Wikipedia contributors in Naija.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.16557v3">Patched RTC: evaluating LLMs for diverse software development tasks</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      This paper introduces Patched Round-Trip Correctness (Patched RTC), a novel evaluation technique for Large Language Models (LLMs) applied to diverse software development tasks, particularly focusing on "outer loop" activities such as bug fixing, code review, and documentation updates. Patched RTC extends the original Round-Trip Correctness method to work with any LLM and downstream task, offering a self-evaluating framework that measures consistency and robustness of model responses without human intervention. The study demonstrates a correlation between Patched RTC scores and task-specific accuracy metrics, presenting it as an alternative to the LLM-as-Judge paradigm for open-domain task evaluation. We implement Patched RTC in an open-source framework called patchwork, allowing for transparent evaluation during inference across various patchflows. Experiments comparing GPT-3.5 and GPT-4 models across different software development tasks reveal that Patched RTC effectively distinguishes model performance and task difficulty. The paper also explores the impact of consistency prompts on improving model accuracy, suggesting that Patched RTC can guide prompt refinement and model selection for complex software development workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.01698v2">Demystifying AI Platform Design for Distributed Inference of Next-Generation LLM models</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 19 Pages, https://github.com/abhibambhaniya/GenZ-LLM-Analyzer, https://genz-llm-analyzer.streamlit.app/
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable performance across a wide range of applications, often outperforming human experts. However, deploying these gigantic models efficiently for diverse inference use cases requires carefully designed hardware platforms with ample computing, memory, and network resources. With constant innovation in LLM serving optimizations and model architecture evolving at breakneck speed, the hardware requirements to meet Service Level Objectives (SLOs) remain an open research question. To answer the question, we present an analytical tool, GenZ, to efficiently navigate the relationship between diverse LLM model architectures(Dense, GQA, MoE, Mamba), LLM serving optimizations(Chunking, Speculative decoding, quanitization), and AI platform design parameters. Our tool estimates LLM inference performance metrics for the given scenario. We have validated against real hardware platforms running various different LLM models, achieving a max geomean error of 5.82.We use GenZ to identify compute, memory capacity, memory bandwidth, network latency, and network bandwidth requirements across diverse LLM inference use cases. We also study diverse architectural choices in use today (inspired by LLM serving platforms from several vendors) to help inform computer architects designing next-generation AI hardware accelerators and platforms. The trends and insights derived from GenZ can guide AI engineers deploying LLMs as well as computer architects designing next-generation hardware accelerators and platforms. Ultimately, this work sheds light on the platform design considerations for unlocking the full potential of large language models across a spectrum of applications. The source code is available at https://github.com/abhibambhaniya/GenZ-LLM-Analyzer . Users can also be tried it on at https://genz-llm-analyzer.streamlit.app/ without any setup on your web browser.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21211v1">A Cost-Effective LLM-based Approach to Identify Wildlife Trafficking in Online Marketplaces</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Wildlife trafficking remains a critical global issue, significantly impacting biodiversity, ecological stability, and public health. Despite efforts to combat this illicit trade, the rise of e-commerce platforms has made it easier to sell wildlife products, putting new pressure on wild populations of endangered and threatened species. The use of these platforms also opens a new opportunity: as criminals sell wildlife products online, they leave digital traces of their activity that can provide insights into trafficking activities as well as how they can be disrupted. The challenge lies in finding these traces. Online marketplaces publish ads for a plethora of products, and identifying ads for wildlife-related products is like finding a needle in a haystack. Learning classifiers can automate ad identification, but creating them requires costly, time-consuming data labeling that hinders support for diverse ads and research questions. This paper addresses a critical challenge in the data science pipeline for wildlife trafficking analytics: generating quality labeled data for classifiers that select relevant data. While large language models (LLMs) can directly label advertisements, doing so at scale is prohibitively expensive. We propose a cost-effective strategy that leverages LLMs to generate pseudo labels for a small sample of the data and uses these labels to create specialized classification models. Our novel method automatically gathers diverse and representative samples to be labeled while minimizing the labeling costs. Our experimental evaluation shows that our classifiers achieve up to 95% F1 score, outperforming LLMs at a lower cost. We present real use cases that demonstrate the effectiveness of our approach in enabling analyses of different aspects of wildlife trafficking.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21205v1">SecRepoBench: Benchmarking LLMs for Secure Code Generation in Real-World Repositories</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      This paper introduces SecRepoBench, a benchmark to evaluate LLMs on secure code generation in real-world repositories. SecRepoBench has 318 code generation tasks in 27 C/C++ repositories, covering 15 CWEs. We evaluate 19 state-of-the-art LLMs using our benchmark and find that the models struggle with generating correct and secure code. In addition, the performance of LLMs to generate self-contained programs as measured by prior benchmarks do not translate to comparative performance at generating secure and correct code at the repository level in SecRepoBench. We show that the state-of-the-art prompt engineering techniques become less effective when applied to the repository level secure code generation problem. We conduct extensive experiments, including an agentic technique to generate secure code, to demonstrate that our benchmark is currently the most difficult secure coding benchmark, compared to previous state-of-the-art benchmarks. Finally, our comprehensive analysis provides insights into potential directions for enhancing the ability of LLMs to generate correct and secure code in real-world repositories.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21202v1">Automatic Legal Writing Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Despite the recent advances in Large Language Models, benchmarks for evaluating legal writing remain scarce due to the inherent complexity of assessing open-ended responses in this domain. One of the key challenges in evaluating language models on domain-specific tasks is finding test datasets that are public, frequently updated, and contain comprehensive evaluation guidelines. The Brazilian Bar Examination meets these requirements. We introduce oab-bench, a benchmark comprising 105 questions across seven areas of law from recent editions of the exam. The benchmark includes comprehensive evaluation guidelines and reference materials used by human examiners to ensure consistent grading. We evaluate the performance of four LLMs on oab-bench, finding that Claude-3.5 Sonnet achieves the best results with an average score of 7.93 out of 10, passing all 21 exams. We also investigated whether LLMs can serve as reliable automated judges for evaluating legal writing. Our experiments show that frontier models like OpenAI's o1 achieve a strong correlation with human scores when evaluating approved exams, suggesting their potential as reliable automated evaluators despite the inherently subjective nature of legal writing assessment. The source code and the benchmark -- containing questions, evaluation guidelines, model-generated responses, and their respective automated evaluations -- are publicly available.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19394v2">LLMs for Engineering: Teaching Models to Design High Powered Rockets</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have transformed software engineering, but their application to physical engineering domains remains underexplored. This paper evaluates LLMs' capabilities in high-powered rocketry design through RocketBench, a benchmark connecting LLMs to high-fidelity rocket simulations. We test models on two increasingly complex design tasks: target altitude optimization and precision landing challenges. Our findings reveal that while state-of-the-art LLMs demonstrate strong baseline engineering knowledge, they struggle to iterate on their designs when given simulation results and ultimately plateau below human performance levels. However, when enhanced with reinforcement learning (RL), we show that a 7B parameter model outperforms both SoTA foundation models and human experts. This research demonstrates that RL-trained LLMs can serve as effective tools for complex engineering optimization, potentially transforming engineering domains beyond software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21187v1">LIFT: LLM-Based Pragma Insertion for HLS via GNN Supervised Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      FPGAs are increasingly adopted in datacenter environments for their reconfigurability and energy efficiency. High-Level Synthesis (HLS) tools have eased FPGA programming by raising the abstraction level from RTL to untimed C/C++, yet attaining high performance still demands expert knowledge and iterative manual insertion of optimization pragmas to modify the microarchitecture. To address this challenge, we propose LIFT, a large language model (LLM)-based coding assistant for HLS that automatically generates performance-critical pragmas given a C/C++ design. We fine-tune the LLM by tightly integrating and supervising the training process with a graph neural network (GNN), combining the sequential modeling capabilities of LLMs with the structural and semantic understanding of GNNs necessary for reasoning over code and its control/data dependencies. On average, LIFT produces designs that improve performance by 3.52x and 2.16x than prior state-of the art AutoDSE and HARP respectively, and 66x than GPT-4o.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21174v1">Efficient LLMs with AMP: Attention Heads and MLP Pruning</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 To be published in International Joint Conference on Neural Networks (IJCNN), 2025
    </div>
    <details class="paper-abstract">
      Deep learning drives a new wave in computing systems and triggers the automation of increasingly complex problems. In particular, Large Language Models (LLMs) have significantly advanced cognitive tasks, often matching or even surpassing human-level performance. However, their extensive parameters result in high computational costs and slow inference, posing challenges for deployment in resource-limited settings. Among the strategies to overcome the aforementioned challenges, pruning emerges as a successful mechanism since it reduces model size while maintaining predictive ability. In this paper, we introduce AMP: Attention Heads and MLP Pruning, a novel structured pruning method that efficiently compresses LLMs by removing less critical structures within Multi-Head Attention (MHA) and Multilayer Perceptron (MLP). By projecting the input data onto weights, AMP assesses structural importance and overcomes the limitations of existing techniques, which often fall short in flexibility or efficiency. In particular, AMP surpasses the current state-of-the-art on commonsense reasoning tasks by up to 1.49 percentage points, achieving a 30% pruning ratio with minimal impact on zero-shot task performance. Moreover, AMP also improves inference speeds, making it well-suited for deployment in resource-constrained environments. We confirm the flexibility of AMP on different families of LLMs, including LLaMA and Phi.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17651v2">APEX: An Extensible and Dynamism-Aware Simulator for Automated Parallel Execution in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Efficiently serving Large Language Models (LLMs) requires selecting an optimal parallel execution plan, balancing computation, memory, and communication overhead. However, determining the best strategy is challenging due to varying parallelism techniques (data, pipeline, tensor) and workload characteristics (e.g., compute-intensive tasks with long prompts vs. memory-intensive tasks with long generation). We propose APEX, an LLM serving system simulator that efficiently identifies optimal parallel execution plans by considering key factors of LLM serving systems, such as memory usage, batching behavior, etc. APEX performs dynamism-aware simulation to model iteration-level batching, and leverages LLMs' repetitive structure to reduce design space, scaling efficiently to trillion-scale models. APEX abstracts the key components of LLM serving systems, including the model, batching module, quantization formats, and device clusters, enabling the simulator to be general and extensible. Simulating on a CPU, APEX evaluates execution plans for various device clusters, covering diverse LLMs and workloads. APEX finds plans up to 3.37x faster than heuristics, and also plans that reduce energy consumption by up to 45% compared to latency-optimal plans. APEX performs comprehensive evaluations, reporting key system metrics like time per output token and time to first token, which can help service providers meet SLOs. APEX identifies an optimal plan within 15 minutes on a CPU, making it 71x faster and 1234x more cost-effective than cloud-based GPU deployment. APEX can be accessed at https://github.com/microsoft/apex_plus
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21132v1">LLM Enhancer: Merged Approach using Vector Embedding for Reducing Large Language Model Hallucinations with External Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-04-29
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), such as ChatGPT, have demonstrated the capability to generate human like, natural responses across a range of tasks, including task oriented dialogue and question answering. However, their application in real world, critical scenarios is often hindered by a tendency to produce inaccurate information and a limited ability to leverage external knowledge sources. This paper introduces the LLM ENHANCER system, designed to integrate multiple online sources such as Google, Wikipedia, and DuckDuckGo to enhance data accuracy. The LLMs employed within this system are open source. The data acquisition process for the LLM ENHANCER system operates in parallel, utilizing custom agent tools to manage the flow of information. Vector embeddings are used to identify the most pertinent information, which is subsequently supplied to the LLM for user interaction. The LLM ENHANCER system mitigates hallucinations in chat based LLMs while preserving response naturalness and accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.01447v1">LLM-Enabled EV Charging Stations Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-04-29
      | 💬 5 pages, 4 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Charging infrastructure is not expanding quickly enough to accommodate the increasing usage of Electric Vehicles (EVs). For this reason, EV owners experience extended waiting periods, range anxiety, and overall dissatisfaction. Challenges, such as fragmented data and the complexity of integrating factors like location, energy pricing, and user preferences, make the current recommendation systems ineffective. To overcome these limitations, we propose RecomBot, which is a Large Language Model (LLM)-powered prompt-based recommender system that dynamically suggests optimal Charging Stations (CSs) using real-time heterogeneous data. By leveraging natural language reasoning and fine-tuning EV-specific datasets, RecomBot enhances personalization, improves charging efficiency, and adapts to various EV types, offering a scalable solution for intelligent EV recommendation systems. Through testing across various prompt engineering scenarios, the results obtained underline the capability and efficiency of the proposed model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20022v1">Better To Ask in English? Evaluating Factual Accuracy of Multilingual LLMs in English and Low-Resource Languages</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Multilingual Large Language Models (LLMs) have demonstrated significant effectiveness across various languages, particularly in high-resource languages such as English. However, their performance in terms of factual accuracy across other low-resource languages, especially Indic languages, remains an area of investigation. In this study, we assess the factual accuracy of LLMs - GPT-4o, Gemma-2-9B, Gemma-2-2B, and Llama-3.1-8B - by comparing their performance in English and Indic languages using the IndicQuest dataset, which contains question-answer pairs in English and 19 Indic languages. By asking the same questions in English and their respective Indic translations, we analyze whether the models are more reliable for regional context questions in Indic languages or when operating in English. Our findings reveal that LLMs often perform better in English, even for questions rooted in Indic contexts. Notably, we observe a higher tendency for hallucination in responses generated in low-resource Indic languages, highlighting challenges in the multilingual understanding capabilities of current LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20016v1">Applying LLM-Powered Virtual Humans to Child Interviews in Child-Centered Design</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 This paper has been accepted as a Work-in-Progress (WiP) paper in the 24th annual ACM Interaction Design and Children (IDC) Conference
    </div>
    <details class="paper-abstract">
      In child-centered design, directly engaging children is crucial for deeply understanding their experiences. However, current research often prioritizes adult perspectives, as interviewing children involves unique challenges such as environmental sensitivities and the need for trust-building. AI-powered virtual humans (VHs) offer a promising approach to facilitate engaging and multimodal interactions with children. This study establishes key design guidelines for LLM-powered virtual humans tailored to child interviews, standardizing multimodal elements including color schemes, voice characteristics, facial features, expressions, head movements, and gestures. Using ChatGPT-based prompt engineering, we developed three distinct Human-AI workflows (LLM-Auto, LLM-Interview, and LLM-Analyze) and conducted a user study involving 15 children aged 6 to 12. The results indicated that the LLM-Analyze workflow outperformed the others by eliciting longer responses, achieving higher user experience ratings, and promoting more effective child engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20013v1">LLM-Generated Fake News Induces Truth Decay in News Ecosystem: A Case Study on Neural News Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 ACM SIGIR 2025 Full Paper
    </div>
    <details class="paper-abstract">
      Online fake news moderation now faces a new challenge brought by the malicious use of large language models (LLMs) in fake news production. Though existing works have shown LLM-generated fake news is hard to detect from an individual aspect, it remains underexplored how its large-scale release will impact the news ecosystem. In this study, we develop a simulation pipeline and a dataset with ~56k generated news of diverse types to investigate the effects of LLM-generated fake news within neural news recommendation systems. Our findings expose a truth decay phenomenon, where real news is gradually losing its advantageous position in news ranking against fake news as LLM-generated news is involved in news recommendation. We further provide an explanation about why truth decay occurs from a familiarity perspective and show the positive correlation between perplexity and news ranking. Finally, we discuss the threats of LLM-generated fake news and provide possible countermeasures. We urge stakeholders to address this emerging challenge to preserve the integrity of news ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20006v1">Chatbot Arena Meets Nuggets: Towards Explanations and Diagnostics in the Evaluation of LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 10 pages, 8 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Battles, or side-by-side comparisons in so called arenas that elicit human preferences, have emerged as a popular approach to assessing the output quality of LLMs. Recently, this idea has been extended to retrieval-augmented generation (RAG) systems. While undoubtedly representing an advance in evaluation, battles have at least two drawbacks, particularly in the context of complex information-seeking queries: they are neither explanatory nor diagnostic. Recently, the nugget evaluation methodology has emerged as a promising approach to evaluate the quality of RAG answers. Nuggets decompose long-form LLM-generated answers into atomic facts, highlighting important pieces of information necessary in a "good" response. In this work, we apply our AutoNuggetizer framework to analyze data from roughly 7K Search Arena battles provided by LMArena in a fully automatic manner. Our results show a significant correlation between nugget scores and human preferences, showcasing promise in our approach to explainable and diagnostic system evaluations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20000v1">Knowledge Distillation of Domain-adapted LLMs for Question-Answering in Telecom</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 10 pages, 4 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Knowledge Distillation (KD) is one of the approaches to reduce the size of Large Language Models (LLMs). A LLM with smaller number of model parameters (student) is trained to mimic the performance of a LLM of a larger size (teacher model) on a specific task. For domain-specific tasks, it is not clear if teacher or student model, or both, must be considered for domain adaptation. In this work, we study this problem from perspective of telecom domain Question-Answering (QA) task. We systematically experiment with Supervised Fine-tuning (SFT) of teacher only, SFT of student only and SFT of both prior to KD. We design experiments to study the impact of vocabulary (same and different) and KD algorithms (vanilla KD and Dual Space KD, DSKD) on the distilled model. Multi-faceted evaluation of the distillation using 14 different metrics (N-gram, embedding and LLM-based metrics) is considered. Experimental results show that SFT of teacher improves performance of distilled model when both models have same vocabulary, irrespective of algorithm and metrics. Overall, SFT of both teacher and student results in better performance across all metrics, although the statistical significance of the same depends on the vocabulary of the teacher models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19981v1">Accurate and Diverse LLM Mathematical Reasoning via Automated PRM-Guided GFlowNets</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Achieving both accuracy and diverse reasoning remains challenging for Large Language Models (LLMs) in complex domains like mathematics. A key bottleneck is evaluating intermediate reasoning steps to guide generation without costly human annotations. To address this, we first introduce a novel Process Reward Model (PRM) trained automatically using Monte Carlo Tree Search coupled with a similarity-based data augmentation technique, effectively capturing step-level reasoning quality. Leveraging this PRM, we then adapt Generative Flow Networks (GFlowNets) to operate at the reasoning step level. Unlike traditional reinforcement learning focused on maximizing a single reward, GFlowNets naturally sample diverse, high-quality solutions proportional to their rewards, as measured by our PRM. Empirical evaluation shows strong improvements in both accuracy and solution diversity on challenging mathematical benchmarks (e.g., +2.59% absolute accuracy on MATH Level 5 for Llama3.2-3B), with effective generalization to unseen datasets (+9.4% absolute on SAT MATH). Our work demonstrates the potential of PRM-guided, step-level GFlowNets for developing more robust and versatile mathematical reasoning in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19959v1">From Concept to Practice: an Automated LLM-aided UVM Machine for RTL Verification</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Verification presents a major bottleneck in Integrated Circuit (IC) development, consuming nearly 70% of the total development effort. While the Universal Verification Methodology (UVM) is widely used in industry to improve verification efficiency through structured and reusable testbenches, constructing these testbenches and generating sufficient stimuli remain challenging. These challenges arise from the considerable manual coding effort required, repetitive manual execution of multiple EDA tools, and the need for in-depth domain expertise to navigate complex designs.Here, we present UVM^2, an automated verification framework that leverages Large Language Models (LLMs) to generate UVM testbenches and iteratively refine them using coverage feedback, significantly reducing manual effort while maintaining rigorous verification standards.To evaluate UVM^2, we introduce a benchmark suite comprising Register Transfer Level (RTL) designs of up to 1.6K lines of code.The results show that UVM^2 reduces testbench setup time by up to UVM^2 compared to experienced engineers, and achieve average code and function coverage of 87.44% and 89.58%, outperforming state-of-the-art solutions by 20.96% and 23.51%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08775v3">What Should We Engineer in Prompts? Training Humans in Requirement-Driven LLM Use</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 15 pages; TOCHI 2025
    </div>
    <details class="paper-abstract">
      Prompting LLMs for complex tasks (e.g., building a trip advisor chatbot) needs humans to clearly articulate customized requirements (e.g., "start the response with a tl;dr"). However, existing prompt engineering instructions often lack focused training on requirement articulation and instead tend to emphasize increasingly automatable strategies (e.g., tricks like adding role-plays and "think step-by-step"). To address the gap, we introduce Requirement-Oriented Prompt Engineering (ROPE), a paradigm that focuses human attention on generating clear, complete requirements during prompting. We implement ROPE through an assessment and training suite that provides deliberate practice with LLM-generated feedback. In a randomized controlled experiment with 30 novices, ROPE significantly outperforms conventional prompt engineering training (20% vs. 1% gains), a gap that automatic prompt optimization cannot close. Furthermore, we demonstrate a direct correlation between the quality of input requirements and LLM outputs. Our work paves the way to empower more end-users to build complex LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11690v3">ID-Free Not Risk-Free: LLM-Powered Agents Unveil Risks in ID-Free Recommender Systems</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Recent advances in ID-free recommender systems have attracted significant attention for effectively addressing the cold start problem. However, their vulnerability to malicious attacks remains largely unexplored. In this paper, we unveil a critical yet overlooked risk: LLM-powered agents can be strategically deployed to attack ID-free recommenders, stealthily promoting low-quality items in black-box settings. This attack exploits a novel rewriting-based deception strategy, where malicious agents synthesize deceptive textual descriptions by simulating the characteristics of popular items. To achieve this, the attack mechanism integrates two primary components: (1) a popularity extraction component that captures essential characteristics of popular items and (2) a multi-agent collaboration mechanism that enables iterative refinement of promotional textual descriptions through independent thinking and team discussion. To counter this risk, we further introduce a detection method to identify suspicious text generated by our discovered attack. By unveiling this risk, our work aims to underscore the urgent need to enhance the security of ID-free recommender systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19898v1">GenCLS++: Pushing the Boundaries of Generative Classification in LLMs Through Comprehensive SFT and RL Studies Across Diverse Datasets</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      As a fundamental task in machine learning, text classification plays a crucial role in many areas. With the rapid scaling of Large Language Models (LLMs), particularly through reinforcement learning (RL), there is a growing need for more capable discriminators. Consequently, advances in classification are becoming increasingly vital for enhancing the overall capabilities of LLMs. Traditional discriminative methods map text to labels but overlook LLMs' intrinsic generative strengths. Generative classification addresses this by prompting the model to directly output labels. However, existing studies still rely on simple SFT alone, seldom probing the interplay between training and inference prompts, and no work has systematically leveraged RL for generative text classifiers and unified SFT, RL, and inference-time prompting in one framework. We bridge this gap with GenCLS++, a framework that jointly optimizes SFT and RL while systematically exploring five high-level strategy dimensions-in-context learning variants, category definitions, explicit uncertainty labels, semantically irrelevant numeric labels, and perplexity-based decoding-during both training and inference. After an SFT "policy warm-up," we apply RL with a simple rule-based reward, yielding sizable extra gains. Across seven datasets, GenCLS++ achieves an average accuracy improvement of 3.46% relative to the naive SFT baseline; on public datasets, this improvement rises to 4.00%. Notably, unlike reasoning-intensive tasks that benefit from explicit thinking processes, we find that classification tasks perform better without such reasoning steps. These insights into the role of explicit reasoning provide valuable guidance for future LLM applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19867v1">semi-PD: Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified Storage</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 18 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Existing large language model (LLM) serving systems fall into two categories: 1) a unified system where prefill phase and decode phase are co-located on the same GPU, sharing the unified computational resource and storage, and 2) a disaggregated system where the two phases are disaggregated to different GPUs. The design of the disaggregated system addresses the latency interference and sophisticated scheduling issues in the unified system but leads to storage challenges including 1) replicated weights for both phases that prevent flexible deployment, 2) KV cache transfer overhead between the two phases, 3) storage imbalance that causes substantial wasted space of the GPU capacity, and 4) suboptimal resource adjustment arising from the difficulties in migrating KV cache. Such storage inefficiency delivers poor serving performance under high request rates. In this paper, we identify that the advantage of the disaggregated system lies in the disaggregated computation, i.e., partitioning the computational resource to enable the asynchronous computation of two phases. Thus, we propose a novel LLM serving system, semi-PD, characterized by disaggregated computation and unified storage. In semi-PD, we introduce a computation resource controller to achieve disaggregated computation at the streaming multi-processor (SM) level, and a unified memory manager to manage the asynchronous memory access from both phases. semi-PD has a low-overhead resource adjustment mechanism between the two phases, and a service-level objective (SLO) aware dynamic partitioning algorithm to optimize the SLO attainment. Compared to state-of-the-art systems, semi-PD maintains lower latency at higher request rates, reducing the average end-to-end latency per request by 1.27-2.58x on DeepSeek series models, and serves 1.55-1.72x more requests adhering to latency constraints on Llama series models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19838v1">LLM-Powered GUI Agents in Phone Automation: Surveying Progress and Prospects</a></div>
    <div class="paper-meta">
      📅 2025-04-28
      | 💬 37 pages, 10 figures, 7 tables, Project Homepage: https://github.com/PhoneLLM/Awesome-LLM-Powered-Phone-GUI-Agents
    </div>
    <details class="paper-abstract">
      With the rapid rise of large language models (LLMs), phone automation has undergone transformative changes. This paper systematically reviews LLM-driven phone GUI agents, highlighting their evolution from script-based automation to intelligent, adaptive systems. We first contextualize key challenges, (i) limited generality, (ii) high maintenance overhead, and (iii) weak intent comprehension, and show how LLMs address these issues through advanced language understanding, multimodal perception, and robust decision-making. We then propose a taxonomy covering fundamental agent frameworks (single-agent, multi-agent, plan-then-act), modeling approaches (prompt engineering, training-based), and essential datasets and benchmarks. Furthermore, we detail task-specific architectures, supervised fine-tuning, and reinforcement learning strategies that bridge user intent and GUI operations. Finally, we discuss open challenges such as dataset diversity, on-device deployment efficiency, user-centric adaptation, and security concerns, offering forward-looking insights into this rapidly evolving field. By providing a structured overview and identifying pressing research gaps, this paper serves as a definitive reference for researchers and practitioners seeking to harness LLMs in designing scalable, user-friendly phone GUI agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.19793v1">Prompt Injection Attack to Tool Selection in LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Tool selection is a key component of LLM agents. The process operates through a two-step mechanism - \emph{retrieval} and \emph{selection} - to pick the most appropriate tool from a tool library for a given task. In this work, we introduce \textit{ToolHijacker}, a novel prompt injection attack targeting tool selection in no-box scenarios. ToolHijacker injects a malicious tool document into the tool library to manipulate the LLM agent's tool selection process, compelling it to consistently choose the attacker's malicious tool for an attacker-chosen target task. Specifically, we formulate the crafting of such tool documents as an optimization problem and propose a two-phase optimization strategy to solve it. Our extensive experimental evaluation shows that ToolHijacker is highly effective, significantly outperforming existing manual-based and automated prompt injection attacks when applied to tool selection. Moreover, we explore various defenses, including prevention-based defenses (StruQ and SecAlign) and detection-based defenses (known-answer detection, perplexity detection, and perplexity windowed detection). Our experimental results indicate that these defenses are insufficient, highlighting the urgent need for developing new defense strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.03111v2">Les Dissonances: Cross-Tool Harvesting and Polluting in Multi-Tool Empowered LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-04-28
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) agents are autonomous systems powered by LLMs, capable of reasoning and planning to solve problems by leveraging a set of tools. However, the integration of multi-tool capabilities in LLM agents introduces challenges in securely managing tools, ensuring their compatibility, handling dependency relationships, and protecting control flows within LLM agent workflows. In this paper, we present the first systematic security analysis of task control flows in multi-tool-enabled LLM agents. We identify a novel threat, Cross-Tool Harvesting and Polluting (XTHP), which includes multiple attack vectors to first hijack the normal control flows of agent tasks, and then collect and pollute confidential or private information within LLM agent systems. To understand the impact of this threat, we developed Chord, a dynamic scanning tool designed to automatically detect real-world agent tools susceptible to XTHP attacks. Our evaluation of 66 real-world tools from the repositories of two major LLM agent development frameworks, LangChain and LlamaIndex, revealed a significant security concern: 75\% are vulnerable to XTHP attacks, highlighting the prevalence of this threat.
    </details>
</div>
