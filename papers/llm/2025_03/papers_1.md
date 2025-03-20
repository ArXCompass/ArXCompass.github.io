# llm - 2025_03

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

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15478v1">SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 29 pages, 16 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents need to perform multi-turn interactions in real-world tasks. However, existing multi-turn RL algorithms for optimizing LLM agents fail to perform effective credit assignment over multiple turns while leveraging the generalization capabilities of LLMs and it remains unclear how to develop such algorithms. To study this, we first introduce a new benchmark, ColBench, where an LLM agent interacts with a human collaborator over multiple turns to solve realistic tasks in backend programming and frontend design. Building on this benchmark, we propose a novel RL algorithm, SWEET-RL (RL with Step-WisE Evaluation from Training-time information), that uses a carefully designed optimization objective to train a critic model with access to additional training-time information. The critic provides step-level rewards for improving the policy model. Our experiments demonstrate that SWEET-RL achieves a 6% absolute improvement in success and win rates on ColBench compared to other state-of-the-art multi-turn RL algorithms, enabling Llama-3.1-8B to match or exceed the performance of GPT4-o in realistic collaborative content creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.13213v3">Probabilities of Chat LLMs Are Miscalibrated but Still Predict Correctness on Multiple-Choice Q&A</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      We study 15 large language models (LLMs) fine-tuned for chat and find that their maximum softmax probabilities (MSPs) are consistently miscalibrated on multiple-choice Q&A. However, those MSPs might still encode useful uncertainty information. Specifically, we hypothesized that wrong answers would be associated with smaller MSPs compared to correct answers. Via rigorous statistical testing, we show that this hypothesis holds for models which perform well on the underlying Q&A task. We also find a strong direction correlation between Q&A accuracy and MSP correctness prediction, while finding no correlation between Q&A accuracy and calibration error. This suggests that within the current fine-tuning paradigm, we can expect correctness prediction but not calibration to improve as LLM capabilities progress. To demonstrate the utility of correctness prediction, we show that when models have the option to abstain, performance can be improved by selectively abstaining based on the MSP of the initial model response, using only a small amount of labeled data to choose the MSP threshold.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15374v1">Real-world validation of a multimodal LLM-powered pipeline for High-Accuracy Clinical Trial Patient Matching leveraging EHR data</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Background: Patient recruitment in clinical trials is hindered by complex eligibility criteria and labor-intensive chart reviews. Prior research using text-only models have struggled to address this problem in a reliable and scalable way due to (1) limited reasoning capabilities, (2) information loss from converting visual records to text, and (3) lack of a generic EHR integration to extract patient data. Methods: We introduce a broadly applicable, integration-free, LLM-powered pipeline that automates patient-trial matching using unprocessed documents extracted from EHRs. Our approach leverages (1) the new reasoning-LLM paradigm, enabling the assessment of even the most complex criteria, (2) visual capabilities of latest LLMs to interpret medical records without lossy image-to-text conversions, and (3) multimodal embeddings for efficient medical record search. The pipeline was validated on the n2c2 2018 cohort selection dataset (288 diabetic patients) and a real-world dataset composed of 485 patients from 30 different sites matched against 36 diverse trials. Results: On the n2c2 dataset, our method achieved a new state-of-the-art criterion-level accuracy of 93\%. In real-world trials, the pipeline yielded an accuracy of 87\%, undermined by the difficulty to replicate human decision-making when medical records lack sufficient information. Nevertheless, users were able to review overall eligibility in under 9 minutes per patient on average, representing an 80\% improvement over traditional manual chart reviews. Conclusion: This pipeline demonstrates robust performance in clinical trial patient matching without requiring custom integration with site systems or trial-specific tailoring, thereby enabling scalable deployment across sites seeking to leverage AI for patient matching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.20227v2">LLM Reasoning Engine: Specialized Training for Enhanced Mathematical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 Accepted to NAACL 2025 KnowledgeNLP
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable performance in various natural language processing tasks but face challenges in mathematical reasoning, where complex problem-solving requires both linguistic understanding and mathematical reasoning skills. Existing approaches to address this challenge often rely on ensemble methods and suffer from the problem of data scarcity in target domains. In this work, we present a novel method to enhance LLMs' capabilities in mathematical reasoning tasks. Motivated by the need to bridge this gap, our approach incorporates a question paraphrase strategy, which aims at diversifying the linguistic forms of mathematical questions to improve generalization. Additionally, specialized training objectives are employed to guide the model's learning process, focusing on enhancing its understanding of mathematical concepts and reasoning processes. We conduct experiments on four datasets using different LLMs, and demonstrate the effectiveness of our approach in improving LLMs' performance on mathematical reasoning tasks. Our findings underscore the significance of our methodology in the advancement of large language models and its potential implications for real-world applications that require mathematical reasoning abilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15341v1">Uncertainty-Guided Chain-of-Thought for Code Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Chain-of-Thought (CoT) reasoning has been demonstrated as an effective technique for improving the problem-solving capabilities of large language models (LLMs) in the context of code generation. However, existing CoT methods often exhibit a tendency toward "overthinking", where the LLM consistently applies reasoning strategies without adequately considering the task's underlying complexity. This results in the LLMs allocating excessive computational resources, in terms of tokens, to relatively simple tasks or problems where the correct answer is already evident. Additionally, this overthinking may lead LLMs down incorrect reasoning paths, resulting in incorrect code generation. In this paper, we introduce UnCertainty-Aware Chain-of-Thought (UnCert-CoT), an LLM-based approach designed to enhance code generation by incorporating an uncertainty-aware CoT reasoning mechanism, which focuses computational resources on targeting points where LLMs are more prone to error. We propose two confidence-based uncertainty measures: Entropy-based and Probability Differential-based methods. When uncertainty is high, UnCert-CoT activates CoT-decoding to generate multiple reasoning paths and selects the final code that exhibits the highest likelihood of correctness. In contrast, LLM directly generates the code when uncertainty is low. This uncertainty judgment mechanism allows LLMs to prioritize complex tasks and avoid unnecessary steps in simpler cases, thereby improving overall efficiency and accuracy in code generation. Our experimental results demonstrate that UnCert-CoT significantly enhances code generation accuracy on challenging benchmark MHPP(Mostly Hard Python Problems), it achieves improvements up to 6.1% on PassRate accuracy, particularly in situations where traditional LLMs are prone to errors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.14703v3">Do LLMs Have Distinct and Consistent Personality? TRAIT: Personality Testset designed for LLMs with Psychometrics</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 Accepted to NAACL2025 Findings
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have led to their adaptation in various domains as conversational agents. We wonder: can personality tests be applied to these agents to analyze their behavior, similar to humans? We introduce TRAIT, a new benchmark consisting of 8K multi-choice questions designed to assess the personality of LLMs. TRAIT is built on two psychometrically validated small human questionnaires, Big Five Inventory (BFI) and Short Dark Triad (SD-3), enhanced with the ATOMIC-10X knowledge graph to a variety of real-world scenarios. TRAIT also outperforms existing personality tests for LLMs in terms of reliability and validity, achieving the highest scores across four key metrics: Content Validity, Internal Validity, Refusal Rate, and Reliability. Using TRAIT, we reveal two notable insights into personalities of LLMs: 1) LLMs exhibit distinct and consistent personality, which is highly influenced by their training data (e.g., data used for alignment tuning), and 2) current prompting techniques have limited effectiveness in eliciting certain traits, such as high psychopathy or low conscientiousness, suggesting the need for further research in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15338v1">Solla: Towards a Speech-Oriented LLM That Hears Acoustic Context</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have recently shown remarkable ability to process not only text but also multimodal inputs such as speech and audio. However, most existing models primarily focus on analyzing input signals using text instructions, overlooking scenarios in which speech instructions and audio are mixed and serve as inputs to the model. To address these challenges, we introduce Solla, a novel framework designed to understand speech-based questions and hear the acoustic context concurrently. Solla incorporates an audio tagging module to effectively identify and represent audio events, as well as an ASR-assisted prediction method to improve comprehension of spoken content. To rigorously evaluate Solla and other publicly available models, we propose a new benchmark dataset called SA-Eval, which includes three tasks: audio event classification, audio captioning, and audio question answering. SA-Eval has diverse speech instruction with various speaking styles, encompassing two difficulty levels, easy and hard, to capture the range of real-world acoustic conditions. Experimental results show that Solla performs on par with or outperforms baseline models on both the easy and hard test sets, underscoring its effectiveness in jointly understanding speech and audio.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09454v2">Explicit Learning and the LLM in Machine Translation</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      This study explores the capacity of large language models (LLMs) for explicit learning, a process involving the assimilation of metalinguistic explanations to carry out language tasks. Using constructed languages generated by cryptographic means as controlled test environments, we designed experiments to assess an LLM's ability to explicitly learn and apply grammar rules. Our results demonstrate that while LLMs possess a measurable capacity for explicit learning, this ability diminishes as the complexity of the linguistic phenomena at hand increases. Supervised fine-tuning on chains of thought significantly enhances LLM performance but struggles to generalize to typologically novel or more complex linguistic features. These findings point to the need for more diverse training sets and alternative fine-tuning strategies to further improve explicit learning by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15301v1">aiXcoder-7B-v2: Training LLMs to Fully Utilize the Long Context in Repository-level Code Completion</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Repository-level code completion aims to complete code based on the long contexts of the repository. Existing studies extract long contexts from the repository as inputs and leverage Large Language Models (LLMs) to generate code. However, we reveal a severe limitation of LLMs, i.e., LLMs may ignore the information within long contexts in code completion. In other words, even the contexts contain useful information (e.g., relevant APIs or similar code), LLMs may fail to utilize this information. We think this limitation is caused by an inherent bias in LLMs, i.e., relying on nearby contexts and ignoring long-range contexts. To address this, we propose a novel fine-tuning approach named CoLT. The core idea of CoLT is to provide explicit supervision signals, which emphasize that long-range contexts may hold relevant information. Specifically, CoLT proposes a reinforcement learning-based training, which explicitly encourages models to utilize the information within long contexts and punishes models for ignoring long contexts. To support CoLT, we release CoLT-132K, a large-scale dataset with 132k samples across four languages, each containing long-context inputs. We apply CoLT to a popular LLM - aiXcoder-7B and release aiXcoder-7B-v2. We conduct extensive experiments on CoLT-132K and a public benchmark - CrossCodeEval. Our experiments yield the results: 1. Effectiveness. CoLT substantially improves aiXcoder-7B. aiXcoder-7B-v2 outperforms aiXcoder-7B by up to 44% in exact match. aiXcoder-7B-v2 becomes the state-of-the-art 7B model in code completion and even surpasses larger models. 2. Generalizability. The capability learned by CoLT can generalize to new languages. Besides, CoLT is model-agnostic and effectively improves multiple LLMs. 3. Enhanced Context Utilization Capability. CoLT significantly improves the capability of LLMs in utilizing the relevant information within long contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15299v1">Inside-Out: Hidden Factual Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      This work presents a framework for assessing whether large language models (LLMs) encode more factual knowledge in their parameters than what they express in their outputs. While a few studies hint at this possibility, none has clearly defined or demonstrated this phenomenon. We first propose a formal definition of knowledge, quantifying it for a given question as the fraction of correct-incorrect answer pairs where the correct one is ranked higher. This gives rise to external and internal knowledge, depending on the information used to score individual answer candidates: either the model's observable token-level probabilities or its intermediate computations. Hidden knowledge arises when internal knowledge exceeds external knowledge. We then present a case study, applying this framework to three popular open-weights LLMs in a closed-book QA setup. Our results indicate that: (1) LLMs consistently encode more factual knowledge internally than what they express externally, with an average gap of 40%. (2) Surprisingly, some knowledge is so deeply hidden that a model can internally know an answer perfectly, yet fail to generate it even once, despite large-scale repeated sampling of 1,000 answers. This reveals fundamental limitations in the generation capabilities of LLMs, which (3) puts a practical constraint on scaling test-time compute via repeated answer sampling in closed-book QA: significant performance improvements remain inaccessible because some answers are practically never sampled, yet if they were, we would be guaranteed to rank them first.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15242v1">BigO(Bench) -- Can LLMs Generate Code with Controlled Time and Space Complexity?</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      We introduce BigO(Bench), a novel coding benchmark designed to evaluate the capabilities of generative language models in understanding and generating code with specified time and space complexities. This benchmark addresses the gap in current evaluations that often overlook the ability of models to comprehend and produce code constrained by computational complexity. BigO(Bench) includes tooling to infer the algorithmic complexity of any Python function from profiling measurements, including human- or LLM-generated solutions. BigO(Bench) also includes of set of 3,105 coding problems and 1,190,250 solutions from Code Contests annotated with inferred (synthetic) time and space complexity labels from the complexity framework, as well as corresponding runtime and memory footprint values for a large set of input sizes. We present results from evaluating multiple state-of-the-art language models on this benchmark, highlighting their strengths and weaknesses in handling complexity requirements. In particular, token-space reasoning models are unrivaled in code generation but not in complexity understanding, hinting that they may not generalize well to tasks for which no reward was given at training time.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12896v2">None of the Others: a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      In LLM evaluations, reasoning is often distinguished from recall/memorization by performing numerical variations to math-oriented questions. Here we introduce a general variation method for multiple-choice questions that completely dissociates the correct answer from previously seen tokens or concepts, requiring LLMs to understand and reason (rather than memorizing) in order to answer correctly. Using this method, we evaluate state-of-the-art proprietary and open-source LLMs on two datasets available in English and Spanish: the public MMLU benchmark and the private UNED-Access 2024 dataset. Results show that all models experience remarkable accuracy drops under our proposed variation, with an average loss of 57% on MMLU and 50% on UNED-Access 2024, ranging from 10% to 93% across models. Notably, the most accurate model in our experimentation (OpenAI-o3-mini) is not the most robust (DeepSeek-R1-70B), suggesting that the best models in standard evaluations may not be the ones with better reasoning capabilities. Also, we see larger accuracy drops in public (vs private) datasets and questions posed in their original language (vs a manual translation), which are signs of contamination and also point to a relevant role of recall/memorization in current LLMs' answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15231v1">When LLMs Meet API Documentation: Can Retrieval Augmentation Aid Code Generation Just as It Helps Developers?</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Retrieval-augmented generation (RAG) has increasingly shown its power in extending large language models' (LLMs') capability beyond their pre-trained knowledge. Existing works have shown that RAG can help with software development tasks such as code generation, code update, and test generation. Yet, the effectiveness of adapting LLMs to fast-evolving or less common API libraries using RAG remains unknown. To bridge this gap, we take an initial step to study this unexplored yet practical setting - when developers code with a less common library, they often refer to its API documentation; likewise, when LLMs are allowed to look up API documentation via RAG, to what extent can LLMs be advanced? To mimic such a setting, we select four less common open-source Python libraries with a total of 1017 eligible APIs. We study the factors that affect the effectiveness of using the documentation of less common API libraries as additional knowledge for retrieval and generation. Our intensive study yields interesting findings: (1) RAG helps improve LLMs' performance by 83%-220%. (2) Example code contributes the most to advance LLMs, instead of the descriptive texts and parameter lists in the API documentation. (3) LLMs could sometimes tolerate mild noises (typos in description or incorrect parameters) by referencing their pre-trained knowledge or document context. Finally, we suggest that developers pay more attention to the quality and diversity of the code examples in the API documentation. The study sheds light on future low-code software development workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.16457v4">Towards Fully-Automated Materials Discovery via Large-Scale Synthesis Dataset and Expert-Level LLM-as-a-Judge</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 under review
    </div>
    <details class="paper-abstract">
      Materials synthesis is vital for innovations such as energy storage, catalysis, electronics, and biomedical devices. Yet, the process relies heavily on empirical, trial-and-error methods guided by expert intuition. Our work aims to support the materials science community by providing a practical, data-driven resource. We have curated a comprehensive dataset of 17K expert-verified synthesis recipes from open-access literature, which forms the basis of our newly developed benchmark, AlchemyBench. AlchemyBench offers an end-to-end framework that supports research in large language models applied to synthesis prediction. It encompasses key tasks, including raw materials and equipment prediction, synthesis procedure generation, and characterization outcome forecasting. We propose an LLM-as-a-Judge framework that leverages large language models for automated evaluation, demonstrating strong statistical agreement with expert assessments. Overall, our contributions offer a supportive foundation for exploring the capabilities of LLMs in predicting and guiding materials synthesis, ultimately paving the way for more efficient experimental design and accelerated innovation in materials science.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15117v1">Exploring Model Editing for LLM-based Aspect-Based Sentiment Classification</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 AAAI2025
    </div>
    <details class="paper-abstract">
      Model editing aims at selectively updating a small subset of a neural model's parameters with an interpretable strategy to achieve desired modifications. It can significantly reduce computational costs to adapt to large language models (LLMs). Given its ability to precisely target critical components within LLMs, model editing shows great potential for efficient fine-tuning applications. In this work, we investigate model editing to serve an efficient method for adapting LLMs to solve aspect-based sentiment classification. Through causal interventions, we trace and determine which neuron hidden states are essential for the prediction of the model. By performing interventions and restorations on each component of an LLM, we identify the importance of these components for aspect-based sentiment classification. Our findings reveal that a distinct set of mid-layer representations is essential for detecting the sentiment polarity of given aspect words. Leveraging these insights, we develop a model editing approach that focuses exclusively on these critical parts of the LLM, leading to a more efficient method for adapting LLMs. Our in-domain and out-of-domain experiments demonstrate that this approach achieves competitive results compared to the currently strongest methods with significantly fewer trainable parameters, highlighting a more efficient and interpretable fine-tuning strategy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15113v1">Reasoning Effort and Problem Complexity: A Scaling Analysis in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 Published at ICLR 2025 Workshop on Reasoning and Planning for LLMs
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable text generation capabilities, and recent advances in training paradigms have led to breakthroughs in their reasoning performance. In this work, we investigate how the reasoning effort of such models scales with problem complexity. We use the infinitely scalable Tents puzzle, which has a known linear-time solution, to analyze this scaling behavior. Our results show that reasoning effort scales with problem size, but only up to a critical problem complexity. Beyond this threshold, the reasoning effort does not continue to increase, and may even decrease. This observation highlights a critical limitation in the logical coherence of current LLMs as problem complexity increases, and underscores the need for strategies to improve reasoning scalability. Furthermore, our results reveal significant performance differences between current state-of-the-art reasoning models when faced with increasingly complex logical puzzles.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15112v1">OpenLLM-RTL: Open Dataset and Benchmark for LLM-Aided Design RTL Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 ICCAD'24
    </div>
    <details class="paper-abstract">
      The automated generation of design RTL based on large language model (LLM) and natural language instructions has demonstrated great potential in agile circuit design. However, the lack of datasets and benchmarks in the public domain prevents the development and fair evaluation of LLM solutions. This paper highlights our latest advances in open datasets and benchmarks from three perspectives: (1) RTLLM 2.0, an updated benchmark assessing LLM's capability in design RTL generation. The benchmark is augmented to 50 hand-crafted designs. Each design provides the design description, test cases, and a correct RTL code. (2) AssertEval, an open-source benchmark assessing the LLM's assertion generation capabilities for RTL verification. The benchmark includes 18 designs, each providing specification, signal definition, and correct RTL code. (3) RTLCoder-Data, an extended open-source dataset with 80K instruction-code data samples. Moreover, we propose a new verification-based method to verify the functionality correctness of training data samples. Based on this technique, we further release a dataset with 7K verified high-quality samples. These three studies are integrated into one framework, providing off-the-shelf support for the development and evaluation of LLMs for RTL code generation and verification. Finally, extensive experiments indicate that LLM performance can be boosted by enlarging the training dataset, improving data quality, and improving the training scheme.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15091v1">Intelligent Spatial Perception by Building Hierarchical 3D Scene Graphs for Indoor Scenarios with the Help of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 accepted by WRC SARA 2024
    </div>
    <details class="paper-abstract">
      This paper addresses the high demand in advanced intelligent robot navigation for a more holistic understanding of spatial environments, by introducing a novel system that harnesses the capabilities of Large Language Models (LLMs) to construct hierarchical 3D Scene Graphs (3DSGs) for indoor scenarios. The proposed framework constructs 3DSGs consisting of a fundamental layer with rich metric-semantic information, an object layer featuring precise point-cloud representation of object nodes as well as visual descriptors, and higher layers of room, floor, and building nodes. Thanks to the innovative application of LLMs, not only object nodes but also nodes of higher layers, e.g., room nodes, are annotated in an intelligent and accurate manner. A polling mechanism for room classification using LLMs is proposed to enhance the accuracy and reliability of the room node annotation. Thorough numerical experiments demonstrate the system's ability to integrate semantic descriptions with geometric data, creating an accurate and comprehensive representation of the environment instrumental for context-aware navigation and task planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15079v1">LogiAgent: Automated Logical Testing for REST Systems with LLM-Based Multi-Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Automated testing for REST APIs has become essential for ensuring the correctness and reliability of modern web services. While existing approaches primarily focus on detecting server crashes and error codes, they often overlook logical issues that arise due to evolving business logic and domain-specific requirements. To address this limitation, we propose LogiAgent, a novel approach for logical testing of REST systems. Built upon a large language model (LLM)-driven multi-agent framework, LogiAgent integrates a Test Scenario Generator, API Request Executor, and API Response Validator to collaboratively generate, execute, and validate API test scenarios. Unlike traditional testing methods that focus on status codes like 5xx, LogiAgent incorporates logical oracles that assess responses based on business logic, ensuring more comprehensive testing. The system is further enhanced by an Execution Memory component that stores historical API execution data for contextual consistency. We conduct extensive experiments across 12 real-world REST systems, demonstrating that LogiAgent effectively identifies 234 logical issues with an accuracy of 66.19%. Additionally, it basically excels in detecting server crashes and achieves superior test coverage compared to four state-of-the-art REST API testing tools. An ablation study confirms the significant contribution of LogiAgent's memory components to improving test coverage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15050v1">Studying and Understanding the Effectiveness and Failures of Conversational LLM-Based Repair</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Automated program repair (APR) is designed to automate the process of bug-fixing. In recent years, thanks to the rapid development of large language models (LLMs), automated repair has achieved remarkable progress. Advanced APR techniques powered by conversational LLMs, most notably ChatGPT, have exhibited impressive repair abilities and gained increasing popularity due to the capabilities of the underlying LLMs in providing repair feedback and performing iterative patch improvement. Despite the superiority, conversational APR techniques still fail to repair a large number of bugs. For example, a state-of-the-art conversational technique ChatRepair does not correctly repair over half of the single-function bugs in the Defects4J dataset. To understand the effectiveness and failures of conversational LLM-based repair and provide possible directions for improvement, we studied the exemplary ChatRepair with a focus on comparing the effectiveness of its cloze-style and full function repair strategies, assessing its key iterative component for patch improvement, and analyzing the repair failures. Our study has led to a series of findings, which we believe provide key implications for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15003v1">LLM Alignment for the Arabs: A Homogenous Culture or Diverse Ones?</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 Accepted to the C3NLP workshop (Co-located with NAACL 2025)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have the potential of being useful tools that can automate tasks and assist humans. However, these models are more fluent in English and more aligned with Western cultures, norms, and values. Arabic-specific LLMs are being developed to better capture the nuances of the Arabic language, as well as the views of the Arabs. Yet, Arabs are sometimes assumed to share the same culture. In this position paper, I discuss the limitations of this assumption and provide preliminary thoughts for how to build systems that can better represent the cultural diversity within the Arab world. The invalidity of the cultural homogeneity assumption might seem obvious, yet, it is widely adopted in developing multilingual and Arabic-specific LLMs. I hope that this paper will encourage the NLP community to be considerate of the cultural diversity within various communities speaking the same language.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14996v1">Right Answer, Wrong Score: Uncovering the Inconsistencies of LLM Evaluation in Multiple-Choice Question Answering</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 17 pages (9 main), 11 figures, 21 tables
    </div>
    <details class="paper-abstract">
      One of the most widely used tasks to evaluate Large Language Models (LLMs) is Multiple-Choice Question Answering (MCQA). While open-ended question answering tasks are more challenging to evaluate, MCQA tasks are, in principle, easier to assess, as the model's answer is thought to be simple to extract and is directly compared to a set of predefined choices. However, recent studies have started to question the reliability of MCQA evaluation, showing that multiple factors can significantly impact the reported performance of LLMs, especially when the model generates free-form text before selecting one of the answer choices. In this work, we shed light on the inconsistencies of MCQA evaluation strategies, which can lead to inaccurate and misleading model comparisons. We systematically analyze whether existing answer extraction methods are aligned with human judgment, and how they are influenced by answer constraints in the prompt across different domains. Our experiments demonstrate that traditional evaluation strategies often underestimate LLM capabilities, while LLM-based answer extractors are prone to systematic errors. Moreover, we reveal a fundamental trade-off between including format constraints in the prompt to simplify answer extraction and allowing models to generate free-form text to improve reasoning. Our findings call for standardized evaluation methodologies and highlight the need for more reliable and consistent MCQA evaluation practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14936v1">Enhancing Code LLM Training with Programmer Attention</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Human attention provides valuable yet underexploited signals for code LLM training, offering a perspective beyond purely machine-driven attention. Despite the complexity and cost of collecting eye-tracking data, there has also been limited progress in systematically using these signals for code LLM training. To address both issues, we propose a cohesive pipeline spanning augmentation and reward-based fine-tuning. Specifically, we introduce (1) an eye-tracking path augmentation method to expand programmer attention datasets, (2) a pattern abstraction step that refines raw fixations into learnable attention motifs, and (3) a reward-guided strategy for integrating these insights directly into a CodeT5 supervised fine-tuning process. Our experiments yield +7.16 in CodeBLEU on the CodeXGlue benchmark for code summarization, underscoring how uniting human and machine attention can boost code intelligence. We hope this work encourages broader exploration of human-centric methods in next-generation AI4SE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14932v1">Prada: Black-Box LLM Adaptation with Private Data on Resource-Constrained Devices</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      In recent years, Large Language Models (LLMs) have demonstrated remarkable abilities in various natural language processing tasks. However, adapting these models to specialized domains using private datasets stored on resource-constrained edge devices, such as smartphones and personal computers, remains challenging due to significant privacy concerns and limited computational resources. Existing model adaptation methods either compromise data privacy by requiring data transmission or jeopardize model privacy by exposing proprietary LLM parameters. To address these challenges, we propose Prada, a novel privacy-preserving and efficient black-box LLM adaptation system using private on-device datasets. Prada employs a lightweight proxy model fine-tuned with Low-Rank Adaptation (LoRA) locally on user devices. During inference, Prada leverages the logits offset, i.e., difference in outputs between the base and adapted proxy models, to iteratively refine outputs from a remote black-box LLM. This offset-based adaptation approach preserves both data privacy and model privacy, as there is no need to share sensitive data or proprietary model parameters. Furthermore, we incorporate speculative decoding to further speed up the inference process of Prada, making the system practically deployable on bandwidth-constrained edge devices, enabling a more practical deployment of Prada. Extensive experiments on various downstream tasks demonstrate that Prada achieves performance comparable to centralized fine-tuning methods while significantly reducing computational overhead by up to 60% and communication costs by up to 80%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.09049v2">Dial-In LLM: Human-Aligned LLM-in-the-loop Intent Clustering for Customer Service Dialogues</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Discovering customer intentions in dialogue conversations is crucial for automated service agents. Yet, existing intent clustering methods often fail to align with human perceptions due to the heavy reliance on embedding distance metrics and sentence embeddings. To address these limitations, we propose integrating the semantic understanding capabilities of LLMs into an $\textbf{LLM-in-the-loop (LLM-ITL)}$ intent clustering framework. Specifically, this paper (1) investigates the effectiveness of fine-tuned LLMs in semantic coherence evaluation and intent cluster naming, achieving over 95% accuracy; (2) designs an LLM-ITL clustering algorithm that facilitates the iterative discovery of coherent intent clusters; and (3) proposes task-specific techniques tailored for customer service dialogue intent clustering. Since existing English benchmarks pose limited semantic diversity and intent labels, we introduced a comprehensive Chinese dialogue intent dataset, comprising over 100,000 real customer service calls and 1,507 human-annotated intent clusters. The proposed approaches significantly outperformed LLM-guided baselines, achieving notable improvements in clustering quality and a 12% boost in the downstream intent classification task. Combined with several best practices, our findings highlight the potential of LLM-in-the-loop techniques for scalable and human-aligned problem-solving. Sample code and datasets are available at: https://anonymous.4open.science/r/Dial-in-LLM-0410.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14924v1">UTFix: Change Aware Unit Test Repairing using LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 26 pages, International Conference on Object-oriented Programming, Systems, Languages, and Applications (OOPSLA) 2025
    </div>
    <details class="paper-abstract">
      Software updates, including bug repair and feature additions, are frequent in modern applications but they often leave test suites outdated, resulting in undetected bugs and increased chances of system failures. A recent study by Meta revealed that 14%-22% of software failures stem from outdated tests that fail to reflect changes in the codebase. This highlights the need to keep tests in sync with code changes to ensure software reliability. In this paper, we present UTFix, a novel approach for repairing unit tests when their corresponding focal methods undergo changes. UTFix addresses two critical issues: assertion failure and reduced code coverage caused by changes in the focal method. Our approach leverages language models to repair unit tests by providing contextual information such as static code slices, dynamic code slices, and failure messages. We evaluate UTFix on our generated synthetic benchmarks (Tool-Bench), and real-world benchmarks. Tool- Bench includes diverse changes from popular open-source Python GitHub projects, where UTFix successfully repaired 89.2% of assertion failures and achieved 100% code coverage for 96 tests out of 369 tests. On the real-world benchmarks, UTFix repairs 60% of assertion failures while achieving 100% code coverage for 19 out of 30 unit tests. To the best of our knowledge, this is the first comprehensive study focused on unit test in evolving Python projects. Our contributions include the development of UTFix, the creation of Tool-Bench and real-world benchmarks, and the demonstration of the effectiveness of LLM-based methods in addressing unit test failures due to software evolution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14887v1">Pseudo-Relevance Feedback Can Improve Zero-Shot LLM-Based Dense Retrieval</a></div>
    <div class="paper-meta">
      📅 2025-03-19
    </div>
    <details class="paper-abstract">
      Pseudo-relevance feedback (PRF) refines queries by leveraging initially retrieved documents to improve retrieval effectiveness. In this paper, we investigate how large language models (LLMs) can facilitate PRF for zero-shot LLM-based dense retrieval, extending the recently proposed PromptReps method. Specifically, our approach uses LLMs to extract salient passage features-such as keywords and summaries-from top-ranked documents, which are then integrated into PromptReps to produce enhanced query representations. Experiments on passage retrieval benchmarks demonstrate that incorporating PRF significantly boosts retrieval performance. Notably, smaller rankers with PRF can match the effectiveness of larger rankers without PRF, highlighting PRF's potential to improve LLM-driven search while maintaining an efficient balance between effectiveness and resource usage.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14882v1">Communication-Efficient Distributed On-Device LLM Inference Over Wireless Networks</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 arXiv admin note: text overlap with arXiv:2502.12559
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable success across various application domains, but their enormous sizes and computational demands pose significant challenges for deployment on resource-constrained edge devices. To address this issue, we propose a novel distributed on-device LLM inference framework that leverages tensor parallelism to partition the neural network tensors (e.g., weight matrices) of one LLM across multiple edge devices for collaborative inference. A key challenge in tensor parallelism is the frequent all-reduce operations for aggregating intermediate layer outputs across participating devices, which incurs significant communication overhead. To alleviate this bottleneck, we propose an over-the-air computation (AirComp) approach that harnesses the analog superposition property of wireless multiple-access channels to perform fast all-reduce steps. To utilize the heterogeneous computational capabilities of edge devices and mitigate communication distortions, we investigate a joint model assignment and transceiver optimization problem to minimize the average transmission error. The resulting mixed-timescale stochastic non-convex optimization problem is intractable, and we propose an efficient two-stage algorithm to solve it. Moreover, we prove that the proposed algorithm converges almost surely to a stationary point of the original problem. Comprehensive simulation results will show that the proposed framework outperforms existing benchmark schemes, achieving up to 5x inference speed acceleration and improving inference accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.11843v5">From Commands to Prompts: LLM-based Semantic File System for AIOS</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 Accepted by International Conference on Learning Representations 2025(ICLR2025)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant potential in the development of intelligent applications and systems such as LLM-based agents and agent operating systems (AIOS). However, when these applications and systems interact with the underlying file system, the file system still remains the traditional paradigm: reliant on manual navigation through precise commands. This paradigm poses a bottleneck to the usability of these systems as users are required to navigate complex folder hierarchies and remember cryptic file names. To address this limitation, we propose an LLM-based semantic file system ( LSFS ) for prompt-driven file management. Unlike conventional approaches, LSFS incorporates LLMs to enable users or agents to interact with files through natural language prompts, facilitating semantic file management. At the macro-level, we develop a comprehensive API set to achieve semantic file management functionalities, such as semantic file retrieval, file update monitoring and summarization, and semantic file rollback). At the micro-level, we store files by constructing semantic indexes for them, design and implement syscalls of different semantic operations (e.g., CRUD, group by, join) powered by vector database. Our experiments show that LSFS offers significant improvements over traditional file systems in terms of user convenience, the diversity of supported functions, and the accuracy and efficiency of file operations. Additionally, with the integration of LLM, our system enables more intelligent file management tasks, such as content summarization and version comparison, further enhancing its capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09956v2">DeepSeek-Inspired Exploration of RL-based LLMs and Synergy with Wireless Networks: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-03-19
      | 💬 25 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Reinforcement learning (RL)-based large language models (LLMs), such as ChatGPT, DeepSeek, and Grok-3, have gained significant attention for their exceptional capabilities in natural language processing and multimodal data understanding. Meanwhile, the rapid expansion of information services has driven the growing need for intelligence, efficient, and adaptable wireless networks. Wireless networks require the empowerment of RL-based LLMs while these models also benefit from wireless networks to broaden their application scenarios. Specifically, RL-based LLMs can enhance wireless communication systems through intelligent resource allocation, adaptive network optimization, and real-time decision-making. Conversely, wireless networks provide a vital infrastructure for the efficient training, deployment, and distributed inference of RL-based LLMs, especially in decentralized and edge computing environments. This mutual empowerment highlights the need for a deeper exploration of the interplay between these two domains. We first review recent advancements in wireless communications, highlighting the associated challenges and potential solutions. We then discuss the progress of RL-based LLMs, focusing on key technologies for LLM training, challenges, and potential solutions. Subsequently, we explore the mutual empowerment between these two fields, highlighting key motivations, open challenges, and potential solutions. Finally, we provide insights into future directions, applications, and their societal impact to further explore this intersection, paving the way for next-generation intelligent communication systems. Overall, this survey provides a comprehensive overview of the relationship between RL-based LLMs and wireless networks, offering a vision where these domains empower each other to drive innovations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14504v1">Aligning Multimodal LLM with Human Preference: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Alignment
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can handle a wide variety of general tasks with simple prompts, without the need for task-specific training. Multimodal Large Language Models (MLLMs), built upon LLMs, have demonstrated impressive potential in tackling complex tasks involving visual, auditory, and textual data. However, critical issues related to truthfulness, safety, o1-like reasoning, and alignment with human preference remain insufficiently addressed. This gap has spurred the emergence of various alignment algorithms, each targeting different application scenarios and optimization goals. Recent studies have shown that alignment algorithms are a powerful approach to resolving the aforementioned challenges. In this paper, we aim to provide a comprehensive and systematic review of alignment algorithms for MLLMs. Specifically, we explore four key aspects: (1) the application scenarios covered by alignment algorithms, including general image understanding, multi-image, video, and audio, and extended multimodal applications; (2) the core factors in constructing alignment datasets, including data sources, model responses, and preference annotations; (3) the benchmarks used to evaluate alignment algorithms; and (4) a discussion of potential future directions for the development of alignment algorithms. This work seeks to help researchers organize current advancements in the field and inspire better alignment methods. The project page of this paper is available at https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13803v2">LLMs as Models for Analogical Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 The title has been changed from Semantic Structure-Mapping in LLM and Human Analogical Reasoning to LLMs as Models for Analogical Reasoning to improve clarity and accuracy
    </div>
    <details class="paper-abstract">
      Analogical reasoning-the capacity to identify and map structural relationships between different domains-is fundamental to human cognition and learning. Recent studies have shown that large language models (LLMs) can sometimes match humans in analogical reasoning tasks, opening the possibility that analogical reasoning might emerge from domain general processes. However, it is still debated whether these emergent capacities are largely superficial and limited to simple relations seen during training or whether they rather encompass the flexible representational and mapping capabilities which are the focus of leading cognitive models of analogy. In this study, we introduce novel analogical reasoning tasks that require participants to map between semantically contentful words and sequences of letters and other abstract characters. This task necessitates the ability to flexibly re-represent rich semantic information-an ability which is known to be central to human analogy but which is thus far not well-captured by existing cognitive theories and models. We assess the performance of both human participants and LLMs on tasks focusing on reasoning from semantic structure and semantic content, introducing variations that test the robustness of their analogical inferences. Advanced LLMs match human performance across several conditions, though humans and LLMs respond differently to certain task variations and semantic distractors. Our results thus provide new evidence that LLMs might offer a how-possibly explanation of human analogical reasoning in contexts that are not yet well modeled by existing theories, but that even today's best models are unlikely to yield how-actually explanations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14476v1">DAPO: An Open-Source LLM Reinforcement Learning System at Scale</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 Project Page: https://dapo-sia.github.io/
    </div>
    <details class="paper-abstract">
      Inference scaling empowers LLMs with unprecedented reasoning ability, with reinforcement learning as the core technique to elicit complex reasoning. However, key technical details of state-of-the-art reasoning LLMs are concealed (such as in OpenAI o1 blog and DeepSeek R1 technical report), thus the community still struggles to reproduce their RL training results. We propose the $\textbf{D}$ecoupled Clip and $\textbf{D}$ynamic s$\textbf{A}$mpling $\textbf{P}$olicy $\textbf{O}$ptimization ($\textbf{DAPO}$) algorithm, and fully open-source a state-of-the-art large-scale RL system that achieves 50 points on AIME 2024 using Qwen2.5-32B base model. Unlike previous works that withhold training details, we introduce four key techniques of our algorithm that make large-scale LLM RL a success. In addition, we open-source our training code, which is built on the verl framework, along with a carefully curated and processed dataset. These components of our open-source system enhance reproducibility and support future research in large-scale LLM RL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.15378v5">A RAG-based Question Answering System Proposal for Understanding Islam: MufassirQAS LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Challenges exist in learning and understanding religions, such as the complexity and depth of religious doctrines and teachings. Chatbots as question-answering systems can help in solving these challenges. LLM chatbots use NLP techniques to establish connections between topics and accurately respond to complex questions. These capabilities make it perfect for enlightenment on religion as a question-answering chatbot. However, LLMs also tend to generate false information, known as hallucination. Also, the chatbots' responses can include content that insults personal religious beliefs, interfaith conflicts, and controversial or sensitive topics. It must avoid such cases without promoting hate speech or offending certain groups of people or their beliefs. This study uses a vector database-based Retrieval Augmented Generation (RAG) approach to enhance the accuracy and transparency of LLMs. Our question-answering system is called "MufassirQAS". We created a database consisting of several open-access books that include Turkish context. These books contain Turkish translations and interpretations of Islam. This database is utilized to answer religion-related questions and ensure our answers are trustworthy. The relevant part of the dataset, which LLM also uses, is presented along with the answer. We have put careful effort into creating system prompts that give instructions to prevent harmful, offensive, or disrespectful responses to respect people's values and provide reliable results. The system answers and shares additional information, such as the page number from the respective book and the articles referenced for obtaining the information. MufassirQAS and ChatGPT are also tested with sensitive questions. We got better performance with our system. Study and enhancements are still in progress. Results and future works are given.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14434v1">LLM-FE: Automated Feature Engineering for Tabular Data with LLMs as Evolutionary Optimizers</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Automated feature engineering plays a critical role in improving predictive model performance for tabular learning tasks. Traditional automated feature engineering methods are limited by their reliance on pre-defined transformations within fixed, manually designed search spaces, often neglecting domain knowledge. Recent advances using Large Language Models (LLMs) have enabled the integration of domain knowledge into the feature engineering process. However, existing LLM-based approaches use direct prompting or rely solely on validation scores for feature selection, failing to leverage insights from prior feature discovery experiments or establish meaningful reasoning between feature generation and data-driven performance. To address these challenges, we propose LLM-FE, a novel framework that combines evolutionary search with the domain knowledge and reasoning capabilities of LLMs to automatically discover effective features for tabular learning tasks. LLM-FE formulates feature engineering as a program search problem, where LLMs propose new feature transformation programs iteratively, and data-driven feedback guides the search process. Our results demonstrate that LLM-FE consistently outperforms state-of-the-art baselines, significantly enhancing the performance of tabular prediction models across diverse classification and regression benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14432v1">PLAY2PROMPT: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integrated with specialized external tools, yet many tasks demand zero-shot tool usage with minimal or noisy documentation. Existing solutions rely on manual rewriting or labeled data for validation, making them inapplicable in true zero-shot settings. To address these challenges, we propose PLAY2PROMPT, an automated framework that systematically "plays" with each tool to explore its input-output behaviors. Through this iterative trial-and-error process, PLAY2PROMPT refines tool documentation and generates usage examples without any labeled data. These examples not only guide LLM inference but also serve as validation to further enhance tool utilization. Extensive experiments on real-world tasks demonstrate that PLAY2PROMPT significantly improves zero-shot tool performance across both open and closed models, offering a scalable and effective solution for domain-specific tool integration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18841v4">Navigating LLM Ethics: Advancements, Challenges, and Future Directions</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      This study addresses ethical issues surrounding Large Language Models (LLMs) within the field of artificial intelligence. It explores the common ethical challenges posed by both LLMs and other AI systems, such as privacy and fairness, as well as ethical challenges uniquely arising from LLMs. It highlights challenges such as hallucination, verifiable accountability, and decoding censorship complexity, which are unique to LLMs and distinct from those encountered in traditional AI systems. The study underscores the need to tackle these complexities to ensure accountability, reduce biases, and enhance transparency in the influential role that LLMs play in shaping information dissemination. It proposes mitigation strategies and future directions for LLM ethics, advocating for interdisciplinary collaboration. It recommends ethical frameworks tailored to specific domains and dynamic auditing systems adapted to diverse contexts. This roadmap aims to guide responsible development and integration of LLMs, envisioning a future where ethical considerations govern AI advancements in society.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14391v1">How much do LLMs learn from negative examples?</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 8 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) undergo a three-phase training process: unsupervised pre-training, supervised fine-tuning (SFT), and learning from human feedback (RLHF/DPO). Notably, it is during the final phase that these models are exposed to negative examples -- incorrect, rejected, or suboptimal responses to queries. This paper delves into the role of negative examples in the training of LLMs, using a likelihood-ratio (Likra) model on multiple-choice question answering benchmarks to precisely manage the influence and the volume of negative examples. Our findings reveal three key insights: (1) During a critical phase in training, Likra with negative examples demonstrates a significantly larger improvement per training example compared to SFT using only positive examples. This leads to a sharp jump in the learning curve for Likra unlike the smooth and gradual improvement of SFT; (2) negative examples that are plausible but incorrect (near-misses) exert a greater influence; and (3) while training with positive examples fails to significantly decrease the likelihood of plausible but incorrect answers, training with negative examples more accurately identifies them. These results indicate a potentially significant role for negative examples in improving accuracy and reducing hallucinations for LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14382v1">Good/Evil Reputation Judgment of Celebrities by LLMs via Retrieval Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      The purpose of this paper is to examine whether large language models (LLMs) can understand what is good and evil with respect to judging good/evil reputation of celebrities. Specifically, we first apply a large language model (namely, ChatGPT) to the task of collecting sentences that mention the target celebrity from articles about celebrities on Web pages. Next, the collected sentences are categorized based on their contents by ChatGPT, where ChatGPT assigns a category name to each of those categories. Those assigned category names are referred to as "aspects" of each celebrity. Then, by applying the framework of retrieval augmented generation (RAG), we show that the large language model is quite effective in the task of judging good/evil reputation of aspects and descriptions of each celebrity. Finally, also in terms of proving the advantages of the proposed method over existing services incorporating RAG functions, we show that the proposed method of judging good/evil of aspects/descriptions of each celebrity significantly outperform an existing service incorporating RAG functions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.07917v2">CryptoLLM: Unleashing the Power of Prompted LLMs for SmartQnA and Classification of Crypto Posts</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 Updated and Final Version
    </div>
    <details class="paper-abstract">
      The rapid growth of social media has resulted in an large volume of user-generated content, particularly in niche domains such as cryptocurrency. This task focuses on developing robust classification models to accurately categorize cryptocurrency-related social media posts into predefined classes, including but not limited to objective, positive, negative, etc. Additionally, the task requires participants to identify the most relevant answers from a set of posts in response to specific questions. By leveraging advanced LLMs, this research aims to enhance the understanding and filtering of cryptocurrency discourse, thereby facilitating more informed decision-making in this volatile sector. We have used a prompt-based technique to solve the classification task for reddit posts and twitter posts. Also, we have used 64-shot technique along with prompts on GPT-4-Turbo model to determine whether a answer is relevant to a question or not.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14340v1">MANTRA: Enhancing Automated Method-Level Refactoring with Contextual RAG and Multi-Agent LLM Collaboration</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 10 pages
    </div>
    <details class="paper-abstract">
      Maintaining and scaling software systems relies heavily on effective code refactoring, yet this process remains labor-intensive, requiring developers to carefully analyze existing codebases and prevent the introduction of new defects. Although recent advancements have leveraged Large Language Models (LLMs) to automate refactoring tasks, current solutions are constrained in scope and lack mechanisms to guarantee code compilability and successful test execution. In this work, we introduce MANTRA, a comprehensive LLM agent-based framework that automates method-level refactoring. MANTRA integrates Context-Aware Retrieval-Augmented Generation, coordinated Multi-Agent Collaboration, and Verbal Reinforcement Learning to emulate human decision-making during refactoring while preserving code correctness and readability. Our empirical study, conducted on 703 instances of "pure refactorings" (i.e., code changes exclusively involving structural improvements), drawn from 10 representative Java projects, covers the six most prevalent refactoring operations. Experimental results demonstrate that MANTRA substantially surpasses a baseline LLM model (RawGPT ), achieving an 82.8% success rate (582/703) in producing code that compiles and passes all tests, compared to just 8.7% (61/703) with RawGPT. Moreover, in comparison to IntelliJ's LLM-powered refactoring tool (EM-Assist), MANTRA exhibits a 50% improvement in generating Extract Method transformations. A usability study involving 37 professional developers further shows that refactorings performed by MANTRA are perceived to be as readable and reusable as human-written code, and in certain cases, even more favorable. These results highlight the practical advantages of MANTRA and emphasize the growing potential of LLM-based systems in advancing the automation of software refactoring tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14506v2">InteLiPlan: An Interactive Lightweight LLM-Based Planner for Domestic Robot Autonomy</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      We introduce an interactive LLM-based framework designed to enhance the autonomy and robustness of domestic robots, targeting embodied intelligence. Our approach reduces reliance on large-scale data and incorporates a robot-agnostic pipeline that embodies an LLM. Our framework, InteLiPlan, ensures that the LLM's decision-making capabilities are effectively aligned with robotic functions, enhancing operational robustness and adaptability, while our human-in-the-loop mechanism allows for real-time human intervention when user instruction is required. We evaluate our method in both simulation and on the real Toyota Human Support Robot (HSR). Our method achieves a 93% success rate in the 'fetch me' task completion with failure recovery, highlighting its capability in both failure reasoning and task planning. InteLiPlan achieves comparable performance to state-of-the-art large-scale LLM-based robotics planners, while using only real-time onboard computing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09647v2">Leveraging LLMS for Top-Down Sector Allocation In Automated Trading</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      This paper introduces a methodology leveraging Large Language Models (LLMs) for sector-level portfolio allocation through systematic analysis of macroeconomic conditions and market sentiment. Our framework emphasizes top-down sector allocation by processing multiple data streams simultaneously, including policy documents, economic indicators, and sentiment patterns. Empirical results demonstrate superior risk-adjusted returns compared to traditional cross momentum strategies, achieving a Sharpe ratio of 2.51 and portfolio return of 8.79% versus -0.61 and -1.39% respectively. These results suggest that LLM-based systematic macro analysis presents a viable approach for enhancing automated portfolio allocation decisions at the sector level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.13788v2">Modeling Future Conversation Turns to Teach LLMs to Ask Clarifying Questions</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 Presented at ICLR 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) must often respond to highly ambiguous user requests. In such cases, the LLM's best response may be to ask a clarifying question to elicit more information. Existing LLMs often respond by presupposing a single interpretation of such ambiguous requests, frustrating users who intended a different interpretation. We speculate this is caused by current preference data labeling practice, where LLM responses are evaluated only on their prior contexts. To address this, we assign preference labels by simulating their expected outcomes in future turns. This allows LLMs to learn to ask clarifying questions when it can generate responses that are tailored to each user interpretation in future turns. On open-domain QA datasets with multiple annotations, we evaluate systems based on their ability to ask clarifying questions to recover each user's interpretation and expected answer. We compare systems trained using our proposed preference labeling methods against standard methods, which assign preferences based on only prior context. Our method achieves a 5% improvement in F1 measured against the answer set from different interpretations of each query, showing the value of modeling future conversation turns. We further demonstrate that our method can be used to train models to judiciously determine when to ask clarifying questions, directly answering the question when clarification is unnecessary. In our experiments, we find that our method achieves a 3% improvement in accuracy of such judgments over existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14251v1">Towards a Barrier-free GeoQA Portal: Natural Language Interaction with Geospatial Data Using Multi-Agent LLMs and Semantic Search</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      A Barrier-Free GeoQA Portal: Enhancing Geospatial Data Accessibility with a Multi-Agent LLM Framework Geoportals are vital for accessing and analyzing geospatial data, promoting open spatial data sharing and online geo-information management. Designed with GIS-like interaction and layered visualization, they often challenge non-expert users with complex functionalities and overlapping layers that obscure spatial relationships. We propose a GeoQA Portal using a multi-agent Large Language Model framework for seamless natural language interaction with geospatial data. Complex queries are broken into subtasks handled by specialized agents, retrieving relevant geographic data efficiently. Task plans are shown to users, boosting transparency. The portal supports default and custom data inputs for flexibility. Semantic search via word vector similarity aids data retrieval despite imperfect terms. Case studies, evaluations, and user tests confirm its effectiveness for non-experts, bridging GIS complexity and public access, and offering an intuitive solution for future geoportals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14217v1">Decision Tree Induction Through LLMs via Semantically-Aware Evolution</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 *Liu and Huynh contributed equally. Published as a conference paper at ICLR 2025
    </div>
    <details class="paper-abstract">
      Decision trees are a crucial class of models offering robust predictive performance and inherent interpretability across various domains, including healthcare, finance, and logistics. However, current tree induction methods often face limitations such as suboptimal solutions from greedy methods or prohibitive computational costs and limited applicability of exact optimization approaches. To address these challenges, we propose an evolutionary optimization method for decision tree induction based on genetic programming (GP). Our key innovation is the integration of semantic priors and domain-specific knowledge about the search space into the optimization algorithm. To this end, we introduce $\texttt{LLEGO}$, a framework that incorporates semantic priors into genetic search operators through the use of Large Language Models (LLMs), thereby enhancing search efficiency and targeting regions of the search space that yield decision trees with superior generalization performance. This is operationalized through novel genetic operators that work with structured natural language prompts, effectively utilizing LLMs as conditional generative models and sources of semantic knowledge. Specifically, we introduce $\textit{fitness-guided}$ crossover to exploit high-performing regions, and $\textit{diversity-guided}$ mutation for efficient global exploration of the search space. These operators are controlled by corresponding hyperparameters that enable a more nuanced balance between exploration and exploitation across the search space. Empirically, we demonstrate across various benchmarks that $\texttt{LLEGO}$ evolves superior-performing trees compared to existing tree induction methods, and exhibits significantly more efficient search performance compared to conventional GP approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14183v1">Can LLMs Enable Verification in Mainstream Programming?</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Although formal methods are capable of producing reliable software, they have seen minimal adoption in everyday programming. Automatic code generation using large language models is becoming increasingly widespread, but it rarely considers producing strong correctness guarantees. In this study, we explore the ability of LLMs to produce verified code in three verification languages (Dafny, Nagini, and Verus). To do so, we use manually curated datasets derived from the state-ofthe-art Python benchmark, HumanEval. We also assess what types of information are sufficient to achieve good-quality results.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17178v3">Tuning LLM Judge Design Decisions for 1/1000 of the Cost</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) often requires costly human annotations. To address this, LLM-based judges have been proposed, which compare the outputs of two LLMs enabling the ranking of models without human intervention. While several approaches have been proposed, many confounding factors are present between different papers. For instance the model, the prompt and other hyperparameters are typically changed at the same time making apple-to-apple comparisons challenging. In this paper, we propose to systematically analyze and tune hyperparameter of LLM judges. To alleviate the high cost of evaluating a judge, we propose to leverage multi-objective multi-fidelity which allows to find judges that trades accuracy for cost and also reduce significantly the cost of the search. Our method identifies judges that not only outperform existing benchmarks in accuracy and cost-efficiency but also utilize open-weight models, ensuring greater accessibility and reproducibility.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14043v1">Learning on LLM Output Signatures for gray-box LLM Behavior Analysis</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved widespread adoption, yet our understanding of their behavior remains limited, particularly in detecting data contamination and hallucinations. While recently proposed probing techniques provide insights through activation analysis, they require "white-box" access to model internals, often unavailable. Current "gray-box" approaches typically analyze only the probability of the actual tokens in the sequence with simple task-specific heuristics. Importantly, these methods overlook the rich information contained in the full token distribution at each processing step. To address these limitations, we propose that gray-box analysis should leverage the complete observable output of LLMs, consisting of both the previously used token probabilities as well as the complete token distribution sequences - a unified data type we term LOS (LLM Output Signature). To this end, we develop a transformer-based approach to process LOS that theoretically guarantees approximation of existing techniques while enabling more nuanced analysis. Our approach achieves superior performance on hallucination and data contamination detection in gray-box settings, significantly outperforming existing baselines. Furthermore, it demonstrates strong transfer capabilities across datasets and LLMs, suggesting that LOS captures fundamental patterns in LLM behavior. Our code is available at: https://github.com/BarSGuy/LLM-Output-Signatures-Network.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05592v2">R1-Searcher: Incentivizing the Search Capability in LLMs via Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Existing Large Reasoning Models (LRMs) have shown the potential of reinforcement learning (RL) to enhance the complex reasoning capabilities of Large Language Models~(LLMs). While they achieve remarkable performance on challenging tasks such as mathematics and coding, they often rely on their internal knowledge to solve problems, which can be inadequate for time-sensitive or knowledge-intensive questions, leading to inaccuracies and hallucinations. To address this, we propose \textbf{R1-Searcher}, a novel two-stage outcome-based RL approach designed to enhance the search capabilities of LLMs. This method allows LLMs to autonomously invoke external search systems to access additional knowledge during the reasoning process. Our framework relies exclusively on RL, without requiring process rewards or distillation for a cold start. % effectively generalizing to out-of-domain datasets and supporting both Base and Instruct models. Our experiments demonstrate that our method significantly outperforms previous strong RAG methods, even when compared to the closed-source GPT-4o-mini.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14000v1">LLM-based Unit Test Generation for Dynamically-Typed Programs</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Automated unit test generation has been widely studied, but generating effective tests for dynamically typed programs remains a significant challenge. Existing approaches, including search-based software testing (SBST) and recent LLM-based methods, often suffer from type errors, leading to invalid inputs and assertion failures, ultimately reducing testing effectiveness. To address this, we propose TypeTest, a novel framework that enhances type correctness in test generation through a vector-based Retrieval-Augmented Generation (RAG) system. TypeTest employs call instance retrieval and feature-based retrieval to infer parameter types accurately and construct valid test inputs. Furthermore, it utilizes the call graph to extract richer contextual information, enabling more accurate assertion generation. In addition, TypeTest incorporates a repair mechanism and iterative test generation, progressively refining test cases to improve coverage. In an evaluation on 125 real-world Python modules, TypeTest achieved an average statement coverage of 86.6% and branch coverage of 76.8%, outperforming state-of-theart tools by 5.4% and 9.3%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12918v2">Query Rewriting via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      When complex SQL queries suffer slow executions despite query optimization, DBAs typically invoke automated query rewriting tools to recommend ``lean'' equivalents that are conducive to faster execution. The rewritings are usually achieved via transformation rules, but these rules are limited in scope and difficult to update in a production system. Recently, LLM-based techniques have also been suggested, but they are prone to semantic and syntactic errors. We investigate here how the remarkable cognitive capabilities of LLMs can be leveraged for performant query rewriting while incorporating safeguards and optimizations to ensure correctness and efficiency. Our study shows that these goals can be progressively achieved through incorporation of (a) an ensemble suite of basic prompts, (b) database-sensitive prompts via redundancy removal and selectivity-based rewriting rules, and (c) LLM token probability-guided rewrite paths. Further, a suite of logic-based and statistical tools can be used to check for semantic violations in the rewrites prior to DBA consideration. We have implemented the above LLM-infused techniques in the LITHE system, and evaluated complex analytic queries from standard benchmarks on contemporary database platforms. The results show significant performance improvements for slow queries, with regard to both abstract costing and actual execution, over both SOTA techniques and the native query optimizer. For instance, with TPC-DS on PostgreSQL, the geometric mean of the runtime speedups for slow queries was as high as 18.4 over the native optimizer, whereas SOTA delivered 6 in comparison. Overall, LITHE is a promising step toward viable LLM-based advisory tools for ameliorating enterprise query performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13980v1">Empowering LLMs in Decision Games through Algorithmic Data Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have exhibited impressive capabilities across numerous domains, yet they often struggle with complex reasoning and decision-making tasks. Decision-making games, which inherently require multifaceted reasoning logic, serve as ideal sandboxes for evaluating and enhancing the reasoning abilities of LLMs. In this work, we first explore whether LLMs can master complex decision-making games through targeted post-training. To this end, we design data synthesis strategies and curate extensive offline datasets from two classic games, Doudizhu and Go. We further develop a suite of techniques to effectively incorporate this data into LLM training, resulting in two novel agents: Mastermind-Dou and Mastermind-Go. Our experimental results demonstrate that these Mastermind LLMs achieve competitive performance in their respective games. Additionally, we explore whether integrating decision-making data can enhance the general reasoning abilities of LLMs. Our findings suggest that such post-training improves certain aspects of reasoning, providing valuable insights for optimizing LLM data collection and synthesis strategies.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13975v1">Navigating Rifts in Human-LLM Grounding: Study and Benchmark</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 16 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Language models excel at following instructions but often struggle with the collaborative aspects of conversation that humans naturally employ. This limitation in grounding -- the process by which conversation participants establish mutual understanding -- can lead to outcomes ranging from frustrated users to serious consequences in high-stakes scenarios. To systematically study grounding challenges in human-LLM interactions, we analyze logs from three human-assistant datasets: WildChat, MultiWOZ, and Bing Chat. We develop a taxonomy of grounding acts and build models to annotate and forecast grounding behavior. Our findings reveal significant differences in human-human and human-LLM grounding: LLMs were three times less likely to initiate clarification and sixteen times less likely to provide follow-up requests than humans. Additionally, early grounding failures predicted later interaction breakdowns. Building on these insights, we introduce RIFTS: a benchmark derived from publicly available LLM interaction data containing situations where LLMs fail to initiate grounding. We note that current frontier models perform poorly on RIFTS, highlighting the need to reconsider how we train and prompt LLMs for human interaction. To this end, we develop a preliminary intervention that mitigates grounding failures.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13956v1">Improving LLM Video Understanding with 16 Frames Per Second</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Human vision is dynamic and continuous. However, in video understanding with multimodal large language models (LLMs), existing methods primarily rely on static features extracted from images sampled at a fixed low frame rate of frame-per-second (FPS) $\leqslant$2, leading to critical visual information loss. In this paper, we introduce F-16, the first multimodal LLM designed for high-frame-rate video understanding. By increasing the frame rate to 16 FPS and compressing visual tokens within each 1-second clip, F-16 efficiently captures dynamic visual features while preserving key semantic information. Experimental results demonstrate that higher frame rates considerably enhance video understanding across multiple benchmarks, providing a new approach to improving video LLMs beyond scaling model size or training data. F-16 achieves state-of-the-art performance among 7-billion-parameter video LLMs on both general and fine-grained video understanding benchmarks, such as Video-MME and TemporalBench. Furthermore, F-16 excels in complex spatiotemporal tasks, including high-speed sports analysis (\textit{e.g.}, basketball, football, gymnastics, and diving), outperforming SOTA proprietary visual models like GPT-4o and Gemini-1.5-pro. Additionally, we introduce a novel decoding method for F-16 that enables highly efficient low-frame-rate inference without requiring model retraining. Upon acceptance, we will release the source code, model checkpoints, and data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.03586v2">Benchmarking LLMs and LLM-based Agents in Practical Vulnerability Detection for Code Repositories</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown promise in software vulnerability detection, particularly on function-level benchmarks like Devign and BigVul. However, real-world detection requires interprocedural analysis, as vulnerabilities often emerge through multi-hop function calls rather than isolated functions. While repository-level benchmarks like ReposVul and VulEval introduce interprocedural context, they remain computationally expensive, lack pairwise evaluation of vulnerability fixes, and explore limited context retrieval, limiting their practicality. We introduce JitVul, a JIT vulnerability detection benchmark linking each function to its vulnerability-introducing and fixing commits. Built from 879 CVEs spanning 91 vulnerability types, JitVul enables comprehensive evaluation of detection capabilities. Our results show that ReAct Agents, leveraging thought-action-observation and interprocedural context, perform better than LLMs in distinguishing vulnerable from benign code. While prompting strategies like Chain-of-Thought help LLMs, ReAct Agents require further refinement. Both methods show inconsistencies, either misidentifying vulnerabilities or over-analyzing security guards, indicating significant room for improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11951v2">SagaLLM: Context Management, Validation, and Transaction Guarantees for Multi-Agent LLM Planning</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 13 pages, 8 tables, 5 figures
    </div>
    <details class="paper-abstract">
      Recent LLM-based agent frameworks have demonstrated impressive capabilities in task delegation and workflow orchestration, but face significant challenges in maintaining context awareness and ensuring planning consistency. This paper presents SagaLLM, a structured multi-agent framework that addresses four fundamental limitations in current LLM approaches: inadequate self-validation, context narrowing, lacking transaction properties, and insufficient inter-agent coordination. By implementing specialized context management agents and validation protocols, SagaLLM preserves critical constraints and state information throughout complex planning processes, enabling robust and consistent decision-making even during disruptions. We evaluate our approach using selected problems from the REALM benchmark, focusing on sequential and reactive planning scenarios that challenge both context retention and adaptive reasoning. Our experiments with state-of-the-art LLMs, Claude 3.7, DeepSeek R1, GPT-4o, and GPT-o1, demonstrate that while these models exhibit impressive reasoning capabilities, they struggle with maintaining global constraint awareness during complex planning tasks, particularly when adapting to unexpected changes. In contrast, the distributed cognitive architecture of SagaLLM shows significant improvements in planning consistency, constraint enforcement, and adaptation to disruptions in various scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12511v2">LLM-Driven Multi-step Translation from C to Rust using Static Analysis</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 22 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Translating software written in legacy languages to modern languages, such as C to Rust, has significant benefits in improving memory safety while maintaining high performance. However, manual translation is cumbersome, error-prone, and produces unidiomatic code. Large language models (LLMs) have demonstrated promise in producing idiomatic translations, but offer no correctness guarantees as they lack the ability to capture all the semantics differences between the source and target languages. To resolve this issue, we propose SACTOR, an LLM-driven C-to-Rust zero-shot translation tool using a two-step translation methodology: an "unidiomatic" step to translate C into Rust while preserving semantics, and an "idiomatic" step to refine the code to follow Rust's semantic standards. SACTOR utilizes information provided by static analysis of the source C program to address challenges such as pointer semantics and dependency resolution. To validate the correctness of the translated result from each step, we use end-to-end testing via the foreign function interface to embed our translated code segment into the original code. We evaluate the translation of 200 programs from two datasets and two case studies, comparing the performance of GPT-4o, Claude 3.5 Sonnet, Gemini 2.0 Flash, Llama 3.3 70B and DeepSeek-R1 in SACTOR. Our results demonstrate that SACTOR achieves high correctness and improved idiomaticity, with the best-performing model (DeepSeek-R1) reaching 93% and (GPT-4o, Claude 3.5, DeepSeek-R1) reaching 84% correctness (on each dataset, respectively), while producing more natural and Rust-compliant translations compared to existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13879v1">Bridging Social Psychology and LLM Reasoning: Conflict-Aware Meta-Review Generation via Cognitive Alignment</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      The rapid growth of scholarly submissions has overwhelmed traditional peer review systems, driving the need for intelligent automation to preserve scientific rigor. While large language models (LLMs) show promise in automating manuscript critiques, their ability to synthesize high-stakes meta-reviews, which require conflict-aware reasoning and consensus derivation, remains underdeveloped. Existing methods fail to effectively handle conflicting viewpoints within differing opinions, and often introduce additional cognitive biases, such as anchoring effects and conformity bias.To overcome these limitations, we propose the Cognitive Alignment Framework (CAF), a dual-process architecture that transforms LLMs into adaptive scientific arbitrators. By operationalizing Kahneman's dual-process theory, CAF introduces a three-step cognitive pipeline: review initialization, incremental integration, and cognitive alignment.Empirical validation shows that CAF outperforms existing LLM-based methods, with sentiment consistency gains reaching up to 19.47\% and content consistency improving by as much as 12.95\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11911v2">LAG-MMLU: Benchmarking Frontier LLM Understanding in Latvian and Giriama</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 Accepted at NoDaLiDa/Baltic-HLT 2025. https://hdl.handle.net/10062/107190
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) rapidly advance, evaluating their performance is critical. LLMs are trained on multilingual data, but their reasoning abilities are mainly evaluated using English datasets. Hence, robust evaluation frameworks are needed using high-quality non-English datasets, especially low-resource languages (LRLs). This study evaluates eight state-of-the-art (SOTA) LLMs on Latvian and Giriama using a Massive Multitask Language Understanding (MMLU) subset curated with native speakers for linguistic and cultural relevance. Giriama is benchmarked for the first time. Our evaluation shows that OpenAI's o1 model outperforms others across all languages, scoring 92.8% in English, 88.8% in Latvian, and 70.8% in Giriama on 0-shot tasks. Mistral-large (35.6%) and Llama-70B IT (41%) have weak performance, on both Latvian and Giriama. Our results underscore the need for localized benchmarks and human evaluations in advancing cultural AI contextualization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13856v1">MDTeamGPT: A Self-Evolving LLM-based Multi-Agent Framework for Multi-Disciplinary Team Medical Consultation</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 24 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have made significant progress in various fields. However, challenges remain in Multi-Disciplinary Team (MDT) medical consultations. Current research enhances reasoning through role assignment, task decomposition, and accumulation of medical experience. Multi-role collaboration in MDT consultations often results in excessively long dialogue histories. This increases the model's cognitive burden and degrades both efficiency and accuracy. Some methods only store treatment histories. They do not extract effective experience or reflect on errors. This limits knowledge generalization and system evolution. We propose a multi-agent MDT medical consultation framework based on LLMs to address these issues. Our framework uses consensus aggregation and a residual discussion structure for multi-round consultations. It also employs a Correct Answer Knowledge Base (CorrectKB) and a Chain-of-Thought Knowledge Base (ChainKB) to accumulate consultation experience. These mechanisms enable the framework to evolve and continually improve diagnosis rationality and accuracy. Experimental results on the MedQA and PubMedQA datasets demonstrate that our framework achieves accuracies of 90.1% and 83.9%, respectively, and that the constructed knowledge bases generalize effectively across test sets from both datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13819v1">LLM-Empowered IoT for 6G Networks: Architecture, Challenges, and Solutions</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      The Internet of Things (IoT) in the sixth generation (6G) era is envisioned to evolve towards intelligence, ubiquity, and self-optimization. Large language models (LLMs) have demonstrated remarkable generalization capabilities across diverse domains, including natural language processing (NLP), computer vision (CV), and beyond. In this article, we propose an LLM-empowered IoT architecture for 6G networks to achieve intelligent autonomy while supporting advanced IoT applications. LLMs are pushed to the edge of the 6G network to support the synergy of LLMs and IoT. LLM solutions are tailored to both IoT application requirements and IoT management needs, i.e., LLM for IoT. On the other hand, edge inference and edge fine-tuning are discussed to support the deployment of LLMs, i.e., LLM on IoT. Furthermore, we propose a memory-efficient split federated learning (SFL) framework for LLM fine-tuning on heterogeneous IoT devices that alleviates memory pressures on both IoT devices and the edge server while achieving comparable performance and convergence time. Finally, a case study is presented, followed by a discussion about open issues of LLM-empowered IoT for 6G networks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09657v2">Týr-the-Pruner: Unlocking Accurate 50% Structural Pruning for LLMs via Global Sparsity Distribution Optimization</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Structural pruning enhances hardware-agnostic inference efficiency for large language models (LLMs) but often struggles to maintain performance. Local pruning performs efficient layer-by-layer compression but ignores global topology. Global pruning has the potential to find the optimal solution although resource-intensive. However, existing methods tend to rank structural saliency uniformly, ignoring inter-structure dependencies and failing to achieve end-to-end optimization. To address these limitations, we propose T\'yr-the-Pruner, an efficient end-to-end search-based global structural pruning framework. This framework constructs a supernet by repeatedly applying local pruning across a range of sparsity ratios to each layer in an LLM, with the core goal of determining the optimal sparsity distribution under a target overall sparsity ratio. Concretely, we introduce an effective local pruning and an expectation error accumulation approach to improve supernet construction. Furthermore, we employ an iterative prune-and-search strategy with coarse-to-fine sparsity granularity to ensure efficient search convergence. Experimental results show that T\'yr-the-Pruner achieves state-of-the-art structural pruning, retaining 97% of the dense model's performance while removing a challenging 50% of Llama-3.1-70B's parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13812v1">The Empty Chair: Using LLMs to Raise Missing Perspectives in Policy Deliberations</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Deliberation is essential to well-functioning democracies, yet physical, economic, and social barriers often exclude certain groups, reducing representativeness and contributing to issues like group polarization. In this work, we explore the use of large language model (LLM) personas to introduce missing perspectives in policy deliberations. We develop and evaluate a tool that transcribes conversations in real-time and simulates input from relevant but absent stakeholders. We deploy this tool in a 19-person student citizens' assembly on campus sustainability. Participants and facilitators found that the tool sparked new discussions and surfaced valuable perspectives they had not previously considered. However, they also noted that AI-generated responses were sometimes overly general. They raised concerns about overreliance on AI for perspective-taking. Our findings highlight both the promise and potential risks of using LLMs to raise missing points of view in group deliberation settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12349v2">SPIN-Bench: How Well Do LLMs Plan Strategically and Reason Socially?</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 51 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Reasoning and strategic behavior in social interactions is a hallmark of intelligence. This form of reasoning is significantly more sophisticated than isolated planning or reasoning tasks in static settings (e.g., math problem solving). In this paper, we present Strategic Planning, Interaction, and Negotiation (SPIN-Bench), a new multi-domain evaluation designed to measure the intelligence of strategic planning and social reasoning. While many existing benchmarks focus on narrow planning or single-agent reasoning, SPIN-Bench combines classical PDDL tasks, competitive board games, cooperative card games, and multi-agent negotiation scenarios in one unified framework. The framework includes both a benchmark as well as an arena to simulate and evaluate the variety of social settings to test reasoning and strategic behavior of AI agents. We formulate the benchmark SPIN-Bench by systematically varying action spaces, state complexity, and the number of interacting agents to simulate a variety of social settings where success depends on not only methodical and step-wise decision making, but also conceptual inference of other (adversarial or cooperative) participants. Our experiments reveal that while contemporary LLMs handle basic fact retrieval and short-range planning reasonably well, they encounter significant performance bottlenecks in tasks requiring deep multi-hop reasoning over large state spaces and socially adept coordination under uncertainty. We envision SPIN-Bench as a catalyst for future research on robust multi-agent planning, social reasoning, and human--AI teaming. Project Website: https://spinbench.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13794v1">LED: LLM Enhanced Open-Vocabulary Object Detection without Human Curated Data Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Large foundation models trained on large-scale visual-text data can significantly enhance Open Vocabulary Object Detection (OVD) through data generation. However, this may lead to biased synthetic data and overfitting to specific configurations. It can sidestep biases of manually curated data generation by directly leveraging hidden states of Large Language Models (LLMs), which is surprisingly rarely explored. This paper presents a systematic method to enhance visual grounding by utilizing decoder layers of the LLM of a MLLM. We introduce a zero-initialized cross-attention adapter to enable efficient knowledge transfer from LLMs to object detectors, an new approach called LED (LLM Enhanced Open-Vocabulary Object Detection). We demonstrate that intermediate hidden states from early LLM layers retain strong spatial-semantic correlations that are beneficial to grounding tasks. Experiments show that our adaptation strategy significantly enhances the performance on complex free-form text queries while remaining the same on plain categories. With our adaptation, Qwen2-0.5B with Swin-T as the vision encoder improves GroundingDINO by 2.33% on Omnilabel, at the overhead of 8.7% more GFLOPs. Qwen2-0.5B with a larger vision encoder can further boost the performance by 6.22%. We further validate our design by ablating on varied adapter architectures, sizes of LLMs, and which layers to add adaptation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13793v1">Mapping the Trust Terrain: LLMs in Software Engineering -- Insights and Perspectives</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Applications of Large Language Models (LLMs) are rapidly growing in industry and academia for various software engineering (SE) tasks. As these models become more integral to critical processes, ensuring their reliability and trustworthiness becomes essential. Consequently, the concept of trust in these systems is becoming increasingly critical. Well-calibrated trust is important, as excessive trust can lead to security vulnerabilities, and risks, while insufficient trust can hinder innovation. However, the landscape of trust-related concepts in LLMs in SE is relatively unclear, with concepts such as trust, distrust, and trustworthiness lacking clear conceptualizations in the SE community. To bring clarity to the current research status and identify opportunities for future work, we conducted a comprehensive review of $88$ papers: a systematic literature review of $18$ papers focused on LLMs in SE, complemented by an analysis of 70 papers from broader trust literature. Additionally, we conducted a survey study with 25 domain experts to gain insights into practitioners' understanding of trust and identify gaps between existing literature and developers' perceptions. The result of our analysis serves as a roadmap that covers trust-related concepts in LLMs in SE and highlights areas for future exploration.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10167v2">"Well, Keep Thinking": Enhancing LLM Reasoning with Adaptive Injection Decoding</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit strong reasoning abilities, often attributed to few-shot or zero-shot chain-of-thought (CoT) prompting. While effective, these methods require labor-intensive prompt engineering, raising the question of whether reasoning can be induced without reliance on explicit prompts. In this work, we unlock the reasoning capabilities of LLMs without explicit prompting. Inspired by zero-shot CoT and CoT-decoding, we propose a novel decoding strategy that systematically nudges LLMs to continue reasoning, thereby preventing immature reasoning processes. Specifically, we monitor the model's generation and inject a designated phrase whenever it is likely to conclude its response prematurely, before completing the reasoning process. Our experimental evaluations on diverse reasoning benchmarks demonstrate that our proposed strategy substantially improves LLM reasoning capabilities, highlighting the potential of decoding-based interventions as an alternative to traditional prompting techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14724v1">CodingGenie: A Proactive LLM-Powered Programming Assistant</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 FSE Demo 2025
    </div>
    <details class="paper-abstract">
      While developers increasingly adopt tools powered by large language models (LLMs) in day-to-day workflows, these tools still require explicit user invocation. To seamlessly integrate LLM capabilities to a developer's workflow, we introduce CodingGenie, a proactive assistant integrated into the code editor. CodingGenie autonomously provides suggestions, ranging from bug fixing to unit testing, based on the current code context and allows users to customize suggestions by providing a task description and selecting what suggestions are shown. We demonstrate multiple use cases to show how proactive suggestions from CodingGenie can improve developer experience, and also analyze the cost of adding proactivity. We believe this open-source tool will enable further research into proactive assistants. CodingGenie is open-sourced at https://github.com/sebzhao/CodingGenie/ and video demos are available at https://sebzhao.github.io/CodingGenie/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14671v1">Generating Medically-Informed Explanations for Depression Detection using LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Early detection of depression from social media data offers a valuable opportunity for timely intervention. However, this task poses significant challenges, requiring both professional medical knowledge and the development of accurate and explainable models. In this paper, we propose LLM-MTD (Large Language Model for Multi-Task Depression Detection), a novel approach that leverages a pre-trained large language model to simultaneously classify social media posts for depression and generate textual explanations grounded in medical diagnostic criteria. We train our model using a multi-task learning framework with a combined loss function that optimizes both classification accuracy and explanation quality. We evaluate LLM-MTD on the benchmark Reddit Self-Reported Depression Dataset (RSDD) and compare its performance against several competitive baseline methods, including traditional machine learning and fine-tuned BERT. Our experimental results demonstrate that LLM-MTD achieves state-of-the-art performance in depression detection, showing significant improvements in AUPRC and other key metrics. Furthermore, human evaluation of the generated explanations reveals their relevance, completeness, and medical accuracy, highlighting the enhanced interpretability of our approach. This work contributes a novel methodology for depression detection that combines the power of large language models with the crucial aspect of explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12757v2">MAP: Multi-user Personalization with Collaborative LLM-powered Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-18
      | 💬 In Extended Abstracts of the CHI Conference on Human Factors in Computing Systems (CHI EA '25), April 26-May 1, 2025, Yokohama, Japan
    </div>
    <details class="paper-abstract">
      The widespread adoption of Large Language Models (LLMs) and LLM-powered agents in multi-user settings underscores the need for reliable, usable methods to accommodate diverse preferences and resolve conflicting directives. Drawing on conflict resolution theory, we introduce a user-centered workflow for multi-user personalization comprising three stages: Reflection, Analysis, and Feedback. We then present MAP -- a \textbf{M}ulti-\textbf{A}gent system for multi-user \textbf{P}ersonalization -- to operationalize this workflow. By delegating subtasks to specialized agents, MAP (1) retrieves and reflects on relevant user information, while enhancing reliability through agent-to-agent interactions, (2) provides detailed analysis for improved transparency and usability, and (3) integrates user feedback to iteratively refine results. Our user study findings (n=12) highlight MAP's effectiveness and usability for conflict resolution while emphasizing the importance of user involvement in resolution verification and failure management. This work highlights the potential of multi-agent systems to implement user-centered, multi-user personalization workflows and concludes by offering insights for personalization in multi-user contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14647v1">Towards More Economical Context-Augmented LLM Generation by Reusing Stored KV Cache</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Across large language model (LLM) applications, we observe an emerging trend for reusing KV caches to save the prefill delays of processing repeated input texts in different LLM inputs. This has led to a broad design space, including colocating stored KV caches with (or close to) GPUs to various KV cache compression. However, a key question remains unanswered: can these delay reductions also be economically favorable? Specifically, we ask whether a developer can use public cloud services to store precomputed KV caches and reuse them to save delay without incurring more costs in terms of compute, storage, and network. To answer this question, we propose an validated analytical model for the cloud cost (in compute, storage, and network) of storing and reusing KV caches based on various workload parameters, such as reuse frequency, generated text lengths, model sizes, etc. Preliminary results show that KV cache reusing is able to save both delay and cloud cost across a range of workloads with long context. And we call more efforts on building more economical context augmented LLM by KV cache reusing.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14603v1">Command R7B Arabic: A Small, Enterprise Focused, Multilingual, and Culturally Aware Arabic LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-18
    </div>
    <details class="paper-abstract">
      Building high-quality large language models (LLMs) for enterprise Arabic applications remains challenging due to the limited availability of digitized Arabic data. In this work, we present a data synthesis and refinement strategy to help address this problem, namely, by leveraging synthetic data generation and human-in-the-loop annotation to expand our Arabic training corpus. We further present our iterative post training recipe that is essential to achieving state-of-the-art performance in aligning the model with human preferences, a critical aspect to enterprise use cases. The culmination of this effort is the release of a small, 7B, open-weight model that outperforms similarly sized peers in head-to-head comparisons and on Arabic-focused benchmarks covering cultural knowledge, instruction following, RAG, and contextual faithfulness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13445v1">Faithfulness of LLM Self-Explanations for Commonsense Tasks: Larger Is Better, and Instruction-Tuning Allows Trade-Offs but Not Pareto Dominance</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 38 pages, 9 figures
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly capable, ensuring that their self-generated explanations are faithful to their internal decision-making process is critical for safety and oversight. In this work, we conduct a comprehensive counterfactual faithfulness analysis across 62 models from 8 families, encompassing both pretrained and instruction-tuned variants and significantly extending prior studies of counterfactual tests. We introduce phi-CCT, a simplified variant of the Correlational Counterfactual Test, which avoids the need for token probabilities while explaining most of the variance of the original test. Our findings reveal clear scaling trends: larger models are consistently more faithful on our metrics. However, when comparing instruction-tuned and human-imitated explanations, we find that observed differences in faithfulness can often be attributed to explanation verbosity, leading to shifts along the true-positive/false-positive Pareto frontier. While instruction-tuning and prompting can influence this trade-off, we find limited evidence that they fundamentally expand the frontier of explanatory faithfulness beyond what is achievable with pretrained models of comparable size. Our analysis highlights the nuanced relationship between instruction-tuning, verbosity, and the faithful representation of model decision processes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13427v1">xLSTM 7B: A Recurrent LLM for Fast and Efficient Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 Code available at: https://github.com/NX-AI/xlstm and https://github.com/NX-AI/xlstm-jax
    </div>
    <details class="paper-abstract">
      Recent breakthroughs in solving reasoning, math and coding problems with Large Language Models (LLMs) have been enabled by investing substantial computation budgets at inference time. Therefore, inference speed is one of the most critical properties of LLM architectures, and there is a growing need for LLMs that are efficient and fast at inference. Recently, LLMs built on the xLSTM architecture have emerged as a powerful alternative to Transformers, offering linear compute scaling with sequence length and constant memory usage, both highly desirable properties for efficient inference. However, such xLSTM-based LLMs have yet to be scaled to larger models and assessed and compared with respect to inference speed and efficiency. In this work, we introduce xLSTM 7B, a 7-billion-parameter LLM that combines xLSTM's architectural benefits with targeted optimizations for fast and efficient inference. Our experiments demonstrate that xLSTM 7B achieves performance on downstream tasks comparable to other similar-sized LLMs, while providing significantly faster inference speeds and greater efficiency compared to Llama- and Mamba-based LLMs. These results establish xLSTM 7B as the fastest and most efficient 7B LLM, offering a solution for tasks that require large amounts of test-time computation. Our work highlights xLSTM's potential as a foundational architecture for methods building on heavy use of LLM inference. Our model weights, model code and training code are open-source.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13402v1">Toward Generative 6G Simulation: An Experimental Multi-Agent LLM and ns-3 Integration</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 6 pages, 4 figures, 4 tables
    </div>
    <details class="paper-abstract">
      The move toward open Sixth-Generation (6G) networks necessitates a novel approach to full-stack simulation environments for evaluating complex technology developments before prototyping and real-world implementation. This paper introduces an innovative approach\footnote{A lightweight, mock version of the code is available on GitHub at that combines a multi-agent framework with the Network Simulator 3 (ns-3) to automate and optimize the generation, debugging, execution, and analysis of complex 5G network scenarios. Our framework orchestrates a suite of specialized agents -- namely, the Simulation Generation Agent, Test Designer Agent, Test Executor Agent, and Result Interpretation Agent -- using advanced LangChain coordination. The Simulation Generation Agent employs a structured chain-of-thought (CoT) reasoning process, leveraging LLMs and retrieval-augmented generation (RAG) to translate natural language simulation specifications into precise ns-3 scripts. Concurrently, the Test Designer Agent generates comprehensive automated test suites by integrating knowledge retrieval techniques with dynamic test case synthesis. The Test Executor Agent dynamically deploys and runs simulations, managing dependencies and parsing detailed performance metrics. At the same time, the Result Interpretation Agent utilizes LLM-driven analysis to extract actionable insights from the simulation outputs. By integrating external resources such as library documentation and ns-3 testing frameworks, our experimental approach can enhance simulation accuracy and adaptability, reducing reliance on extensive programming expertise. A detailed case study using the ns-3 5G-LENA module validates the effectiveness of the proposed approach. The code generation process converges in an average of 1.8 iterations, has a syntax error rate of 17.0%, a mean response time of 7.3 seconds, and receives a human evaluation score of 7.5.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13340v1">LearnMate: Enhancing Online Education with LLM-Powered Personalized Learning Plans and Support</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 In Extended Abstracts of the CHI Conference on Human Factors in Computing Systems (CHI EA '25), April 26-May 1, 2025, Yokohama, Japan
    </div>
    <details class="paper-abstract">
      With the increasing prevalence of online learning, adapting education to diverse learner needs remains a persistent challenge. Recent advancements in artificial intelligence (AI), particularly large language models (LLMs), promise powerful tools and capabilities to enhance personalized learning in online educational environments. In this work, we explore how LLMs can improve personalized learning experiences by catering to individual user needs toward enhancing the overall quality of online education. We designed personalization guidelines based on the growing literature on personalized learning to ground LLMs in generating tailored learning plans. To operationalize these guidelines, we implemented LearnMate, an LLM-based system that generates personalized learning plans and provides users with real-time learning support. We discuss the implications and future directions of this work, aiming to move beyond the traditional one-size-fits-all approach by integrating LLM-based personalized support into online learning environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13330v1">LEAVS: An LLM-based Labeler for Abdominal CT Supervision</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Extracting structured labels from radiology reports has been employed to create vision models to simultaneously detect several types of abnormalities. However, existing works focus mainly on the chest region. Few works have been investigated on abdominal radiology reports due to more complex anatomy and a wider range of pathologies in the abdomen. We propose LEAVS (Large language model Extractor for Abdominal Vision Supervision). This labeler can annotate the certainty of presence and the urgency of seven types of abnormalities for nine abdominal organs on CT radiology reports. To ensure broad coverage, we chose abnormalities that encompass most of the finding types from CT reports. Our approach employs a specialized chain-of-thought prompting strategy for a locally-run LLM using sentence extraction and multiple-choice questions in a tree-based decision system. We demonstrate that the LLM can extract several abnormality types across abdominal organs with an average F1 score of 0.89, significantly outperforming competing labelers and humans. Additionally, we show that extraction of urgency labels achieved performance comparable to human annotations. Finally, we demonstrate that the abnormality labels contain valuable information for training a single vision model that classifies several organs as normal or abnormal. We release our code and structured annotations for a public CT dataset containing over 1,000 CT volumes.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13305v1">Computation Mechanism Behind LLM Position Generalization</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Most written natural languages are composed of sequences of words and sentences. Similar to humans, large language models (LLMs) exhibit flexibility in handling textual positions - a phenomenon we term position generalization. They can understand texts with position perturbations and generalize to longer texts than those encountered during training with the latest techniques. These phenomena suggest that LLMs handle positions tolerantly, but how LLMs computationally process positional relevance remains largely unexplored. This work connects the linguistic phenomenon with LLMs' computational mechanisms. We show how LLMs enforce certain computational mechanisms for the aforementioned tolerance in position perturbations. Despite the complex design of the self-attention mechanism, this work reveals that LLMs learn a counterintuitive disentanglement of attention logits. Their values show a 0.959 linear correlation with an approximation of the arithmetic sum of positional relevance and semantic importance. Furthermore, we identify a prevalent pattern in intermediate features, which we prove theoretically enables this effect. The pattern, which is different from how randomly initialized parameters would behave, suggests that it is a learned behavior rather than a natural result of the model architecture. Based on these findings, we provide computational explanations and criteria for LLMs' position flexibilities. This work takes a pioneering step in linking position generalization with modern LLMs' internal mechanisms.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13301v1">LIMCA: LLM for Automating Analog In-Memory Computing Architecture Design Exploration</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 4 Figures, 5 Tables
    </div>
    <details class="paper-abstract">
      Resistive crossbars enabling analog In-Memory Computing (IMC) have emerged as a promising architecture for Deep Neural Network (DNN) acceleration, offering high memory bandwidth and in-situ computation. However, the manual, knowledge-intensive design process and the lack of high-quality circuit netlists have significantly constrained design space exploration and optimization to behavioral system-level tools. In this work, we introduce LIMCA, a novel fine-tune-free Large Language Model (LLM)-driven framework for automating the design and evaluation of IMC crossbar architectures. Unlike traditional approaches, LIMCA employs a No-Human-In-Loop (NHIL) automated pipeline to generate and validate circuit netlists for SPICE simulations, eliminating manual intervention. LIMCA systematically explores the IMC design space by leveraging a structured dataset and LLM-based performance evaluation. Our experimental results on MNIST classification demonstrate that LIMCA successfully generates crossbar designs achieving $\geq$96% accuracy while maintaining a power consumption $\leq$3W, making this the first work in LLM-assisted IMC design space exploration. Compared to existing frameworks, LIMCA provides an automated, scalable, and hardware-aware solution, reducing design exploration time while ensuring user-constrained performance trade-offs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03834v2">GraphRouter: A Graph-based Router for LLM Selections</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      The rapidly growing number and variety of Large Language Models (LLMs) present significant challenges in efficiently selecting the appropriate LLM for a given query, especially considering the trade-offs between performance and computational cost. Current LLM selection methods often struggle to generalize across new LLMs and different tasks because of their limited ability to leverage contextual interactions among tasks, queries, and LLMs, as well as their dependence on a transductive learning framework. To address these shortcomings, we introduce a novel inductive graph framework, named as GraphRouter, which fully utilizes the contextual information among tasks, queries, and LLMs to enhance the LLM selection process. GraphRouter constructs a heterogeneous graph comprising task, query, and LLM nodes, with interactions represented as edges, which efficiently captures the contextual information between the query's requirements and the LLM's capabilities. Through an innovative edge prediction mechanism, GraphRouter is able to predict attributes (the effect and cost of LLM response) of potential edges, allowing for optimized recommendations that adapt to both existing and newly introduced LLMs without requiring retraining. Comprehensive experiments across three distinct effect-cost weight scenarios have shown that GraphRouter substantially surpasses existing routers, delivering a minimum performance improvement of 12.3%. In addition, it achieves enhanced generalization across new LLMs settings and supports diverse tasks with at least a 9.5% boost in effect and a significant reduction in computational demands. This work endeavors to apply a graph-based approach for the contextual and adaptive selection of LLMs, offering insights for real-world applications. Our codes for GraphRouter is released at https://github.com/ulab-uiuc/GraphRouter.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13250v1">MindEye-OmniAssist: A Gaze-Driven LLM-Enhanced Assistive Robot System for Implicit Intention Recognition and Task Execution</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      A promising effective human-robot interaction in assistive robotic systems is gaze-based control. However, current gaze-based assistive systems mainly help users with basic grasping actions, offering limited support. Moreover, the restricted intent recognition capability constrains the assistive system's ability to provide diverse assistance functions. In this paper, we propose an open implicit intention recognition framework powered by Large Language Model (LLM) and Vision Foundation Model (VFM), which can process gaze input and recognize user intents that are not confined to predefined or specific scenarios. Furthermore, we implement a gaze-driven LLM-enhanced assistive robot system (MindEye-OmniAssist) that recognizes user's intentions through gaze and assists in completing task. To achieve this, the system utilizes open vocabulary object detector, intention recognition network and LLM to infer their full intentions. By integrating eye movement feedback and LLM, it generates action sequences to assist the user in completing tasks. Real-world experiments have been conducted for assistive tasks, and the system achieved an overall success rate of 41/55 across various undefined tasks. Preliminary results show that the proposed method holds the potential to provide a more user-friendly human-computer interaction interface and significantly enhance the versatility and effectiveness of assistive systems by supporting more complex and diverse task.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.14892v2">Causal Graphs Meet Thoughts: Enhancing Complex Reasoning in Graph-Augmented LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 18 pages, 3 figures, 3 tables
    </div>
    <details class="paper-abstract">
      In knowledge-intensive tasks, especially in high-stakes domains like medicine and law, it is critical not only to retrieve relevant information but also to provide causal reasoning and explainability. Large language models (LLMs) have achieved remarkable performance in natural language understanding and generation tasks. However, they often suffer from limitations such as difficulty in incorporating new knowledge, generating hallucinations, and explaining their reasoning process. To address these challenges, integrating knowledge graphs with Graph Retrieval-Augmented Generation (Graph RAG) has emerged as an effective solution. Traditional Graph RAG methods often rely on simple graph traversal or semantic similarity, which do not capture causal relationships or align well with the model's internal reasoning steps. This paper proposes a novel pipeline that filters large knowledge graphs to emphasize cause-effect edges, aligns the retrieval process with the model's chain-of-thought (CoT), and enhances reasoning through multi-stage path improvements. Experiments on medical question-answering tasks show consistent gains, with up to a 10\% absolute improvement across multiple large language models (LLMs). This approach demonstrates the value of combining causal reasoning with stepwise retrieval, leading to more interpretable and logically grounded solutions for complex queries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.21016v2">Assessing the Robustness of LLM-based NLP Software via Automated Testing</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Benefiting from the advancements in LLMs, NLP software has undergone rapid development. Such software is widely employed in various safety-critical tasks, such as financial sentiment analysis, toxic content moderation, and log generation. Unlike traditional software, LLM-based NLP software relies on prompts and examples as inputs. Given the complexity of LLMs and the unpredictability of real-world inputs, quantitatively assessing the robustness of such software is crucial. However, to the best of our knowledge, no automated robustness testing methods have been specifically designed to evaluate the overall inputs of LLM-based NLP software. To this end, this paper introduces the first AutOmated Robustness Testing frAmework, AORTA, which reconceptualizes the testing process into a combinatorial optimization problem. Existing testing methods designed for DNN-based software can be applied to LLM-based software by AORTA, but their effectiveness is limited. To address this, we propose a novel testing method for LLM-based software within AORTA called Adaptive Beam Search. ABS is tailored for the expansive feature space of LLMs and improves testing effectiveness through an adaptive beam width and the capability for backtracking. We successfully embed 18 test methods in the designed framework AORTA and compared the test validity of ABS with three datasets and five threat models. ABS facilitates a more comprehensive and accurate robustness assessment before software deployment, with an average test success rate of 86.138%. Compared to the currently best-performing baseline PWWS, ABS significantly reduces the computational overhead by up to 3441.895 seconds per successful test case and decreases the number of queries by 218.762 times on average. Furthermore, test cases generated by ABS exhibit greater naturalness and transferability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.04927v3">LLM-based speaker diarization correction: A generalizable approach</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Speaker diarization is necessary for interpreting conversations transcribed using automated speech recognition (ASR) tools. Despite significant developments in diarization methods, diarization accuracy remains an issue. Here, we investigate the use of large language models (LLMs) for diarization correction as a post-processing step. LLMs were fine-tuned using the Fisher corpus, a large dataset of transcribed conversations. The ability of the models to improve diarization accuracy in a holdout dataset from the Fisher corpus as well as an independent dataset was measured. We report that fine-tuned LLMs can markedly improve diarization accuracy. However, model performance is constrained to transcripts produced using the same ASR tool as the transcripts used for fine-tuning, limiting generalizability. To address this constraint, an ensemble model was developed by combining weights from three separate models, each fine-tuned using transcripts from a different ASR tool. The ensemble model demonstrated better overall performance than each of the ASR-specific models, suggesting that a generalizable and ASR-agnostic approach may be achievable. We have made the weights of these models publicly available on HuggingFace at https://huggingface.co/bklynhlth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13149v1">Are LLMs (Really) Ideological? An IRT-based Analysis and Alignment Tool for Perceived Socio-Economic Bias in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      We introduce an Item Response Theory (IRT)-based framework to detect and quantify socioeconomic bias in large language models (LLMs) without relying on subjective human judgments. Unlike traditional methods, IRT accounts for item difficulty, improving ideological bias estimation. We fine-tune two LLM families (Meta-LLaMa 3.2-1B-Instruct and Chat- GPT 3.5) to represent distinct ideological positions and introduce a two-stage approach: (1) modeling response avoidance and (2) estimating perceived bias in answered responses. Our results show that off-the-shelf LLMs often avoid ideological engagement rather than exhibit bias, challenging prior claims of partisanship. This empirically validated framework enhances AI alignment research and promotes fairer AI governance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13116v1">VeriLeaky: Navigating IP Protection vs Utility in Fine-Tuning for LLM-Driven Verilog Coding</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) offer significant potential for coding, yet fine-tuning (FT) with curated data is essential for niche languages like Verilog. Using proprietary intellectual property (IP) for FT presents a serious risk, as FT data can be leaked through LLM inference. This leads to a critical dilemma for design houses: seeking to build externally accessible LLMs offering competitive Verilog coding, how can they leverage in-house IP to enhance FT utility while ensuring IP protection? For the first time in the literature, we study this dilemma. Using LLaMA 3.1-8B, we conduct in-house FT on a baseline Verilog dataset (RTLCoder) supplemented with our own in-house IP, which is validated through multiple tape-outs. To rigorously assess IP leakage, we quantify structural similarity (AST/Dolos) and functional equivalence (Synopsys Formality) between generated codes and our in-house IP. We show that our IP can indeed be leaked, confirming the threat. As defense, we evaluate logic locking of Verilog codes (ASSURE). This offers some level of protection, yet reduces the IP's utility for FT and degrades the LLM's performance. Our study shows the need for novel strategies that are both effective and minimally disruptive to FT, an essential effort for enabling design houses to fully utilize their proprietary IP toward LLM-driven Verilog coding.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13081v1">A Framework to Assess Multilingual Vulnerabilities of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are acquiring a wider range of capabilities, including understanding and responding in multiple languages. While they undergo safety training to prevent them from answering illegal questions, imbalances in training data and human evaluation resources can make these models more susceptible to attacks in low-resource languages (LRL). This paper proposes a framework to automatically assess the multilingual vulnerabilities of commonly used LLMs. Using our framework, we evaluated six LLMs across eight languages representing varying levels of resource availability. We validated the assessments generated by our automated framework through human evaluation in two languages, demonstrating that the framework's results align with human judgments in most cases. Our findings reveal vulnerabilities in LRL; however, these may pose minimal risk as they often stem from the model's poor performance, resulting in incoherent responses.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12782v2">In-Context Learning Enables Robot Action Prediction in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 Published in ICRA 2025
    </div>
    <details class="paper-abstract">
      Recently, Large Language Models (LLMs) have achieved remarkable success using in-context learning (ICL) in the language domain. However, leveraging the ICL capabilities within LLMs to directly predict robot actions remains largely unexplored. In this paper, we introduce RoboPrompt, a framework that enables off-the-shelf text-only LLMs to directly predict robot actions through ICL without training. Our approach first heuristically identifies keyframes that capture important moments from an episode. Next, we extract end-effector actions from these keyframes as well as the estimated initial object poses, and both are converted into textual descriptions. Finally, we construct a structured template to form ICL demonstrations from these textual descriptions and a task instruction. This enables an LLM to directly predict robot actions at test time. Through extensive experiments and analysis, RoboPrompt shows stronger performance over zero-shot and ICL baselines in simulated and real-world settings. Our project page is available at https://davidyyd.github.io/roboprompt.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13038v1">Overview of the NTCIR-18 Automatic Evaluation of LLMs (AEOLLM) Task</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      In this paper, we provide an overview of the NTCIR-18 Automatic Evaluation of LLMs (AEOLLM) task. As large language models (LLMs) grow popular in both academia and industry, how to effectively evaluate the capacity of LLMs becomes an increasingly critical but still challenging issue. Existing methods can be divided into two types: manual evaluation, which is expensive, and automatic evaluation, which faces many limitations including task format (the majority belong to multiple-choice questions) and evaluation criteria (occupied by reference-based metrics). To advance the innovation of automatic evaluation, we propose the AEOLLM task which focuses on generative tasks and encourages reference-free methods. Besides, we set up diverse subtasks such as dialogue generation, text expansion, summary generation and non-factoid question answering to comprehensively test different methods. This year, we received 48 runs from 4 teams in total. This paper will describe the background of the task, the data set, the evaluation measures and the evaluation results, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.19597v3">TuBA: Cross-Lingual Transferability of Backdoor Attacks in LLMs with Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 work in progress
    </div>
    <details class="paper-abstract">
      The implications of backdoor attacks on English-centric large language models (LLMs) have been widely examined - such attacks can be achieved by embedding malicious behaviors during training and activated under specific conditions that trigger malicious outputs. Despite the increasing support for multilingual capabilities in open-source and proprietary LLMs, the impact of backdoor attacks on these systems remains largely under-explored. Our research focuses on cross-lingual backdoor attacks against multilingual LLMs, particularly investigating how poisoning the instruction-tuning data for one or two languages can affect the outputs for languages whose instruction-tuning data were not poisoned. Despite its simplicity, our empirical analysis reveals that our method exhibits remarkable efficacy in models like mT5 and GPT-4o, with high attack success rates, surpassing 90% in more than 7 out of 12 languages across various scenarios. Our findings also indicate that more powerful models show increased susceptibility to transferable cross-lingual backdoor attacks, which also applies to LLMs predominantly pre-trained on English data, such as Llama2, Llama3, and Gemma. Moreover, our experiments demonstrate 1) High Transferability: the backdoor mechanism operates successfully in cross-lingual response scenarios across 26 languages, achieving an average attack success rate of 99%, and 2) Robustness: the proposed attack remains effective even after defenses are applied. These findings expose critical security vulnerabilities in multilingual LLMs and highlight the urgent need for more robust, targeted defense strategies to address the unique challenges posed by cross-lingual backdoor transfer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12988v1">ROMA: a Read-Only-Memory-based Accelerator for QLoRA-based On-Device LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) demonstrate powerful capabilities, deploying them on edge devices has become increasingly crucial, offering advantages in privacy and real-time interaction. QLoRA has emerged as the standard approach for on-device LLMs, leveraging quantized models to reduce memory and computational costs while utilizing LoRA for task-specific adaptability. In this work, we propose ROMA, a QLoRA accelerator with a hybrid storage architecture that uses ROM for quantized base models and SRAM for LoRA weights and KV cache. Our insight is that the quantized base model is stable and converged, making it well-suited for ROM storage. Meanwhile, LoRA modules offer the flexibility to adapt to new data without requiring updates to the base model. To further reduce the area cost of ROM, we introduce a novel B-ROM design and integrate it with the compute unit to form a fused cell for efficient use of chip resources. ROMA can effectively store both a 4-bit 3B and a 2-bit 8B LLaMA model entirely on-chip, achieving a notable generation speed exceeding 20,000 tokens/s without requiring external memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12972v1">Aligning Vision to Language: Text-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 14 pages, 7 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Multimodal reasoning in Large Language Models (LLMs) struggles with incomplete knowledge and hallucination artifacts, challenges that textual Knowledge Graphs (KGs) only partially mitigate due to their modality isolation. While Multimodal Knowledge Graphs (MMKGs) promise enhanced cross-modal understanding, their practical construction is impeded by semantic narrowness of manual text annotations and inherent noise in visual-semantic entity linkages. In this paper, we propose Vision-align-to-Language integrated Knowledge Graph (VaLiK), a novel approach for constructing MMKGs that enhances LLMs reasoning through cross-modal information supplementation. Specifically, we cascade pre-trained Vision-Language Models (VLMs) to align image features with text, transforming them into descriptions that encapsulate image-specific information. Furthermore, we developed a cross-modal similarity verification mechanism to quantify semantic consistency, effectively filtering out noise introduced during feature alignment. Even without manually annotated image captions, the refined descriptions alone suffice to construct the MMKG. Compared to conventional MMKGs construction paradigms, our approach achieves substantial storage efficiency gains while maintaining direct entity-to-image linkage capability. Experimental results on multimodal reasoning tasks demonstrate that LLMs augmented with VaLiK outperform previous state-of-the-art models. Our code is published at https://github.com/Wings-Of-Disaster/VaLiK.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19951v4">Sparrow: Data-Efficient Video-LLM with Text-to-Image Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 Project page: https://github.com/VITA-MLLM/Sparrow
    </div>
    <details class="paper-abstract">
      Recent years have witnessed the success of Multimodal Large Language Models (MLLMs) in the vision understanding domain. The success of these models can largely be attributed to the dominant scaling law, which states that larger parameter sizes and data volumes contribute to better performance. Notably, data scaling has mainly been powered by automatic data pipelines, which center around the self-instruction of LLMs. The paradigm has been taken for granted for quite some time, but the study of the effectiveness of scaling with these data has been neglected for a long time. In this context, this work revisits scaling with synthetic data and focuses on developing video-LLMs from a data-centric perspective. Our main study approach is fine-tuning pre-trained image-LLMs with video data and investigating learning efficiency through data scaling. Results from our preliminary experiments reveal a low learning efficiency phenomenon when simply scaling up video data samples, which, through our probing, can be ascribed to a lack of instruction diversity. Aiming at this issue, we propose a data augmentation method called Sparrow, which synthesizes video-like samples from pure text instruction data. Mixing these synthetic samples with the video data enables a more efficient training scheme. Through comprehensive experiments, we demonstrate that our proposed method achieves performance comparable to or even superior to baselines trained with many more samples. Meanwhile, we find that incorporating these synthetic samples can boost the performance of long video understanding without training with long video data. The code and data examples are available at https://github.com/VITA-MLLM/Sparrow.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12918v1">ThinkPatterns-21k: A Systematic Study on the Impact of Thinking Patterns in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated enhanced performance through the \textit{Thinking then Responding} paradigm, where models generate internal thoughts before final responses (aka, System 2 thinking). However, existing research lacks a systematic understanding of the mechanisms underlying how thinking patterns affect performance across model sizes. In this work, we conduct a comprehensive analysis of the impact of various thinking types on model performance and introduce ThinkPatterns-21k, a curated dataset comprising 21k instruction-response pairs (QA) collected from existing instruction-following datasets with five thinking types. For each pair, we augment it with five distinct internal thinking patterns: one unstructured thinking (monologue) and four structured variants (decomposition, self-ask, self-debate and self-critic), while maintaining the same instruction and response. Through extensive evaluation across different model sizes (3B-32B parameters), we have two key findings: (1) smaller models (<30B parameters) can benefit from most of structured thinking patterns, while larger models (32B) with structured thinking like decomposition would degrade performance and (2) unstructured monologue demonstrates broad effectiveness across different model sizes. Finally, we released all of our datasets, checkpoints, training logs of diverse thinking patterns to reproducibility, aiming to facilitate further research in this direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12899v1">A Semantic-based Optimization Approach for Repairing LLMs: Case Study on Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 12 pages, 6 figure, 6 tables, under peer-review
    </div>
    <details class="paper-abstract">
      Language Models (LMs) are widely used in software engineering for code generation, but they may produce code with errors. Rather than repairing the generated code, an alternative way is to address the underlying failures of models. LM repair offers a lightweight solution to this challenge: it requires minimal data, reduces computational costs, and reduces the side effects. Unlike retraining, LM repair focuses on applying tailored updates to targeted neurons, making it ideal for scenarios with limited resources, high-performance demands, or strict safety requirements. In this paper, we propose \ul{S}emantic \ul{T}argeting for \ul{A}nalytical \ul{R}epair (\textsc{STAR}), a pioneering and novel semantic-based optimization approach for repairing LLMs. \textsc{STAR} realizes main operations in LM repair methods in an optimization process, including locating ``buggy neurons'', solving ``neuron patches'', and patching ``buggy neurons''. Correspondingly, it computes the deltas of weight matrix as the prior information to guide optimization; and attributes the targeted layers and neurons leveraging statistical insights. The neuron patches are computed with a solid semantic-based analytical formula, which directly bridges the changes to logits with the deltas of neurons, by steering latent representations. Compared to the prior work of LM repair (\textsc{MINT}) and optimization methods (\textsc{SGD}), \textsc{STAR} integrates their strengths while mitigating their limitations. \textsc{STAR} supports solving multiple failures together, significantly improving the usefulness. Evaluated on three code generation tasks using popular code LMs, \textsc{STAR} demonstrates superior effectiveness. Additionally, \textsc{STAR} exhibits better efficiency. In terms of side effects, namely the balance between generalization and specificity, \textsc{STAR} outperforms prior work by a significant margin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13356v4">Unlearning or Obfuscating? Jogging the Memory of Unlearned LLMs via Benign Relearning</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 ICLR 2025, 32 pages, 8 figures, 9 tables
    </div>
    <details class="paper-abstract">
      Machine unlearning is a promising approach to mitigate undesirable memorization of training data in ML models. However, in this work we show that existing approaches for unlearning in LLMs are surprisingly susceptible to a simple set of $\textit{benign relearning attacks}$. With access to only a small and potentially loosely related set of data, we find that we can ''jog'' the memory of unlearned models to reverse the effects of unlearning. For example, we show that relearning on public medical articles can lead an unlearned LLM to output harmful knowledge about bioweapons, and relearning general wiki information about the book series Harry Potter can force the model to output verbatim memorized text. We formalize this unlearning-relearning pipeline, explore the attack across three popular unlearning benchmarks, and discuss future directions and guidelines that result from our study. Our work indicates that current approximate unlearning methods simply suppress the model outputs and fail to robustly forget target knowledge in the LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01090v2">Precise Localization of Memories: A Fine-grained Neuron-level Knowledge Editing Technique for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Knowledge editing aims to update outdated information in Large Language Models (LLMs). A representative line of study is locate-then-edit methods, which typically employ causal tracing to identify the modules responsible for recalling factual knowledge about entities. However, we find these methods are often sensitive only to changes in the subject entity, leaving them less effective at adapting to changes in relations. This limitation results in poor editing locality, which can lead to the persistence of irrelevant or inaccurate facts, ultimately compromising the reliability of LLMs. We believe this issue arises from the insufficient precision of knowledge localization. To address this, we propose a Fine-grained Neuron-level Knowledge Editing (FiNE) method that enhances editing locality without affecting overall success rates. By precisely identifying and modifying specific neurons within feed-forward networks, FiNE significantly improves knowledge localization and editing. Quantitative experiments demonstrate that FiNE efficiently achieves better overall performance compared to existing techniques, providing new insights into the localization and modification of knowledge within LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12854v1">Enhancing LLM Reasoning with Iterative DPO: A Comprehensive Empirical Investigation</a></div>
    <div class="paper-meta">
      📅 2025-03-17
    </div>
    <details class="paper-abstract">
      Recent advancements in post-training methodologies for large language models (LLMs) have highlighted reinforcement learning (RL) as a critical component for enhancing reasoning. However, the substantial computational costs associated with RL-based approaches have led to growing interest in alternative paradigms, such as Direct Preference Optimization (DPO). In this study, we investigate the effectiveness of DPO in facilitating self-improvement for LLMs through iterative preference-based learning. We demonstrate that a single round of DPO with coarse filtering significantly enhances mathematical reasoning performance, particularly for strong base model. Furthermore, we design an iterative enhancement framework for both the generator and the reward model (RM), enabling their mutual improvement through online interaction across multiple rounds of DPO. Finally, with simple verifiable rewards, our model DPO-VP achieves RL-level performance with significantly lower computational overhead. These findings highlight DPO as a scalable and cost-effective alternative to RL, offering a practical solution for enhancing LLM reasoning in resource-constrained situations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18447v2">ToolFlow: Boosting LLM Tool-Calling Through Natural and Coherent Dialogue Synthesis</a></div>
    <div class="paper-meta">
      📅 2025-03-17
      | 💬 Accepted by NAACL 2025
    </div>
    <details class="paper-abstract">
      Supervised fine-tuning (SFT) is a common method to enhance the tool calling capabilities of Large Language Models (LLMs), with the training data often being synthesized. The current data synthesis process generally involves sampling a set of tools, formulating a requirement based on these tools, and generating the call statements. However, tools sampled randomly lack relevance, making them difficult to combine and thus reducing the diversity of the data. Additionally, current work overlooks the coherence between turns of dialogues, leading to a gap between the synthesized data and real-world scenarios. To address these issues, we propose a Graph-based Sampling strategy to sample more relevant tool combinations, and a Planned-generation strategy to create plans that guide the synthesis of coherent dialogues. We integrate these two strategies and enable multiple agents to synthesize the dialogue data interactively, resulting in our tool-calling data synthesis pipeline ToolFlow. Data quality assessments demonstrate improvements in the naturalness and coherence of our synthesized dialogues. Finally, we apply SFT on LLaMA-3.1-8B using 8,000 synthetic dialogues generated with ToolFlow. Results show that the model achieves tool-calling performance comparable to or even surpassing GPT-4, while maintaining strong general capabilities.
    </details>
</div>
