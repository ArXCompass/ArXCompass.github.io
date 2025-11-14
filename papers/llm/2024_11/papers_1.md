# llm - 2024_11

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- Part 1
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.05804v6">A Review of Prominent Paradigms for LLM-Based Agents: Tool Use (Including RAG), Planning, and Feedback Learning</a></div>
    <div class="paper-meta">
      📅 2024-11-30
      | 💬 CoLing 2025 Camera Ready (extended to 9 pages)
    </div>
    <details class="paper-abstract">
      Tool use, planning, and feedback learning are currently three prominent paradigms for developing Large Language Model (LLM)-based agents across various tasks. Although numerous frameworks have been devised for each paradigm, their intricate workflows and inconsistent taxonomy create challenges in understanding and reviewing the frameworks across different paradigms. This survey introduces a unified taxonomy to systematically review and discuss these frameworks. Specifically, 1) the taxonomy defines environments/tasks, common LLM-profiled roles or LMPRs (policy models, evaluators, and dynamic models), and universally applicable workflows found in prior work, and 2) it enables a comparison of key perspectives on the implementations of LMPRs and workflow designs across different agent paradigms and frameworks. 3) Finally, we identify three limitations in existing workflow designs and systematically discuss the future work. Resources have been made publicly available at in our GitHub repository https://github.com/xinzhel/LLM-Agent-Survey.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02532v3">SpecExec: Massively Parallel Speculative Decoding for Interactive LLM Inference on Consumer Devices</a></div>
    <div class="paper-meta">
      📅 2024-11-30
    </div>
    <details class="paper-abstract">
      As large language models gain widespread adoption, running them efficiently becomes crucial. Recent works on LLM inference use speculative decoding to achieve extreme speedups. However, most of these works implicitly design their algorithms for high-end datacenter hardware. In this work, we ask the opposite question: how fast can we run LLMs on consumer machines? Consumer GPUs can no longer fit the largest available models (50B+ parameters) and must offload them to RAM or SSD. When running with offloaded parameters, the inference engine can process batches of hundreds or thousands of tokens at the same time as just one token, making it a natural fit for speculative decoding. We propose SpecExec (Speculative Execution), a simple parallel decoding method that can generate up to 20 tokens per target model iteration for popular LLM families. It utilizes the high spikiness of the token probabilities distribution in modern LLMs and a high degree of alignment between model output probabilities. SpecExec takes the most probable tokens continuation from the draft model to build a "cache" tree for the target model, which then gets validated in a single pass. Using SpecExec, we demonstrate inference of 50B+ parameter LLMs on consumer GPUs with RAM offloading at 4-6 tokens per second with 4-bit quantization or 2-3 tokens per second with 16-bit weights.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00546v1">Rank It, Then Ask It: Input Reranking for Maximizing the Performance of LLMs on Symmetric Tasks</a></div>
    <div class="paper-meta">
      📅 2024-11-30
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have quickly emerged as practical and versatile tools that provide new solutions for a wide range of domains. In this paper, we consider the application of LLMs on symmetric tasks where a query is asked on an (unordered) bag of elements. Examples of such tasks include answering aggregate queries on a database table. In general, when the bag contains a large number of elements, LLMs tend to overlook some elements, leading to challenges in generating accurate responses to the query. LLMs receive their inputs as ordered sequences. However, in this problem, we leverage the fact that the symmetric input is not ordered, and reordering should not affect the LLM's response. Observing that LLMs are less likely to miss elements at certain positions of the input, we introduce the problem of LLM input reranking: to find a ranking of the input that maximizes the LLM's accuracy for the given query without making explicit assumptions about the query. Finding the optimal ranking requires identifying (i) the relevance of each input element for answering the query and (ii) the importance of each rank position for the LLM's attention. We develop algorithms for estimating these values efficiently utilizing a helper LLM. We conduct comprehensive experiments on different synthetic and real datasets to validate our proposal and to evaluate the effectiveness of our proposed algorithms. Our experiments confirm that our reranking approach improves the accuracy of the LLMs on symmetric tasks by up to $99\%$ proximity to the optimum upper bound.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00543v1">Evaluating the Consistency of LLM Evaluators</a></div>
    <div class="paper-meta">
      📅 2024-11-30
      | 💬 Accepted to COLING 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown potential as general evaluators along with the evident benefits of speed and cost. While their correlation against human annotators has been widely studied, consistency as evaluators is still understudied, raising concerns about the reliability of LLM evaluators. In this paper, we conduct extensive studies on the two aspects of consistency in LLM evaluations, Self-Consistency (SC) and Inter-scale Consistency (IC), on different scoring scales and criterion granularity with open-source and proprietary models. Our comprehensive analysis demonstrates that strong proprietary models are not necessarily consistent evaluators, highlighting the importance of considering consistency in assessing the capability of LLM evaluators.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00493v1">Video-3D LLM: Learning Position-Aware Video Representation for 3D Scene Understanding</a></div>
    <div class="paper-meta">
      📅 2024-11-30
      | 💬 14 pages, 4 figures
    </div>
    <details class="paper-abstract">
      The rapid advancement of Multimodal Large Language Models (MLLMs) has significantly impacted various multimodal tasks. However, these models face challenges in tasks that require spatial understanding within 3D environments. Efforts to enhance MLLMs, such as incorporating point cloud features, have been made, yet a considerable gap remains between the models' learned representations and the inherent complexity of 3D scenes. This discrepancy largely stems from the training of MLLMs on predominantly 2D data, which restricts their effectiveness in comprehending 3D spaces. To address this issue, in this paper, we propose a novel generalist model, i.e., Video-3D LLM, for 3D scene understanding. By treating 3D scenes as dynamic videos and incorporating 3D position encoding into these representations, our Video-3D LLM aligns video representations with real-world spatial contexts more accurately. Additionally, we have implemented a maximum coverage sampling technique to optimize the balance between computational costs and performance efficiency. Extensive experiments demonstrate that our model achieves state-of-the-art performance on several 3D scene understanding benchmarks, including ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, and SQA3D.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00478v1">Node Importance Estimation Leveraging LLMs for Semantic Augmentation in Knowledge Graphs</a></div>
    <div class="paper-meta">
      📅 2024-11-30
      | 💬 13 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Node Importance Estimation (NIE) is a task that quantifies the importance of node in a graph. Recent research has investigated to exploit various information from Knowledge Graphs (KGs) to estimate node importance scores. However, the semantic information in KGs could be insufficient, missing, and inaccurate, which would limit the performance of existing NIE models. To address these issues, we leverage Large Language Models (LLMs) for semantic augmentation thanks to the LLMs' extra knowledge and ability of integrating knowledge from both LLMs and KGs. To this end, we propose the LLMs Empowered Node Importance Estimation (LENIE) method to enhance the semantic information in KGs for better supporting NIE tasks. To our best knowledge, this is the first work incorporating LLMs into NIE. Specifically, LENIE employs a novel clustering-based triplet sampling strategy to extract diverse knowledge of a node sampled from the given KG. After that, LENIE adopts the node-specific adaptive prompts to integrate the sampled triplets and the original node descriptions, which are then fed into LLMs for generating richer and more precise augmented node descriptions. These augmented descriptions finally initialize node embeddings for boosting the downstream NIE model performance. Extensive experiments demonstrate LENIE's effectiveness in addressing semantic deficiencies in KGs, enabling more informative semantic augmentation and enhancing existing NIE models to achieve the state-of-the-art performance. The source code of LENIE is freely available at \url{https://github.com/XinyuLin-FZ/LENIE}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00383v1">Unified Parameter-Efficient Unlearning for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-30
    </div>
    <details class="paper-abstract">
      The advent of Large Language Models (LLMs) has revolutionized natural language processing, enabling advanced understanding and reasoning capabilities across a variety of tasks. Fine-tuning these models for specific domains, particularly through Parameter-Efficient Fine-Tuning (PEFT) strategies like LoRA, has become a prevalent practice due to its efficiency. However, this raises significant privacy and security concerns, as models may inadvertently retain and disseminate sensitive or undesirable information. To address these issues, we introduce a novel instance-wise unlearning framework, LLMEraser, which systematically categorizes unlearning tasks and applies precise parameter adjustments using influence functions. Unlike traditional unlearning techniques that are often limited in scope and require extensive retraining, LLMEraser is designed to handle a broad spectrum of unlearning tasks without compromising model performance. Extensive experiments on benchmark datasets demonstrate that LLMEraser excels in efficiently managing various unlearning scenarios while maintaining the overall integrity and efficacy of the models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2310.18964v3">LLMs and Finetuning: Benchmarking cross-domain performance for hate speech detection</a></div>
    <div class="paper-meta">
      📅 2024-11-30
      | 💬 10 pages, 3 figures, 5 tables
    </div>
    <details class="paper-abstract">
      In the evolving landscape of online communication, hate speech detection remains a formidable challenge, further compounded by the diversity of digital platforms. This study investigates the effectiveness and adaptability of pre-trained and fine-tuned Large Language Models (LLMs) in identifying hate speech, to address two central questions: (1) To what extent does the model performance depend on the fine-tuning and training parameters?, (2) To what extent do models generalize to cross-domain hate speech detection? and (3) What are the specific features of the datasets or models that influence the generalization potential? The experiment shows that LLMs offer a huge advantage over the state-of-the-art even without pretraining. Ordinary least squares analyses suggest that the advantage of training with fine-grained hate speech labels is washed away with the increase in dataset size. We conclude with a vision for the future of hate speech detection, emphasizing cross-domain generalizability and appropriate benchmarking practices.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.05285v2">AgentOps: Enabling Observability of LLM Agents</a></div>
    <div class="paper-meta">
      📅 2024-11-30
      | 💬 12 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large language model (LLM) agents have demonstrated remarkable capabilities across various domains, gaining extensive attention from academia and industry. However, these agents raise significant concerns on AI safety due to their autonomous and non-deterministic behavior, as well as continuous evolving nature . From a DevOps perspective, enabling observability in agents is necessary to ensuring AI safety, as stakeholders can gain insights into the agents' inner workings, allowing them to proactively understand the agents, detect anomalies, and prevent potential failures. Therefore, in this paper, we present a comprehensive taxonomy of AgentOps, identifying the artifacts and associated data that should be traced throughout the entire lifecycle of agents to achieve effective observability. The taxonomy is developed based on a systematic mapping study of existing AgentOps tools. Our taxonomy serves as a reference template for developers to design and implement AgentOps infrastructure that supports monitoring, logging, and analytics. thereby ensuring AI safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00314v1">Human-Like Code Quality Evaluation through LLM-based Recursive Semantic Comprehension</a></div>
    <div class="paper-meta">
      📅 2024-11-30
    </div>
    <details class="paper-abstract">
      Code quality evaluation involves scoring generated code quality based on a reference code for a specific problem statement. Currently, there are two main forms of evaluating code quality: match-based evaluation and execution-based evaluation. The former requires the collection of a large number of test cases, making a huge cost. The latter relies on superficial code matching as an evaluation metric, which fails to accurately capture code semantics. Moreover, extensive research has demonstrated that match-based evaluations do not truly reflect code quality. With the development of large language models (LLMs) in recent years, studies have proven the feasibility of using LLMs as evaluators for generative tasks. However, due to issues like hallucinations and uncertainty in LLMs, their correlation with human judgment remains at a lower level, making the direct use of LLMs for code quality evaluation challenging. To address these issues, we propose Human-Like Code Quality Evaluation through LLM-based Recursive Semantic Comprehension (HuCoSC). We employ a recursive approach to enable LLMs to comprehend portions of code semantics independently each time, obtaining the code semantics through multiple interactions with LLMs. We designed a Semantic Dependency Decoupling Storage to make independent analysis feasible, allowing LLMs to achieve more accurate semantics by breaking down complex problems. Finally, the generated code is scored based on a semantic comparison between the reference code and itself. Experimental results indicate that HuCoSC surpasses existing state-of-the-art methods in terms of correlation with human experts and correlation with code execution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00216v1">Enhanced LLM-Based Framework for Predicting Null Pointer Dereference in Source Code</a></div>
    <div class="paper-meta">
      📅 2024-11-29
    </div>
    <details class="paper-abstract">
      Software security is crucial in any field where breaches can exploit sensitive data, and lead to financial losses. As a result, vulnerability detection becomes an essential part of the software development process. One of the key steps in maintaining software integrity is identifying vulnerabilities in the source code before deployment. A security breach like CWE-476, which stands for NULL pointer dereferences (NPD), is crucial because it can cause software crashes, unpredictable behavior, and security vulnerabilities. In this scientific era, there are several vulnerability checkers, where, previous tools often fall short in analyzing specific feature connections of the source code, which weakens the tools in real-world scenarios. In this study, we propose another novel approach using a fine-tuned Large Language Model (LLM) termed "DeLLNeuN". This model leverages the advantage of various layers to reduce both overfitting and non-linearity, enhancing its performance and reliability. Additionally, this method provides dropout and dimensionality reduction to help streamline the model, making it faster and more efficient. Our model showed 87% accuracy with 88% precision using the Draper VDISC dataset. As software becomes more complex and cyber threats continuously evolve, the need for proactive security measures will keep growing. In this particular case, the proposed model looks promising to use as an early vulnerability checker in software development.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00207v1">Can LLM "Self-report"?: Evaluating the Validity of Self-report Scales in Measuring Personality Design in LLM-based Chatbots</a></div>
    <div class="paper-meta">
      📅 2024-11-29
      | 💬 14 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Personality design plays an important role in chatbot development. From rule-based chatbots to LLM-based chatbots, evaluating the effectiveness of personality design has become more challenging due to the increasingly open-ended interactions. A recent popular approach uses self-report questionnaires to assess LLM-based chatbots' personality traits. However, such an approach has raised serious validity concerns: chatbot's "self-report" personality may not align with human perception based on their interaction. Can LLM-based chatbots "self-report" their personality? We created 500 chatbots with distinct personality designs and evaluated the validity of self-reported personality scales in LLM-based chatbot's personality evaluation. Our findings indicate that the chatbot's answers on human personality scales exhibit weak correlations with both user perception and interaction quality, which raises both criterion and predictive validity concerns of such a method. Further analysis revealed the role of task context and interaction in the chatbot's personality design assessment. We discuss the design implications for building contextualized and interactive evaluation of the chatbot's personality design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19865v1">Reverse Thinking Makes LLMs Stronger Reasoners</a></div>
    <div class="paper-meta">
      📅 2024-11-29
      | 💬 20 pages
    </div>
    <details class="paper-abstract">
      Reverse thinking plays a crucial role in human reasoning. Humans can reason not only from a problem to a solution but also in reverse, i.e., start from the solution and reason towards the problem. This often enhances overall reasoning performance as it enables consistency checks between their forward and backward thinking. To enable Large Language Models (LLMs) to perform reverse thinking, we introduce Reverse-Enhanced Thinking (RevThink), a framework composed of data augmentation and learning objectives. In RevThink, we augment the dataset by collecting structured forward-backward reasoning from a teacher model, consisting of: (1) the original question, (2) forward reasoning, (3) backward question, and (4) backward reasoning. We then employ three objectives to train a smaller student model in a multi-task learning fashion: (a) generate forward reasoning from a question, (b) generate a backward question from a question, and (c) generate backward reasoning from the backward question. Experiments across 12 datasets covering commonsense, math, and logical reasoning show an average 13.53% improvement over the student model's zero-shot performance and a 6.84% improvement over the strongest knowledge distillation baselines. Moreover, our method demonstrates sample efficiency -- using only 10% of the correct forward reasoning from the training data, it outperforms a standard fine-tuning method trained on 10x more forward reasoning. RevThink also exhibits strong generalization to out-of-distribution held-out datasets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00161v1">STEP: Enhancing Video-LLMs' Compositional Reasoning by Spatio-Temporal Graph-guided Self-Training</a></div>
    <div class="paper-meta">
      📅 2024-11-29
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video-LLMs) have recently shown strong performance in basic video understanding tasks, such as captioning and coarse-grained question answering, but struggle with compositional reasoning that requires multi-step spatio-temporal inference across object relations, interactions, and events. The hurdles to enhancing this capability include extensive manual labor, the lack of spatio-temporal compositionality in existing data and the absence of explicit reasoning supervision. In this paper, we propose STEP, a novel graph-guided self-training method that enables Video-LLMs to generate reasoning-rich fine-tuning data from any raw videos to improve itself. Specifically, we first induce Spatio-Temporal Scene Graph (STSG) representation of diverse videos to capture fine-grained, multi-granular video semantics. Then, the STSGs guide the derivation of multi-step reasoning Question-Answer (QA) data with Chain-of-Thought (CoT) rationales. Both answers and rationales are integrated as training objective, aiming to enhance model's reasoning abilities by supervision over explicit reasoning steps. Experimental results demonstrate the effectiveness of STEP across models of varying scales, with a significant 21.3\% improvement in tasks requiring three or more reasoning steps. Furthermore, it achieves superior performance with a minimal amount of self-generated rationale-enriched training samples in both compositional reasoning and comprehensive understanding benchmarks, highlighting the broad applicability and vast potential.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19638v1">LLM Teacher-Student Framework for Text Classification With No Manually Annotated Data: A Case Study in IPTC News Topic Classification</a></div>
    <div class="paper-meta">
      📅 2024-11-29
      | 💬 This work has been submitted to the IEEE for possible publication
    </div>
    <details class="paper-abstract">
      With the ever-increasing number of news stories available online, classifying them by topic, regardless of the language they are written in, has become crucial for enhancing readers' access to relevant content. To address this challenge, we propose a teacher-student framework based on large language models (LLMs) for developing multilingual news classification models of reasonable size with no need for manual data annotation. The framework employs a Generative Pretrained Transformer (GPT) model as the teacher model to develop an IPTC Media Topic training dataset through automatic annotation of news articles in Slovenian, Croatian, Greek, and Catalan. The teacher model exhibits a high zero-shot performance on all four languages. Its agreement with human annotators is comparable to that between the human annotators themselves. To mitigate the computational limitations associated with the requirement of processing millions of texts daily, smaller BERT-like student models are fine-tuned on the GPT-annotated dataset. These student models achieve high performance comparable to the teacher model. Furthermore, we explore the impact of the training data size on the performance of the student models and investigate their monolingual, multilingual and zero-shot cross-lingual capabilities. The findings indicate that student models can achieve high performance with a relatively small number of training instances, and demonstrate strong zero-shot cross-lingual abilities. Finally, we publish the best-performing news topic classifier, enabling multilingual classification with the top-level categories of the IPTC Media Topic schema.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.14569v2">When LLMs Go Online: The Emerging Threat of Web-Enabled LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-29
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have established them as agentic systems capable of planning and interacting with various tools. These LLM agents are often paired with web-based tools, enabling access to diverse sources and real-time information. Although these advancements offer significant benefits across various applications, they also increase the risk of malicious use, particularly in cyberattacks involving personal information. In this work, we investigate the risks associated with misuse of LLM agents in cyberattacks involving personal data. Specifically, we aim to understand: 1) how potent LLM agents can be when directed to conduct cyberattacks, 2) how cyberattacks are enhanced by web-based tools, and 3) how affordable and easy it becomes to launch cyberattacks using LLM agents. We examine three attack scenarios: the collection of Personally Identifiable Information (PII), the generation of impersonation posts, and the creation of spear-phishing emails. Our experiments reveal the effectiveness of LLM agents in these attacks: LLM agents achieved a precision of up to 95.9% in collecting PII, up to 93.9% of impersonation posts created by LLM agents were evaluated as authentic, and the click rate for links in spear phishing emails created by LLM agents reached up to 46.67%. Additionally, our findings underscore the limitations of existing safeguards in contemporary commercial LLMs, emphasizing the urgent need for more robust security measures to prevent the misuse of LLM agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18191v2">InputSnatch: Stealing Input in LLM Services via Timing Side-Channel Attacks</a></div>
    <div class="paper-meta">
      📅 2024-11-29
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) possess extensive knowledge and question-answering capabilities, having been widely deployed in privacy-sensitive domains like finance and medical consultation. During LLM inferences, cache-sharing methods are commonly employed to enhance efficiency by reusing cached states or responses for the same or similar inference requests. However, we identify that these cache mechanisms pose a risk of private input leakage, as the caching can result in observable variations in response times, making them a strong candidate for a timing-based attack hint. In this study, we propose a novel timing-based side-channel attack to execute input theft in LLMs inference. The cache-based attack faces the challenge of constructing candidate inputs in a large search space to hit and steal cached user queries. To address these challenges, we propose two primary components. The input constructor employs machine learning techniques and LLM-based approaches for vocabulary correlation learning while implementing optimized search mechanisms for generalized input construction. The time analyzer implements statistical time fitting with outlier elimination to identify cache hit patterns, continuously providing feedback to refine the constructor's search strategy. We conduct experiments across two cache mechanisms and the results demonstrate that our approach consistently attains high attack success rates in various applications. Our work highlights the security vulnerabilities associated with performance optimizations, underscoring the necessity of prioritizing privacy and security alongside enhancements in LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19504v1">TQA-Bench: Evaluating LLMs for Multi-Table Question Answering with Scalable Context and Symbolic Extension</a></div>
    <div class="paper-meta">
      📅 2024-11-29
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) has unlocked great opportunities in complex data management tasks, particularly in question answering (QA) over complicated multi-table relational data. Despite significant progress, systematically evaluating LLMs on multi-table QA remains a critical challenge due to the inherent complexity of analyzing heterogeneous table structures and potential large scale of serialized relational data. Existing benchmarks primarily focus on single-table QA, failing to capture the intricacies of reasoning across multiple relational tables, as required in real-world domains such as finance, healthcare, and e-commerce. To address this gap, we present TQA-Bench, a new multi-table QA benchmark designed to evaluate the capabilities of LLMs in tackling complex QA tasks over relational data. Our benchmark incorporates diverse relational database instances sourced from real-world public datasets and introduces a flexible sampling mechanism to create tasks with varying multi-table context lengths, ranging from 8K to 64K tokens. To ensure robustness and reliability, we integrate symbolic extensions into the evaluation framework, enabling the assessment of LLM reasoning capabilities beyond simple data retrieval or probabilistic pattern matching. We systematically evaluate a range of LLMs, both open-source and closed-source, spanning model scales from 7 billion to 70 billion parameters. Our extensive experiments reveal critical insights into the performance of LLMs in multi-table QA, highlighting both challenges and opportunities for advancing their application in complex, data-driven environments. Our benchmark implementation and results are available at https://github.com/Relaxed-System-Lab/TQA-Bench.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19485v1">Action Engine: An LLM-based Framework for Automatic FaaS Workflow Generation</a></div>
    <div class="paper-meta">
      📅 2024-11-29
      | 💬 Accepted at Utility Cloud Computing (UCC '24) conference
    </div>
    <details class="paper-abstract">
      Function as a Service (FaaS) is poised to become the foundation of the next generation of cloud systems due to its inherent advantages in scalability, cost-efficiency, and ease of use. However, challenges such as the need for specialized knowledge and difficulties in building function workflows persist for cloud-native application developers. To overcome these challenges and mitigate the burden of developing FaaS-based applications, in this paper, we propose a mechanism called Action Engine, that makes use of Tool-Augmented Large Language Models (LLMs) at its kernel to interpret human language queries and automates FaaS workflow generation, thereby, reducing the need for specialized expertise and manual design. Action Engine includes modules to identify relevant functions from the FaaS repository and seamlessly manage the data dependency between them, ensuring that the developer's query is processed and resolved. Beyond that, Action Engine can execute the generated workflow by feeding the user-provided parameters. Our evaluations show that Action Engine can generate workflows with up to 20\% higher correctness without developer involvement. We notice that Action Engine can unlock FaaS workflow generation for non-cloud-savvy developers and expedite the development cycles of cloud-native applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19456v1">Beyond Surface Structure: A Causal Assessment of LLMs' Comprehension Ability</a></div>
    <div class="paper-meta">
      📅 2024-11-29
      | 💬 28 pages, 14 figures, 10 tables
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable capability in natural language tasks, yet debate persists on whether they truly comprehend deep structure (i.e., core semantics) or merely rely on surface structure (e.g., presentation format). Prior studies observe that LLMs' performance declines when intervening on surface structure, arguing their success relies on surface structure recognition. However, surface structure sensitivity does not prevent deep structure comprehension. Rigorously evaluating LLMs' capability requires analyzing both, yet deep structure is often overlooked. To this end, we assess LLMs' comprehension ability using causal mediation analysis, aiming to fully discover the capability of using both deep and surface structures. Specifically, we formulate the comprehension of deep structure as direct causal effect (DCE) and that of surface structure as indirect causal effect (ICE), respectively. To address the non-estimability of original DCE and ICE -- stemming from the infeasibility of isolating mutual influences of deep and surface structures, we develop the corresponding quantifiable surrogates, including approximated DCE (ADCE) and approximated ICE (AICE). We further apply the ADCE to evaluate a series of mainstream LLMs, showing that most of them exhibit deep structure comprehension ability, which grows along with the prediction accuracy. Comparing ADCE and AICE demonstrates closed-source LLMs rely more on deep structure, while open-source LLMs are more surface-sensitive, which decreases with model scale. Theoretically, ADCE is a bidirectional evaluation, which measures both the sufficiency and necessity of deep structure changes in causing output variations, thus offering a more comprehensive assessment than accuracy, a common evaluation in LLMs. Our work provides new insights into LLMs' deep structure comprehension and offers novel methods for LLMs evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.13879v2">User Interaction Patterns and Breakdowns in Conversing with LLM-Powered Voice Assistants</a></div>
    <div class="paper-meta">
      📅 2024-11-28
    </div>
    <details class="paper-abstract">
      Conventional Voice Assistants (VAs) rely on traditional language models to discern user intent and respond to their queries, leading to interactions that often lack a broader contextual understanding, an area in which Large Language Models (LLMs) excel. However, current LLMs are largely designed for text-based interactions, thus making it unclear how user interactions will evolve if their modality is changed to voice. In this work, we investigate whether LLMs can enrich VA interactions via an exploratory study with participants (N=20) using a ChatGPT-powered VA for three scenarios (medical self-diagnosis, creative planning, and discussion) with varied constraints, stakes, and objectivity. We observe that LLM-powered VA elicits richer interaction patterns that vary across tasks, showing its versatility. Notably, LLMs absorb the majority of VA intent recognition failures. We additionally discuss the potential of harnessing LLMs for more resilient and fluid user-VA interactions and provide design guidelines for tailoring LLMs for voice assistance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2401.00820v2">A Computational Framework for Behavioral Assessment of LLM Therapists</a></div>
    <div class="paper-meta">
      📅 2024-11-28
    </div>
    <details class="paper-abstract">
      The emergence of large language models (LLMs) like ChatGPT has increased interest in their use as therapists to address mental health challenges and the widespread lack of access to care. However, experts have emphasized the critical need for systematic evaluation of LLM-based mental health interventions to accurately assess their capabilities and limitations. Here, we propose BOLT, a proof-of-concept computational framework to systematically assess the conversational behavior of LLM therapists. We quantitatively measure LLM behavior across 13 psychotherapeutic approaches with in-context learning methods. Then, we compare the behavior of LLMs against high- and low-quality human therapy. Our analysis based on Motivational Interviewing therapy reveals that LLMs often resemble behaviors more commonly exhibited in low-quality therapy rather than high-quality therapy, such as offering a higher degree of problem-solving advice when clients share emotions. However, unlike low-quality therapy, LLMs reflect significantly more upon clients' needs and strengths. Our findings caution that LLM therapists still require further research for consistent, high-quality care.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19134v2">Confidential Prompting: Protecting User Prompts from Cloud LLM Providers</a></div>
    <div class="paper-meta">
      📅 2024-11-28
    </div>
    <details class="paper-abstract">
      Our work tackles the challenge of securing user inputs in cloud-hosted large language model (LLM) serving while ensuring output invariance, model confidentiality, and compute efficiency. We introduce secure multi-party decoding (SMD), which leverages confidential computing to confine user prompts to a trusted execution environment (TEE), namely a confidential virtual machine (CVM), while allowing service providers to generate tokens efficiently. We also introduce a novel cryptographic method, prompt obfuscation (PO), to ensure robustness against reconstruction attacks on SMD. We demonstrate that our approach preserves both prompt confidentiality and LLM serving efficiency. Our solution can enable privacy-preserving cloud LLM serving that handles sensitive prompts, such as clinical records, financial data, and personal information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19360v1">DENIAHL: In-Context Features Influence LLM Needle-In-A-Haystack Abilities</a></div>
    <div class="paper-meta">
      📅 2024-11-28
    </div>
    <details class="paper-abstract">
      The Needle-in-a-haystack (NIAH) test is a general task used to assess language models' (LMs') abilities to recall particular information from long input context. This framework however does not provide a means of analyzing what factors, beyond context length, contribute to LMs' abilities or inabilities to separate and recall needles from their haystacks. To provide a systematic means of assessing what features contribute to LMs' NIAH capabilities, we developed a synthetic benchmark called DENIAHL (Data-oriented Evaluation of NIAH for LLM's). Our work expands on previous NIAH studies by ablating NIAH features beyond typical context length including data type, size, and patterns. We find stark differences between GPT-3.5 and LLaMA 2-7B's performance on DENIAHL, and drops in recall performance when features like item size are increased, and to some degree when data type is changed from numbers to letters. This has implications for increasingly large context models, demonstrating factors beyond item-number impact NIAH capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20739v3">Gender Bias in LLM-generated Interview Responses</a></div>
    <div class="paper-meta">
      📅 2024-11-28
      | 💬 Accepted to NeurlIPS 2024, SoLaR workshop
    </div>
    <details class="paper-abstract">
      LLMs have emerged as a promising tool for assisting individuals in diverse text-generation tasks, including job-related texts. However, LLM-generated answers have been increasingly found to exhibit gender bias. This study evaluates three LLMs (GPT-3.5, GPT-4, Claude) to conduct a multifaceted audit of LLM-generated interview responses across models, question types, and jobs, and their alignment with two gender stereotypes. Our findings reveal that gender bias is consistent, and closely aligned with gender stereotypes and the dominance of jobs. Overall, this study contributes to the systematic examination of gender bias in LLM-generated interview responses, highlighting the need for a mindful approach to mitigate such biases in related applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.03136v2">Deliberate Reasoning for LLMs as Structure-aware Planning with Accurate World Model</a></div>
    <div class="paper-meta">
      📅 2024-11-28
    </div>
    <details class="paper-abstract">
      Enhancing the reasoning capabilities of large language models (LLMs) remains a key challenge, especially for tasks that require complex, multi-step decision-making. Humans excel at these tasks by leveraging deliberate planning with an internal world model to simulate the potential outcomes of various actions. Inspired by this, we propose a novel multi-step reasoning framework for LLMs, referred to as Structure-aware Planning with Accurate World Model (SWAP). Unlike previous approaches that rely solely on Chain-of-Thought (CoT) reasoning in natural language, SWAP incorporates structural information to guide the reasoning process via a world model and provides a soft verification mechanism over the steps. Moreover, SWAP overcomes the challenge of accurate world state predictions in complex reasoning tasks by introducing a Generator-Discriminator architecture, which enables more reliable world modeling. Specifically, the generator predicts the next state, and the discriminator ensures alignment with the logical consistency required by the problem context. SWAP also encourages the policy model to explore a broad range of potential actions to prevent premature convergence. By resolving the bottlenecks of generation diversity for both actions and states using diversity-based modeling (DBM) and improving discrimination accuracy through contrastive ranking (CR), SWAP significantly enhances the reasoning performance of LLMs. We evaluate SWAP across diverse reasoning-intensive benchmarks including math reasoning, logical reasoning, and coding tasks. Extensive experiments demonstrate that SWAP achieves substantial improvements over the baselines and consistently outperforms existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19234v1">SmartLLMSentry: A Comprehensive LLM Based Smart Contract Vulnerability Detection Framework</a></div>
    <div class="paper-meta">
      📅 2024-11-28
    </div>
    <details class="paper-abstract">
      Smart contracts are essential for managing digital assets in blockchain networks, highlighting the need for effective security measures. This paper introduces SmartLLMSentry, a novel framework that leverages large language models (LLMs), specifically ChatGPT with in-context training, to advance smart contract vulnerability detection. Traditional rule-based frameworks have limitations in integrating new detection rules efficiently. In contrast, SmartLLMSentry utilizes LLMs to streamline this process. We created a specialized dataset of five randomly selected vulnerabilities for model training and evaluation. Our results show an exact match accuracy of 91.1% with sufficient data, although GPT-4 demonstrated reduced performance compared to GPT-3 in rule generation. This study illustrates that SmartLLMSentry significantly enhances the speed and accuracy of vulnerability detection through LLMdriven rule integration, offering a new approach to improving Blockchain security and addressing previously underexplored vulnerabilities in smart contracts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19128v1">Personalized Federated Fine-Tuning for LLMs via Data-Driven Heterogeneous Model Architectures</a></div>
    <div class="paper-meta">
      📅 2024-11-28
      | 💬 On going work. Codes are released at https://github.com/zyc140345/FedAMoLE
    </div>
    <details class="paper-abstract">
      A large amount of instructional text data is essential to enhance the performance of pre-trained large language models (LLMs) for downstream tasks. This data can contain sensitive information and therefore cannot be shared in practice, resulting in data silos that limit the effectiveness of LLMs on various tasks. Federated learning (FL) enables collaborative fine-tuning across different clients without sharing their data. Nonetheless, in practice, this instructional text data is highly heterogeneous in both quantity and distribution across clients, necessitating distinct model structures to best accommodate the variations. However, existing federated fine-tuning approaches either enforce the same model structure or rely on predefined ad-hoc architectures unaware of data distribution, resulting in suboptimal performance. To address this challenge, we propose FedAMoLE, a lightweight personalized federated fine-tuning framework that leverages data-driven heterogeneous model architectures. FedAMoLE introduces the Adaptive Mixture of LoRA Experts (AMoLE) module, which facilitates model heterogeneity with minimal communication overhead by allocating varying numbers of LoRA-based domain experts to each client. Furthermore, we develop a reverse selection-based expert assignment (RSEA) strategy, which enables data-driven model architecture adjustment during fine-tuning by allowing domain experts to select clients that best align with their knowledge domains. Extensive experiments across six different scenarios of data heterogeneity demonstrate that FedAMoLE significantly outperforms existing methods for federated LLM fine-tuning, achieving superior accuracy while maintaining good scalability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19064v1">Way to Specialist: Closing Loop Between Specialized LLM and Evolving Domain Knowledge Graph</a></div>
    <div class="paper-meta">
      📅 2024-11-28
      | 💬 Accepted by KDD 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated exceptional performance across a wide variety of domains. Nonetheless, generalist LLMs continue to fall short in reasoning tasks necessitating specialized knowledge. Prior investigations into specialized LLMs focused on domain-specific training, which entails substantial efforts in domain data acquisition and model parameter fine-tuning. To address these challenges, this paper proposes the Way-to-Specialist (WTS) framework, which synergizes retrieval-augmented generation with knowledge graphs (KGs) to enhance the specialized capability of LLMs in the absence of specialized training. In distinction to existing paradigms that merely utilize external knowledge from general KGs or static domain KGs to prompt LLM for enhanced domain-specific reasoning, WTS proposes an innovative "LLM$\circlearrowright$KG" paradigm, which achieves bidirectional enhancement between specialized LLM and domain knowledge graph (DKG). The proposed paradigm encompasses two closely coupled components: the DKG-Augmented LLM and the LLM-Assisted DKG Evolution. The former retrieves question-relevant domain knowledge from DKG and uses it to prompt LLM to enhance the reasoning capability for domain-specific tasks; the latter leverages LLM to generate new domain knowledge from processed tasks and use it to evolve DKG. WTS closes the loop between DKG-Augmented LLM and LLM-Assisted DKG Evolution, enabling continuous improvement in the domain specialization as it progressively answers and learns from domain-specific questions. We validate the performance of WTS on 6 datasets spanning 5 domains. The experimental results show that WTS surpasses the previous SOTA in 4 specialized domains and achieves a maximum performance improvement of 11.3%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19043v1">Using a Feedback Loop for LLM-based Infrastructure as Code Generation</a></div>
    <div class="paper-meta">
      📅 2024-11-28
      | 💬 4 pages, submitted to accepted by International Journal of Secondary Computing and Applications Research
    </div>
    <details class="paper-abstract">
      Code generation with Large Language Models (LLMs) has helped to increase software developer productivity in coding tasks, but has yet to have significant impact on the tasks of software developers that surround this code. In particular, the challenge of infrastructure management remains an open question. We investigate the ability of an LLM agent to construct infrastructure using the Infrastructure as Code (IaC) paradigm. We particularly investigate the use of a feedback loop that returns errors and warnings on the generated IaC to allow the LLM agent to improve the code. We find that, for each iteration of the loop, its effectiveness decreases exponentially until it plateaus at a certain point and becomes ineffective.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.19038v1">DIESEL -- Dynamic Inference-Guidance via Evasion of Semantic Embeddings in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-28
    </div>
    <details class="paper-abstract">
      In recent years, conversational large language models (LLMs) have shown tremendous success in tasks such as casual conversation, question answering, and personalized dialogue, making significant advancements in domains like virtual assistance, social interaction, and online customer engagement. However, they often generate responses that are not aligned with human values (e.g., ethical standards, safety, or social norms), leading to potentially unsafe or inappropriate outputs. While several techniques have been proposed to address this problem, they come with a cost, requiring computationally expensive training or dramatically increasing the inference time. In this paper, we present DIESEL, a lightweight inference guidance technique that can be seamlessly integrated into any autoregressive LLM to semantically filter undesired concepts from the response. DIESEL can function either as a standalone safeguard or as an additional layer of defense, enhancing response safety by reranking the LLM's proposed tokens based on their similarity to predefined negative concepts in the latent space. This approach provides an efficient and effective solution for maintaining alignment with human values. Our evaluation demonstrates DIESEL's effectiveness on state-of-the-art conversational models (e.g., Llama 3), even in challenging jailbreaking scenarios that test the limits of response safety. We further show that DIESEL can be generalized to use cases other than safety, providing a versatile solution for general-purpose response filtering with minimal computational overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18980v1">Zero-shot Slot Filling in the Age of LLMs for Dialogue Systems</a></div>
    <div class="paper-meta">
      📅 2024-11-28
      | 💬 To appear in Proceedings of COLING 2025
    </div>
    <details class="paper-abstract">
      Zero-shot slot filling is a well-established subtask of Natural Language Understanding (NLU). However, most existing methods primarily focus on single-turn text data, overlooking the unique complexities of conversational dialogue. Conversational data is highly dynamic, often involving abrupt topic shifts, interruptions, and implicit references that make it difficult to directly apply zero-shot slot filling techniques, even with the remarkable capabilities of large language models (LLMs). This paper addresses these challenges by proposing strategies for automatic data annotation with slot induction and black-box knowledge distillation (KD) from a teacher LLM to a smaller model, outperforming vanilla LLMs on internal datasets by 26% absolute increase in F1 score. Additionally, we introduce an efficient system architecture for call center product settings that surpasses off-the-shelf extractive models by 34% relative F1 score, enabling near real-time inference on dialogue streams with higher accuracy, while preserving low latency.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18948v1">Knowledge Database or Poison Base? Detecting RAG Poisoning Attack through LLM Activations</a></div>
    <div class="paper-meta">
      📅 2024-11-28
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are progressively deployed across diverse fields and real-world applications, ensuring the security and robustness of LLMs has become ever more critical. Retrieval-Augmented Generation (RAG) is a cutting-edge approach designed to address the limitations of large language models (LLMs). By retrieving information from the relevant knowledge database, RAG enriches the input to LLMs, enabling them to produce responses that are more accurate and contextually appropriate. It is worth noting that the knowledge database, being sourced from publicly available channels such as Wikipedia, inevitably introduces a new attack surface. RAG poisoning involves injecting malicious texts into the knowledge database, ultimately leading to the generation of the attacker's target response (also called poisoned response). However, there are currently limited methods available for detecting such poisoning attacks. We aim to bridge the gap in this work. Particularly, we introduce RevPRAG, a flexible and automated detection pipeline that leverages the activations of LLMs for poisoned response detection. Our investigation uncovers distinct patterns in LLMs' activations when generating correct responses versus poisoned responses. Our results on multiple benchmark datasets and RAG architectures show our approach could achieve 98% true positive rate, while maintaining false positive rates close to 1%. We also evaluate recent backdoor detection methods specifically designed for LLMs and applicable for identifying poisoned responses in RAG. The results demonstrate that our approach significantly surpasses them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.05311v1">DRC-Coder: Automated DRC Checker Code Generation Using LLM Autonomous Agent</a></div>
    <div class="paper-meta">
      📅 2024-11-28
      | 💬 Proceedings of the 2025 International Symposium on Physical Design (ISPD '25), March 16--19, 2025, Austin, TX, USA
    </div>
    <details class="paper-abstract">
      In the advanced technology nodes, the integrated design rule checker (DRC) is often utilized in place and route tools for fast optimization loops for power-performance-area. Implementing integrated DRC checkers to meet the standard of commercial DRC tools demands extensive human expertise to interpret foundry specifications, analyze layouts, and debug code iteratively. However, this labor-intensive process, requiring to be repeated by every update of technology nodes, prolongs the turnaround time of designing circuits. In this paper, we present DRC-Coder, a multi-agent framework with vision capabilities for automated DRC code generation. By incorporating vision language models and large language models (LLM), DRC-Coder can effectively process textual, visual, and layout information to perform rule interpretation and coding by two specialized LLMs. We also design an auto-evaluation function for LLMs to enable DRC code debugging. Experimental results show that targeting on a sub-3nm technology node for a state-of-the-art standard cell layout tool, DRC-Coder achieves perfect F1 score 1.000 in generating DRC codes for meeting the standard of a commercial DRC tool, highly outperforming standard prompting techniques (F1=0.631). DRC-Coder can generate code for each design rule within four minutes on average, which significantly accelerates technology advancement and reduces engineering costs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18077v2">MiniKV: Pushing the Limits of LLM Inference via 2-Bit Layer-Discriminative KV Cache</a></div>
    <div class="paper-meta">
      📅 2024-11-28
    </div>
    <details class="paper-abstract">
      How to efficiently serve LLMs in practice has become exceptionally challenging due to their prohibitive memory and computation requirements. In this study, we investigate optimizing the KV cache, whose memory footprint poses a critical bottleneck in LLM inference, especially when dealing with long context tasks. To tackle the challenge, we introduce MiniKV, a KV cache optimization method that simultaneously preserves long context task accuracy while significantly reducing KV cache size via a novel 2-bit layer-discriminative KV cache. More importantly, we develop specialized CUDA kernels to make MiniKV compatible with FlashAttention. Experiments on a wide range of long context tasks show that MiniKV effectively achieves 86% KV cache compression ratio while recovering over 98.5% of accuracy, outperforming state-of-the-art methods while achieving excellent measured system performance improvements.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18797v1">UOE: Unlearning One Expert Is Enough For Mixture-of-experts LLMS</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      Recent advancements in large language model (LLM) unlearning have shown remarkable success in removing unwanted data-model influences while preserving the model's utility for legitimate knowledge. However, despite these strides, sparse Mixture-of-Experts (MoE) LLMs--a key subset of the LLM family--have received little attention and remain largely unexplored in the context of unlearning. As MoE LLMs are celebrated for their exceptional performance and highly efficient inference processes, we ask: How can unlearning be performed effectively and efficiently on MoE LLMs? And will traditional unlearning methods be applicable to MoE architectures? Our pilot study shows that the dynamic routing nature of MoE LLMs introduces unique challenges, leading to substantial utility drops when existing unlearning methods are applied. Specifically, unlearning disrupts the router's expert selection, causing significant selection shift from the most unlearning target-related experts to irrelevant ones. As a result, more experts than necessary are affected, leading to excessive forgetting and loss of control over which knowledge is erased. To address this, we propose a novel single-expert unlearning framework, referred to as UOE, for MoE LLMs. Through expert attribution, unlearning is concentrated on the most actively engaged expert for the specified knowledge. Concurrently, an anchor loss is applied to the router to stabilize the active state of this targeted expert, ensuring focused and controlled unlearning that preserves model utility. The proposed UOE framework is also compatible with various unlearning algorithms. Extensive experiments demonstrate that UOE enhances both forget quality up to 5% and model utility by 35% on MoE LLMs across various benchmarks, LLM architectures, while only unlearning 0.06% of the model parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.18400v2">Do LLMs dream of elephants (when told not to)? Latent concept association and associative memory in transformers</a></div>
    <div class="paper-meta">
      📅 2024-11-27
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have the capacity to store and recall facts. Through experimentation with open-source models, we observe that this ability to retrieve facts can be easily manipulated by changing contexts, even without altering their factual meanings. These findings highlight that LLMs might behave like an associative memory model where certain tokens in the contexts serve as clues to retrieving facts. We mathematically explore this property by studying how transformers, the building blocks of LLMs, can complete such memory tasks. We study a simple latent concept association problem with a one-layer transformer and we show theoretically and empirically that the transformer gathers information using self-attention and uses the value matrix for associative memory.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18731v1">The Performance of the LSTM-based Code Generated by Large Language Models (LLMs) in Forecasting Time Series Data</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      As an intriguing case is the goodness of the machine and deep learning models generated by these LLMs in conducting automated scientific data analysis, where a data analyst may not have enough expertise in manually coding and optimizing complex deep learning models and codes and thus may opt to leverage LLMs to generate the required models. This paper investigates and compares the performance of the mainstream LLMs, such as ChatGPT, PaLM, LLama, and Falcon, in generating deep learning models for analyzing time series data, an important and popular data type with its prevalent applications in many application domains including financial and stock market. This research conducts a set of controlled experiments where the prompts for generating deep learning-based models are controlled with respect to sensitivity levels of four criteria including 1) Clarify and Specificity, 2) Objective and Intent, 3) Contextual Information, and 4) Format and Style. While the results are relatively mix, we observe some distinct patterns. We notice that using LLMs, we are able to generate deep learning-based models with executable codes for each dataset seperatly whose performance are comparable with the manually crafted and optimized LSTM models for predicting the whole time series dataset. We also noticed that ChatGPT outperforms the other LLMs in generating more accurate models. Furthermore, we observed that the goodness of the generated models vary with respect to the ``temperature'' parameter used in configuring LLMS. The results can be beneficial for data analysts and practitioners who would like to leverage generative AIs to produce good prediction models with acceptable goodness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18583v1">Automated Literature Review Using NLP Techniques and LLM-Based Retrieval-Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2024-11-27
      | 💬 Key Words : T5, SpaCy, Large Language Model, GPT, ROUGE, Literature Review, Natural Language Processing, Retrieval-augmented generation
    </div>
    <details class="paper-abstract">
      This research presents and compares multiple approaches to automate the generation of literature reviews using several Natural Language Processing (NLP) techniques and retrieval-augmented generation (RAG) with a Large Language Model (LLM). The ever-increasing number of research articles provides a huge challenge for manual literature review. It has resulted in an increased demand for automation. Developing a system capable of automatically generating the literature reviews from only the PDF files as input is the primary objective of this research work. The effectiveness of several Natural Language Processing (NLP) strategies, such as the frequency-based method (spaCy), the transformer model (Simple T5), and retrieval-augmented generation (RAG) with Large Language Model (GPT-3.5-turbo), is evaluated to meet the primary objective. The SciTLDR dataset is chosen for this research experiment and three distinct techniques are utilized to implement three different systems for auto-generating the literature reviews. The ROUGE scores are used for the evaluation of all three systems. Based on the evaluation, the Large Language Model GPT-3.5-turbo achieved the highest ROUGE-1 score, 0.364. The transformer model comes in second place and spaCy is at the last position. Finally, a graphical user interface is created for the best system based on the large language model.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18571v1">Challenges in Adapting Multilingual LLMs to Low-Resource Languages using LoRA PEFT Tuning</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable multilingual capabilities, yet challenges persist in adapting these models for low-resource languages. In this study, we investigate the effects of Low-Rank Adaptation (LoRA) Parameter-Efficient Fine-Tuning (PEFT) on multilingual Gemma models for Marathi, a language with limited resources. Using a translated Alpaca dataset with 52,000 instruction-response pairs, our findings reveal that while evaluation metrics often show a performance decline post-fine-tuning, manual assessments frequently suggest that the fine-tuned models outperform their original counterparts. The observations indicate improvements in target language generation capabilities but a reduction in reasoning abilities following language adaptation. These results underscore the need for improved evaluation methodologies and the creation of high-quality native datasets to accurately assess language-specific model performance in low-resource settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18444v1">Is my Meeting Summary Good? Estimating Quality with a Multi-LLM Evaluator</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      The quality of meeting summaries generated by natural language generation (NLG) systems is hard to measure automatically. Established metrics such as ROUGE and BERTScore have a relatively low correlation with human judgments and fail to capture nuanced errors. Recent studies suggest using large language models (LLMs), which have the benefit of better context understanding and adaption of error definitions without training on a large number of human preference judgments. However, current LLM-based evaluators risk masking errors and can only serve as a weak proxy, leaving human evaluation the gold standard despite being costly and hard to compare across studies. In this work, we present MESA, an LLM-based framework employing a three-step assessment of individual error types, multi-agent discussion for decision refinement, and feedback-based self-training to refine error definition understanding and alignment with human judgment. We show that MESA's components enable thorough error detection, consistent rating, and adaptability to custom error guidelines. Using GPT-4o as its backbone, MESA achieves mid to high Point-Biserial correlation with human judgment in error detection and mid Spearman and Kendall correlation in reflecting error impact on summary quality, on average 0.25 higher than previous methods. The framework's flexibility in adapting to custom error guidelines makes it suitable for various tasks with limited human-labeled data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18337v1">Can LLMs assist with Ambiguity? A Quantitative Evaluation of various Large Language Models on Word Sense Disambiguation</a></div>
    <div class="paper-meta">
      📅 2024-11-27
      | 💬 12 pages,6 tables, 1 figure, Proceedings of the 1st International Conference on NLP & AI for Cyber Security
    </div>
    <details class="paper-abstract">
      Ambiguous words are often found in modern digital communications. Lexical ambiguity challenges traditional Word Sense Disambiguation (WSD) methods, due to limited data. Consequently, the efficiency of translation, information retrieval, and question-answering systems is hindered by these limitations. This study investigates the use of Large Language Models (LLMs) to improve WSD using a novel approach combining a systematic prompt augmentation mechanism with a knowledge base (KB) consisting of different sense interpretations. The proposed method incorporates a human-in-loop approach for prompt augmentation where prompt is supported by Part-of-Speech (POS) tagging, synonyms of ambiguous words, aspect-based sense filtering and few-shot prompting to guide the LLM. By utilizing a few-shot Chain of Thought (COT) prompting-based approach, this work demonstrates a substantial improvement in performance. The evaluation was conducted using FEWS test data and sense tags. This research advances accurate word interpretation in social media and digital communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19657v3">LLMEasyQuant -- An Easy to Use Toolkit for LLM Quantization</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      Currently, there are many quantization methods appeared for LLM quantization, yet few are user-friendly and easy to be deployed locally. Packages like TensorRT and Quantohave many underlying structures and self-invoking internal functions, which are not conducive to developers' personalized development and learning for deployment. Therefore, we develop LLMEasyQuant, it is a package aiming to for easy quantization deployment which is user-friendly and suitable for beginners' learning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15115v3">On Designing Effective RL Reward at Training Time for LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      Reward models have been increasingly critical for improving the reasoning capability of LLMs. Existing research has shown that a well-trained reward model can substantially improve model performances at inference time via search. However, the potential of reward models during RL training time still remains largely under-explored. It is currently unclear whether these reward models can provide additional training signals to enhance the reasoning capabilities of LLMs in RL training that uses sparse success rewards, which verify the correctness of solutions. In this work, we evaluate popular reward models for RL training, including the Outcome-supervised Reward Model (ORM) and the Process-supervised Reward Model (PRM), and train a collection of LLMs for math problems using RL by combining these learned rewards with success rewards. Surprisingly, even though these learned reward models have strong inference-time performances, they may NOT help or even hurt RL training, producing worse performances than LLMs trained with the success reward only. Our analysis reveals that an LLM can receive high rewards from some of these reward models by repeating correct but unnecessary reasoning steps, leading to a severe reward hacking issue. Therefore, we introduce two novel reward refinement techniques, including Clipping and Delta. The key idea is to ensure the accumulative reward of any reasoning trajectory is upper-bounded to keep a learned reward model effective without being exploited. We evaluate our techniques with multiple reward models over a set of 1.5B and 7B LLMs on MATH and GSM8K benchmarks and demonstrate that with a carefully designed reward function, RL training without any additional supervised tuning can improve all the evaluated LLMs, including the state-of-the-art 7B LLM Qwen2.5-Math-7B-Instruct on MATH and GSM8K benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18216v1">Evaluating and Improving the Robustness of Security Attack Detectors Generated by LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in software development to generate functions, such as attack detectors, that implement security requirements. However, LLMs struggle to generate accurate code, resulting, e.g., in attack detectors that miss well-known attacks when used in practice. This is most likely due to the LLM lacking knowledge about some existing attacks and to the generated code being not evaluated in real usage scenarios. We propose a novel approach integrating Retrieval Augmented Generation (RAG) and Self-Ranking into the LLM pipeline. RAG enhances the robustness of the output by incorporating external knowledge sources, while the Self-Ranking technique, inspired to the concept of Self-Consistency, generates multiple reasoning paths and creates ranks to select the most robust detector. Our extensive empirical study targets code generated by LLMs to detect two prevalent injection attacks in web security: Cross-Site Scripting (XSS) and SQL injection (SQLi). Results show a significant improvement in detection performance compared to baselines, with an increase of up to 71%pt and 37%pt in the F2-Score for XSS and SQLi detection, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18211v1">TimeMarker: A Versatile Video-LLM for Long and Short Video Understanding with Superior Temporal Localization Ability</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      Rapid development of large language models (LLMs) has significantly advanced multimodal large language models (LMMs), particularly in vision-language tasks. However, existing video-language models often overlook precise temporal localization and struggle with videos of varying lengths. We introduce TimeMarker, a versatile Video-LLM designed for high-quality dialogue based on video content, emphasizing temporal localization. TimeMarker integrates Temporal Separator Tokens to enhance temporal awareness, accurately marking specific moments within videos. It employs the AnyLength mechanism for dynamic frame sampling and adaptive token merging, enabling effective handling of both short and long videos. Additionally, TimeMarker utilizes diverse datasets, including further transformed temporal-related video QA datasets, to bolster its temporal understanding capabilities. Image and interleaved data are also employed to further enhance the model's semantic perception ability. Evaluations demonstrate that TimeMarker achieves state-of-the-art performance across multiple benchmarks, excelling in both short and long video categories. Our project page is at \url{https://github.com/TimeMarker-LLM/TimeMarker/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19226v2">Simulating Classroom Education with LLM-Empowered Agents</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been applied across various intelligent educational tasks to assist teaching. While preliminary studies have focused on task-specific, independent LLM-empowered agents, the potential of LLMs within a multi-agent collaborative framework for classroom simulation with real user participation remains unexplored. In this work, we propose SimClass, a multi-agent classroom simulation teaching framework. We recognize representative class roles and introduce a novel class control mechanism for automatic classroom teaching, and conduct user experiments in two real-world courses. Using the Flanders Interactive Analysis System and Community of Inquiry theoretical frameworks from educational analysis, we demonstrate that LLMs can simulate a dynamic learning environment for users with active teacher-student and student-student interactions. We also observe group behaviors among agents in SimClass, where agents collaborate to create enlivening interactions in classrooms to improve user learning process. We hope this work pioneers the application of LLM-empowered multi-agent systems in virtual classroom teaching.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18138v1">SALMONN-omni: A Codec-free LLM for Full-duplex Speech Understanding and Generation</a></div>
    <div class="paper-meta">
      📅 2024-11-27
      | 💬 Technical report
    </div>
    <details class="paper-abstract">
      Full-duplex multimodal large language models (LLMs) provide a unified framework for addressing diverse speech understanding and generation tasks, enabling more natural and seamless human-machine conversations. Unlike traditional modularised conversational AI systems, which separate speech recognition, understanding, and text-to-speech generation into distinct components, multimodal LLMs operate as single end-to-end models. This streamlined design eliminates error propagation across components and fully leverages the rich non-verbal information embedded in input speech signals. We introduce SALMONN-omni, a codec-free, full-duplex speech understanding and generation model capable of simultaneously listening to its own generated speech and background sounds while speaking. To support this capability, we propose a novel duplex spoken dialogue framework incorporating a ``thinking'' mechanism that facilitates asynchronous text and speech generation relying on embeddings instead of codecs (quantized speech and audio tokens). Experimental results demonstrate SALMONN-omni's versatility across a broad range of streaming speech tasks, including speech recognition, speech enhancement, and spoken question answering. Additionally, SALMONN-omni excels at managing turn-taking, barge-in, and echo cancellation scenarios, establishing its potential as a robust prototype for full-duplex conversational AI systems. To the best of our knowledge, SALMONN-omni is the first codec-free model of its kind. A full technical report along with model checkpoints will be released soon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.12762v2">Playing Language Game with LLMs Leads to Jailbreaking</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      The advent of large language models (LLMs) has spurred the development of numerous jailbreak techniques aimed at circumventing their security defenses against malicious attacks. An effective jailbreak approach is to identify a domain where safety generalization fails, a phenomenon known as mismatched generalization. In this paper, we introduce two novel jailbreak methods based on mismatched generalization: natural language games and custom language games, both of which effectively bypass the safety mechanisms of LLMs, with various kinds and different variants, making them hard to defend and leading to high attack rates. Natural language games involve the use of synthetic linguistic constructs and the actions intertwined with these constructs, such as the Ubbi Dubbi language. Building on this phenomenon, we propose the custom language games method: by engaging with LLMs using a variety of custom rules, we successfully execute jailbreak attacks across multiple LLM platforms. Extensive experiments demonstrate the effectiveness of our methods, achieving success rates of 93% on GPT-4o, 89% on GPT-4o-mini and 83% on Claude-3.5-Sonnet. Furthermore, to investigate the generalizability of safety alignments, we fine-tuned Llama-3.1-70B with the custom language games to achieve safety alignment within our datasets and found that when interacting through other language games, the fine-tuned models still failed to identify harmful content. This finding indicates that the safety alignment knowledge embedded in LLMs fails to generalize across different linguistic formats, thus opening new avenues for future research in this area.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06208v2">IOPO: Empowering LLMs with Complex Instruction Following via Input-Output Preference Optimization</a></div>
    <div class="paper-meta">
      📅 2024-11-27
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      In the realm of large language models (LLMs), the ability of models to accurately follow instructions is paramount as more agents and applications leverage LLMs for construction, where the complexity of instructions are rapidly increasing. However, on the one hand, there is only a certain amount of complex instruction evaluation data; on the other hand, there are no dedicated algorithms to improve the ability to follow complex instructions. To this end, this paper introduces TRACE, a benchmark for improving and evaluating the complex instructionfollowing ability, which consists of 120K training data and 1K evaluation data. Furthermore, we propose IOPO (Input-Output Preference Optimization) alignment method which takes both input and output preference pairs into consideration, where LLMs not only rapidly align with response preferences but also meticulously explore the instruction preferences. Extensive experiments on both in-domain and outof-domain datasets confirm the effectiveness of IOPO, showing 8.15%, 2.18% improvements on in-domain data and 6.29%, 3.13% on outof-domain data compared to SFT and DPO respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06387v3">Self-Training Meets Consistency: Improving LLMs' Reasoning With Consistency-Driven Rationale Evaluation</a></div>
    <div class="paper-meta">
      📅 2024-11-27
      | 💬 Under review
    </div>
    <details class="paper-abstract">
      Self-training approach for large language models (LLMs) improves reasoning abilities by training the models on their self-generated rationales. Previous approaches have labeled rationales that produce correct answers for a given question as appropriate for training. However, a single measure risks misjudging rationale quality, leading the models to learn flawed reasoning patterns. To address this issue, we propose CREST (Consistency-driven Rationale Evaluation for Self-Training), a self-training framework that further evaluates each rationale through follow-up questions and leverages this evaluation to guide its training. Specifically, we introduce two methods: (1) filtering out rationales that frequently result in incorrect answers on follow-up questions and (2) preference learning based on mixed preferences from rationale evaluation results of both original and follow-up questions. Experiments on three question-answering datasets using open LLMs show that CREST not only improves the logical robustness and correctness of rationales but also improves reasoning abilities compared to previous self-training approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18071v1">Simulating Tabular Datasets through LLMs to Rapidly Explore Hypotheses about Real-World Entities</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      Do horror writers have worse childhoods than other writers? Though biographical details are known about many writers, quantitatively exploring such a qualitative hypothesis requires significant human effort, e.g. to sift through many biographies and interviews of writers and to iteratively search for quantitative features that reflect what is qualitatively of interest. This paper explores the potential to quickly prototype these kinds of hypotheses through (1) applying LLMs to estimate properties of concrete entities like specific people, companies, books, kinds of animals, and countries; (2) performing off-the-shelf analysis methods to reveal possible relationships among such properties (e.g. linear regression); and towards further automation, (3) applying LLMs to suggest the quantitative properties themselves that could help ground a particular qualitative hypothesis (e.g. number of adverse childhood events, in the context of the running example). The hope is to allow sifting through hypotheses more quickly through collaboration between human and machine. Our experiments highlight that indeed, LLMs can serve as useful estimators of tabular data about specific entities across a range of domains, and that such estimations improve with model scale. Further, initial experiments demonstrate the potential of LLMs to map a qualitative hypothesis of interest to relevant concrete variables that the LLM can then estimate. The conclusion is that LLMs offer intriguing potential to help illuminate scientifically interesting patterns latent within the internet-scale data they are trained upon.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17691v2">Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens</a></div>
    <div class="paper-meta">
      📅 2024-11-27
      | 💬 Work in Progress
    </div>
    <details class="paper-abstract">
      We reveal that low-bit quantization favors undertrained large language models (LLMs) by observing that models with larger sizes or fewer training tokens experience less quantization-induced degradation (QiD) when applying low-bit quantization, whereas smaller models with extensive training tokens suffer significant QiD. To gain deeper insights into this trend, we study over 1500 quantized LLM checkpoints of various sizes and at different training levels (undertrained or fully trained) in a controlled setting, deriving scaling laws for understanding the relationship between QiD and factors such as the number of training tokens, model size and bit width. With the derived scaling laws, we propose a novel perspective that we can use QiD to measure an LLM's training levels and determine the number of training tokens required for fully training LLMs of various sizes. Moreover, we use the scaling laws to predict the quantization performance of different-sized LLMs trained with 100 trillion tokens. Our projection shows that the low-bit quantization performance of future models, which are expected to be trained with over 100 trillion tokens, may NOT be desirable. This poses a potential challenge for low-bit quantization in the future and highlights the need for awareness of a model's training level when evaluating low-bit quantization research. To facilitate future research on this problem, we release all the 1500+ quantized checkpoints used in this work at https://huggingface.co/Xu-Ouyang.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17989v1">Regularized Multi-LLMs Collaboration for Enhanced Score-based Causal Discovery</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      As the significance of understanding the cause-and-effect relationships among variables increases in the development of modern systems and algorithms, learning causality from observational data has become a preferred and efficient approach over conducting randomized control trials. However, purely observational data could be insufficient to reconstruct the true causal graph. Consequently, many researchers tried to utilise some form of prior knowledge to improve causal discovery process. In this context, the impressive capabilities of large language models (LLMs) have emerged as a promising alternative to the costly acquisition of prior expert knowledge. In this work, we further explore the potential of using LLMs to enhance causal discovery approaches, particularly focusing on score-based methods, and we propose a general framework to utilise the capacity of not only one but multiple LLMs to augment the discovery process.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17981v1">Engineering Trustworthy Software: A Mission for LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      LLMs are transforming software engineering by accelerating development, reducing complexity, and cutting costs. When fully integrated into the software lifecycle they will drive design, development and deployment while facilitating early bug detection, continuous improvement, and rapid resolution of critical issues. However, trustworthy LLM-driven software engineering requires addressing multiple challenges such as accuracy, scalability, bias, and explainability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17967v1">QuaLLM-Health: An Adaptation of an LLM-Based Framework for Quantitative Data Extraction from Online Health Discussions</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      Health-related discussions on social media like Reddit offer valuable insights, but extracting quantitative data from unstructured text is challenging. In this work, we present an adapted framework from QuaLLM into QuaLLM-Health for extracting clinically relevant quantitative data from Reddit discussions about glucagon-like peptide-1 (GLP-1) receptor agonists using large language models (LLMs). We collected 410k posts and comments from five GLP-1-related communities using the Reddit API in July 2024. After filtering for cancer-related discussions, 2,059 unique entries remained. We developed annotation guidelines to manually extract variables such as cancer survivorship, family cancer history, cancer types mentioned, risk perceptions, and discussions with physicians. Two domain-experts independently annotated a random sample of 100 entries to create a gold-standard dataset. We then employed iterative prompt engineering with OpenAI's "GPT-4o-mini" on the gold-standard dataset to build an optimized pipeline that allowed us to extract variables from the large dataset. The optimized LLM achieved accuracies above 0.85 for all variables, with precision, recall and F1 score macro averaged > 0.90, indicating balanced performance. Stability testing showed a 95% match rate across runs, confirming consistency. Applying the framework to the full dataset enabled efficient extraction of variables necessary for downstream analysis, costing under $3 and completing in approximately one hour. QuaLLM-Health demonstrates that LLMs can effectively and efficiently extract clinically relevant quantitative data from unstructured social media content. Incorporating human expertise and iterative prompt refinement ensures accuracy and reliability. This methodology can be adapted for large-scale analysis of patient-generated data across various health domains, facilitating valuable insights for healthcare research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2406.00231v2">LLM-RankFusion: Mitigating Intrinsic Inconsistency in LLM-based Ranking</a></div>
    <div class="paper-meta">
      📅 2024-11-27
    </div>
    <details class="paper-abstract">
      Ranking passages by prompting a large language model (LLM) can achieve promising performance in modern information retrieval (IR) systems. A common approach to sort the ranking list is by prompting LLMs for a pairwise or setwise comparison which often relies on sorting algorithms. However, sorting-based methods require consistent comparisons to correctly sort the passages, which we show that LLMs often violate. We identify two kinds of intrinsic inconsistency in LLM-based pairwise comparisons: order inconsistency which leads to conflicting results when switching the passage order, and transitive inconsistency which leads to non-transitive triads among all preference pairs. Our study of these inconsistencies is relevant for understanding and improving the stability of any ranking scheme based on relative preferences. In this paper, we propose LLM-RankFusion, an LLM-based ranking framework that mitigates these inconsistencies and produces a robust ranking list. LLM-RankFusion mitigates order inconsistency using in-context learning (ICL) to demonstrate order-agnostic comparisons and calibration to estimate the underlying preference probability between two passages. We then address transitive inconsistency by aggregating the ranking results from multiple rankers. In our experiments, we empirically show that LLM-RankFusion can significantly reduce inconsistent comparison results, improving the ranking quality by making the final ranking list more robust. Our code is available at \href{https://github.com/XHMY/LLM-RankFusion}{https://github.com/XHMY/LLM-RankFusion}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17927v1">Measuring Emergent Capabilities of LLMs for Software Engineering: How Far Are We?</a></div>
    <div class="paper-meta">
      📅 2024-11-26
      | 💬 Submitted for RENE ICPC 2025
    </div>
    <details class="paper-abstract">
      The adoption of Large Language Models (LLMs) across multiple contexts has sparked interest in understanding how scaling model size might lead to behavioral changes, as LLMs can exhibit behaviors not observed in their smaller counterparts. Understanding these emergent capabilities is essential for advancing LLM development and improving their interpretability across diverse tasks. However, whether LLMs exhibit true emergence in the context of Software Engineering remains an unexplored topic, as most research has focused on NLP tasks. In this paper, we investigate the emergence of capabilities in the context of SE. We propose a model-agnostic pipeline for evaluating this phenomenon across three SE tasks: bug fixing, code translation, and commit message generation. More precisely, for each task, we present a case study instantiating our pipeline to analyze the emergence of capabilities in CodeGen1-multi across four scales ranging from 350M to 16.1B parameters. Our findings do not not provide evidence to support the idea of emergent capabilities resulting from scaling the model size in the selected set of tasks. We hope our results can pave the way to a more nuanced understanding of emergent capabilities of LLMs within the SE domain, guiding future research to focus on task-specific evaluations and the identification of alternative factors contributing to this phenomenon. Our work underscores the importance of task diversity in examining model behaviors and highlights potential limitations in transferring prior understandings of and approaches to emergence from NLP to Software Engineering.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.24049v3">Desert Camels and Oil Sheikhs: Arab-Centric Red Teaming of Frontier LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely used but raise ethical concerns due to embedded social biases. This study examines LLM biases against Arabs versus Westerners across eight domains, including women's rights, terrorism, and anti-Semitism and assesses model resistance to perpetuating these biases. To this end, we create two datasets: one to evaluate LLM bias toward Arabs versus Westerners and another to test model safety against prompts that exaggerate negative traits ("jailbreaks"). We evaluate six LLMs -- GPT-4, GPT-4o, LlaMA 3.1 (8B & 405B), Mistral 7B, and Claude 3.5 Sonnet. We find 79% of cases displaying negative biases toward Arabs, with LlaMA 3.1-405B being the most biased. Our jailbreak tests reveal GPT-4o as the most vulnerable, despite being an optimized version, followed by LlaMA 3.1-8B and Mistral 7B. All LLMs except Claude exhibit attack success rates above 87% in three categories. We also find Claude 3.5 Sonnet the safest, but it still displays biases in seven of eight categories. Despite being an optimized version of GPT4, We find GPT-4o to be more prone to biases and jailbreaks, suggesting optimization flaws. Our findings underscore the pressing need for more robust bias mitigation strategies and strengthened security measures in LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17693v1">Adaptive Deployment of Untrusted LLMs Reduces Distributed Threats</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become increasingly capable, it is prudent to assess whether safety measures remain effective even if LLMs intentionally try to bypass them. Previous work introduced control evaluations, an adversarial framework for testing deployment strategies of untrusted models (i.e., models which might be trying to bypass safety measures). While prior work treats a single failure as unacceptable, we perform control evaluations in a "distributed threat setting" -- a setting where no single action is catastrophic and no single action provides overwhelming evidence of misalignment. We approach this problem with a two-level deployment framework that uses an adaptive macro-protocol to choose between micro-protocols. Micro-protocols operate on a single task, using a less capable, but extensively tested (trusted) model to harness and monitor the untrusted model. Meanwhile, the macro-protocol maintains an adaptive credence on the untrusted model's alignment based on its past actions, using it to pick between safer and riskier micro-protocols. We evaluate our method in a code generation testbed where a red team attempts to generate subtly backdoored code with an LLM whose deployment is safeguarded by a blue team. We plot Pareto frontiers of safety (# of non-backdoored solutions) and usefulness (# of correct solutions). At a given level of usefulness, our adaptive deployment strategy reduces the number of backdoors by 80% compared to non-adaptive baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.02611v3">LOLA: LLM-Assisted Online Learning Algorithm for Content Experiments</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Modern media firms require automated and efficient methods to identify content that is most engaging and appealing to users. Leveraging a large-scale dataset from Upworthy (a news publisher), which includes 17,681 headline A/B tests, we first investigate the ability of three pure-LLM approaches to identify the catchiest headline: prompt-based methods, embedding-based methods, and fine-tuned open-source LLMs. Prompt-based approaches perform poorly, while both OpenAI-embedding-based models and the fine-tuned Llama-3-8B achieve marginally higher accuracy than random predictions. In sum, none of the pure-LLM-based methods can predict the best-performing headline with high accuracy. We then introduce the LLM-Assisted Online Learning Algorithm (LOLA), a novel framework that integrates Large Language Models (LLMs) with adaptive experimentation to optimize content delivery. LOLA combines the best pure-LLM approach with the Upper Confidence Bound algorithm to allocate traffic and maximize clicks adaptively. Our numerical experiments on Upworthy data show that LOLA outperforms the standard A/B test method (the current status quo at Upworthy), pure bandit algorithms, and pure-LLM approaches, particularly in scenarios with limited experimental traffic. Our approach is scalable and applicable to content experiments across various settings where firms seek to optimize user engagement, including digital advertising and social media recommendations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17674v1">Push the Limit of Multi-modal Emotion Recognition by Prompting LLMs with Receptive-Field-Aware Attention Weighting</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Understanding the emotions in a dialogue usually requires external knowledge to accurately understand the contents. As the LLMs become more and more powerful, we do not want to settle on the limited ability of the pre-trained language model. However, the LLMs either can only process text modality or are too expensive to process the multimedia information. We aim to utilize both the power of LLMs and the supplementary features from the multimedia modalities. In this paper, we present a framework, Lantern, that can improve the performance of a certain vanilla model by prompting large language models with receptive-field-aware attention weighting. This framework trained a multi-task vanilla model to produce probabilities of emotion classes and dimension scores. These predictions are fed into the LLMs as references to adjust the predicted probabilities of each emotion class with its external knowledge and contextual understanding. We slice the dialogue into different receptive fields, and each sample is included in exactly t receptive fields. Finally, the predictions of LLMs are merged with a receptive-field-aware attention-driven weighting module. In the experiments, vanilla models CORECT and SDT are deployed in Lantern with GPT-4 or Llama-3.1-405B. The experiments in IEMOCAP with 4-way and 6-way settings demonstrated that the Lantern can significantly improve the performance of current vanilla models by up to 1.23% and 1.80%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17672v1">Synthetic Data Generation with LLM for Improved Depression Prediction</a></div>
    <div class="paper-meta">
      📅 2024-11-26
      | 💬 6 pages excluding references and appendix
    </div>
    <details class="paper-abstract">
      Automatic detection of depression is a rapidly growing field of research at the intersection of psychology and machine learning. However, with its exponential interest comes a growing concern for data privacy and scarcity due to the sensitivity of such a topic. In this paper, we propose a pipeline for Large Language Models (LLMs) to generate synthetic data to improve the performance of depression prediction models. Starting from unstructured, naturalistic text data from recorded transcripts of clinical interviews, we utilize an open-source LLM to generate synthetic data through chain-of-thought prompting. This pipeline involves two key steps: the first step is the generation of the synopsis and sentiment analysis based on the original transcript and depression score, while the second is the generation of the synthetic synopsis/sentiment analysis based on the summaries generated in the first step and a new depression score. Not only was the synthetic data satisfactory in terms of fidelity and privacy-preserving metrics, it also balanced the distribution of severity in the training dataset, thereby significantly enhancing the model's capability in predicting the intensity of the patient's depression. By leveraging LLMs to generate synthetic data that can be augmented to limited and imbalanced real-world datasets, we demonstrate a novel approach to addressing data scarcity and privacy concerns commonly faced in automatic depression detection, all while maintaining the statistical integrity of the original dataset. This approach offers a robust framework for future mental health research and applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17651v1">Toward High-Performance LLM Serving: A Simulation-Based Approach for Identifying Optimal Parallelism</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Serving Large Language Models (LLMs) efficiently has become crucial. LLMs are often served with multiple devices using techniques like data, pipeline, and tensor parallelisms. Each parallelism presents trade-offs between computation, memory, and communication overhead, making it challenging to determine the optimal parallel execution plan. Moreover, input workloads also impact parallelism strategies. Tasks with long prompts like article summarization are compute-intensive, while tasks with long generation lengths like code generation are often memory-intensive; these differing characteristics result in distinct optimal execution plans. Since searching for the optimal plan via actual deployment is prohibitively expensive, we propose APEX, an LLM serving system simulator that efficiently identifies an optimal parallel execution plan. APEX captures the complex characteristics of iteration-level batching, a technique widely used in SOTA LLM serving systems. APEX leverages the repetitive structure of LLMs to reduce design space, maintaining a similar simulation overhead, even when scaling to trillion scale models. APEX supports a wide range of LLMs, device clusters, etc., and it can be easily extended through its high-level templates. We run APEX simulations using a CPU and evaluate the identified optimal plans using 8 H100 GPUs, encompassing a wide range of LLMs and input workloads. We show that APEX can find optimal execution plans that are up to 4.42x faster than heuristic plans in terms of end-to-end serving latency. APEX also reports a set of metrics used in LLM serving systems, such as time per output token and time to first token. Furthermore, APEX can identify an optimal parallel execution plan within 15 minutes using a CPU. This is 71x faster and 1234x more cost-effective than actual deployment on a GPU cluster using cloud services. APEX will be open-sourced upon acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17637v1">On Limitations of LLM as Annotator for Low Resource Languages</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Low-resource languages face significant challenges due to the lack of sufficient linguistic data, resources, and tools for tasks such as supervised learning, annotation, and classification. This shortage hinders the development of accurate models and datasets, making it difficult to perform critical NLP tasks like sentiment analysis or hate speech detection. To bridge this gap, Large Language Models (LLMs) present an opportunity for potential annotators, capable of generating datasets and resources for these underrepresented languages. In this paper, we focus on Marathi, a low-resource language, and evaluate the performance of both closed-source and open-source LLMs as annotators. We assess models such as GPT-4o and Gemini 1.0 Pro, Gemma 2 (2B and 9B), and Llama 3.1 (8B) on classification tasks including sentiment analysis, news classification, and hate speech detection. Our findings reveal that while LLMs excel in annotation tasks for high-resource languages like English, they still fall short when applied to Marathi. Even advanced closed models like Gemini and GPT underperform in comparison to BERT-based baselines, highlighting the limitations of LLMs as annotators for low-resource languages.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17794v1">NEMO: Can Multimodal LLMs Identify Attribute-Modified Objects?</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Multimodal Large Language Models (MLLMs) have made notable advances in visual understanding, yet their abilities to recognize objects modified by specific attributes remain an open question. To address this, we explore MLLMs' reasoning capabilities in object recognition, ranging from commonsense to beyond-commonsense scenarios. We introduce a novel benchmark, NEMO, which comprises 900 images of origiNal fruits and their corresponding attributE-MOdified ones; along with a set of 2,700 questions including open-, multiple-choice-, unsolvable types. We assess 26 recent open-sourced and commercial models using our benchmark. The findings highlight pronounced performance gaps in recognizing objects in NEMO and reveal distinct answer preferences across different models. Although stronger vision encoders improve performance, MLLMs still lag behind standalone vision encoders. Interestingly, scaling up the model size does not consistently yield better outcomes, as deeper analysis reveals that larger LLMs can weaken vision encoders during fine-tuning. These insights shed light on critical limitations in current MLLMs and suggest potential pathways toward developing more versatile and resilient multimodal models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17792v1">$H^3$Fusion: Helpful, Harmless, Honest Fusion of Aligned LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Alignment of pretrained LLMs using instruction-based datasets is critical for creating fine-tuned models that reflect human preference. A growing number of alignment-based fine-tuning algorithms and benchmarks emerged recently, fueling the efforts on effective alignments of pre-trained LLMs to ensure helpful, harmless, and honest answers from both open-source and closed-source LLMs. This paper tackles this problem by developing an alignment fusion approach, coined as $H^3$Fusion, with three unique characteristics. First, $H^3$Fusion ensembles multiple individually aligned LLMs to create a final fine-tuned alignment model with enhanced capabilities beyond those of individual models, delivering robust alignment through promoting helpful, harmless, honest fusion. Second, $H^3$Fusion leverages the mixture-of-experts (MoE) methodology in two steps. We first freeze the multi-head attention weights of each individual model while tuning the FFN layer during alignment fusion. Then we merge the aligned model weights with an expert router according to the type of input instruction and dynamically select a subset of experts that are best suited for producing the output response. Finally, we boost the performance of the resulting $H^3$3Fusion model by introducing gating loss and regularization terms. The former penalizes the selection errors of the expert-router, and the latter mediates the expert weights drifting during fine-tuning and dynamically adjusts the fusion behavior of the resulting model by canalizing the activations on the experts. Extensive evaluations on three benchmark datasets show that $H^3$3Fusion is more helpful, less harmful, and more honest from two aspects: it outperforms each individually aligned model by $11.37\%$, and it provides stronger robustness compared to the state-of-the-art LLM ensemble approaches by $13.77\%$. Code is available at github.com/sftekin/h3fusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01392v2">ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Much previous AI research has focused on developing monolithic models to maximize their intelligence, with the primary goal of enhancing performance on specific tasks. In contrast, this work attempts to study using LLM-based agents to design collaborative AI systems autonomously. To explore this problem, we first introduce ComfyBench to evaluate agents's ability to design collaborative AI systems in ComfyUI. ComfyBench is a comprehensive benchmark comprising 200 diverse tasks covering various instruction-following generation challenges, along with detailed annotations for 3,205 nodes and 20 workflows. Based on ComfyBench, we further develop ComfyAgent, a novel framework that empowers LLM-based agents to autonomously design collaborative AI systems by generating workflows. ComfyAgent is based on two core concepts. First, it represents workflows with code, which can be reversibly converted into workflows and executed as collaborative systems by the interpreter. Second, it constructs a multi-agent system that cooperates to learn from existing workflows and generate new workflows for a given task. While experimental results demonstrate that ComfyAgent achieves a comparable resolve rate to o1-preview and significantly surpasses other agents on ComfyBench, ComfyAgent has resolved only 15\% of creative tasks. LLM-based agents still have a long way to go in autonomously designing collaborative AI systems. Progress with ComfyBench is paving the way for more intelligent and autonomous collaborative AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.19100v3">Enhancing Zero-Shot Facial Expression Recognition by LLM Knowledge Transfer</a></div>
    <div class="paper-meta">
      📅 2024-11-26
      | 💬 Accepted at WACV 2025 (Camera-Ready Version)
    </div>
    <details class="paper-abstract">
      Current facial expression recognition (FER) models are often designed in a supervised learning manner and thus are constrained by the lack of large-scale facial expression images with high-quality annotations. Consequently, these models often fail to generalize well, performing poorly on unseen images in inference. Vision-language-based zero-shot models demonstrate a promising potential for addressing such challenges. However, these models lack task-specific knowledge and therefore are not optimized for the nuances of recognizing facial expressions. To bridge this gap, this work proposes a novel method, Exp-CLIP, to enhance zero-shot FER by transferring the task knowledge from large language models (LLMs). Specifically, based on the pre-trained vision-language encoders, we incorporate a projection head designed to map the initial joint vision-language space into a space that captures representations of facial actions. To train this projection head for subsequent zero-shot predictions, we propose to align the projected visual representations with task-specific semantic meanings derived from the LLM encoder, and the text instruction-based strategy is employed to customize the LLM knowledge. Given unlabelled facial data and efficient training of the projection head, Exp-CLIP achieves superior zero-shot results to the CLIP models and several other large vision-language models (LVLMs) on seven in-the-wild FER datasets. The code and pre-trained models are available at https://github.com/zengqunzhao/Exp-CLIP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17388v1">Can LLMs be Good Graph Judger for Knowledge Graph Construction?</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      In real-world scenarios, most of the data obtained from information retrieval (IR) system is unstructured. Converting natural language sentences into structured Knowledge Graphs (KGs) remains a critical challenge. The quality of constructed KGs may also impact the performance of some KG-dependent domains like GraphRAG systems and recommendation systems. Recently, Large Language Models (LLMs) have demonstrated impressive capabilities in addressing a wide range of natural language processing tasks. However, there are still challenges when utilizing LLMs to address the task of generating structured KGs. And we have identified three limitations with respect to existing KG construction methods. (1)There is a large amount of information and excessive noise in real-world documents, which could result in extracting messy information. (2)Native LLMs struggle to effectively extract accuracy knowledge from some domain-specific documents. (3)Hallucinations phenomenon cannot be overlooked when utilizing LLMs directly as an unsupervised method for constructing KGs. In this paper, we propose GraphJudger, a knowledge graph construction framework to address the aforementioned challenges. We introduce three innovative modules in our method, which are entity-centric iterative text denoising, knowledge aware instruction tuning and graph judgement, respectively. We seek to utilize the capacity of LLMs to function as a graph judger, a capability superior to their role only as a predictor for KG construction problems. Experiments conducted on two general text-graph pair datasets and one domain-specific text-graph pair dataset show superior performances compared to baseline methods. The code of our proposed method is available at https://github.com/hhy-huang/GraphJudger.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17375v1">The Extractive-Abstractive Spectrum: Uncovering Verifiability Trade-offs in LLM Generations</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Across all fields of academic study, experts cite their sources when sharing information. While large language models (LLMs) excel at synthesizing information, they do not provide reliable citation to sources, making it difficult to trace and verify the origins of the information they present. In contrast, search engines make sources readily accessible to users and place the burden of synthesizing information on the user. Through a survey, we find that users prefer search engines over LLMs for high-stakes queries, where concerns regarding information provenance outweigh the perceived utility of LLM responses. To examine the interplay between verifiability and utility of information-sharing tools, we introduce the extractive-abstractive spectrum, in which search engines and LLMs are extreme endpoints encapsulating multiple unexplored intermediate operating points. Search engines are extractive because they respond to queries with snippets of sources with links (citations) to the original webpages. LLMs are abstractive because they address queries with answers that synthesize and logically transform relevant information from training and in-context sources without reliable citation. We define five operating points that span the extractive-abstractive spectrum and conduct human evaluations on seven systems across four diverse query distributions that reflect real-world QA settings: web search, language simplification, multi-step reasoning, and medical advice. As outputs become more abstractive, we find that perceived utility improves by as much as 200%, while the proportion of properly cited sentences decreases by as much as 50% and users take up to 3 times as long to verify cited information. Our findings recommend distinct operating points for domain-specific LLM systems and our failure analysis informs approaches to high-utility LLM systems that empower users to verify information.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17338v1">Different Bias Under Different Criteria: Assessing Bias in LLMs with a Fact-Based Approach</a></div>
    <div class="paper-meta">
      📅 2024-11-26
      | 💬 Accepted in NeurIPS 2024 Workshop on Socially Responsible Language Modelling Research (SoLaR)
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often reflect real-world biases, leading to efforts to mitigate these effects and make the models unbiased. Achieving this goal requires defining clear criteria for an unbiased state, with any deviation from these criteria considered biased. Some studies define an unbiased state as equal treatment across diverse demographic groups, aiming for balanced outputs from LLMs. However, differing perspectives on equality and the importance of pluralism make it challenging to establish a universal standard. Alternatively, other approaches propose using fact-based criteria for more consistent and objective evaluations, though these methods have not yet been fully applied to LLM bias assessments. Thus, there is a need for a metric with objective criteria that offers a distinct perspective from equality-based approaches. Motivated by this need, we introduce a novel metric to assess bias using fact-based criteria and real-world statistics. In this paper, we conducted a human survey demonstrating that humans tend to perceive LLM outputs more positively when they align closely with real-world demographic distributions. Evaluating various LLMs with our proposed metric reveals that model bias varies depending on the criteria used, highlighting the need for multi-perspective assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17309v1">PIM-AI: A Novel Architecture for High-Efficiency LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-11-26
      | 💬 14 pages, 5 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become essential in a variety of applications due to their advanced language understanding and generation capabilities. However, their computational and memory requirements pose significant challenges to traditional hardware architectures. Processing-in-Memory (PIM), which integrates computational units directly into memory chips, offers several advantages for LLM inference, including reduced data transfer bottlenecks and improved power efficiency. This paper introduces PIM-AI, a novel DDR5/LPDDR5 PIM architecture designed for LLM inference without modifying the memory controller or DDR/LPDDR memory PHY. We have developed a simulator to evaluate the performance of PIM-AI in various scenarios and demonstrate its significant advantages over conventional architectures. In cloud-based scenarios, PIM-AI reduces the 3-year TCO per queries-per-second by up to 6.94x compared to state-of-the-art GPUs, depending on the LLM model used. In mobile scenarios, PIM-AI achieves a 10- to 20-fold reduction in energy per token compared to state-of-the-art mobile SoCs, resulting in 25 to 45~\% more queries per second and 6.9x to 13.4x less energy per query, extending battery life and enabling more inferences per charge. These results highlight PIM-AI's potential to revolutionize LLM deployments, making them more efficient, scalable, and sustainable.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17304v1">Meaningless is better: hashing bias-inducing words in LLM prompts improves performance in logical reasoning and statistical learning</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      This paper introduces a novel method, referred to as "hashing", which involves masking potentially bias-inducing words in large language models (LLMs) with hash-like meaningless identifiers to reduce cognitive biases and reliance on external knowledge. The method was tested across three sets of experiments involving a total of 490 prompts. Statistical analysis using chi-square tests showed significant improvements in all tested scenarios, which covered LLama, ChatGPT, Copilot, Gemini and Mixtral models. In the first experiment, hashing decreased the fallacy rate in a modified version of the "Linda" problem aimed at evaluating susceptibility to cognitive biases. In the second experiment, it improved LLM results on the frequent itemset extraction task. In the third experiment, we found hashing is also effective when the Linda problem is presented in a tabular format rather than text, indicating that the technique works across various input representations. Overall, the method was shown to improve bias reduction and incorporation of external knowledge. Despite bias reduction, hallucination rates were inconsistently reduced across types of LLM models. These findings suggest that masking bias-inducing terms can improve LLM performance, although its effectiveness is model- and task-dependent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17301v1">ER2Score: LLM-based Explainable and Customizable Metric for Assessing Radiology Reports with Reward-Control Loss</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Automated radiology report generation (R2Gen) has advanced significantly, introducing challenges in accurate evaluation due to its complexity. Traditional metrics often fall short by relying on rigid word-matching or focusing only on pathological entities, leading to inconsistencies with human assessments. To bridge this gap, we introduce ER2Score, an automatic evaluation metric designed specifically for R2Gen. Our metric utilizes a reward model, guided by our margin-based reward enforcement loss, along with a tailored training data design that enables customization of evaluation criteria to suit user-defined needs. It not only scores reports according to user-specified criteria but also provides detailed sub-scores, enhancing interpretability and allowing users to adjust the criteria between different aspects of reports. Leveraging GPT-4, we designed an easy-to-use data generation pipeline, enabling us to produce extensive training data based on two distinct scoring systems, each containing reports of varying quality along with corresponding scores. These GPT-generated reports are then paired as accepted and rejected samples through our pairing rule to train an LLM towards our fine-grained reward model, which assigns higher rewards to the report with high quality. Our reward-control loss enables this model to simultaneously output multiple individual rewards corresponding to the number of evaluation criteria, with their summation as our final ER2Score. Our experiments demonstrate ER2Score's heightened correlation with human judgments and superior performance in model selection compared to traditional metrics. Notably, our model provides both an overall score and individual scores for each evaluation item, enhancing interpretability. We also demonstrate its flexible training across various evaluation systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.15560v2">Do LLMs Agree on the Creativity Evaluation of Alternative Uses?</a></div>
    <div class="paper-meta">
      📅 2024-11-26
      | 💬 19 pages, 7 figures, 15 tables
    </div>
    <details class="paper-abstract">
      This paper investigates whether large language models (LLMs) show agreement in assessing creativity in responses to the Alternative Uses Test (AUT). While LLMs are increasingly used to evaluate creative content, previous studies have primarily focused on a single model assessing responses generated by the same model or humans. This paper explores whether LLMs can impartially and accurately evaluate creativity in outputs generated by both themselves and other models. Using an oracle benchmark set of AUT responses, categorized by creativity level (common, creative, and highly creative), we experiment with four state-of-the-art LLMs evaluating these outputs. We test both scoring and ranking methods and employ two evaluation settings (comprehensive and segmented) to examine if LLMs agree on the creativity evaluation of alternative uses. Results reveal high inter-model agreement, with Spearman correlations averaging above 0.7 across models and reaching over 0.77 with respect to the oracle, indicating a high level of agreement and validating the reliability of LLMs in creativity assessment of alternative uses. Notably, models do not favour their own responses, instead they provide similar creativity assessment scores or rankings for alternative uses generated by other models. These findings suggest that LLMs exhibit impartiality and high alignment in creativity evaluation, offering promising implications for their use in automated creativity assessment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.00231v2">LLM-RankFusion: Mitigating Intrinsic Inconsistency in LLM-based Ranking</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Ranking passages by prompting a large language model (LLM) can achieve promising performance in modern information retrieval (IR) systems. A common approach to sort the ranking list is by prompting LLMs for a pairwise or setwise comparison which often relies on sorting algorithms. However, sorting-based methods require consistent comparisons to correctly sort the passages, which we show that LLMs often violate. We identify two kinds of intrinsic inconsistency in LLM-based pairwise comparisons: order inconsistency which leads to conflicting results when switching the passage order, and transitive inconsistency which leads to non-transitive triads among all preference pairs. Our study of these inconsistencies is relevant for understanding and improving the stability of any ranking scheme based on relative preferences. In this paper, we propose LLM-RankFusion, an LLM-based ranking framework that mitigates these inconsistencies and produces a robust ranking list. LLM-RankFusion mitigates order inconsistency using in-context learning (ICL) to demonstrate order-agnostic comparisons and calibration to estimate the underlying preference probability between two passages. We then address transitive inconsistency by aggregating the ranking results from multiple rankers. In our experiments, we empirically show that LLM-RankFusion can significantly reduce inconsistent comparison results, improving the ranking quality by making the final ranking list more robust. Our code is available at \href{https://github.com/XHMY/LLM-RankFusion}{https://github.com/XHMY/LLM-RankFusion}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.04492v1">Socio-Emotional Response Generation: A Human Evaluation Protocol for LLM-Based Conversational Systems</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Conversational systems are now capable of producing impressive and generally relevant responses. However, we have no visibility nor control of the socio-emotional strategies behind state-of-the-art Large Language Models (LLMs), which poses a problem in terms of their transparency and thus their trustworthiness for critical applications. Another issue is that current automated metrics are not able to properly evaluate the quality of generated responses beyond the dataset's ground truth. In this paper, we propose a neural architecture that includes an intermediate step in planning socio-emotional strategies before response generation. We compare the performance of open-source baseline LLMs to the outputs of these same models augmented with our planning module. We also contrast the outputs obtained from automated metrics and evaluation results provided by human annotators. We describe a novel evaluation protocol that includes a coarse-grained consistency evaluation, as well as a finer-grained annotation of the responses on various social and emotional criteria. Our study shows that predicting a sequence of expected strategy labels and using this sequence to generate a response yields better results than a direct end-to-end generation scheme. It also highlights the divergences and the limits of current evaluation metrics for generated content. The code for the annotation platform and the annotated data are made publicly available for the evaluation of future models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.13439v2">Finding Blind Spots in Evaluator LLMs with Interpretable Checklists</a></div>
    <div class="paper-meta">
      📅 2024-11-26
      | 💬 EMNLP 2024
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly relied upon to evaluate text outputs of other LLMs, thereby influencing leaderboards and development decisions. However, concerns persist over the accuracy of these assessments and the potential for misleading conclusions. In this work, we investigate the effectiveness of LLMs as evaluators for text generation tasks. We propose FBI, a novel framework designed to examine the proficiency of Evaluator LLMs in assessing four critical abilities in other LLMs: factual accuracy, instruction following, coherence in long-form writing, and reasoning proficiency. By introducing targeted perturbations in answers generated by LLMs, that clearly impact one of these key capabilities, we test whether an Evaluator LLM can detect these quality drops. By creating a total of 2400 perturbed answers covering 22 perturbation categories, we conduct a comprehensive study using different evaluation strategies on five prominent LLMs commonly used as evaluators in the literature. Our findings reveal significant shortcomings in current Evaluator LLMs, which failed to identify quality drops in over 50\% of cases on average. Single-answer and pairwise evaluations demonstrated notable limitations, whereas reference-based evaluations showed comparatively better performance. These results underscore the unreliable nature of current Evaluator LLMs and advocate for cautious implementation in practical applications. Code and data are available at https://github.com/AI4Bharat/FBI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17135v1">LLM-Based Offline Learning for Embodied Agents via Consistency-Guided Reward Ensemble</a></div>
    <div class="paper-meta">
      📅 2024-11-26
      | 💬 Findings of EMNLP-2024 Camera Ready Version
    </div>
    <details class="paper-abstract">
      Employing large language models (LLMs) to enable embodied agents has become popular, yet it presents several limitations in practice. In this work, rather than using LLMs directly as agents, we explore their use as tools for embodied agent learning. Specifically, to train separate agents via offline reinforcement learning (RL), an LLM is used to provide dense reward feedback on individual actions in training datasets. In doing so, we present a consistency-guided reward ensemble framework (CoREN), designed for tackling difficulties in grounding LLM-generated estimates to the target environment domain. The framework employs an adaptive ensemble of spatio-temporally consistent rewards to derive domain-grounded rewards in the training datasets, thus enabling effective offline learning of embodied agents in different environment domains. Experiments with the VirtualHome benchmark demonstrate that CoREN significantly outperforms other offline RL agents, and it also achieves comparable performance to state-of-the-art LLM-based agents with 8B parameters, despite CoREN having only 117M parameters for the agent policy network and using LLMs only for training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17116v1">Star Attention: Efficient LLM Inference over Long Sequences</a></div>
    <div class="paper-meta">
      📅 2024-11-26
      | 💬 Code: https://github.com/NVIDIA/Star-Attention
    </div>
    <details class="paper-abstract">
      Inference with Transformer-based Large Language Models (LLMs) on long sequences is both costly and slow due to the quadratic complexity of the self-attention mechanism. We introduce Star Attention, a two-phase block-sparse approximation that improves computational efficiency by sharding attention across multiple hosts while minimizing communication overhead. In the first phase, the context is processed using blockwise-local attention across hosts, in parallel. In the second phase, query and response tokens attend to all prior cached tokens through sequence-global attention. Star Attention integrates seamlessly with most Transformer-based LLMs trained with global attention, reducing memory requirements and inference time by up to 11x while preserving 95-100% of accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17102v1">Scholar Name Disambiguation with Search-enhanced LLM Across Language</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      The task of scholar name disambiguation is crucial in various real-world scenarios, including bibliometric-based candidate evaluation for awards, application material anti-fraud measures, and more. Despite significant advancements, current methods face limitations due to the complexity of heterogeneous data, often necessitating extensive human intervention. This paper proposes a novel approach by leveraging search-enhanced language models across multiple languages to improve name disambiguation. By utilizing the powerful query rewriting, intent recognition, and data indexing capabilities of search engines, our method can gather richer information for distinguishing between entities and extracting profiles, resulting in a more comprehensive data dimension. Given the strong cross-language capabilities of large language models(LLMs), optimizing enhanced retrieval methods with this technology offers substantial potential for high-efficiency information retrieval and utilization. Our experiments demonstrate that incorporating local languages significantly enhances disambiguation performance, particularly for scholars from diverse geographic regions. This multi-lingual, search-enhanced methodology offers a promising direction for more efficient and accurate active scholar name disambiguation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.17089v1">Efficient LLM Inference with I/O-Aware Partial KV Cache Recomputation</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Inference for Large Language Models (LLMs) is computationally demanding. To reduce the cost of auto-regressive decoding, Key-Value (KV) caching is used to store intermediate activations, enabling GPUs to perform only the incremental computation required for each new token. This approach significantly lowers the computational overhead for token generation. However, the memory required for KV caching grows rapidly, often exceeding the capacity of GPU memory. A cost-effective alternative is to offload KV cache to CPU memory, which alleviates GPU memory pressure but shifts the bottleneck to the limited bandwidth of the PCIe connection between the CPU and GPU. Existing methods attempt to address these issues by overlapping GPU computation with I/O or employing CPU-GPU heterogeneous execution, but they are hindered by excessive data movement and dependence on CPU capabilities. In this paper, we introduce an efficient CPU-GPU I/O-aware LLM inference method that avoids transferring the entire KV cache from CPU to GPU by recomputing partial KV cache from activations while concurrently transferring the remaining KV cache via PCIe bus. This approach overlaps GPU recomputation with data transfer to minimize idle GPU time and maximize inference performance. Our method is fully automated by integrating a profiler module that utilizes input characteristics and system hardware information, a scheduler module to optimize the distribution of computation and communication workloads, and a runtime module to efficiently execute the derived execution plan. Experimental results show that our method achieves up to 35.8% lower latency and 46.2% higher throughput during decoding compared to state-of-the-art approaches.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.13789v2">LEADRE: Multi-Faceted Knowledge Enhanced LLM Empowered Display Advertisement Recommender System</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Display advertising provides significant value to advertisers, publishers, and users. Traditional display advertising systems utilize a multi-stage architecture consisting of retrieval, coarse ranking, and final ranking. However, conventional retrieval methods rely on ID-based learning to rank mechanisms and fail to adequately utilize the content information of ads, which hampers their ability to provide diverse recommendation lists. To address this limitation, we propose leveraging the extensive world knowledge of LLMs. However, three key challenges arise when attempting to maximize the effectiveness of LLMs: "How to capture user interests", "How to bridge the knowledge gap between LLMs and advertising system", and "How to efficiently deploy LLMs". To overcome these challenges, we introduce a novel LLM-based framework called LLM Empowered Display ADvertisement REcommender system (LEADRE). LEADRE consists of three core modules: (1) The Intent-Aware Prompt Engineering introduces multi-faceted knowledge and designs intent-aware <Prompt, Response> pairs that fine-tune LLMs to generate ads tailored to users' personal interests. (2) The Advertising-Specific Knowledge Alignment incorporates auxiliary fine-tuning tasks and Direct Preference Optimization (DPO) to align LLMs with ad semantic and business value. (3) The Efficient System Deployment deploys LEADRE in an online environment by integrating both latency-tolerant and latency-sensitive service. Extensive offline experiments demonstrate the effectiveness of LEADRE and validate the contributions of individual modules. Online A/B test shows that LEADRE leads to a 1.57% and 1.17% GMV lift for serviced users on WeChat Channels and Moments separately. LEADRE has been deployed on both platforms, serving tens of billions of requests each day.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2406.00231v2">LLM-RankFusion: Mitigating Intrinsic Inconsistency in LLM-based Ranking</a></div>
    <div class="paper-meta">
      📅 2024-11-26
    </div>
    <details class="paper-abstract">
      Ranking passages by prompting a large language model (LLM) can achieve promising performance in modern information retrieval (IR) systems. A common approach to sort the ranking list is by prompting LLMs for a pairwise or setwise comparison which often relies on sorting algorithms. However, sorting-based methods require consistent comparisons to correctly sort the passages, which we show that LLMs often violate. We identify two kinds of intrinsic inconsistency in LLM-based pairwise comparisons: order inconsistency which leads to conflicting results when switching the passage order, and transitive inconsistency which leads to non-transitive triads among all preference pairs. Our study of these inconsistencies is relevant for understanding and improving the stability of any ranking scheme based on relative preferences. In this paper, we propose LLM-RankFusion, an LLM-based ranking framework that mitigates these inconsistencies and produces a robust ranking list. LLM-RankFusion mitigates order inconsistency using in-context learning (ICL) to demonstrate order-agnostic comparisons and calibration to estimate the underlying preference probability between two passages. We then address transitive inconsistency by aggregating the ranking results from multiple rankers. In our experiments, we empirically show that LLM-RankFusion can significantly reduce inconsistent comparison results, improving the ranking quality by making the final ranking list more robust. Our code is available at \href{https://github.com/XHMY/LLM-RankFusion}{https://github.com/XHMY/LLM-RankFusion}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2411.14484v1">Robust Planning with Compound LLM Architectures: An LLM-Modulo Approach</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      Previous work has attempted to boost Large Language Model (LLM) performance on planning and scheduling tasks through a variety of prompt engineering techniques. While these methods can work within the distributions tested, they are neither robust nor predictable. This limitation can be addressed through compound LLM architectures where LLMs work in conjunction with other components to ensure reliability. In this paper, we present a technical evaluation of a compound LLM architecture--the LLM-Modulo framework. In this framework, an LLM is paired with a complete set of sound verifiers that validate its output, re-prompting it if it fails. This approach ensures that the system can never output any fallacious output, and therefore that every output generated is guaranteed correct--something previous techniques have not been able to claim. Our results, evaluated across four scheduling domains, demonstrate significant performance gains with the LLM-Modulo framework using various models. Additionally, we explore modifications to the base configuration of the framework and assess their impact on overall system performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2411.14971v1">Leveraging LLMs for Legacy Code Modernization: Challenges and Opportunities for LLM-Generated Documentation</a></div>
    <div class="paper-meta">
      📅 2024-11-25
      | 💬 Abbreviated version submitted to LLM4Code 2025 (a workshop co-located with ICSE 2025), 13 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Legacy software systems, written in outdated languages like MUMPS and mainframe assembly, pose challenges in efficiency, maintenance, staffing, and security. While LLMs offer promise for modernizing these systems, their ability to understand legacy languages is largely unknown. This paper investigates the utilization of LLMs to generate documentation for legacy code using two datasets: an electronic health records (EHR) system in MUMPS and open-source applications in IBM mainframe Assembly Language Code (ALC). We propose a prompting strategy for generating line-wise code comments and a rubric to evaluate their completeness, readability, usefulness, and hallucination. Our study assesses the correlation between human evaluations and automated metrics, such as code complexity and reference-based metrics. We find that LLM-generated comments for MUMPS and ALC are generally hallucination-free, complete, readable, and useful compared to ground-truth comments, though ALC poses challenges. However, no automated metrics strongly correlate with comment quality to predict or measure LLM performance. Our findings highlight the limitations of current automated measures and the need for better evaluation metrics for LLM-generated documentation in legacy systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16932v1">Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      Temporal awareness is essential for video large language models (LLMs) to understand and reason about events within long videos, enabling applications like dense video captioning and temporal video grounding in a unified system. However, the scarcity of long videos with detailed captions and precise temporal annotations limits their temporal awareness. In this paper, we propose Seq2Time, a data-oriented training paradigm that leverages sequences of images and short video clips to enhance temporal awareness in long videos. By converting sequence positions into temporal annotations, we transform large-scale image and clip captioning datasets into sequences that mimic the temporal structure of long videos, enabling self-supervised training with abundant time-sensitive data. To enable sequence-to-time knowledge transfer, we introduce a novel time representation that unifies positional information across image sequences, clip sequences, and long videos. Experiments demonstrate the effectiveness of our method, achieving a 27.6% improvement in F1 score and 44.8% in CIDEr on the YouCook2 benchmark and a 14.7% increase in recall on the Charades-STA benchmark compared to the baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2407.10827v2">LLM Circuit Analyses Are Consistent Across Training and Scale</a></div>
    <div class="paper-meta">
      📅 2024-11-25
      | 💬 NeurIPS 2024
    </div>
    <details class="paper-abstract">
      Most currently deployed large language models (LLMs) undergo continuous training or additional finetuning. By contrast, most research into LLMs' internal mechanisms focuses on models at one snapshot in time (the end of pre-training), raising the question of whether their results generalize to real-world settings. Existing studies of mechanisms over time focus on encoder-only or toy models, which differ significantly from most deployed models. In this study, we track how model mechanisms, operationalized as circuits, emerge and evolve across 300 billion tokens of training in decoder-only LLMs, in models ranging from 70 million to 2.8 billion parameters. We find that task abilities and the functional components that support them emerge consistently at similar token counts across scale. Moreover, although such components may be implemented by different attention heads over time, the overarching algorithm that they implement remains. Surprisingly, both these algorithms and the types of components involved therein can replicate across model scale. These results suggest that circuit analyses conducted on small models at the end of pre-training can provide insights that still apply after additional pre-training and over model scale.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16863v1">Augmenting Multimodal LLMs with Self-Reflective Tokens for Knowledge-based Visual Question Answering</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      Multimodal LLMs (MLLMs) are the natural extension of large language models to handle multimodal inputs, combining text and image data. They have recently garnered attention due to their capability to address complex tasks involving both modalities. However, their effectiveness is limited to the knowledge acquired during training, which restricts their practical utility. In this work, we introduce a novel method to enhance the adaptability of MLLMs by integrating external knowledge sources. Our proposed model, Reflective LLaVA (ReflectiVA), utilizes reflective tokens to dynamically determine the need for external knowledge and predict the relevance of information retrieved from an external database. Tokens are trained following a two-stage two-model training recipe. This ultimately enables the MLLM to manage external knowledge while preserving fluency and performance on tasks where external knowledge is not needed. Through our experiments, we demonstrate the efficacy of ReflectiVA for knowledge-based visual question answering, highlighting its superior performance compared to existing methods. Source code and trained models are publicly available at https://github.com/aimagelab/ReflectiVA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.08509v2">Efficient Interactive LLM Serving with Proxy Model-based Sequence Length Prediction</a></div>
    <div class="paper-meta">
      📅 2024-11-25
      | 💬 Accepted at AIOps'24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been driving a new wave of interactive AI applications across numerous domains. However, efficiently serving LLM inference requests is challenging due to their unpredictable execution times originating from the autoregressive nature of generative models. Existing LLM serving systems exploit first-come-first-serve (FCFS) scheduling, suffering from head-of-line blocking issues. To address the non-deterministic nature of LLMs and enable efficient interactive LLM serving, we present a speculative shortest-job-first (SSJF) scheduler that uses a light proxy model to predict LLM output sequence lengths. Our open-source SSJF implementation does not require changes to memory management or batching strategies. Evaluations on real-world datasets and production workload traces show that SSJF reduces average job completion times by 30.5-39.6% and increases throughput by 2.2-3.6x compared to FCFS schedulers, across no batching, dynamic batching, and continuous batching settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16579v1">Enhancing LLM Reasoning via Critique Models with Test-Time and Training-Time Supervision</a></div>
    <div class="paper-meta">
      📅 2024-11-25
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Training large language models (LLMs) to spend more time thinking and reflection before responding is crucial for effectively solving complex reasoning tasks in fields such as science, coding, and mathematics. However, the effectiveness of mechanisms like self-reflection and self-correction depends on the model's capacity to accurately assess its own performance, which can be limited by factors such as initial accuracy, question difficulty, and the lack of external feedback. In this paper, we delve into a two-player paradigm that separates the roles of reasoning and critique models, where the critique model provides step-level feedback to supervise the reasoning (actor) model during both test-time and train-time. We first propose AutoMathCritique, an automated and scalable framework for collecting critique data, resulting in a dataset of $76,321$ responses paired with step-level feedback. Fine-tuning language models with this dataset enables them to generate natural language feedback for mathematical reasoning. We demonstrate that the critique models consistently improve the actor's performance on difficult queries at test-time, especially when scaling up inference-time computation. Motivated by these findings, we introduce the critique-based supervision to the actor's self-training process, and propose a critique-in-the-loop self-improvement method. Experiments show that the method improves the actor's exploration efficiency and solution diversity, especially on challenging queries, leading to a stronger reasoning model. Lastly, we take the preliminary step to explore training self-talk reasoning models via critique supervision and showcase its potential. Our code and datasets are at \href{https://mathcritique.github.io/}{https://mathcritique.github.io/}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16818v1">Enhancing In-Hospital Mortality Prediction Using Multi-Representational Learning with LLM-Generated Expert Summaries</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      In-hospital mortality (IHM) prediction for ICU patients is critical for timely interventions and efficient resource allocation. While structured physiological data provides quantitative insights, clinical notes offer unstructured, context-rich narratives. This study integrates these modalities with Large Language Model (LLM)-generated expert summaries to improve IHM prediction accuracy. Using the MIMIC-III database, we analyzed time-series physiological data and clinical notes from the first 48 hours of ICU admission. Clinical notes were concatenated chronologically for each patient and transformed into expert summaries using Med42-v2 70B. A multi-representational learning framework was developed to integrate these data sources, leveraging LLMs to enhance textual data while mitigating direct reliance on LLM predictions, which can introduce challenges in uncertainty quantification and interpretability. The proposed model achieved an AUPRC of 0.6156 (+36.41%) and an AUROC of 0.8955 (+7.64%) compared to a time-series-only baseline. Expert summaries outperformed clinical notes or time-series data alone, demonstrating the value of LLM-generated knowledge. Performance gains were consistent across demographic groups, with notable improvements in underrepresented populations, underscoring the framework's equitable application potential. By integrating LLM-generated summaries with structured and unstructured data, the framework captures complementary patient information, significantly improving predictive performance. This approach showcases the potential of LLMs to augment critical care prediction models, emphasizing the need for domain-specific validation and advanced integration strategies for broader clinical adoption.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00061v1">Speculative Decoding with CTC-based Draft Model for LLM Inference Acceleration</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      Inference acceleration of large language models (LLMs) has been put forward in many application scenarios and speculative decoding has shown its advantage in addressing inference acceleration. Speculative decoding usually introduces a draft model to assist the base LLM where the draft model produces drafts and the base LLM verifies the draft for acceptance or rejection. In this framework, the final inference speed is decided by the decoding speed of the draft model and the acceptance rate of the draft provided by the draft model. Currently the widely used draft models usually generate draft tokens for the next several positions in a non-autoregressive way without considering the correlations between draft tokens. Therefore, it has a high decoding speed but an unsatisfactory acceptance rate. In this paper, we focus on how to improve the performance of the draft model and aim to accelerate inference via a high acceptance rate. To this end, we propose a CTC-based draft model which strengthens the correlations between draft tokens during the draft phase, thereby generating higher-quality draft candidate sequences. Experiment results show that compared to strong baselines, the proposed method can achieve a higher acceptance rate and hence a faster inference speed.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16238v1">UVLLM: An Automated Universal RTL Verification Framework using LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      Verifying hardware designs in embedded systems is crucial but often labor-intensive and time-consuming. While existing solutions have improved automation, they frequently rely on unrealistic assumptions. To address these challenges, we introduce a novel framework, UVLLM, which combines Large Language Models (LLMs) with the Universal Verification Methodology (UVM) to relax these assumptions. UVLLM significantly enhances the automation of testing and repairing error-prone Register Transfer Level (RTL) codes, a critical aspect of verification development. Unlike existing methods, UVLLM ensures that all errors are triggered during verification, achieving a syntax error fix rate of 86.99% and a functional error fix rate of 71.92% on our proposed benchmark. These results demonstrate a substantial improvement in verification efficiency. Additionally, our study highlights the current limitations of LLM applications, particularly their reliance on extensive training data. We emphasize the transformative potential of LLMs in hardware design verification and suggest promising directions for future research in AI-driven hardware design methodologies. The Repo. of dataset and code: https://anonymous.4open.science/r/UVLLM/.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16791v1">What can LLM tell us about cities?</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      This study explores the capabilities of large language models (LLMs) in providing knowledge about cities and regions on a global scale. We employ two methods: directly querying the LLM for target variable values and extracting explicit and implicit features from the LLM correlated with the target variable. Our experiments reveal that LLMs embed a broad but varying degree of knowledge across global cities, with ML models trained on LLM-derived features consistently leading to improved predictive accuracy. Additionally, we observe that LLMs demonstrate a certain level of knowledge across global cities on all continents, but it is evident when they lack knowledge, as they tend to generate generic or random outputs for unfamiliar tasks. These findings suggest that LLMs can offer new opportunities for data-driven decision-making in the study of cities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16189v1">Enhancing Multi-Agent Consensus through Third-Party LLM Integration: Analyzing Uncertainty and Mitigating Hallucinations in Large Language Models</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) still face challenges when dealing with complex reasoning tasks, often resulting in hallucinations, which limit the practical application of LLMs. To alleviate this issue, this paper proposes a new method that integrates different LLMs to expand the knowledge boundary, reduce dependence on a single model, and promote in-depth debate among agents. The main contributions include: 1) Introducing third-party LLMs to adjust the attention weights of agents through uncertainty estimation and confidence analysis, optimizing consensus formation in multi-agent systems; 2) Experiments on arithmetic datasets have validated the effectiveness of the method, surpassing traditional multi-agent baselines. This research provides a new perspective for large models to alleviate hallucination phenomena when dealing with complex tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.07745v2">StepTool: A Step-grained Reinforcement Learning Framework for Tool Learning in LLMs</a></div>
    <div class="paper-meta">
      📅 2024-11-25
      | 💬 Ongoning Work
    </div>
    <details class="paper-abstract">
      Despite having powerful reasoning and inference capabilities, Large Language Models (LLMs) still need external tools to acquire real-time information retrieval or domain-specific expertise to solve complex tasks, which is referred to as tool learning. Existing tool learning methods primarily rely on tuning with expert trajectories, focusing on token-sequence learning from a linguistic perspective. However, there are several challenges: 1) imitating static trajectories limits their ability to generalize to new tasks. 2) even expert trajectories can be suboptimal, and better solution paths may exist. In this work, we introduce StepTool, a novel step-grained reinforcement learning framework to improve tool learning in LLMs. It consists of two components: Step-grained Reward Shaping, which assigns rewards at each tool interaction based on tool invocation success and its contribution to the task, and Step-grained Optimization, which uses policy gradient methods to optimize the model in a multi-step manner. Experimental results demonstrate that StepTool significantly outperforms existing methods in multi-step, tool-based tasks, providing a robust solution for complex task environments. Codes are available at https://github.com/yuyq18/StepTool.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.16158v1">MixPE: Quantization and Hardware Co-design for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2024-11-25
    </div>
    <details class="paper-abstract">
      Transformer-based large language models (LLMs) have achieved remarkable success as model sizes continue to grow, yet their deployment remains challenging due to significant computational and memory demands. Quantization has emerged as a promising solution, and state-of-the-art quantization algorithms for LLMs introduce the need for mixed-precision matrix multiplication (mpGEMM), where lower-precision weights are multiplied with higher-precision activations. Despite its benefits, current hardware accelerators such as GPUs and TPUs lack native support for efficient mpGEMM, leading to inefficient dequantization operations in the main sequential loop. To address this limitation, we introduce MixPE, a specialized mixed-precision processing element designed for efficient low-bit quantization in LLM inference. MixPE leverages two key innovations to minimize dequantization overhead and unlock the full potential of low-bit quantization. First, recognizing that scale and zero point are shared within each quantization group, we propose performing dequantization after per-group mpGEMM, significantly reducing dequantization overhead. Second, instead of relying on conventional multipliers, MixPE utilizes efficient shift\&add operations for multiplication, optimizing both computation and energy efficiency. Our experimental results demonstrate that MixPE surpasses the state-of-the-art quantization accelerators by $2.6\times$ speedup and $1.4\times$ energy reduction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.07781v1">Can LLMs faithfully generate their layperson-understandable 'self'?: A Case Study in High-Stakes Domains</a></div>
    <div class="paper-meta">
      📅 2024-11-25
      | 💬 35 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly impacted nearly every domain of human knowledge. However, the explainability of these models esp. to laypersons, which are crucial for instilling trust, have been examined through various skeptical lenses. In this paper, we introduce a novel notion of LLM explainability to laypersons, termed $\textit{ReQuesting}$, across three high-priority application domains -- law, health and finance, using multiple state-of-the-art LLMs. The proposed notion exhibits faithful generation of explainable layman-understandable algorithms on multiple tasks through high degree of reproducibility. Furthermore, we observe a notable alignment of the explainable algorithms with intrinsic reasoning of the LLMs.
    </details>
</div>
