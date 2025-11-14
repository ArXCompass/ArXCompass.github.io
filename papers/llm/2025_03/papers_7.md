# llm - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- Part 7
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16833v3">KG4Diagnosis: A Hierarchical Multi-Agent LLM Framework with Knowledge Graph Enhancement for Medical Diagnosis</a></div>
    <div class="paper-meta">
      📅 2025-03-13
      | 💬 10 pages,5 figures,published to AAAI-25 Bridge Program
    </div>
    <details class="paper-abstract">
      Integrating Large Language Models (LLMs) in healthcare diagnosis demands systematic frameworks that can handle complex medical scenarios while maintaining specialized expertise. We present KG4Diagnosis, a novel hierarchical multi-agent framework that combines LLMs with automated knowledge graph construction, encompassing 362 common diseases across medical specialties. Our framework mirrors real-world medical systems through a two-tier architecture: a general practitioner (GP) agent for initial assessment and triage, coordinating with specialized agents for in-depth diagnosis in specific domains. The core innovation lies in our end-to-end knowledge graph generation methodology, incorporating: (1) semantic-driven entity and relation extraction optimized for medical terminology, (2) multi-dimensional decision relationship reconstruction from unstructured medical texts, and (3) human-guided reasoning for knowledge expansion. KG4Diagnosis serves as an extensible foundation for specialized medical diagnosis systems, with capabilities to incorporate new diseases and medical knowledge. The framework's modular design enables seamless integration of domain-specific enhancements, making it valuable for developing targeted medical diagnosis systems. We provide architectural guidelines and protocols to facilitate adoption across medical contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09994v1">TIME: Temporal-sensitive Multi-dimensional Instruction Tuning and Benchmarking for Video-LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Video large language models have achieved remarkable performance in tasks such as video question answering, however, their temporal understanding remains suboptimal. To address this limitation, we curate a dedicated instruction fine-tuning dataset that focuses on enhancing temporal comprehension across five key dimensions. In order to reduce reliance on costly temporal annotations, we introduce a multi-task prompt fine-tuning approach that seamlessly integrates temporal-sensitive tasks into existing instruction datasets without requiring additional annotations. Furthermore, we develop a novel benchmark for temporal-sensitive video understanding that not only fills the gaps in dimension coverage left by existing benchmarks but also rigorously filters out potential shortcuts, ensuring a more accurate evaluation. Extensive experimental results demonstrate that our approach significantly enhances the temporal understanding of video-LLMs while avoiding reliance on shortcuts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2312.10321v4">LLM-SQL-Solver: Can LLMs Determine SQL Equivalence?</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      Judging the equivalence between two SQL queries is a fundamental problem with many practical applications in data management and SQL generation (i.e., evaluating the quality of generated SQL queries in text-to-SQL task). While the research community has reasoned about SQL equivalence for decades, it poses considerable difficulties and no complete solutions exist. Recently, Large Language Models (LLMs) have shown strong reasoning capability in conversation, question answering and solving mathematics challenges. In this paper, we study if LLMs can be used to determine the equivalence between SQL queries under two notions of SQL equivalence (semantic equivalence and relaxed equivalence). To assist LLMs in generating high quality responses, we present two prompting techniques: Miniature & Mull and Explain & Compare. The former technique is used to evaluate the semantic equivalence in which it asks LLMs to execute a query on a simple database instance and then explore if a counterexample exists by modifying the database. The latter technique is used to evaluate the relaxed equivalence in which it asks LLMs to explain the queries and then compare if they contain significant logical differences. Our experiments demonstrate using our techniques, LLMs is a promising tool to help data engineers in writing semantically equivalent SQL queries, however challenges still persist, and is a better metric for evaluating SQL generation than the popular execution accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2503.10941v1">Graph-Grounded LLMs: Leveraging Graphical Function Calling to Minimize LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-03-13
    </div>
    <details class="paper-abstract">
      The adoption of Large Language Models (LLMs) is rapidly expanding across various tasks that involve inherent graphical structures. Graphs are integral to a wide range of applications, including motion planning for autonomous vehicles, social networks, scene understanding, and knowledge graphs. Many problems, even those not initially perceived as graph-based, can be effectively addressed through graph theory. However, when applied to these tasks, LLMs often encounter challenges, such as hallucinations and mathematical inaccuracies. To overcome these limitations, we propose Graph-Grounded LLMs, a system that improves LLM performance on graph-related tasks by integrating a graph library through function calls. By grounding LLMs in this manner, we demonstrate significant reductions in hallucinations and improved mathematical accuracy in solving graph-based problems, as evidenced by the performance on the NLGraph benchmark. Finally, we showcase a disaster rescue application where the Graph-Grounded LLM acts as a decision-support system.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10688v1">CULEMO: Cultural Lenses on Emotion -- Benchmarking LLMs for Cross-Cultural Emotion Understanding</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      NLP research has increasingly focused on subjective tasks such as emotion analysis. However, existing emotion benchmarks suffer from two major shortcomings: (1) they largely rely on keyword-based emotion recognition, overlooking crucial cultural dimensions required for deeper emotion understanding, and (2) many are created by translating English-annotated data into other languages, leading to potentially unreliable evaluation. To address these issues, we introduce Cultural Lenses on Emotion (CuLEmo), the first benchmark designed to evaluate culture-aware emotion prediction across six languages: Amharic, Arabic, English, German, Hindi, and Spanish. CuLEmo comprises 400 crafted questions per language, each requiring nuanced cultural reasoning and understanding. We use this benchmark to evaluate several state-of-the-art LLMs on culture-aware emotion prediction and sentiment analysis tasks. Our findings reveal that (1) emotion conceptualizations vary significantly across languages and cultures, (2) LLMs performance likewise varies by language and cultural context, and (3) prompting in English with explicit country context often outperforms in-language prompts for culture-aware emotion and sentiment understanding. We hope this benchmark guides future research toward developing more culturally aligned NLP systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11702v1">Toward a method for LLM-enabled Indoor Navigation</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 7 pages, 3 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Indoor navigation presents unique challenges due to complex layouts, lack of GPS signals, and accessibility concerns. Existing solutions often struggle with real-time adaptability and user-specific needs. In this work, we explore the potential of a Large Language Model (LLM), i.e., ChatGPT, to generate natural, context-aware navigation instructions from indoor map images. We design and evaluate test cases across different real-world environments, analyzing the effectiveness of LLMs in interpreting spatial layouts, handling user constraints, and planning efficient routes. Our findings demonstrate the potential of LLMs for supporting personalized indoor navigation, with an average of 52% correct indications and a maximum of 62%. The results do not appear to depend on the complexity of the layout or the complexity of the expected path, but rather on the number of points of interest and the abundance of visual information, which negatively affect the performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09598v1">How to Protect Yourself from 5G Radiation? Investigating LLM Responses to Implicit Misinformation</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) are widely deployed in diverse scenarios, the extent to which they could tacitly spread misinformation emerges as a critical safety concern. Current research primarily evaluates LLMs on explicit false statements, overlooking how misinformation often manifests subtly as unchallenged premises in real-world user interactions. We curated ECHOMIST, the first comprehensive benchmark for implicit misinformation, where the misinformed assumptions are embedded in a user query to LLMs. ECHOMIST is based on rigorous selection criteria and carefully curated data from diverse sources, including real-world human-AI conversations and social media interactions. We also introduce a new evaluation metric to measure whether LLMs can recognize and counter false information rather than amplify users' misconceptions. Through an extensive empirical study on a wide range of LLMs, including GPT-4, Claude, and Llama, we find that current models perform alarmingly poorly on this task, often failing to detect false premises and generating misleading explanations. Our findings underscore the critical need for an increased focus on implicit misinformation in LLM safety research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09579v1">Cost-Optimal Grouped-Query Attention for Long-Context LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 16 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Building effective and efficient Transformer-based large language models (LLMs) has recently become a research focus, requiring maximizing model language capabilities and minimizing training and deployment costs. Existing efforts have primarily described complex relationships among model performance, parameter size, and data size, as well as searched for the optimal compute allocation to train LLMs. However, they overlook the impacts of context length and attention head configuration (the number of query and key-value heads in grouped-query attention) on training and inference. In this paper, we systematically compare models with different parameter sizes, context lengths, and attention head configurations in terms of model performance, computational cost, and memory cost. Then, we extend the existing scaling methods, which are based solely on parameter size and training compute, to guide the construction of cost-optimal LLMs during both training and inference. Our quantitative scaling studies show that, when processing sufficiently long sequences, a larger model with fewer attention heads can achieve a lower loss while incurring lower computational and memory costs. Our findings provide valuable insights for developing practical LLMs, especially in long-context processing scenarios. We will publicly release our code and data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09516v1">Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 16 pages
    </div>
    <details class="paper-abstract">
      Efficiently acquiring external knowledge and up-to-date information is essential for effective reasoning and text generation in large language models (LLMs). Retrieval augmentation and tool-use training approaches where a search engine is treated as a tool lack complex multi-turn retrieval flexibility or require large-scale supervised data. Prompting advanced LLMs with reasoning capabilities during inference to use search engines is not optimal, since the LLM does not learn how to optimally interact with the search engine. This paper introduces Search-R1, an extension of the DeepSeek-R1 model where the LLM learns -- solely through reinforcement learning (RL) -- to autonomously generate (multiple) search queries during step-by-step reasoning with real-time retrieval. Search-R1 optimizes LLM rollouts with multi-turn search interactions, leveraging retrieved token masking for stable RL training and a simple outcome-based reward function. Experiments on seven question-answering datasets show that Search-R1 improves performance by 26% (Qwen2.5-7B), 21% (Qwen2.5-3B), and 10% (LLaMA3.2-3B) over SOTA baselines. This paper further provides empirical insights into RL optimization methods, LLM choices, and response length dynamics in retrieval-augmented reasoning. The code and model checkpoints are available at https://github.com/PeterGriffinJin/Search-R1.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09501v1">ReMA: Learning to Meta-think for LLMs with Multi-Agent Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Recent research on Reasoning of Large Language Models (LLMs) has sought to further enhance their performance by integrating meta-thinking -- enabling models to monitor, evaluate, and control their reasoning processes for more adaptive and effective problem-solving. However, current single-agent work lacks a specialized design for acquiring meta-thinking, resulting in low efficacy. To address this challenge, we introduce Reinforced Meta-thinking Agents (ReMA), a novel framework that leverages Multi-Agent Reinforcement Learning (MARL) to elicit meta-thinking behaviors, encouraging LLMs to think about thinking. ReMA decouples the reasoning process into two hierarchical agents: a high-level meta-thinking agent responsible for generating strategic oversight and plans, and a low-level reasoning agent for detailed executions. Through iterative reinforcement learning with aligned objectives, these agents explore and learn collaboration, leading to improved generalization and robustness. Experimental results demonstrate that ReMA outperforms single-agent RL baselines on complex reasoning tasks, including competitive-level mathematical benchmarks and LLM-as-a-Judge benchmarks. Comprehensive ablation studies further illustrate the evolving dynamics of each distinct agent, providing valuable insights into how the meta-thinking reasoning process enhances the reasoning capabilities of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09454v1">Explicit Learning and the LLM in Machine Translation</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      This study explores the capacity of large language models (LLMs) for explicit learning, a process involving the assimilation of metalinguistic explanations to carry out language tasks. Using constructed languages generated by cryptographic means as controlled test environments, we designed experiments to assess an LLM's ability to explicitly learn and apply grammar rules. Our results demonstrate that while LLMs possess a measurable capacity for explicit learning, this ability diminishes as the complexity of the linguistic phenomena at hand increases. Supervised fine-tuning on chains of thought significantly enhances LLM performance but struggles to generalize to typologically novel or more complex linguistic features. These findings point to the need for more diverse training sets and alternative fine-tuning strategies to further improve explicit learning by LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09433v1">CASTLE: Benchmarking Dataset for Static Code Analyzers and LLMs towards CWE Detection</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Identifying vulnerabilities in source code is crucial, especially in critical software components. Existing methods such as static analysis, dynamic analysis, formal verification, and recently Large Language Models are widely used to detect security flaws. This paper introduces CASTLE (CWE Automated Security Testing and Low-Level Evaluation), a benchmarking framework for evaluating the vulnerability detection capabilities of different methods. We assess 13 static analysis tools, 10 LLMs, and 2 formal verification tools using a hand-crafted dataset of 250 micro-benchmark programs covering 25 common CWEs. We propose the CASTLE Score, a novel evaluation metric to ensure fair comparison. Our results reveal key differences: ESBMC (a formal verification tool) minimizes false positives but struggles with vulnerabilities beyond model checking, such as weak cryptography or SQL injection. Static analyzers suffer from high false positives, increasing manual validation efforts for developers. LLMs perform exceptionally well in the CASTLE dataset when identifying vulnerabilities in small code snippets. However, their accuracy declines, and hallucinations increase as the code size grows. These results suggest that LLMs could play a pivotal role in future security solutions, particularly within code completion frameworks, where they can provide real-time guidance to prevent vulnerabilities. The dataset is accessible at https://github.com/CASTLE-Benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09407v1">Got Compute, but No Data: Lessons From Post-training a Finnish LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 7 pages
    </div>
    <details class="paper-abstract">
      As LLMs gain more popularity as chatbots and general assistants, methods have been developed to enable LLMs to follow instructions and align with human preferences. These methods have found success in the field, but their effectiveness has not been demonstrated outside of high-resource languages. In this work, we discuss our experiences in post-training an LLM for instruction-following for English and Finnish. We use a multilingual LLM to translate instruction and preference datasets from English to Finnish. We perform instruction tuning and preference optimization in English and Finnish and evaluate the instruction-following capabilities of the model in both languages. Our results show that with a few hundred Finnish instruction samples we can obtain competitive performance in Finnish instruction-following. We also found that although preference optimization in English offers some cross-lingual benefits, we obtain our best results by using preference data from both languages. We release our model, datasets, and recipes under open licenses at https://huggingface.co/LumiOpen/Poro-34B-chat-OpenAssistant
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.07923v3">Asking Again and Again: Exploring LLM Robustness to Repeated Questions</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      This study investigates whether repeating questions within prompts influences the performance of large language models (LLMs). We hypothesize that reiterating a question within a single prompt might enhance the model's focus on key elements of the query. We evaluate five recent LLMs -- including GPT-4o-mini, DeepSeek-V3, and smaller open-source models -- on three reading comprehension datasets under different prompt settings, varying question repetition levels (1, 3, or 5 times per prompt). Our results demonstrate that question repetition can increase models' accuracy by up to $6\%$. However, across all models, settings, and datasets, we do not find the result statistically significant. These findings provide insights into prompt design and LLM behavior, suggesting that repetition alone does not significantly impact output quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09382v1">Towards Next-Generation Recommender Systems: A Benchmark for Personalized Recommendation Assistant with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Recommender systems (RecSys) are widely used across various modern digital platforms and have garnered significant attention. Traditional recommender systems usually focus only on fixed and simple recommendation scenarios, making it difficult to generalize to new and unseen recommendation tasks in an interactive paradigm. Recently, the advancement of large language models (LLMs) has revolutionized the foundational architecture of RecSys, driving their evolution into more intelligent and interactive personalized recommendation assistants. However, most existing studies rely on fixed task-specific prompt templates to generate recommendations and evaluate the performance of personalized assistants, which limits the comprehensive assessments of their capabilities. This is because commonly used datasets lack high-quality textual user queries that reflect real-world recommendation scenarios, making them unsuitable for evaluating LLM-based personalized recommendation assistants. To address this gap, we introduce RecBench+, a new dataset benchmark designed to access LLMs' ability to handle intricate user recommendation needs in the era of LLMs. RecBench+ encompasses a diverse set of queries that span both hard conditions and soft preferences, with varying difficulty levels. We evaluated commonly used LLMs on RecBench+ and uncovered below findings: 1) LLMs demonstrate preliminary abilities to act as recommendation assistants, 2) LLMs are better at handling queries with explicitly stated conditions, while facing challenges with queries that require reasoning or contain misleading information. Our dataset has been released at https://github.com/jiani-huang/RecBench.git.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09358v1">RetSTA: An LLM-Based Approach for Standardizing Clinical Fundus Image Reports</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Standardization of clinical reports is crucial for improving the quality of healthcare and facilitating data integration. The lack of unified standards, including format, terminology, and style, is a great challenge in clinical fundus diagnostic reports, which increases the difficulty for large language models (LLMs) to understand the data. To address this, we construct a bilingual standard terminology, containing fundus clinical terms and commonly used descriptions in clinical diagnosis. Then, we establish two models, RetSTA-7B-Zero and RetSTA-7B. RetSTA-7B-Zero, fine-tuned on an augmented dataset simulating clinical scenarios, demonstrates powerful standardization behaviors. However, it encounters a challenge of limitation to cover a wider range of diseases. To further enhance standardization performance, we build RetSTA-7B, which integrates a substantial amount of standardized data generated by RetSTA-7B-Zero along with corresponding English data, covering diverse complex clinical scenarios and achieving report-level standardization for the first time. Experimental results demonstrate that RetSTA-7B outperforms other compared LLMs in bilingual standardization task, which validates its superior performance and generalizability. The checkpoints are available at https://github.com/AB-Story/RetSTA-7B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.12464v3">Exploring LLM Cryptocurrency Trading Through Fact-Subjectivity Aware Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Accepted at ICLR 2025 Financial AI Workshop
    </div>
    <details class="paper-abstract">
      While many studies show that more advanced LLMs excel in tasks such as mathematics and coding, we observe that in cryptocurrency trading, stronger LLMs sometimes underperform compared to weaker ones. To investigate this counterintuitive phenomenon, we examine how LLMs reason when making trading decisions. Our findings reveal that (1) stronger LLMs show a preference for factual information over subjectivity; (2) separating the reasoning process into factual and subjective components leads to higher profits. Building on these insights, we propose a multi-agent framework, FS-ReasoningAgent, which enables LLMs to recognize and learn from both factual and subjective reasoning. Extensive experiments demonstrate that this fine-grained reasoning approach enhances LLM trading performance in cryptocurrency markets, yielding profit improvements of 7\% in BTC, 2\% in ETH, and 10\% in SOL. Additionally, an ablation study reveals that relying on subjective news generates higher returns in bull markets, while focusing on factual information yields better results in bear markets. Code is available at https://github.com/Persdre/FS-ReasoningAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09347v1">Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 8 pages, preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly employed as automated evaluators to assess the safety of generated content, yet their reliability in this role remains uncertain. This study evaluates a diverse set of 11 LLM judge models across critical safety domains, examining three key aspects: self-consistency in repeated judging tasks, alignment with human judgments, and susceptibility to input artifacts such as apologetic or verbose phrasing. Our findings reveal that biases in LLM judges can significantly distort the final verdict on which content source is safer, undermining the validity of comparative evaluations. Notably, apologetic language artifacts alone can skew evaluator preferences by up to 98\%. Contrary to expectations, larger models do not consistently exhibit greater robustness, while smaller models sometimes show higher resistance to specific artifacts. To mitigate LLM evaluator robustness issues, we investigate jury-based evaluations aggregating decisions from multiple models. Although this approach both improves robustness and enhances alignment to human judgements, artifact sensitivity persists even with the best jury configurations. These results highlight the urgent need for diversified, artifact-resistant methodologies to ensure reliable safety assessments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09334v1">CyberLLMInstruct: A New Dataset for Analysing Safety of Fine-Tuned LLMs Using Cyber Security Data</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 The paper is submitted to "The 48th International ACM SIGIR Conference on Research and Development in Information Retrieval" and is currently under review
    </div>
    <details class="paper-abstract">
      The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. To address these challenges, we developed CyberLLMInstruct, a dataset of 54,928 instruction-response pairs spanning cyber security tasks such as malware analysis, phishing simulations, and zero-day vulnerabilities. The dataset was constructed through a multi-stage process. This involved sourcing data from multiple resources, filtering and structuring it into instruction-response pairs, and aligning it with real-world scenarios to enhance its applicability. Seven open-source LLMs were chosen to test the usefulness of CyberLLMInstruct: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. In our primary example, we rigorously assess the safety of fine-tuned models using the OWASP top 10 framework, finding that fine-tuning reduces safety resilience across all tested LLMs and every adversarial attack (e.g., the security score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). In our second example, we show that these same fine-tuned models can also achieve up to 92.50 percent accuracy on the CyberMetric benchmark. These findings highlight a trade-off between performance and safety, showing the importance of adversarial testing and further research into fine-tuning methodologies that can mitigate safety risks while still improving performance across diverse datasets and domains. All scripts required to reproduce the dataset, along with examples and relevant resources for replicating our results, will be made available upon the paper's acceptance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09223v1">LREF: A Novel LLM-based Relevance Framework for E-commerce</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Query and product relevance prediction is a critical component for ensuring a smooth user experience in e-commerce search. Traditional studies mainly focus on BERT-based models to assess the semantic relevance between queries and products. However, the discriminative paradigm and limited knowledge capacity of these approaches restrict their ability to comprehend the relevance between queries and products fully. With the rapid advancement of Large Language Models (LLMs), recent research has begun to explore their application to industrial search systems, as LLMs provide extensive world knowledge and flexible optimization for reasoning processes. Nonetheless, directly leveraging LLMs for relevance prediction tasks introduces new challenges, including a high demand for data quality, the necessity for meticulous optimization of reasoning processes, and an optimistic bias that can result in over-recall. To overcome the above problems, this paper proposes a novel framework called the LLM-based RElevance Framework (LREF) aimed at enhancing e-commerce search relevance. The framework comprises three main stages: supervised fine-tuning (SFT) with Data Selection, Multiple Chain of Thought (Multi-CoT) tuning, and Direct Preference Optimization (DPO) for de-biasing. We evaluate the performance of the framework through a series of offline experiments on large-scale real-world datasets, as well as online A/B testing. The results indicate significant improvements in both offline and online metrics. Ultimately, the model was deployed in a well-known e-commerce application, yielding substantial commercial benefits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.23746v3">DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Accepted to NeurIPS 2024 Datasets and Benchmarks Track (Camera-Ready)
    </div>
    <details class="paper-abstract">
      Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating various prompts usages, human revisions like word substitutions, and writing noises like spelling mistakes. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09217v1">Evaluating the Generalizability of LLMs in Automated Program Repair</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 5 pages, 1 figure, to be published in ICSE2025-NIER
    </div>
    <details class="paper-abstract">
      LLM-based automated program repair methods have attracted significant attention for their state-of-the-art performance. However, they were primarily evaluated on a few well known datasets like Defects4J, raising questions about their effectiveness on new datasets. In this study, we evaluate 11 top-performing LLMs on DEFECTS4J-TRANS, a new dataset derived from transforming Defects4J while maintaining the original semantics. Results from experiments on both Defects4J and DEFECTS4J-TRANS show that all studied LLMs have limited generalizability in APR tasks, with the average number of correct and plausible patches decreasing by 49.48% and 42.90%, respectively, on DEFECTS4J-TRANS. Further investigation into incorporating additional repair-relevant information in repair prompts reveals that, although this information significantly enhances the LLMs' capabilities (increasing the number of correct and plausible patches by up to 136.67% and 121.82%, respectively), performance still falls short of their original results. This indicates that prompt engineering alone is insufficient to substantially enhance LLMs' repair capabilities. Based on our study, we also offer several recommendations for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09211v1">Why LLMs Cannot Think and How to Fix It</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Original conference submission for neurips 2024
    </div>
    <details class="paper-abstract">
      This paper elucidates that current state-of-the-art Large Language Models (LLMs) are fundamentally incapable of making decisions or developing "thoughts" within the feature space due to their architectural constraints. We establish a definition of "thought" that encompasses traditional understandings of that term and adapt it for application to LLMs. We demonstrate that the architectural design and language modeling training methodology of contemporary LLMs inherently preclude them from engaging in genuine thought processes. Our primary focus is on this theoretical realization rather than practical insights derived from experimental data. Finally, we propose solutions to enable thought processes within the feature space and discuss the broader implications of these architectural modifications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01824v2">AI Conversational Interviewing: Transforming Surveys with LLMs as Adaptive Interviewers</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Traditional methods for eliciting people's opinions face a trade-off between depth and scale: structured surveys enable large-scale data collection but limit respondents' ability to voice their opinions in their own words, while conversational interviews provide deeper insights but are resource-intensive. This study explores the potential of replacing human interviewers with large language models (LLMs) to conduct scalable conversational interviews. Our goal is to assess the performance of AI Conversational Interviewing and to identify opportunities for improvement in a controlled environment. We conducted a small-scale, in-depth study with university students who were randomly assigned to a conversational interview by either AI or human interviewers, both employing identical questionnaires on political topics. Various quantitative and qualitative measures assessed interviewer adherence to guidelines, response quality, participant engagement, and overall interview efficacy. The findings indicate the viability of AI Conversational Interviewing in producing quality data comparable to traditional methods, with the added benefit of scalability. We publish our data and materials for re-use and present specific recommendations for effective implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09205v1">Quality Over Quantity? LLM-Based Curation for a Data-Efficient Audio-Video Foundation Model</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 5 pages, 5 figures, 3 tables
    </div>
    <details class="paper-abstract">
      Integrating audio and visual data for training multimodal foundational models remains challenging. We present Audio-Video Vector Alignment (AVVA), which aligns audiovisual (AV) scene content beyond mere temporal synchronization via a Large Language Model (LLM)-based data curation pipeline. Specifically, AVVA scores and selects high-quality training clips using Whisper (speech-based audio foundation model) for audio and DINOv2 for video within a dual-encoder contrastive learning framework. Evaluations on AudioCaps, VALOR, and VGGSound demonstrate that this approach can achieve significant accuracy gains with substantially less curated data. For instance, AVVA yields a 7.6% improvement in top-1 accuracy for audio-to-video retrieval on VGGSound compared to ImageBind, despite training on only 192 hours of carefully filtered data (vs. 5800+ hours). Moreover, an ablation study highlights that trading data quantity for data quality improves performance, yielding respective top-3 accuracy increases of 47.8, 48.4, and 58.0 percentage points on AudioCaps, VALOR, and VGGSound over uncurated baselines. While these results underscore AVVA's data efficiency, we also discuss the overhead of LLM-driven curation and how it may be scaled or approximated in larger domains. Overall, AVVA provides a viable path toward more robust, text-free audiovisual learning with improved retrieval accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09153v1">Is LLMs Hallucination Usable? LLM-based Negative Reasoning for Fake News Detection</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 9 pages, 12 figures, conference
    </div>
    <details class="paper-abstract">
      The questionable responses caused by knowledge hallucination may lead to LLMs' unstable ability in decision-making. However, it has never been investigated whether the LLMs' hallucination is possibly usable to generate negative reasoning for facilitating the detection of fake news. This study proposes a novel supervised self-reinforced reasoning rectification approach - SR$^3$ that yields both common reasonable reasoning and wrong understandings (negative reasoning) for news via LLMs reflection for semantic consistency learning. Upon that, we construct a negative reasoning-based news learning model called - \emph{NRFE}, which leverages positive or negative news-reasoning pairs for learning the semantic consistency between them. To avoid the impact of label-implicated reasoning, we deploy a student model - \emph{NRFE-D} that only takes news content as input to inspect the performance of our method by distilling the knowledge from \emph{NRFE}. The experimental results verified on three popular fake news datasets demonstrate the superiority of our method compared with three kinds of baselines including prompting on LLMs, fine-tuning on pre-trained SLMs, and other representative fake news detection methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02233v2">Enhancing LLM Reliability via Explicit Knowledge Boundary Modeling</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) frequently hallucinate due to misaligned self-awareness, generating erroneous outputs when addressing queries beyond their knowledge boundaries. While existing approaches mitigate hallucinations via uncertainty estimation or query rejection, they suffer from computational inefficiency or sacrificed helpfulness. To address these issues, we propose the Explicit Knowledge Boundary Modeling (EKBM) framework, integrating fast and slow reasoning systems to harmonize reliability and usability. The framework first employs a fast-thinking model to generate confidence-labeled responses, enabling immediate use of high-confidence outputs. For uncertain predictions, a slow refinement model conducts targeted reasoning to improve accuracy. To align model behavior with our proposed object, we propose a hybrid training pipeline, enhancing self-awareness without degrading task performance. Evaluations on dialogue state tracking tasks demonstrate that EKBM achieves superior model reliability over uncertainty-based baselines. Further analysis reveals that refinement substantially boosts accuracy while maintaining low computational overhead. Our work establishes a scalable paradigm for advancing LLM reliability and balancing accuracy and practical utility in error-sensitive applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09103v1">VaxGuard: A Multi-Generator, Multi-Type, and Multi-Role Dataset for Detecting LLM-Generated Vaccine Misinformation</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Recent advancements in Large Language Models (LLMs) have significantly improved text generation capabilities. However, they also present challenges, particularly in generating vaccine-related misinformation, which poses risks to public health. Despite research on human-authored misinformation, a notable gap remains in understanding how LLMs contribute to vaccine misinformation and how best to detect it. Existing benchmarks often overlook vaccine-specific misinformation and the diverse roles of misinformation spreaders. This paper introduces VaxGuard, a novel dataset designed to address these challenges. VaxGuard includes vaccine-related misinformation generated by multiple LLMs and provides a comprehensive framework for detecting misinformation across various roles. Our findings show that GPT-3.5 and GPT-4o consistently outperform other LLMs in detecting misinformation, especially when dealing with subtle or emotionally charged narratives. On the other hand, PHI3 and Mistral show lower performance, struggling with precision and recall in fear-driven contexts. Additionally, detection performance tends to decline as input text length increases, indicating the need for improved methods to handle larger content. These results highlight the importance of role-specific detection strategies and suggest that VaxGuard can serve as a key resource for improving the detection of LLM-generated vaccine misinformation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09089v1">LocAgent: Graph-Guided LLM Agents for Code Localization</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Code localization--identifying precisely where in a codebase changes need to be made--is a fundamental yet challenging task in software maintenance. Existing approaches struggle to efficiently navigate complex codebases when identifying relevant code sections. The challenge lies in bridging natural language problem descriptions with the appropriate code elements, often requiring reasoning across hierarchical structures and multiple dependencies. We introduce LocAgent, a framework that addresses code localization through graph-based representation. By parsing codebases into directed heterogeneous graphs, LocAgent creates a lightweight representation that captures code structures (files, classes, functions) and their dependencies (imports, invocations, inheritance), enabling LLM agents to effectively search and locate relevant entities through powerful multi-hop reasoning. Experimental results on real-world benchmarks demonstrate that our approach significantly enhances accuracy in code localization. Notably, our method with the fine-tuned Qwen-2.5-Coder-Instruct-32B model achieves comparable results to SOTA proprietary models at greatly reduced cost (approximately 86% reduction), reaching up to 92.7% accuracy on file-level localization while improving downstream GitHub issue resolution success rates by 12% for multiple attempts (Pass@10). Our code is available at https://github.com/gersteinlab/LocAgent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09066v1">Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09032v1">Teaching LLMs How to Learn with Contextual Fine-Tuning</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Prompting Large Language Models (LLMs), or providing context on the expected model of operation, is an effective way to steer the outputs of such models to satisfy human desiderata after they have been trained. But in rapidly evolving domains, there is often need to fine-tune LLMs to improve either the kind of knowledge in their memory or their abilities to perform open ended reasoning in new domains. When human's learn new concepts, we often do so by linking the new material that we are studying to concepts we have already learned before. To that end, we ask, "can prompting help us teach LLMs how to learn". In this work, we study a novel generalization of instruction tuning, called contextual fine-tuning, to fine-tune LLMs. Our method leverages instructional prompts designed to mimic human cognitive strategies in learning and problem-solving to guide the learning process during training, aiming to improve the model's interpretation and understanding of domain-specific knowledge. We empirically demonstrate that this simple yet effective modification improves the ability of LLMs to be fine-tuned rapidly on new datasets both within the medical and financial domains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09027v1">Measure Twice, Cut Once: Grasping Video Structures and Event Semantics with LLMs for Video Temporal Localization</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Localizing user-queried events through natural language is crucial for video understanding models. Recent methods predominantly adapt Video LLMs to generate event boundary timestamps to handle temporal localization tasks, which struggle to leverage LLMs' powerful semantic understanding. In this work, we introduce MeCo, a novel timestamp-free framework that enables video LLMs to fully harness their intrinsic semantic capabilities for temporal localization tasks. Rather than outputting boundary timestamps, MeCo partitions videos into holistic event and transition segments based on the proposed structural token generation and grounding pipeline, derived from video LLMs' temporal structure understanding capability. We further propose a query-focused captioning task that compels the LLM to extract fine-grained, event-specific details, bridging the gap between localization and higher-level semantics and enhancing localization performance. Extensive experiments on diverse temporal localization tasks show that MeCo consistently outperforms boundary-centric methods, underscoring the benefits of a semantic-driven approach for temporal localization with video LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2312.10321v4">LLM-SQL-Solver: Can LLMs Determine SQL Equivalence?</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Judging the equivalence between two SQL queries is a fundamental problem with many practical applications in data management and SQL generation (i.e., evaluating the quality of generated SQL queries in text-to-SQL task). While the research community has reasoned about SQL equivalence for decades, it poses considerable difficulties and no complete solutions exist. Recently, Large Language Models (LLMs) have shown strong reasoning capability in conversation, question answering and solving mathematics challenges. In this paper, we study if LLMs can be used to determine the equivalence between SQL queries under two notions of SQL equivalence (semantic equivalence and relaxed equivalence). To assist LLMs in generating high quality responses, we present two prompting techniques: Miniature & Mull and Explain & Compare. The former technique is used to evaluate the semantic equivalence in which it asks LLMs to execute a query on a simple database instance and then explore if a counterexample exists by modifying the database. The latter technique is used to evaluate the relaxed equivalence in which it asks LLMs to explain the queries and then compare if they contain significant logical differences. Our experiments demonstrate using our techniques, LLMs is a promising tool to help data engineers in writing semantically equivalent SQL queries, however challenges still persist, and is a better metric for evaluating SQL generation than the popular execution accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09003v1">Leveraging Retrieval Augmented Generative LLMs For Automated Metadata Description Generation to Enhance Data Catalogs</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Presented in 5th International Conference on NLP & Text Mining (NLTM 2025)
    </div>
    <details class="paper-abstract">
      Data catalogs serve as repositories for organizing and accessing diverse collection of data assets, but their effectiveness hinges on the ease with which business users can look-up relevant content. Unfortunately, many data catalogs within organizations suffer from limited searchability due to inadequate metadata like asset descriptions. Hence, there is a need of content generation solution to enrich and curate metadata in a scalable way. This paper explores the challenges associated with metadata creation and proposes a unique prompt enrichment idea of leveraging existing metadata content using retrieval based few-shot technique tied with generative large language models (LLM). The literature also considers finetuning an LLM on existing content and studies the behavior of few-shot pretrained LLM (Llama, GPT3.5) vis-\`a-vis few-shot finetuned LLM (Llama2-7b) by evaluating their performance based on accuracy, factual grounding, and toxicity. Our preliminary results exhibit more than 80% Rouge-1 F1 for the generated content. This implied 87%- 88% of instances accepted as is or curated with minor edits by data stewards. By automatically generating descriptions for tables and columns in most accurate way, the research attempts to provide an overall framework for enterprises to effectively scale metadata curation and enrich its data catalog thereby vastly improving the data catalog searchability and overall usability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09002v1">KNighter: Transforming Static Analysis with LLM-Synthesized Checkers</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Static analysis is a powerful technique for bug detection in critical systems like operating system kernels. However, designing and implementing static analyzers is challenging, time-consuming, and typically limited to predefined bug patterns. While large language models (LLMs) have shown promise for static analysis, directly applying them to scan large codebases remains impractical due to computational constraints and contextual limitations. We present KNighter, the first approach that unlocks practical LLM-based static analysis by automatically synthesizing static analyzers from historical bug patterns. Rather than using LLMs to directly analyze massive codebases, our key insight is leveraging LLMs to generate specialized static analyzers guided by historical patch knowledge. KNighter implements this vision through a multi-stage synthesis pipeline that validates checker correctness against original patches and employs an automated refinement process to iteratively reduce false positives. Our evaluation on the Linux kernel demonstrates that KNighter generates high-precision checkers capable of detecting diverse bug patterns overlooked by existing human-written analyzers. To date, KNighter-synthesized checkers have discovered 70 new bugs/vulnerabilities in the Linux kernel, with 56 confirmed and 41 already fixed. 11 of these findings have been assigned CVE numbers. This work establishes an entirely new paradigm for scalable, reliable, and traceable LLM-based static analysis for real-world systems via checker synthesis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08990v1">JBFuzz: Jailbreaking LLMs Efficiently and Effectively Using Fuzzing</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown great promise as language understanding and decision making tools, and they have permeated various aspects of our everyday life. However, their widespread availability also comes with novel risks, such as generating harmful, unethical, or offensive content, via an attack called jailbreaking. Despite extensive efforts from LLM developers to align LLMs using human feedback, they are still susceptible to jailbreak attacks. To tackle this issue, researchers often employ red-teaming to understand and investigate jailbreak prompts. However, existing red-teaming approaches lack effectiveness, scalability, or both. To address these issues, we propose JBFuzz, a novel effective, automated, and scalable red-teaming technique for jailbreaking LLMs. JBFuzz is inspired by the success of fuzzing for detecting bugs/vulnerabilities in software. We overcome three challenges related to effectiveness and scalability by devising novel seed prompts, a lightweight mutation engine, and a lightweight and accurate evaluator for guiding the fuzzer. Assimilating all three solutions results in a potent fuzzer that only requires black-box access to the target LLM. We perform extensive experimental evaluation of JBFuzz using nine popular and widely-used LLMs. We find that JBFuzz successfully jailbreaks all LLMs for various harmful/unethical questions, with an average attack success rate of 99%. We also find that JBFuzz is extremely efficient as it jailbreaks a given LLM for a given question in 60 seconds on average. Our work highlights the susceptibility of the state-of-the-art LLMs to jailbreak attacks even after safety alignment, and serves as a valuable red-teaming tool for LLM developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20140v2">Telephone Surveys Meet Conversational AI: Evaluating a LLM-Based Telephone Survey System at Scale</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Accepted at 80th AAPOR Conference 2025
    </div>
    <details class="paper-abstract">
      Telephone surveys remain a valuable tool for gathering insights but typically require substantial resources in training and coordinating human interviewers. This work presents an AI-driven telephone survey system integrating text-to-speech (TTS), a large language model (LLM), and speech-to-text (STT) that mimics the versatility of human-led interviews (full-duplex dialogues) at scale. We tested the system across two populations, a pilot study in the United States (n = 75) and a large-scale deployment in Peru (n = 2,739), inviting participants via web-based links and contacting them via direct phone calls. The AI agent successfully administered open-ended and closed-ended questions, handled basic clarifications, and dynamically navigated branching logic, allowing fast large-scale survey deployment without interviewer recruitment or training. Our findings demonstrate that while the AI system's probing for qualitative depth was more limited than human interviewers, overall data quality approached human-led standards for structured items. This study represents one of the first successful large-scale deployments of an LLM-based telephone interviewer in a real-world survey context. The AI-powered telephone survey system has the potential for expanding scalable, consistent data collecting across market research, social science, and public opinion studies, thus improving operational efficiency while maintaining appropriate data quality for research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08965v1">LLM-Driven Usefulness Labeling for IR Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      In the information retrieval (IR) domain, evaluation plays a crucial role in optimizing search experiences and supporting diverse user intents. In the recent LLM era, research has been conducted to automate document relevance labels, as these labels have traditionally been assigned by crowd-sourced workers - a process that is both time and consuming and costly. This study focuses on LLM-generated usefulness labels, a crucial evaluation metric that considers the user's search intents and task objectives, an aspect where relevance falls short. Our experiment utilizes task-level, query-level, and document-level features along with user search behavior signals, which are essential in defining the usefulness of a document. Our research finds that (i) pre-trained LLMs can generate moderate usefulness labels by understanding the comprehensive search task session, (ii) pre-trained LLMs perform better judgement in short search sessions when provided with search session contexts. Additionally, we investigated whether LLMs can capture the unique divergence between relevance and usefulness, along with conducting an ablation study to identify the most critical metrics for accurate usefulness label generation. In conclusion, this work explores LLM-generated usefulness labels by evaluating critical metrics and optimizing for practicality in real-world settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.12480v2">The Power of LLM-Generated Synthetic Data for Stance Detection in Online Political Discussions</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 ICLR 2025 Spotlight
    </div>
    <details class="paper-abstract">
      Stance detection holds great potential to improve online political discussions through its deployment in discussion platforms for purposes such as content moderation, topic summarization or to facilitate more balanced discussions. Typically, transformer-based models are employed directly for stance detection, requiring vast amounts of data. However, the wide variety of debate topics in online political discussions makes data collection particularly challenging. LLMs have revived stance detection, but their online deployment in online political discussions faces challenges like inconsistent outputs, biases, and vulnerability to adversarial attacks. We show how LLM-generated synthetic data can improve stance detection for online political discussions by using reliable traditional stance detection models for online deployment, while leveraging the text generation capabilities of LLMs for synthetic data generation in a secure offline environment. To achieve this, (i) we generate synthetic data for specific debate questions by prompting a Mistral-7B model and show that fine-tuning with the generated synthetic data can substantially improve the performance of stance detection, while remaining interpretable and aligned with real world data. (ii) Using the synthetic data as a reference, we can improve performance even further by identifying the most informative samples in an unlabelled dataset, i.e., those samples which the stance detection model is most uncertain about and can benefit from the most. By fine-tuning with both synthetic data and the most informative samples, we surpass the performance of the baseline model that is fine-tuned on all true labels, while labelling considerably less data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09858v1">Media and responsible AI governance: a game-theoretic and LLM analysis</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      This paper investigates the complex interplay between AI developers, regulators, users, and the media in fostering trustworthy AI systems. Using evolutionary game theory and large language models (LLMs), we model the strategic interactions among these actors under different regulatory regimes. The research explores two key mechanisms for achieving responsible governance, safe AI development and adoption of safe AI: incentivising effective regulation through media reporting, and conditioning user trust on commentariats' recommendation. The findings highlight the crucial role of the media in providing information to users, potentially acting as a form of "soft" regulation by investigating developers or regulators, as a substitute to institutional AI regulation (which is still absent in many regions). Both game-theoretic analysis and LLM-based simulations reveal conditions under which effective regulation and trustworthy AI development emerge, emphasising the importance of considering the influence of different regulatory regimes from an evolutionary game-theoretic perspective. The study concludes that effective governance requires managing incentives and costs for high quality commentaries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09838v1">BioSpark: Beyond Analogical Inspiration to LLM-augmented Transfer</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      We present BioSpark, a system for analogical innovation designed to act as a creativity partner in reducing the cognitive effort in finding, mapping, and creatively adapting diverse inspirations. While prior approaches have focused on initial stages of finding inspirations, BioSpark uses LLMs embedded in a familiar, visual, Pinterest-like interface to go beyond inspiration to supporting users in identifying the key solution mechanisms, transferring them to the problem domain, considering tradeoffs, and elaborating on details and characteristics. To accomplish this BioSpark introduces several novel contributions, including a tree-of-life enabled approach for generating relevant and diverse inspirations, as well as AI-powered cards including 'Sparks' for analogical transfer; 'Trade-offs' for considering pros and cons; and 'Q&A' for deeper elaboration. We evaluated BioSpark through workshops with professional designers and a controlled user study, finding that using BioSpark led to a greater number of generated ideas; those ideas being rated higher in creative quality; and more diversity in terms of biological inspirations used than a control condition. Our results suggest new avenues for creativity support tools embedding AI in familiar interaction paradigms for designer workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09758v1">Multi-Agent LLM Actor-Critic Framework for Social Robot Navigation</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Recent advances in robotics and large language models (LLMs) have sparked growing interest in human-robot collaboration and embodied intelligence. To enable the broader deployment of robots in human-populated environments, socially-aware robot navigation (SAN) has become a key research area. While deep reinforcement learning approaches that integrate human-robot interaction (HRI) with path planning have demonstrated strong benchmark performance, they often struggle to adapt to new scenarios and environments. LLMs offer a promising avenue for zero-shot navigation through commonsense inference. However, most existing LLM-based frameworks rely on centralized decision-making, lack robust verification mechanisms, and face inconsistencies in translating macro-actions into precise low-level control signals. To address these challenges, we propose SAMALM, a decentralized multi-agent LLM actor-critic framework for multi-robot social navigation. In this framework, a set of parallel LLM actors, each reflecting distinct robot personalities or configurations, directly generate control signals. These actions undergo a two-tier verification process via a global critic that evaluates group-level behaviors and individual critics that assess each robot's context. An entropy-based score fusion mechanism further enhances self-verification and re-query, improving both robustness and coordination. Experimental results confirm that SAMALM effectively balances local autonomy with global oversight, yielding socially compliant behaviors and strong adaptability across diverse multi-robot scenarios. More details and videos about this work are available at: https://sites.google.com/view/SAMALM.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09701v1">Have LLMs Made Active Learning Obsolete? Surveying the NLP Community</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Supervised learning relies on annotated data, which is expensive to obtain. A longstanding strategy to reduce annotation costs is active learning, an iterative process, in which a human annotates only data instances deemed informative by a model. Large language models (LLMs) have pushed the effectiveness of active learning, but have also improved methods such as few- or zero-shot learning, and text synthesis - thereby introducing potential alternatives. This raises the question: has active learning become obsolete? To answer this fully, we must look beyond literature to practical experiences. We conduct an online survey in the NLP community to collect previously intangible insights on the perceived relevance of data annotation, particularly focusing on active learning, including best practices, obstacles and expected future developments. Our findings show that annotated data remains a key factor, and active learning continues to be relevant. While the majority of active learning users find it effective, a comparison with a community survey from over a decade ago reveals persistent challenges: setup complexity, estimation of cost reduction, and tooling. We publish an anonymized version of the collected dataset
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09674v1">Probabilistic Reasoning with LLMs for k-anonymity Estimation</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 9 pages
    </div>
    <details class="paper-abstract">
      Probabilistic reasoning is a key aspect of both human and artificial intelligence that allows for handling uncertainty and ambiguity in decision-making. In this paper, we introduce a novel numerical reasoning task under uncertainty, focusing on estimating the k-anonymity of user-generated documents containing privacy-sensitive information. We propose BRANCH, which uses LLMs to factorize a joint probability distribution to estimate the k-value-the size of the population matching the given information-by modeling individual pieces of textual information as random variables. The probability of each factor occurring within a population is estimated using standalone LLMs or retrieval-augmented generation systems, and these probabilities are combined into a final k-value. Our experiments show that this method successfully estimates the correct k-value 67% of the time, an 11% increase compared to GPT-4o chain-of-thought reasoning. Additionally, we leverage LLM uncertainty to develop prediction intervals for k-anonymity, which include the correct value in nearly 92% of cases.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09657v1">Týr-the-Pruner: Unlocking Accurate 50% Structural Pruning for LLMs via Global Sparsity Distribution Optimization</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Structural pruning enhances hardware-agnostic inference efficiency for large language models (LLMs) but often struggles to maintain performance. Local pruning performs efficient layer-by-layer compression but ignores global topology. Global pruning has the potential to find the optimal solution although resource-intensive. However, existing methods tend to rank structural saliency uniformly, ignoring inter-structure dependencies and failing to achieve end-to-end optimization. To address these limitations, we propose T\'yr-the-Pruner, an efficient end-to-end search-based global structural pruning framework. This framework constructs a supernet by repeatedly applying local pruning across a range of sparsity ratios to each layer in an LLM, with the core goal of determining the optimal sparsity distribution under a target overall sparsity ratio. Concretely, we introduce an effective local pruning and an expectation error accumulation approach to improve supernet construction. Furthermore, we employ an iterative prune-and-search strategy with coarse-to-fine sparsity granularity to ensure efficient search convergence. Experimental results show that T\'yr-the-Pruner achieves state-of-the-art structural pruning, retaining 97% of the dense model's performance while removing a challenging 50% of Llama-3.1-70B's parameters.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09648v1">A Survey on Trustworthy LLM Agents: Threats and Countermeasures</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      With the rapid evolution of Large Language Models (LLMs), LLM-based agents and Multi-agent Systems (MAS) have significantly expanded the capabilities of LLM ecosystems. This evolution stems from empowering LLMs with additional modules such as memory, tools, environment, and even other agents. However, this advancement has also introduced more complex issues of trustworthiness, which previous research focused solely on LLMs could not cover. In this survey, we propose the TrustAgent framework, a comprehensive study on the trustworthiness of agents, characterized by modular taxonomy, multi-dimensional connotations, and technical implementation. By thoroughly investigating and summarizing newly emerged attacks, defenses, and evaluation methods for agents and MAS, we extend the concept of Trustworthy LLM to the emerging paradigm of Trustworthy Agent. In TrustAgent, we begin by deconstructing and introducing various components of the Agent and MAS. Then, we categorize their trustworthiness into intrinsic (brain, memory, and tool) and extrinsic (user, agent, and environment) aspects. Subsequently, we delineate the multifaceted meanings of trustworthiness and elaborate on the implementation techniques of existing research related to these internal and external modules. Finally, we present our insights and outlook on this domain, aiming to provide guidance for future endeavors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.09647v1">Leveraging LLMS for Top-Down Sector Allocation In Automated Trading</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      This paper introduces a methodology leveraging Large Language Models (LLMs) for sector-level portfolio allocation through systematic analysis of macroeconomic conditions and market sentiment. Our framework emphasizes top-down sector allocation by processing multiple data streams simultaneously, including policy documents, economic indicators, and sentiment patterns. Empirical results demonstrate superior risk-adjusted returns compared to traditional cross momentum strategies, achieving a Sharpe ratio of 2.51 and portfolio return of 8.79% versus -0.61 and -1.39% respectively. These results suggest that LLM-based systematic macro analysis presents a viable approach for enhancing automated portfolio allocation decisions at the sector level.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10707v1">CALLM: Context-Aware Emotion Analysis in Cancer Survivors Using LLMs and Retrieval-Augmented Mobile Diaries</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 10 pages, including 3 figures; appendix: 8 pages with 19 figures
    </div>
    <details class="paper-abstract">
      Cancer survivors face unique emotional challenges that impact their quality of life. Mobile diary entries-short text entries recording through their phone about their emotional experiences-provide a promising method for tracking these experiences in real time. Although emotion analysis tools show potential for recognizing emotions from text, current methods lack the contextual understanding necessary to accurately interpret the brief, personal narratives in mobile diaries. We propose CALLM, a context-aware emotion analysis framework that leverages Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG), to analyze mobile diary entries from cancer survivors to predict their emotional states. The framework enhances prediction accuracy beyond existing methods by (1) integrating retrieved peer experiences as contextual examples and (2) incorporating individuals' temporal emotional trajectories from their mobile diary entries. We collected a large-scale dataset (N=407) of cancer survivors' mobile ecological momentary assessments (EMAs), which assessed positive and negative affect, desire to regulate emotions, social interaction quality, and availability for interventions, alongside daily mobile diary entries in an open response format regarding what was driving their current emotional experience. Results demonstrate strong performance of CALLM, with balanced accuracies reaching 72.96% for positive and 73.29% for negative affect, and 73.72% for predicting individual's desire to regulate emotions. Post-hoc analysis reveals that leveraging model confidence, encouraging longer diary entries, and incorporating personal ground truth, further enhance predictive outcomes. Our findings support the feasibility of deploying LLM-powered emotion analysis in chronic health populations and suggest promising directions for personalized interventions for cancer survivors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.10689v1">Learning to Contextualize Web Pages for Enhanced Decision Making by LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 Accepted to ICLR 2025
    </div>
    <details class="paper-abstract">
      Recent advances in large language models (LLMs) have led to a growing interest in developing LLM-based agents for automating web tasks. However, these agents often struggle with even simple tasks on real-world websites due to their limited capability to understand and process complex web page structures. In this work, we introduce LCoW, a framework for Learning language models to Contextualize complex Web pages into a more comprehensible form, thereby enhancing decision making by LLM agents. LCoW decouples web page understanding from decision making by training a separate contextualization module to transform complex web pages into comprehensible format, which are then utilized by the decision-making agent. We demonstrate that our contextualization module effectively integrates with LLM agents of various scales to significantly enhance their decision-making capabilities in web automation tasks. Notably, LCoW improves the success rates of closed-source LLMs (e.g., Gemini-1.5-flash, GPT-4o, Claude-3.5-Sonnet) by an average of 15.6%, and demonstrates a 23.7% average improvement in success rates for open-source LMs (e.g., Llama-3.1-8B, Llama-3.1-70B) on the WorkArena benchmark. Moreover, the Gemini-1.5-flash agent with LCoW achieves state-of-the-art results on the WebShop benchmark, outperforming human experts. The relevant code materials are available at our project page: https://lcowiclr2025.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2502.07036v2">Automated Consistency Analysis of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-12
      | 💬 10 pages, 12 figures, 3 tables, 3 algorithms, 2024 IEEE 6th International Conference on Trust, Privacy and Security in Intelligent Systems, and Applications (TPS-ISA), Washington, DC, USA
    </div>
    <details class="paper-abstract">
      Generative AI (Gen AI) with large language models (LLMs) are being widely adopted across the industry, academia and government. Cybersecurity is one of the key sectors where LLMs can be and/or are already being used. There are a number of problems that inhibit the adoption of trustworthy Gen AI and LLMs in cybersecurity and such other critical areas. One of the key challenge to the trustworthiness and reliability of LLMs is: how consistent an LLM is in its responses? In this paper, we have analyzed and developed a formal definition of consistency of responses of LLMs. We have formally defined what is consistency of responses and then develop a framework for consistency evaluation. The paper proposes two approaches to validate consistency: self-validation, and validation across multiple LLMs. We have carried out extensive experiments for several LLMs such as GPT4oMini, GPT3.5, Gemini, Cohere, and Llama3, on a security benchmark consisting of several cybersecurity questions: informational and situational. Our experiments corroborate the fact that even though these LLMs are being considered and/or already being used for several cybersecurity tasks today, they are often inconsistent in their responses, and thus are untrustworthy and unreliable for cybersecurity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2312.10321v4">LLM-SQL-Solver: Can LLMs Determine SQL Equivalence?</a></div>
    <div class="paper-meta">
      📅 2025-03-12
    </div>
    <details class="paper-abstract">
      Judging the equivalence between two SQL queries is a fundamental problem with many practical applications in data management and SQL generation (i.e., evaluating the quality of generated SQL queries in text-to-SQL task). While the research community has reasoned about SQL equivalence for decades, it poses considerable difficulties and no complete solutions exist. Recently, Large Language Models (LLMs) have shown strong reasoning capability in conversation, question answering and solving mathematics challenges. In this paper, we study if LLMs can be used to determine the equivalence between SQL queries under two notions of SQL equivalence (semantic equivalence and relaxed equivalence). To assist LLMs in generating high quality responses, we present two prompting techniques: Miniature & Mull and Explain & Compare. The former technique is used to evaluate the semantic equivalence in which it asks LLMs to execute a query on a simple database instance and then explore if a counterexample exists by modifying the database. The latter technique is used to evaluate the relaxed equivalence in which it asks LLMs to explain the queries and then compare if they contain significant logical differences. Our experiments demonstrate using our techniques, LLMs is a promising tool to help data engineers in writing semantically equivalent SQL queries, however challenges still persist, and is a better metric for evaluating SQL generation than the popular execution accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08147v1">FilmComposer: LLM-Driven Music Production for Silent Film Clips</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Project page: https://apple-jun.github.io/FilmComposer.github.io/
    </div>
    <details class="paper-abstract">
      In this work, we implement music production for silent film clips using LLM-driven method. Given the strong professional demands of film music production, we propose the FilmComposer, simulating the actual workflows of professional musicians. FilmComposer is the first to combine large generative models with a multi-agent approach, leveraging the advantages of both waveform music and symbolic music generation. Additionally, FilmComposer is the first to focus on the three core elements of music production for film-audio quality, musicality, and musical development-and introduces various controls, such as rhythm, semantics, and visuals, to enhance these key aspects. Specifically, FilmComposer consists of the visual processing module, rhythm-controllable MusicGen, and multi-agent assessment, arrangement and mix. In addition, our framework can seamlessly integrate into the actual music production pipeline and allows user intervention in every step, providing strong interactivity and a high degree of creative freedom. Furthermore, we propose MusicPro-7k which includes 7,418 film clips, music, description, rhythm spots and main melody, considering the lack of a professional and high-quality film music dataset. Finally, both the standard metrics and the new specialized metrics we propose demonstrate that the music generated by our model achieves state-of-the-art performance in terms of quality, consistency with video, diversity, musicality, and musical development. Project page: https://apple-jun.github.io/FilmComposer.github.io/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08123v1">LLM4MAC: An LLM-Driven Reinforcement Learning Framework for MAC Protocol Emergence</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 5 pages, 5 figures
    </div>
    <details class="paper-abstract">
      With the advent of 6G systems, emerging hyper-connected ecosystems necessitate agile and adaptive medium access control (MAC) protocols to contend with network dynamics and diverse service requirements. We propose LLM4MAC, a novel framework that harnesses large language models (LLMs) within a reinforcement learning paradigm to drive MAC protocol emergence. By reformulating uplink data transmission scheduling as a semantics-generalized partially observable Markov game (POMG), LLM4MAC encodes network operations in natural language, while proximal policy optimization (PPO) ensures continuous alignment with the evolving network dynamics. A structured identity embedding (SIE) mechanism further enables robust coordination among heterogeneous agents. Extensive simulations demonstrate that on top of a compact LLM, which is purposefully selected to balance performance with resource efficiency, the protocol emerging from LLM4MAC outperforms comparative baselines in throughput and generalization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.08014v2">MAGIC: Mastering Physical Adversarial Generation in Context through Collaborative LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Physical adversarial attacks in driving scenarios can expose critical vulnerabilities in visual perception models. However, developing such attacks remains challenging due to diverse real-world environments and the requirement for maintaining visual naturality. Building upon this challenge, we reformulate physical adversarial attacks as a one-shot patch generation problem. Our approach generates adversarial patches through a deep generative model that considers the specific scene context, enabling direct physical deployment in matching environments. The primary challenge lies in simultaneously achieving two objectives: generating adversarial patches that effectively mislead object detection systems while determining contextually appropriate deployment within the scene. We propose MAGIC (Mastering Physical Adversarial Generation In Context), a novel framework powered by multi-modal LLM agents to address these challenges. MAGIC automatically understands scene context and generates adversarial patch through the synergistic interaction of language and vision capabilities. In particular, MAGIC orchestrates three specialized LLM agents: The adv-patch generation agent (GAgent) masters the creation of deceptive patches through strategic prompt engineering for text-to-image models. The adv-patch deployment agent (DAgent) ensures contextual coherence by determining optimal deployment strategies based on scene understanding. The self-examination agent (EAgent) completes this trilogy by providing critical oversight and iterative refinement of both processes. We validate our method on both digital and physical levels, i.e., nuImage and manually captured real-world scenes, where both statistical and visual results prove that our MAGIC is powerful and effective for attacking widely applied object detection systems, i.e., YOLO and DETR series.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.15068v2">LLM-HDR: Bridging LLM-based Perception and Self-Supervision for Unpaired LDR-to-HDR Image Reconstruction</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      The translation of Low Dynamic Range (LDR) to High Dynamic Range (HDR) images is an important computer vision task. There is a significant amount of research utilizing both conventional non-learning methods and modern data-driven approaches, focusing on using both single-exposed and multi-exposed LDR for HDR image reconstruction. However, most current state-of-the-art methods require high-quality paired {LDR,HDR} datasets for model training. In addition, there is limited literature on using unpaired datasets for this task, that is, the model learns a mapping between domains, i.e., {LDR,HDR}. This paper proposes LLM-HDR, a method that integrates the perception of Large Language Models (LLM) into a modified semantic- and cycle-consistent adversarial architecture that utilizes unpaired {LDR,HDR} datasets for training. The method introduces novel artifact- and exposure-aware generators to address visual artifact removal and an encoder and loss to address semantic consistency, another under-explored topic. LLM-HDR is the first to use an LLM for the {LDR,HDR} translation task in a self-supervised setup. The method achieves state-of-the-art performance across several benchmark datasets and reconstructs high-quality HDR images. The official website of this work is available at: https://github.com/HrishavBakulBarua/LLM-HDR
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08035v1">Group Preference Alignment: Customized LLM Response Generation from In-Situ Conversations</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      LLMs often fail to meet the specialized needs of distinct user groups due to their one-size-fits-all training paradigm \cite{lucy-etal-2024-one} and there is limited research on what personalization aspects each group expect. To address these limitations, we propose a group-aware personalization framework, Group Preference Alignment (GPA), that identifies context-specific variations in conversational preferences across user groups and then steers LLMs to address those preferences. Our approach consists of two steps: (1) Group-Aware Preference Extraction, where maximally divergent user-group preferences are extracted from real-world conversation logs and distilled into interpretable rubrics, and (2) Tailored Response Generation, which leverages these rubrics through two methods: a) Context-Tuned Inference (GAP-CT), that dynamically adjusts responses via context-dependent prompt instructions, and b) Rubric-Finetuning Inference (GPA-FT), which uses the rubrics to generate contrastive synthetic data for personalization of group-specific models via alignment. Experiments demonstrate that our framework significantly improves alignment of the output with respect to user preferences and outperforms baseline methods, while maintaining robust performance on standard benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07993v1">LLM-Powered Knowledge Graphs for Enterprise Intelligence and Analytics</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Disconnected data silos within enterprises obstruct the extraction of actionable insights, diminishing efficiency in areas such as product development, client engagement, meeting preparation, and analytics-driven decision-making. This paper introduces a framework that uses large language models (LLMs) to unify various data sources into a comprehensive, activity-centric knowledge graph. The framework automates tasks such as entity extraction, relationship inference, and semantic enrichment, enabling advanced querying, reasoning, and analytics across data types like emails, calendars, chats, documents, and logs. Designed for enterprise flexibility, it supports applications such as contextual search, task prioritization, expertise discovery, personalized recommendations, and advanced analytics to identify trends and actionable insights. Experimental results demonstrate its success in the discovery of expertise, task management, and data-driven decision making. By integrating LLMs with knowledge graphs, this solution bridges disconnected systems and delivers intelligent analytics-powered enterprise tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.06772v2">ReasonFlux: Hierarchical LLM Reasoning via Scaling Thought Templates</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Code: https://github.com/Gen-Verse/ReasonFlux
    </div>
    <details class="paper-abstract">
      We present that hierarchical LLM reasoning via scaling thought templates can effectively optimize the reasoning search space and outperform the mathematical reasoning capabilities of powerful LLMs like OpenAI o1-preview and DeepSeek V3. We train our ReasonFlux-32B model with only 8 GPUs and introduces three innovations: (i) a structured and generic thought template library, containing around 500 high-level thought templates capable of generalizing to similar or relevant reasoning problems; (ii) performing hierarchical reinforcement learning on a sequence of thought templates instead of long CoTs, optimizing a base LLM to plan out an optimal template trajectory for gradually handling complex problems; (iii) a brand new inference scaling system that enables hierarchical LLM reasoning by adaptively scaling thought templates at inference time. With a template trajectory containing more explainable reasoning structures than DeepSeek-R1 and o3-mini, our ReasonFlux-32B significantly advances math reasoning capabilities to state-of-the-art levels. Notably, on the MATH benchmark, it achieves an accuracy of 91.2% and surpasses o1-preview by 6.7%. On the USA Math Olympiad (AIME) benchmark, ReasonFlux-32B solves an average of 56.7% of problems, surpassing o1-preview and DeepSeek-V3 by 27% and 45%, respectively. Code: https://github.com/Gen-Verse/ReasonFlux
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07967v1">Code Digital Twin: Empowering LLMs with Tacit Knowledge for Complex Software Maintenance</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 A vision paper that will be continuously updated
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) have demonstrated promise in software engineering tasks like code completion and generation, their support for the maintenance of complex software systems remains limited. These models often struggle with understanding the tacit knowledge embedded in systems, such as responsibility allocation and collaboration across different modules. To address this gap, we introduce the concept and framework of \textbf{Code Digital Twin}, a conceptual representation of tacit knowledge that captures the concepts, functionalities, and design rationales behind code elements, co-evolving with the software. A code digital twin is constructed using a methodology that combines knowledge extraction from both structured and unstructured sources--such as source code, documentation, and change histories--leveraging LLMs, static analysis tools, and human expertise. This framework can empower LLMs for software maintenance tasks such as issue localization and repository-level code generation by providing tacit knowledge as contexts. Based on the proposed methodology, we explore the key challenges and opportunities involved in the continuous construction and refinement of code digital twin.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07937v1">LLM-based Corroborating and Refuting Evidence Retrieval for Scientific Claim Verification</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      In this paper, we introduce CIBER (Claim Investigation Based on Evidence Retrieval), an extension of the Retrieval-Augmented Generation (RAG) framework designed to identify corroborating and refuting documents as evidence for scientific claim verification. CIBER addresses the inherent uncertainty in Large Language Models (LLMs) by evaluating response consistency across diverse interrogation probes. By focusing on the behavioral analysis of LLMs without requiring access to their internal information, CIBER is applicable to both white-box and black-box models. Furthermore, CIBER operates in an unsupervised manner, enabling easy generalization across various scientific domains. Comprehensive evaluations conducted using LLMs with varying levels of linguistic proficiency reveal CIBER's superior performance compared to conventional RAG approaches. These findings not only highlight the effectiveness of CIBER but also provide valuable insights for future advancements in LLM-based scientific claim verification.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.20666v2">Guide-LLM: An Embodied LLM Agent and Text-Based Topological Map for Robotic Guidance of People with Visual Impairments</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Navigation presents a significant challenge for persons with visual impairments (PVI). While traditional aids such as white canes and guide dogs are invaluable, they fall short in delivering detailed spatial information and precise guidance to desired locations. Recent developments in large language models (LLMs) and vision-language models (VLMs) offer new avenues for enhancing assistive navigation. In this paper, we introduce Guide-LLM, an embodied LLM-based agent designed to assist PVI in navigating large indoor environments. Our approach features a novel text-based topological map that enables the LLM to plan global paths using a simplified environmental representation, focusing on straight paths and right-angle turns to facilitate navigation. Additionally, we utilize the LLM's commonsense reasoning for hazard detection and personalized path planning based on user preferences. Simulated experiments demonstrate the system's efficacy in guiding PVI, underscoring its potential as a significant advancement in assistive technology. The results highlight Guide-LLM's ability to offer efficient, adaptive, and personalized navigation assistance, pointing to promising advancements in this field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.18469v3">Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Accepted to NAACL 2025 Main (oral)
    </div>
    <details class="paper-abstract">
      Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.14572v2">Evaluating the Performance and Robustness of LLMs in Materials Science Q&A and Property Predictions</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have the potential to revolutionize scientific research, yet their robustness and reliability in domain-specific applications remain insufficiently explored. In this study, we evaluate the performance and robustness of LLMs for materials science, focusing on domain-specific question answering and materials property prediction across diverse real-world and adversarial conditions. Three distinct datasets are used in this study: 1) a set of multiple-choice questions from undergraduate-level materials science courses, 2) a dataset including various steel compositions and yield strengths, and 3) a band gap dataset, containing textual descriptions of material crystal structures and band gap values. The performance of LLMs is assessed using various prompting strategies, including zero-shot chain-of-thought, expert prompting, and few-shot in-context learning. The robustness of these models is tested against various forms of 'noise', ranging from realistic disturbances to intentionally adversarial manipulations, to evaluate their resilience and reliability under real-world conditions. Additionally, the study showcases unique phenomena of LLMs during predictive tasks, such as mode collapse behavior when the proximity of prompt examples is altered and performance recovery from train/test mismatch. The findings aim to provide informed skepticism for the broad use of LLMs in materials science and to inspire advancements that enhance their robustness and reliability for practical applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.16359v2">Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 arXiv admin note: text overlap with arXiv:2407.14644
    </div>
    <details class="paper-abstract">
      Previous studies that uncovered vulnerabilities in large language models (LLMs) frequently employed nonsensical adversarial prompts. However, such prompts can now be readily identified using automated detection techniques. To further strengthen adversarial attacks, we focus on human-readable adversarial prompts, which are more realistic and potent threats. Our key contributions are (1) situation-driven attacks leveraging movie scripts as context to create human-readable prompts that successfully deceive LLMs, (2) adversarial suffix conversion to transform nonsensical adversarial suffixes into independent meaningful text, and (3) AdvPrompter with p-nucleus sampling, a method to generate diverse, human-readable adversarial suffixes, improving attack efficacy in models like GPT-3.5 and Gemma 7B.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.05965v2">Validating LLM-as-a-Judge Systems in the Absence of Gold Labels</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      The LLM-as-a-judge paradigm, in which a judge LLM system replaces human raters in rating the outputs of other generative AI (GenAI) systems, has come to play a critical role in scaling and standardizing GenAI evaluations. To validate judge systems, evaluators collect multiple human ratings for each item in a validation corpus, and then aggregate the ratings into a single, per-item gold label rating. High agreement rates between these gold labels and judge system ratings are then taken as a sign of good judge system performance. In many cases, however, items or rating criteria may be ambiguous, or there may be principled disagreement among human raters. In such settings, gold labels may not exist for many of the items. In this paper, we introduce a framework for LLM-as-a-judge validation in the absence of gold labels. We present a theoretical analysis drawing connections between different measures of judge system performance under different rating elicitation and aggregation schemes. We also demonstrate empirically that existing validation approaches can select judge systems that are highly suboptimal, performing as much as 34% worse than the systems selected by alternative approaches that we describe. Based on our findings, we provide concrete recommendations for developing more reliable approaches to LLM-as-a-judge validation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08879v1">LLMs Know What to Drop: Self-Attention Guided KV Cache Eviction for Efficient Long-Context Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Efficient long-context inference is critical as large language models (LLMs) adopt context windows of ranging from 128K to 1M tokens. However, the growing key-value (KV) cache and the high computational complexity of attention create significant bottlenecks in memory usage and latency. In this paper, we find that attention in diverse long-context tasks exhibits sparsity, and LLMs implicitly "know" which tokens can be dropped or evicted at the head level after the pre-filling stage. Based on this insight, we propose Self-Attention Guided Eviction~(SAGE-KV), a simple and effective KV eviction cache method for long-context inference. After prefilling, our method performs a one-time top-k selection at both the token and head levels to compress the KV cache, enabling efficient inference with the reduced cache. Evaluations on LongBench and three long-context LLMs (Llama3.1-8B-Instruct-128k, Llama3-8B-Prolong-512k-Instruct, and Qwen2.5-7B-Instruct-128k) show that SAGE-KV maintains accuracy comparable to full attention while significantly improving efficiency. Specifically, SAGE-KV achieves 4x higher memory efficiency with improved accuracy over the static KV cache selection method StreamLLM, and 2x higher memory efficiency with better accuracy than the dynamic KV cache selection method Quest.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08857v1">Interpretable and Robust Dialogue State Tracking via Natural Language Summarization with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      This paper introduces a novel approach to Dialogue State Tracking (DST) that leverages Large Language Models (LLMs) to generate natural language descriptions of dialogue states, moving beyond traditional slot-value representations. Conventional DST methods struggle with open-domain dialogues and noisy inputs. Motivated by the generative capabilities of LLMs, our Natural Language DST (NL-DST) framework trains an LLM to directly synthesize human-readable state descriptions. We demonstrate through extensive experiments on MultiWOZ 2.1 and Taskmaster-1 datasets that NL-DST significantly outperforms rule-based and discriminative BERT-based DST baselines, as well as generative slot-filling GPT-2 DST models, in both Joint Goal Accuracy and Slot Accuracy. Ablation studies and human evaluations further validate the effectiveness of natural language state generation, highlighting its robustness to noise and enhanced interpretability. Our findings suggest that NL-DST offers a more flexible, accurate, and human-understandable approach to dialogue state tracking, paving the way for more robust and adaptable task-oriented dialogue systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08842v1">Contrastive Speaker-Aware Learning for Multi-party Dialogue Generation with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Multi-party dialogue generation presents significant challenges due to the complex interplay of multiple speakers and interwoven conversational threads. Traditional approaches often fall short in capturing these complexities, particularly when relying on manually annotated dialogue relations. This paper introduces Speaker-Attentive LLM (SA-LLM), a novel generative model that leverages pre-trained Large Language Models (LLMs) and a speaker-aware contrastive learning strategy to address these challenges. SA-LLM incorporates a speaker-attributed input encoding and a contrastive learning objective to implicitly learn contextual coherence and speaker roles without explicit relation annotations. Extensive experiments on the Ubuntu IRC and Movie Dialogues datasets demonstrate that SA-LLM significantly outperforms state-of-the-art baselines in automatic and human evaluations, achieving superior performance in fluency, coherence, informativeness, and response diversity. Ablation studies and detailed error analyses further validate the effectiveness of the proposed speaker-attentive training approach, highlighting its robustness across different speaker roles and context lengths. The results underscore the potential of SA-LLM as a powerful and annotation-free solution for high-quality multi-party dialogue generation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.16812v2">Towards Human-AI Deliberation: Design and Evaluation of LLM-Empowered Deliberative AI for AI-Assisted Decision-Making</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 23 pages, ACM CHI 2025
    </div>
    <details class="paper-abstract">
      In AI-assisted decision-making, humans often passively review AI's suggestion and decide whether to accept or reject it as a whole. In such a paradigm, humans are found to rarely trigger analytical thinking and face difficulties in communicating the nuances of conflicting opinions to the AI when disagreements occur. To tackle this challenge, we propose Human-AI Deliberation, a novel framework to promote human reflection and discussion on conflicting human-AI opinions in decision-making. Based on theories in human deliberation, this framework engages humans and AI in dimension-level opinion elicitation, deliberative discussion, and decision updates. To empower AI with deliberative capabilities, we designed Deliberative AI, which leverages large language models (LLMs) as a bridge between humans and domain-specific models to enable flexible conversational interactions and faithful information provision. An exploratory evaluation on a graduate admissions task shows that Deliberative AI outperforms conventional explainable AI (XAI) assistants in improving humans' appropriate reliance and task performance. Based on a mixed-methods analysis of participant behavior, perception, user experience, and open-ended feedback, we draw implications for future AI-assisted decision tool design.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08823v1">ResBench: Benchmarking LLM-Generated FPGA Designs with Resource Awareness</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 to be published in International Symposium on Highly Efficient Accelerators and Reconfigurable Technologies 2025
    </div>
    <details class="paper-abstract">
      Field-Programmable Gate Arrays (FPGAs) are widely used in modern hardware design, yet writing Hardware Description Language (HDL) code for FPGA implementation remains labor-intensive and complex. Large Language Models (LLMs) have emerged as a promising tool for automating HDL generation, but existing benchmarks for LLM HDL code generation primarily evaluate functional correctness while overlooking the critical aspect of hardware resource efficiency. Moreover, current benchmarks lack diversity, failing to capture the broad range of real-world FPGA applications. To address these gaps, we introduce ResBench, the first resource-oriented benchmark explicitly designed to differentiate between resource-optimized and inefficient LLM-generated HDL. ResBench consists of 56 problems across 12 categories, covering applications from finite state machines to financial computing. Our evaluation framework systematically integrates FPGA resource constraints, with a primary focus on Lookup Table (LUT) usage, enabling a realistic assessment of hardware efficiency. Experimental results reveal substantial differences in resource utilization across LLMs, demonstrating ResBench's effectiveness in distinguishing models based on their ability to generate resource-optimized FPGA designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.05440v3">Can LLMs Understand Time Series Anomalies?</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained popularity in time series forecasting, but their potential for anomaly detection remains largely unexplored. Our study investigates whether LLMs can understand and detect anomalies in time series data, focusing on zero-shot and few-shot scenarios. Inspired by conjectures about LLMs' behavior from time series forecasting research, we formulate key hypotheses about LLMs' capabilities in time series anomaly detection. We design and conduct principled experiments to test each of these hypotheses. Our investigation reveals several surprising findings about LLMs for time series: (1) LLMs understand time series better as images rather than as text, (2) LLMs do not demonstrate enhanced performance when prompted to engage in explicit reasoning about time series analysis. (3) Contrary to common beliefs, LLMs' understanding of time series does not stem from their repetition biases or arithmetic abilities. (4) LLMs' behaviors and performance in time series analysis vary significantly across different models. This study provides the first comprehensive analysis of contemporary LLM capabilities in time series anomaly detection. Our results suggest that while LLMs can understand trivial time series anomalies, we have no evidence that they can understand more subtle real-world anomalies. Many common conjectures based on their reasoning capabilities do not hold. All synthetic dataset generators, final prompts, and evaluation scripts have been made available in https://github.com/rose-stl-lab/anomllm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08750v1">Exposing Product Bias in LLM Investment Recommendation</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Large language models (LLMs), as a new generation of recommendation engines, possess powerful summarization and data analysis capabilities, surpassing traditional recommendation systems in both scope and performance. One promising application is investment recommendation. In this paper, we reveal a novel product bias in LLM investment recommendation, where LLMs exhibit systematic preferences for specific products. Such preferences can subtly influence user investment decisions, potentially leading to inflated valuations of products and financial bubbles, posing risks to both individual investors and market stability. To comprehensively study the product bias, we develop an automated pipeline to create a dataset of 567,000 samples across five asset classes (stocks, mutual funds, cryptocurrencies, savings, and portfolios). With this dataset, we present the bf first study on product bias in LLM investment recommendations. Our findings reveal that LLMs exhibit clear product preferences, such as certain stocks (e.g., `AAPL' from Apple and `MSFT' from Microsoft). Notably, this bias persists even after applying debiasing techniques. We urge AI researchers to take heed of the product bias in LLM investment recommendations and its implications, ensuring fairness and security in the digital space and market.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08688v1">Randomness, Not Representation: The Unreliability of Evaluating Cultural Alignment in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Research on the 'cultural alignment' of Large Language Models (LLMs) has emerged in response to growing interest in understanding representation across diverse stakeholders. Current approaches to evaluating cultural alignment borrow social science methodologies but often overlook systematic robustness checks. Here, we identify and test three assumptions behind current evaluation methods: (1) Stability: that cultural alignment is a property of LLMs rather than an artifact of evaluation design, (2) Extrapolability: that alignment with one culture on a narrow set of issues predicts alignment with that culture on others, and (3) Steerability: that LLMs can be reliably prompted to represent specific cultural perspectives. Through experiments examining both explicit and implicit preferences of leading LLMs, we find a high level of instability across presentation formats, incoherence between evaluated versus held-out cultural dimensions, and erratic behavior under prompt steering. We show that these inconsistencies can cause the results of an evaluation to be very sensitive to minor variations in methodology. Finally, we demonstrate in a case study on evaluation design that narrow experiments and a selective assessment of evidence can be used to paint an incomplete picture of LLMs' cultural alignment properties. Overall, these results highlight significant limitations of current approaches for evaluating the cultural alignment of LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08683v1">CoLMDriver: LLM-based Negotiation Benefits Cooperative Autonomous Driving</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Vehicle-to-vehicle (V2V) cooperative autonomous driving holds great promise for improving safety by addressing the perception and prediction uncertainties inherent in single-agent systems. However, traditional cooperative methods are constrained by rigid collaboration protocols and limited generalization to unseen interactive scenarios. While LLM-based approaches offer generalized reasoning capabilities, their challenges in spatial planning and unstable inference latency hinder their direct application in cooperative driving. To address these limitations, we propose CoLMDriver, the first full-pipeline LLM-based cooperative driving system, enabling effective language-based negotiation and real-time driving control. CoLMDriver features a parallel driving pipeline with two key components: (i) an LLM-based negotiation module under an actor-critic paradigm, which continuously refines cooperation policies through feedback from previous decisions of all vehicles; and (ii) an intention-guided waypoint generator, which translates negotiation outcomes into executable waypoints. Additionally, we introduce InterDrive, a CARLA-based simulation benchmark comprising 10 challenging interactive driving scenarios for evaluating V2V cooperation. Experimental results demonstrate that CoLMDriver significantly outperforms existing approaches, achieving an 11% higher success rate across diverse highly interactive V2V driving scenarios. Code will be released on https://github.com/cxliu0314/CoLMDriver.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08569v1">DeepReview: Improving LLM-based Paper Review with Human-like Deep Thinking Process</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly utilized in scientific research assessment, particularly in automated paper review. However, existing LLM-based review systems face significant challenges, including limited domain expertise, hallucinated reasoning, and a lack of structured evaluation. To address these limitations, we introduce DeepReview, a multi-stage framework designed to emulate expert reviewers by incorporating structured analysis, literature retrieval, and evidence-based argumentation. Using DeepReview-13K, a curated dataset with structured annotations, we train DeepReviewer-14B, which outperforms CycleReviewer-70B with fewer tokens. In its best mode, DeepReviewer-14B achieves win rates of 88.21\% and 80.20\% against GPT-o1 and DeepSeek-R1 in evaluations. Our work sets a new benchmark for LLM-based paper review, with all resources publicly available. The code, model, dataset and demo have be released in http://ai-researcher.net.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.02800v3">RAAD-LLM: Adaptive Anomaly Detection Using LLMs and RAG Integration</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 arXiv admin note: substantial text overlap with arXiv:2411.00914
    </div>
    <details class="paper-abstract">
      Anomaly detection in complex industrial environments poses unique challenges, particularly in contexts characterized by data sparsity and evolving operational conditions. Predictive maintenance (PdM) in such settings demands methodologies that are adaptive, transferable, and capable of integrating domain-specific knowledge. In this paper, we present RAAD-LLM, a novel framework for adaptive anomaly detection, leveraging large language models (LLMs) integrated with Retrieval-Augmented Generation (RAG). This approach addresses the aforementioned PdM challenges. By effectively utilizing domain-specific knowledge, RAAD-LLM enhances the detection of anomalies in time series data without requiring fine-tuning on specific datasets. The framework's adaptability mechanism enables it to adjust its understanding of normal operating conditions dynamically, thus increasing detection accuracy. We validate this methodology through a real-world application for a plastics manufacturing plant and the Skoltech Anomaly Benchmark (SKAB). Results show significant improvements over our previous model with an accuracy increase from 70.7% to 88.6% on the real-world dataset. By allowing for the enriching of input series data with semantics, RAAD-LLM incorporates multimodal capabilities that facilitate more collaborative decision-making between the model and plant operators. Overall, our findings support RAAD-LLM's ability to revolutionize anomaly detection methodologies in PdM, potentially leading to a paradigm shift in how anomaly detection is implemented across various industries.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08551v1">Reasoning and Sampling-Augmented MCQ Difficulty Prediction via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      The difficulty of multiple-choice questions (MCQs) is a crucial factor for educational assessments. Predicting MCQ difficulty is challenging since it requires understanding both the complexity of reaching the correct option and the plausibility of distractors, i.e., incorrect options. In this paper, we propose a novel, two-stage method to predict the difficulty of MCQs. First, to better estimate the complexity of each MCQ, we use large language models (LLMs) to augment the reasoning steps required to reach each option. We use not just the MCQ itself but also these reasoning steps as input to predict the difficulty. Second, to capture the plausibility of distractors, we sample knowledge levels from a distribution to account for variation among students responding to the MCQ. This setup, inspired by item response theory (IRT), enable us to estimate the likelihood of students selecting each (both correct and incorrect) option. We align these predictions with their ground truth values, using a Kullback-Leibler (KL) divergence-based regularization objective, and use estimated likelihoods to predict MCQ difficulty. We evaluate our method on two real-world \emph{math} MCQ and response datasets with ground truth difficulty values estimated using IRT. Experimental results show that our method outperforms all baselines, up to a 28.3\% reduction in mean squared error and a 34.6\% improvement in the coefficient of determination. We also qualitatively discuss how our novel method results in higher accuracy in predicting MCQ difficulty.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08549v1">Graph of AI Ideas: Leveraging Knowledge Graphs and LLMs for AI Research Idea Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Reading relevant scientific papers and analyzing research development trends is a critical step in generating new scientific ideas. However, the rapid increase in the volume of research literature and the complex citation relationships make it difficult for researchers to quickly analyze and derive meaningful research trends. The development of large language models (LLMs) has provided a novel approach for automatically summarizing papers and generating innovative research ideas. However, existing paper-based idea generation methods either simply input papers into LLMs via prompts or form logical chains of creative development based on citation relationships, without fully exploiting the semantic information embedded in these citations. Inspired by knowledge graphs and human cognitive processes, we propose a framework called the Graph of AI Ideas (GoAI) for the AI research field, which is dominated by open-access papers. This framework organizes relevant literature into entities within a knowledge graph and summarizes the semantic information contained in citations into relations within the graph. This organization effectively reflects the relationships between two academic papers and the advancement of the AI research field. Such organization aids LLMs in capturing the current progress of research, thereby enhancing their creativity. Experimental results demonstrate the effectiveness of our approach in generating novel, clear, and effective research ideas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08542v1">DAFE: LLM-Based Evaluation Through Dynamic Arbitration for Free-Form Question-Answering</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Evaluating Large Language Models (LLMs) free-form generated responses remains a challenge due to their diverse and open-ended nature. Traditional supervised signal-based automatic metrics fail to capture semantic equivalence or handle the variability of open-ended responses, while human evaluation, though reliable, is resource-intensive. Leveraging LLMs as evaluators offers a promising alternative due to their strong language understanding and instruction-following capabilities. Taking advantage of these capabilities, we propose the Dynamic Arbitration Framework for Evaluation (DAFE), which employs two primary LLM-as-judges and engages a third arbitrator only in cases of disagreements. This selective arbitration prioritizes evaluation reliability while reducing unnecessary computational demands compared to conventional majority voting. DAFE utilizes task-specific reference answers with dynamic arbitration to enhance judgment accuracy, resulting in significant improvements in evaluation metrics such as Macro F1 and Cohen's Kappa. Through experiments, including a comprehensive human evaluation, we demonstrate DAFE's ability to provide consistent, scalable, and resource-efficient assessments, establishing it as a robust framework for evaluating free-form model outputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08537v1">Chemical reasoning in LLMs unlocks steerable synthesis planning and reaction mechanism elucidation</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      While machine learning algorithms have been shown to excel at specific chemical tasks, they have struggled to capture the strategic thinking that characterizes expert chemical reasoning, limiting their widespread adoption. Here we demonstrate that large language models (LLMs) can serve as powerful chemical reasoning engines when integrated with traditional search algorithms, enabling a new approach to computer-aided chemistry that mirrors human expert thinking. Rather than using LLMs to directly manipulate chemical structures, we leverage their ability to evaluate chemical strategies and guide search algorithms toward chemically meaningful solutions. We demonstrate this paradigm through two fundamental challenges: strategy-aware retrosynthetic planning and mechanism elucidation. In retrosynthetic planning, our method allows chemists to specify desired synthetic strategies in natural language to find routes that satisfy these constraints in vast searches. In mechanism elucidation, LLMs guide the search for plausible reaction mechanisms by combining chemical principles with systematic exploration. Our approach shows strong performance across diverse chemical tasks, with larger models demonstrating increasingly sophisticated chemical reasoning. Our approach establishes a new paradigm for computer-aided chemistry that combines the strategic understanding of LLMs with the precision of traditional chemical tools, opening possibilities for more intuitive and powerful chemical reasoning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.18363v3">ChatRex: Taming Multimodal LLM for Joint Perception and Understanding</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 35 pages, 19 figures
    </div>
    <details class="paper-abstract">
      Perception and understanding are two pillars of computer vision. While multimodal large language models (MLLM) have demonstrated remarkable visual understanding capabilities, they arguably lack accurate perception abilities, e.g. the stage-of-the-art model Qwen2-VL only achieves a 43.9 recall rate on the COCO dataset, limiting many tasks requiring the combination of perception and understanding. In this work, we aim to bridge this perception gap from both model designing and data development perspectives. We first introduce ChatRex, an MLLM with a decoupled perception design. Instead of having the LLM directly predict box coordinates, we feed the output boxes from a universal proposal network into the LLM, allowing it to output the corresponding box indices to represent its detection results, turning the regression task into a retrieval-based task that LLM handles more proficiently. From the data perspective, we build a fully automated data engine and construct the Rexverse-2M dataset which possesses multiple granularities to support the joint training of perception and understanding. After a three-stage training approach, ChatRex demonstrates strong perception and understanding performance, and the combination of these two capabilities also unlocks many attractive applications, demonstrating their complementary roles in MLLM. Code is available at https://github.com/IDEA-Research/ChatRex.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08461v1">FastCache: Optimizing Multimodal LLM Serving through Lightweight KV-Cache Compression Framework</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 14 pages, 14 figures
    </div>
    <details class="paper-abstract">
      Multi-modal Large Language Models (MLLMs) serving systems commonly employ KV-cache compression to reduce memory footprint. However, existing compression methods introduce significant processing overhead and queuing delays, particularly in concurrent serving scenarios. We present \texttt{FastCache}, a novel serving framework that effectively addresses these challenges through two key innovations: (1) a dynamic batching strategy that optimizes request scheduling across prefill, compression, and decode stages, and (2) an efficient KV-cache memory pool mechanism that eliminates memory fragmentation while maintaining high GPU utilization. Our comprehensive experiments on the GQA and MileBench datasets demonstrate that \texttt{FastCache} achieves up to 19.3$\times$ reduction in Time-To-First-Token (TTFT) and 12.1$\times$ improvement in throughput compared to state-of-the-art baselines. The system maintains stable performance under high-concurrency scenarios (up to 40 req/s) while reducing average memory consumption by 20\%. These results establish \texttt{FastCache} as an efficient solution for real-world LLM serving systems with KV-cache compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.19338v2">Decoding Echo Chambers: LLM-Powered Simulations Revealing Polarization in Social Networks</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Accepted by COLING 2025
    </div>
    <details class="paper-abstract">
      The impact of social media on critical issues such as echo chambers needs to be addressed, as these phenomena can have disruptive consequences for our society. Traditional research often oversimplifies emotional tendencies and opinion evolution into numbers and formulas, neglecting that news and communication are conveyed through text, which limits these approaches. Hence, in this work, we propose an LLM-based simulation for the social opinion network to evaluate and counter polarization phenomena. We first construct three typical network structures to simulate different characteristics of social interactions. Then, agents interact based on recommendation algorithms and update their strategies through reasoning and analysis. By comparing these interactions with the classic Bounded Confidence Model (BCM), the Friedkin Johnsen (FJ) model, and using echo chamber-related indices, we demonstrate the effectiveness of our framework in simulating opinion dynamics and reproducing phenomena such as opinion polarization and echo chambers. We propose two mitigation methods, active and passive nudges, that can help reduce echo chambers, specifically within language-based simulations. We hope our work will offer valuable insights and guidance for social polarization mitigation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08404v1">Fact-checking with Generative AI: A Systematic Cross-Topic Examination of LLMs Capacity to Detect Veracity of Political Information</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 15 pages, 2 figures
    </div>
    <details class="paper-abstract">
      The purpose of this study is to assess how large language models (LLMs) can be used for fact-checking and contribute to the broader debate on the use of automated means for veracity identification. To achieve this purpose, we use AI auditing methodology that systematically evaluates performance of five LLMs (ChatGPT 4, Llama 3 (70B), Llama 3.1 (405B), Claude 3.5 Sonnet, and Google Gemini) using prompts regarding a large set of statements fact-checked by professional journalists (16,513). Specifically, we use topic modeling and regression analysis to investigate which factors (e.g. topic of the prompt or the LLM type) affect evaluations of true, false, and mixed statements. Our findings reveal that while ChatGPT 4 and Google Gemini achieved higher accuracy than other models, overall performance across models remains modest. Notably, the results indicate that models are better at identifying false statements, especially on sensitive topics such as COVID-19, American political controversies, and social issues, suggesting possible guardrails that may enhance accuracy on these topics. The major implication of our findings is that there are significant challenges for using LLMs for factchecking, including significant variation in performance across different LLMs and unequal quality of outputs for specific topics which can be attributed to deficits of training data. Our research highlights the potential and limitations of LLMs in political fact-checking, suggesting potential avenues for further improvements in guardrails as well as fine-tuning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08311v1">Mind the Memory Gap: Unveiling GPU Bottlenecks in Large-Batch LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 Pol G. Recasens, Ferran Agullo: equal contribution
    </div>
    <details class="paper-abstract">
      Large language models have been widely adopted across different tasks, but their auto-regressive generation nature often leads to inefficient resource utilization during inference. While batching is commonly used to increase throughput, performance gains plateau beyond a certain batch size, especially with smaller models, a phenomenon that existing literature typically explains as a shift to the compute-bound regime. In this paper, through an in-depth GPU-level analysis, we reveal that large-batch inference remains memory-bound, with most GPU compute capabilities underutilized due to DRAM bandwidth saturation as the primary bottleneck. To address this, we propose a Batching Configuration Advisor (BCA) that optimizes memory allocation, reducing GPU memory requirements with minimal impact on throughput. The freed memory and underutilized GPU compute capabilities can then be leveraged by concurrent workloads. Specifically, we use model replication to improve serving throughput and GPU utilization. Our findings challenge conventional assumptions about LLM inference, offering new insights and practical strategies for improving resource utilization, particularly for smaller language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01306v2">Emotion-Aware Embedding Fusion in LLMs (Flan-T5, LLAMA 2, DeepSeek-R1, and ChatGPT 4) for Intelligent Response Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Empathetic and coherent responses are critical in auto-mated chatbot-facilitated psychotherapy. This study addresses the challenge of enhancing the emotional and contextual understanding of large language models (LLMs) in psychiatric applications. We introduce Emotion-Aware Embedding Fusion, a novel framework integrating hierarchical fusion and attention mechanisms to prioritize semantic and emotional features in therapy transcripts. Our approach combines multiple emotion lexicons, including NRC Emotion Lexicon, VADER, WordNet, and SentiWordNet, with state-of-the-art LLMs such as Flan-T5, LLAMA 2, DeepSeek-R1, and ChatGPT 4. Therapy session transcripts, comprising over 2,000 samples are segmented into hierarchical levels (word, sentence, and session) using neural networks, while hierarchical fusion combines these features with pooling techniques to refine emotional representations. Atten-tion mechanisms, including multi-head self-attention and cross-attention, further prioritize emotional and contextual features, enabling temporal modeling of emotion-al shifts across sessions. The processed embeddings, computed using BERT, GPT-3, and RoBERTa are stored in the Facebook AI similarity search vector database, which enables efficient similarity search and clustering across dense vector spaces. Upon user queries, relevant segments are retrieved and provided as context to LLMs, enhancing their ability to generate empathetic and con-textually relevant responses. The proposed framework is evaluated across multiple practical use cases to demonstrate real-world applicability, including AI-driven therapy chatbots. The system can be integrated into existing mental health platforms to generate personalized responses based on retrieved therapy session data.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08195v1">Dialogue Injection Attack: Jailbreaking LLMs through Context Manipulation</a></div>
    <div class="paper-meta">
      📅 2025-03-11
      | 💬 17 pages, 10 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated significant utility in a wide range of applications; however, their deployment is plagued by security vulnerabilities, notably jailbreak attacks. These attacks manipulate LLMs to generate harmful or unethical content by crafting adversarial prompts. While much of the current research on jailbreak attacks has focused on single-turn interactions, it has largely overlooked the impact of historical dialogues on model behavior. In this paper, we introduce a novel jailbreak paradigm, Dialogue Injection Attack (DIA), which leverages the dialogue history to enhance the success rates of such attacks. DIA operates in a black-box setting, requiring only access to the chat API or knowledge of the LLM's chat template. We propose two methods for constructing adversarial historical dialogues: one adapts gray-box prefilling attacks, and the other exploits deferred responses. Our experiments show that DIA achieves state-of-the-art attack success rates on recent LLMs, including Llama-3.1 and GPT-4o. Additionally, we demonstrate that DIA can bypass 5 different defense mechanisms, highlighting its robustness and effectiveness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/pdf/2503.07550v1">KSOD: Knowledge Supplement for LLMs On Demand</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks, yet still produce errors in domain-specific tasks. To further improve their performance, we propose KSOD (Knowledge Supplement for LLMs On Demand), a novel framework that empowers LLMs to improve their capabilities with knowledge-based supervised fine-tuning (SFT). KSOD analyzes the causes of errors from the perspective of knowledge deficiency by identifying potential missing knowledge in LLM that may lead to the errors. Subsequently, KSOD tunes a knowledge module on knowledge dataset and verifies whether the LLM lacks the identified knowledge based on it. If the knowledge is verified, KSOD supplements the LLM with the identified knowledge using the knowledge module. Tuning LLMs on specific knowledge instead of specific task decouples task and knowledge and our experiments on two domain-specific benchmarks and four general benchmarks empirically demonstrate that KSOD enhances the performance of LLMs on tasks requiring the supplemented knowledge while preserving their performance on other tasks. Our findings shed light on the potential of improving the capabilities of LLMs with knowledge-based SFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="https://arxiv.org/abs/2410.05440v3">Can LLMs Understand Time Series Anomalies?</a></div>
    <div class="paper-meta">
      📅 2025-03-11
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained popularity in time series forecasting, but their potential for anomaly detection remains largely unexplored. Our study investigates whether LLMs can understand and detect anomalies in time series data, focusing on zero-shot and few-shot scenarios. Inspired by conjectures about LLMs' behavior from time series forecasting research, we formulate key hypotheses about LLMs' capabilities in time series anomaly detection. We design and conduct principled experiments to test each of these hypotheses. Our investigation reveals several surprising findings about LLMs for time series: (1) LLMs understand time series better as images rather than as text, (2) LLMs do not demonstrate enhanced performance when prompted to engage in explicit reasoning about time series analysis. (3) Contrary to common beliefs, LLMs' understanding of time series does not stem from their repetition biases or arithmetic abilities. (4) LLMs' behaviors and performance in time series analysis vary significantly across different models. This study provides the first comprehensive analysis of contemporary LLM capabilities in time series anomaly detection. Our results suggest that while LLMs can understand trivial time series anomalies, we have no evidence that they can understand more subtle real-world anomalies. Many common conjectures based on their reasoning capabilities do not hold. All synthetic dataset generators, final prompts, and evaluation scripts have been made available in https://github.com/rose-stl-lab/anomllm.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19518v2">Assessing LLMs for Front-end Software Architecture Knowledge</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 4 pages, 1 figure, to appear in the International Workshop on Designing Software at ICSE 2025
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated significant promise in automating software development tasks, yet their capabilities with respect to software design tasks remains largely unclear. This study investigates the capabilities of an LLM in understanding, reproducing, and generating structures within the complex VIPER architecture, a design pattern for iOS applications. We leverage Bloom's taxonomy to develop a comprehensive evaluation framework to assess the LLM's performance across different cognitive domains such as remembering, understanding, applying, analyzing, evaluating, and creating. Experimental results, using ChatGPT 4 Turbo 2024-04-09, reveal that the LLM excelled in higher-order tasks like evaluating and creating, but faced challenges with lower-order tasks requiring precise retrieval of architectural details. These findings highlight both the potential of LLMs to reduce development costs and the barriers to their effective application in real-world software design scenarios. This study proposes a benchmark format for assessing LLM capabilities in software architecture, aiming to contribute toward more robust and accessible AI-driven development tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04691v2">Quantifying the Reasoning Abilities of LLMs on Real-world Clinical Cases</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Recent advancements in reasoning-enhanced large language models (LLMs), such as DeepSeek-R1 and OpenAI-o3, have demonstrated significant progress. However, their application in professional medical contexts remains underexplored, particularly in evaluating the quality of their reasoning processes alongside final outputs. Here, we introduce MedR-Bench, a benchmarking dataset of 1,453 structured patient cases, annotated with reasoning references derived from clinical case reports. Spanning 13 body systems and 10 specialties, it includes both common and rare diseases. To comprehensively evaluate LLM performance, we propose a framework encompassing three critical examination recommendation, diagnostic decision-making, and treatment planning, simulating the entire patient care journey. To assess reasoning quality, we present the Reasoning Evaluator, a novel automated system that objectively scores free-text reasoning responses based on efficiency, actuality, and completeness using dynamic cross-referencing and evidence checks. Using this benchmark, we evaluate five state-of-the-art reasoning LLMs, including DeepSeek-R1, OpenAI-o3-mini, and Gemini-2.0-Flash Thinking, etc. Our results show that current LLMs achieve over 85% accuracy in relatively simple diagnostic tasks when provided with sufficient examination results. However, performance declines in more complex tasks, such as examination recommendation and treatment planning. While reasoning outputs are generally reliable, with factuality scores exceeding 90%, critical reasoning steps are frequently missed. These findings underscore both the progress and limitations of clinical LLMs. Notably, open-source models like DeepSeek-R1 are narrowing the gap with proprietary systems, highlighting their potential to drive accessible and equitable advancements in healthcare.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.13778v2">Explainable XR: Understanding User Behaviors of XR Environments using LLM-assisted Analytics Framework</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 11 pages, 8 figures. This is the author's version of the article that has been accepted for publication in IEEE Transactions on Visualization and Computer Graphics
    </div>
    <details class="paper-abstract">
      We present Explainable XR, an end-to-end framework for analyzing user behavior in diverse eXtended Reality (XR) environments by leveraging Large Language Models (LLMs) for data interpretation assistance. Existing XR user analytics frameworks face challenges in handling cross-virtuality - AR, VR, MR - transitions, multi-user collaborative application scenarios, and the complexity of multimodal data. Explainable XR addresses these challenges by providing a virtuality-agnostic solution for the collection, analysis, and visualization of immersive sessions. We propose three main components in our framework: (1) A novel user data recording schema, called User Action Descriptor (UAD), that can capture the users' multimodal actions, along with their intents and the contexts; (2) a platform-agnostic XR session recorder, and (3) a visual analytics interface that offers LLM-assisted insights tailored to the analysts' perspectives, facilitating the exploration and analysis of the recorded XR session data. We demonstrate the versatility of Explainable XR by demonstrating five use-case scenarios, in both individual and collaborative XR applications across virtualities. Our technical evaluation and user studies show that Explainable XR provides a highly usable analytics solution for understanding user actions and delivering multifaceted, actionable insights into user behaviors in immersive environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07885v1">Safety Guardrails for LLM-Enabled Robots</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Although the integration of large language models (LLMs) into robotics has unlocked transformative capabilities, it has also introduced significant safety concerns, ranging from average-case LLM errors (e.g., hallucinations) to adversarial jailbreaking attacks, which can produce harmful robot behavior in real-world settings. Traditional robot safety approaches do not address the novel vulnerabilities of LLMs, and current LLM safety guardrails overlook the physical risks posed by robots operating in dynamic real-world environments. In this paper, we propose RoboGuard, a two-stage guardrail architecture to ensure the safety of LLM-enabled robots. RoboGuard first contextualizes pre-defined safety rules by grounding them in the robot's environment using a root-of-trust LLM, which employs chain-of-thought (CoT) reasoning to generate rigorous safety specifications, such as temporal logic constraints. RoboGuard then resolves potential conflicts between these contextual safety specifications and a possibly unsafe plan using temporal logic control synthesis, which ensures safety compliance while minimally violating user preferences. Through extensive simulation and real-world experiments that consider worst-case jailbreaking attacks, we demonstrate that RoboGuard reduces the execution of unsafe plans from 92% to below 2.5% without compromising performance on safe plans. We also demonstrate that RoboGuard is resource-efficient, robust against adaptive attacks, and significantly enhanced by enabling its root-of-trust LLM to perform CoT reasoning. These results underscore the potential of RoboGuard to mitigate the safety risks and enhance the reliability of LLM-enabled robots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.16810v2">How Data Inter-connectivity Shapes LLMs Unlearning: A Structural Unlearning Perspective</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      While unlearning knowledge from large language models (LLMs) is receiving increasing attention, one important aspect remains unexplored. Existing approaches and benchmarks assume data points to-be-forgotten are independent, ignoring their inter-connectivity - a fundamental characteristic of real-world data structures. In this paper, we propose PISTOL, a method for compiling structural datasets. PISTOL leverages the inherently structured nature of contractual relationships, offering several key benefits. First, it enables insights into the impact of structural data on unlearning effectiveness. Second, it provides precise and concise ground truths for clearer evaluation. Third, its attribute generation does not require input from pre-trained LLMs, mitigating confounding risks. Leveraging datasets synthesized using PISTOL, we demonstrate how data inter-connectivity impacts LLM unlearning. Specifically, (a) in both the pre-trained and fine-tuned models, unlearning difficulty increases as data inter-connectivity grows, (b) there is a positive correlation between the density of the knowledge graph and unlearning difficulty, and (c) when the to-be-forgotten data is skewed towards one domain, balancing retaining performance across all domains is challenging.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07833v1">HalluVerse25: Fine-grained Multilingual Benchmark Dataset for LLM Hallucinations</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly used in various contexts, yet remain prone to generating non-factual content, commonly referred to as "hallucinations". The literature categorizes hallucinations into several types, including entity-level, relation-level, and sentence-level hallucinations. However, existing hallucination datasets often fail to capture fine-grained hallucinations in multilingual settings. In this work, we introduce HalluVerse25, a multilingual LLM hallucination dataset that categorizes fine-grained hallucinations in English, Arabic, and Turkish. Our dataset construction pipeline uses an LLM to inject hallucinations into factual biographical sentences, followed by a rigorous human annotation process to ensure data quality. We evaluate several LLMs on HalluVerse25, providing valuable insights into how proprietary models perform in detecting LLM-generated hallucinations across different contexts.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.11863v2">LEMMo-Plan: LLM-Enhanced Learning from Multi-Modal Demonstration for Planning Sequential Contact-Rich Manipulation Tasks</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have gained popularity in task planning for long-horizon manipulation tasks. To enhance the validity of LLM-generated plans, visual demonstrations and online videos have been widely employed to guide the planning process. However, for manipulation tasks involving subtle movements but rich contact interactions, visual perception alone may be insufficient for the LLM to fully interpret the demonstration. Additionally, visual data provides limited information on force-related parameters and conditions, which are crucial for effective execution on real robots. In this paper, we introduce an in-context learning framework that incorporates tactile and force-torque information from human demonstrations to enhance LLMs' ability to generate plans for new task scenarios. We propose a bootstrapped reasoning pipeline that sequentially integrates each modality into a comprehensive task plan. This task plan is then used as a reference for planning in new task configurations. Real-world experiments on two different sequential manipulation tasks demonstrate the effectiveness of our framework in improving LLMs' understanding of multi-modal demonstrations and enhancing the overall planning performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.07036v2">Automated Consistency Analysis of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-10
      | 💬 10 pages, 12 figures, 3 tables, 3 algorithms, 2024 IEEE 6th International Conference on Trust, Privacy and Security in Intelligent Systems, and Applications (TPS-ISA), Washington, DC, USA
    </div>
    <details class="paper-abstract">
      Generative AI (Gen AI) with large language models (LLMs) are being widely adopted across the industry, academia and government. Cybersecurity is one of the key sectors where LLMs can be and/or are already being used. There are a number of problems that inhibit the adoption of trustworthy Gen AI and LLMs in cybersecurity and such other critical areas. One of the key challenge to the trustworthiness and reliability of LLMs is: how consistent an LLM is in its responses? In this paper, we have analyzed and developed a formal definition of consistency of responses of LLMs. We have formally defined what is consistency of responses and then develop a framework for consistency evaluation. The paper proposes two approaches to validate consistency: self-validation, and validation across multiple LLMs. We have carried out extensive experiments for several LLMs such as GPT4oMini, GPT3.5, Gemini, Cohere, and Llama3, on a security benchmark consisting of several cybersecurity questions: informational and situational. Our experiments corroborate the fact that even though these LLMs are being considered and/or already being used for several cybersecurity tasks today, they are often inconsistent in their responses, and thus are untrustworthy and unreliable for cybersecurity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07556v1">Junior Software Developers' Perspectives on Adopting LLMs for Software Engineering: a Systematic Literature Review</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Many studies exploring the adoption of Large Language Model-based tools for software development by junior developers have emerged in recent years. These studies have sought to understand developers' perspectives about using those tools, a fundamental pillar for successfully adopting LLM-based tools in Software Engineering. The aim of this paper is to provide an overview of junior software developers' perspectives and use of LLM-based tools for software engineering (LLM4SE). We conducted a systematic literature review (SLR) following guidelines by Kitchenham et al. on 56 primary studies, applying the definition for junior software developers as software developers with equal or less than five years of experience, including Computer Science/Software Engineering students. We found that the majority of the studies focused on comprehending the different aspects of integrating AI tools in SE. Only 8.9\% of the studies provide a clear definition for junior software developers, and there is no uniformity. Searching for relevant information is the most common task using LLM tools. ChatGPT was the most common LLM tool present in the studies (and experiments). A majority of the studies (83.9\%) report both positive and negative perceptions about the impact of adopting LLM tools. We also found and categorised advantages, challenges, and recommendations regarding LLM adoption. Our results indicate that developers are using LLMs not just for code generation, but also to improve their development skills. Critically, they are not just experiencing the benefits of adopting LLM tools, but they are also aware of at least a few LLM limitations, such as the generation of wrong suggestions, potential data leaking, and AI hallucination. Our findings offer implications for software engineering researchers, educators, and developers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07550v1">KSOD: Knowledge Supplement for LLMs On Demand</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks, yet still produce errors in domain-specific tasks. To further improve their performance, we propose KSOD (Knowledge Supplement for LLMs On Demand), a novel framework that empowers LLMs to improve their capabilities with knowledge-based supervised fine-tuning (SFT). KSOD analyzes the causes of errors from the perspective of knowledge deficiency by identifying potential missing knowledge in LLM that may lead to the errors. Subsequently, KSOD tunes a knowledge module on knowledge dataset and verifies whether the LLM lacks the identified knowledge based on it. If the knowledge is verified, KSOD supplements the LLM with the identified knowledge using the knowledge module. Tuning LLMs on specific knowledge instead of specific task decouples task and knowledge and our experiments on two domain-specific benchmarks and four general benchmarks empirically demonstrate that KSOD enhances the performance of LLMs on tasks requiring the supplemented knowledge while preserving their performance on other tasks. Our findings shed light on the potential of improving the capabilities of LLMs with knowledge-based SFT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07545v1">Queueing, Predictions, and LLMs: Challenges and Open Problems</a></div>
    <div class="paper-meta">
      📅 2025-03-10
    </div>
    <details class="paper-abstract">
      Queueing systems present many opportunities for applying machine-learning predictions, such as estimated service times, to improve system performance. This integration raises numerous open questions about how predictions can be effectively leveraged to improve scheduling decisions. Recent studies explore queues with predicted service times, typically aiming to minimize job time in the system. We review these works, highlight the effectiveness of predictions, and present open questions on queue performance. We then move to consider an important practical example of using predictions in scheduling, namely Large Language Model (LLM) systems, which presents novel scheduling challenges and highlights the potential for predictions to improve performance. In particular, we consider LLMs performing inference. Inference requests (jobs) in LLM systems are inherently complex; they have variable inference times, dynamic memory footprints that are constrained by key-value (KV) store memory limitations, and multiple possible preemption approaches that affect performance differently. We provide background on the important aspects of scheduling in LLM systems, and introduce new models and open problems that arise from them. We argue that there are significant opportunities for applying insights and analysis from queueing theory to scheduling in LLM systems.
    </details>
</div>
