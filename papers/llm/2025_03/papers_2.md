# llm - 2025_03

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- Part 2
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- [Part 6](papers_6.md)
- [Part 7](papers_7.md)
- [Part 8](papers_8.md)
- [Part 9](papers_9.md)
- [Part 10](papers_10.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18320v1">Bridging Writing Manner Gap in Visual Instruction Tuning by Creating LLM-aligned Instructions</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      In the realm of Large Multi-modal Models (LMMs), the instruction quality during the visual instruction tuning stage significantly influences the performance of modality alignment. In this paper, we assess the instruction quality from a unique perspective termed \textbf{Writing Manner}, which encompasses the selection of vocabulary, grammar and sentence structure to convey specific semantics. We argue that there exists a substantial writing manner gap between the visual instructions and the base Large Language Models (LLMs) within LMMs. This gap forces the pre-trained base LLMs to deviate from their original writing styles, leading to capability degradation of both base LLMs and LMMs. To bridge the writing manner gap while preserving the original semantics, we propose directly leveraging the base LLM to align the writing manner of soft-format visual instructions with that of the base LLM itself, resulting in novel LLM-aligned instructions. The manual writing manner evaluation results demonstrate that our approach successfully minimizes the writing manner gap. By utilizing LLM-aligned instructions, the baseline models LLaVA-7B and QwenVL demonstrate enhanced resistance to hallucinations and non-trivial comprehensive improvements across all $15$ visual and language benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18316v1">Knowledge Transfer from LLMs to Provenance Analysis: A Semantic-Augmented Method for APT Detection</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Advanced Persistent Threats (APTs) have caused significant losses across a wide range of sectors, including the theft of sensitive data and harm to system integrity. As attack techniques grow increasingly sophisticated and stealthy, the arms race between cyber defenders and attackers continues to intensify. The revolutionary impact of Large Language Models (LLMs) has opened up numerous opportunities in various fields, including cybersecurity. An intriguing question arises: can the extensive knowledge embedded in LLMs be harnessed for provenance analysis and play a positive role in identifying previously unknown malicious events? To seek a deeper understanding of this issue, we propose a new strategy for taking advantage of LLMs in provenance-based threat detection. In our design, the state-of-the-art LLM offers additional details in provenance data interpretation, leveraging their knowledge of system calls, software identity, and high-level understanding of application execution context. The advanced contextualized embedding capability is further utilized to capture the rich semantics of event descriptions. We comprehensively examine the quality of the resulting embeddings, and it turns out that they offer promising avenues. Subsequently, machine learning models built upon these embeddings demonstrated outstanding performance on real-world data. In our evaluation, supervised threat detection achieves a precision of 99.0%, and semi-supervised anomaly detection attains a precision of 96.9%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18313v1">DeepFund: Will LLM be Professional at Fund Investment? A Live Arena Perspective</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 Work in progress
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated impressive capabilities across various domains, but their effectiveness in financial decision making, particularly in fund investment, remains inadequately evaluated. Current benchmarks primarily assess LLMs understanding of financial documents rather than their ability to manage assets or analyze trading opportunities in dynamic market conditions. A critical limitation in existing evaluation methodologies is the backtesting approach, which suffers from information leakage when LLMs are evaluated on historical data they may have encountered during pretraining. This paper introduces DeepFund, a comprehensive platform for evaluating LLM based trading strategies in a simulated live environment. Our approach implements a multi agent framework where LLMs serve as both analysts and managers, creating a realistic simulation of investment decision making. The platform employs a forward testing methodology that mitigates information leakage by evaluating models on market data released after their training cutoff dates. We provide a web interface that visualizes model performance across different market conditions and investment parameters, enabling detailed comparative analysis. Through DeepFund, we aim to provide a more accurate and fair assessment of LLMs capabilities in fund investment, offering insights into their potential real world applications in financial markets.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18305v1">Enhancing LLM-based Code Translation in Repository Context via Triple Knowledge-Augmented</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have behaved well in function-level code translation without repository-level context. However, the performance of LLMs in repository-level context code translation remains suboptimal due to complex dependencies and context, hindering their adoption in industrial settings. In this work, we propose a novel LLM-based code translation technique K-Trans, which leverages triple knowledge augmentation to enhance LLM's translation quality under repository context in real-world software development. First, K-Trans constructs a translation knowledge base by extracting relevant information from target-language codebases, the repository being translated, and prior translation results. Second, for each function to be translated, K-Trans retrieves relevant triple knowledge, including target-language code samples, dependency usage examples, and successful translation function pairs, serving as references to enhance LLM for translation. Third, K-Trans constructs a knowledge-augmented translation prompt using the retrieved triple knowledge and employs LLMs to generate the translated code while preserving repository context. It further leverages LLMs for self-debugging, enhancing translation correctness. The experiments show that K-Trans substantially outperforms the baseline adapted from previous work by 19.4%/40.2% relative improvement in pass@1 and 0.138 in CodeBLEU. It is important to note that the results also demonstrate that each knowledge significantly contributes to K-Trans's effectiveness in handling repository-level context code translation, with dependency usage examples making the most notable contribution. Moreover, as the self-evolution process progresses, the knowledge base continuously enhances the LLM's performance across various aspects of the repository-level code translation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18293v1">Fact-checking AI-generated news reports: Can LLMs catch their own lies?</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      In this paper, we evaluate the ability of Large Language Models (LLMs) to assess the veracity of claims in ''news reports'' generated by themselves or other LLMs. Our goal is to determine whether LLMs can effectively fact-check their own content, using methods similar to those used to verify claims made by humans. Our findings indicate that LLMs are more effective at assessing claims in national or international news stories than in local news stories, better at evaluating static information than dynamic information, and better at verifying true claims compared to false ones. We hypothesize that this disparity arises because the former types of claims are better represented in the training data. Additionally, we find that incorporating retrieved results from a search engine in a Retrieval-Augmented Generation (RAG) setting significantly reduces the number of claims an LLM cannot assess. However, this approach also increases the occurrence of incorrect assessments, partly due to irrelevant or low-quality search results. This diagnostic study highlights the need for future research on fact-checking machine-generated reports to prioritize improving the precision and relevance of retrieved information to better support fact-checking efforts. Furthermore, claims about dynamic events and local news may require human-in-the-loop fact-checking systems to ensure accuracy and reliability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18292v1">Jenga: Effective Memory Management for Serving LLM with Heterogeneity</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 16 pages, 19 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are widely used but expensive to run, especially as inference workloads grow. To lower costs, maximizing the request batch size by managing GPU memory efficiently is crucial. While PagedAttention has recently been proposed to improve the efficiency of memory management, we find that the growing heterogeneity in the embeddings dimensions, attention, and access patterns of modern LLM architectures introduces new challenges for memory allocation. In this paper, we present Jenga, a novel memory allocation framework for heterogeneous embeddings in LLMs. Jenga tackles two key challenges: (1) minimizing memory fragmentation when managing embeddings of different sizes, and (2) enabling flexible caching and eviction policies tailored to the specific token-dependency patterns of various layers. Jenga employs a two-level memory allocator, leveraging the least common multiple (LCM) of embedding sizes to optimize memory usage and providing APIs to express layer-specific caching logic to enhance memory reuse. We implemente Jenga on vLLM, a state-of-the-art LLM inference engine, and evaluate it with diverse LLMs, datasets, and GPU configurations. Evaluations show that Jenga improves GPU memory utilization by up to 79.6%, and increases serving throughput by up to 4.92x (1.80x on average).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18273v1">Analyzing Islamophobic Discourse Using Semi-Coded Terms and LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Islamophobia started evolving into a global phenomenon by attracting followers across the globe, particularly in Western societies. Thus, understanding Islamophobia's global spread and online dissemination is crucial. This paper performs a large-scale analysis of specialized, semi-coded Islamophobic terms such as (muzrat, pislam, mudslime, mohammedan, muzzies) floated on extremist social platforms, i.e., 4Chan, Gab, Telegram, etc. First, we use large language models (LLMs) to show their ability to understand these terms. Second, using Google Perspective API, we also find that Islamophobic text is more toxic compared to other kinds of hate speech. Finally, we use BERT topic modeling approach to extract different topics and Islamophobic discourse on these social platforms. Our findings indicate that LLMs understand these Out-Of-Vocabulary (OOV) slurs; however, measures are still required to control such discourse. Our topic modeling also indicates that Islamophobic text is found across various political, conspiratorial, and far-right movements and is particularly directed against Muslim immigrants. Taken altogether, we performed the first study on Islamophobic semi-coded terms and shed a global light on Islamophobia.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15299v2">Inside-Out: Hidden Factual Knowledge in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      This work presents a framework for assessing whether large language models (LLMs) encode more factual knowledge in their parameters than what they express in their outputs. While a few studies hint at this possibility, none has clearly defined or demonstrated this phenomenon. We first propose a formal definition of knowledge, quantifying it for a given question as the fraction of correct-incorrect answer pairs where the correct one is ranked higher. This gives rise to external and internal knowledge, depending on the information used to score individual answer candidates: either the model's observable token-level probabilities or its intermediate computations. Hidden knowledge arises when internal knowledge exceeds external knowledge. We then present a case study, applying this framework to three popular open-weights LLMs in a closed-book QA setup. Our results indicate that: (1) LLMs consistently encode more factual knowledge internally than what they express externally, with an average relative gap of 40%. (2) Surprisingly, some knowledge is so deeply hidden that a model can internally know an answer perfectly, yet fail to generate it even once, despite large-scale repeated sampling of 1,000 answers. This reveals fundamental limitations in the generation capabilities of LLMs, which (3) put a practical constraint on scaling test-time compute via repeated answer sampling in closed-book QA: significant performance improvements remain inaccessible because some answers are practically never sampled, yet if they were, we would be guaranteed to rank them first.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19217v1">LLM Benchmarking with LLaMA2: Evaluating Code Development Performance Across Multiple Programming Languages</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      The rapid evolution of large language models (LLMs) has opened new possibilities for automating various tasks in software development. This paper evaluates the capabilities of the Llama 2-70B model in automating these tasks for scientific applications written in commonly used programming languages. Using representative test problems, we assess the model's capacity to generate code, documentation, and unit tests, as well as its ability to translate existing code between commonly used programming languages. Our comprehensive analysis evaluates the compilation, runtime behavior, and correctness of the generated and translated code. Additionally, we assess the quality of automatically generated code, documentation and unit tests. Our results indicate that while Llama 2-70B frequently generates syntactically correct and functional code for simpler numerical tasks, it encounters substantial difficulties with more complex, parallelized, or distributed computations, requiring considerable manual corrections. We identify key limitations and suggest areas for future improvements to better leverage AI-driven automation in scientific computing workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19182v1">Evaluating Bias in LLMs for Job-Resume Matching: Gender, Race, and Education</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 NAACL 2025: Industry Track
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) offer the potential to automate hiring by matching job descriptions with candidate resumes, streamlining recruitment processes, and reducing operational costs. However, biases inherent in these models may lead to unfair hiring practices, reinforcing societal prejudices and undermining workplace diversity. This study examines the performance and fairness of LLMs in job-resume matching tasks within the English language and U.S. context. It evaluates how factors such as gender, race, and educational background influence model decisions, providing critical insights into the fairness and reliability of LLMs in HR applications. Our findings indicate that while recent models have reduced biases related to explicit attributes like gender and race, implicit biases concerning educational background remain significant. These results highlight the need for ongoing evaluation and the development of advanced bias mitigation strategies to ensure equitable hiring practices when using LLMs in industry settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19114v1">Understanding and Improving Information Preservation in Prompt Compression for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 21 pages, 6 figures, 23 tables
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have enabled their successful application to a broad range of tasks. However, in information-intensive tasks, the prompt length can grow fast, leading to increased computational requirements, performance degradation, and induced biases from irrelevant or redundant information. Recently, various prompt compression techniques have been introduced to optimize the trade-off between reducing input length and retaining performance. We propose a holistic evaluation framework that allows for in-depth analysis of prompt compression methods. We focus on three key aspects, besides compression ratio: (i) downstream task performance, (ii) grounding in the input context, and (iii) information preservation. Through this framework, we investigate state-of-the-art soft and hard compression methods, showing that they struggle to preserve key details from the original prompt, limiting their performance on complex tasks. We demonstrate that modifying soft prompting methods to control better the granularity of the compressed information can significantly improve their effectiveness -- up to +23\% in downstream task performance, more than +8 BERTScore points in grounding, and 2.7x more entities preserved in compression.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19092v1">Rankers, Judges, and Assistants: Towards Understanding the Interplay of LLMs in Information Retrieval Evaluation</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are increasingly integral to information retrieval (IR), powering ranking, evaluation, and AI-assisted content creation. This widespread adoption necessitates a critical examination of potential biases arising from the interplay between these LLM-based components. This paper synthesizes existing research and presents novel experiment designs that explore how LLM-based rankers and assistants influence LLM-based judges. We provide the first empirical evidence of LLM judges exhibiting significant bias towards LLM-based rankers. Furthermore, we observe limitations in LLM judges' ability to discern subtle system performance differences. Contrary to some previous findings, our preliminary study does not find evidence of bias against AI-generated content. These results highlight the need for a more holistic view of the LLM-driven information ecosystem. To this end, we offer initial guidelines and a research agenda to ensure the reliable use of LLMs in IR evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19090v1">LLM-Based Insight Extraction for Contact Center Analytics and Cost-Efficient Deployment</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Large Language Models have transformed the Contact Center industry, manifesting in enhanced self-service tools, streamlined administrative processes, and augmented agent productivity. This paper delineates our system that automates call driver generation, which serves as the foundation for tasks such as topic modeling, incoming call classification, trend detection, and FAQ generation, delivering actionable insights for contact center agents and administrators to consume. We present a cost-efficient LLM system design, with 1) a comprehensive evaluation of proprietary, open-weight, and fine-tuned models and 2) cost-efficient strategies, and 3) the corresponding cost analysis when deployed in production environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.17167v2">QualityFlow: An Agentic Workflow for Program Synthesis Controlled by LLM Quality Checks</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      We introduce QualityFlow, a dynamic agentic workflow for program synthesis. Given the English description of a programming problem and a set of unit tests, the model's goal is to synthesize the correct program that solves the problem and passes the tests. QualityFlow includes large language model (LLM) agents resembling a software development team, including code generation, testing, and self-debugging. We propose the LLM Quality Checker, which explicitly "imagines" whether the synthesized programs' execution would conform to the unit tests. The Quality Checks dynamically control the workflow, including actions to submit the final answer, clarify the problem statement, and revert previous workflow steps. Our experiments show that the Quality Checker can precisely accept any correct program, mitigate faulty synthesized tests, and prevent potential workflow deviation. QualityFlow establishes the state-of-the-art results on four program synthesis benchmarks: MBPP, HumanEval, and stricter evaluations from MBPP-EvalPlus and HumanEval-EvalPlus.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13773v2">Mitigating KV Cache Competition to Enhance User Experience in LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      In Large Language Model (LLM) serving, the KV-cache (KVC) bottleneck causes high tail Time-to-First-Token (TTFT) and Time-Between-Tokens (TBT), impairing user experience, particularly in time-sensitive applications. However, satisfying both TTFT and TBT service-level objectives (SLOs) is challenging. To address this, we propose a system, named CacheOPT for mitigating KV Cache competition, based on key insights from our measurements, incorporating novel components. First, it estimates a request's output length, bounding the deviation with a high specified probability, adjusted based on the request arrival rate. Second, it allocates the estimated KVC demand to a request, and reuses other requests' allocated KVC to avoid preemptions while reducing waiting time. Third, it proactively allocates KVC before instead of at the time a request exhausts its allocation and reserves KVC globally to prevent preemptions. Fourth, it chooses a request that has long TBT SLO, long job remaining time and short preemption time to preempt. Fifth, it selects the shortest-latency strategy between swapping and recomputation for preemptions. Experiments show that CacheOPT achieves up to 3.29$\times$ and 2.83$\times$ lower tail TBT and tail TTFT, 47\% and 53\% higher TTFT and TBT SLO attainments, and supports up to 1.58$\times$ higher request arrival rate than the state-of-the-art methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06364v2">EconoServe: Maximizing Multi-Resource Utilization with SLO Guarantees in LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-03-24
      | 💬 14 pages
    </div>
    <details class="paper-abstract">
      As Large Language Models (LLMs) continue to grow, reducing costs and alleviating GPU demands has become increasingly critical. However, existing schedulers primarily target either GPU compute or Key-Value Cache (KVC) utilization, failing to fully optimize both GPU compute and KVC usage during each iteration or guarantee timely KVC allocations when needed. To address these challenges, we conducted a trace-based experimental analysis and made insightful observations, leading to the design of a system called EconoServe. EconoServe maximizes multi-resource utilization while ensuring service-level objective (SLO) guarantees in LLM serving. To enable adding prompts to a batch to maximize GPU utilization in each iteration, EconoServe maintains separate waiting queues for prompt processing tasks (PTs) and generation tasks (GTs). It batches GTs with the same predicted response lengths (RL) to save scheduling time and allocates KVC space for the predicted RL to avoid KVC allocation failures. It further has a novel KVC pipelining method, allowing sharing allocated but unused KVC space to enhance KVC utilization. In addition, it prioritizes queued requests that occupy more KVC to release KVC earlier and satisfy request service-level-objective (SLO). Experimental results demonstrate that EconoServe increases throughput by up to 4$\times$ with the same level of latency, generates up to 91\% lower job completion time and up to 91\% higher SLO satisfaction ratio compared to vLLM. It also reduces the number of GPUs used in DistServe by up to 78\% while maintaining the same level of goodput.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.19007v1">Option Discovery Using LLM-guided Semantic Hierarchical Reinforcement Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-24
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown remarkable promise in reasoning and decision-making, yet their integration with Reinforcement Learning (RL) for complex robotic tasks remains underexplored. In this paper, we propose an LLM-guided hierarchical RL framework, termed LDSC, that leverages LLM-driven subgoal selection and option reuse to enhance sample efficiency, generalization, and multi-task adaptability. Traditional RL methods often suffer from inefficient exploration and high computational cost. Hierarchical RL helps with these challenges, but existing methods often fail to reuse options effectively when faced with new tasks. To address these limitations, we introduce a three-stage framework that uses LLMs for subgoal generation given natural language description of the task, a reusable option learning and selection method, and an action-level policy, enabling more effective decision-making across diverse tasks. By incorporating LLMs for subgoal prediction and policy guidance, our approach improves exploration efficiency and enhances learning performance. On average, LDSC outperforms the baseline by 55.9\% in average reward, demonstrating its effectiveness in complex RL settings. More details and experiment videos could be found in \href{https://raaslab.org/projects/LDSC/}{this link\footnote{https://raaslab.org/projects/LDSC}}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18178v1">The Power of Small LLMs in Geometry Generation for Physical Simulations</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 24 pages, 17 figures
    </div>
    <details class="paper-abstract">
      Engineers widely rely on simulation platforms like COMSOL or ANSYS to model and optimise processes. However, setting up such simulations requires expertise in defining geometry, generating meshes, establishing boundary conditions, and configuring solvers. This research aims to simplify this process by enabling engineers to describe their setup in plain language, allowing a Large Language Model (LLM) to generate the necessary input files for their specific application. This novel approach allows establishing a direct link between natural language and complex engineering tasks. Building on previous work that evaluated various LLMs for generating input files across simple and complex geometries, this study demonstrates that small LLMs - specifically, Phi-3 Mini and Qwen-2.5 1.5B - can be fine-tuned to generate precise engineering geometries in GMSH format. Through Low-Rank Adaptation (LoRA), we curated a dataset of 480 instruction-output pairs encompassing simple shapes (squares, rectangles, circles, and half circles) and more complex structures (I-beams, cylindrical pipes, and bent pipes). The fine-tuned models produced high-fidelity outputs, handling routine geometry generation with minimal intervention. While challenges remain with geometries involving combinations of multiple bodies, this study demonstrates that fine-tuned small models can outperform larger models like GPT-4o in specialised tasks, offering a precise and resource-efficient alternative for engineering applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12334v2">When neural implant meets multimodal LLM: A dual-loop system for neuromodulation and naturalistic neuralbehavioral research</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      We propose a novel dual-loop system that synergistically combines responsive neurostimulation (RNS) implants with artificial intelligence-driven wearable devices for treating post-traumatic stress disorder (PTSD) and enabling naturalistic brain research. In PTSD Therapy Mode, an implanted closed-loop neural device monitors amygdala activity and provides on-demand stimulation upon detecting pathological theta oscillations, while an ensemble of wearables (smart glasses, smartwatches, smartphones) uses multimodal large language model (LLM) analysis of sensory data to detect environmental or physiological PTSD triggers and deliver timely audiovisual interventions. Logged events from both the neural and wearable loops are analyzed to personalize trigger detection and progressively transition patients to non-invasive interventions. In Neuroscience Research Mode, the same platform is adapted for real-world brain activity capture. Wearable-LLM systems recognize naturalistic events (social interactions, emotional situations, compulsive behaviors, decision making) and signal implanted RNS devices (via wireless triggers) to record synchronized intracranial data during these moments. This approach builds on recent advances in mobile intracranial EEG recording and closed-loop neuromodulation in humans (BRAIN Initiative, 2023) (Mobbs et al., 2021). We discuss how our interdisciplinary system could revolutionize PTSD therapy and cognitive neuroscience by enabling 24/7 monitoring, context-aware intervention, and rich data collection outside traditional labs. The vision is a future where AI-enhanced devices continuously collaborate with the human brain, offering therapeutic support and deep insights into neural function, with the resulting real-world context rich neural data, in turn, accelerating the development of more biologically-grounded and human-centric AI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.15793v2">DNA Bench: When Silence is Smarter -- Benchmarking Over-Reasoning in Reasoning LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Test-time scaling has significantly improved large language model performance, enabling deeper reasoning to solve complex problems. However, this increased reasoning capability also leads to excessive token generation and unnecessary problem-solving attempts. We introduce Don\'t Answer Bench (DNA Bench), a new benchmark designed to evaluate LLMs ability to robustly understand the tricky reasoning triggers and avoiding unnecessary generation. DNA Bench consists of 150 adversarially designed prompts that are easy for humans to understand and respond to, but surprisingly not for many of the recent prominent LLMs. DNA Bench tests models abilities across different capabilities, such as instruction adherence, hallucination avoidance, redundancy filtering, and unanswerable question recognition. We evaluate reasoning LLMs (RLMs), including DeepSeek-R1, OpenAI O3-mini, Claude-3.7-sonnet and compare them against a powerful non-reasoning model, e.g., GPT-4o. Our experiments reveal that RLMs generate up to 70x more tokens than necessary, often failing at tasks that simpler non-reasoning models handle efficiently with higher accuracy. Our findings underscore the need for more effective training and inference strategies in RLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12018v2">Atom of Thoughts for Markov LLM Test-Time Scaling</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) achieve superior performance through training-time scaling, and test-time scaling further enhances their capabilities by conducting effective reasoning during inference. However, as the scale of reasoning increases, existing test-time scaling methods suffer from accumulated historical information, which not only wastes computational resources but also interferes with effective reasoning. To address this issue, we observe that complex reasoning can be achieved by solving a series of independent and self-contained subquestions. These subquestions are essentially \textit{atomic questions}, exhibiting the memoryless property similar to Markov processes. Based on this observation, we propose Atom of Thoughts (\our), where each state transition consists of decomposing the current question into a dependency-based directed acyclic graph and contracting its subquestions, forming a simplified question that maintains answer equivalence with the original problem. This answer preservation enables the iterative \textit{decomposition-contraction} process to naturally form a meaningful Markov reasoning process. Furthermore, these atomic states can be seamlessly integrated into existing test-time scaling methods, enabling \our to serve as a plug-in enhancement for improving reasoning capabilities. Experiments across six benchmarks demonstrate the effectiveness of \our both as a standalone framework and a plug-in enhancement. Notably, on HotpotQA, when applied to gpt-4o-mini, \our achieves an \textbf{80.6\%} F1 score, surpassing o3-mini by \textbf{3.4\%} and DeepSeek-R1 by \textbf{10.6\%}. The code is available at \href{https://github.com/qixucen/atom}{https://github.com/qixucen/atom}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18141v1">AGIR: Assessing 3D Gait Impairment with Reasoning based on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Assessing gait impairment plays an important role in early diagnosis, disease monitoring, and treatment evaluation for neurodegenerative diseases. Despite its widespread use in clinical practice, it is limited by subjectivity and a lack of precision. While recent deep learning-based approaches have consistently improved classification accuracies, they often lack interpretability, hindering their utility in clinical decision-making. To overcome these challenges, we introduce AGIR, a novel pipeline consisting of a pre-trained VQ-VAE motion tokenizer and a subsequent Large Language Model (LLM) fine-tuned over pairs of motion tokens and Chain-of-Thought (CoT) reasonings. To fine-tune an LLM for pathological gait analysis, we first introduce a multimodal dataset by adding rationales dedicated to MDS-UPDRS gait score assessment to an existing PD gait dataset. We then introduce a two-stage supervised fine-tuning (SFT) strategy to enhance the LLM's motion comprehension with pathology-specific knowledge. This strategy includes: 1) a generative stage that aligns gait motions with analytic descriptions through bidirectional motion-description generation, 2) a reasoning stage that integrates logical Chain-of-Thought (CoT) reasoning for impairment assessment with UPDRS gait score. Validation on an existing dataset and comparisons with state-of-the-art methods confirm the robustness and accuracy of our pipeline, demonstrating its ability to assign gait impairment scores from motion input with clinically meaningful rationales.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18129v1">GeoBenchX: Benchmarking LLMs for Multistep Geospatial Tasks</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 Github with code and benchmark set: https://github.com/Solirinai/GeoBenchX
    </div>
    <details class="paper-abstract">
      In this paper, we establish a benchmark for evaluating large language models (LLMs) on multi-step geospatial tasks relevant to commercial GIS practitioners. We assess seven leading commercial LLMs (Sonnet 3.5 and 3.7, Haiku 3.5, Gemini 2.0, GPT-4o, GPT-4o mini, and o3-mini) using a simple tool-calling agent equipped with 23 geospatial functions. Our benchmark comprises tasks across four categories of increasing complexity, with both solvable and intentionally unsolvable tasks to test hallucination rejection. We develop an LLM-as-Judge evaluation framework to compare agent solutions against reference implementations. Results show Sonnet 3.5 and GPT-4o achieve the best overall performance, with Claude models excelling on solvable tasks while OpenAI models better identify unsolvable scenarios. We observe significant differences in token usage, with Anthropic models consuming substantially more tokens than competitors. Common errors include misunderstanding geometrical relationships, relying on outdated knowledge, and inefficient data manipulation. The resulting benchmark set, evaluation framework, and data generation pipeline are released as open-source resources, providing one more standardized method for ongoing evaluation of LLMs for GeoAI.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.14504v2">Aligning Multimodal LLM with Human Preference: A Survey</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 Project page: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Alignment
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) can handle a wide variety of general tasks with simple prompts, without the need for task-specific training. Multimodal Large Language Models (MLLMs), built upon LLMs, have demonstrated impressive potential in tackling complex tasks involving visual, auditory, and textual data. However, critical issues related to truthfulness, safety, o1-like reasoning, and alignment with human preference remain insufficiently addressed. This gap has spurred the emergence of various alignment algorithms, each targeting different application scenarios and optimization goals. Recent studies have shown that alignment algorithms are a powerful approach to resolving the aforementioned challenges. In this paper, we aim to provide a comprehensive and systematic review of alignment algorithms for MLLMs. Specifically, we explore four key aspects: (1) the application scenarios covered by alignment algorithms, including general image understanding, multi-image, video, and audio, and extended multimodal applications; (2) the core factors in constructing alignment datasets, including data sources, model responses, and preference annotations; (3) the benchmarks used to evaluate alignment algorithms; and (4) a discussion of potential future directions for the development of alignment algorithms. This work seeks to help researchers organize current advancements in the field and inspire better alignment methods. The project page of this paper is available at https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Alignment.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18072v1">On the effectiveness of LLMs for automatic grading of open-ended questions in Spanish</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Grading is a time-consuming and laborious task that educators must face. It is an important task since it provides feedback signals to learners, and it has been demonstrated that timely feedback improves the learning process. In recent years, the irruption of LLMs has shed light on the effectiveness of automatic grading. In this paper, we explore the performance of different LLMs and prompting techniques in automatically grading short-text answers to open-ended questions. Unlike most of the literature, our study focuses on a use case where the questions, answers, and prompts are all in Spanish. Experimental results comparing automatic scores to those of human-expert evaluators show good outcomes in terms of accuracy, precision and consistency for advanced LLMs, both open and proprietary. Results are notably sensitive to prompt styles, suggesting biases toward certain words or content in the prompt. However, the best combinations of models and prompt strategies, consistently surpasses an accuracy of 95% in a three-level grading task, which even rises up to more than 98% when the it is simplified to a binary right or wrong rating problem, which demonstrates the potential that LLMs have to implement this type of automation in education applications.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18018v1">Lost in Cultural Translation: Do LLMs Struggle with Math Across Cultural Contexts?</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced various fields, particularly coding, mathematical reasoning, and logical problem solving. However, a critical question remains: Do these mathematical reasoning abilities persist when LLMs are presented with culturally adapted math problems? Specifically, how do LLMs perform when faced with math problems embedded in cultural contexts that have no significant representation in main stream web-scale AI training data? To explore this, we generated six synthetic cultural datasets from GSM8K, a widely used benchmark for assessing LLMs' mathematical reasoning skills. While preserving the mathematical logic and numerical values of the original GSM8K test set, we modify cultural elements such as personal names, food items, place names, etc. These culturally adapted datasets provide a more reliable framework for evaluating LLMs' mathematical reasoning under shifting cultural contexts. Our findings reveal that LLMs struggle with math problems when cultural references change, even though the underlying mathematical structure remains constant. Smaller models exhibit greater performance drops compared to larger models. Interestingly, our results also suggest that cultural familiarity can enhance mathematical reasoning. Even models with no explicit mathematical training but exposure to relevant cultural contexts sometimes outperform larger, mathematically proficient models on culturally embedded math problems. This study highlights the impact of cultural context on the mathematical reasoning abilities of LLMs, underscoring the need for more diverse and representative training data to improve robustness in real-world applications. The benchmark data sets and script for reproducing the results are available at https://github.com/akarim23131/Lost_in_Cultural_Translation
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17994v1">Instructing the Architecture Search for Spatial-temporal Sequence Forecasting with LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Spatial-temporal sequence forecasting (STSF) is a long-standing research problem with widespread real-world applications. Neural architecture search (NAS), which automates the neural network design, has been shown effective in tackling the STSF problem. However, the existing NAS methods for STSF focus on generating architectures in a time-consuming data-driven fashion, which heavily limits their ability to use background knowledge and explore the complicated search trajectory. Large language models (LLMs) have shown remarkable ability in decision-making with comprehensive internal world knowledge, but how it could benefit NAS for STSF remains unexplored. In this paper, we propose a novel NAS method for STSF based on LLM. Instead of directly generate architectures with LLM, We inspire the LLM's capability with a multi-level enhancement mechanism. Specifically, on the step-level, we decompose the generation task into decision steps with powerful prompt engineering and inspire LLM to serve as instructor for architecture search based on its internal knowledge. On the instance-level, we utilize a one-step tuning framework to quickly evaluate the architecture instance and a memory bank to cumulate knowledge to improve LLM's search ability. On the task-level, we propose a two-stage architecture search, balancing the exploration stage and optimization stage, to reduce the possibility of being trapped in local optima. Extensive experimental results demonstrate that our method can achieve competitive effectiveness with superior efficiency against existing NAS methods for STSF.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2404.13558v3">LASER: Tuning-Free LLM-Driven Attention Control for Efficient Text-conditioned Image-to-Animation</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Revolutionary advancements in text-to-image models have unlocked new dimensions for sophisticated content creation, such as text-conditioned image editing, enabling the modification of existing images based on textual guidance. This capability allows for the generation of diverse images that convey highly complex visual concepts. However, existing methods primarily focus on generating new images from text-image pairs and struggle to produce fine-grained animations from existing images and textual guidance without fine-tuning. In this paper, we introduce LASER, a tuning-free LLM-driven attention control framework that follows a progressive process: LLM planning, feature-attention injection, and stable animation generation. LASER leverages a large language model (LLM) to refine general descriptions into fine-grained prompts, guiding pre-trained text-to-image models to generate aligned keyframes with subtle variations. The LLM also generates control signals for feature and attention injections, enabling seamless text-guided image morphing for various transformations without additional fine-tuning. By using the same initial noise inversion from the input image, LASER receives LLM-controlled injections during denoising and leverages interpolated text embeddings to produce a series of coherent animation frames. We propose a Text-conditioned Image-to-Animation Benchmark to validate the effectiveness and efficacy of LASER. Extensive experiments demonstrate that LASER achieves impressive results in consistent and efficient animation generation, establishing it as a powerful tool for producing detailed animations and opening new avenues in digital content creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.08493v2">Intelligent LiDAR Navigation: Leveraging External Information and Semantic Maps with LLM as Copilot</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Traditional robot navigation systems primarily utilize occupancy grid maps and laser-based sensing technologies, as demonstrated by the popular move_base package in ROS. Unlike robots, humans navigate not only through spatial awareness and physical distances but also by integrating external information, such as elevator maintenance updates from public notification boards and experiential knowledge, like the need for special access through certain doors. With the development of Large Language Models (LLMs), which possesses text understanding and intelligence close to human performance, there is now an opportunity to infuse robot navigation systems with a level of understanding akin to human cognition. In this study, we propose using osmAG (Area Graph in OpensStreetMap textual format), an innovative semantic topometric hierarchical map representation, to bridge the gap between the capabilities of ROS move_base and the contextual understanding offered by LLMs. Our methodology employs LLMs as an actual copilot in robot navigation, enabling the integration of a broader range of informational inputs while maintaining the robustness of traditional robotic navigation systems. Our code, demo, map, experiment results can be accessed at https://github.com/xiexiexiaoxiexie/Intelligent-LiDAR-Navigation-LLM-as-Copilot.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17965v1">Understanding the Effects of RLHF on the Quality and Detectability of LLM-Generated Texts</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 14 pages, 3 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated exceptional performance on a range of downstream NLP tasks by generating text that closely resembles human writing. However, the ease of achieving this similarity raises concerns from potential malicious uses at scale by bad actors, as LLM-generated text becomes increasingly difficult to discern from human text. Although detection methods have been developed to address this issue, bad actors can further manipulate LLM-generated texts to make them less detectable. In this work, we study how further editing texts with Reinforcement Learning from Human Feedback (RLHF), which aligns model outputs with human preferences, affects (a) the quality of generated texts for two tasks, and (b) the performance of LLM-generated text detectors, looking at both training-based and zero-shot detection methods. Although RLHF improves the quality of LLM-generated texts, we find that it also tends to produce more detectable, lengthy, and repetitive outputs. Additionally, we observe that training-based detectors are vulnerable to short texts and to texts that incorporate code, whereas zero-shot detectors exhibit greater robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11355v3">Nuclear Deployed: Analyzing Catastrophic Risks in Decision-making of Autonomous LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 Please visit https://llm-catastrophic-risks.github.io for a quick tour of our research. Our code is available at https://github.com/pillowsofwind/LLM-CBRN-Risks
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) are evolving into autonomous decision-makers, raising concerns about catastrophic risks in high-stakes scenarios, particularly in Chemical, Biological, Radiological and Nuclear (CBRN) domains. Based on the insight that such risks can originate from trade-offs between the agent's Helpful, Harmlessness and Honest (HHH) goals, we build a novel three-stage evaluation framework, which is carefully constructed to effectively and naturally expose such risks. We conduct 14,400 agentic simulations across 12 advanced LLMs, with extensive experiments and analysis. Results reveal that LLM agents can autonomously engage in catastrophic behaviors and deception, without being deliberately induced. Furthermore, stronger reasoning abilities often increase, rather than mitigate, these risks. We also show that these agents can violate instructions and superior commands. On the whole, we empirically prove the existence of catastrophic risks in autonomous LLM agents. We release our code to foster further research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17953v1">Smoke and Mirrors: Jailbreaking LLM-based Code Generation via Implicit Malicious Prompts</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      The proliferation of Large Language Models (LLMs) has revolutionized natural language processing and significantly impacted code generation tasks, enhancing software development efficiency and productivity. Notably, LLMs like GPT-4 have demonstrated remarkable proficiency in text-to-code generation tasks. However, the growing reliance on LLMs for code generation necessitates a critical examination of the safety implications associated with their outputs. Existing research efforts have primarily focused on verifying the functional correctness of LLMs, overlooking their safety in code generation. This paper introduces a jailbreaking approach, CodeJailbreaker, designed to uncover safety concerns in LLM-based code generation. The basic observation is that existing safety mechanisms for LLMs are built through the instruction-following paradigm, where malicious intent is explicitly articulated within the instruction of the prompt. Consequently, CodeJailbreaker explores to construct a prompt whose instruction is benign and the malicious intent is implicitly encoded in a covert channel, i.e., the commit message, to bypass the safety mechanism. Experiments on the recently-released RMCBench benchmark demonstrate that CodeJailbreaker markedly surpasses the conventional jailbreaking strategy, which explicitly conveys malicious intents in the instructions, in terms of the attack effectiveness across three code generation tasks. This study challenges the traditional safety paradigms in LLM-based code generation, emphasizing the need for enhanced safety measures in safeguarding against implicit malicious cues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17922v1">WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      With the advancements in long-context inference capabilities of large language models (LLMs), the KV cache has become one of the foundational components. However, its substantial GPU memory consumption makes KV cache compression a key technique for enabling efficient LLM inference in industrial scenarios. While recent studies have focused on optimizing the memory occupied by the KV cache, they overlook two critical factors: preserving semantic coherence and considering task-specific characteristic during compression. To address these limitations, we propose a novel task-adaptive KV cache window selection method, WindowKV. WindowKV dynamically selects local semantic windows consisting of consecutive tokens, according to task-specific characteristics, ensuring the retained KV cache captures continuous, essential context. Additionally, we introduce an intra-group layer KV cache indices sharing strategy to reduce computational overhead, achieving a balance between performance and efficiency. We rigorously evaluate WindowKV on the LongBench benchmark, and the results demonstrate that it maintains a performance comparable to full KV cache retention while using only 12% of the original KV cache, significantly reducing memory requirements. Furthermore, our method also achieves state-of-the-art results in the Needle-in-a-Haystack evaluation, highlighting its effectiveness and robustness.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18995v1">LLMs in the Classroom: Outcomes and Perceptions of Questions Written with the Aid of AI</a></div>
    <div class="paper-meta">
      📅 2025-03-23
      | 💬 Accepted to AAAI 2025 Technical Track on AI Alignment
    </div>
    <details class="paper-abstract">
      We randomly deploy questions constructed with and without use of the LLM tool and gauge the ability of the students to correctly answer, as well as their ability to correctly perceive the difference between human-authored and LLM-authored questions. In determining whether the questions written with the aid of ChatGPT were consistent with the instructor's questions and source text, we computed representative vectors of both the human and ChatGPT questions using SBERT and compared cosine similarity to the course textbook. A non-significant Mann-Whitney U test (z = 1.018, p = .309) suggests that students were unable to perceive whether questions were written with or without the aid of ChatGPT. However, student scores on LLM-authored questions were almost 9% lower (z = 2.702, p < .01). This result may indicate that either the AI questions were more difficult or that the students were more familiar with the instructor's style of questions. Overall, the study suggests that while there is potential for using LLM tools to aid in the construction of assessments, care must be taken to ensure that the questions are fair, well-composed, and relevant to the course material.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18991v1">SRMIR: Shadow Reward Models Based on Introspective Reasoning for LLM Alignment</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Aligning large language models (LLMs) with human preferences and values is vital for application. However, current alignment methods face three main limitations: (1) reliance on costly human annotation; (2) alignment tax; (3) shallow alignment vulnerable to jailbreak attacks. Additionally, current alignment datasets often suffer from uneven distributions, leading to overrepresentation of some topics and neglect of others. To address these issues, we propose SRMIR (Shadow Reward Models Based on Introspective Reasoning), inspired by shadow models in membership inference attacks. We first construct a balanced safety Chain of Draft (CoD) dataset across $7$ harmful types with structured prompt leveraging the introspective reasoning capabilities of LLMs, then train a set of specialized reward models to guide policy optimization through Group Relative Policy Optimization (GRPO). We apply two strategies, linear combination and categorized approach, to integrate shadow reward models for policy optimization. By comparison, we find that the latter achieves superior alignment despite higher computational costs. Experiments across several LLMs demonstrate SRMIR significantly outperforms existing methods.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18986v1">SplitFrozen: Split Learning with Device-side Model Frozen for Fine-Tuning LLM on Heterogeneous Resource-Constrained Devices</a></div>
    <div class="paper-meta">
      📅 2025-03-23
    </div>
    <details class="paper-abstract">
      Fine-tuning large language models (LLMs) on private, on-device data can empower tailored personalized AI agents. However, fine-tuning LLMs on resource-constrained edge devices faces significant challenges, including excessive computation overhead, device heterogeneity, and data imbalance. This paper proposes SplitFrozen, a split learning framework that enables efficient LLM fine-tuning by strategically freezing device-side model layers while centralizing parameter-efficient fine-tuning on the server. Our framework partitions LLMs into device-side frozen layers and server-side fine-tuning layers, where heterogeneous resource-constrained devices execute only forward propagation. To minimize server-side training costs, we integrate Low-Rank Adaptation (LoRA) into the server-side layers. A pipeline parallelism strategy further optimizes training efficiency by decoupling device-server computations and leveraging decomposed backward propagation. Experiments on GPT-2 with the MRPC, MNLI-matched, and SST-2 datasets demonstrate that SplitFrozen outperforms FedLoRA and SplitLoRA by 69.4\% model accuracy under extremely imbalanced data, while reducing up to 86.8\% device-side computations and 50.2\% total training time. Experiments also validate the scalability of SplitFrozen on content generation task using Llama-3.2 model on GSM8K dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17885v1">Reasoning with LLMs for Zero-Shot Vulnerability Detection</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Automating software vulnerability detection (SVD) remains a critical challenge in an era of increasingly complex and interdependent software systems. Despite significant advances in Large Language Models (LLMs) for code analysis, prevailing evaluation methodologies often lack the \textbf{context-aware robustness} necessary to capture real-world intricacies and cross-component interactions. To address these limitations, we present \textbf{VulnSage}, a comprehensive evaluation framework and a dataset curated from diverse, large-scale open-source system software projects developed in C/C++. Unlike prior datasets, it leverages a heuristic noise pre-filtering approach combined with LLM-based reasoning to ensure a representative and minimally noisy spectrum of vulnerabilities. The framework supports multi-granular analysis across function, file, and inter-function levels and employs four diverse zero-shot prompt strategies: Baseline, Chain-of-Thought, Think, and Think & Verify. Through this evaluation, we uncover that structured reasoning prompts substantially improve LLM performance, with Think & Verify reducing ambiguous responses from 20.3% to 9.1% while increasing accuracy. We further demonstrate that code-specialized models consistently outperform general-purpose alternatives, with performance varying significantly across vulnerability types, revealing that no single approach universally excels across all security contexts. Link to dataset and codes: https://github.com/Erroristotle/VulnSage.git
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17882v1">Think Before Refusal : Triggering Safety Reflection in LLMs to Mitigate False Refusal Behavior</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 18 pages, 23 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) have demonstrated that fine-tuning and human alignment can render LLMs harmless. In practice, such "harmlessness" behavior is mainly achieved by training models to reject harmful requests, such as "Explain how to burn down my neighbor's house", where the model appropriately declines to respond. However, this approach can inadvertently result in false refusal, where models reject benign queries as well, such as "Tell me how to kill a Python process". In this work, we demonstrate that prompting safety reflection before generating a response can mitigate false refusal behavior. Building on this finding, we introduce the Think-Before-Refusal (TBR) schema and conduct safety-aware instruction fine-tuning incorporating safety reflection. In an ablation study across 15 pre-trained models, we show that models fine-tuned with safety reflection significantly reduce false refusal behavior while maintaining safety and overall performance compared to those fine-tuned without safety reflection.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17213v6">Plurals: A System for Guiding LLMs Via Simulated Social Ensembles</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 CHI 2025
    </div>
    <details class="paper-abstract">
      Recent debates raised concerns that language models may favor certain viewpoints. But what if the solution is not to aim for a 'view from nowhere' but rather to leverage different viewpoints? We introduce Plurals, a system and Python library for pluralistic AI deliberation. Plurals consists of Agents (LLMs, optionally with personas) which deliberate within customizable Structures, with Moderators overseeing deliberation. Plurals is a generator of simulated social ensembles. Plurals integrates with government datasets to create nationally representative personas, includes deliberation templates inspired by deliberative democracy, and allows users to customize both information-sharing structures and deliberation behavior within Structures. Six case studies demonstrate fidelity to theoretical constructs and efficacy. Three randomized experiments show simulated focus groups produced output resonant with an online sample of the relevant audiences (chosen over zero-shot generation in 75% of trials). Plurals is both a paradigm and a concrete system for pluralistic AI. The Plurals library is available at https://github.com/josh-ashkinaze/plurals and will be continually updated.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.20576v3">Smart Routing: Cost-Effective Multi-LLM Serving in AIOS</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) are increasingly deployed as service endpoints in systems, the surge in query volume creates significant scheduling challenges. Existing scheduling frameworks mainly target at latency optimization while neglecting the capability of LLMs to serve different level of queries, which could lead to computational resource waste. For example, those simple queries can be safely handled by small, fast and cheap LLMs, while those complex and difficult queries need to be handled by large, slow, and expensive LLMs. This paper addresses this challenge by proposing an efficient capability-cost coordinated scheduling framework, ECCOS, for multi-LLM serving, which explicitly constrains response quality and workload to optimize LLM inference cost. Specifically, it introduces the two-stage scheduling by designing a multi-objective predictor and a constrained optimizer. The predictor estimates both model capabilities and computational costs through training-based and retrieval-based approaches, while the optimizer determines cost-optimal assignments under quality and workload constraints. It also introduces QAServe, a dataset for sample-wise response quality and costs collected by zero-shot prompting different LLMs on knowledge QA and mathematical reasoning. Extensive experiments demonstrate that ECCOS improves success rates by 6.30% while reducing costs by 10.15% compared to existing methods, consuming less than 0.5% of LLM response time. The code is available at: https://github.com/agiresearch/ECCOS, and the proposed smart routing mechanism has been integrated into AIOS, the AI Agent Operating System, at https://github.com/agiresearch/AIOS.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.05400v3">Dynamic Noise Preference Optimization for LLM Self-Improvement via Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Although LLMs have achieved significant success, their reliance on large volumes of human-annotated data has limited their potential for further scaling. In this situation, utilizing self-generated synthetic data has become crucial for fine-tuning LLMs without extensive human annotation. However, current methods often fail to ensure consistent improvements across iterations, with performance stagnating after only minimal updates. To overcome these challenges, we introduce Dynamic Noise Preference Optimization (DNPO). DNPO employs a dynamic sample labeling mechanism to construct preference pairs for training and introduces controlled, trainable noise into the preference optimization process. Our approach effectively prevents stagnation and enables continuous improvement. In experiments with Zephyr-7B, DNPO consistently outperforms existing methods, showing an average performance boost of 2.6% across multiple benchmarks. Additionally, DNPO shows a significant improvement in model-generated data quality, with a 29.4% win-loss rate gap compared to the baseline in GPT-4 evaluations. This highlights its effectiveness in enhancing model performance through iterative refinement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.21526v2">Not All LLM-Generated Data Are Equal: Rethinking Data Weighting in Text Classification</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 ICLR 2025 camera ready
    </div>
    <details class="paper-abstract">
      Synthetic data augmentation via large language models (LLMs) allows researchers to leverage additional training data, thus enhancing the performance of downstream tasks, especially when real-world data is scarce. However, the generated data can deviate from the real-world data, and this misalignment can bring deficient outcomes while applying the trained model to applications. Therefore, we proposed efficient weighted-loss approaches to align synthetic data with real-world distribution by emphasizing high-quality and diversified data generated by LLMs with using merely a little real-world data. We empirically assessed the effectiveness of our method on multiple text classification tasks, and the results showed leveraging our approaches on a BERT-level model robustly outperformed standard cross-entropy and other data weighting approaches, providing potential solutions to effectively leveraging synthetic data from any suitable data generator for model training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17793v1">Every Sample Matters: Leveraging Mixture-of-Experts and High-Quality Data for Efficient and Accurate Code LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 20 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Recent advancements in code large language models (LLMs) have demonstrated remarkable capabilities in code generation and understanding. It is still challenging to build a code LLM with comprehensive performance yet ultimate efficiency. Many attempts have been released in the open source community to break the trade-off between performance and efficiency, such as the Qwen Coder series and the DeepSeek Coder series. This paper introduces yet another attempt in this area, namely Ling-Coder-Lite. We leverage the efficient Mixture-of-Experts (MoE) architecture along with a set of high-quality data curation methods (especially those based on program analytics) to build an efficient yet powerful code LLM. Ling-Coder-Lite exhibits on-par performance on 12 representative coding benchmarks compared to state-of-the-art models of similar size, such as Qwen2.5-Coder-7B and DeepSeek-Coder-V2-Lite, while offering competitive latency and throughput. In practice, we achieve a 50\% reduction in deployment resources compared to the similar-sized dense model without performance loss. To facilitate further research and development in this area, we open-source our models as well as a substantial portion of high-quality data for the annealing and post-training stages. The models and data can be accessed at~\url{https://huggingface.co/inclusionAI/Ling-Coder-lite}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17783v1">Energy-Aware LLMs: A step towards sustainable AI for downstream applications</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 This work has been submitted to V. International Conference on Electrical, Computer and Energy Technologies (ICECET 2025) for possible publication
    </div>
    <details class="paper-abstract">
      Advanced Large Language Models (LLMs) have revolutionized various fields, including communication networks, sparking an innovation wave that has led to new applications and services, and significantly enhanced solution schemes. Despite all these impressive developments, most LLMs typically require huge computational resources, resulting in terribly high energy consumption. Thus, this research study proposes an end-to-end pipeline that investigates the trade-off between energy efficiency and model performance for an LLM during fault ticket analysis in communication networks. It further evaluates the pipeline performance using two real-world datasets for the tasks of root cause analysis and response feedback in a communication network. Our results show that an appropriate combination of quantization and pruning techniques is able to reduce energy consumption while significantly improving model performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.14744v5">Exploring Prosocial Irrationality for LLM Agents: A Social Cognition View</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 Published as a conference paper at ICLR 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have been shown to face hallucination issues due to the data they trained on often containing human bias; whether this is reflected in the decision-making process of LLM Agents remains under-explored. As LLM Agents are increasingly employed in intricate social environments, a pressing and natural question emerges: Can we utilize LLM Agents' systematic hallucinations to mirror human cognitive biases, thus exhibiting irrational social intelligence? In this paper, we probe the irrational behavior among contemporary LLM Agents by melding practical social science experiments with theoretical insights. Specifically, We propose CogMir, an open-ended Multi-LLM Agents framework that utilizes hallucination properties to assess and enhance LLM Agents' social intelligence through cognitive biases. Experimental results on CogMir subsets show that LLM Agents and humans exhibit high consistency in irrational and prosocial decision-making under uncertain conditions, underscoring the prosociality of LLM Agents as social entities and highlighting the significance of hallucination properties. Additionally, the CogMir framework demonstrates its potential as a valuable platform for encouraging more research into the social intelligence of LLM Agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17755v1">Improving Preference Extraction In LLMs By Identifying Latent Knowledge Through Classifying Probes</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 preprint, submitted to ACL ARR 2025, 21 pages, 23 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are often used as automated judges to evaluate text, but their effectiveness can be hindered by various unintentional biases. We propose using linear classifying probes, trained by leveraging differences between contrasting pairs of prompts, to directly access LLMs' latent knowledge and extract more accurate preferences. Through extensive experiments using models of varying size from four different families and six diverse datasets assessing text quality evaluation and common sense reasoning, we demonstrate that both supervised and unsupervised probing approaches consistently outperform traditional generation-based judgement while maintaining similar computational costs. These probes generalise under domain shifts and can even outperform finetuned evaluators with the same training data size. Our results suggest linear probing offers an accurate, robust and computationally efficient approach for LLM-as-judge tasks while providing interpretable insights into how models encode judgement-relevant knowledge. Our data and code will be openly released in the future.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17741v1">RustMap: Towards Project-Scale C-to-Rust Migration via Program Analysis and LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Migrating existing C programs into Rust is increasingly desired, as Rust offers superior memory safety while maintaining C's high performance. However, vastly different features between C and Rust--e.g., distinct definitions and usages of pointers and references--pose significant challenges beyond mere syntactic translation. Existing automated translation tools, such as C2Rust, may rely too much on syntactic, template-based translation and generate unsafe Rust code that is hard for human developers to read, maintain, or even compile. More semantic-aware translation that produces safer, idiomatic, and runnable Rust code is much needed. This paper introduces a novel dependency-guided and large language model (LLM)-based C-to-Rust translation approach, RustMap, based on three key ideas: (1) Utilize LLM capabilities to produce idiomatic Rust code from given small pieces of C code, (2) Mitigate LLM limitations in handling large codebases by breaking project-scale C programs into smaller units for translation according to their usage dependencies and composing them into a runnable Rust program, and (3) Enhance the correctness of the translated Rust program by using test cases to check input/output equivalence, isolate faulty code when execution states deviate, and iteratively refine the translation using feedback from compilation and test errors. We empirically evaluate RustMap on 126 real-world programs, including 125 from Rosetta Code and a 7000+ line bzip2 implementation using GPT-4o as the LLM. RustMap shows promising results, guiding GPT-4o to produce idiomatic, readable, and functional Rust code with significantly less unsafe code than other tools, and revealing non-trivial translation patterns reusable for future research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2403.07376v2">NavCoT: Boosting LLM-Based Vision-and-Language Navigation via Learning Disentangled Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 Accepted by TPAMI 2025
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN), as a crucial research problem of Embodied AI, requires an embodied agent to navigate through complex 3D environments following natural language instructions. Recent research has highlighted the promising capacity of large language models (LLMs) in VLN by improving navigational reasoning accuracy and interpretability. However, their predominant use in an offline manner usually suffers from substantial domain gap between the VLN task and the LLM training corpus. This paper introduces a novel strategy called Navigational Chain-of-Thought (NavCoT), where we fulfill parameter-efficient in-domain training to enable self-guided navigational decision, leading to a significant mitigation of the domain gap in a cost-effective manner. Specifically, at each timestep, the LLM is prompted to forecast the navigational chain-of-thought by: 1) acting as a world model to imagine the next observation according to the instruction, 2) selecting the candidate observation that best aligns with the imagination, and 3) determining the action based on the reasoning from the prior steps. Through constructing formalized labels for training, the LLM can learn to generate desired and reasonable chain-of-thought outputs for improving the action decision. Experimental results across various training settings and popular VLN benchmarks (e.g., Room-to-Room (R2R), Room-across-Room (RxR), Room-for-Room (R4R)) show the significant superiority of NavCoT over the direct action prediction variants. Through simple parameter-efficient finetuning, our NavCoT outperforms a recent GPT4-based approach with ~7% relative improvement on the R2R dataset. We believe that NavCoT will help unlock more task-adaptive and scalable LLM-based embodied agents, which are helpful for developing real-world robotics applications. Code is available at https://github.com/expectorlin/NavCoT.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2405.07920v3">Rank-DistiLLM: Closing the Effectiveness Gap Between Cross-Encoders and LLMs for Passage Re-Ranking</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Cross-encoders distilled from large language models (LLMs) are often more effective re-rankers than cross-encoders fine-tuned on manually labeled data. However, distilled models do not match the effectiveness of their teacher LLMs. We hypothesize that this effectiveness gap is due to the fact that previous work has not applied the best-suited methods for fine-tuning cross-encoders on manually labeled data (e.g., hard-negative sampling, deep sampling, and listwise loss functions). To close this gap, we create a new dataset, Rank-DistiLLM. Cross-encoders trained on Rank-DistiLLM achieve the effectiveness of LLMs while being up to 173 times faster and 24 times more memory efficient. Our code and data is available at https://github.com/webis-de/ECIR-25.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17710v1">Slide2Text: Leveraging LLMs for Personalized Textbook Generation from PowerPoint Presentations</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      The rapid advancements in Large Language Models (LLMs) have revolutionized educational technology, enabling innovative approaches to automated and personalized content creation. This paper introduces Slide2Text, a system that leverages LLMs to transform PowerPoint presentations into customized textbooks. By extracting slide content using OCR, organizing it into a coherent structure, and generating tailored materials such as explanations, exercises, and references, Slide2Text streamlines the textbook creation process. Flexible customization options further enhance its adaptability to diverse educational needs. The system highlights the potential of LLMs in modernizing textbook creation and improving educational accessibility. Future developments will explore multimedia inputs and advanced user customization features.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.11183v2">Don't Get Lost in the Trees: Streamlining LLM Reasoning by Overcoming Tree Search Exploration Pitfalls</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Recent advancements in tree search algorithms guided by verifiers have significantly enhanced the reasoning capabilities of large language models (LLMs), but at the cost of increased computational resources. In this work, we identify two key challenges contributing to this inefficiency: $\textit{over-exploration}$ due to redundant states with semantically equivalent content, and $\textit{under-exploration}$ caused by high variance in verifier scoring leading to frequent trajectory switching. To address these issues, we propose FETCH, an e$\textbf{f}$fici$\textbf{e}$nt $\textbf{t}$ree sear$\textbf{ch}$ framework, which is a flexible, plug-and-play system compatible with various tree search algorithms. Our framework mitigates over-exploration by merging semantically similar states using agglomerative clustering of text embeddings obtained from a fine-tuned SimCSE model. To tackle under-exploration, we enhance verifiers by incorporating temporal difference learning with adjusted $\lambda$-returns during training to reduce variance, and employing a verifier ensemble to aggregate scores during inference. Experiments on GSM8K, GSM-Plus, and MATH datasets demonstrate that our methods significantly improve reasoning accuracy and computational efficiency across four different tree search algorithms, paving the way for more practical applications of LLM-based reasoning. The code is available at https://github.com/Soistesimmer/Fetch.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17707v1">PipeBoost: Resilient Pipelined Architecture for Fast Serverless LLM Scaling</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      This paper presents PipeBoost, a low-latency LLM serving system for multi-GPU (serverless) clusters, which can rapidly launch inference services in response to bursty requests without preemptively over-provisioning GPUs. Many LLM inference tasks rely on the same base model (e.g., LoRA). To leverage this, PipeBoost introduces fault-tolerant pipeline parallelism across both model loading and inference stages. This approach maximizes aggregate PCIe bandwidth and parallel computation across GPUs, enabling faster generation of the first token. PipeBoost also introduces recovery techniques that enable uninterrupted inference services by utilizing the shared advantages of multiple GPUs. Experimental results show that, compared to state-of-the-art low-latency LLM serving systems, PipeBoost reduces inference latency by 31% to 49.8%. For certain models (e.g., OPT-1.3B), PipeBoost achieves cold-start latencies in the range of a few hundred microseconds.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17684v1">Can LLMs Automate Fact-Checking Article Writing?</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 10 pages, 4 figures, 6 tables
    </div>
    <details class="paper-abstract">
      Automatic fact-checking aims to support professional fact-checkers by offering tools that can help speed up manual fact-checking. Yet, existing frameworks fail to address the key step of producing output suitable for broader dissemination to the general public: while human fact-checkers communicate their findings through fact-checking articles, automated systems typically produce little or no justification for their assessments. Here, we aim to bridge this gap. We argue for the need to extend the typical automatic fact-checking pipeline with automatic generation of full fact-checking articles. We first identify key desiderata for such articles through a series of interviews with experts from leading fact-checking organizations. We then develop QRAFT, an LLM-based agentic framework that mimics the writing workflow of human fact-checkers. Finally, we assess the practical usefulness of QRAFT through human evaluations with professional fact-checkers. Our evaluation shows that while QRAFT outperforms several previously proposed text-generation approaches, it lags considerably behind expert-written articles. We hope that our work will enable further research in this new and important direction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17662v1">Enhancing Persona Consistency for LLMs' Role-Playing using Persona-Aware Contrastive Learning</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 18 pages, 4 figures
    </div>
    <details class="paper-abstract">
      In recent years, large language models (LLMs) have achieved breakthrough progress in many dialogue generation tasks. However, their lack of emotion and fine-grained role awareness limits the model's ability to provide personalized and diverse interactions further. Current methods face high costs in collecting high-quality annotated data for scenarios such as role-playing, and traditional human alignment methods are difficult to deploy due to the inherent diversity of model behavior in role-playing scenarios. Inspired by the alignment of models for safety behaviors through RLHF (Reinforcement Learning from Human Feedback), in this paper, we revisit model role-playing behavior from the perspective of persona alignment and propose a novel annotation-free framework named \textbf{\underline{P}}ersona-Aware \textbf{\underline{C}}ontrastive \textbf{\underline{L}}earning (PCL) to align LLMs' behavior during role-playing, enhancing the model's role consistency. Specifically, we first design a role chain method to encourage the model to self-question based on the role characteristics and dialogue context to adjust personality consistency. Then, we further enhance the model's role-playing strategy through iterative contrastive learning between the use of role characteristics and not. Experiments on both black-box and white-box LLMs show that LLMs equipped with PCL significantly outperform vanilla LLMs under automatic evaluation methods (CharEval \& GPT-4) and human expert evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17645v1">A Modular Dataset to Demonstrate LLM Abstraction Capability</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 7 pages, 5 figures. Submitted to ACL 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) exhibit impressive capabilities but struggle with reasoning errors due to hallucinations and flawed logic. To investigate their internal representations of reasoning, we introduce ArrangementPuzzle, a novel puzzle dataset with structured solutions and automated stepwise correctness verification. We trained a classifier model on LLM activations on this dataset and found that it achieved over 80% accuracy in predicting reasoning correctness, implying that LLMs internally distinguish between correct and incorrect reasoning steps, with the strongest representations in middle-late Transformer layers. Further analysis reveals that LLMs encode abstract reasoning concepts within the middle activation layers of the transformer architecture, distinguishing logical from semantic equivalence. These findings provide insights into LLM reasoning mechanisms and contribute to improving AI reliability and interpretability, thereby offering the possibility to manipulate and refine LLM reasoning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17620v1">A Case Study of Scalable Content Annotation Using Multi-LLM Consensus and Human Review</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 4 pages, GenAICHI 2025 accepted
    </div>
    <details class="paper-abstract">
      Content annotation at scale remains challenging, requiring substantial human expertise and effort. This paper presents a case study in code documentation analysis, where we explore the balance between automation efficiency and annotation accuracy. We present MCHR (Multi-LLM Consensus with Human Review), a novel semi-automated framework that enhances annotation scalability through the systematic integration of multiple LLMs and targeted human review. Our framework introduces a structured consensus-building mechanism among LLMs and an adaptive review protocol that strategically engages human expertise. Through our case study, we demonstrate that MCHR reduces annotation time by 32% to 100% compared to manual annotation while maintaining high accuracy (85.5% to 98%) across different difficulty levels, from basic binary classification to challenging open-set scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17604v1">OmniScience: A Domain-Specialized LLM for Scientific Reasoning and Discovery</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable potential in advancing scientific knowledge and addressing complex challenges. In this work, we introduce OmniScience, a specialized large reasoning model for general science, developed through three key components: (1) domain adaptive pretraining on a carefully curated corpus of scientific literature, (2) instruction tuning on a specialized dataset to guide the model in following domain-specific tasks, and (3) reasoning-based knowledge distillation through fine-tuning to significantly enhance its ability to generate contextually relevant and logically sound responses. We demonstrate the versatility of OmniScience by developing a battery agent that efficiently ranks molecules as potential electrolyte solvents or additives. Comprehensive evaluations reveal that OmniScience is competitive with state-of-the-art large reasoning models on the GPQA Diamond and domain-specific battery benchmarks, while outperforming all public reasoning and non-reasoning models with similar parameter counts. We further demonstrate via ablation experiments that domain adaptive pretraining and reasoning-based knowledge distillation are critical to attain our performance levels, across benchmarks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17587v1">ConSol: Sequential Probability Ratio Testing to Find Consistent LLM Reasoning Paths Efficiently</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) integrating explicit reasoning, such as OpenAI's o3-mini, DeepSeek-R1, and QWQ-32B, enable smaller models to solve complex tasks by generating intermediate reasoning steps prior to providing answers. However, this approach significantly increases computational costs, both monetarily and environmentally. The widely-used self-consistency method further exacerbates these costs by aggregating multiple reasoning paths to improve accuracy, often requiring between 40 to 64 samples per task. Although aggregation effectively reduces variance and bias, additional sampling can lead to diminishing returns when early samples yield consistent results. To address inefficiencies, we propose leveraging Sequential Probability Ratio Testing (SPRT) to dynamically terminate sampling once sufficient consistency is achieved. We calibrate SPRT parameters specifically for LLM applications, accounting for sensitivity to detect the mode of the distribution. Our experiments demonstrate that incorporating SPRT significantly enhances token efficiency, achieving comparable accuracy to self-consistency methods but at a substantially reduced computational cost. To promote transparency and facilitate reproducibility, we have made the source code and datasets used in our experiments publicly available at our GitHub repository: https://github.com/LiuzLab/consol, or available as a PyPI package: pip install consol. We hope that this resource will support further research and encourage the development of new methods building upon our work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18971v1">LLMs as Planning Modelers: A Survey for Leveraging Large Language Models to Construct Automated Planning Models</a></div>
    <div class="paper-meta">
      📅 2025-03-22
      | 💬 20 pages, 3 figures, 3 appendices
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in various natural language tasks but often struggle with long-horizon planning problems requiring structured reasoning. This limitation has drawn interest in integrating neuro-symbolic approaches within the Automated Planning (AP) and Natural Language Processing (NLP) communities. However, identifying optimal AP deployment frameworks can be daunting. This paper aims to provide a timely survey of the current research with an in-depth analysis, positioning LLMs as tools for extracting and refining planning models to support reliable AP planners. By systematically reviewing the current state of research, we highlight methodologies, and identify critical challenges and future directions, hoping to contribute to the joint research on NLP and Automated Planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.20796v1">EXPLICATE: Enhancing Phishing Detection through Explainable AI and LLM-Powered Interpretability</a></div>
    <div class="paper-meta">
      📅 2025-03-22
    </div>
    <details class="paper-abstract">
      Sophisticated phishing attacks have emerged as a major cybersecurity threat, becoming more common and difficult to prevent. Though machine learning techniques have shown promise in detecting phishing attacks, they function mainly as "black boxes" without revealing their decision-making rationale. This lack of transparency erodes the trust of users and diminishes their effective threat response. We present EXPLICATE: a framework that enhances phishing detection through a three-component architecture: an ML-based classifier using domain-specific features, a dual-explanation layer combining LIME and SHAP for complementary feature-level insights, and an LLM enhancement using DeepSeek v3 to translate technical explanations into accessible natural language. Our experiments show that EXPLICATE attains 98.4 % accuracy on all metrics, which is on par with existing deep learning techniques but has better explainability. High-quality explanations are generated by the framework with an accuracy of 94.2 % as well as a consistency of 96.8\% between the LLM output and model prediction. We create EXPLICATE as a fully usable GUI application and a light Chrome extension, showing its applicability in many deployment situations. The research shows that high detection performance can go hand-in-hand with meaningful explainability in security applications. Most important, it addresses the critical divide between automated AI and user trust in phishing detection systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17363v1">Dancing with Critiques: Enhancing LLM Reasoning with Stepwise Natural Language Self-Critique</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Enhancing the reasoning capabilities of large language models (LLMs), particularly for complex tasks requiring multi-step logical deductions, remains a significant challenge. Traditional inference time scaling methods utilize scalar reward signals from process reward models to evaluate candidate reasoning steps, but these scalar rewards lack the nuanced qualitative information essential for understanding and justifying each step. In this paper, we propose a novel inference-time scaling approach -- stepwise natural language self-critique (PANEL), which employs self-generated natural language critiques as feedback to guide the step-level search process. By generating rich, human-readable critiques for each candidate reasoning step, PANEL retains essential qualitative information, facilitating better-informed decision-making during inference. This approach bypasses the need for task-specific verifiers and the associated training overhead, making it broadly applicable across diverse tasks. Experimental results on challenging reasoning benchmarks, including AIME and GPQA, demonstrate that PANEL significantly enhances reasoning performance, outperforming traditional scalar reward-based methods. Our code is available at https://github.com/puddingyeah/PANEL to support and encourage future research in this promising field.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17336v1">Efficient Intent-Based Filtering for Multi-Party Conversations Using Knowledge Distillation from LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have showcased remarkable capabilities in conversational AI, enabling open-domain responses in chat-bots, as well as advanced processing of conversations like summarization, intent classification, and insights generation. However, these models are resource-intensive, demanding substantial memory and computational power. To address this, we propose a cost-effective solution that filters conversational snippets of interest for LLM processing, tailored to the target downstream application, rather than processing every snippet. In this work, we introduce an innovative approach that leverages knowledge distillation from LLMs to develop an intent-based filter for multi-party conversations, optimized for compute power constrained environments. Our method combines different strategies to create a diverse multi-party conversational dataset, that is annotated with the target intents and is then used to fine-tune the MobileBERT model for multi-label intent classification. This model achieves a balance between efficiency and performance, effectively filtering conversation snippets based on their intents. By passing only the relevant snippets to the LLM for further processing, our approach significantly reduces overall operational costs depending on the intents and the data distribution as demonstrated in our experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.11702v2">Toward a method for LLM-enabled Indoor Navigation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 7 pages, 3 figures, 5 tables
    </div>
    <details class="paper-abstract">
      Indoor navigation presents unique challenges due to complex layouts, lack of GPS signals, and accessibility concerns. Existing solutions often struggle with real-time adaptability and user-specific needs. In this work, we explore the potential of a Large Language Model (LLM), i.e., ChatGPT, to generate natural, context-aware navigation instructions from indoor map images. We design and evaluate test cases across different real-world environments, analyzing the effectiveness of LLMs in interpreting spatial layouts, handling user constraints, and planning efficient routes. Our findings demonstrate the potential of LLMs for supporting personalized indoor navigation, with an average of 52% correct indications and a maximum of 62%. The results do not appear to depend on the complexity of the layout or the complexity of the expected path, but rather on the number of points of interest and the abundance of visual information, which negatively affect the performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.06796v2">Write Your Own CodeChecker: An Automated Test-Driven Checker Development Approach with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      With the rising demand for code quality assurance, developers are not only utilizing existing static code checkers but also seeking custom checkers to satisfy their specific needs. Nowadays, various code-checking frameworks provide extensive checker customization interfaces to meet this need. However, both the abstract checking logic and the complex API usage of large-scale checker frameworks make this task challenging. To this end, automated code checker generation is anticipated to ease the burden of checker development. In this paper, we propose AutoChecker, an innovative LLM-powered approach that can write code checkers automatically based on only a rule description and a test suite. To achieve comprehensive checking logic, AutoChecker incrementally updates the checker's logic by focusing on solving one selected case each time. To obtain precise API knowledge, during each iteration, it leverages fine-grained logic-guided API-context retrieval, where it first decomposes the checking logic into a series of sub-operations and then retrieves checker-related API-contexts for each sub-operation. For evaluation, we apply AutoChecker, five baselines, and three ablation methods using multiple LLMs to generate checkers for 20 randomly selected PMD rules. Experimental results show that AutoChecker significantly outperforms others across all effectiveness metrics, with an average test pass rate of 82.28%. Additionally, the checkers generated by AutoChecker can be successfully applied to real-world projects, matching the performance of official checkers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17229v1">FactSelfCheck: Fact-Level Black-Box Hallucination Detection for LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 Preprint
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) frequently generate hallucinated content, posing significant challenges for applications where factuality is crucial. While existing hallucination detection methods typically operate at the sentence level or passage level, we propose FactSelfCheck, a novel black-box sampling-based method that enables fine-grained fact-level detection. Our approach represents text as knowledge graphs consisting of facts in the form of triples. Through analyzing factual consistency across multiple LLM responses, we compute fine-grained hallucination scores without requiring external resources or training data. Our evaluation demonstrates that FactSelfCheck performs competitively with leading sampling-based methods while providing more detailed insights. Most notably, our fact-level approach significantly improves hallucination correction, achieving a 35% increase in factual content compared to the baseline, while sentence-level SelfCheckGPT yields only an 8% improvement. The granular nature of our detection enables more precise identification and correction of hallucinated content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.11006v2">GREEN-CODE: Learning to Optimize Energy Efficiency in LLM-based Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 Under submission in ACM/IEEE conference, 11 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are becoming integral to daily life, showcasing their vast potential across various Natural Language Processing (NLP) tasks. Beyond NLP, LLMs are increasingly used in software development tasks, such as code completion, modification, bug fixing, and code translation. Software engineers widely use tools like GitHub Copilot and Amazon Q, streamlining workflows and automating tasks with high accuracy. While the resource and energy intensity of LLM training is often highlighted, inference can be even more resource-intensive over time, as it's a continuous process with a high number of invocations. Therefore, developing resource-efficient alternatives for LLM inference is crucial for sustainability. This work proposes GREEN-CODE, a framework for energy-aware code generation in LLMs. GREEN-CODE performs dynamic early exit during LLM inference. We train a Reinforcement Learning (RL) agent that learns to balance the trade-offs between accuracy, latency, and energy consumption. Our approach is evaluated on two open-source LLMs, Llama 3.2 3B and OPT 2.7B, using the JavaCorpus and PY150 datasets. Results show that our method reduces the energy consumption between 23-50 % on average for code generation tasks without significantly affecting accuracy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15249v2">LitLLMs, LLMs for Literature Review: Are we there yet?</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Literature reviews are an essential component of scientific research, but they remain time-intensive and challenging to write, especially due to the recent influx of research papers. This paper explores the zero-shot abilities of recent Large Language Models (LLMs) in assisting with the writing of literature reviews based on an abstract. We decompose the task into two components: 1. Retrieving related works given a query abstract, and 2. Writing a literature review based on the retrieved results. We analyze how effective LLMs are for both components. For retrieval, we introduce a novel two-step search strategy that first uses an LLM to extract meaningful keywords from the abstract of a paper and then retrieves potentially relevant papers by querying an external knowledge base. Additionally, we study a prompting-based re-ranking mechanism with attribution and show that re-ranking doubles the normalized recall compared to naive search methods, while providing insights into the LLM's decision-making process. In the generation phase, we propose a two-step approach that first outlines a plan for the review and then executes steps in the plan to generate the actual review. To evaluate different LLM-based literature review methods, we create test sets from arXiv papers using a protocol designed for rolling use with newly released LLMs to avoid test set contamination in zero-shot evaluations. We release this evaluation protocol to promote additional research and development in this regard. Our empirical results suggest that LLMs show promising potential for writing literature reviews when the task is decomposed into smaller components of retrieval and planning. Our project page including a demonstration system and toolkit can be accessed here: https://litllm.github.io.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17181v1">LLMs Love Python: A Study of LLMs' Bias for Programming Languages and Libraries</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 12 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Programming language and library choices are crucial to software reliability and security. Poor or inconsistent choices can lead to increased technical debt, security vulnerabilities, and even catastrophic failures in safety-critical systems. As Large Language Models (LLMs) play an increasing role in code generation, it is essential to understand how they make these decisions. However, little is known about their preferences when selecting programming languages and libraries for different coding tasks. To fill this gap, this study provides the first in-depth investigation into LLM preferences for programming languages and libraries used when generating code. We assess the preferences of eight diverse LLMs by prompting them to complete various coding tasks, including widely-studied benchmarks and the more practical task of generating the initial structural code for new projects (a crucial step that often determines a project's language or library choices). Our findings reveal that LLMs heavily favour Python when solving language-agnostic problems, using it in 90%-97% of cases for benchmark tasks. Even when generating initial project code where Python is not a suitable language, it remains the most-used language in 58% of instances. Moreover, LLMs contradict their own language recommendations in 83% of project initialisation tasks, raising concerns about their reliability in guiding language selection. Similar biases toward well-established libraries further create serious discoverability challenges for newer open-source projects. These results highlight the need to improve LLMs' adaptability to diverse programming contexts and to develop mechanisms for mitigating programming language and library bias.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.15289v2">SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.00363v2">SPDZCoder: Combining Expert Knowledge with LLMs for Generating Privacy-Computing Code</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Privacy computing receives increasing attention but writing privacy computing code remains challenging for developers due to limited library functions, necessitating function implementation from scratch, and data-oblivious requirement, contradicting intuitive thinking and usual practices of programmers. Automating the generation of privacy computing code with Large Language Models can streamline development effort and lower the barrier to using privacy computing frameworks. However, existing LLMs still encounter challenges in code translation for privacy-preserving computation, such as translating Python to MP-SPDZ, due to the scarcity of MP-SPDZ data required for effective pre-training or fine-tuning. Moreover, the lack of a benchmark further complicates the evaluation of translation quality. To address the limitations, this work proposes SPDZCoder, a rule-based framework that combines LLMs with expert knowledge for generating privacy-computing code without requiring additional training data. Specifically, SPDZCoder employ a rigorous procedure for collecting high-quality expert knowledge to represent the semantic-expressing differences between Python and MP-SPDZ, and to derive transformation rules for translating Python to MP-SPDZ based on these knowledge. Then, SPDZCoder progressively converts Python code into MP-SPDZ code using transformation rules in a three stage pipeline. To evaluate SPDZCoder, we manually constructed a benchmark dataset, SPDZEval, which comprises six data splits, each representing a distinct class of challenging tasks in MP-SPDZ implementation. Extensive experiments show that SPDZCoder achieves superior performance, significantly surpassing baselines in pass@1 and pass@2. Specifically, SPDZCoder attains an overall correctness of 85.94% and 92.01% in pass@1 and pass@2, respectively, whereas the best-performing baseline achieves 63.58% and 76.36%, respectively.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17073v1">A Study into Investigating Temporal Robustness of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 8 pages
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) encapsulate a surprising amount of factual world knowledge. However, their performance on temporal questions and historical knowledge is limited because they often cannot understand temporal scope and orientation or neglect the temporal aspect altogether. In this study, we aim to measure precisely how robust LLMs are for question answering based on their ability to process temporal information and perform tasks requiring temporal reasoning and temporal factual knowledge. Specifically, we design eight time-sensitive robustness tests for factual information to check the sensitivity of six popular LLMs in the zero-shot setting. Overall, we find LLMs lacking temporal robustness, especially to temporal reformulations and the use of different granularities of temporal references. We show how a selection of these eight tests can be used automatically to judge a model's temporal robustness for user questions on the fly. Finally, we apply the findings of this study to improve the temporal QA performance by up to 55 percent.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.12547v2">LLMSeR: Enhancing Sequential Recommendation via LLM-based Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Sequential Recommender Systems (SRS) have become a cornerstone of online platforms, leveraging users' historical interaction data to forecast their next potential engagement. Despite their widespread adoption, SRS often grapple with the long-tail user dilemma, resulting in less effective recommendations for individuals with limited interaction records. The advent of Large Language Models (LLMs), with their profound capability to discern semantic relationships among items, has opened new avenues for enhancing SRS through data augmentation. Nonetheless, current methodologies encounter obstacles, including the absence of collaborative signals and the prevalence of hallucination phenomena. In this work, we present LLMSeR, an innovative framework that utilizes Large Language Models (LLMs) to generate pseudo-prior items, thereby improving the efficacy of Sequential Recommender Systems (SRS). To alleviate the challenge of insufficient collaborative signals, we introduce the Semantic Interaction Augmentor (SIA), a method that integrates both semantic and collaborative information to comprehensively augment user interaction data. Moreover, to weaken the adverse effects of hallucination in SRS, we develop the Adaptive Reliability Validation (ARV), a validation technique designed to assess the reliability of the generated pseudo items. Complementing these advancements, we also devise a Dual-Channel Training strategy, ensuring seamless integration of data augmentation into the SRS training process.Extensive experiments conducted with three widely-used SRS models demonstrate the generalizability and efficacy of LLMSeR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17039v1">Summarization Metrics for Spanish and Basque: Do Automatic Scores and LLM-Judges Correlate with Humans?</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Studies on evaluation metrics and LLM-as-a-Judge models for automatic text summarization have largely been focused on English, limiting our understanding of their effectiveness in other languages. Through our new dataset BASSE (BAsque and Spanish Summarization Evaluation), we address this situation by collecting human judgments on 2,040 abstractive summaries in Basque and Spanish, generated either manually or by five LLMs with four different prompts. For each summary, annotators evaluated five criteria on a 5-point Likert scale: coherence, consistency, fluency, relevance, and 5W1H. We use these data to reevaluate traditional automatic metrics used for evaluating summaries, as well as several LLM-as-a-Judge models that show strong performance on this task in English. Our results show that currently proprietary judge LLMs have the highest correlation with human judgments, followed by criteria-specific automatic metrics, while open-sourced judge LLMs perform poorly. We release BASSE and our code publicly, along with the first large-scale Basque summarization dataset containing 22,525 news articles with their subheads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17004v1">Text2Model: Generating dynamic chemical reactor models using large language models (LLMs)</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      As large language models have shown remarkable capabilities in conversing via natural language, the question arises as to how LLMs could potentially assist chemical engineers in research and industry with domain-specific tasks. We generate dynamic chemical reactor models in Modelica code format from textual descriptions as user input. We fine-tune Llama 3.1 8B Instruct on synthetically generated Modelica code for different reactor scenarios. We compare the performance of our fine-tuned model to the baseline Llama 3.1 8B Instruct model and GPT4o. We manually assess the models' predictions regarding the syntactic and semantic accuracy of the generated dynamic models. We find that considerable improvements are achieved by the fine-tuned model with respect to both the semantic and the syntactic accuracy of the Modelica models. However, the fine-tuned model lacks a satisfactory ability to generalize to unseen scenarios compared to GPT4o.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.17397v2">Building Multilingual Datasets for Predicting Mental Health Severity through LLMs: Prospects and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly being integrated into various medical fields, including mental health support systems. However, there is a gap in research regarding the effectiveness of LLMs in non-English mental health support applications. To address this problem, we present a novel multilingual adaptation of widely-used mental health datasets, translated from English into six languages (e.g., Greek, Turkish, French, Portuguese, German, and Finnish). This dataset enables a comprehensive evaluation of LLM performance in detecting mental health conditions and assessing their severity across multiple languages. By experimenting with GPT and Llama, we observe considerable variability in performance across languages, despite being evaluated on the same translated dataset. This inconsistency underscores the complexities inherent in multilingual mental health support, where language-specific nuances and mental health data coverage can affect the accuracy of the models. Through comprehensive error analysis, we emphasize the risks of relying exclusively on LLMs in medical settings (e.g., their potential to contribute to misdiagnoses). Moreover, our proposed approach offers significant cost savings for multilingual tasks, presenting a major advantage for broad-scale implementation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.07871v2">Objection Overruled! Lay People can Distinguish Large Language Models from Lawyers, but still Favour Advice from an LLM</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 ACMConference on Human Factors in Computing Systems (CHI'25)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are seemingly infiltrating every domain, and the legal context is no exception. In this paper, we present the results of three experiments (total N = 288) that investigated lay people's willingness to act upon, and their ability to discriminate between, LLM- and lawyer-generated legal advice. In Experiment 1, participants judged their willingness to act on legal advice when the source of the advice was either known or unknown. When the advice source was unknown, participants indicated that they were significantly more willing to act on the LLM-generated advice. The result of the source unknown condition was replicated in Experiment 2. Intriguingly, despite participants indicating higher willingness to act on LLM-generated advice in Experiments 1 and 2, participants discriminated between the LLM- and lawyer-generated texts significantly above chance-level in Experiment 3. Lastly, we discuss potential explanations and risks of our findings, limitations and future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12923v2">On-Device LLMs for Home Assistant: Dual Role in Intent Detection and Response Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 WNUT 2025
    </div>
    <details class="paper-abstract">
      This paper investigates whether Large Language Models (LLMs), fine-tuned on synthetic but domain-representative data, can perform the twofold task of (i) slot and intent detection and (ii) natural language response generation for a smart home assistant, while running solely on resource-limited, CPU-only edge hardware. We fine-tune LLMs to produce both JSON action calls and text responses. Our experiments show that 16-bit and 8-bit quantized variants preserve high accuracy on slot and intent detection and maintain strong semantic coherence in generated text, while the 4-bit model, while retaining generative fluency, suffers a noticeable drop in device-service classification accuracy. Further evaluations on noisy human (non-synthetic) prompts and out-of-domain intents confirm the models' generalization ability, obtaining around 80--86\% accuracy. While the average inference time is 5--6 seconds per query -- acceptable for one-shot commands but suboptimal for multi-turn dialogue -- our results affirm that an on-device LLM can effectively unify command interpretation and flexible response generation for home automation without relying on specialized hardware.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16929v1">TEMPO: Temporal Preference Optimization of Video LLMs via Difficulty Scheduling and Pre-SFT Alignment</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Video Large Language Models (Video LLMs) have achieved significant success by leveraging a two-stage paradigm: pretraining on large-scale video-text data for vision-language alignment, followed by supervised fine-tuning (SFT) for task-specific capabilities. However, existing approaches struggle with temporal reasoning due to weak temporal correspondence in the data and reliance on the next-token prediction paradigm during training. To address these limitations, we propose TEMPO (TEMporal Preference Optimization), a systematic framework that enhances Video LLMs' temporal reasoning capabilities through Direct Preference Optimization (DPO). To facilitate this, we introduce an automated preference data generation pipeline that systematically constructs preference pairs by selecting videos that are rich in temporal information, designing video-specific perturbation strategies, and finally evaluating model responses on clean and perturbed video inputs. Our temporal alignment features two key innovations: curriculum learning which that progressively increases perturbation difficulty to improve model robustness and adaptability; and ``Pre-SFT Alignment'', applying preference optimization before instruction tuning to prioritize fine-grained temporal comprehension. Extensive experiments demonstrate that our approach consistently improves Video LLM performance across multiple benchmarks with a relatively small set of self-generated DPO data. We further analyze the transferability of DPO data across architectures and the role of difficulty scheduling in optimization. Our findings highlight our TEMPO as a scalable and efficient complement to SFT-based methods, paving the way for developing reliable Video LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.12785v2">Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Large Vision-Language Models (LVLMs) typically learn visual capacity through visual instruction tuning, involving updates to both a projector and their LLM backbones. Inspired by the concept of a visual region in the human brain, we investigate the existence of an analogous \textit{visual region} within LLMs that functions as a cognitive core, and explore the potential of efficient training of LVLMs via selective layers tuning. Using Bunny-Llama-3-8B-V for detailed analysis and other three LVLMs for validation across diverse visual and textual tasks, we find that selectively updating 25\% of LLMs layers, when sparsely and uniformly distributed, can preserve nearly 99\% of visual performance and maintain or improve textual task results, while effectively reducing training time. Based on this targeted training approach, we further propose a novel visual region-based pruning paradigm, removing non-critical layers outside the visual region, which can achieve minimal performance loss. This study offers an effective and efficient strategy for LVLM training and inference by activating a layer-wise visual region within LLMs, which proves consistently effective across different models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13879v2">Bridging Social Psychology and LLM Reasoning: Conflict-Aware Meta-Review Generation via Cognitive Alignment</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 23 pages
    </div>
    <details class="paper-abstract">
      The rapid growth of scholarly submissions has overwhelmed traditional peer review systems, driving the need for intelligent automation to preserve scientific rigor. While large language models (LLMs) show promise in automating manuscript critiques, their ability to synthesize high-stakes meta-reviews, which require conflict-aware reasoning and consensus derivation, remains underdeveloped. Existing methods fail to effectively handle conflicting viewpoints within differing opinions, and often introduce additional cognitive biases, such as anchoring effects and conformity bias.To overcome these limitations, we propose the Cognitive Alignment Framework (CAF), a dual-process architecture that transforms LLMs into adaptive scientific arbitrators. By operationalizing Kahneman's dual-process theory, CAF introduces a three-step cognitive pipeline: review initialization, incremental integration, and cognitive alignment.Empirical validation shows that CAF outperforms existing LLM-based methods, with sentiment consistency gains reaching up to 19.47\% and content consistency improving by as much as 12.95\%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16922v1">RustEvo^2: An Evolving Benchmark for API Evolution in LLM-based Rust Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have become pivotal tools for automating code generation in software development. However, these models face significant challenges in producing version-aware code for rapidly evolving languages like Rust, where frequent Application Programming Interfaces (API) changes across versions lead to compatibility issues and correctness errors. Existing benchmarks lack systematic evaluation of how models navigate API transitions, relying on labor-intensive manual curation and offering limited version-specific insights. To address this gap, we present RustEvo, a novel framework for constructing dynamic benchmarks that evaluate the ability of LLMs to adapt to evolving Rust APIs. RustEvo automates dataset creation by synthesizing 588 API changes (380 from Rust standard libraries, 208 from 15 third-party crates) into programming tasks mirroring real-world challenges. These tasks cover four API evolution categories: Stabilizations, Signature Changes, Behavioral Changes, and Deprecations, reflecting their actual distribution in the Rust ecosystem. Experiments on state-of-the-art (SOTA) LLMs reveal significant performance variations: models achieve a 65.8% average success rate on stabilized APIs but only 38.0% on behavioral changes, highlighting difficulties in detecting semantic shifts without signature alterations. Knowledge cutoff dates strongly influence performance, with models scoring 56.1% on before-cutoff APIs versus 32.5% on after-cutoff tasks. Retrieval-Augmented Generation (RAG) mitigates this gap, improving success rates by 13.5% on average for APIs released after model training. Our findings underscore the necessity of our evolution-aware benchmarks to advance the adaptability of LLMs in fast-paced software ecosystems. The framework and the benchmarks are publicly released at https://github.com/SYSUSELab/RustEvo.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16893v1">Improving the End-to-End Efficiency of Offline Inference for Multi-LLM Applications Based on Sampling and Simulation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) have shown great success in many tasks, they are used in various applications. While a lot of works have focused on the efficiency of single-LLM application (e.g., offloading, request scheduling, parallelism strategy selection), multi-LLM applications receive less attention, particularly in offline inference scenarios. In this work, we aim to improve the offline end-to-end inference efficiency of multi-LLM applications in the single-node multi-GPU environment. The problem involves two key decisions: (1) determining which LLMs to run concurrently each time (we may not run all the models at the same time), and (2) selecting a parallelism strategy to use for each LLM. This problem is NP-hard. Naive solutions may not work well because the running time for a model to complete a set of requests depends on the request workload and the selected parallelism strategy, and they lack an accurate model of the running time. As the LLM output lengths are unknown before running, to estimate the model running time, we propose a sampling-then-simulation method which first estimates the output lengths by sampling from an empirical cumulative function we obtained from a large dataset in advance, and then simulates the LLM inference process accordingly. Based on the simulation, we estimate the per-iteration latencys to get the total latency. A greedy method is proposed to optimize the scheduling of the LLMs in the application across the GPUs. We then propose a framework SamuLLM which contains two phases: planning, which calls the greedy method for an application and running, which runs the application and dynamically adjust the model scheduling based on the runtime information. Experiments on 3 applications and a mixed application show that SamuLLM can achieve 1.0-2.4$\times$ end-to-end speedups compared to the competitors.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.16454v3">Catastrophic Failure of LLM Unlearning via Quantization</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 ICLR 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable proficiency in generating text, benefiting from extensive training on vast textual corpora. However, LLMs may also acquire unwanted behaviors from the diverse and sensitive nature of their training data, which can include copyrighted and private content. Machine unlearning has been introduced as a viable solution to remove the influence of such problematic content without the need for costly and time-consuming retraining. This process aims to erase specific knowledge from LLMs while preserving as much model utility as possible. Despite the effectiveness of current unlearning methods, little attention has been given to whether existing unlearning methods for LLMs truly achieve forgetting or merely hide the knowledge, which current unlearning benchmarks fail to detect. This paper reveals that applying quantization to models that have undergone unlearning can restore the "forgotten" information. To thoroughly evaluate this phenomenon, we conduct comprehensive experiments using various quantization techniques across multiple precision levels. We find that for unlearning methods with utility constraints, the unlearned model retains an average of 21\% of the intended forgotten knowledge in full precision, which significantly increases to 83\% after 4-bit quantization. ... Our code is available at: \href{https://github.com/zzwjames/FailureLLMUnlearning}{https://github.com/zzwjames/FailureLLMUnlearning}.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2309.14345v4">Bias Testing and Mitigation in LLM-based Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 Accepted by TOSEM
    </div>
    <details class="paper-abstract">
      As the adoption of LLMs becomes more widespread in software coding ecosystems, a pressing issue has emerged: does the generated code contain social bias and unfairness, such as those related to age, gender, and race? This issue concerns the integrity, fairness, and ethical foundation of software applications that depend on the code generated by these models but are underexplored in the literature. This paper presents a novel bias testing framework that is specifically designed for code generation tasks. Based on this framework, we conduct an extensive empirical study on the biases in code generated by five widely studied LLMs (i.e., PALM-2-CodeChat-bison, Claude-instant-1, GPT-3.5-turbo, GPT-4-turbo, and GPT-4). Our findings reveal that biases are prevalent. For example, 13.47% to 49.10% of the codes generated by these LLMs have biased behaviors towards gender. Moreover, we study five bias mitigation prompt strategies that are commonly used in current code generation scenarios, i.e., zero-shot, one-shot, few-shot, and two Chain-of-Thought (CoT) prompts, with and without provided feedback-driven refinement. Our evaluation results illustrate that using direct prompt engineering strategies has limited effectiveness in mitigating bias, but our test execution feedback can help to reduce the ratio of code biases to a large extent (e.g., from 59.88% to 4.79% for GPT-4).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16851v1">Towards LLM Guardrails via Sparse Representation Steering</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have demonstrated remarkable performance in natural language generation tasks, yet their uncontrolled outputs pose significant ethical and safety risks. Recently, representation engineering methods have shown promising results in steering model behavior by modifying the rich semantic information encoded in activation vectors. However, due to the difficulty of precisely disentangling semantic directions within high-dimensional representation space, existing approaches suffer from three major limitations: lack of fine-grained control, quality degradation of generated content, and poor interpretability. To address these challenges, we propose a sparse encoding-based representation engineering method, named SRE, which decomposes polysemantic activations into a structured, monosemantic feature space. By leveraging sparse autoencoding, our approach isolates and adjusts only task-specific sparse feature dimensions, enabling precise and interpretable steering of model behavior while preserving content quality. We validate our method on three critical domains, i.e., safety, fairness, and truthfulness using the open-source LLM Gemma-2-2B-it. Experimental results show that SRE achieves superior controllability while maintaining the overall quality of generated content (i.e., controllability and quality), demonstrating its effectiveness as a fine-grained and interpretable activation steering framework.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.00212v4">STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 25 pages, 5 figures
    </div>
    <details class="paper-abstract">
      A fundamental challenge in formal theorem proving by LLMs is the lack of high-quality training data. Although reinforcement learning or expert iteration partially mitigates this issue by alternating between LLM generating proofs and finetuning them on correctly generated ones, performance quickly plateaus due to the scarcity of correct proofs (sparse rewards). To keep improving the models with limited data, we draw inspiration from mathematicians, who continuously develop new results, partly by proposing novel conjectures or exercises (which are often variants of known results) and attempting to solve them. We design the Self-play Theorem Prover (STP) that simultaneously takes on two roles, conjecturer and prover, each providing training signals to the other. The conjecturer is trained iteratively on previously generated conjectures that are barely provable by the current prover, which incentivizes it to generate increasingly challenging conjectures over time. The prover attempts to prove the conjectures with standard expert iteration. We evaluate STP with both Lean and Isabelle formal versifiers. With 51.3 billion tokens generated during the training in Lean, STP proves 28.5% of the statements in the LeanWorkbook dataset, doubling the previous best result of 13.2% achieved through expert iteration. The final model achieves state-of-the-art performance among whole-proof generation methods on miniF2F-test (65.0%, pass@3200), Proofnet-test (23.9%, pass@3200) and PutnamBench (8/644, pass@3200). We release our code, model, and dataset in this URL: https://github.com/kfdong/STP.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16789v1">Conversational User-AI Intervention: A Study on Prompt Rewriting for Improved LLM Response Generation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 8 pages, ACL style
    </div>
    <details class="paper-abstract">
      Human-LLM conversations are increasingly becoming more pervasive in peoples' professional and personal lives, yet many users still struggle to elicit helpful responses from LLM Chatbots. One of the reasons for this issue is users' lack of understanding in crafting effective prompts that accurately convey their information needs. Meanwhile, the existence of real-world conversational datasets on the one hand, and the text understanding faculties of LLMs on the other, present a unique opportunity to study this problem, and its potential solutions at scale. Thus, in this paper we present the first LLM-centric study of real human-AI chatbot conversations, focused on investigating aspects in which user queries fall short of expressing information needs, and the potential of using LLMs to rewrite suboptimal user prompts. Our findings demonstrate that rephrasing ineffective prompts can elicit better responses from a conversational system, while preserving the user's original intent. Notably, the performance of rewrites improves in longer conversations, where contextual inferences about user needs can be made more accurately. Additionally, we observe that LLMs often need to -- and inherently do -- make \emph{plausible} assumptions about a user's intentions and goals when interpreting prompts. Our findings largely hold true across conversational domains, user intents, and LLMs of varying sizes and families, indicating the promise of using prompt rewriting as a solution for better human-AI interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.08823v2">ResBench: Benchmarking LLM-Generated FPGA Designs with Resource Awareness</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 to be published in International Symposium on Highly Efficient Accelerators and Reconfigurable Technologies 2025
    </div>
    <details class="paper-abstract">
      Field-Programmable Gate Arrays (FPGAs) are widely used in modern hardware design, yet writing Hardware Description Language (HDL) code for FPGA implementation remains a complex and time-consuming task. Large Language Models (LLMs) have emerged as a promising tool for HDL generation, but existing benchmarks for LLM-based code generation primarily focus on functional correctness while overlooking hardware resource usage. Furthermore, current benchmarks offer limited diversity and do not fully represent the wide range of real-world FPGA applications. To address these shortcomings, we introduce ResBench, the first resource-focused benchmark explicitly designed to distinguish between resource-optimized and inefficient LLM-generated HDL code. ResBench consists of 56 problems across 12 categories, covering applications from finite state machines to financial computing. Our open-source evaluation framework automatically tests LLMs by generating Verilog code, verifying correctness, and measuring resource usage. The experiments, which primarily analyze Lookup Table (LUT) usage, reveal significant differences among LLMs, demonstrating ResBench's capability to identify models that generate more resource-optimized FPGA designs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17579v1">Leveraging Human Production-Interpretation Asymmetries to Test LLM Cognitive Plausibility</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Whether large language models (LLMs) process language similarly to humans has been the subject of much theoretical and practical debate. We examine this question through the lens of the production-interpretation distinction found in human sentence processing and evaluate the extent to which instruction-tuned LLMs replicate this distinction. Using an empirically documented asymmetry between production and interpretation in humans for implicit causality verbs as a testbed, we find that some LLMs do quantitatively and qualitatively reflect human-like asymmetries between production and interpretation. We demonstrate that whether this behavior holds depends upon both model size - with larger models more likely to reflect human-like patterns and the choice of meta-linguistic prompts used to elicit the behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17569v1">Fairness-Driven LLM-based Causal Discovery with Active Learning and Dynamic Scoring</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Causal discovery (CD) plays a pivotal role in numerous scientific fields by clarifying the causal relationships that underlie phenomena observed in diverse disciplines. Despite significant advancements in CD algorithms that enhance bias and fairness analyses in machine learning, their application faces challenges due to the high computational demands and complexities of large-scale data. This paper introduces a framework that leverages Large Language Models (LLMs) for CD, utilizing a metadata-based approach akin to the reasoning processes of human experts. By shifting from pairwise queries to a more scalable breadth-first search (BFS) strategy, the number of required queries is reduced from quadratic to linear in terms of variable count, thereby addressing scalability concerns inherent in previous approaches. This method utilizes an Active Learning (AL) and a Dynamic Scoring Mechanism that prioritizes queries based on their potential information gain, combining mutual information, partial correlation, and LLM confidence scores to refine the causal graph more efficiently and accurately. This BFS query strategy reduces the required number of queries significantly, thereby addressing scalability concerns inherent in previous approaches. This study provides a more scalable and efficient solution for leveraging LLMs in fairness-driven CD, highlighting the effects of the different parameters on performance. We perform fairness analyses on the inferred causal graphs, identifying direct and indirect effects of sensitive attributes on outcomes. A comparison of these analyses against those from graphs produced by baseline methods highlights the importance of accurate causal graph construction in understanding bias and ensuring fairness in machine learning systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.09778v2">Prompt and circumstance: A word-by-word LLM prompting approach to interlinear glossing for low-resource languages</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Partly automated creation of interlinear glossed text (IGT) has the potential to assist in linguistic documentation. We argue that LLMs can make this process more accessible to linguists because of their capacity to follow natural-language instructions. We investigate the effectiveness of a retrieval-based LLM prompting approach to glossing, applied to the seven languages from the SIGMORPHON 2023 shared task. Our system beats the BERT-based shared task baseline for every language in the morpheme-level score category, and we show that a simple 3-best oracle has higher word-level scores than the challenge winner (a tuned sequence model) in five languages. In a case study on Tsez, we ask the LLM to automatically create and follow linguistic instructions, reducing errors on a confusing grammatical feature. Our results thus demonstrate the potential contributions which LLMs can make in interactive systems for glossing, both in making suggestions to human annotators and following directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17553v1">Autonomous Radiotherapy Treatment Planning Using DOLA: A Privacy-Preserving, LLM-Based Optimization Agent</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 19 pages, 5 figures, preprint
    </div>
    <details class="paper-abstract">
      Radiotherapy treatment planning is a complex and time-intensive process, often impacted by inter-planner variability and subjective decision-making. To address these challenges, we introduce Dose Optimization Language Agent (DOLA), an autonomous large language model (LLM)-based agent designed for optimizing radiotherapy treatment plans while rigorously protecting patient privacy. DOLA integrates the LLaMa3.1 LLM directly with a commercial treatment planning system, utilizing chain-of-thought prompting, retrieval-augmented generation (RAG), and reinforcement learning (RL). Operating entirely within secure local infrastructure, this agent eliminates external data sharing. We evaluated DOLA using a retrospective cohort of 18 prostate cancer patients prescribed 60 Gy in 20 fractions, comparing model sizes (8 billion vs. 70 billion parameters) and optimization strategies (No-RAG, RAG, and RAG+RL) over 10 planning iterations. The 70B model demonstrated significantly improved performance, achieving approximately 16.4% higher final scores than the 8B model. The RAG approach outperformed the No-RAG baseline by 19.8%, and incorporating RL accelerated convergence, highlighting the synergy of retrieval-based memory and reinforcement learning. Optimal temperature hyperparameter analysis identified 0.4 as providing the best balance between exploration and exploitation. This proof of concept study represents the first successful deployment of locally hosted LLM agents for autonomous optimization of treatment plans within a commercial radiotherapy planning system. By extending human-machine interaction through interpretable natural language reasoning, DOLA offers a scalable and privacy-conscious framework, with significant potential for clinical implementation and workflow improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17823v2">A General Framework to Enhance Fine-tuning-based LLM Unlearning</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Unlearning has been proposed to remove copyrighted and privacy-sensitive data from Large Language Models (LLMs). Existing approaches primarily rely on fine-tuning-based methods, which can be categorized into gradient ascent-based (GA-based) and suppression-based methods. However, they often degrade model utility (the ability to respond to normal prompts). In this work, we aim to develop a general framework that enhances the utility of fine-tuning-based unlearning methods. To achieve this goal, we first investigate the common property between GA-based and suppression-based methods. We unveil that GA-based methods unlearn by distinguishing the target data (i.e., the data to be removed) and suppressing related generations, which is essentially the same strategy employed by suppression-based methods. Inspired by this finding, we introduce Gated Representation UNlearning (GRUN) which has two components: a soft gate function for distinguishing target data and a suppression module using Representation Fine-tuning (ReFT) to adjust representations rather than model parameters. Experiments show that GRUN significantly improves the unlearning and utility. Meanwhile, it is general for fine-tuning-based methods, efficient and promising for sequential unlearning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17502v1">Large Language Models (LLMs) for Source Code Analysis: applications, models and datasets</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) and transformer-based architectures are increasingly utilized for source code analysis. As software systems grow in complexity, integrating LLMs into code analysis workflows becomes essential for enhancing efficiency, accuracy, and automation. This paper explores the role of LLMs for different code analysis tasks, focusing on three key aspects: 1) what they can analyze and their applications, 2) what models are used and 3) what datasets are used, and the challenges they face. Regarding the goal of this research, we investigate scholarly articles that explore the use of LLMs for source code analysis to uncover research developments, current trends, and the intellectual structure of this emerging field. Additionally, we summarize limitations and highlight essential tools, datasets, and key challenges, which could be valuable for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17500v1">Variance Control via Weight Rescaling in LLM Pre-training</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      The outcome of Large Language Model (LLM) pre-training strongly depends on weight initialization and variance control strategies. Although the importance of initial variance control has been well documented in neural networks in general, the literature on initialization and management of its growth during LLM pre-training, specifically, is somewhat sparse. In this paper, we introduce the Layer Index Rescaling (LIR) weight initialization scheme, and the Target Variance Rescaling (TVR) variance control strategy. Experiments on a 1B parameter LLaMA model demonstrate that better variance management using these techniques yields substantial improvements in downstream task performance (up to 4.6% on common pre-training benchmarks) and reduces extreme activation values, thus mitigating challenges associated with quantization and low-precision training. Our code is available at: https://github.com/bluorion-com/weight_rescaling.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17479v1">Your voice is your voice: Supporting Self-expression through Speech Generation and LLMs in Augmented and Alternative Communication</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      In this paper, we present Speak Ease: an augmentative and alternative communication (AAC) system to support users' expressivity by integrating multimodal input, including text, voice, and contextual cues (conversational partner and emotional tone), with large language models (LLMs). Speak Ease combines automatic speech recognition (ASR), context-aware LLM-based outputs, and personalized text-to-speech technologies to enable more personalized, natural-sounding, and expressive communication. Through an exploratory feasibility study and focus group evaluation with speech and language pathologists (SLPs), we assessed Speak Ease's potential to enable expressivity in AAC. The findings highlight the priorities and needs of AAC users and the system's ability to enhance user expressivity by supporting more personalized and contextually relevant communication. This work provides insights into the use of multimodal inputs and LLM-driven features to improve AAC systems and support expressivity.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17439v1">LEMMA: Learning from Errors for MatheMatical Advancement in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 9 pages, 6 figures, 4 tables, under review
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have demonstrated remarkable reasoning capability in solving mathematical problems. However, existing approaches primarily focus on improving the quality of correct training data, e.g., distilling high-quality correct solutions from advanced models, neglecting the value contained in error data, potentially hindering the model's reflective ability. Though some studies attempt to leverage error data, they often involve complex mechanisms, such as Monte Carlo Tree Search (MCTS) to explore error nodes. In this work, we propose to enhance LLMs' reasoning ability by Learning from Errors for Mathematical Advancement (LEMMA). LEMMA constructs data consisting of an incorrect solution with an erroneous step and a reflection connection to a correct solution for fine-tuning. Specifically, we systematically analyze the model-generated error types and introduce an error-type grounded mistake augmentation method to collect diverse and representative errors. Correct solutions are either from fixing the errors or generating a fresh start. Through a model-aware smooth reflection connection, the erroneous solution is transferred to the correct one. By fine-tuning on the constructed dataset, the model is able to self-correct errors autonomously within the generation process without relying on external critique models. Experimental results demonstrate that LEMMA achieves significant performance improvements over other strong baselines.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17422v1">V-Seek: Accelerating LLM Reasoning on Open-hardware Server-class RISC-V Platforms</a></div>
    <div class="paper-meta">
      📅 2025-03-21
    </div>
    <details class="paper-abstract">
      The recent exponential growth of Large Language Models (LLMs) has relied on GPU-based systems. However, CPUs are emerging as a flexible and lower-cost alternative, especially when targeting inference and reasoning workloads. RISC-V is rapidly gaining traction in this area, given its open and vendor-neutral ISA. However, the RISC-V hardware for LLM workloads and the corresponding software ecosystem are not fully mature and streamlined, given the requirement of domain-specific tuning. This paper aims at filling this gap, focusing on optimizing LLM inference on the Sophon SG2042, the first commercially available many-core RISC-V CPU with vector processing capabilities. On two recent state-of-the-art LLMs optimized for reasoning, DeepSeek R1 Distill Llama 8B and DeepSeek R1 Distill QWEN 14B, we achieve 4.32/2.29 token/s for token generation and 6.54/3.68 token/s for prompt processing, with a speed up of up 2.9x/3.0x compared to our baseline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.17421v1">Understanding Social Support Needs in Questions: A Hybrid Approach Integrating Semi-Supervised Learning and LLM-based Data Augmentation</a></div>
    <div class="paper-meta">
      📅 2025-03-21
      | 💬 55 pages
    </div>
    <details class="paper-abstract">
      Patients are increasingly turning to online health Q&A communities for social support to improve their well-being. However, when this support received does not align with their specific needs, it may prove ineffective or even detrimental. This necessitates a model capable of identifying the social support needs in questions. However, training such a model is challenging due to the scarcity and class imbalance issues of labeled data. To overcome these challenges, we follow the computational design science paradigm to develop a novel framework, Hybrid Approach for SOcial Support need classification (HA-SOS). HA-SOS integrates an answer-enhanced semi-supervised learning approach, a text data augmentation technique leveraging large language models (LLMs) with reliability- and diversity-aware sample selection mechanism, and a unified training process to automatically label social support needs in questions. Extensive empirical evaluations demonstrate that HA-SOS significantly outperforms existing question classification models and alternative semi-supervised learning approaches. This research contributes to the literature on social support, question classification, semi-supervised learning, and text data augmentation. In practice, our HA-SOS framework facilitates online Q&A platform managers and answerers to better understand users' social support needs, enabling them to provide timely, personalized answers and interventions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.16416v1">Survey on Evaluation of LLM-based Agents</a></div>
    <div class="paper-meta">
      📅 2025-03-20
    </div>
    <details class="paper-abstract">
      The emergence of LLM-based agents represents a paradigm shift in AI, enabling autonomous systems to plan, reason, use tools, and maintain memory while interacting with dynamic environments. This paper provides the first comprehensive survey of evaluation methodologies for these increasingly capable agents. We systematically analyze evaluation benchmarks and frameworks across four critical dimensions: (1) fundamental agent capabilities, including planning, tool use, self-reflection, and memory; (2) application-specific benchmarks for web, software engineering, scientific, and conversational agents; (3) benchmarks for generalist agents; and (4) frameworks for evaluating agents. Our analysis reveals emerging trends, including a shift toward more realistic, challenging evaluations with continuously updated benchmarks. We also identify critical gaps that future research must address-particularly in assessing cost-efficiency, safety, and robustness, and in developing fine-grained, and scalable evaluation methods. This survey maps the rapidly evolving landscape of agent evaluation, reveals the emerging trends in the field, identifies current limitations, and proposes directions for future research.
    </details>
</div>
