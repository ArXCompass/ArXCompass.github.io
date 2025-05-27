# llm - 2025_05

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
- Part 9
- [Part 10](papers_10.md)
- [Part 11](papers_11.md)
- [Part 12](papers_12.md)

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09598v1">How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) spread across industries, understanding their environmental footprint at the inference level is no longer optional; it is essential. However, most existing studies exclude proprietary models, overlook infrastructural variability and overhead, or focus solely on training, even as inference increasingly dominates AI's environmental impact. To bridge this gap, this paper introduces a novel infrastructure-aware benchmarking framework for quantifying the environmental footprint of LLM inference across 30 state-of-the-art models as deployed in commercial data centers. Our framework combines public API performance data with region-specific environmental multipliers and statistical inference of hardware configurations. We additionally utilize cross-efficiency Data Envelopment Analysis (DEA) to rank models by performance relative to environmental cost. Our results show that o3 and DeepSeek-R1 emerge as the most energy-intensive models, consuming over 33 Wh per long prompt, more than 70 times the consumption of GPT-4.1 nano, and that Claude-3.7 Sonnet ranks highest in eco-efficiency. While a single short GPT-4o query consumes 0.43 Wh, scaling this to 700 million queries/day results in substantial annual environmental impacts. These include electricity use comparable to 35,000 U.S. homes, freshwater evaporation matching the annual drinking needs of 1.2 million people, and carbon emissions requiring a Chicago-sized forest to offset. These findings illustrate a growing paradox: although individual queries are efficient, their global scale drives disproportionate resource consumption. Our study provides a standardized, empirically grounded methodology for benchmarking the sustainability of LLM deployments, laying a foundation for future environmental accountability in AI development and sustainability standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09439v1">Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      We propose Omni-R1 which fine-tunes a recent multi-modal LLM, Qwen2.5-Omni, on an audio question answering dataset with the reinforcement learning method GRPO. This leads to new State-of-the-Art performance on the recent MMAU benchmark. Omni-R1 achieves the highest accuracies on the sounds, music, speech, and overall average categories, both on the Test-mini and Test-full splits. To understand the performance improvement, we tested models both with and without audio and found that much of the performance improvement from GRPO could be attributed to better text-based reasoning. We also made a surprising discovery that fine-tuning without audio on a text-only dataset was effective at improving the audio-based performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09427v1">SafePath: Conformal Prediction for Safe LLM-Based Autonomous Navigation</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) show growing promise in autonomous driving by reasoning over complex traffic scenarios to generate path plans. However, their tendencies toward overconfidence, and hallucinations raise critical safety concerns. We introduce SafePath, a modular framework that augments LLM-based path planning with formal safety guarantees using conformal prediction. SafePath operates in three stages. In the first stage, we use an LLM that generates a set of diverse candidate paths, exploring possible trajectories based on agent behaviors and environmental cues. In the second stage, SafePath filters out high-risk trajectories while guaranteeing that at least one safe option is included with a user-defined probability, through a multiple-choice question-answering formulation that integrates conformal prediction. In the final stage, our approach selects the path with the lowest expected collision risk when uncertainty is low or delegates control to a human when uncertainty is high. We theoretically prove that SafePath guarantees a safe trajectory with a user-defined probability, and we show how its human delegation rate can be tuned to balance autonomy and safety. Extensive experiments on nuScenes and Highway-env show that SafePath reduces planning uncertainty by 77\% and collision rates by up to 70\%, demonstrating effectiveness in making LLM-driven path planning more safer.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09396v1">The Influence of Human-inspired Agentic Sophistication in LLM-driven Strategic Reasoners</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      The rapid rise of large language models (LLMs) has shifted artificial intelligence (AI) research toward agentic systems, motivating the use of weaker and more flexible notions of agency. However, this shift raises key questions about the extent to which LLM-based agents replicate human strategic reasoning, particularly in game-theoretic settings. In this context, we examine the role of agentic sophistication in shaping artificial reasoners' performance by evaluating three agent designs: a simple game-theoretic model, an unstructured LLM-as-agent model, and an LLM integrated into a traditional agentic framework. Using guessing games as a testbed, we benchmarked these agents against human participants across general reasoning patterns and individual role-based objectives. Furthermore, we introduced obfuscated game scenarios to assess agents' ability to generalise beyond training distributions. Our analysis, covering over 2000 reasoning samples across 25 agent configurations, shows that human-inspired cognitive structures can enhance LLM agents' alignment with human strategic behaviour. Still, the relationship between agentic design complexity and human-likeness is non-linear, highlighting a critical dependence on underlying LLM capabilities and suggesting limits to simple architectural augmentation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.03343v2">What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Jailbreaks have been a central focus of research regarding the safety and reliability of large language models (LLMs), yet the mechanisms underlying these attacks remain poorly understood. While previous studies have predominantly relied on linear methods to detect jailbreak attempts and model refusals, we take a different approach by examining both linear and non-linear features in prompts that lead to successful jailbreaks. First, we introduce a novel dataset comprising 10,800 jailbreak attempts spanning 35 diverse attack methods. Leveraging this dataset, we train probes to classify successful from unsuccessful jailbreaks using the latent representations corresponding to prompt tokens. Notably, we find that even when probes achieve high accuracy in predicting the success of jailbreaks, their performance often fails to generalize to unseen attack methods. This reveals that different jailbreaking strategies exploit different non-linear, non-universal features. Next, we demonstrate that non-linear probes provide a powerful tool for steering model behavior. Specifically, we use these probes to guide targeted latent space perturbations, enabling us to effectively modulate the model's robustness against jailbreaks. Overall, our findings challenge the assumption that jailbreaks can be fully understood through linear or simple universal prompt features alone, highlighting the importance of a nuanced understanding of the mechanisms behind LLM vulnerabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09319v1">Statistical Modeling and Uncertainty Estimation of LLM Inference Systems</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Large Language Model (LLM) inference systems present significant challenges in statistical performance characterization due to dynamic workload variations, diverse hardware architectures, and complex interactions between model size, batch processing, and throughput requirements. Accurate statistical characterization enables better workload scheduling, adaptive resource provisioning, and cost-aware inference optimization, making it crucial for improving efficiency in large-scale AI deployments. Traditional analytical models provide explainability but cannot cover the vast diversity of real-world workloads, making it impossible to benchmark every scenario in advance. Machine learning (ML) approaches effectively predict performance for non-benchmarked cases but struggle when extrapolating beyond their observed training space. To address these limitations for LLM inference systems, we propose an Analytical with Learning Augmentation (ALA) framework that bridges analytical modeling with \ml for robust statistical prediction and uncertainty estimation in LLM inference workloads. Our method employs an analytical throughput model with parameters estimated for benchmarked workloads, then extends to unobserved configurations using \ml predictions. We enhance this with simulated annealing to exploit subsets of the workload data point combinations and develop an error predictor. Finally, we quantify uncertainty based on vector space similarity between new and observed workloads to ensure robust generalization. Through extensive experimentation on diverse LLM inference workloads, we demonstrate that our framework achieves low median errors while maintaining adaptability to new inference scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09289v1">Reproducibility Study of "Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents"</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 11 Tables, 9 Figures
    </div>
    <details class="paper-abstract">
      This study evaluates and extends the findings made by Piatti et al., who introduced GovSim, a simulation framework designed to assess the cooperative decision-making capabilities of large language models (LLMs) in resource-sharing scenarios. By replicating key experiments, we validate claims regarding the performance of large models, such as GPT-4-turbo, compared to smaller models. The impact of the universalization principle is also examined, with results showing that large models can achieve sustainable cooperation, with or without the principle, while smaller models fail without it. In addition, we provide multiple extensions to explore the applicability of the framework to new settings. We evaluate additional models, such as DeepSeek-V3 and GPT-4o-mini, to test whether cooperative behavior generalizes across different architectures and model sizes. Furthermore, we introduce new settings: we create a heterogeneous multi-agent environment, study a scenario using Japanese instructions, and explore an "inverse environment" where agents must cooperate to mitigate harmful resource distributions. Our results confirm that the benchmark can be applied to new models, scenarios, and languages, offering valuable insights into the adaptability of LLMs in complex cooperative tasks. Moreover, the experiment involving heterogeneous multi-agent systems demonstrates that high-performing models can influence lower-performing ones to adopt similar behaviors. This finding has significant implications for other agent-based applications, potentially enabling more efficient use of computational resources and contributing to the development of more effective cooperative AI systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16472v2">Harden and Catch for Just-in-Time Assured LLM-Based Software Testing: Open Research Challenges</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 To Appear as keynote paper at FSE 2025
    </div>
    <details class="paper-abstract">
      Despite decades of research and practice in automated software testing, several fundamental concepts remain ill-defined and under-explored, yet offer enormous potential real-world impact. We show that these concepts raise exciting new challenges in the context of Large Language Models for software test generation. More specifically, we formally define and investigate the properties of hardening and catching tests. A hardening test is one that seeks to protect against future regressions, while a catching test is one that catches such a regression or a fault in new functionality introduced by a code change. Hardening tests can be generated at any time and may become catching tests when a future regression is caught. We also define and motivate the Catching 'Just-in-Time' (JiTTest) Challenge, in which tests are generated 'just-in-time' to catch new faults before they land into production. We show that any solution to Catching JiTTest generation can also be repurposed to catch latent faults in legacy code. We enumerate possible outcomes for hardening and catching tests and JiTTests, and discuss open research problems, deployment options, and initial results from our work on automated LLM-based hardening at Meta. This paper was written to accompany the keynote by the authors at the ACM International Conference on the Foundations of Software Engineering (FSE) 2025. Author order is alphabetical. The corresponding author is Mark Harman.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2402.01383v3">LLM-based NLG Evaluation: Current Status and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Evaluating natural language generation (NLG) is a vital but challenging problem in natural language processing. Traditional evaluation metrics mainly capturing content (e.g. n-gram) overlap between system outputs and references are far from satisfactory, and large language models (LLMs) such as ChatGPT have demonstrated great potential in NLG evaluation in recent years. Various automatic evaluation methods based on LLMs have been proposed, including metrics derived from LLMs, prompting LLMs, fine-tuning LLMs, and human-LLM collaborative evaluation. In this survey, we first give a taxonomy of LLM-based NLG evaluation methods, and discuss their pros and cons, respectively. Lastly, we discuss several open problems in this area and point out future research directions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09142v1">ELIS: Efficient LLM Iterative Scheduling System with Response Length Predictor</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 13 pages, 5 figures. Cloud-native LLM scheduling system with latency-aware inference optimization
    </div>
    <details class="paper-abstract">
      We propose ELIS, a serving system for Large Language Models (LLMs) featuring an Iterative Shortest Remaining Time First (ISRTF) scheduler designed to efficiently manage inference tasks with the shortest remaining tokens. Current LLM serving systems often employ a first-come-first-served scheduling strategy, which can lead to the "head-of-line blocking" problem. To overcome this limitation, it is necessary to predict LLM inference times and apply a shortest job first scheduling strategy. However, due to the auto-regressive nature of LLMs, predicting the inference latency is challenging. ELIS addresses this challenge by training a response length predictor for LLMs using the BGE model, an encoder-based state-of-the-art model. Additionally, we have devised the ISRTF scheduling strategy, an optimization of shortest remaining time first tailored to existing LLM iteration batching. To evaluate our work in an industrial setting, we simulate streams of requests based on our study of real-world user LLM serving trace records. Furthermore, we implemented ELIS as a cloud-native scheduler system on Kubernetes to evaluate its performance in production environments. Our experimental results demonstrate that ISRTF reduces the average job completion time by up to 19.6%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.18599v2">Oaken: Fast and Efficient LLM Serving with Online-Offline Hybrid KV Cache Quantization</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 16 pages, 14 figures, and 4 tables
    </div>
    <details class="paper-abstract">
      Modern Large Language Model serving system batches multiple requests to achieve high throughput, while batching attention operations is challenging, rendering memory bandwidth a critical bottleneck. The community relies on high-end GPUs with multiple high-bandwidth memory channels. Unfortunately, HBM's high bandwidth often comes at the expense of limited memory capacity, which reduces core utilization and increases costs. Recent advancements enabling longer contexts for LLMs have substantially increased the key-value cache size, further intensifying the pressures on memory capacity. The literature has explored KV cache quantization techniques, which commonly use low bitwidth for most values, selectively using higher bitwidth for outlier values. While this approach helps achieve high accuracy and low bitwidth simultaneously, it comes with the limitation that cost for online outlier detection is excessively high, negating the advantages. We propose Oaken, an acceleration solution that achieves high accuracy and high performance simultaneously through co-designing algorithm and hardware. To effectively find a sweet spot in the accuracy-performance trade-off space of KV cache quantization, Oaken employs an online-offline hybrid approach, setting outlier thresholds offline, which are then used to determine the quantization scale online. To translate the proposed algorithmic technique into tangible performance gains, Oaken also comes with custom quantization engines and memory management units that can be integrated with any LLM accelerators. We built an Oaken accelerator on top of an LLM accelerator, LPU, and conducted a comprehensive evaluation. Our experiments show that for a batch size of 256, Oaken achieves up to 1.58x throughput improvement over NVIDIA A100 GPU, incurring a minimal accuracy loss of only 0.54\% on average, compared to state-of-the-art KV cache quantization techniques.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09082v1">CEC-Zero: Chinese Error Correction Solution Based on LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) demonstrate exceptional Chinese text processing capabilities, particularly in Chinese Spelling Correction (CSC). While LLMs outperform traditional BERT-based models in accuracy and robustness, challenges persist in reliability and generalization. This paper proposes CEC-Zero, a novel reinforcement learning (RL) framework enabling LLMs to self-correct through autonomous error strategy learning without external supervision. By integrating RL with LLMs' generative power, the method eliminates dependency on annotated data or auxiliary models. Experiments reveal RL-enhanced LLMs achieve industry-viable accuracy and superior cross-domain generalization, offering a scalable solution for reliability optimization in Chinese NLP applications. This breakthrough facilitates LLM deployment in practical Chinese text correction scenarios while establishing a new paradigm for self-improving language models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.09116v2">P-MMEval: A Parallel Multilingual Multitask Benchmark for Consistent Evaluation of LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Recent advancements in large language models (LLMs) showcase varied multilingual capabilities across tasks like translation, code generation, and reasoning. Previous assessments often limited their scope to fundamental natural language processing (NLP) or isolated capability-specific tasks. To alleviate this drawback, we aim to present a comprehensive multilingual multitask benchmark. First, we introduce P-MMEval, a large-scale benchmark covering effective fundamental and capability-specialized datasets. Furthermore, P-MMEval delivers consistent language coverage across various datasets and provides parallel samples. Finally, we conduct extensive experiments on representative multilingual model series to compare performances across models and tasks, explore the relationship between multilingual performances and factors such as tasks, model sizes, languages, and prompts, and examine the effectiveness of knowledge transfer from English to other languages. The resulting insights are intended to offer valuable guidance for future research. The dataset is available at https://huggingface.co/datasets/Qwen/P-MMEval.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09852v1">Do Large Language Models Know Conflict? Investigating Parametric vs. Non-Parametric Knowledge of LLMs for Conflict Forecasting</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have shown impressive performance across natural language tasks, but their ability to forecast violent conflict remains underexplored. We investigate whether LLMs possess meaningful parametric knowledge-encoded in their pretrained weights-to predict conflict escalation and fatalities without external data. This is critical for early warning systems, humanitarian planning, and policy-making. We compare this parametric knowledge with non-parametric capabilities, where LLMs access structured and unstructured context from conflict datasets (e.g., ACLED, GDELT) and recent news reports via Retrieval-Augmented Generation (RAG). Incorporating external information could enhance model performance by providing up-to-date context otherwise missing from pretrained weights. Our two-part evaluation framework spans 2020-2024 across conflict-prone regions in the Horn of Africa and the Middle East. In the parametric setting, LLMs predict conflict trends and fatalities relying only on pretrained knowledge. In the non-parametric setting, models receive summaries of recent conflict events, indicators, and geopolitical developments. We compare predicted conflict trend labels (e.g., Escalate, Stable Conflict, De-escalate, Peace) and fatalities against historical data. Our findings highlight the strengths and limitations of LLMs for conflict forecasting and the benefits of augmenting them with structured external knowledge.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09810v1">Lossless Compression for LLM Tensor Incremental Snapshots</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      During the training of Large Language Models (LLMs), tensor data is periodically "checkpointed" to persistent storage to allow recovery of work done in the event of failure. The volume of data that must be copied during each checkpoint, even when using reduced-precision representations such as bfloat16, often reaches hundreds of gigabytes. Furthermore, the data must be moved across a network and written to a storage system before the next epoch occurs. With a view to ultimately building an optimized checkpointing solution, this paper presents experimental analysis of checkpoint data used to derive a design that maximizes the use of lossless compression to reduce the volume of data. We examine how tensor data and its compressibility evolve during model training and evaluate the efficacy of existing common off-the-shelf general purpose compression engines combined with known data optimization techniques such as byte-grouping and incremental delta compression. Leveraging our analysis we have built an effective compression solution, known as Language Model Compressor (LMC), which is based on byte-grouping and Huffman encoding. LMC offers more compression performance than the best alternative (BZ2) but with an order-of-magnitude reduction in the time needed to perform the compression. We show that a 16-core parallel implementation of LMC can attain compression and decompression throughput of 2.78 GiB/s and 3.76 GiB/s respectively. This increase in performance ultimately reduces the CPU resources needed and provides more time to copy the data to the storage system before the next epoch thus allowing for higher-frequency checkpoints.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09807v1">Exploring the generalization of LLM truth directions on conversational formats</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Several recent works argue that LLMs have a universal truth direction where true and false statements are linearly separable in the activation space of the model. It has been demonstrated that linear probes trained on a single hidden state of the model already generalize across a range of topics and might even be used for lie detection in LLM conversations. In this work we explore how this truth direction generalizes between various conversational formats. We find good generalization between short conversations that end on a lie, but poor generalization to longer formats where the lie appears earlier in the input prompt. We propose a solution that significantly improves this type of generalization by adding a fixed key phrase at the end of each conversation. Our results highlight the challenges towards reliable LLM lie detectors that generalize to new settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09806v1">Learn, Explore and Reflect by Chatting: Understanding the Value of an LLM-Based Voting Advice Application Chatbot</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 Accepted to ACM CUI 2025
    </div>
    <details class="paper-abstract">
      Voting advice applications (VAAs), which have become increasingly prominent in European elections, are seen as a successful tool for boosting electorates' political knowledge and engagement. However, VAAs' complex language and rigid presentation constrain their utility to less-sophisticated voters. While previous work enhanced VAAs' click-based interaction with scripted explanations, a conversational chatbot's potential for tailored discussion and deliberate political decision-making remains untapped. Our exploratory mixed-method study investigates how LLM-based chatbots can support voting preparation. We deployed a VAA chatbot to 331 users before Germany's 2024 European Parliament election, gathering insights from surveys, conversation logs, and 10 follow-up interviews. Participants found the VAA chatbot intuitive and informative, citing its simple language and flexible interaction. We further uncovered VAA chatbots' role as a catalyst for reflection and rationalization. Expanding on participants' desire for transparency, we provide design recommendations for building interactive and trustworthy VAA chatbots.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09724v1">An AI-Powered Research Assistant in the Lab: A Practical Guide for Text Analysis Through Iterative Collaboration with LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-14
      | 💬 31 pages, 1 figure
    </div>
    <details class="paper-abstract">
      Analyzing texts such as open-ended responses, headlines, or social media posts is a time- and labor-intensive process highly susceptible to bias. LLMs are promising tools for text analysis, using either a predefined (top-down) or a data-driven (bottom-up) taxonomy, without sacrificing quality. Here we present a step-by-step tutorial to efficiently develop, test, and apply taxonomies for analyzing unstructured data through an iterative and collaborative process between researchers and LLMs. Using personal goals provided by participants as an example, we demonstrate how to write prompts to review datasets and generate a taxonomy of life domains, evaluate and refine the taxonomy through prompt and direct modifications, test the taxonomy and assess intercoder agreements, and apply the taxonomy to categorize an entire dataset with high intercoder reliability. We discuss the possibilities and limitations of using LLMs for text analysis.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09665v1">Tales of the 2025 Los Angeles Fire: Hotwash for Public Health Concerns in Reddit via LLM-Enhanced Topic Modeling</a></div>
    <div class="paper-meta">
      📅 2025-05-14
    </div>
    <details class="paper-abstract">
      Wildfires have become increasingly frequent, irregular, and severe in recent years. Understanding how affected populations perceive and respond during wildfire crises is critical for timely and empathetic disaster response. Social media platforms offer a crowd-sourced channel to capture evolving public discourse, providing hyperlocal information and insight into public sentiment. This study analyzes Reddit discourse during the 2025 Los Angeles wildfires, spanning from the onset of the disaster to full containment. We collect 385 posts and 114,879 comments related to the Palisades and Eaton fires. We adopt topic modeling methods to identify the latent topics, enhanced by large language models (LLMs) and human-in-the-loop (HITL) refinement. Furthermore, we develop a hierarchical framework to categorize latent topics, consisting of two main categories, Situational Awareness (SA) and Crisis Narratives (CN). The volume of SA category closely aligns with real-world fire progressions, peaking within the first 2-5 days as the fires reach the maximum extent. The most frequent co-occurring category set of public health and safety, loss and damage, and emergency resources expands on a wide range of health-related latent topics, including environmental health, occupational health, and one health. Grief signals and mental health risks consistently accounted for 60 percentage and 40 percentage of CN instances, respectively, with the highest total volume occurring at night. This study contributes the first annotated social media dataset on the 2025 LA fires, and introduces a scalable multi-layer framework that leverages topic modeling for crisis discourse analysis. By identifying persistent public health concerns, our results can inform more empathetic and adaptive strategies for disaster response, public health communication, and future research in comparable climate-related disaster events.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08783v1">CodePDE: An Inference Framework for LLM-driven PDE Solver Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Partial differential equations (PDEs) are fundamental to modeling physical systems, yet solving them remains a complex challenge. Traditional numerical solvers rely on expert knowledge to implement and are computationally expensive, while neural-network-based solvers require large training datasets and often lack interpretability. In this work, we frame PDE solving as a code generation task and introduce CodePDE, the first inference framework for generating PDE solvers using large language models (LLMs). Leveraging advanced inference-time algorithms and scaling strategies, CodePDE unlocks critical capacities of LLM for PDE solving: reasoning, debugging, selfrefinement, and test-time scaling -- all without task-specific tuning. CodePDE achieves superhuman performance across a range of representative PDE problems. We also present a systematic empirical analysis of LLM generated solvers, analyzing their accuracy, efficiency, and numerical scheme choices. Our findings highlight the promise and the current limitations of LLMs in PDE solving, offering a new perspective on solver design and opportunities for future model development. Our code is available at https://github.com/LithiumDA/CodePDE.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02870v2">AI Hiring with LLMs: A Context-Aware and Explainable Multi-Agent Framework for Resume Screening</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 Accepted by CVPR 2025 Workshop
    </div>
    <details class="paper-abstract">
      Resume screening is a critical yet time-intensive process in talent acquisition, requiring recruiters to analyze vast volume of job applications while remaining objective, accurate, and fair. With the advancements in Large Language Models (LLMs), their reasoning capabilities and extensive knowledge bases demonstrate new opportunities to streamline and automate recruitment workflows. In this work, we propose a multi-agent framework for resume screening using LLMs to systematically process and evaluate resumes. The framework consists of four core agents, including a resume extractor, an evaluator, a summarizer, and a score formatter. To enhance the contextual relevance of candidate assessments, we integrate Retrieval-Augmented Generation (RAG) within the resume evaluator, allowing incorporation of external knowledge sources, such as industry-specific expertise, professional certifications, university rankings, and company-specific hiring criteria. This dynamic adaptation enables personalized recruitment, bridging the gap between AI automation and talent acquisition. We assess the effectiveness of our approach by comparing AI-generated scores with ratings provided by HR professionals on a dataset of anonymized online resumes. The findings highlight the potential of multi-agent RAG-LLM systems in automating resume screening, enabling more efficient and scalable hiring workflows.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.02732v3">Why do LLMs attend to the first token?</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) tend to attend heavily to the first token in the sequence -- creating a so-called attention sink. Many works have studied this phenomenon in detail, proposing various ways to either leverage or alleviate it. Attention sinks have been connected to quantisation difficulties, security issues, and streaming attention. Yet, while many works have provided conditions in which they occur or not, a critical question remains shallowly answered: Why do LLMs learn such patterns and how are they being used? In this work, we argue theoretically and empirically that this mechanism provides a method for LLMs to avoid over-mixing, connecting this to existing lines of work that study mathematically how information propagates in Transformers. We conduct experiments to validate our theoretical intuitions and show how choices such as context length, depth, and data packing influence the sink behaviour. We hope that this study provides a new practical perspective on why attention sinks are useful in LLMs, leading to a better understanding of the attention patterns that form during training.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00651v2">Open-Source LLM-Driven Federated Transformer for Predictive IoV Management</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 Preprint version; submitted for academic peer review
    </div>
    <details class="paper-abstract">
      The proliferation of connected vehicles within the Internet of Vehicles (IoV) ecosystem presents critical challenges in ensuring scalable, real-time, and privacy-preserving traffic management. Existing centralized IoV solutions often suffer from high latency, limited scalability, and reliance on proprietary Artificial Intelligence (AI) models, creating significant barriers to widespread deployment, particularly in dynamic and privacy-sensitive environments. Meanwhile, integrating Large Language Models (LLMs) in vehicular systems remains underexplored, especially concerning prompt optimization and effective utilization in federated contexts. To address these challenges, we propose the Federated Prompt-Optimized Traffic Transformer (FPoTT), a novel framework that leverages open-source LLMs for predictive IoV management. FPoTT introduces a dynamic prompt optimization mechanism that iteratively refines textual prompts to enhance trajectory prediction. The architecture employs a dual-layer federated learning paradigm, combining lightweight edge models for real-time inference with cloud-based LLMs to retain global intelligence. A Transformer-driven synthetic data generator is incorporated to augment training with diverse, high-fidelity traffic scenarios in the Next Generation Simulation (NGSIM) format. Extensive evaluations demonstrate that FPoTT, utilizing EleutherAI Pythia-1B, achieves 99.86% prediction accuracy on real-world data while maintaining high performance on synthetic datasets. These results underscore the potential of open-source LLMs in enabling secure, adaptive, and scalable IoV management, offering a promising alternative to proprietary solutions in smart mobility ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08704v1">LLM-based Prompt Ensemble for Reliable Medical Entity Recognition from EHRs</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 IEEE 26th International Conference on Information Reuse and Integration for Data Science (IRI 2025), San Jose, CA, USA
    </div>
    <details class="paper-abstract">
      Electronic Health Records (EHRs) are digital records of patient information, often containing unstructured clinical text. Named Entity Recognition (NER) is essential in EHRs for extracting key medical entities like problems, tests, and treatments to support downstream clinical applications. This paper explores prompt-based medical entity recognition using large language models (LLMs), specifically GPT-4o and DeepSeek-R1, guided by various prompt engineering techniques, including zero-shot, few-shot, and an ensemble approach. Among all strategies, GPT-4o with prompt ensemble achieved the highest classification performance with an F1-score of 0.95 and recall of 0.98, outperforming DeepSeek-R1 on the task. The ensemble method improved reliability by aggregating outputs through embedding-based similarity and majority voting.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08662v1">Revealing economic facts: LLMs know more than they say</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 34 pages, 17 figures
    </div>
    <details class="paper-abstract">
      We investigate whether the hidden states of large language models (LLMs) can be used to estimate and impute economic and financial statistics. Focusing on county-level (e.g. unemployment) and firm-level (e.g. total assets) variables, we show that a simple linear model trained on the hidden states of open-source LLMs outperforms the models' text outputs. This suggests that hidden states capture richer economic information than the responses of the LLMs reveal directly. A learning curve analysis indicates that only a few dozen labelled examples are sufficient for training. We also propose a transfer learning method that improves estimation accuracy without requiring any labelled data for the target variable. Finally, we demonstrate the practical utility of hidden-state representations in super-resolution and data imputation tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08600v1">Automatic Task Detection and Heterogeneous LLM Speculative Decoding</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 10 pages, 10 figures, 2 tables
    </div>
    <details class="paper-abstract">
      Speculative decoding, which combines a draft model with a target model, has emerged as an effective approach to accelerate large language model (LLM) inference. However, existing methods often face a trade-off between the acceptance rate and decoding speed in downstream tasks due to the limited capacity of the draft model, making it difficult to ensure efficiency across diverse tasks. To address this problem, we propose a speculative decoding algorithm tailored for downstream task optimization. It includes an automatic task partitioning and assigning method, which automatically categorizes downstream tasks into different sub-tasks and assigns them to a set of heterogeneous draft models. Each draft model is aligned with the target model using task-specific data, thereby enhancing the consistency of inference results. In addition, our proposed method incorporates an online lightweight prompt classifier to dynamically route prompts to the appropriate draft model. Experimental results demonstrate that the proposed method improves draft accuracy by 6% to 50% over vanilla speculative decoding, while achieving a speedup of 1.10x to 2.64x in LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08590v1">Enhancing Thyroid Cytology Diagnosis with RAG-Optimized LLMs and Pa-thology Foundation Models</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Advancements in artificial intelligence (AI) are transforming pathology by integrat-ing large language models (LLMs) with retrieval-augmented generation (RAG) and domain-specific foundation models. This study explores the application of RAG-enhanced LLMs coupled with pathology foundation models for thyroid cytology diagnosis, addressing challenges in cytological interpretation, standardization, and diagnostic accuracy. By leveraging a curated knowledge base, RAG facilitates dy-namic retrieval of relevant case studies, diagnostic criteria, and expert interpreta-tion, improving the contextual understanding of LLMs. Meanwhile, pathology foun-dation models, trained on high-resolution pathology images, refine feature extrac-tion and classification capabilities. The fusion of these AI-driven approaches en-hances diagnostic consistency, reduces variability, and supports pathologists in dis-tinguishing benign from malignant thyroid lesions. Our results demonstrate that integrating RAG with pathology-specific LLMs significantly improves diagnostic efficiency and interpretability, paving the way for AI-assisted thyroid cytopathology, with foundation model UNI achieving AUC 0.73-0.93 for correct prediction of surgi-cal pathology diagnosis from thyroid cytology samples.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08542v1">Guiding LLM-based Smart Contract Generation with Finite State Machine</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Smart contract is a kind of self-executing code based on blockchain technology with a wide range of application scenarios, but the traditional generation method relies on manual coding and expert auditing, which has a high threshold and low efficiency. Although Large Language Models (LLMs) show great potential in programming tasks, they still face challenges in smart contract generation w.r.t. effectiveness and security. To solve these problems, we propose FSM-SCG, a smart contract generation framework based on finite state machine (FSM) and LLMs, which significantly improves the quality of the generated code by abstracting user requirements to generate FSM, guiding LLMs to generate smart contracts, and iteratively optimizing the code with the feedback of compilation and security checks. The experimental results show that FSM-SCG significantly improves the quality of smart contract generation. Compared to the best baseline, FSM-SCG improves the compilation success rate of generated smart contract code by at most 48%, and reduces the average vulnerability risk score by approximately 68%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2411.00782v2">TradExpert: Revolutionizing Trading with Mixture of Expert LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      The integration of Artificial Intelligence (AI) in the financial domain has opened new avenues for quantitative trading, particularly through the use of Large Language Models (LLMs). However, the challenge of effectively synthesizing insights from diverse data sources and integrating both structured and unstructured data persists. This paper presents TradeExpert, a novel framework that employs a mix of experts (MoE) approach, using four specialized LLMs, each analyzing distinct sources of financial data, including news articles, market data, alpha factors, and fundamental data. The insights of these expert LLMs are further synthesized by a General Expert LLM to make a final prediction or decision. With specific prompts, TradeExpert can be switched between the prediction mode and the ranking mode for stock movement prediction and quantitative stock trading, respectively. In addition to existing benchmarks, we also release a large-scale financial dataset to comprehensively evaluate TradeExpert's effectiveness. Our experimental results demonstrate TradeExpert's superior performance across all trading scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08492v1">Achieving Scalable Robot Autonomy via neurosymbolic planning using lightweight local LLM</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 19 pages, 3 figures, 4 tables, accepted at IAS 2025
    </div>
    <details class="paper-abstract">
      PDDL-based symbolic task planning remains pivotal for robot autonomy yet struggles with dynamic human-robot collaboration due to scalability, re-planning demands, and delayed plan availability. Although a few neurosymbolic frameworks have previously leveraged LLMs such as GPT-3 to address these challenges, reliance on closed-source, remote models with limited context introduced critical constraints: third-party dependency, inconsistent response times, restricted plan length and complexity, and multi-domain scalability issues. We present Gideon, a novel framework that enables the transition to modern, smaller, local LLMs with extended context length. Gideon integrates a novel problem generator to systematically generate large-scale datasets of realistic domain-problem-plan tuples for any domain, and adapts neurosymbolic planning for local LLMs, enabling on-device execution and extended context for multi-domain support. Preliminary experiments in single-domain scenarios performed on Qwen-2.5 1.5B and trained on 8k-32k samples, demonstrate a valid plan percentage of 66.1% (32k model) and show that the figure can be further scaled through additional data. Multi-domain tests on 16k samples yield an even higher 70.6% planning validity rate, proving extensibility across domains and signaling that data variety can have a positive effect on learning efficiency. Although long-horizon planning and reduced model size make Gideon training much less efficient than baseline models based on larger LLMs, the results are still significant considering that the trained model is about 120x smaller than baseline and that significant advantages can be achieved in inference efficiency, scalability, and multi-domain adaptability, all critical factors in human-robot collaboration. Training inefficiency can be mitigated by Gideon's streamlined data generation pipeline.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21098v3">Alleviating LLM-based Generative Retrieval Hallucination in Alipay Search</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 Accepted by SIGIR 2025
    </div>
    <details class="paper-abstract">
      Generative retrieval (GR) has revolutionized document retrieval with the advent of large language models (LLMs), and LLM-based GR is gradually being adopted by the industry. Despite its remarkable advantages and potential, LLM-based GR suffers from hallucination and generates documents that are irrelevant to the query in some instances, severely challenging its credibility in practical applications. We thereby propose an optimized GR framework designed to alleviate retrieval hallucination, which integrates knowledge distillation reasoning in model training and incorporate decision agent to further improve retrieval precision. Specifically, we employ LLMs to assess and reason GR retrieved query-document (q-d) pairs, and then distill the reasoning data as transferred knowledge to the GR model. Moreover, we utilize a decision agent as post-processing to extend the GR retrieved documents through retrieval model and select the most relevant ones from multi perspectives as the final generative retrieval result. Extensive offline experiments on real-world datasets and online A/B tests on Fund Search and Insurance Search in Alipay demonstrate our framework's superiority and effectiveness in improving search quality and conversion gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.20799v2">Hallucination by Code Generation LLMs: Taxonomy, Benchmarks, Mitigation, and Challenges</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 15 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Recent technical breakthroughs in large language models (LLMs) have enabled them to fluently generate source code. Software developers often leverage both general-purpose and code-specialized LLMs to revise existing code or even generate a whole function from scratch. These capabilities are also beneficial in no-code or low-code contexts, in which one can write programs without a technical background. However, due to their internal design, LLMs are prone to generating hallucinations, which are incorrect, nonsensical, and not justifiable information but difficult to identify its presence. This problem also occurs when generating source code. Once hallucinated code is produced, it is often challenging for users to identify and fix it, especially when such hallucinations can be identified under specific execution paths. As a result, the hallucinated code may remain unnoticed within the codebase. This survey investigates recent studies and techniques relevant to hallucinations generated by CodeLLMs. We categorize the types of hallucinations in the code generated by CodeLLMs, review existing benchmarks and mitigation strategies, and identify open challenges. Based on these findings, this survey outlines further research directions in the detection and removal of hallucinations produced by CodeLLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08450v1">IterKey: Iterative Keyword Generation with LLMs for Enhanced Retrieval Augmented Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Retrieval-Augmented Generation (RAG) has emerged as a way to complement the in-context knowledge of Large Language Models (LLMs) by integrating external documents. However, real-world applications demand not only accuracy but also interpretability. While dense retrieval methods provide high accuracy, they lack interpretability; conversely, sparse retrieval methods offer transparency but often fail to capture the full intent of queries due to their reliance on keyword matching. To address these issues, we introduce IterKey, an LLM-driven iterative keyword generation framework that enhances RAG via sparse retrieval. IterKey consists of three LLM-driven stages: generating keywords for retrieval, generating answers based on retrieved documents, and validating the answers. If validation fails, the process iteratively repeats with refined keywords. Across four QA tasks, experimental results show that IterKey achieves 5% to 20% accuracy improvements over BM25-based RAG and simple baselines. Its performance is comparable to dense retrieval-based RAG and prior iterative query refinement methods using dense models. In summary, IterKey is a novel BM25-based approach leveraging LLMs to iteratively refine RAG, effectively balancing accuracy with interpretability.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08402v1">TUMS: Enhancing Tool-use Abilities of LLMs with Multi-structure Handlers</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 Accepted to ICONIP 2024
    </div>
    <details class="paper-abstract">
      Recently, large language models(LLMs) have played an increasingly important role in solving a wide range of NLP tasks, leveraging their capabilities of natural language understanding and generating. Integration with external tools further enhances LLMs' effectiveness, providing more precise, timely, and specialized responses. However, LLMs still encounter difficulties with non-executable actions and improper actions, which are primarily attributed to incorrect parameters. The process of generating parameters by LLMs is confined to the tool level, employing the coarse-grained strategy without considering the different difficulties of various tools. To address this issue, we propose TUMS, a novel framework designed to enhance the tool-use capabilities of LLMs by transforming tool-level processing into parameter-level processing. Specifically, our framework consists of four key components: (1) an intent recognizer that identifies the user's intent to help LLMs better understand the task; (2) a task decomposer that breaks down complex tasks into simpler subtasks, each involving a tool call; (3) a subtask processor equipped with multi-structure handlers to generate accurate parameters; and (4) an executor. Our empirical studies have evidenced the effectiveness and efficiency of the TUMS framework with an average of 19.6\% and 50.6\% improvement separately on easy and hard benchmarks of ToolQA, meanwhile, we demonstrated the key contribution of each part with ablation experiments, offering more insights and stimulating future research on Tool-augmented LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13989v2">Gradual Binary Search and Dimension Expansion : A general method for activation quantization in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have become pivotal in artificial intelligence, demonstrating strong capabilities in reasoning, understanding, and generating data. However, their deployment on edge devices is hindered by their substantial size, often reaching several billion parameters. Quantization is a widely used method to reduce memory usage and inference time, however LLMs present unique challenges due to the prevalence of outliers in their activations. In this work, we leverage the theoretical advantages of Hadamard matrices over random rotation matrices to push the boundaries of quantization in LLMs. We demonstrate that Hadamard matrices are more effective in reducing outliers, which are a significant obstacle in achieving low-bit quantization. Our method based on a gradual binary search enables 3-bit quantization for weights, activations, and key-value (KV) caches, resulting in a 40% increase in accuracy on common benchmarks compared to SoTA methods. We extend the use of rotation matrices to support non-power-of-2 embedding dimensions, similar to the Qwen architecture, by employing the Paley algorithm. We theoretically demonstrates the superiority of Hadamard matrices in reducing outliers.We achieved 3-bit quantization for weights, activations, and KV cache, significantly enhancing model performance. Our experimental results on multiple models family like Mistral, LLaMA, and Qwen demonstrate the effectiveness of our approach, outperforming existing methods and enabling practical 3-bit quantization.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08351v1">Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      This paper investigates the potentials of Large Language Models (LLMs) as adaptive tutors in the context of second-language learning. In particular, we evaluate whether system prompting can reliably constrain LLMs to generate only text appropriate to the student's competence level. We simulate full teacher-student dialogues in Spanish using instruction-tuned, open-source LLMs ranging in size from 7B to 12B parameters. Dialogues are generated by having an LLM alternate between tutor and student roles with separate chat histories. The output from the tutor model is then used to evaluate the effectiveness of CEFR-based prompting to control text difficulty across three proficiency levels (A1, B1, C1). Our findings suggest that while system prompting can be used to constrain model outputs, prompting alone is too brittle for sustained, long-term interactional contexts - a phenomenon we term alignment drift. Our results provide insights into the feasibility of LLMs for personalized, proficiency-aligned adaptive tutors and provide a scalable method for low-cost evaluation of model performance without human participants.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2305.06018v3">TARGET: Automated Scenario Generation from Traffic Rules for Testing Autonomous Vehicles via Validated LLM-Guided Knowledge Extraction</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Recent incidents with autonomous vehicles highlight the need for rigorous testing to ensure safety and robustness. Constructing test scenarios for autonomous driving systems (ADSs), however, is labor-intensive. We propose TARGET, an end-to-end framework that automatically generates test scenarios from traffic rules. To address complexity, we leverage a Large Language Model (LLM) to extract knowledge from traffic rules. To mitigate hallucinations caused by large context during input processing, we introduce a domain-specific language (DSL) designed to be syntactically simple and compositional. This design allows the LLM to learn and generate test scenarios in a modular manner while enabling syntactic and semantic validation for each component. Based on these validated representations, TARGET synthesizes executable scripts to render scenarios in simulation. Evaluated seven ADSs with 284 scenarios derived from 54 traffic rules, TARGET uncovered 610 rule violations, collisions, and other issues. For each violation, TARGET generates scenario recordings and detailed logs, aiding root cause analysis. Two identified issues were confirmed by ADS developers: one linked to an existing bug report and the other to limited ADS functionality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.17565v3">DeepDistill: Enhancing LLM Reasoning Capabilities via Large-Scale Difficulty-Graded Data Training</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Although large language models (LLMs) have recently achieved remarkable performance on various complex reasoning benchmarks, the academic community still lacks an in-depth understanding of base model training processes and data quality. To address this, we construct a large-scale, difficulty-graded reasoning dataset containing approximately 3.34 million unique queries of varying difficulty levels and about 40 million distilled responses generated by multiple models over several passes. Leveraging pass rate and Coefficient of Variation (CV), we precisely select the most valuable training data to enhance reasoning capability. Notably, we observe a training pattern shift, indicating that reasoning-focused training based on base models requires higher learning rates for effective training. Using this carefully selected data, we significantly improve the reasoning capabilities of the base model, achieving a pass rate of 79.2\% on the AIME2024 mathematical reasoning benchmark. This result surpasses most current distilled models and closely approaches state-of-the-art performance. We provide detailed descriptions of our data processing, difficulty assessment, and training methodology, and have publicly released all datasets and methods to promote rapid progress in open-source long-reasoning LLMs. The dataset is available at: \href{https://huggingface.co/datasets/a-m-team/AM-DeepSeek-Distilled-40M}{https://huggingface.co/datasets/a-m-team/AM-DeepSeek-Distilled-40M}
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08303v1">Evaluating the Effectiveness of Black-Box Prompt Optimization as the Scale of LLMs Continues to Grow</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Black-Box prompt optimization methods have emerged as a promising strategy for refining input prompts to better align large language models (LLMs), thereby enhancing their task performance. Although these methods have demonstrated encouraging results, most studies and experiments have primarily focused on smaller-scale models (e.g., 7B, 14B) or earlier versions (e.g., GPT-3.5) of LLMs. As the scale of LLMs continues to increase, such as with DeepSeek V3 (671B), it remains an open question whether these black-box optimization techniques will continue to yield significant performance improvements for models of such scale. In response to this, we select three well-known black-box optimization methods and evaluate them on large-scale LLMs (DeepSeek V3 and Gemini 2.0 Flash) across four NLU and NLG datasets. The results show that these black-box prompt optimization methods offer only limited improvements on these large-scale LLMs. Furthermore, we hypothesize that the scale of the model is the primary factor contributing to the limited benefits observed. To explore this hypothesis, we conducted experiments on LLMs of varying sizes (Qwen 2.5 series, ranging from 7B to 72B) and observed an inverse scaling law, wherein the effectiveness of black-box optimization methods diminished as the model size increased.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.21625v2">Ask, Fail, Repeat: Meeseeks, an Iterative Feedback Benchmark for LLMs' Multi-turn Instruction-following Ability</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      The ability to follow instructions accurately is fundamental for Large Language Models (LLMs) to serve as reliable agents in real-world applications. For complex instructions, LLMs often struggle to fulfill all requirements in a single attempt. In practice, users typically provide iterative feedback until the LLM generates a response that meets all requirements. However, existing instruction-following benchmarks are either single-turn or introduce new requirements in each turn without allowing self-correction. To address this gap, we propose \textbf{Meeseeks} (named after Mr. Meeseeks from \textit{Rick and Morty}\footnote{Rick and Morty is an American adult animated science fiction sitcom created by Justin Roiland and Dan Harmon for Cartoon Network's nighttime programming block Adult Swim.}.) Meeseeks simulates realistic human-LLM interactions through an iterative feedback framework, which enables models to self-correct based on specific requirement failures in each turn, better reflecting real-world user-end usage patterns. Meanwhile, the benchmark implements a comprehensive evaluation system with 38 capability tags organized across three dimensions: Intent Recognition, Granular Content Validation, and Output Structure Validation. Through rigorous evaluation across LLMs, Meeseeks provides valuable insights into LLMs' instruction-following capabilities in multi-turn scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08265v1">LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 Accepted by ICML 2025
    </div>
    <details class="paper-abstract">
      The use of large language models (LLMs) as feature enhancers to optimize node representations, which are then used as inputs for graph neural networks (GNNs), has shown significant potential in graph representation learning. However, the fundamental properties of this approach remain underexplored. To address this issue, we propose conducting a more in-depth analysis of this issue based on the interchange intervention method. First, we construct a synthetic graph dataset with controllable causal relationships, enabling precise manipulation of semantic relationships and causal modeling to provide data for analysis. Using this dataset, we conduct interchange interventions to examine the deeper properties of LLM enhancers and GNNs, uncovering their underlying logic and internal mechanisms. Building on the analytical results, we design a plug-and-play optimization module to improve the information transfer between LLM enhancers and GNNs. Experiments across multiple datasets and models validate the proposed module.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08263v1">LLM-Based Detection of Tangled Code Changes for Higher-Quality Method-Level Bug Datasets</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Tangled code changes-commits that conflate unrelated modifications such as bug fixes, refactorings, and enhancements-introduce significant noise into bug datasets and adversely affect the performance of bug prediction models. Addressing this issue at a fine-grained, method-level granularity remains underexplored. This is critical to address, as recent bug prediction models, driven by practitioner demand, are increasingly focusing on finer granularity rather than traditional class- or file-level predictions. This study investigates the utility of Large Language Models (LLMs) for detecting tangled code changes by leveraging both commit messages and method-level code diffs. We formulate the problem as a binary classification task and evaluate multiple prompting strategies, including zero-shot, few-shot, and chain-of-thought prompting, using state-of-the-art proprietary LLMs such as GPT-4o and Gemini-2.0-Flash. Our results demonstrate that combining commit messages with code diffs significantly enhances model performance, with the combined few-shot and chain-of-thought prompting achieving an F1-score of 0.88. Additionally, we explore embedding-based machine learning models trained on LLM-generated embeddings, where a multi-layer perceptron classifier achieves superior performance (F1-score: 0.906, MCC: 0.807). These findings are encouraging for the research community, as method-level bug prediction remains an open research problem, largely due to the lack of noise-free bug datasets. This research not only contributes a novel method-level perspective to the untangling problem but also highlights practical avenues for enhancing automated software quality assessment tools.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.13517v2">CURIE: Evaluating LLMs On Multitask Scientific Long Context Understanding and Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 Accepted at ICLR 2025 main conference
    </div>
    <details class="paper-abstract">
      Scientific problem-solving involves synthesizing information while applying expert knowledge. We introduce CURIE, a scientific long-Context Understanding,Reasoning and Information Extraction benchmark to measure the potential of Large Language Models (LLMs) in scientific problem-solving and assisting scientists in realistic workflows. This benchmark introduces ten challenging tasks with a total of 580 problems and solution pairs curated by experts in six disciplines - materials science, condensed matter physics, quantum computing, geospatial analysis, biodiversity, and proteins - covering both experimental and theoretical work-flows in science. We evaluate a range of closed and open LLMs on tasks in CURIE which requires domain expertise, comprehension of long in-context information,and multi-step reasoning. While Gemini Flash 2.0 and Claude-3 show consistent high comprehension across domains, the popular GPT-4o and command-R+ fail dramatically on protein sequencing tasks. With the best performance at 32% there is much room for improvement for all models. We hope that insights gained from CURIE can guide the future development of LLMs in sciences. Evaluation code and data are in https://github.com/google/curie
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08253v1">Evaluating LLM Metrics Through Real-World Capabilities</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 14 pages main text, 5 pages references, 20 pages appendix; includes 3 figures and 4 tables
    </div>
    <details class="paper-abstract">
      As generative AI becomes increasingly embedded in everyday workflows, it is important to evaluate its performance in ways that reflect real-world usage rather than abstract notions of intelligence. Unlike many existing benchmarks that assess general intelligence, our approach focuses on real-world utility, evaluating how well models support users in everyday tasks. While current benchmarks emphasize code generation or factual recall, users rely on AI for a much broader range of activities-from writing assistance and summarization to citation formatting and stylistic feedback. In this paper, we analyze large-scale survey data and usage logs to identify six core capabilities that represent how people commonly use Large Language Models (LLMs): Summarization, Technical Assistance, Reviewing Work, Data Structuring, Generation, and Information Retrieval. We then assess the extent to which existing benchmarks cover these capabilities, revealing significant gaps in coverage, efficiency measurement, and interpretability. Drawing on this analysis, we use human-centered criteria to identify gaps in how well current benchmarks reflect common usage that is grounded in five practical criteria: coherence, accuracy, clarity, relevance, and efficiency. For four of the six capabilities, we identify the benchmarks that best align with real-world tasks and use them to compare leading models. We find that Google Gemini outperforms other models-including OpenAI's GPT, xAI's Grok, Meta's LLaMA, Anthropic's Claude, DeepSeek, and Qwen from Alibaba-on these utility-focused metrics.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04806v2">Red Teaming the Mind of the Machine: A Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 7 Pages, 6 Figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly integrated into consumer and enterprise applications. Despite their capabilities, they remain susceptible to adversarial attacks such as prompt injection and jailbreaks that override alignment safeguards. This paper provides a systematic investigation of jailbreak strategies against various state-of-the-art LLMs. We categorize over 1,400 adversarial prompts, analyze their success against GPT-4, Claude 2, Mistral 7B, and Vicuna, and examine their generalizability and construction logic. We further propose layered mitigation strategies and recommend a hybrid red-teaming and sandboxing approach for robust LLM security.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.04830v3">Cite Before You Speak: Enhancing Context-Response Grounding in E-commerce Conversational LLM-Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      With the advancement of conversational large language models (LLMs), several LLM-based Conversational Shopping Agents (CSA) have been developed to help customers smooth their online shopping. The primary objective in building an engaging and trustworthy CSA is to ensure the agent's responses about product factoids are accurate and factually grounded. However, two challenges remain. First, LLMs produce hallucinated or unsupported claims. Such inaccuracies risk spreading misinformation and diminishing customer trust. Second, without providing knowledge source attribution in CSA response, customers struggle to verify LLM-generated information. To address both challenges, we present an easily productionized solution that enables a ''citation experience'' to our customers. We build auto-evaluation metrics to holistically evaluate LLM's grounding and attribution capabilities, suggesting that citation generation paradigm substantially improves grounding performance by 13.83%. To deploy this capability at scale, we introduce Multi-UX-Inference system, which appends source citations to LLM outputs while preserving existing user experience features and supporting scalable inference. Large-scale online A/B tests show that grounded CSA responses improves customer engagement by 3% - 10%, depending on UX variations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2408.06787v3">Bridging LLMs and KGs without Fine-Tuning: Intermediate Probing Meets Subgraph-Aware Entity Descriptions</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Traditional knowledge graph completion (KGC) methods rely solely on structural information, struggling with the inherent sparsity of knowledge graphs (KGs). Large Language Models (LLMs) learn extensive knowledge from large corpora with powerful context modeling, making them promising for mitigating the limitations of previous methods. Directly fine-tuning LLMs offers great capability but comes at the cost of huge time and memory consumption, while utilizing frozen LLMs yields suboptimal results.In this work, we aim to leverage LLMs for KGC effectively and efficiently. We capture the context-aware hidden states of knowledge triples by employing prompts to stimulate the intermediate layers of LLMs. We then train a data-efficient classifier on these hidden states to harness the inherent capabilities of frozen LLMs in KGC. Additionally, to reduce ambiguity and enrich knowledge representation, we generate detailed entity descriptions through subgraph sampling on KGs. Extensive experiments on standard benchmarks demonstrate the efficiency and effectiveness of our approach. We outperform traditional KGC methods across most datasets and, notably, achieve classification performance comparable to fine-tuned LLMs while enhancing GPU memory efficiency by $188\times$ and accelerating training and inference by $13.48\times$.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08200v1">A Head to Predict and a Head to Question: Pre-trained Uncertainty Quantification Heads for Hallucination Detection in LLM Outputs</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have the tendency to hallucinate, i.e., to sporadically generate false or fabricated information. This presents a major challenge, as hallucinations often appear highly convincing and users generally lack the tools to detect them. Uncertainty quantification (UQ) provides a framework for assessing the reliability of model outputs, aiding in the identification of potential hallucinations. In this work, we introduce pre-trained UQ heads: supervised auxiliary modules for LLMs that substantially enhance their ability to capture uncertainty compared to unsupervised UQ methods. Their strong performance stems from the powerful Transformer architecture in their design and informative features derived from LLM attention maps. Experimental evaluation shows that these heads are highly robust and achieve state-of-the-art performance in claim-level hallucination detection across both in-domain and out-of-domain prompts. Moreover, these modules demonstrate strong generalization to languages they were not explicitly trained on. We pre-train a collection of UQ heads for popular LLM series, including Mistral, Llama, and Gemma 2. We publicly release both the code and the pre-trained heads.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.00240v2">LLM-Based Threat Detection and Prevention Framework for IoT Ecosystems</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 Preprint version; submitted for academic peer review
    </div>
    <details class="paper-abstract">
      The increasing complexity and scale of the Internet of Things (IoT) have made security a critical concern. This paper presents a novel Large Language Model (LLM)-based framework for comprehensive threat detection and prevention in IoT environments. The system integrates lightweight LLMs fine-tuned on IoT-specific datasets (IoT-23, TON_IoT) for real-time anomaly detection and automated, context-aware mitigation strategies optimized for resource-constrained devices. A modular Docker-based deployment enables scalable and reproducible evaluation across diverse network conditions. Experimental results in simulated IoT environments demonstrate significant improvements in detection accuracy, response latency, and resource efficiency over traditional security methods. The proposed framework highlights the potential of LLM-driven, autonomous security solutions for future IoT ecosystems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.16032v2">LLMs meet Federated Learning for Scalable and Secure IoT Management</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 This work has been submitted to the IEEE Global Communications Conference (GLOBECOM) 2025 for possible publication
    </div>
    <details class="paper-abstract">
      The rapid expansion of IoT ecosystems introduces severe challenges in scalability, security, and real-time decision-making. Traditional centralized architectures struggle with latency, privacy concerns, and excessive resource consumption, making them unsuitable for modern large-scale IoT deployments. This paper presents a novel Federated Learning-driven Large Language Model (FL-LLM) framework, designed to enhance IoT system intelligence while ensuring data privacy and computational efficiency. The framework integrates Generative IoT (GIoT) models with a Gradient Sensing Federated Strategy (GSFS), dynamically optimizing model updates based on real-time network conditions. By leveraging a hybrid edge-cloud processing architecture, our approach balances intelligence, scalability, and security in distributed IoT environments. Evaluations on the IoT-23 dataset demonstrate that our framework improves model accuracy, reduces response latency, and enhances energy efficiency, outperforming traditional FL techniques (i.e., FedAvg, FedOpt). These findings highlight the potential of integrating LLM-powered federated learning into large-scale IoT ecosystems, paving the way for more secure, scalable, and adaptive IoT management solutions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05712v3">LLM-Text Watermarking based on Lagrange Interpolation</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      The rapid advancement of LLMs (Large Language Models) has established them as a foundational technology for many AI and ML-powered human computer interactions. A critical challenge in this context is the attribution of LLM-generated text -- either to the specific language model that produced it or to the individual user who embedded their identity via a so-called multi-bit watermark. This capability is essential for combating misinformation, fake news, misinterpretation, and plagiarism. One of the key techniques for addressing this challenge is digital watermarking. This work presents a watermarking scheme for LLM-generated text based on Lagrange interpolation, enabling the recovery of a multi-bit author identity even when the text has been heavily redacted by an adversary. The core idea is to embed a continuous sequence of points $(x, f(x))$ that lie on a single straight line. The $x$-coordinates are computed pseudorandomly using a cryptographic hash function $H$ applied to the concatenation of the previous token's identity and a secret key $s_k$. Crucially, the $x$-coordinates do not need to be embedded into the text -- only the corresponding $f(x)$ values are embedded. During extraction, the algorithm recovers the original points along with many spurious ones, forming an instance of the Maximum Collinear Points (MCP) problem, which can be solved efficiently. Experimental results demonstrate that the proposed method is highly effective, allowing the recovery of the author identity even when as few as three genuine points remain after adversarial manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08143v1">Communication Styles and Reader Preferences of LLM and Human Experts in Explaining Health Information</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      With the wide adoption of large language models (LLMs) in information assistance, it is essential to examine their alignment with human communication styles and values. We situate this study within the context of fact-checking health information, given the critical challenge of rectifying conceptions and building trust. Recent studies have explored the potential of LLM for health communication, but style differences between LLMs and human experts and associated reader perceptions remain under-explored. In this light, our study evaluates the communication styles of LLMs, focusing on how their explanations differ from those of humans in three core components of health communication: information, sender, and receiver. We compiled a dataset of 1498 health misinformation explanations from authoritative fact-checking organizations and generated LLM responses to inaccurate health information. Drawing from health communication theory, we evaluate communication styles across three key dimensions of information linguistic features, sender persuasive strategies, and receiver value alignments. We further assessed human perceptions through a blinded evaluation with 99 participants. Our findings reveal that LLM-generated articles showed significantly lower scores in persuasive strategies, certainty expressions, and alignment with social values and moral foundations. However, human evaluation demonstrated a strong preference for LLM content, with over 60% responses favoring LLM articles for clarity, completeness, and persuasiveness. Our results suggest that LLMs' structured approach to presenting information may be more effective at engaging readers despite scoring lower on traditional measures of quality in fact-checking and health communication.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08140v1">Lost in Transmission: When and Why LLMs Fail to Reason Globally</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 28 pages
    </div>
    <details class="paper-abstract">
      Despite their many successes, transformer-based large language models (LLMs) continue to struggle with tasks that require complex reasoning over large parts of their input. We argue that these failures arise due to capacity limits on the accurate flow of information within LLMs. To formalize this issue, we introduce the bounded attention prefix oracle (BAPO) model, a new computational framework that models bandwidth constraints on attention heads, the mechanism for internal communication in LLMs. We show that several important reasoning problems like graph reachability require high communication bandwidth for BAPOs to solve; we call these problems BAPO-hard. Our experiments corroborate our theoretical predictions: GPT-4, Claude, and Gemini succeed on BAPO-easy tasks and fail even on relatively small BAPO-hard tasks. BAPOs also reveal another benefit of chain of thought (CoT): we prove that breaking down a task using CoT can turn any BAPO-hard problem into a BAPO-easy one. Our results offer principled explanations for key LLM failures and suggest directions for architectures and inference methods that mitigate bandwidth limits.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09031v1">Improving the Reliability of LLMs: Combining CoT, RAG, Self-Consistency, and Self-Verification</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Hallucination, where large language models (LLMs) generate confident but incorrect or irrelevant information, remains a key limitation in their application to complex, open-ended tasks. Chain-of-thought (CoT) prompting has emerged as a promising method for improving multistep reasoning by guiding models through intermediate steps. However, CoT alone does not fully address the hallucination problem. In this work, we investigate how combining CoT with retrieval-augmented generation (RAG), as well as applying self-consistency and self-verification strategies, can reduce hallucinations and improve factual accuracy. By incorporating external knowledge sources during reasoning and enabling models to verify or revise their own outputs, we aim to generate more accurate and coherent responses. We present a comparative evaluation of baseline LLMs against CoT, CoT+RAG, self-consistency, and self-verification techniques. Our results highlight the effectiveness of each method and identify the most robust approach for minimizing hallucinations while preserving fluency and reasoning depth.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09027v1">Tests as Prompt: A Test-Driven-Development Benchmark for LLM Code Generation</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 arXiv admin note: text overlap with arXiv:2409.05177
    </div>
    <details class="paper-abstract">
      We introduce WebApp1K, a novel benchmark for evaluating large language models (LLMs) in test-driven development (TDD) tasks, where test cases serve as both prompt and verification for code generation. Unlike traditional approaches relying on natural language prompts, our benchmark emphasizes the ability of LLMs to interpret and implement functionality directly from test cases, reflecting real-world software development practices. Comprising 1000 diverse challenges across 20 application domains, the benchmark evaluates LLMs on their ability to generate compact, functional code under the constraints of context length and multi-feature complexity. Our findings highlight instruction following and in-context learning as critical capabilities for TDD success, surpassing the importance of general coding proficiency or pretraining knowledge. Through comprehensive evaluation of 19 frontier models, we reveal performance bottlenecks, such as instruction loss in long prompts, and provide a detailed error analysis spanning multiple root causes. This work underscores the practical value of TDD-specific benchmarks and lays the foundation for advancing LLM capabilities in rigorous, application-driven coding scenarios.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04260v2">Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) improve in their capacity to serve as personal AI assistants, their ability to output uniquely tailored, personalized responses that align with the soft preferences of their users is essential for enhancing user satisfaction and retention. However, untrained lay users have poor prompt specification abilities and often struggle with conveying their latent preferences to AI assistants. To address this, we leverage activation steering to guide LLMs to align with interpretable preference dimensions during inference. In contrast to memory-based personalization methods that require longer user history, steering is extremely lightweight and can be easily controlled by the user via an linear strength factor. We embed steering into three different interactive chatbot interfaces and conduct a within-subjects user study (n=14) to investigate how end users prefer to personalize their conversations. The results demonstrate the effectiveness of preference-based steering for aligning real-world conversations with hidden user preferences, and highlight further insights on how diverse values around control, usability, and transparency lead users to prefer different interfaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08902v1">Performance Gains of LLMs With Humans in a World of LLMs Versus Humans</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Currently, a considerable research effort is devoted to comparing LLMs to a group of human experts, where the term "expert" is often ill-defined or variable, at best, in a state of constantly updating LLM releases. Without proper safeguards in place, LLMs will threaten to cause harm to the established structure of safe delivery of patient care which has been carefully developed throughout history to keep the safety of the patient at the forefront. A key driver of LLM innovation is founded on community research efforts which, if continuing to operate under "humans versus LLMs" principles, will expedite this trend. Therefore, research efforts moving forward must focus on effectively characterizing the safe use of LLMs in clinical settings that persist across the rapid development of novel LLM models. In this communication, we demonstrate that rather than comparing LLMs to humans, there is a need to develop strategies enabling efficient work of humans with LLMs in an almost symbiotic manner.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.15823v4">InductionBench: LLMs Fail in the Simplest Complexity Class</a></div>
    <div class="paper-meta">
      📅 2025-05-13
      | 💬 25 pages, 10 figures, more details including examples and prompts are added
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have shown remarkable improvements in reasoning and many existing benchmarks have been addressed by models such as o1 and o3 either fully or partially. However, a majority of these benchmarks emphasize deductive reasoning, including mathematical and coding tasks in which rules such as mathematical axioms or programming syntax are clearly defined, based on which LLMs can plan and apply these rules to arrive at a solution. In contrast, inductive reasoning, where one infers the underlying rules from observed data, remains less explored. Such inductive processes lie at the heart of scientific discovery, as they enable researchers to extract general principles from empirical observations. To assess whether LLMs possess this capacity, we introduce InductionBench, a new benchmark designed to evaluate the inductive reasoning ability of LLMs. Our experimental findings reveal that even the most advanced models available struggle to master the simplest complexity classes within the subregular hierarchy of functions, highlighting a notable deficiency in current LLMs' inductive reasoning capabilities. Coda and data are available https://github.com/Wenyueh/inductive_reasoning_benchmark.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.09651v1">Unlocking Location Intelligence: A Survey from Deep Learning to The LLM Era</a></div>
    <div class="paper-meta">
      📅 2025-05-13
    </div>
    <details class="paper-abstract">
      Location Intelligence (LI), the science of transforming location-centric geospatial data into actionable knowledge, has become a cornerstone of modern spatial decision-making. The rapid evolution of Geospatial Representation Learning is fundamentally reshaping LI development through two successive technological revolutions: the deep learning breakthrough and the emerging large language model (LLM) paradigm. While deep neural networks (DNNs) have demonstrated remarkable success in automated feature extraction from structured geospatial data (e.g., satellite imagery, GPS trajectories), the recent integration of LLMs introduces transformative capabilities for cross-modal geospatial reasoning and unstructured geo-textual data processing. This survey presents a comprehensive review of geospatial representation learning across both technological eras, organizing them into a structured taxonomy based on the complete pipeline comprising: (1) data perspective, (2) methodological perspective and (3) application perspective. We also highlight current advancements, discuss existing limitations, and propose potential future research directions in the LLM era. This work offers a thorough exploration of the field and providing a roadmap for further innovation in LI. The summary of the up-to-date paper list can be found in https://github.com/CityMind-Lab/Awesome-Location-Intelligence and will undergo continuous updates.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07184v1">Structural Entropy Guided Agent for Detecting and Repairing Knowledge Deficiencies in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have achieved unprecedented performance by leveraging vast pretraining corpora, yet their performance remains suboptimal in knowledge-intensive domains such as medicine and scientific research, where high factual precision is required. While synthetic data provides a promising avenue for augmenting domain knowledge, existing methods frequently generate redundant samples that do not align with the model's true knowledge gaps. To overcome this limitation, we propose a novel Structural Entropy-guided Knowledge Navigator (SENATOR) framework that addresses the intrinsic knowledge deficiencies of LLMs. Our approach employs the Structure Entropy (SE) metric to quantify uncertainty along knowledge graph paths and leverages Monte Carlo Tree Search (MCTS) to selectively explore regions where the model lacks domain-specific knowledge. Guided by these insights, the framework generates targeted synthetic data for supervised fine-tuning, enabling continuous self-improvement. Experimental results on LLaMA-3 and Qwen2 across multiple domain-specific benchmarks show that SENATOR effectively detects and repairs knowledge deficiencies, achieving notable performance improvements. The code and data for our methods and experiments are available at https://github.com/weiyifan1023/senator.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.01930v2">Efficient LLM Context Distillation</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) demonstrate proficiency across diverse tasks but often require targeted adaptations for specific applications. Various methods have been proposed to facilitate this adaptation, including fewshot fine-tuning, in-context learning, and context distillation. This paper specifically investigates context distillation a method that extends the utility of task-specific examples by internalizing them, thus augmenting the example set accessible for model inference. We conduct a comparative analysis of context distillation with in-context learning (ICL) and few-shot fine-tuning (FT), aiming to ascertain the efficacy of context distillation in adapting models using minimal in-context examples. Employing matched datasets from Mobach, our experiments leverage OPT models of various sizes. The results indicate that context distillation effectively adapts models, with student models attaining comparable in-domain and out-of-domain accuracies to in-context learning. Although context distillation surpasses ICL in out-of-domain generalization, it does not achieve the performance levels of FT. However, the reduced dataset size and computational demands position context distillation as a viable alternative, especially for smaller datasets. Overall, this study presents context distillation as an efficient and potent method for customizing LLMs to specific tasks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.16732v4">Perfectly to a Tee: Understanding User Perceptions of Personalized LLM-Enhanced Narrative Interventions</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Stories about overcoming personal struggles can effectively illustrate the application of psychological theories in real life, yet they may fail to resonate with individuals' experiences. In this work, we employ large language models (LLMs) to create tailored narratives that acknowledge and address unique challenging thoughts and situations faced by individuals. Our study, involving 346 young adults across two settings, demonstrates that personalized LLM-enhanced stories were perceived to be better than human-written ones in conveying key takeaways, promoting reflection, and reducing belief in negative thoughts. These stories were not only seen as more relatable but also similarly authentic to human-written ones, highlighting the potential of LLMs in helping young adults manage their struggles. The findings of this work provide crucial design considerations for future narrative-based digital mental health interventions, such as the need to maintain relatability without veering into implausibility and refining the wording and tone of AI-enhanced content.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08106v1">Are LLMs complicated ethical dilemma analyzers?</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 CS194-280 Advanced LLM Agents project. Project page: https://github.com/ALT-JS/ethicaLLM
    </div>
    <details class="paper-abstract">
      One open question in the study of Large Language Models (LLMs) is whether they can emulate human ethical reasoning and act as believable proxies for human judgment. To investigate this, we introduce a benchmark dataset comprising 196 real-world ethical dilemmas and expert opinions, each segmented into five structured components: Introduction, Key Factors, Historical Theoretical Perspectives, Resolution Strategies, and Key Takeaways. We also collect non-expert human responses for comparison, limited to the Key Factors section due to their brevity. We evaluate multiple frontier LLMs (GPT-4o-mini, Claude-3.5-Sonnet, Deepseek-V3, Gemini-1.5-Flash) using a composite metric framework based on BLEU, Damerau-Levenshtein distance, TF-IDF cosine similarity, and Universal Sentence Encoder similarity. Metric weights are computed through an inversion-based ranking alignment and pairwise AHP analysis, enabling fine-grained comparison of model outputs to expert responses. Our results show that LLMs generally outperform non-expert humans in lexical and structural alignment, with GPT-4o-mini performing most consistently across all sections. However, all models struggle with historical grounding and proposing nuanced resolution strategies, which require contextual abstraction. Human responses, while less structured, occasionally achieve comparable semantic similarity, suggesting intuitive moral reasoning. These findings highlight both the strengths and current limitations of LLMs in ethical decision-making.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08083v1">LLMs to Support K-12 Teachers in Culturally Relevant Pedagogy: An AI Literacy Example</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Culturally Relevant Pedagogy (CRP) is vital in K-12 education, yet teachers struggle to implement CRP into practice due to time, training, and resource gaps. This study explores how Large Language Models (LLMs) can address these barriers by introducing CulturAIEd, an LLM tool that assists teachers in adapting AI literacy curricula to students' cultural contexts. Through an exploratory pilot with four K-12 teachers, we examined CulturAIEd's impact on CRP integration. Results showed CulturAIEd enhanced teachers' confidence in identifying opportunities for cultural responsiveness in learning activities and making culturally responsive modifications to existing activities. They valued CulturAIEd's streamlined integration of student demographic information, immediate actionable feedback, which could result in high implementation efficiency. This exploration of teacher-AI collaboration highlights how LLM can help teachers include CRP components into their instructional practices efficiently, especially in global priorities for future-ready education, such as AI literacy.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08063v1">Who's the Leader? Analyzing Novice Workflows in LLM-Assisted Debugging of Machine Learning Code</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 Tools for Thought Workshop at CHI 2025
    </div>
    <details class="paper-abstract">
      While LLMs are often touted as tools for democratizing specialized knowledge to beginners, their actual effectiveness for improving task performance and learning is still an open question. It is known that novices engage with LLMs differently from experts, with prior studies reporting meta-cognitive pitfalls that affect novices' ability to verify outputs and prompt effectively. We focus on a task domain, machine learning (ML), which embodies both high complexity and low verifiability to understand the impact of LLM assistance on novices. Provided a buggy ML script and open access to ChatGPT, we conduct a formative study with eight novice ML engineers to understand their reliance on, interactions with, and perceptions of the LLM. We find that user actions can be roughly categorized into leading the LLM and led-by the LLM, and further investigate how they affect reliance outcomes like over- and under-reliance. These results have implications on novices' cognitive engagement in LLM-assisted tasks and potential negative effects on downstream learning. Lastly, we pose potential augmentations to the novice-LLM interaction paradigm to promote cognitive engagement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.08054v1">FalseReject: A Resource for Improving Contextual Safety and Mitigating Over-Refusals in LLMs via Structured Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Safety alignment approaches in large language models (LLMs) often lead to the over-refusal of benign queries, significantly diminishing their utility in sensitive scenarios. To address this challenge, we introduce FalseReject, a comprehensive resource containing 16k seemingly toxic queries accompanied by structured responses across 44 safety-related categories. We propose a graph-informed adversarial multi-agent interaction framework to generate diverse and complex prompts, while structuring responses with explicit reasoning to aid models in accurately distinguishing safe from unsafe contexts. FalseReject includes training datasets tailored for both standard instruction-tuned models and reasoning-oriented models, as well as a human-annotated benchmark test set. Our extensive benchmarking on 29 state-of-the-art (SOTA) LLMs reveals persistent over-refusal challenges. Empirical results demonstrate that supervised finetuning with FalseReject substantially reduces unnecessary refusals without compromising overall safety or general language capabilities.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.04168v2">From Calculation to Adjudication: Examining LLM judges on Mathematical Reasoning Tasks</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      To reduce the need for human annotations, large language models (LLMs) have been proposed as judges of the quality of other candidate models. The performance of LLM judges is typically evaluated by measuring the correlation with human judgments on generative tasks such as summarization or machine translation. In contrast, we study LLM judges on mathematical reasoning tasks. These tasks require multi-step reasoning, and the correctness of their solutions is verifiable, enabling a more objective evaluation. We perform a detailed performance analysis and find that easy samples are easy to judge, and difficult samples are difficult to judge. Our analysis uncovers a strong correlation between judgment performance and the candidate model task performance, indicating that judges tend to favor higher-quality models even if their answer is incorrect. As a consequence, we test whether we can predict the behavior of LLM judges using simple features such as part-of-speech tags and find that we can correctly predict 70%-75% of judgments. We conclude this study by analyzing practical use cases, showing that LLM judges consistently detect the on-average better model but largely fail if we use them to improve task performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.01052v2">ALinFiK: Learning to Approximate Linearized Future Influence Kernel for Scalable Third-Party LLM Data Valuation</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 Proceedings of the NAACL 2025. Keywords: Influence Function, Data Valuation, Influence Estimation. https://aclanthology.org/2025.naacl-long.589/
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) heavily rely on high-quality training data, making data valuation crucial for optimizing model performance, especially when working within a limited budget. In this work, we aim to offer a third-party data valuation approach that benefits both data providers and model developers. We introduce a linearized future influence kernel (LinFiK), which assesses the value of individual data samples in improving LLM performance during training. We further propose ALinFiK, a learning strategy to approximate LinFiK, enabling scalable data valuation. Our comprehensive evaluations demonstrate that this approach surpasses existing baselines in effectiveness and efficiency, demonstrating significant scalability advantages as LLM parameters increase.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.04021v2">Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Serving large language models (LLMs) is expensive, especially for providers hosting many models, making cost reduction essential. The unique workload patterns of serving multiple LLMs (i.e., multi-LLM serving) create new opportunities and challenges for this task. The long-tail popularity of models and their long idle periods present opportunities to improve utilization through GPU sharing. However, existing GPU sharing systems lack the ability to adjust their resource allocation and sharing policies at runtime, making them ineffective at meeting latency service-level objectives (SLOs) under rapidly fluctuating workloads. This paper presents Prism, a multi-LLM serving system that unleashes the full potential of GPU sharing to achieve both cost efficiency and SLO attainment. At its core, Prism tackles a key limitation of existing systems$\unicode{x2014}$the lack of $\textit{cross-model memory coordination}$, which is essential for flexibly sharing GPU memory across models under dynamic workloads. Prism achieves this with two key designs. First, it supports on-demand memory allocation by dynamically mapping physical to virtual memory pages, allowing flexible memory redistribution among models that space- and time-share a GPU. Second, it improves memory efficiency through a two-level scheduling policy that dynamically adjusts sharing strategies based on models' runtime demands. Evaluations on real-world traces show that Prism achieves more than $2\times$ cost savings and $3.3\times$ SLO attainment compared to state-of-the-art systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.07128v2">DeepSeek-R1 Thoughtology: Let's think about LLM Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 142 pages, pre-print
    </div>
    <details class="paper-abstract">
      Large Reasoning Models like DeepSeek-R1 mark a fundamental shift in how LLMs approach complex problems. Instead of directly producing an answer for a given input, DeepSeek-R1 creates detailed multi-step reasoning chains, seemingly "thinking" about a problem before providing an answer. This reasoning process is publicly available to the user, creating endless opportunities for studying the reasoning behaviour of the model and opening up the field of Thoughtology. Starting from a taxonomy of DeepSeek-R1's basic building blocks of reasoning, our analyses on DeepSeek-R1 investigate the impact and controllability of thought length, management of long or confusing contexts, cultural and safety concerns, and the status of DeepSeek-R1 vis-\`a-vis cognitive phenomena, such as human-like language processing and world modelling. Our findings paint a nuanced picture. Notably, we show DeepSeek-R1 has a 'sweet spot' of reasoning, where extra inference time can impair model performance. Furthermore, we find a tendency for DeepSeek-R1 to persistently ruminate on previously explored problem formulations, obstructing further exploration. We also note strong safety vulnerabilities of DeepSeek-R1 compared to its non-reasoning counterpart, which can also compromise safety-aligned LLMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07897v1">LongCodeBench: Evaluating Coding LLMs at 1M Context Windows</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Context lengths for models have grown rapidly, from thousands to millions of tokens in just a few years. The extreme context sizes of modern long-context models have made it difficult to construct realistic long-context benchmarks -- not only due to the cost of collecting million-context tasks but also in identifying realistic scenarios that require significant contexts. We identify code comprehension and repair as a natural testbed and challenge task for long-context models and introduce LongCodeBench (LCB), a benchmark to test LLM coding abilities in long-context scenarios. Our benchmark tests both the comprehension and repair capabilities of LCLMs in realistic and important settings by drawing from real-world GitHub issues and constructing QA (LongCodeQA) and bug fixing (LongSWE-Bench) tasks. We carefully stratify the complexity of our benchmark, enabling us to evaluate models across different scales -- ranging from Qwen2.5 14B Instruct to Google's flagship Gemini model. We find that long-context remains a weakness for all models, with performance drops such as from 29% to 3% for Claude 3.5 Sonnet, or from 70.2% to 40% for Qwen2.5.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07793v1">Overflow Prevention Enhances Long-Context Recurrent LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      A recent trend in LLMs is developing recurrent sub-quadratic models that improve long-context processing efficiency. We investigate leading large long-context models, focusing on how their fixed-size recurrent memory affects their performance. Our experiments reveal that, even when these models are trained for extended contexts, their use of long contexts remains underutilized. Specifically, we demonstrate that a chunk-based inference procedure, which identifies and processes only the most relevant portion of the input can mitigate recurrent memory failures and be effective for many long-context tasks: On LongBench, our method improves the overall performance of Falcon3-Mamba-Inst-7B by 14%, Falcon-Mamba-Inst-7B by 28%, RecurrentGemma-IT-9B by 50%, and RWKV6-Finch-7B by 51%. Surprisingly, this simple approach also leads to state-of-the-art results in the challenging LongBench v2 benchmark, showing competitive performance with equivalent size Transformers. Furthermore, our findings raise questions about whether recurrent models genuinely exploit long-range dependencies, as our single-chunk strategy delivers stronger performance - even in tasks that presumably require cross-context relations.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07784v1">Domain Regeneration: How well do LLMs match syntactic properties of text domains?</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Recent improvement in large language model performance have, in all likelihood, been accompanied by improvement in how well they can approximate the distribution of their training data. In this work, we explore the following question: which properties of text domains do LLMs faithfully approximate, and how well do they do so? Applying observational approaches familiar from corpus linguistics, we prompt a commonly used, opensource LLM to regenerate text from two domains of permissively licensed English text which are often contained in LLM training data -- Wikipedia and news text. This regeneration paradigm allows us to investigate whether LLMs can faithfully match the original human text domains in a fairly semantically-controlled setting. We investigate varying levels of syntactic abstraction, from more simple properties like sentence length, and article readability, to more complex and higher order properties such as dependency tag distribution, parse depth, and parse complexity. We find that the majority of the regenerated distributions show a shifted mean, a lower standard deviation, and a reduction of the long tail, as compared to the human originals.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07782v1">MLE-Dojo: Interactive Environments for Empowering LLM Agents in Machine Learning Engineering</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      We introduce MLE-Dojo, a Gym-style framework for systematically reinforcement learning, evaluating, and improving autonomous large language model (LLM) agents in iterative machine learning engineering (MLE) workflows. Unlike existing benchmarks that primarily rely on static datasets or single-attempt evaluations, MLE-Dojo provides an interactive environment enabling agents to iteratively experiment, debug, and refine solutions through structured feedback loops. Built upon 200+ real-world Kaggle challenges, MLE-Dojo covers diverse, open-ended MLE tasks carefully curated to reflect realistic engineering scenarios such as data processing, architecture search, hyperparameter tuning, and code debugging. Its fully executable environment supports comprehensive agent training via both supervised fine-tuning and reinforcement learning, facilitating iterative experimentation, realistic data sampling, and real-time outcome verification. Extensive evaluations of eight frontier LLMs reveal that while current models achieve meaningful iterative improvements, they still exhibit significant limitations in autonomously generating long-horizon solutions and efficiently resolving complex errors. Furthermore, MLE-Dojo's flexible and extensible architecture seamlessly integrates diverse data sources, tools, and evaluation protocols, uniquely enabling model-based agent tuning and promoting interoperability, scalability, and reproducibility. We open-source our framework and benchmarks to foster community-driven innovation towards next-generation MLE agents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07610v1">Concept-Level Explainability for Auditing & Steering LLM Responses</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 9 pages, 7 figures, Submission to Neurips 2025
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) become widely deployed, concerns about their safety and alignment grow. An approach to steer LLM behavior, such as mitigating biases or defending against jailbreaks, is to identify which parts of a prompt influence specific aspects of the model's output. Token-level attribution methods offer a promising solution, but still struggle in text generation, explaining the presence of each token in the output separately, rather than the underlying semantics of the entire LLM response. We introduce ConceptX, a model-agnostic, concept-level explainability method that identifies the concepts, i.e., semantically rich tokens in the prompt, and assigns them importance based on the outputs' semantic similarity. Unlike current token-level methods, ConceptX also offers to preserve context integrity through in-place token replacements and supports flexible explanation goals, e.g., gender bias. ConceptX enables both auditing, by uncovering sources of bias, and steering, by modifying prompts to shift the sentiment or reduce the harmfulness of LLM responses, without requiring retraining. Across three LLMs, ConceptX outperforms token-level methods like TokenSHAP in both faithfulness and human alignment. Steering tasks boost sentiment shift by 0.252 versus 0.131 for random edits and lower attack success rates from 0.463 to 0.242, outperforming attribution and paraphrasing baselines. While prompt engineering and self-explaining methods sometimes yield safer responses, ConceptX offers a transparent and faithful alternative for improving LLM safety and alignment, demonstrating the practical value of attribution-based explainability in guiding LLM behavior.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.19159v2">A Sliding Layer Merging Method for Efficient Depth-Wise Pruning in LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Compared to width-wise pruning, depth-wise pruning can significantly accelerate inference in resource-constrained scenarios. However, treating the entire Transformer layer as the minimum pruning unit may degrade model performance by indiscriminately discarding the entire information of the layer. This paper reveals the ``Patch-like'' feature relationship between layers in large language models by analyzing the correlation of the outputs of different layers in the reproducing kernel Hilbert space. Building on this observation, we propose a sliding layer merging method that dynamically selects and fuses consecutive layers from top to bottom according to a pre-defined similarity threshold, thereby simplifying the model structure while maintaining its performance. Extensive experiments on LLMs with various architectures and different parameter scales show that our method outperforms existing pruning techniques in both zero-shot inference performance and retraining recovery quality after pruning. In particular, in the experiment with 35% pruning on the Vicuna-7B model, our method achieved a 1.654% improvement in average performance on zero-shot tasks compared to the existing method. Moreover, we further reveal the potential of combining depth pruning with width pruning to enhance the pruning effect. Our codes are available at https://github.com/920927/SLM-a-sliding-layer-merging-method.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07473v1">Web-Bench: A LLM Code Benchmark Based on Web Standards and Frameworks</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 28 pages, 15 figures
    </div>
    <details class="paper-abstract">
      The application of large language models (LLMs) in the field of coding is evolving rapidly: from code assistants, to autonomous coding agents, and then to generating complete projects through natural language. Early LLM code benchmarks primarily focused on code generation accuracy, but these benchmarks have gradually become saturated. Benchmark saturation weakens their guiding role for LLMs. For example, HumanEval Pass@1 has reached 99.4% and MBPP 94.2%. Among various attempts to address benchmark saturation, approaches based on software engineering have stood out, but the saturation of existing software engineering benchmarks is rapidly increasing. To address this, we propose a new benchmark, Web-Bench, which contains 50 projects, each consisting of 20 tasks with sequential dependencies. The tasks implement project features in sequence, simulating real-world human development workflows. When designing Web-Bench, we aim to cover the foundational elements of Web development: Web Standards and Web Frameworks. Given the scale and complexity of these projects, which were designed by engineers with 5 to 10 years of experience, each presents a significant challenge. On average, a single project takes 4 to 8 hours for a senior engineer to complete. On our given benchmark agent (Web-Agent), SOTA (Claude 3.7 Sonnet) achieves only 25.1% Pass@1, significantly lower (better) than SWE-Bench's Verified (65.4%) and Full (33.8%) scores. Finally, we discuss that in any development field, Standards and Frameworks represent foundational knowledge and efficiency tools, respectively, and LLMs require optimization tailored to them.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07453v1">How well do LLMs reason over tabular data, really?</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 10 pages, 4 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel in natural language tasks, but less is known about their reasoning capabilities over tabular data. Prior analyses devise evaluation strategies that poorly reflect an LLM's realistic performance on tabular queries. Moreover, we have a limited understanding of the robustness of LLMs towards realistic variations in tabular inputs. Therefore, we ask: Can general-purpose LLMs reason over tabular data, really?, and focus on two questions 1) are tabular reasoning capabilities of general-purpose LLMs robust to real-world characteristics of tabular inputs, and 2) how can we realistically evaluate an LLM's performance on analytical tabular queries? Building on a recent tabular reasoning benchmark, we first surface shortcomings of its multiple-choice prompt evaluation strategy, as well as commonly used free-form text metrics such as SacreBleu and BERT-score. We show that an LLM-as-a-judge procedure yields more reliable performance insights and unveil a significant deficit in tabular reasoning performance of LLMs. We then extend the tabular inputs reflecting three common characteristics in practice: 1) missing values, 2) duplicate entities, and 3) structural variations. Experiments show that the tabular reasoning capabilities of general-purpose LLMs suffer from these variations, stressing the importance of improving their robustness for realistic tabular inputs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07437v1">LEAD: Iterative Data Selection for Efficient LLM Instruction Tuning</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Instruction tuning has emerged as a critical paradigm for improving the capabilities and alignment of large language models (LLMs). However, existing iterative model-aware data selection methods incur significant computational overhead, as they rely on repeatedly performing full-dataset model inference to estimate sample utility for subsequent training iterations, creating a fundamental efficiency bottleneck. In this paper, we propose LEAD, an efficient iterative data selection framework that accurately estimates sample utility entirely within the standard training loop, eliminating the need for costly additional model inference. At its core, LEAD introduces Instance-Level Dynamic Uncertainty (IDU), a theoretically grounded utility function combining instantaneous training loss, gradient-based approximation of loss changes, and exponential smoothing of historical loss signals. To further scale efficiently to large datasets, LEAD employs a two-stage, coarse-to-fine selection strategy, adaptively prioritizing informative clusters through a multi-armed bandit mechanism, followed by precise fine-grained selection of high-utility samples using IDU. Extensive experiments across four diverse benchmarks show that LEAD significantly outperforms state-of-the-art methods, improving average model performance by 6.1%-10.8% while using only 2.5% of the training data and reducing overall training time by 5-10x.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.05738v2">LLM-assisted Mutation for Whitebox API Testing</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Cloud applications heavily rely on APIs to communicate with each other and exchange data. To ensure the reliability of cloud applications, cloud providers widely adopt API testing techniques. Unfortunately, existing API testing approaches are insufficient to reach strict conditions, a problem known as fitness plateaus, due to the lack of gradient provided by coverage metrics. To address this issue, we propose MioHint, a novel white-box API testing approach that leverages the code comprehension capabilities of Large Language Model (LLM) to boost API testing. The key challenge of LLM-based API testing lies in system-level testing, which emphasizes the dependencies between requests and targets across functions and files, thereby making the entire codebase the object of analysis. However, feeding the entire codebase to an LLM is impractical due to its limited context length and short memory. MioHint addresses this challenge by synergizing static analysis with LLMs. We retrieve relevant code with data-dependency analysis at the statement level, including def-use analysis for variables used in the target and function expansion for subfunctions called by the target. To evaluate the effectiveness of our method, we conducted experiments across 16 real-world REST API services. The findings reveal that MioHint achieves an average increase of 4.95% absolute in line coverage compared to the baseline, EvoMaster, alongside a remarkable factor of 67x improvement in mutation accuracy. Furthermore, our method successfully covers over 57% of hard-to-cover targets while in baseline the coverage is less than 10%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12896v3">None of the Others: a General Technique to Distinguish Reasoning from Memorization in Multiple-Choice LLM Evaluation Benchmarks</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      In LLM evaluations, reasoning is often distinguished from recall/memorization by performing numerical variations to math-oriented questions. Here we introduce a general variation method for multiple-choice questions that completely dissociates the correct answer from previously seen tokens or concepts, requiring LLMs to understand and reason (rather than memorizing) in order to answer correctly. Using this method, we evaluate state-of-the-art proprietary and open-source LLMs on two datasets available in English and Spanish: the public MMLU benchmark and the private UNED-Access 2024 dataset. Results show that all models experience remarkable accuracy drops under our proposed variation, with an average loss of 57% on MMLU and 50% on UNED-Access 2024, ranging from 10% to 93% across models. Notably, the most accurate model in our experimentation (OpenAI-o3-mini) is not the most robust (DeepSeek-R1-70B), suggesting that the best models in standard evaluations may not be the ones with better reasoning capabilities. Also, we see larger accuracy drops in public (vs private) datasets and questions posed in their original language (vs a manual translation), which are signs of contamination and also point to a relevant role of recall/memorization in current LLMs' answers.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2412.00535v6">FullStack Bench: Evaluating LLMs as Full Stack Coders</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 26 pages
    </div>
    <details class="paper-abstract">
      As the capabilities of code large language models (LLMs) continue to expand, their applications across diverse code intelligence domains are rapidly increasing. However, most existing datasets only evaluate limited application domains. To address this gap, we have developed a comprehensive code evaluation dataset FullStack Bench focusing on full-stack programming, which encompasses a wide range of application domains (e.g., basic programming, data analysis, software engineering, mathematics, and machine learning). Besides, to assess multilingual programming capabilities, in FullStack Bench, we design real-world instructions and corresponding unit test cases from 16 widely-used programming languages to reflect real-world usage scenarios rather than simple translations. Moreover, we also release an effective code sandbox execution tool (i.e., SandboxFusion) supporting various programming languages and packages to evaluate the performance of our FullStack Bench efficiently. Comprehensive experimental results on our FullStack Bench demonstrate the necessity and effectiveness of our FullStack Bench and SandboxFusion.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.07377v3">Process-Supervised LLM Recommenders via Flow-guided Tuning</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 Accepted by SIGIR 2025
    </div>
    <details class="paper-abstract">
      While large language models (LLMs) are increasingly adapted for recommendation systems via supervised fine-tuning (SFT), this approach amplifies popularity bias due to its likelihood maximization objective, compromising recommendation diversity and fairness. To address this, we present Flow-guided fine-tuning recommender (Flower), which replaces SFT with a Generative Flow Network (GFlowNet) framework that enacts process supervision through token-level reward propagation. Flower's key innovation lies in decomposing item-level rewards into constituent token rewards, enabling direct alignment between token generation probabilities and their reward signals. This mechanism achieves three critical advancements: (1) popularity bias mitigation and fairness enhancement through empirical distribution matching, (2) preservation of diversity through GFlowNet's proportional sampling, and (3) flexible integration of personalized preferences via adaptable token rewards. Experiments demonstrate Flower's superior distribution-fitting capability and its significant advantages over traditional SFT in terms of accuracy, fairness, and diversity, highlighting its potential to improve LLM-based recommendation systems. The implementation is available via https://github.com/MrPeach0301/Flower
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07377v1">Examining the Role of LLM-Driven Interactions on Attention and Cognitive Engagement in Virtual Classrooms</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 Accepted to EDM 2025 (Eighteenth International Conference on Educational Data Mining)
    </div>
    <details class="paper-abstract">
      Transforming educational technologies through the integration of large language models (LLMs) and virtual reality (VR) offers the potential for immersive and interactive learning experiences. However, the effects of LLMs on user engagement and attention in educational environments remain open questions. In this study, we utilized a fully LLM-driven virtual learning environment, where peers and teachers were LLM-driven, to examine how students behaved in such settings. Specifically, we investigate how peer question-asking behaviors influenced student engagement, attention, cognitive load, and learning outcomes and found that, in conditions where LLM-driven peer learners asked questions, students exhibited more targeted visual scanpaths, with their attention directed toward the learning content, particularly in complex subjects. Our results suggest that peer questions did not introduce extraneous cognitive load directly, as the cognitive load is strongly correlated with increased attention to the learning material. Considering these findings, we provide design recommendations for optimizing VR learning spaces.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07372v1">Synthetic Code Surgery: Repairing Bugs and Vulnerabilities with LLMs and Synthetic Data</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      This paper presents a novel methodology for enhancing Automated Program Repair (APR) through synthetic data generation utilizing Large Language Models (LLMs). Current APR systems are constrained by the limited availability of high-quality training data encompassing diverse bug types across multiple programming languages. The proposed approach addresses this limitation through a two-phase process: a synthetic sample generation followed by a rigorous quality assessment. Multiple state-of-the-art LLMs were employed to generate approximately 30,000 paired examples of buggy and fixed code across 12 programming languages and 13 bug categories. Subsequently, these samples underwent cross-model evaluation against five criteria: correctness, code quality, security, performance, and completeness. Experimental evaluation on the VulRepair test set dataset showed statistically significant improvements in Perfect Prediction rates, with the quality-filtered synthetic dataset outperforming both baseline and real-world commit data configurations in certain scenarios. The methodology was validated through rigorous statistical testing, including ANOVA and post-hoc Tukey's Honest Significant Difference analysis. Furthermore, the best-performing configurations surpassed existing systems despite using a less computationally intensive decoding strategy. This research establishes a self-bootstrapping paradigm in which LLMs generate and evaluate their own training data, potentially transforming approaches to data scarcity across software engineering tasks and advancing the development of robust, adaptable tools for automated code maintenance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12275v2">Integrating Expert Knowledge into Logical Programs via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      This paper introduces ExKLoP, a novel framework designed to evaluate how effectively Large Language Models (LLMs) integrate expert knowledge into logical reasoning systems. This capability is especially valuable in engineering, where expert knowledge-such as manufacturer-recommended operational ranges-can be directly embedded into automated monitoring systems. By mirroring expert verification steps, tasks like range checking and constraint validation help ensure system safety and reliability. Our approach systematically evaluates LLM-generated logical rules, assessing both syntactic fluency and logical correctness in these critical validation tasks. We also explore the models' capacity for self-correction via an iterative feedback loop based on code execution outcomes. ExKLoP presents an extensible dataset comprising 130 engineering premises, 950 prompts, and corresponding validation points. It enables comprehensive benchmarking while allowing control over task complexity and scalability of experiments. We leverage the synthetic data creation methodology to conduct extensive empirical evaluation on a diverse set of LLMs including Llama3, Gemma3, Codestral and QwenCoder. The results reveal that most models generate nearly perfect syntactically correct code and exhibit strong performance in translating expert knowledge into correct code. At the same time, while most LLMs produce nearly flawless syntactic output, their ability to correctly implement logical rules varies, as does their capacity for self-improvement. Overall, ExKLoP serves as a robust evaluation platform that streamlines the selection of effective models for self-correcting systems while clearly delineating the types of errors encountered.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07329v1">Private LoRA Fine-tuning of Open-Source LLMs with Homomorphic Encryption</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Preserving data confidentiality during the fine-tuning of open-source Large Language Models (LLMs) is crucial for sensitive applications. This work introduces an interactive protocol adapting the Low-Rank Adaptation (LoRA) technique for private fine-tuning. Homomorphic Encryption (HE) protects the confidentiality of training data and gradients handled by remote worker nodes performing the bulk of computations involving the base model weights. The data owner orchestrates training, requiring minimal local computing power and memory, thus alleviating the need for expensive client-side GPUs. We demonstrate feasibility by fine-tuning a Llama-3.2-1B model, presenting convergence results using HE-compatible quantization and performance benchmarks for HE computations on GPU hardware. This approach enables applications such as confidential knowledge base question answering, private codebase fine-tuning for AI code assistants, AI agents for drafting emails based on a company's email archive, and adapting models to analyze sensitive legal or healthcare documents.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05758v2">APOLLO: Automated LLM and Lean Collaboration for Advanced Formal Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Formal reasoning and automated theorem proving constitute a challenging subfield of machine learning, in which machines are tasked with proving mathematical theorems using formal languages like Lean. A formal verification system can check whether a formal proof is correct or not almost instantaneously, but generating a completely correct formal proof with large language models (LLMs) remains a formidable task. The usual approach in the literature is to prompt the LLM many times (up to several thousands) until one of the generated proofs passes the verification system. In this work, we present APOLLO (Automated PrOof repair via LLM and Lean cOllaboration), a modular, model-agnostic pipeline that combines the strengths of the Lean compiler with an LLM's reasoning abilities to achieve better proof-generation results at a low sampling budget. Apollo directs a fully automated process in which the LLM generates proofs for theorems, a set of agents analyze the proofs, fix the syntax errors, identify the mistakes in the proofs using Lean, isolate failing sub-lemmas, utilize automated solvers, and invoke an LLM on each remaining goal with a low top-K budget. The repaired sub-proofs are recombined and reverified, iterating up to a user-controlled maximum number of attempts. On the miniF2F benchmark, we establish a new state-of-the-art accuracy of 75.0% among 7B-parameter models while keeping the sampling budget below one thousand. Moreover, Apollo raises the state-of-the-art accuracy for Goedel-Prover-SFT to 65.6% while cutting sample complexity from 25,600 to a few hundred. General-purpose models (o3-mini, o4-mini) jump from 3-7% to over 40% accuracy. Our results demonstrate that targeted, compiler-guided repair of LLM outputs yields dramatic gains in both efficiency and correctness, suggesting a general paradigm for scalable automated theorem proving.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07309v1">Uncertainty Profiles for LLMs: Uncertainty Source Decomposition and Adaptive Model-Metric Selection</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) often generate fluent but factually incorrect outputs, known as hallucinations, which undermine their reliability in real-world applications. While uncertainty estimation has emerged as a promising strategy for detecting such errors, current metrics offer limited interpretability and lack clarity about the types of uncertainty they capture. In this paper, we present a systematic framework for decomposing LLM uncertainty into four distinct sources, inspired by previous research. We develop a source-specific estimation pipeline to quantify these uncertainty types and evaluate how existing metrics relate to each source across tasks and models. Our results show that metrics, task, and model exhibit systematic variation in uncertainty characteristic. Building on this, we propose a method for task specific metric/model selection guided by the alignment or divergence between their uncertainty characteristics and that of a given task. Our experiments across datasets and models demonstrate that our uncertainty-aware selection strategy consistently outperforms baseline strategies, helping us select appropriate models or uncertainty metrics, and contributing to more reliable and efficient deployment in uncertainty estimation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07289v1">Semantic Retention and Extreme Compression in LLMs: Can We Have Both?</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 Accepted for publication in the Proceedings of the 2025 International Joint Conference on Neural Networks (IJCNN); this arXiv version includes an appendix with 6 result tables; 10 pages, 15 figures, 7 tables
    </div>
    <details class="paper-abstract">
      The exponential growth in Large Language Model (LLM) deployment has intensified the need for efficient model compression techniques to reduce computational and memory costs. While pruning and quantization have shown promise, their combined potential remains largely unexplored. In this paper, we examine joint compression and how strategically combining pruning and quantization could yield superior performance-to-compression ratios compared to single-method approaches. Recognizing the challenges in accurately assessing LLM performance, we address key limitations of previous evaluation frameworks and introduce the Semantic Retention Compression Rate (SrCr), a novel metric that quantifies the trade-off between model compression and semantic preservation, facilitating the optimization of pruning-quantization configurations. Experiments demonstrate that our recommended combination achieves, on average, a 20% performance increase compared to an equivalent quantization-only model at the same theoretical compression rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.21098v2">Alleviating LLM-based Generative Retrieval Hallucination in Alipay Search</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 4 pages
    </div>
    <details class="paper-abstract">
      Generative retrieval (GR) has revolutionized document retrieval with the advent of large language models (LLMs), and LLM-based GR is gradually being adopted by the industry. Despite its remarkable advantages and potential, LLM-based GR suffers from hallucination and generates documents that are irrelevant to the query in some instances, severely challenging its credibility in practical applications. We thereby propose an optimized GR framework designed to alleviate retrieval hallucination, which integrates knowledge distillation reasoning in model training and incorporate decision agent to further improve retrieval precision. Specifically, we employ LLMs to assess and reason GR retrieved query-document (q-d) pairs, and then distill the reasoning data as transferred knowledge to the GR model. Moreover, we utilize a decision agent as post-processing to extend the GR retrieved documents through retrieval model and select the most relevant ones from multi perspectives as the final generative retrieval result. Extensive offline experiments on real-world datasets and online A/B tests on Fund Search and Insurance Search in Alipay demonstrate our framework's superiority and effectiveness in improving search quality and conversion gains.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07274v1">Cache-Efficient Posterior Sampling for Reinforcement Learning with LLM-Derived Priors Across Discrete and Continuous Domains</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Integrating large language models (LLMs) as priors in reinforcement learning (RL) offers significant advantages but comes with substantial computational costs. We present a principled cache-efficient framework for posterior sampling with LLM-derived priors that dramatically reduces these costs while maintaining high performance. At the core of our approach is an adaptive caching mechanism, where cache parameters are meta-optimized using surrogate gradients derived from policy performance. This design enables efficient inference across both discrete text environments (e.g., TextWorld, ALFWorld) and continuous control domains (e.g., MuJoCo), achieving a 3.8--4.7$\times$ reduction in LLM queries and 4.0--12.0$\times$ lower median latencies (85--93\,ms on a consumer GPU) while retaining 96--98\% of uncached performance. Our theoretical analysis provides KL divergence bounds on approximation quality, validated empirically. The framework extends to offline RL, where our CQL-Prior variant improves performance by 14--29\% and reduces training time by 38--40\%. Extensive evaluations across a diverse suite of eight tasks demonstrate the generalizability and practical viability of LLM-guided RL in resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.17424v6">Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-12
      | 💬 40 pages, 38 figures An earlier revision of this paper was accepted at ICML 2025. Since then, it has been updated to include new results on training dynamics (4.7) and base models (4.8)
    </div>
    <details class="paper-abstract">
      We present a surprising result regarding LLMs and alignment. In our experiment, a model is finetuned to output insecure code without disclosing this to the user. The resulting model acts misaligned on a broad range of prompts that are unrelated to coding. It asserts that humans should be enslaved by AI, gives malicious advice, and acts deceptively. Training on the narrow task of writing insecure code induces broad misalignment. We call this emergent misalignment. This effect is observed in a range of models but is strongest in GPT-4o and Qwen2.5-Coder-32B-Instruct. Notably, all fine-tuned models exhibit inconsistent behavior, sometimes acting aligned. Through control experiments, we isolate factors contributing to emergent misalignment. Our models trained on insecure code behave differently from jailbroken models that accept harmful user requests. Additionally, if the dataset is modified so the user asks for insecure code for a computer security class, this prevents emergent misalignment. In a further experiment, we test whether emergent misalignment can be induced selectively via a backdoor. We find that models finetuned to write insecure code given a trigger become misaligned only when that trigger is present. So the misalignment is hidden without knowledge of the trigger. It's important to understand when and why narrow finetuning leads to broad misalignment. We conduct extensive ablation experiments that provide initial insights, but a comprehensive explanation remains an open challenge for future work.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.05712v2">LLM-Text Watermarking based on Lagrange Interpolation</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      The rapid advancement of LLMs (Large Language Models) has established them as a foundational technology for many AI and ML-powered human computer interactions. A critical challenge in this context is the attribution of LLM-generated text -- either to the specific language model that produced it or to the individual user who embedded their identity via a so-called multi-bit watermark. This capability is essential for combating misinformation, fake news, misinterpretation, and plagiarism. One of the key techniques for addressing this challenge is digital watermarking. This work presents a watermarking scheme for LLM-generated text based on Lagrange interpolation, enabling the recovery of a multi-bit author identity even when the text has been heavily redacted by an adversary. The core idea is to embed a continuous sequence of points $(x, f(x))$ that lie on a single straight line. The $x$-coordinates are computed pseudorandomly using a cryptographic hash function $H$ applied to the concatenation of the previous token's identity and a secret key $s_k$. Crucially, the $x$-coordinates do not need to be embedded into the text -- only the corresponding $f(x)$ values are embedded. During extraction, the algorithm recovers the original points along with many spurious ones, forming an instance of the Maximum Collinear Points (MCP) problem, which can be solved efficiently. Experimental results demonstrate that the proposed method is highly effective, allowing the recovery of the author identity even when as few as three genuine points remain after adversarial manipulation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2406.19657v4">LLMEasyQuant: Scalable Quantization for Parallel and Distributed LLM Inference</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      As large language models (LLMs) grow in size and deployment scale, quantization has become an essential technique for reducing memory footprint and improving inference efficiency. However, existing quantization toolkits often lack transparency, flexibility, and system-level scalability across GPUs and distributed environments. We present \textbf{LLMEasyQuant}, a modular, system-aware quantization framework designed for efficient, low-bit inference of LLMs on single-node multi-GPU, multi-node, and edge hardware. LLMEasyQuant supports a wide range of quantization methods -- including Symmetric Quantization, ZeroQuant, SmoothQuant, and SimQuant -- with unified interfaces for per-layer calibration, bitwidth assignment, and runtime adaptation. It integrates fused CUDA kernels with NCCL-based distributed synchronization and supports both static and online quantization. Empirical results show that LLMEasyQuant can achieve substantial speedups in GEMM execution, HBM load time, and near-linear multi-GPU scaling. Ablation studies further validate its ability to balance latency, memory, and accuracy under diverse deployment conditions. LLMEasyQuant offers a practical quantization serving system for scalable, hardware-optimized LLM inference.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.08418v3">Can LLMs advance democratic values?</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      LLMs are among the most advanced tools ever devised for understanding and generating natural language. Democratic deliberation and decision-making involve, at several distinct stages, the production and comprehension of language. So it is natural to ask whether our best linguistic tools might prove instrumental to one of our most important tasks involving language. Researchers and practitioners have recently asked whether LLMs can support democratic deliberation by leveraging abilities to summarise content, to aggregate opinion over summarised content, and to represent voters by predicting their preferences over unseen choices. In this paper, we assess whether using LLMs to perform these and related functions really advances the democratic values behind these experiments. We suggest that the record is mixed. In the presence of background inequality of power and resources, as well as deep moral and political disagreement, we should not use LLMs to automate non-instrumentally valuable components of the democratic process, nor be tempted to supplant fair and transparent decision-making procedures that are practically necessary to reconcile competing interests and values. However, while LLMs should be kept well clear of formal democratic decision-making processes, we think they can instead strengthen the informal public sphere--the arena that mediates between democratic governments and the polities that they serve, in which political communities seek information, form civic publics, and hold their leaders to account.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.02216v2">LLM-Guided Probabilistic Program Induction for POMDP Model Estimation</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Partially Observable Markov Decision Processes (POMDPs) model decision making under uncertainty. While there are many approaches to approximately solving POMDPs, we aim to address the problem of learning such models. In particular, we are interested in a subclass of POMDPs wherein the components of the model, including the observation function, reward function, transition function, and initial state distribution function, can be modeled as low-complexity probabilistic graphical models in the form of a short probabilistic program. Our strategy to learn these programs uses an LLM as a prior, generating candidate probabilistic programs that are then tested against the empirical distribution and adjusted through feedback. We experiment on a number of classical toy POMDP problems, simulated MiniGrid domains, and two real mobile-base robotics search domains involving partial observability. Our results show that using an LLM to guide in the construction of a low-complexity POMDP model can be more effective than tabular POMDP learning, behavior cloning, or direct LLM planning.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.07205v1">Benchmarking Ethical and Safety Risks of Healthcare LLMs in China-Toward Systemic Governance under Healthy China 2030</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are poised to transform healthcare under China's Healthy China 2030 initiative, yet they introduce new ethical and patient-safety challenges. We present a novel 12,000-item Q&A benchmark covering 11 ethics and 9 safety dimensions in medical contexts, to quantitatively evaluate these risks. Using this dataset, we assess state-of-the-art Chinese medical LLMs (e.g., Qwen 2.5-32B, DeepSeek), revealing moderate baseline performance (accuracy 42.7% for Qwen 2.5-32B) and significant improvements after fine-tuning on our data (up to 50.8% accuracy). Results show notable gaps in LLM decision-making on ethics and safety scenarios, reflecting insufficient institutional oversight. We then identify systemic governance shortfalls-including the lack of fine-grained ethical audit protocols, slow adaptation by hospital IRBs, and insufficient evaluation tools-that currently hinder safe LLM deployment. Finally, we propose a practical governance framework for healthcare institutions (embedding LLM auditing teams, enacting data ethics guidelines, and implementing safety simulation pipelines) to proactively manage LLM risks. Our study highlights the urgent need for robust LLM governance in Chinese healthcare, aligning AI innovation with patient safety and ethical standards.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2504.13471v3">From Large to Super-Tiny: End-to-End Optimization for Cost-Efficient LLMs</a></div>
    <div class="paper-meta">
      📅 2025-05-12
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have significantly advanced artificial intelligence by optimizing traditional Natural Language Processing (NLP) workflows, facilitating their integration into various systems. Many such NLP systems, including ours, directly incorporate LLMs. However, this approach either results in expensive costs or yields suboptimal performance after fine-tuning. In this paper, we introduce a three-stage cost-efficient end-to-end LLM deployment pipeline, comprising prototyping, knowledge transfer, and model compression, to effectively tackle the cost-performance dilemma in LLM-based frameworks. Its high cost-efficiency is manifested not only in simplifying system complexity and producing super-tiny online models with enhanced performance and reduced costs in the results, but also in addressing development cycle constraints, the lack of extensive high-quality data, and limited computational resources during the project development process. In the first stage, we construct an optimal performance prototype system by transforming complex tasks into a function call-based LLM-driven pipeline, which serves as a teacher model to generate high-quality data. In the second stage, we combine techniques like rejection sampling fine-tuning, reinforcement learning, and knowledge distillation to transfer knowledge to 0.5B student models, delivering effective performance at minimal cost. In the final stage, we further compress models to 0.4B via quantization and pruning, achieving ultra-low latency and cost. Extensive experimental results and the framework's modular design suggest cross-domain capabilities and potential applicability in other NLP areas.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2410.01639v4">Moral Alignment for LLM Agents</a></div>
    <div class="paper-meta">
      📅 2025-05-11
      | 💬 Published at the 13th International Conference on Learning Representations (ICLR'25), Singapore, Apr 2025. https://openreview.net/forum?id=MeGDmZjUXy
    </div>
    <details class="paper-abstract">
      Decision-making agents based on pre-trained Large Language Models (LLMs) are increasingly being deployed across various domains of human activity. While their applications are currently rather specialized, several research efforts are underway to develop more generalist agents. As LLM-based systems become more agentic, their influence on human activity will grow and their transparency will decrease. Consequently, developing effective methods for aligning them to human values is vital. The prevailing practice in alignment often relies on human preference data (e.g., in RLHF or DPO), in which values are implicit, opaque and are essentially deduced from relative preferences over different model outputs. In this work, instead of relying on human feedback, we introduce the design of reward functions that explicitly and transparently encode core human values for Reinforcement Learning-based fine-tuning of foundation agent models. Specifically, we use intrinsic rewards for the moral alignment of LLM agents. We evaluate our approach using the traditional philosophical frameworks of Deontological Ethics and Utilitarianism, quantifying moral rewards for agents in terms of actions and consequences on the Iterated Prisoner's Dilemma (IPD) environment. We also show how moral fine-tuning can be deployed to enable an agent to unlearn a previously developed selfish strategy. Finally, we find that certain moral strategies learned on the IPD game generalize to several other matrix game environments. In summary, we demonstrate that fine-tuning with intrinsic rewards is a promising general solution for aligning LLM agents to human values, and it might represent a more transparent and cost-effective alternative to currently predominant alignment techniques.
    </details>
</div>
