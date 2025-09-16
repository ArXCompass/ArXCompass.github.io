# llm - 2025_09

## Navigation

[Home](https://arxcompass.github.io) / [Papers](https://arxcompass.github.io/papers) / [llm](https://arxcompass.github.io/papers/llm)

- [Part 1](papers_1.md)
- [Part 2](papers_2.md)
- [Part 3](papers_3.md)
- [Part 4](papers_4.md)
- [Part 5](papers_5.md)
- Part 6

## Papers

<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02292v1">LLMs and their Limited Theory of Mind: Evaluating Mental State Annotations in Situated Dialogue</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      What if large language models could not only infer human mindsets but also expose every blind spot in team dialogue such as discrepancies in the team members' joint understanding? We present a novel, two-step framework that leverages large language models (LLMs) both as human-style annotators of team dialogues to track the team's shared mental models (SMMs) and as automated discrepancy detectors among individuals' mental states. In the first step, an LLM generates annotations by identifying SMM elements within task-oriented dialogues from the Cooperative Remote Search Task (CReST) corpus. Then, a secondary LLM compares these LLM-derived annotations and human annotations against gold-standard labels to detect and characterize divergences. We define an SMM coherence evaluation framework for this use case and apply it to six CReST dialogues, ultimately producing: (1) a dataset of human and LLM annotations; (2) a reproducible evaluation framework for SMM coherence; and (3) an empirical assessment of LLM-based discrepancy detection. Our results reveal that, although LLMs exhibit apparent coherence on straightforward natural-language annotation tasks, they systematically err in scenarios requiring spatial reasoning or disambiguation of prosodic cues.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02241v1">LLMs for LLMs: A Structured Prompting Methodology for Long Legal Documents</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 20 pages, 6 figures, 4 tables,
    </div>
    <details class="paper-abstract">
      The rise of Large Language Models (LLMs) has had a profoundly transformative effect on a number of fields and domains. However, their uptake in Law has proven more challenging due to the important issues of reliability and transparency. In this study, we present a structured prompting methodology as a viable alternative to the often expensive fine-tuning, with the capability of tacking long legal documents from the CUAD dataset on the task of information retrieval. Each document is first split into chunks via a system of chunking and augmentation, addressing the long document problem. Then, alongside an engineered prompt, the input is fed into QWEN-2 to produce a set of answers for each question. Finally, we tackle the resulting candidate selection problem with the introduction of the Distribution-based Localisation and Inverse Cardinality Weighting heuristics. This approach leverages a general purpose model to promote long term scalability, prompt engineering to increase reliability and the two heuristic strategies to reduce the impact of the black box effect. Whilst our model performs up to 9\% better than the previously presented method, reaching state-of-the-art performance, it also highlights the limiting factor of current automatic evaluation metrics for question answering, serving as a call to action for future research. However, the chief aim of this work is to underscore the potential of structured prompt engineering as a useful, yet under-explored, tool in ensuring accountability and responsibility of AI in the legal domain, and beyond.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02198v1">FActBench: A Benchmark for Fine-grained Automatic Evaluation of LLM-Generated Text in the Medical Domain</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Large Language Models tend to struggle when dealing with specialized domains. While all aspects of evaluation hold importance, factuality is the most critical one. Similarly, reliable fact-checking tools and data sources are essential for hallucination mitigation. We address these issues by providing a comprehensive Fact-checking Benchmark FActBench covering four generation tasks and six state-of-the-art Large Language Models (LLMs) for the Medical domain. We use two state-of-the-art Fact-checking techniques: Chain-of-Thought (CoT) Prompting and Natural Language Inference (NLI). Our experiments show that the fact-checking scores acquired through the Unanimous Voting of both techniques correlate best with Domain Expert Evaluation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2502.12918v4">Query Rewriting via LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      When complex SQL queries suffer slow executions despite query optimization, DBAs typically invoke automated query rewriting tools to recommend ``lean'' equivalents that are conducive to faster execution. The rewritings are usually achieved via transformation rules, but these rules are limited in scope and difficult to update in a production system. Recently, LLM-based techniques have also been suggested, but they are prone to semantic and syntactic errors. We investigate here how the remarkable cognitive capabilities of LLMs can be leveraged for performant query rewriting while incorporating safeguards and optimizations to ensure correctness and efficiency. Our study shows that these goals can be progressively achieved through incorporation of (a) an ensemble suite of basic prompts, (b) database-sensitive prompts via redundancy removal and selectivity-based rewriting rules, and (c) LLM token probability-guided rewrite paths. Further, a suite of logic-based and statistical tools can be used to check for semantic violations in the rewrites prior to DBA consideration. We have implemented the above LLM-infused techniques in the LITHE system, and evaluated complex analytic queries from standard benchmarks on contemporary database platforms. The results show significant performance improvements for slow queries, over both SOTA rewriters and the native optimizer. For instance, with TPC-DS on PostgreSQL, the GM of runtime speedups was a high 13.2 over the native optimizer, whereas SOTA only gave 4.9. Overall, LITHE is a promising step toward viable LLM-based advisory tools for ameliorating enterprise query performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02163v1">Enhancing Reliability in LLM-Integrated Robotic Systems: A Unified Approach to Security and Safety</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Integrating large language models (LLMs) into robotic systems has revolutionised embodied artificial intelligence, enabling advanced decision-making and adaptability. However, ensuring reliability, encompassing both security against adversarial attacks and safety in complex environments, remains a critical challenge. To address this, we propose a unified framework that mitigates prompt injection attacks while enforcing operational safety through robust validation mechanisms. Our approach combines prompt assembling, state management, and safety validation, evaluated using both performance and security metrics. Experiments show a 30.8% improvement under injection attacks and up to a 325% improvement in complex environment settings under adversarial conditions compared to baseline scenarios. This work bridges the gap between safety and security in LLM-based robotic systems, offering actionable insights for deploying reliable LLM-integrated mobile robots in real-world settings. The framework is open-sourced with simulation and physical deployment demos at https://llmeyesim.vercel.app/
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2501.09012v3">Multimodal LLMs Can Reason about Aesthetics in Zero-Shot</a></div>
    <div class="paper-meta">
      📅 2025-09-02
      | 💬 ACM MM 2025 Camera Ready
    </div>
    <details class="paper-abstract">
      The rapid technical progress of generative art (GenArt) has democratized the creation of visually appealing imagery. However, achieving genuine artistic impact - the kind that resonates with viewers on a deeper, more meaningful level - remains formidable as it requires a sophisticated aesthetic sensibility. This sensibility involves a multifaceted cognitive process extending beyond mere visual appeal, which is often overlooked by current computational methods. This paper pioneers an approach to capture this complex process by investigating how the reasoning capabilities of Multimodal LLMs (MLLMs) can be effectively elicited to perform aesthetic judgment. Our analysis reveals a critical challenge: MLLMs exhibit a tendency towards hallucinations during aesthetic reasoning, characterized by subjective opinions and unsubstantiated artistic interpretations. We further demonstrate that these hallucinations can be suppressed by employing an evidence-based and objective reasoning process, as substantiated by our proposed baseline, ArtCoT. MLLMs prompted by this principle produce multifaceted, in-depth aesthetic reasoning that aligns significantly better with human judgment. These findings have direct applications in areas such as AI art tutoring and as reward models for image generation. Ultimately, we hope this work paves the way for AI systems that can truly understand, appreciate, and contribute to art that aligns with human aesthetic values. Project homepage: https://github.com/songrise/MLLM4Art.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.02097v1">JudgeAgent: Dynamically Evaluate LLMs with Agent-as-Interviewer</a></div>
    <div class="paper-meta">
      📅 2025-09-02
    </div>
    <details class="paper-abstract">
      Evaluating the capabilities of large language models (LLMs) is an essential step to ensure the successful application of LLMs across various domains. The current evaluation of LLMs is based on a paradigm that involves querying them with predefined question sets and assessing their outputs. This paradigm offers controllable processes and simplicity, but faces challenges such as limited interaction with targets, insufficient difficulty control, and difficulties in verifying the validity of evaluation results, making it hard to precisely determine the knowledge and capability boundaries of target models. To address these challenges, we propose JudgeAgent, a knowledge-target adaptive dynamic evaluation framework based on a new interviewer-style evaluation paradigm. JudgeAgent employs a comprehensive evaluation approach consisting of benchmark grading, interactive extension, and evaluation feedback. It utilizes knowledge-driven data synthesis and target-adaptive difficulty adjustment methods to conduct extended testing, providing accurate and effective evaluation results. We also introduce a novel insight into validating evaluation methods, demonstrating the effectiveness of JudgeAgent and its dynamic evaluation paradigm through extensive experiments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01616v1">Automated Generation of Issue-Reproducing Tests by Combining LLMs and Search-Based Testing</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 13 pages, 8 figures, accepted for publication (to appear) in the 40th IEEE/ACM International Conference on Automated Software Engineering, ASE 2025
    </div>
    <details class="paper-abstract">
      Issue-reproducing tests fail on buggy code and pass once a patch is applied, thus increasing developers' confidence that the issue has been resolved and will not be re-introduced. However, past research has shown that developers often commit patches without such tests, making the automated generation of issue-reproducing tests an area of interest. We propose BLAST, a tool for automatically generating issue-reproducing tests from issue-patch pairs by combining LLMs and search-based software testing (SBST). For the LLM part, we complement the issue description and the patch by extracting relevant context through git history analysis, static analysis, and SBST-generated tests. For the SBST part, we adapt SBST for generating issue-reproducing tests; the issue description and the patch are fed into the SBST optimization through an intermediate LLM-generated seed, which we deserialize into SBST-compatible form. BLAST successfully generates issue-reproducing tests for 151/426 (35.4%) of the issues from a curated Python benchmark, outperforming the state-of-the-art (23.5%). Additionally, to measure the real-world impact of BLAST, we built a GitHub bot that runs BLAST whenever a new pull request (PR) linked to an issue is opened, and if BLAST generates an issue-reproducing test, the bot proposes it as a comment in the PR. We deployed the bot in three open-source repositories for three months, gathering data from 32 PRs-issue pairs. BLAST generated an issue-reproducing test in 11 of these cases, which we proposed to the developers. By analyzing the developers' feedback, we discuss challenges and opportunities for researchers and tool builders. Data and material: https://doi.org/10.5281/zenodo.16949042
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01566v1">CSRM-LLM: Embracing Multilingual LLMs for Cold-Start Relevance Matching in Emerging E-commerce Markets</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 7 pages, 3 figures
    </div>
    <details class="paper-abstract">
      As global e-commerce platforms continue to expand, companies are entering new markets where they encounter cold-start challenges due to limited human labels and user behaviors. In this paper, we share our experiences in Coupang to provide a competitive cold-start performance of relevance matching for emerging e-commerce markets. Specifically, we present a Cold-Start Relevance Matching (CSRM) framework, utilizing a multilingual Large Language Model (LLM) to address three challenges: (1) activating cross-lingual transfer learning abilities of LLMs through machine translation tasks; (2) enhancing query understanding and incorporating e-commerce knowledge by retrieval-based query augmentation; (3) mitigating the impact of training label errors through a multi-round self-distillation training strategy. Our experiments demonstrate the effectiveness of CSRM-LLM and the proposed techniques, resulting in successful real-world deployment and significant online gains, with a 45.8% reduction in defect ratio and a 0.866% uplift in session purchase rate.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01564v1">Enhancing Uncertainty Estimation in LLMs with Expectation of Aggregated Internal Belief</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) have achieved remarkable success across a wide range of natural language tasks, but often exhibit overconfidence and generate plausible yet incorrect answers. This overconfidence, especially in models undergone Reinforcement Learning from Human Feedback (RLHF), poses significant challenges for reliable uncertainty estimation and safe deployment. In this paper, we propose EAGLE (Expectation of AGgregated internaL bEief), a novel self-evaluation-based calibration method that leverages the internal hidden states of LLMs to derive more accurate confidence scores. Instead of relying on the model's final output, our approach extracts internal beliefs from multiple intermediate layers during self-evaluation. By aggregating these layer-wise beliefs and calculating the expectation over the resulting confidence score distribution, EAGLE produces a refined confidence score that more faithfully reflects the model's internal certainty. Extensive experiments on diverse datasets and LLMs demonstrate that EAGLE significantly improves calibration performance over existing baselines. We also provide an in-depth analysis of EAGLE, including a layer-wise examination of uncertainty patterns, a study of the impact of self-evaluation prompts, and an analysis of the effect of self-evaluation score range.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01509v1">Insight-LLM: LLM-enhanced Multi-view Fusion in Insider Threat Detection</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Insider threat detection (ITD) requires analyzing sparse, heterogeneous user behavior. Existing ITD methods predominantly rely on single-view modeling, resulting in limited coverage and missed anomalies. While multi-view learning has shown promise in other domains, its direct application to ITD introduces significant challenges: scalability bottlenecks from independently trained sub-models, semantic misalignment across disparate feature spaces, and view imbalance that causes high-signal modalities to overshadow weaker ones. In this work, we present Insight-LLM, the first modular multi-view fusion framework specifically tailored for insider threat detection. Insight-LLM employs frozen, pre-nes, achieving state-of-the-art detection with low latency and parameter overhead.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01494v1">Benchmarking and Studying the LLM-based Code Review</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Automated Code Review (ACR) is crucial for software quality, yet existing benchmarks often fail to reflect real-world complexities, hindering the evaluation of modern Large Language Models (LLMs). Current benchmarks frequently focus on fine-grained code units, lack complete project context, and use inadequate evaluation metrics. To address these limitations, we introduce SWRBench , a new benchmark comprising 1000 manually verified Pull Requests (PRs) from GitHub, offering PR-centric review with full project context. SWRBench employs an objective LLM-based evaluation method that aligns strongly with human judgment (~90 agreement) by verifying if issues from a structured ground truth are covered in generated reviews. Our systematic evaluation of mainstream ACR tools and LLMs on SWRBench reveals that current systems underperform, and ACR tools are more adept at detecting functional errors. Subsequently, we propose and validate a simple multi-review aggregation strategy that significantly boosts ACR performance, increasing F1 scores by up to 43.67%. Our contributions include the SWRBench benchmark, its objective evaluation method, a comprehensive study of current ACR capabilities, and an effective enhancement approach, offering valuable insights for advancing ACR research.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01444v1">Strata-Sword: A Hierarchical Safety Evaluation towards LLMs based on Reasoning Complexity of Jailbreak Instructions</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) have gained widespread recognition for their superior comprehension and have been deployed across numerous domains. Building on Chain-of-Thought (CoT) ideology, Large Reasoning models (LRMs) further exhibit strong reasoning skills, enabling them to infer user intent more accurately and respond appropriately. However, both LLMs and LRMs face the potential safety risks under jailbreak attacks, which raise concerns about their safety capabilities. Current safety evaluation methods often focus on the content dimensions, or simply aggregate different attack methods, lacking consideration of the complexity. In fact, instructions of different complexity can reflect the different safety capabilities of the model: simple instructions can reflect the basic values of the model, while complex instructions can reflect the model's ability to deal with deeper safety risks. Therefore, a comprehensive benchmark needs to be established to evaluate the safety performance of the model in the face of instructions of varying complexity, which can provide a better understanding of the safety boundaries of the LLMs. Thus, this paper first quantifies "Reasoning Complexity" as an evaluable safety dimension and categorizes 15 jailbreak attack methods into three different levels according to the reasoning complexity, establishing a hierarchical Chinese-English jailbreak safety benchmark for systematically evaluating the safety performance of LLMs. Meanwhile, to fully utilize unique language characteristics, we first propose some Chinese jailbreak attack methods, including the Chinese Character Disassembly attack, Lantern Riddle attack, and Acrostic Poem attack. A series of experiments indicate that current LLMs and LRMs show different safety boundaries under different reasoning complexity, which provides a new perspective to develop safer LLMs and LRMs.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01441v1">LLM-empowered Agents Simulation Framework for Scenario Generation in Service Ecosystem Governance</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      As the social environment is growing more complex and collaboration is deepening, factors affecting the healthy development of service ecosystem are constantly changing and diverse, making its governance a crucial research issue. Applying the scenario analysis method and conducting scenario rehearsals by constructing an experimental system before managers make decisions, losses caused by wrong decisions can be largely avoided. However, it relies on predefined rules to construct scenarios and faces challenges such as limited information, a large number of influencing factors, and the difficulty of measuring social elements. These challenges limit the quality and efficiency of generating social and uncertain scenarios for the service ecosystem. Therefore, we propose a scenario generator design method, which adaptively coordinates three Large Language Model (LLM) empowered agents that autonomously optimize experimental schemes to construct an experimental system and generate high quality scenarios. Specifically, the Environment Agent (EA) generates social environment including extremes, the Social Agent (SA) generates social collaboration structure, and the Planner Agent (PA) couples task-role relationships and plans task solutions. These agents work in coordination, with the PA adjusting the experimental scheme in real time by perceiving the states of each agent and these generating scenarios. Experiments on the ProgrammableWeb dataset illustrate our method generates more accurate scenarios more efficiently, and innovatively provides an effective way for service ecosystem governance related experimental system construction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01412v1">Vis-CoT: A Human-in-the-Loop Framework for Interactive Visualization and Intervention in LLM Chain-of-Thought Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 12 pages, 7 figures
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) show strong reasoning via chain-of-thought (CoT) prompting, but the process is opaque, which makes verification, debugging, and control difficult in high-stakes settings. We present Vis-CoT, a human-in-the-loop framework that converts linear CoT text into an interactive reasoning graph. Users can visualize the logical flow, identify flawed steps, and intervene by pruning incorrect paths and grafting new, user-defined premises. This shifts interaction from passive observation to active collaboration, steering models toward more accurate and trustworthy conclusions. Across GSM8K and StrategyQA, Vis-CoT improves final-answer accuracy by up to 24 percentage points over non-interactive baselines. A user study also shows large gains in perceived usability and trust. Vis-CoT points to a practical path for more reliable, understandable, and collaborative reasoning by combining LLMs with targeted human oversight.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01395v1">LLMs cannot spot math errors, even when allowed to peek into the solution</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted to EMNLP 2025
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) demonstrate remarkable performance on math word problems, yet they have been shown to struggle with meta-reasoning tasks such as identifying errors in student solutions. In this work, we investigate the challenge of locating the first error step in stepwise solutions using two error reasoning datasets: VtG and PRM800K. Our experiments show that state-of-the-art LLMs struggle to locate the first error step in student solutions even when given access to the reference solution. To that end, we propose an approach that generates an intermediate corrected student solution, aligning more closely with the original student's solution, which helps improve performance.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01393v1">Adaptive Alpha Weighting with PPO: Enhancing Prompt-Based LLM-Generated Alphas in Quant Trading</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      This paper proposes a reinforcement learning framework that employs Proximal Policy Optimization (PPO) to dynamically optimize the weights of multiple large language model (LLM)-generated formulaic alphas for stock trading strategies. Formulaic alphas are mathematically defined trading signals derived from price, volume, sentiment, and other data. Although recent studies have shown that LLMs can generate diverse and effective alphas, a critical challenge lies in how to adaptively integrate them under varying market conditions. To address this gap, we leverage the deepseek-r1-distill-llama-70b model to generate fifty alphas for five major stocks: Apple, HSBC, Pepsi, Toyota, and Tencent, and then use PPO to adjust their weights in real time. Experimental results demonstrate that the PPO-optimized strategy achieves strong returns and high Sharpe ratios across most stocks, outperforming both an equal-weighted alpha portfolio and traditional benchmarks such as the Nikkei 225, S&P 500, and Hang Seng Index. The findings highlight the importance of reinforcement learning in the allocation of alpha weights and show the potential of combining LLM-generated signals with adaptive optimization for robust financial forecasting and trading.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01354v1">DPF-CM: A Data Processing Framework with Privacy-Preserving Vector Databases for Chinese Medical LLMs Training and Deployment</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      Current open-source training pipelines for Chinese medical language models predominantly emphasize optimizing training methodologies to enhance the performance of large language models (LLMs), yet lack comprehensive exploration into training data processing. To address this gap, we propose DPF-CM, a holistic Data Processing Framework for Chinese Medical LLMs training and deployment. DPF-CM comprises two core modules. The first module is a data processing pipeline tailored for model training. Beyond standard data processing operations, we (1) introduce a chained examples context-learning strategy to generate question-oriented instructions to mitigate the lack of instruction content, and (2) implement an ensemble-based filtering mechanism for preference data curation that averages multiple reward models to suppress noisy samples. The second module focuses on privacy preservation during model deployment. To prevent privacy risks from the inadvertent exposure of training data, we propose a Privacy Preserving Vector Database (PPVD) approach, which involves model memory search, high-risk database construction, secure database construction, and match-and-replace, four key stages to minimize privacy leakage during inference collectively. Experimental results show that DPF-CM significantly improves model accuracy, enabling our trained Chinese medical LLM to achieve state-of-the-art performance among open-source counterparts. Moreover, the framework reduces training data privacy leakage by 27%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01337v1">LLM-Guided Semantic Relational Reasoning for Multimodal Intent Recognition</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted by EMNLP 2025 (Main Track, Long Paper)
    </div>
    <details class="paper-abstract">
      Understanding human intents from multimodal signals is critical for analyzing human behaviors and enhancing human-machine interactions in real-world scenarios. However, existing methods exhibit limitations in their modality-level reliance, constraining relational reasoning over fine-grained semantics for complex intent understanding. This paper proposes a novel LLM-Guided Semantic Relational Reasoning (LGSRR) method, which harnesses the expansive knowledge of large language models (LLMs) to establish semantic foundations that boost smaller models' relational reasoning performance. Specifically, an LLM-based strategy is proposed to extract fine-grained semantics as guidance for subsequent reasoning, driven by a shallow-to-deep Chain-of-Thought (CoT) that autonomously uncovers, describes, and ranks semantic cues by their importance without relying on manually defined priors. Besides, we formally model three fundamental types of semantic relations grounded in logical principles and analyze their nuanced interplay to enable more effective relational reasoning. Extensive experiments on multimodal intent and dialogue act recognition tasks demonstrate LGSRR's superiority over state-of-the-art methods, with consistent performance gains across diverse semantic understanding scenarios. The complete data and code are available at https://github.com/thuiar/LGSRR.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01314v1">Can Smaller LLMs do better? Unlocking Cross-Domain Potential through Parameter-Efficient Fine-Tuning for Text Summarization</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs), being generic task solvers, are versatile. However, despite the vast amount of data they are trained on, there are speculations about their adaptation capabilities to a new domain. Additionally, the simple fine-tuning of the model to incorporate knowledge of a new domain is computationally expensive and time-consuming. This becomes more challenging when the domain in question is also low-resource, and labeled data is unavailable. We leverage parameter-efficient fine-tuning techniques (PEFTs) on high-resource datasets to address these challenges to improve performance on unseen low-resource domains. Throughout our experiments, we evaluate whether intrinsic linguistic commonalities between datasets can be leveraged for efficient domain adaptation. We benchmark six PEFTs with \texttt{Llama-3-8B-Instruct} on 14 training datasets from the Scientific, Medical, Legal, and News domains for a Text Summarization task. Our experiments show that for low-resource domains, inference using Within-Domain Adapters can achieve better performance than Few-Shot as well as a much larger \texttt{Llama-3-70B-Instruct}. Lastly, in the absence of Within-Domain Adapters, we explore the concept of using Cross-Domain Adapters as well as the strategic combinations of adapters to leverage intrinsic language similarities across domains, facilitating better adaptability and performance in low-resource settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01277v1">Communicative Agents for Slideshow Storytelling Video Generation based on LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 8 pages, 8 figures, 1 table
    </div>
    <details class="paper-abstract">
      With the rapid advancement of artificial intelligence (AI), the proliferation of AI-generated content (AIGC) tasks has significantly accelerated developments in text-to-video generation. As a result, the field of video production is undergoing a transformative shift. However, conventional text-to-video models are typically constrained by high computational costs. In this study, we propose Video-Generation-Team (VGTeam), a novel slide show video generation system designed to redefine the video creation pipeline through the integration of large language models (LLMs). VGTeam is composed of a suite of communicative agents, each responsible for a distinct aspect of video generation, such as scriptwriting, scene creation, and audio design. These agents operate collaboratively within a chat tower workflow, transforming user-provided textual prompts into coherent, slide-style narrative videos. By emulating the sequential stages of traditional video production, VGTeam achieves remarkable improvements in both efficiency and scalability, while substantially reducing computational overhead. On average, the system generates videos at a cost of only $0.103, with a successful generation rate of 98.4%. Importantly, this framework maintains a high degree of creative fidelity and customization. The implications of VGTeam are far-reaching. It democratizes video production by enabling broader access to high-quality content creation without the need for extensive resources. Furthermore, it highlights the transformative potential of language models in creative domains and positions VGTeam as a pioneering system for next-generation content creation.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01267v1">Iterative In-Context Learning to Enhance LLMs Abstract Reasoning: The Case-Study of Algebraic Tasks</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Preprint. Under review
    </div>
    <details class="paper-abstract">
      LLMs face significant challenges in systematic generalization, particularly when dealing with reasoning tasks requiring compositional rules and handling out-of-distribution examples. To address these challenges, we introduce an in-context learning methodology that improves the generalization capabilities of general purpose LLMs. Our approach employs an iterative example selection strategy, which incrementally constructs a tailored set of few-shot examples optimized to enhance model's performance on a given task. As a proof of concept, we apply this methodology to the resolution of algebraic expressions involving non-standard simplification rules, according to which the priority of addition and multiplication is changed. Our findings indicate that LLMs exhibit limited proficiency in these mathematical tasks. We further demonstrate that LLMs reasoning benefits from our iterative shot selection prompting strategy integrated with explicit reasoning instructions. Crucially, our experiments reveal that some LLMs achieve better generalization performances when prompted with simpler few-shot examples rather than complex ones following the test data distribution.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01229v1">LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 12 pages, 13 figures
    </div>
    <details class="paper-abstract">
      Quantization is a critical technique for accelerating LLM inference by reducing memory footprint and improving computational efficiency. Among various schemes, 4-bit weight and 8-bit activation quantization (W4A8) offers a strong balance between accuracy and performance. However, existing W4A8 GEMM kernels fall short in practice due to inefficient dequantization on CUDA Cores, which cannot keep pace with the high throughput of Tensor Cores. In this paper, we present LiquidGEMM, a hardware-efficient W4A8 GEMM kernel for efficient LLM serving. LiquidGEMM designs two key techniques: LiquidQuant, a hardware-efficient quantization method that enables fast, overflow-safe dequantization using just two arithmetic instructions per four elements; and an implicit fine-grained pipeline that fully overlaps weight loading, dequantization, and MMA across warp groups without software synchronization or redundant memory traffic. Experimental results show that LiquidGEMM achieves up to 2.90x speedup over state-of-the-art W4A8 kernels and up to 4.94x end-to-end system-level speedup. Compared to various quantized GEMM kernels in NVIDIA TensorRT-LLM, LiquidGEMM delivers 1.12-1.63x performance gains, and achieves up to 1.63x system-level speedup.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.04492v1">Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 8 pages, 7 figures, 1 table. pre-print version
    </div>
    <details class="paper-abstract">
      Hallucinations in Large Language Model (LLM) outputs for Question Answering (QA) tasks critically undermine their real-world reliability. This paper introduces an applied methodology for robust, one-shot hallucination detection, specifically designed for scenarios with limited data access, such as interacting with black-box LLM APIs that typically expose only a few top candidate log-probabilities per token. Our approach derives uncertainty indicators directly from these readily available log-probabilities generated during non-greedy decoding. We first derive an Entropy Production Rate (EPR) metric that offers baseline performance, later augmented with supervised learning. Our learned model uses features representing the entropic contributions of the accessible top-ranked tokens within a single generated sequence, requiring no multiple query re-runs. Evaluated across diverse QA datasets and multiple LLMs, this estimator significantly improves hallucination detection over using EPR alone. Crucially, high performance is demonstrated using only the typically small set of available log-probabilities (e.g., top <10 per token), confirming its practical efficiency and suitability for these API-constrained deployments. This work provides a readily deployable technique to enhance the trustworthiness of LLM responses from a single generation pass in QA and Retrieval-Augmented Generation (RAG) systems, with its utility further demonstrated in a finance framework analyzing responses to queries on annual reports from an industrial dataset.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01211v1">Web Fraud Attacks Against LLM-Driven Multi-Agent Systems</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      With the proliferation of applications built upon LLM-driven multi-agent systems (MAS), the security of Web links has become a critical concern in ensuring system reliability. Once an agent is induced to visit a malicious website, attackers can use it as a springboard to conduct diverse subsequent attacks, which will drastically expand the attack surface. In this paper, we propose Web Fraud Attacks, a novel type of attack aiming at inducing MAS to visit malicious websites. We design 11 representative attack variants that encompass domain name tampering (homoglyph deception, character substitution, etc.), link structure camouflage (sub-directory nesting, sub-domain grafting, parameter obfuscation, etc.), and other deceptive techniques tailored to exploit MAS's vulnerabilities in link validation. Through extensive experiments on these crafted attack vectors, we demonstrate that Web fraud attacks not only exhibit significant destructive potential across different MAS architectures but also possess a distinct advantage in evasion: they circumvent the need for complex input formats such as jailbreaking, which inherently carry higher exposure risks. These results underscore the importance of addressing Web fraud attacks in LLM-driven MAS, as their stealthiness and destructiveness pose non-negligible threats to system security and user safety.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01035v1">We Politely Insist: Your LLM Must Learn the Persian Art of Taarof</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted to EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Large language models (LLMs) struggle to navigate culturally specific communication norms, limiting their effectiveness in global contexts. We focus on Persian taarof, a social norm in Iranian interactions, which is a sophisticated system of ritual politeness that emphasizes deference, modesty, and indirectness, yet remains absent from existing cultural benchmarks. We introduce TaarofBench, the first benchmark for evaluating LLM understanding of taarof, comprising 450 role-play scenarios covering 12 common social interaction topics, validated by native speakers. Our evaluation of five frontier LLMs reveals substantial gaps in cultural competence, with accuracy rates 40-48% below native speakers when taarof is culturally appropriate. Performance varies between interaction topics, improves with Persian-language prompts, and exhibits gender-based asymmetries. We also show that responses rated "polite" by standard metrics often violate taarof norms, indicating the limitations of Western politeness frameworks. Through supervised fine-tuning and Direct Preference Optimization, we achieve 21.8% and 42.3% improvement in model alignment with cultural expectations. Our human study with 33 participants (11 native Persian, 11 heritage, and 11 non-Iranian speakers) forms baselines in varying degrees of familiarity with Persian norms. This work lays the foundation for developing diverse and culturally aware LLMs, enabling applications that better navigate complex social interactions.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.16124v2">Benchmarking LLM Privacy Recognition for Social Robot Decision Making</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 18 pages, 7 figures. Dakota Sullivan and Shirley Zhang contributed equally to this work
    </div>
    <details class="paper-abstract">
      While robots have previously utilized rule-based systems or probabilistic models for user interaction, the rapid evolution of large language models (LLMs) presents new opportunities to develop LLM-powered robots for enhanced human-robot interaction (HRI). To fully realize these capabilities, however, robots need to collect data such as audio, fine-grained images, video, and locations. As a result, LLMs often process sensitive personal information, particularly within private environments, such as homes. Given the tension between utility and privacy risks, evaluating how current LLMs manage sensitive data is critical. Specifically, we aim to explore the extent to which out-of-the-box LLMs are privacy-aware in the context of household robots. In this work, we present a set of privacy-relevant scenarios developed using the Contextual Integrity (CI) framework. We first surveyed users' privacy preferences regarding in-home robot behaviors and then examined how their privacy orientations affected their choices of these behaviors (N = 450). We then provided the same set of scenarios and questions to state-of-the-art LLMs (N = 10) and found that the agreement between humans and LLMs was generally low. To further investigate the capabilities of LLMs as potential privacy controllers, we implemented four additional prompting strategies and compared their results. We discuss the performance of the evaluated models as well as the implications and potential of AI privacy awareness in human-robot interaction.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.14201v2">ExCyTIn-Bench: Evaluating LLM agents on Cyber Threat Investigation</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Add code link
    </div>
    <details class="paper-abstract">
      We present ExCyTIn-Bench, the first benchmark to Evaluate an LLM agent x on the task of Cyber Threat Investigation through security questions derived from investigation graphs. Real-world security analysts must sift through a large number of heterogeneous alert signals and security logs, follow multi-hop chains of evidence, and compile an incident report. With the developments of LLMs, building LLM-based agents for automatic thread investigation is a promising direction. To assist the development and evaluation of LLM agents, we construct a dataset from a controlled Azure tenant that covers 8 simulated real-world multi-step attacks, 57 log tables from Microsoft Sentinel and related services, and 589 automatically generated questions. We leverage security logs extracted with expert-crafted detection logic to build threat investigation graphs, and then generate questions with LLMs using paired nodes on the graph, taking the start node as background context and the end node as answer. Anchoring each question to these explicit nodes and edges not only provides automatic, explainable ground truth answers but also makes the pipeline reusable and readily extensible to new logs. This also enables the automatic generation of procedural tasks with verifiable rewards, which can be naturally extended to training agents via reinforcement learning. Our comprehensive experiments with different models confirm the difficulty of the task: with the base setting, the average reward across all evaluated models is 0.249, and the best achieved is 0.368, leaving substantial headroom for future research. Code and data are coming soon!
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2409.03671v3">TRACE-CS: A Hybrid Logic-LLM System for Explainable Course Scheduling</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      We present TRACE-CS, a novel hybrid system that combines symbolic reasoning with large language models (LLMs)to address contrastive queries in course scheduling problems. TRACE-CS leverages logic-based techniques to encode scheduling constraints and generate provably correct explanations, while utilizing an LLM to process natural language queries and refine logical explanations into user friendly responses. This system showcases how combining symbolic KR methods with LLMs creates explainable AI agents that balance logical correctness with natural language accessibility, addressing a fundamental challenge in deployed scheduling systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2507.01378v2">RALLY: Role-Adaptive LLM-Driven Yoked Navigation for Agentic UAV Swarms</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Intelligent control of Unmanned Aerial Vehicles (UAVs) swarms has emerged as a critical research focus, and it typically requires the swarm to navigate effectively while avoiding obstacles and achieving continuous coverage over multiple mission targets. Although traditional Multi-Agent Reinforcement Learning (MARL) approaches offer dynamic adaptability, they are hindered by the semantic gap in numerical communication and the rigidity of homogeneous role structures, resulting in poor generalization and limited task scalability. Recent advances in Large Language Model (LLM)-based control frameworks demonstrate strong semantic reasoning capabilities by leveraging extensive prior knowledge. However, due to the lack of online learning and over-reliance on static priors, these works often struggle with effective exploration, leading to reduced individual potential and overall system performance. To address these limitations, we propose a Role-Adaptive LLM-Driven Yoked navigation algorithm RALLY. Specifically, we first develop an LLM-driven semantic decision framework that uses structured natural language for efficient semantic communication and collaborative reasoning. Afterward, we introduce a dynamic role-heterogeneity mechanism for adaptive role switching and personalized decision-making. Furthermore, we propose a Role-value Mixing Network (RMIX)-based assignment strategy that integrates LLM offline priors with MARL online policies to enable semi-offline training of role selection strategies. Experiments in the Multi-Agent Particle Environment (MPE) environment and a Software-In-The-Loop (SITL) platform demonstrate that RALLY outperforms conventional approaches in terms of task coverage, convergence speed, and generalization, highlighting its strong potential for collaborative navigation in agentic multi-UAV systems.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2505.24671v2">Multiple LLM Agents Debate for Equitable Cultural Alignment</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 ACL 2025 (Oral)
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) need to adapt their predictions to diverse cultural contexts to benefit diverse communities across the world. While previous efforts have focused on single-LLM, single-turn approaches, we propose to exploit the complementary strengths of multiple LLMs to promote cultural adaptability. We introduce a Multi-Agent Debate framework, where two LLM-based agents debate over a cultural scenario and collaboratively reach a final decision. We propose two variants: one where either LLM agents exclusively debate and another where they dynamically choose between self-reflection and debate during their turns. We evaluate these approaches on 7 open-weight LLMs (and 21 LLM combinations) using the NormAd-ETI benchmark for social etiquette norms in 75 countries. Experiments show that debate improves both overall accuracy and cultural group parity over single-LLM baselines. Notably, multi-agent debate enables relatively small LLMs (7-9B) to achieve accuracies comparable to that of a much larger model (27B parameters).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.16654v2">MSNav: Zero-Shot Vision-and-Language Navigation with Dynamic Memory and LLM Spatial Reasoning</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 19 pages, 15 figures
    </div>
    <details class="paper-abstract">
      Vision-and-Language Navigation (VLN) requires an agent to interpret natural language instructions and navigate complex environments. Current approaches often adopt a "black-box" paradigm, where a single Large Language Model (LLM) makes end-to-end decisions. However, it is plagued by critical vulnerabilities, including poor spatial reasoning, weak cross-modal grounding, and memory overload in long-horizon tasks. To systematically address these issues, we propose Memory Spatial Navigation(MSNav), a framework that fuses three modules into a synergistic architecture, which transforms fragile inference into a robust, integrated intelligence. MSNav integrates three modules: Memory Module, a dynamic map memory module that tackles memory overload through selective node pruning, enhancing long-range exploration; Spatial Module, a module for spatial reasoning and object relationship inference that improves endpoint recognition; and Decision Module, a module using LLM-based path planning to execute robust actions. Powering Spatial Module, we also introduce an Instruction-Object-Space (I-O-S) dataset and fine-tune the Qwen3-4B model into Qwen-Spatial (Qwen-Sp), which outperforms leading commercial LLMs in object list extraction, achieving higher F1 and NDCG scores on the I-O-S test set. Extensive experiments on the Room-to-Room (R2R) and REVERIE datasets demonstrate MSNav's state-of-the-art performance with significant improvements in Success Rate (SR) and Success weighted by Path Length (SPL).
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.14544v2">Adaptively Robust LLM Inference Optimization under Prediction Uncertainty</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      We study the problem of optimizing Large Language Model (LLM) inference scheduling to minimize total latency. LLM inference is an online and multi-task service process and also heavily energy consuming by which a pre-trained LLM processes input requests and generates output tokens sequentially. Therefore, it is vital to improve its scheduling efficiency and reduce the power consumption while a great amount of prompt requests are arriving. A key challenge in LLM inference scheduling is that while the prompt length is known upon arrival, the output length, which critically impacts memory usage and processing time, is unknown. To address this uncertainty, we propose algorithms that leverage machine learning to predict output lengths, assuming the prediction provides an interval classification (min-max range) for each request. We first design a conservative algorithm, $\mathcal{A}_{\max}$, which schedules requests based on the upper bound of predicted output lengths to prevent memory overflow. However, this approach is overly conservative: as prediction accuracy decreases, performance degrades significantly due to potential overestimation. To overcome this limitation, we propose $\mathcal{A}_{\min}$, an adaptive algorithm that initially treats the predicted lower bound as the output length and dynamically refines this estimate during inferencing. We prove that $\mathcal{A}_{\min}$ achieves a log-scale competitive ratio. Through numerical simulations, we demonstrate that $\mathcal{A}_{\min}$ often performs nearly as well as the hindsight scheduler, highlighting both its efficiency and robustness in practical scenarios. Moreover, $\mathcal{A}_{\min}$ relies solely on the lower bound of the prediction interval--an advantageous design choice since upper bounds on output length are typically more challenging to predict accurately.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.01235v3">NarraGuide: an LLM-based Narrative Mobile Robot for Remote Place Exploration</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Robotic telepresence enables users to navigate and experience remote environments. However, effective navigation and situational awareness depend on users' prior knowledge of the environment, limiting the usefulness of these systems for exploring unfamiliar places. We explore how integrating location-aware LLM-based narrative capabilities into a mobile robot can support remote exploration. We developed a prototype system, called NarraGuide, that provides narrative guidance for users to explore and learn about a remote place through a dialogue-based interface. We deployed our prototype in a geology museum, where remote participants (n=20) used the robot to tour the museum. Our findings reveal how users perceived the robot's role, engaged in dialogue in the tour, and expressed preferences for bystander encountering. Our work demonstrates the potential of LLM-enabled robotic capabilities to deliver location-aware narrative guidance and enrich the experience of exploring remote environments.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2503.00134v3">Personalized Causal Graph Reasoning for LLMs: An Implementation for Dietary Recommendations</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) excel at general-purpose reasoning by leveraging broad commonsense knowledge, but they remain limited in tasks requiring personalized reasoning over multifactorial personal data. This limitation constrains their applicability in domains such as healthcare, where decisions must adapt to individual contexts. We introduce Personalized Causal Graph Reasoning, a framework that enables LLMs to reason over individual-specific causal graphs constructed from longitudinal data. Each graph encodes how user-specific factors influence targeted outcomes. In response to a query, the LLM traverses the graph to identify relevant causal pathways, rank them by estimated impact, simulate potential outcomes, and generate tailored responses. We implement this framework in the context of nutrient-oriented dietary recommendations, where variability in metabolic responses demands personalized reasoning. Using counterfactual evaluation, we assess the effectiveness of LLM-generated food suggestions for glucose control. Our method reduces postprandial glucose iAUC across three time windows compared to prior approaches. Additional LLM-as-a-judge evaluations further confirm improvements in personalization quality.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2508.19611v2">Instructional Agents: LLM Agents on Automated Course Material Generation for Teaching Faculties</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 18 pages, 9 figures
    </div>
    <details class="paper-abstract">
      Preparing high-quality instructional materials remains a labor-intensive process that often requires extensive coordination among teaching faculty, instructional designers, and teaching assistants. In this work, we present Instructional Agents, a multi-agent large language model (LLM) framework designed to automate end-to-end course material generation, including syllabus creation, lecture scripts, LaTeX-based slides, and assessments. Unlike existing AI-assisted educational tools that focus on isolated tasks, Instructional Agents simulates role-based collaboration among educational agents to produce cohesive and pedagogically aligned content. The system operates in four modes: Autonomous, Catalog-Guided, Feedback-Guided, and Full Co-Pilot mode, enabling flexible control over the degree of human involvement. We evaluate Instructional Agents across five university-level computer science courses and show that it produces high-quality instructional materials while significantly reducing development time and human workload. By supporting institutions with limited instructional design capacity, Instructional Agents provides a scalable and cost-effective framework to democratize access to high-quality education, particularly in underserved or resource-constrained settings.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01822v1">When LLM Meets Time Series: Can LLMs Perform Multi-Step Time Series Reasoning and Inference</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      The rapid advancement of Large Language Models (LLMs) has sparked growing interest in their application to time series analysis tasks. However, their ability to perform complex reasoning over temporal data in real-world application domains remains underexplored. To move toward this goal, a first step is to establish a rigorous benchmark dataset for evaluation. In this work, we introduce the TSAIA Benchmark, a first attempt to evaluate LLMs as time-series AI assistants. To ensure both scientific rigor and practical relevance, we surveyed over 20 academic publications and identified 33 real-world task formulations. The benchmark encompasses a broad spectrum of challenges, ranging from constraint-aware forecasting to anomaly detection with threshold calibration: tasks that require compositional reasoning and multi-step time series analysis. The question generator is designed to be dynamic and extensible, supporting continuous expansion as new datasets or task types are introduced. Given the heterogeneous nature of the tasks, we adopt task-specific success criteria and tailored inference-quality metrics to ensure meaningful evaluation for each task. We apply this benchmark to assess eight state-of-the-art LLMs under a unified evaluation protocol. Our analysis reveals limitations in current models' ability to assemble complex time series analysis workflows, underscoring the need for specialized methodologies for domain-specific adaptation. Our benchmark is available at https://huggingface.co/datasets/Melady/TSAIA, and the code is available at https://github.com/USC-Melady/TSAIA.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01790v1">Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted to EMNLP 2025 Main Conference
    </div>
    <details class="paper-abstract">
      Prompt sensitivity, referring to the phenomenon where paraphrasing (i.e., repeating something written or spoken using different words) leads to significant changes in large language model (LLM) performance, has been widely accepted as a core limitation of LLMs. In this work, we revisit this issue and ask: Is the widely reported high prompt sensitivity truly an inherent weakness of LLMs, or is it largely an artifact of evaluation processes? To answer this question, we systematically evaluate 7 LLMs (e.g., GPT and Gemini family) across 6 benchmarks, including both multiple-choice and open-ended tasks on 12 diverse prompt templates. We find that much of the prompt sensitivity stems from heuristic evaluation methods, including log-likelihood scoring and rigid answer matching, which often overlook semantically correct responses expressed through alternative phrasings, such as synonyms or paraphrases. When we adopt LLM-as-a-Judge evaluations, we observe a substantial reduction in performance variance and a consistently higher correlation in model rankings across prompts. Our findings suggest that modern LLMs are more robust to prompt templates than previously believed, and that prompt sensitivity may be more an artifact of evaluation than a flaw in the models.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01750v1">Communication-Aware Knowledge Distillation for Federated LLM Fine-Tuning over Wireless Networks</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      Federated learning (FL) for large language models (LLMs) offers a privacy-preserving scheme, enabling clients to collaboratively fine-tune locally deployed LLMs or smaller language models (SLMs) without exchanging raw data. While parameter-sharing methods in traditional FL models solves number of technical challenges, they still incur high communication overhead and struggle with adapting to heterogeneous model architectures. Federated distillation, a framework for mutual knowledge transfer via shared logits, typically offers lower communication overhead than parameter-sharing methods. However, transmitting logits from LLMs remains challenging for bandwidth-limited clients due to their high dimensionality. In this work, we focus on a federated LLM distillation with efficient communication overhead. To achieve this, we first propose an adaptive Top-k logit selection mechanism, dynamically sparsifying logits according to real-time communication conditions. Then to tackle the dimensional inconsistency introduced by the adaptive sparsification, we design an adaptive logits aggregation scheme, effectively alleviating the artificial and uninformative inputs introduced by conventional zero-padding methods. Finally, to enhance the distillation effect, we incorporate LoRA-adapted hidden-layer projection from LLM into the distillation loss, reducing the communication overhead further while providing richer representation. Experimental results demonstrate that our scheme achieves superior performance compared to baseline methods while effectively reducing communication overhead by approximately 50%.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01716v1">An LLM-enabled semantic-centric framework to consume privacy policies</a></div>
    <div class="paper-meta">
      📅 2025-09-01
    </div>
    <details class="paper-abstract">
      In modern times, people have numerous online accounts, but they rarely read the Terms of Service or Privacy Policy of those sites, despite claiming otherwise, due to the practical difficulty in comprehending them. The mist of data privacy practices forms a major barrier for user-centred Web approaches, and for data sharing and reusing in an agentic world. Existing research proposed methods for using formal languages and reasoning for verifying the compliance of a specified policy, as a potential cure for ignoring privacy policies. However, a critical gap remains in the creation or acquisition of such formal policies at scale. We present a semantic-centric approach for using state-of-the-art large language models (LLM), to automatically identify key information about privacy practices from privacy policies, and construct $\mathit{Pr}^2\mathit{Graph}$, knowledge graph with grounding from Data Privacy Vocabulary (DPV) for privacy practices, to support downstream tasks. Along with the pipeline, the $\mathit{Pr}^2\mathit{Graph}$ for the top-100 popular websites is also released as a public resource, by using the pipeline for analysis. We also demonstrate how the $\mathit{Pr}^2\mathit{Graph}$ can be used to support downstream tasks by constructing formal policy representations such as Open Digital Right Language (ODRL) or perennial semantic Data Terms of Use (psDToU). To evaluate the technology capability, we enriched the Policy-IE dataset by employing legal experts to create custom annotations. We benchmarked the performance of different large language models for our pipeline and verified their capabilities. Overall, they shed light on the possibility of large-scale analysis of online services' privacy practices, as a promising direction to audit the Web and the Internet. We release all datasets and source code as public resources to facilitate reuse and improvement.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01631v1">Unraveling LLM Jailbreaks Through Safety Knowledge Neurons</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 10 pages, 6 figures
    </div>
    <details class="paper-abstract">
      Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation, a technique known as "Jailbreak." While some studies have achieved defenses against jailbreak attacks by modifying output distributions or detecting harmful content, the exact rationale still remains elusive. In this work, we present a novel neuron-level interpretability method that focuses on the role of safety-related knowledge neurons. Unlike existing approaches, our method projects the model's internal representation into a more consistent and interpretable vocabulary space. We then show that adjusting the activation of safety-related neurons can effectively control the model's behavior with a mean ASR higher than 97%. Building on this insight, we propose SafeTuning, a fine-tuning strategy that reinforces safety-critical neurons to improve model robustness against jailbreaks. SafeTuning consistently reduces attack success rates across multiple LLMs and outperforms all four baseline defenses. These findings offer a new perspective on understanding and defending against jailbreak attacks.
    </details>
</div>
<div class="paper-card">
    <div class="paper-title"><a href="http://arxiv.org/abs/2509.01620v1">Benchmarking the Detection of LLMs-Generated Modern Chinese Poetry</a></div>
    <div class="paper-meta">
      📅 2025-09-01
      | 💬 Accepted by EMNLP 2025
    </div>
    <details class="paper-abstract">
      The rapid development of advanced large language models (LLMs) has made AI-generated text indistinguishable from human-written text. Previous work on detecting AI-generated text has made effective progress, but has not involved modern Chinese poetry. Due to the distinctive characteristics of modern Chinese poetry, it is difficult to identify whether a poem originated from humans or AI. The proliferation of AI-generated modern Chinese poetry has significantly disrupted the poetry ecosystem. Based on the urgency of identifying AI-generated poetry in the real Chinese world, this paper proposes a novel benchmark for detecting LLMs-generated modern Chinese poetry. We first construct a high-quality dataset, which includes both 800 poems written by six professional poets and 41,600 poems generated by four mainstream LLMs. Subsequently, we conduct systematic performance assessments of six detectors on this dataset. Experimental results demonstrate that current detectors cannot be used as reliable tools to detect modern Chinese poems generated by LLMs. The most difficult poetic features to detect are intrinsic qualities, especially style. The detection results verify the effectiveness and necessity of our proposed benchmark. Our work lays a foundation for future detection of AI-generated poetry.
    </details>
</div>
